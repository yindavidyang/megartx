"""Eight-file streaming policy for the attention purpose only.

The inherited CompactEvidence serializer, failure reserve and cross-process
flock remain shared by every writer. Fixed extents count once as files, never
as a second reservation. Completion is durable metadata, not zero-fill or size.
No native runtime imports and no authority to attest storage or execution.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from . import prefill_attention_plan as capture_plan
from .prefill_attention_validation import CAPTURE_FILE, CAPTURE_SCHEMA
from .prefill_storage_plan import CompactEvidence, FAILURE_FILE, FAILURE_RESERVE_BYTES

RAW_FILES = {name: layout['bytes'] for name, layout in capture_plan.raw_layouts().items()}
STATE_FILE = 'attention-stream-state.json'
STATE_TEMP = 'attention-stream-state-pending.json'
STATE_BYTES = 8192
INVALID_FILES = {FAILURE_FILE, 'attention-failure.json', 'INVALIDATED.json'}


def _geometry(name):
    layout = capture_plan.raw_layouts()[name]
    rows = layout['shape'][0] if layout['role'] in ('k', 'v') else 10 * layout['shape'][1]
    return rows, layout['bytes'] // rows


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate streaming-state key')
        result[key] = value
    return result


class StreamingEvidence(CompactEvidence):
    """Purpose-local raw layout with the inherited single evidence budget.

    A contribution is exactly one selected K/V row or one selected Q/O head.
    Persist pending before writing, fsync the payload, then mark completion.
    An interruption between these operations invalidates the whole capture.
    """
    def __init__(self, directory, limit=8 << 20, metadata_limit=2 << 20):
        path = Path(directory).absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Symlink evidence path refused')
        super().__init__(path, limit, metadata_limit)

    @staticmethod
    def _name(name, raw=False):
        if raw:
            if name not in RAW_FILES:
                raise ValueError('Only the eight attention raw files are admitted')
            return
        CompactEvidence._name(name, False)

    def _sizes(self):
        total = metadata = 2  # inherited external run.exit, independent of folder spelling
        count = 0
        for path in self.directory.iterdir():
            count += 1
            info = path.lstat()
            if count > 256 or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Bounded single-link regular evidence entries required')
            if path.name == '.budget-lock' and info.st_size:
                raise ValueError('Budget lock must not contain payload')
            if path.name.endswith('.bf16') and path.name not in RAW_FILES:
                raise ValueError('Previous-purpose or unknown raw retention is forbidden')
            if path.name in RAW_FILES and info.st_size != RAW_FILES[path.name]:
                raise ValueError('Attention raw extent changed')
            if path.name in INVALID_FILES and info.st_size > FAILURE_RESERVE_BYTES:
                raise ValueError('Failure evidence exceeds its bounded extent')
            total += info.st_size
            metadata += 0 if path.name in RAW_FILES else info.st_size
        return total, metadata

    @contextmanager
    def _locked(self):
        fd = os.open(self.directory / '.budget-lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._sizes()
            yield
        finally:
            os.close(fd)

    def _reserve(self, count, *, raw=False, failure=False):
        total, metadata = self._sizes()
        reserve = 0 if failure or (self.directory / FAILURE_FILE).exists() else FAILURE_RESERVE_BYTES
        if total + count + reserve > self.limit or metadata + (0 if raw else count) + reserve > self.metadata_limit:
            raise ValueError('Shared attention evidence budget exceeded before write')

    @staticmethod
    def _write_exact(fd, data):
        # Short writes are an invalidated context, never silently credited.
        if os.write(fd, data) != len(data):
            raise OSError('Short evidence write')
        os.fsync(fd)

    def _create_bytes(self, name, data):
        fd = os.open(self.directory / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            self._write_exact(fd, data)
        finally:
            os.close(fd)

    def _failure(self, error):
        """Called while holding the one lock; preserve the first shared failure."""
        if (self.directory / FAILURE_FILE).exists():
            return
        data = _encode({'schema': 'megartx-prefill-attention-stream-failure-v1',
                        'status': 'invalidated', 'error_type': type(error).__name__,
                        'error': str(error)[:1024]})
        try:
            self._reserve(len(data), failure=True)
            self._create_bytes(FAILURE_FILE, data)
        except BaseException as failure:
            if hasattr(error, 'add_note'):
                error.add_note('Failure evidence could not be saved: ' + type(failure).__name__)

    def _valid(self):
        if any((self.directory / name).exists() for name in INVALID_FILES):
            raise ValueError('Invalidated attention context')
        if (self.directory / STATE_TEMP).exists():
            raise ValueError('Interrupted streaming-state update')

    def _save_state(self, value):
        data = _encode(value)
        if len(data) > STATE_BYTES:
            raise ValueError('Bounded streaming-state extent exceeded')
        data += b' ' * (STATE_BYTES - len(data))
        # Both the old state and this temporary publication count until rename.
        self._reserve(STATE_BYTES)
        self._create_bytes(STATE_TEMP, data)
        os.replace(self.directory / STATE_TEMP, self.directory / STATE_FILE)
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _state(self):
        self._valid()
        fd = os.open(self.directory / STATE_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != STATE_BYTES:
                raise ValueError('Exact bounded streaming state required')
            raw = os.read(fd, STATE_BYTES + 1)
            if len(raw) != STATE_BYTES:
                raise ValueError('Incomplete streaming state')
        finally:
            os.close(fd)
        value = json.loads(raw, object_pairs_hook=_pairs)
        expected = {'schema', 'capture_plan_sha256', 'phase', 'pending', 'files', 'manifest'}
        if (type(value) is not dict or set(value) != expected
                or value['schema'] != 'megartx-prefill-attention-stream-v1'
                or value['phase'] not in ('active', 'sealed') or value['pending'] is not None
                or type(value['files']) is not dict or set(value['files']) != set(RAW_FILES)):
            raise ValueError('Incomplete or interrupted attention streaming state')
        capture_plan.sha(value['capture_plan_sha256'])
        for name, item in value['files'].items():
            rows, _ = _geometry(name)
            if type(item) is not dict or set(item) != {'identity', 'completed'}:
                raise ValueError('Exact raw completion schema required')
            bits = bytes.fromhex(item['completed'])
            if len(bits) != (rows + 7) // 8 or (rows % 8 and bits[-1] >> (rows % 8)):
                raise ValueError('Raw completion bitmap extent changed')
            info = (self.directory / name).lstat()
            if item['identity'] != [info.st_dev, info.st_ino] or info.st_size != RAW_FILES[name]:
                raise ValueError('Preallocated raw identity/extent changed')
        return value

    def create_capture(self, capture_plan_sha256):
        capture_plan.sha(capture_plan_sha256)
        with self._locked():
            try:
                self._valid()
                if any((self.directory / name).exists() for name in (*RAW_FILES, STATE_FILE, CAPTURE_FILE)):
                    raise ValueError('Capture creation is exclusive; resume and retries are forbidden')
                self._reserve(sum(RAW_FILES.values()), raw=True)
                # Account every real extent exactly once, and keep room for the
                # fixed state plus its replacement temporary and failure record.
                total, metadata = self._sizes()
                if (total + sum(RAW_FILES.values()) + 2 * STATE_BYTES + FAILURE_RESERVE_BYTES > self.limit
                        or metadata + 2 * STATE_BYTES + FAILURE_RESERVE_BYTES > self.metadata_limit):
                    raise ValueError('Capture allocation and metadata reserve exceed shared budget')
                files = {}
                for name, size in RAW_FILES.items():
                    fd = os.open(self.directory / name, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                    try:
                        os.ftruncate(fd, size)
                        os.fsync(fd)
                        info = os.fstat(fd)
                        if info.st_size != size:
                            raise OSError('Preallocation extent mismatch')
                        rows, _ = _geometry(name)
                        files[name] = {'identity': [info.st_dev, info.st_ino],
                                       'completed': bytes((rows + 7) // 8).hex()}
                    finally:
                        os.close(fd)
                self._save_state({'schema': 'megartx-prefill-attention-stream-v1',
                    'capture_plan_sha256': capture_plan_sha256, 'phase': 'active', 'pending': None,
                    'files': files, 'manifest': None})
            except BaseException as error:
                self._failure(error)
                raise

    def contribute(self, name, offset, data):
        with self._locked():
            try:
                value = self._state()
                if value['phase'] != 'active' or name not in RAW_FILES:
                    raise ValueError('Active selected raw contribution required')
                rows, width = _geometry(name)
                if (type(offset) is not int or offset < 0 or offset % width
                        or offset // width >= rows or type(data) is not bytes or len(data) != width):
                    raise ValueError('Contribution must be one exact selected row/head extent')
                capture_plan.finite_words(data, width // 2)
                row = offset // width
                bits = bytearray.fromhex(value['files'][name]['completed'])
                if bits[row // 8] & (1 << (row % 8)):
                    raise ValueError('Duplicate attention contribution')
                value['pending'] = [name, offset, width]
                self._save_state(value)
                fd = os.open(self.directory / name, os.O_WRONLY | os.O_NOFOLLOW)
                try:
                    info = os.fstat(fd)
                    if [info.st_dev, info.st_ino] != value['files'][name]['identity']:
                        raise ValueError('Raw target replaced during contribution')
                    if os.pwrite(fd, data, offset) != width:
                        raise OSError('Short raw contribution write')
                    os.fsync(fd)
                finally:
                    os.close(fd)
                bits[row // 8] |= 1 << (row % 8)
                value['files'][name]['completed'] = bits.hex()
                value['pending'] = None
                self._save_state(value)
            except BaseException as error:
                self._failure(error)
                raise

    def writer_row(self, layer, position, key, value):
        for name, offset, data in capture_plan.select_writer_row(layer, position, key, value):
            self.contribute(name, offset, data)

    def head_row(self, layer, position, head, role, raw):
        self.contribute(*capture_plan.select_head_row(layer, position, head, role, raw))

    def read_slice(self, name, offset, size):
        with self._locked():
            value = self._state()
            if (name not in RAW_FILES or type(offset) is not int or type(size) is not int
                    or offset < 0 or not 0 < size <= 4096 or offset + size > RAW_FILES[name]):
                raise ValueError('Bounded retained raw slice required')
            _, width = _geometry(name)
            bits = bytes.fromhex(value['files'][name]['completed'])
            if any(not bits[row // 8] & (1 << (row % 8))
                   for row in range(offset // width, (offset + size - 1) // width + 1)):
                raise ValueError('Unwritten retained raw slice')
            fd = os.open(self.directory / name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                if [info.st_dev, info.st_ino] != value['files'][name]['identity']:
                    raise ValueError('Retained raw target replaced')
                data = os.pread(fd, size, offset)
                if len(data) != size:
                    raise ValueError('Short retained raw read')
                return data
            finally:
                os.close(fd)

    def read_selection(self, layer, position):
        capture_plan.coordinates(layer)
        capture_plan.integer(position, 0, 2048)
        prefix = f'attention-layer-{layer:02d}-'
        result = []
        for role in ('k', 'v'):
            name = prefix + role + '.bf16'
            _, width = _geometry(name)
            result.append(self.read_slice(name, position * width, width))
        return tuple(result)

    def _manifest(self, value):
        result = {}
        for name, size in RAW_FILES.items():
            rows, _ = _geometry(name)
            bits = bytes.fromhex(value['files'][name]['completed'])
            if any(not bits[row // 8] & (1 << (row % 8)) for row in range(rows)):
                raise ValueError('Missing attention contribution; zero-filled holes are not samples')
            fd = os.open(self.directory / name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                if ([before.st_dev, before.st_ino] != value['files'][name]['identity']
                        or before.st_size != size or before.st_nlink != 1):
                    raise ValueError('Raw file changed before sealing')
                hasher = hashlib.sha256()
                count = 0
                while chunk := os.read(fd, 65536):
                    count += len(chunk)
                    if count > size:
                        raise ValueError('Raw extent grew while sealing')
                    capture_plan.finite_words(chunk, len(chunk) // 2)
                    hasher.update(chunk)
                after = os.fstat(fd)
                identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
                if count != size or identity(before) != identity(after):
                    raise ValueError('Raw file changed while sealing')
                result[name] = {'bytes': size, 'sha256': hasher.hexdigest()}
            finally:
                os.close(fd)
        return result

    def seal_raw(self):
        with self._locked():
            try:
                value = self._state()
                manifest = self._manifest(value)
                if value['phase'] == 'sealed':
                    if value['manifest'] != manifest:
                        raise ValueError('Sealed raw manifest changed')
                    return manifest
                value.update(phase='sealed', manifest=manifest)
                self._save_state(value)
                return manifest
            except BaseException as error:
                self._failure(error)
                raise

    def finalize_capture(self, metadata):
        """Save only completed raw capture; caller still owns external gates."""
        with self._locked():
            try:
                value = self._state()
                expected = {'schema', 'plan_sha256', 'native_plan_sha256', 'prompt_sha256',
                    'request_identity_sha256', 'inherited_storage_binding_sha256', 'raw_manifest', 'records'}
                if (value['phase'] != 'sealed' or type(metadata) is not dict or set(metadata) != expected
                        or metadata['schema'] != CAPTURE_SCHEMA
                        or metadata['plan_sha256'] != value['capture_plan_sha256']
                        or metadata['raw_manifest'] != value['manifest']
                        or metadata['raw_manifest'] != self._manifest(value)
                        or type(metadata['records']) is not list or len(metadata['records']) != 12):
                    raise ValueError('Complete bound sealed capture metadata required')
                for field in ('native_plan_sha256', 'prompt_sha256', 'request_identity_sha256',
                              'inherited_storage_binding_sha256'):
                    capture_plan.sha(metadata[field])
                data = _encode(metadata)
                if len(data) > 1 << 20:
                    raise ValueError('Capture metadata extent exceeded')
                self._reserve(len(data))
                self._create_bytes(CAPTURE_FILE, data)
            except BaseException as error:
                self._failure(error)
                raise

    def raw(self, name, data):
        raise ValueError('Attention raw files require tracked streaming contributions')


AttentionEvidence = StreamingEvidence

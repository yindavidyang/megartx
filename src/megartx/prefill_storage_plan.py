"""Default-off source admission and compact evidence for one storage control.

Importing this module is CPU-only. Freezing a plan is not GPU clearance and a
storage receipt is deliberately distinct from the historical native fit proof.
"""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat

from . import prefill_diagnostic_plan as legacy
from . import m1_map_borrow_lineage as map_borrow
from . import prefill_attention_lineage as attention_lineage
ATTENTION_HELPER_SHA256 = '6ddcf45d7ad75888e63d374119a0c11c887361752b15c5ef308c30370b6d9dfa'

BASE = legacy.BASE
BOUNDS = legacy.BOUNDS
RUNNER_POLICY = legacy.RUNNER_POLICY
digest = legacy.digest
file_sha = legacy.file_sha
remaining = legacy.remaining
checkpoint_identity = legacy.checkpoint_identity
api_request_id = legacy.api_request_id
native_request_identity = legacy.native_request_identity
native_ledger_comparison = legacy.native_ledger_comparison
server_args = legacy.server_args

SCHEMA = 'megartx-prefill-storage-plan-v1'
PURPOSE = 'first-nine-frame-storage-frontier-control'
CONTROL_SPEC = {
    'max_contexts': 1, 'max_requests_per_context': 1, 'max_retries': 0,
    'compute_dtype': 'bfloat16', 'kv_dtype': 'bfloat16',
    'capture_frames': 9, 'capture_end': 2049, 'checked_layer_frames': 270,
    'pre_rows': 212355, 'processed_rows': 61470, 'post_rows': 273825,
    'metadata_only_decode_inputs': 254, 'single_row_equivalent_reads': 547650,
    'transferred_bytes_per_context': 4024934400,
    'transfer_limit_bytes_per_context': 4 << 30,
    'max_metadata_bytes': 2 << 20, 'max_evidence_bytes': 8 << 20,
    'failure_evidence_reserve_bytes': 4096,
    'kernel_page_tokens': 16,
    'sample_positions': [15, 16, 1023, 1024, 2047, 2048],
    'sample_combined_kv_rows': 180, 'sample_kv_bytes': 1351680,
    'raw_head_rows': 2, 'raw_head_bytes': 1048576,
    'physical_policy': 'full_context', 'storage_comparison': 'exact_bf16_bits',
    'same_path_reference': 'unavailable', 'sampled_repeatability_qualified': False,
    'independent_arithmetic_qualified': False, 'quality_qualified': False,
    'numerical_qualified': False, 'performance_qualified': False,
}
CATALOG_SOURCE = 'docs/prefill/current-main-source-reconciliation.json'
PREFILL_CATALOG_SOURCE = 'docs/prefill/native-storage-source-reconciliation.json'
COMPOSITION_PARENTS = {
    'main_parent': '036ec1c63fd41a740a8a9b67e76ee294b1c74990',
    'main_parent_tree': '2b9165ad156503850f0ee81a38288ae180dc0ab2',
    'prefill_parent': '6493affbcf0827f8702e736636bb4f2549d07305',
    'prefill_parent_tree': '257ef9b3ab5a5a88d606eb22c461540de4bf4dc9',
    'common_base': 'ca6c385562aff30e91e8bb6a7f4228c9d766c30d',
}
PARENT_LEDGERS = {
    PREFILL_CATALOG_SOURCE: 'ac0774e3aae335edd2abc8e68d655e9b91179a50e232deaf1f3a02827d3aceec',
    'docs/prefill/main-reconciliation-lineage.json': '644b1ab27a77427ef7a1d1ea36534fbd6b397bf2f61c0244546233071beef757',
    'docs/evidence/m1-warmed-timing-source-pins.json': 'da428cd43a7466fde3ea757858923e72fe9f0ec2c428d9234ac11a6517af3410',
    'docs/evidence/m1-warmed-evidence-source-pins.json': '25ab6914c2fe4858c08e36cc6ee000137237b7c4774e6afb8afd5b0512b2b80a',
    'docs/evidence/m1-tma-descriptor-source-pins.json': '583f4b637c2a82857b6f422d8a0ec1562ee3ccd7ef456f6d7d2f0e7a811bb811',
}
EXECUTING_HELPERS = {
    'scripts/m1_owned_processes.py': 'b193ac3b991723d82f3d92b4908d5dfd5e04fce32c0b9d62435cc66f95797253',
    'scripts/prefill_owned_processes.py': 'df01a29a8089db09530eef3d36efc04d2871033b4fa6b357eb73a0aca99dae4b',
}
# Canonical ordered {path, main_parent_sha256, prefill_parent_sha256} records,
# independently extracted from the two exact Git parents for this composition.
# This closes the inventory against missing, duplicate, unknown or relabeled
# lineage records without rewriting either historical ledger.
RECONCILED_PARENTS_SHA256 = '2fc5978f026e5fe641b2f7ad5ed5b7e16269c5ae51e197260efcf5cb259e6113'
EXTRA_SOURCES = (
    'src/megartx/prefill_storage_plan.py',
    'src/megartx/prefill_storage.py',
    'src/megartx/prefill_storage_native.py',
    'src/megartx/prefill_storage_process.py',
    'scripts/prefill_storage_client.py',
    'numerical_reference/prefill_native_control.py',
    CATALOG_SOURCE,
    'docs/prefill/native-storage-control.md',
    'docs/prefill/current-main-composition.md',
)
SOURCES = (*legacy.SOURCES, *EXTRA_SOURCES, map_borrow.CATALOG_SOURCE, map_borrow.HELPER_SOURCE)
SOURCES = (*SOURCES, attention_lineage.CATALOG_SOURCE, attention_lineage.HELPER_SOURCE)
CHECKER_SOURCE = 'numerical_reference/prefill_native_control.py'
FAILURE_FILE = 'storage-failure.json'
FAILURE_RESERVE_BYTES = 4096
TELEMETRY_FIELDS = ['memory.used', 'memory.free', 'utilization.gpu', 'power.draw',
                    'temperature.gpu', 'clocks.sm', 'clocks.mem', 'pstate']
TELEMETRY_HEADER = {'schema': 'megartx-prefill-storage-telemetry-v1',
                    'columns': ['monotonic_ns', 'unix_ns', 'values', 'exit'],
                    'fields': TELEMETRY_FIELDS}
RAW_FILES = {
    f'kv-layer-{layer:02d}-position-{position:04d}.bf16': (4096 if layer % 6 == 5 else 8192)
    for layer in range(30) for position in CONTROL_SPEC['sample_positions']
}
RAW_FILES.update({f'head-position-{position}.bf16': 524288 for position in (2047, 2048)})
LEGACY_FIELDS = {
    'schema', 'base', 'proposal_sha256', 'source_head', 'source_hashes',
    'checkpoint_revision', 'checkpoint_identity', 'tokens', 'prompt_sha256',
    'bounds', 'plan_sha256', 'adapter_site', 'runner_policy',
}


def _json_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON field in storage evidence')
        value[key] = item
    return value


def _read_json(path, limit=1 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Storage evidence requires a bounded regular file: ' + path.name)
    return json.loads(path.read_text(), object_pairs_hook=_json_pairs,
                      parse_constant=lambda text: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def _sha(value):
    return type(value) is str and bool(re.fullmatch('[0-9a-f]{64}', value))


def _exact(value, expected):
    """Canonical JSON comparison also rejects bool/int and float/int aliases."""
    return type(value) is type(expected) and digest(value) == digest(expected)


def _source_file(root, relative, expected):
    path = Path(root) / relative
    if (any((Path(root) / Path(*Path(relative).parts[:i])).is_symlink()
            for i in range(1, len(Path(relative).parts) + 1))
            or not path.is_file() or not _sha(expected) or file_sha(path) != expected):
        raise ValueError('Storage repository source drift: ' + relative)
    return path


def validate_plan(value, root=None):
    if (type(value) is not dict or set(value) != LEGACY_FIELDS | {'purpose', 'control_spec'}
            or value['schema'] != SCHEMA or value['purpose'] != PURPOSE
            or not _exact(value['control_spec'], CONTROL_SPEC)):
        raise ValueError('Unknown storage control plan or scope')
    if digest({k: v for k, v in value.items() if k != 'plan_sha256'}) != value['plan_sha256']:
        raise ValueError('Storage plan digest changed')
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    hashes = value['source_hashes']
    if type(hashes) is not dict or set(hashes) != set(SOURCES):
        raise ValueError('Complete storage source vector required')
    for relative, expected in hashes.items():
        _source_file(root, relative, expected)
    # Reuse unchanged historical checkpoint/token/runner validation in memory.
    # Its synthetic digest is never persisted or passed to native hooks.
    common = {k: copy.deepcopy(v) for k, v in value.items() if k in LEGACY_FIELDS}
    common['schema'] = 'megartx-prefill-native-plan-v3'
    common['source_hashes'] = {p: hashes[p] for p in legacy.SOURCES}
    common['plan_sha256'] = digest({k: v for k, v in common.items() if k != 'plan_sha256'})
    legacy.validate_plan(common, root)
    validate_source_catalog(root, hashes)
    return value


def validate_source_catalog(root, hashes):
    """Admit one reviewed two-parent composition, never a per-file hash union."""
    # BEGIN ATTENTION SOURCE ADMISSION
    _source_file(root, attention_lineage.HELPER_SOURCE, ATTENTION_HELPER_SHA256)
    return attention_lineage.validate_storage_catalog(root, hashes)
    # END ATTENTION SOURCE ADMISSION
    # BEGIN MAP-BORROW SOURCE ADMISSION
    helper_sha = '7e4f093d9d0dc0c9a1cd9640ce3d59e3a2092144255326af3d02677f1d8233e7'
    _source_file(root, map_borrow.HELPER_SOURCE, helper_sha)
    helper_path = Path(map_borrow.__file__)
    if (not helper_path.is_absolute() or map_borrow.__spec__.origin != str(helper_path)
            or helper_path.resolve() != Path(__file__).with_name('m1_map_borrow_lineage.py').resolve()):
        raise ValueError('Unknown map-borrow helper origin')
    _source_file(helper_path.parent, helper_path.name, helper_sha)
    composition = map_borrow.load_catalog(root)
    def terminal(path, expected_hash):
        return map_borrow.terminal_sha(composition, path, expected_hash)
    # END MAP-BORROW SOURCE ADMISSION
    catalog = _read_json(Path(root) / CATALOG_SOURCE)
    if type(hashes) is not dict or set(hashes) != set(SOURCES):
        raise ValueError('Unknown, mixed, or incomplete storage source vector')
    for relative in (CATALOG_SOURCE, map_borrow.CATALOG_SOURCE, map_borrow.HELPER_SOURCE):
        _source_file(root, relative, hashes[relative])
    expected = {path: hashes[path] for path in SOURCES
                if path not in (CATALOG_SOURCE, map_borrow.CATALOG_SOURCE, map_borrow.HELPER_SOURCE)}
    if (catalog.get('schema') != 'megartx-prefill-current-main-source-reconciliation-v1'
            or catalog.get('purpose') != PURPOSE
            or any(catalog.get(k) != v for k, v in COMPOSITION_PARENTS.items())
            or catalog.get('parent_ledgers') != PARENT_LEDGERS
            or catalog.get('executing_helpers') != EXECUTING_HELPERS
            or {path: terminal(path, pin) for path, pin in catalog.get('runtime_source_hashes', {}).items()} != expected
            or any(catalog.get(key) is not False for key in
                   ('gpu_executed', 'gpu_authorized', 'numerical_qualified', 'quality_qualified', 'performance_qualified'))):
        raise ValueError('Unknown, mixed, or wrong-purpose storage source catalog')
    for path, expected_hash in PARENT_LEDGERS.items():
        _source_file(root, path, expected_hash)
    for path, expected_hash in EXECUTING_HELPERS.items():
        if hashes[path] != expected_hash:
            raise ValueError('Mixed or unreviewed purpose-selected ownership helper')
        _source_file(root, path, expected_hash)
    records = catalog.get('superseded_sources')
    if (type(records) is not list or any(type(row) is not dict or set(row) != {
            'path', 'main_parent_sha256', 'prefill_parent_sha256', 'current_sha256'} for row in records)):
        raise ValueError('Exact composition source lineage required')
    parent_records = [{k:v for k,v in row.items() if k != 'current_sha256'} for row in records]
    if digest(parent_records) != RECONCILED_PARENTS_SHA256:
        raise ValueError('Unknown, mixed, missing, or duplicate composition source lineage')
    changed = {row['path']:row['current_sha256'] for row in records
               if row['prefill_parent_sha256'] is not None
               and row['current_sha256'] != row['prefill_parent_sha256']}
    previous = {row['path']:row['prefill_parent_sha256'] for row in records if row['path'] in changed}
    if (catalog.get('changed_source_hashes') != changed
            or catalog.get('previous_source_hashes') != previous):
        raise ValueError('Composition terminal delta differs from exact parent lineage')
    for row in records:
        _source_file(root, row['path'], terminal(row['path'], row['current_sha256']))
    parent_catalog = _read_json(Path(root) / PREFILL_CATALOG_SOURCE)
    historical = catalog.get('unchanged_historical_catalogs')
    if (type(historical) is not dict or not historical
            or historical != parent_catalog.get('unchanged_historical_catalogs')):
        raise ValueError('Immutable historical catalog vector absent')
    for path, expected_hash in historical.items():
        if not path.startswith('docs/prefill/') or '..' in Path(path).parts:
            raise ValueError('Unknown historical catalog path')
        _source_file(root, path, expected_hash)
    equivalence = catalog.get('default_fit_equivalence')
    if (type(equivalence) is not dict
            or equivalence.get('method') != 'exact_parent_branch_ast_and_extracted_lifecycle'
            or equivalence.get('test_path') != 'tests/test_prefill_storage_compatibility.py'
            or set(equivalence.get('additional_tests', {})) != {'tests/test_shared_launcher_composition.py'}
            or set(equivalence.get('source_fixtures', {})) != {
                'tests/fixtures/shared-launcher-composition/main.source',
                'tests/fixtures/shared-launcher-composition/prefill.source'}):
        raise ValueError('Current-source default-fit equivalence binding absent')
    _source_file(root, equivalence['test_path'], equivalence.get('test_sha256'))
    for path, expected_hash in {**equivalence['additional_tests'], **equivalence['source_fixtures']}.items():
        _source_file(root, path, expected_hash)
    return catalog


def load_plan(path, root=None):
    return validate_plan(_read_json(path), root)


def freeze_plan(tokens, source_head, root, checkpoint, adapter_site=None):
    value = legacy.freeze_plan(tokens, source_head, root, checkpoint, adapter_site)
    value.update(schema=SCHEMA, purpose=PURPOSE, control_spec=copy.deepcopy(CONTROL_SPEC),
                 source_hashes={p: file_sha(Path(root) / p) for p in SOURCES})
    value['plan_sha256'] = digest({k: v for k, v in value.items() if k != 'plan_sha256'})
    validate_source_catalog(root, value['source_hashes'])
    return value


def require_clearance(path, plan):
    value = _read_json(path, 65536)
    expected = {'schema': 'megartx-prefill-storage-clearance-v1',
                'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
                'cpu_review_passed': True, 'parent_gpu_slot_clearance': True}
    if not _exact(value, expected):
        raise ValueError('Exact storage CPU review and parent GPU slot clearance required')
    return value


def verify_adapter_sources(plan):
    """Check actual imported helper origins and the independent checker file.

    This function imports only CPU-safe project modules. The provider verifies
    the numerical checker object it loads separately against this exact path.
    """
    origins = legacy.verify_adapter_sources(plan)
    for relative in (*EXTRA_SOURCES, map_borrow.HELPER_SOURCE, attention_lineage.HELPER_SOURCE):
        if not relative.startswith('src/megartx/'):
            continue
        name = 'megartx.' + Path(relative).stem
        module = importlib.import_module(name)
        expected = Path(plan['adapter_site']) / 'megartx' / Path(relative).name
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        actual = getattr(module, '__file__', None)
        if (not origin or not actual or Path(origin).resolve() != expected
                or Path(actual).resolve() != expected or expected.is_symlink()
                or file_sha(expected) != plan['source_hashes'][relative]):
            raise ValueError('Executing storage adapter source/origin drift: ' + name)
        origins[name] = {'origin': str(expected), 'sha256': plan['source_hashes'][relative]}
    source_root = Path(os.environ.get('MEGARTX_PREFILL_NATIVE_SOURCE_ROOT',
                                     Path(__file__).resolve().parents[2]))
    checker = _source_file(source_root, CHECKER_SOURCE, plan['source_hashes'][CHECKER_SOURCE])
    origins['prefill_native_control'] = {'origin': str(checker.resolve()),
                                        'sha256': plan['source_hashes'][CHECKER_SOURCE]}
    return origins


class CompactEvidence:
    """One cross-process total and metadata budget, including temporary files.

    All writers share one flock. Existing directory entries count before every
    exclusive create/append; hidden files and unknown temporary files count as
    metadata. Symlinks, directories, special files, and hardlinks are refused.
    Raw data uses exclusive .bf16 files, never JSON or a second unbudgeted copy.
    """
    def __init__(self, directory, limit=8 << 20, metadata_limit=2 << 20):
        if (type(limit) is not int or not 0 < limit <= 8 << 20
                or type(metadata_limit) is not int or not 0 < metadata_limit <= 2 << 20):
            raise ValueError('Storage evidence limits may only be reduced')
        self.directory, self.limit, self.metadata_limit = Path(directory), limit, metadata_limit
        if self.directory.is_symlink():
            raise ValueError('Evidence directory symlink refused')
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.directory.is_dir():
            raise ValueError('Evidence directory required')

    @staticmethod
    def _name(name, raw=False):
        if (type(name) is not str or Path(name).name != name or name.startswith('.')
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)):
            raise ValueError('Evidence filename must be a literal basename')
        if raw != name.endswith('.bf16') or (raw and name not in RAW_FILES):
            raise ValueError('Raw evidence requires an exclusive .bf16 filename')
        if not raw and not name.endswith(('.json', '.jsonl')):
            raise ValueError('Metadata evidence requires a JSON filename')

    def _sizes(self):
        # The owned launcher retains exactly one two-byte run.exit control
        # signal outside the evidence folder. Reserve it before every write.
        total = metadata = 2 if self.directory.name == 'prefill-storage' else 0
        for path in self.directory.iterdir():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Evidence symlink, hardlink, or nonregular entry refused')
            if path.name == FAILURE_FILE and info.st_size > FAILURE_RESERVE_BYTES:
                raise ValueError('Storage failure evidence exceeds its reserved extent')
            if path.name in RAW_FILES and info.st_size != RAW_FILES[path.name]:
                raise ValueError('Raw evidence file extent changed')
            total += info.st_size
            if path.name not in RAW_FILES:
                metadata += info.st_size
        return total, metadata

    def _write(self, name, data, *, append=False, raw=False, initial_data=b''):
        import fcntl
        self._name(name, raw)
        if type(data) is not bytes or not data:
            raise ValueError('Nonempty exact evidence bytes required')
        if name == FAILURE_FILE and (append or raw or len(data) > FAILURE_RESERVE_BYTES):
            raise ValueError('Exactly one bounded storage failure record is admitted')
        if raw and (append or len(data) != RAW_FILES[name]):
            raise ValueError('Raw BF16 evidence requires the exact selected row byte extent')
        # O_NOFOLLOW protects the lock before opening, rather than noticing a
        # symlink only after a lock has already been acquired on another file.
        lock_fd = os.open(self.directory / '.budget-lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            total, metadata = self._sizes()
            if initial_data and not (self.directory / name).exists():
                data = initial_data + data
            failure_exists = (self.directory / FAILURE_FILE).exists()
            if name == 'storage.json' and failure_exists:
                raise ValueError('Storage failure evidence forbids successful publication')
            reserved = FAILURE_RESERVE_BYTES if name != FAILURE_FILE and not failure_exists else 0
            if (total + len(data) + reserved > self.limit
                    or metadata + (0 if raw else len(data)) + reserved > self.metadata_limit):
                raise ValueError('Storage evidence overflow rejected before write')
            flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if append else os.O_EXCL)
            fd = os.open(self.directory / name, flags, 0o600)
            with os.fdopen(fd, 'ab' if append else 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(lock_fd)

    def write(self, name, value, append=False):
        if name == 'telemetry.jsonl':
            if (not append or type(value) is not dict
                    or set(value) != {'monotonic_ns', 'unix_ns', 'fields', 'values', 'exit'}
                    or value['fields'] != TELEMETRY_FIELDS
                    or any(type(value[k]) is not int or value[k] <= 0 for k in ('monotonic_ns', 'unix_ns'))
                    or type(value['exit']) is not int
                    or type(value['values']) is not list or not 1 <= len(value['values']) <= 8
                    or any(type(v) is not str or len(v) > 16 for v in value['values'])
                    or (value['exit'] == 0 and len(value['values']) != 8)):
                raise ValueError('Exact bounded GPU telemetry fields and values required')
            record = [value['monotonic_ns'], value['unix_ns'], value['values'], value['exit']]
            encode = lambda data: (json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n').encode()
            data = encode(record)
            if len(data) > 130:
                raise ValueError('Compact telemetry record exceeds admitted 130-byte bound')
            self._write(name, data, append=True, initial_data=encode(TELEMETRY_HEADER))
            return
        if name in {'frames.jsonl', 'head-invocations.jsonl', 'fit.json'}:
            raise ValueError('Legacy full transcript/fit is not storage-control evidence')
        if name == 'observer.json':
            if type(value) is not dict:
                raise ValueError('Storage observer requires a scalar record')
            value = {**value, 'observed_fit': False, 'independent_comparison_pending': True}
        data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        self._write(name, data, append=append)

    def raw(self, name, data):
        self._write(name, data, raw=True)

    def sizes(self):
        import fcntl
        fd = os.open(self.directory / '.budget-lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            total, metadata = self._sizes()
            if total > self.limit or metadata > self.metadata_limit:
                raise ValueError('Storage evidence budget already exceeded')
            return {'total_bytes': total, 'metadata_bytes': metadata}
        finally:
            os.close(fd)


Evidence = CompactEvidence


def validate_binding(plan, directory, owned_identities=None, require_live=True):
    from .prefill_runner_binding import validate_binding as validate_native_binding
    binding = validate_native_binding(plan, directory, owned_identities, require_live)
    # The additional storage hook binding is independently emitted by the new
    # provider. A historical observer alone can never admit this client's POST.
    value = _read_json(Path(directory) / 'storage-binding.json', 65536)
    expected = {'schema': 'megartx-prefill-storage-binding-v1',
                'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
                'owner_pid': binding['owner_pid'], 'owner_start_ticks': binding['owner_start_ticks'],
                'purpose': PURPOSE, 'writer_hook_bound': True,
                'checker_sha256': plan['source_hashes'][CHECKER_SOURCE]}
    if not _exact(value, expected):
        raise ValueError('Exact owned storage writer and checker binding required')
    return binding


def raw_manifest(directory):
    """Hash the exact private sample set without returning raw tensor values."""
    import struct
    directory = Path(directory)
    actual = {p.name for p in directory.iterdir() if p.name.endswith('.bf16')}
    if actual != set(RAW_FILES):
        raise ValueError('Exact all-layer selected K/V and two raw head files required')
    result = {}
    for name, size in sorted(RAW_FILES.items()):
        path = directory / name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != size):
            raise ValueError('Selected raw BF16 file extent/ownership changed: ' + name)
        data = path.read_bytes()
        if len(data) != size or any(word & 0x7f80 == 0x7f80 for (word,) in struct.iter_unpack('<H', data)):
            raise ValueError('Selected raw BF16 file contains nonfinite or incomplete values: ' + name)
        result[name] = {'bytes': size, 'sha256': hashlib.sha256(data).hexdigest()}
    return result


CONTROL_COUNTS = {
    'capture_frames': 9, 'capture_end': 2049, 'checked_layer_frames': 270,
    'pre_rows': 212355, 'processed_rows': 61470, 'post_rows': 273825,
    'transferred_bytes': 4024934400, 'metadata_only_decode_inputs': 254,
    'sample_combined_kv_rows': 180, 'raw_head_rows': 2,
}
QUALIFICATIONS = {
    'sampled_repeatability_qualified': False, 'independent_arithmetic_qualified': False,
    'quality_qualified': False, 'numerical_qualified': False, 'performance_qualified': False,
}


def validate_control(plan, value, output_ids_sha256=None):
    fixed = {'schema': 'megartx-prefill-storage-control-v1',
             'plan_sha256': plan['plan_sha256'], 'status': 'storage_frontier_observed',
             'storage_exact': True, 'frontier_verified': True, 'storage_callbacks_restored': True,
             **CONTROL_COUNTS, **QUALIFICATIONS}
    if type(value) is not dict or any(not _exact(value.get(key), expected) for key, expected in fixed.items()):
        raise ValueError('Complete exact storage/frontier control proof required')
    for key in ('raw_samples_sha256', 'frame_records_sha256', 'head_records_sha256',
                'sample_records_sha256', 'sample_hashes_sha256'):
        if not _sha(value.get(key)):
            raise ValueError('Storage all-row/head/sample root is absent: ' + key)
    frontier = value.get('frontier')
    frontier_fixed = {'checked_heads': 263, 'emitted_tokens': 256, 'decode_inputs': 255,
                      'committed_length': 2303, 'uncached_output_position': 2303,
                      'token_ids_encoding': 'canonical-compact-json-integer-array-utf8'}
    if (type(frontier) is not dict or set(frontier) != set(frontier_fixed) | {'token_ids_sha256'}
            or any(not _exact(frontier.get(key), expected) for key, expected in frontier_fixed.items())
            or not _sha(frontier.get('token_ids_sha256'))
            or (output_ids_sha256 is not None and frontier['token_ids_sha256'] != output_ids_sha256)):
        raise ValueError('Independent frontier/token transport proof differs')
    roots = value.get('storage_roots')
    if (type(roots) is not dict or set(roots) != {'pre', 'processed', 'post'}
            or any(not _sha(v) for v in roots.values())
            or not _sha(value.get('actual_sampling_params_sha256'))
            or type(value.get('processed_digest_host_bytes')) is not int
            or value['processed_digest_host_bytes'] <= 0
            or value.get('natural_positive_correction_coverage', 'missing') is not None):
        raise ValueError('Bounded all-row roots and actual sampler identity required')
    transfer = value.get('transfer')
    domains = transfer.get('domains') if type(transfer) is dict else None
    expected_domains = {'pre': 1550868480, 'processed': 461598720, 'post': 2012467200,
                        'raw_heads': 1048576, 'head_index': 2104,
                        'selected_hidden': 2962432, 'head_finite_scalar': 261,
                        'writer_slot_metadata': 552720, 'inherited_metadata_upper_bound': 41811968}
    if (type(transfer) is not dict
            or set(transfer) != {'limit_bytes', 'transferred_bytes', 'copy_calls', 'domains'}
            or not _exact(transfer.get('limit_bytes'), 4 << 30)
            or type(transfer.get('transferred_bytes')) is not int
            or not 4024934400 <= transfer['transferred_bytes'] <= 4 << 30
            or type(transfer.get('copy_calls')) is not int or transfer['copy_calls'] <= 0
            or type(domains) is not dict
            or set(domains) != set(expected_domains) | {'manager_table_metadata'}
            or any(not _exact(domains.get(k), v) for k, v in expected_domains.items())
            or any(type(v) is not int or v <= 0 for v in domains.values())
            or sum(domains.values()) != transfer['transferred_bytes']):
        raise ValueError('Complete exact D2H domains and shared transfer cap required')
    manager = value.get('manager')
    expected_manager_keys = {'manager_block_tokens', 'kernel_block_tokens', 'manager_to_kernel_ratios',
                             'append_calls', 'manager_table_sha256', 'expected_address_source'}
    if (type(manager) is not dict or set(manager) != expected_manager_keys
            or manager['expected_address_source'] != 'actual_append_block_ids_before_native_subdivision'
            or not _sha(manager['manager_table_sha256'])
            or type(manager['append_calls']) is not int or manager['append_calls'] <= 0):
        raise ValueError('Actual pre-subdivision manager provenance required')
    sizes, kernels, ratios = (manager[k] for k in (
        'manager_block_tokens', 'kernel_block_tokens', 'manager_to_kernel_ratios'))
    if (any(type(v) is not list for v in (sizes, kernels, ratios))
            or not 1 <= len(sizes) <= 30 or len(sizes) != len(kernels) or len(sizes) != len(ratios)
            or any(type(b) is not int or type(k) is not int or type(r) is not int
                   or k != 16 or r < 1 or b != k*r for b, k, r in zip(sizes, kernels, ratios))):
        raise ValueError('Actual manager/kernel subdivision geometry differs')
    # Each group is reconciled before and after all263frames; both persistent
    # and gathered tables are copied. Counts cannot undercut required pages or
    # exceed capacity144. Exact source-derived lower/upper bounds per group.
    if not 566016 * len(sizes) <= domains['manager_table_metadata'] <= 585984 * len(sizes):
        raise ValueError('Actual manager table transfer allowance is outside source-derived bounds')
    return value


def _records(directory, name, count):
    path = Path(directory) / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 << 20:
        raise ValueError('Bounded compact storage transcript required: ' + name)
    records = []
    with path.open() as stream:
        for line in stream:
            if len(line) > 65536 or len(records) >= count:
                raise ValueError('Compact storage transcript record/count overflow: ' + name)
            record = json.loads(line, object_pairs_hook=_json_pairs)
            if type(record) is not dict:
                raise ValueError('Compact storage transcript requires records: ' + name)
            records.append(record)
    if len(records) != count:
        raise ValueError('Complete compact storage transcript required: ' + name)
    return records


def _validate_transcripts(plan, directory, control, samples):
    frames = _records(directory, 'control-frames.jsonl', 263)
    heads = _records(directory, 'heads.jsonl', 263)
    if (control['frame_records_sha256'] != file_sha(Path(directory) / 'control-frames.jsonl')
            or control['head_records_sha256'] != file_sha(Path(directory) / 'heads.jsonl')):
        raise ValueError('Compact frame/head transcript root changed')
    for sequence, (frame, head) in enumerate(zip(frames, heads)):
        start = sequence * 256 if sequence < 8 else 2048 + sequence - 8
        end = start + (256 if sequence < 8 else 1)
        expected_frame = {'sequence': sequence, 'start': start, 'end': end,
                          'queries_complete': True}
        if any(not _exact(frame.get(k), v) for k, v in expected_frame.items()):
            raise ValueError('Compact storage frame identity/order differs')
        if not _sha(frame.get('input_ids_sha256')):
            raise ValueError('Compact storage frame input identity absent')
        if sequence < 8 and frame['input_ids_sha256'] != digest(plan['tokens'][start:end]):
            raise ValueError('Prompt frame input digest differs from frozen private prompt')
        expected_head = {'sequence': sequence, 'logit_position': end - 1,
                         'predicts_position': end, 'expected_sampler_discard': sequence < 7}
        if (any(not _exact(head.get(k), v) for k, v in expected_head.items())
                or not _sha(head.get('hidden_bits_sha256'))
                or (head.get('logit_bits_sha256') is not None and not _sha(head['logit_bits_sha256']))):
            raise ValueError('Compact native head identity/order differs')
        expected_frame.update(plan_sha256=plan['plan_sha256'], storage_checked=sequence < 9)
        if any(not _exact(frame.get(k), v) for k, v in expected_frame.items()):
            raise ValueError('Storage checked/metadata-only frame coverage differs')
        if (head.get('plan_sha256') != plan['plan_sha256']
                or head.get('before_sampler_transforms') is not True
                or not _exact(head.get('selected_row_index'), 255 if sequence < 8 else 0)
                or head.get('native_logits_dtype') != 'torch.bfloat16'
                or not _exact(head.get('suppressed_token_count'), 0)):
            raise ValueError('Actual pre-sampler native head selection/dtype differs')
        if end - 1 in (2047, 2048):
            name = f'head-position-{end - 1}.bf16'
            if head.get('logit_bits_sha256') != samples[name]['sha256']:
                raise ValueError('Retained raw head differs from native head event')
        elif head.get('logit_bits_sha256') is not None:
            raise ValueError('Only the two admitted raw native head rows may be retained')
    expected_counts = {'pre': 0, 'processed': 0, 'post': 0}
    for frame in frames:
        if frame['storage_checked']:
            for layer in range(30):
                low = 0 if layer % 6 == 5 else max(0, frame['start'] - 1023)
                expected_counts['pre'] += frame['start'] - low
                expected_counts['processed'] += frame['end'] - frame['start']
                expected_counts['post'] += frame['end'] - low
            roots = frame.get('storage_roots')
            if (not _exact(frame.get('storage_counts'), expected_counts)
                    or type(roots) is not dict or set(roots) != set(expected_counts)
                    or any(not _sha(v) for v in roots.values())):
                raise ValueError('Complete all-layer per-frame counts/roots required')
        elif frame.get('storage_counts') is not None or frame.get('storage_roots') is not None:
            raise ValueError('Later continuation must remain metadata-only')
    if frames[8]['storage_roots'] != control['storage_roots']:
        raise ValueError('Storage final checked roots differ from completed capture')
    emissions = _records(directory, 'samples.jsonl', 256)
    if control['sample_records_sha256'] != file_sha(Path(directory) / 'samples.jsonl'):
        raise ValueError('Scalar sample transcript root changed')
    sample_hashes = []
    for index, event in enumerate(emissions):
        expected = {'output_index': index, 'head_sequence': index + 7,
                    'head_phase': 'final_prompt' if index == 0 else 'decode',
                    'cached_length': 2048 + index, 'pending_anchor_position': 2048 + index,
                    'anchor_kv_written': False}
        if (set(event) != set(expected) | {'sample_sha256'}
                or any(not _exact(event.get(k), v) for k, v in expected.items())
                or not _sha(event.get('sample_sha256'))):
            raise ValueError('Exact ordered native output/head/uncached-anchor sample bindings required')
        sample_hashes.append(event['sample_sha256'])
        if index < 255 and event['sample_sha256'] != frames[8 + index]['input_ids_sha256']:
            raise ValueError('Consumed decode input differs from preceding native sample')
    if digest(sample_hashes) != control['sample_hashes_sha256']:
        raise ValueError('Ordered native sample scalar digest root changed')
    return frames, heads



def validate_telemetry(directory):
    path = Path(directory) / 'telemetry.jsonl'
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 << 20:
        raise ValueError('Complete bounded compact GPU telemetry required')
    count, previous, free_min = 0, None, None
    with path.open() as stream:
        header = stream.readline(65537)
        if len(header) > 65536 or not _exact(json.loads(header), TELEMETRY_HEADER):
            raise ValueError('Compact telemetry field schema changed')
        for line in stream:
            if len(line) > 130 or count >= 9000:
                raise ValueError('Compact telemetry record budget changed')
            record = json.loads(line)
            if (type(record) is not list or len(record) != 4
                    or any(type(v) is not int or v <= 0 for v in record[:2])
                    or not _exact(record[3], 0)
                    or type(record[2]) is not list or len(record[2]) != 8
                    or any(type(v) is not str or len(v) > 16 for v in record[2])
                    or (previous is not None and record[0] <= previous)):
                raise ValueError('Telemetry loss, failure, or timestamp disorder')
            try:
                free = int(record[2][1])
            except (TypeError, ValueError) as error:
                raise ValueError('Exact GPU free-memory telemetry required') from error
            if free < 2048:
                raise ValueError('GPU free-memory floor violated')
            free_min = free if free_min is None else min(free_min, free)
            previous = record[0]
            count += 1
    if not count:
        raise ValueError('Telemetry observations are missing')
    return {'schema': TELEMETRY_HEADER['schema'], 'records': count,
            'minimum_gpu_free_mib': free_min, 'sha256': file_sha(path)}


def publish_storage(evidence, plan, ownership):
    """Publish only this storage scope after exact token proof and owned cleanup.

    Never writes fit.json, reuses a historical receipt, or upgrades storage to
    sampled repeatability, independent arithmetic, quality, or performance.
    """
    directory = evidence.directory
    if (directory / FAILURE_FILE).exists():
        raise ValueError('Storage failure evidence forbids successful publication')
    evidence.sizes()
    names = ('observer.json', 'client.json', 'loaded.json', 'geometry.json',
             'client-stream.json', 'control.json', 'storage-client.json')
    observer, client, loaded, geometry, stream, control, mode = [
        _read_json(directory / name, 65536 if name != 'loaded.json' else 1 << 20) for name in names]
    binding = validate_binding(plan, directory, require_live=False)
    comparison = native_ledger_comparison(plan, client, observer, stream)
    if comparison['failed_fields']:
        raise ValueError('Storage native/client transport ledger mismatch: ' + ','.join(comparison['failed_fields']))
    validate_control(plan, control, client['output_ids_sha256'])
    expected_mode = {'schema': 'megartx-prefill-storage-client-v1', 'purpose': PURPOSE,
                     'plan_sha256': plan['plan_sha256'], 'status': 'complete',
                     'control_sha256': digest(control), 'output_ids_sha256': client['output_ids_sha256'],
                     'sample_hashes_sha256': control['sample_hashes_sha256'],
                     'storage_capture_end': 2049, 'metadata_only_decode_inputs': 254,
                     'numerical_qualified': False, 'performance_qualified': False}
    if not _exact(mode, expected_mode):
        raise ValueError('Storage-specific client completion receipt differs')
    if (type(ownership) is not dict
            or not {'cleanup_complete', 'failure', 'owned_identities_remaining',
                    'owned_gpu_pids_remaining', 'cleanup_errors'} <= set(ownership)
            or ownership.get('cleanup_complete') is not True
            or ownership.get('failure') is not None
            or any(ownership.get(key) != [] for key in ('owned_identities_remaining',
                   'owned_gpu_pids_remaining', 'cleanup_errors'))
            or loaded.get('plan_sha256') != plan['plan_sha256']
            or loaded.get('mutable_lease_granted') is not False
            or not _exact(loaded.get('identity', {}).get('owner_pid'), binding['owner_pid'])
            or not _exact(loaded.get('identity', {}).get('owner_start_ticks'), binding['owner_start_ticks'])
            or geometry.get('physical_policy') != 'full_context'
            or not _exact(geometry.get('capacity_tokens'), 2304)
            or geometry.get('actual_owned_page_ranges_disjoint') is not True):
        raise ValueError('Storage publication requires bound loaded owner and complete owned cleanup')
    scratch = observer.get('observer_gpu_scratch', {})
    cap = plan['bounds']['max_observer_gpu_scratch_bytes']
    if (scratch.get('domain') != 'incremental_gpu_allocator_bytes'
            or not _exact(scratch.get('cap_bytes'), cap)
            or type(scratch.get('measured_phases')) is not int or scratch['measured_phases'] < 789
            or any(type(scratch.get(k)) is not int or not 0 <= scratch[k] <= cap for k in (
                'managed_tensor_simultaneous_peak_bytes', 'measured_phase_allocator_increment_peak_bytes',
                'measured_phase_reserved_increment_peak_bytes'))
            or scratch.get('counter_policy') != 'process_global_resets_with_explicit_runwide_peak_preservation'
            or any(type(scratch.get(k)) is not int or scratch[k] < 0 for k in (
                'runwide_allocator_allocated_peak_bytes', 'runwide_allocator_reserved_peak_bytes'))
            or scratch.get('host_heap_excluded') is not True):
        raise ValueError('Storage publication requires complete bounded GPU scratch telemetry')
    samples = raw_manifest(directory)
    if control['raw_samples_sha256'] != digest(samples):
        raise ValueError('Retained raw storage sample root changed')
    _validate_transcripts(plan, directory, control, samples)
    telemetry = validate_telemetry(directory)
    sizes = evidence.sizes()
    evidence.write('storage.json', {
        'schema': 'megartx-prefill-storage-receipt-v1', 'status': 'observed_storage_frontier',
        'purpose': PURPOSE, 'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
        'control_spec_sha256': digest(plan['control_spec']), 'control_sha256': digest(control),
        'prompt_tokens': 2048, 'chunk_tokens': 256, 'capacity_tokens': 2304,
        'contexts': 1, 'emitted_outputs': 256, 'decode_input_rows': 255,
        'storage_capture_end': 2049, 'metadata_only_decode_inputs': 254,
        'storage_exact': True, 'frontier_verified': True, 'cleanup_complete': True,
        'source_and_owner_bound': True, 'same_path_reference': 'unavailable',
        **QUALIFICATIONS, 'observer_gpu_scratch': scratch, 'telemetry': telemetry,
        'evidence_bytes_before_publication': sizes,
        'evidence_files_sha256': {p.name: file_sha(p) for p in directory.iterdir()
            if p.is_file() and not p.name.startswith('.') and p.name != 'request.json'},
    })

"""Owned stdlib-only CPU analysis of the frozen attention capture.

The caller must finish the current native storage/frontier/client/cleanup gates
first and supply their current independently established identities. This
launcher cannot replace those gates. The native caller also owns full runtime
source-plan validation before and after this standalone CPU worker. It reports
errors, never acceptance.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from . import prefill_attention_plan as plan
from .prefill_attention_validation import read_capture

HELPER_PATH = 'scripts/prefill_owned_processes.py'
HELPER_SHA256 = 'df01a29a8089db09530eef3d36efc04d2871033b4fa6b357eb73a0aca99dae4b'
MAX_RSS_BYTES = 512 << 20
MAX_SECONDS = 300
MAX_RESULT_BYTES = 512 << 10
MAX_ERROR_BYTES = 16384
HOST_FREE_FLOOR_BYTES = 8 << 30
MAX_MEMINFO_BYTES = 65536


def _proc_kib_field(path, field):
    """Read one exact KiB field from a bounded Linux proc text record."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        chunks = []
        remaining = MAX_MEMINFO_BYTES + 1
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b''.join(chunks)
    finally:
        os.close(fd)
    if len(raw) > MAX_MEMINFO_BYTES:
        raise ValueError('Bounded proc memory observation overflow')
    fields = [line.split() for line in raw.decode('ascii').splitlines()
              if line.startswith(field + ':')]
    if (len(fields) != 1 or len(fields[0]) != 3 or fields[0][2] != 'kB'
            or not fields[0][1].isascii() or not fields[0][1].isdigit()):
        raise ValueError('Exact proc ' + field + ' kB observation required')
    return int(fields[0][1]) * 1024


def _host_available_bytes(path='/proc/meminfo'):
    """Bounded Linux MemAvailable observation; missing data fails closed."""
    return _proc_kib_field(path, 'MemAvailable')


def _worker_peak_rss_bytes(path='/proc/self/status'):
    """Peak of the current post-exec address space, not inherited ru_maxrss.

    Linux getrusage can retain the pre-exec parent's RSS highwater. VmHWM is
    attached to the new mm after exec, matching this isolated CPU worker.
    """
    return _proc_kib_field(path, 'VmHWM')


def _require_host_free():
    available = _host_available_bytes()
    if available < HOST_FREE_FLOOR_BYTES:
        raise ValueError('CPU analysis host MemAvailable below 8 GiB floor')
    return available


def _helper(ownership, root):
    """Bind the actual imported helper object, not an adjacent lookalike."""
    module = sys.modules.get(type(ownership).__module__)
    expected = root / HELPER_PATH
    if (module is None or getattr(module, 'OwnedProcesses', None) is not type(ownership)
            or Path(getattr(module, '__file__', '')).absolute() != expected
            or getattr(getattr(module, '__spec__', None), 'origin', None) != str(expected)
            or plan._source_hash(expected) != HELPER_SHA256):
        raise ValueError('Analysis requires the exact admitted executing prefill ownership helper')
    return module


def _allowance(deadline, now):
    if type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError('Shared monotonic lifecycle deadline required')
    seconds = min(MAX_SECONDS, math.floor(deadline - now))
    if seconds < 1:
        raise ValueError('Shared lifecycle has no remaining CPU analysis allowance')
    return seconds, min(deadline, now + seconds)


def _assert_result(result, manifest):
    if (type(result) is not dict or result.get('schema') != 'megartx-prefill-attention-errors-v1'
            or result.get('status') != 'independent_attention_errors_computed'
            or result.get('input_manifest') != manifest
            or result.get('native_arithmetic_acceptance', 'missing') is not None
            or any(result.get(key) is not False for key in
                   ('numerical_qualified', 'quality_qualified', 'performance_qualified',
                    'native_execution_attested', 'sampled_repeatability_qualified'))
            or result.get('external_native_storage_frontier_validation_required') is not True
            or type(result.get('cases')) is not list or len(result['cases']) != 60
            or result.get('aggregate', {}).get('coordinates') != 480):
        raise ValueError('Independent oracle manifest, sample coverage or non-acceptance contract differs')


def run_analysis(evidence, specification, *, expected_request_sha256,
                 expected_storage_binding_sha256, deadline, ownership, source_root=None):
    """Return metrics/resources; publication remains the current caller's job.

    A separate owned child uses the same source-pinned ownership implementation;
    it is not falsely counted as a descendant of the already stopped server.
    RLIMIT_AS is a hard address-space ceiling and thus a conservative RSS bound.
    Linux RLIMIT_RSS is deliberately unused. Parent wall time includes startup,
    read/validation, arithmetic, result transport, and process completion.
    """
    root = Path(source_root or Path(__file__).resolve().parents[2]).absolute()
    module = _helper(ownership, root)
    started = time.monotonic()
    seconds, stop = _allowance(deadline, started)
    minimum_host_available = _require_host_free()
    host_samples = 1
    initial = read_capture(evidence.directory, specification,
        expected_request_sha256=expected_request_sha256,
        expected_storage_binding_sha256=expected_storage_binding_sha256, source_root=root)
    capture_digest = plan.digest(initial['capture'])
    manifest = initial['capture']['raw_manifest']
    del initial
    if time.monotonic() >= stop:
        raise ValueError('Analysis validation consumed its remaining shared deadline')
    worker = root / 'scripts/prefill_attention_analyze.py'
    if worker.is_symlink() or not worker.is_file():
        raise ValueError('Exact isolated analysis worker source required')
    supervisor = module.read_process(os.getpid())
    if supervisor.pid != os.getpid():
        raise ValueError('Current analysis supervisor identity unavailable')
    owned = module.OwnedProcesses(supervisor)
    process = None
    cleanup = None
    primary = None
    output = bytearray()
    errors = bytearray()
    identity = None
    try:
        minimum_host_available = min(minimum_host_available, _require_host_free())
        host_samples += 1
        # -I -S excludes ambient PYTHONPATH, site initialization and user hooks.
        # stdin remains unopened by the worker until admission and registration.
        process = subprocess.Popen([sys.executable, '-I', '-S', str(worker),
            str(seconds), str(stop)], cwd=root, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
            env={'PATH': os.defpath, 'LANG': 'C.UTF-8'})
        identity = module.read_process(process.pid)
        if identity.ppid != supervisor.pid:
            raise ValueError('Analysis child is not owned by the current launcher')
        owned.register(identity)
        packet = {'directory': str(evidence.directory), 'specification': specification,
            'request_sha256': expected_request_sha256,
            'storage_sha256': expected_storage_binding_sha256,
            'capture_sha256': capture_digest, 'source_root': str(root)}
        raw = json.dumps(packet, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > 65536:
            raise ValueError('Analysis invocation metadata exceeds fixed bound')
        # Invocation transport is also deadline bounded: never block in a
        # large pipe write while a child is starting or has stopped reading.
        os.set_blocking(process.stdin.fileno(), False)
        sent = 0
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE, None)
            selector.register(process.stdout, selectors.EVENT_READ, (output, MAX_RESULT_BYTES))
            selector.register(process.stderr, selectors.EVENT_READ, (errors, MAX_ERROR_BYTES))
            while selector.get_map():
                remaining = stop - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Owned CPU reference exceeded shared/analysis wall deadline')
                minimum_host_available = min(minimum_host_available, _require_host_free())
                host_samples += 1
                alive = owned.observe(module.snapshot(), time.monotonic())
                if any(p.identity != identity.identity for p in alive):
                    raise ValueError('Unexpected descendant of stdlib-only CPU analysis worker')
                if any(p.rss_bytes > MAX_RSS_BYTES for p in alive):
                    raise ValueError('Analysis process RSS exceeded its hard resource contract')
                for key, _ in selector.select(min(.05, remaining)):
                    if key.data is None:
                        count = os.write(key.fileobj.fileno(), raw[sent:sent + 4096])
                        if count <= 0:
                            raise OSError('Short analysis invocation transport')
                        sent += count
                        if sent == len(raw):
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                        continue
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target, limit = key.data
                    if len(target) + len(chunk) > limit:
                        raise ValueError('Bounded analysis output transport overflow')
                    target.extend(chunk)
        remaining = stop - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Analysis result arrived after its shared deadline')
        process.wait(timeout=remaining)
        if process.returncode != 0:
            raise ValueError('CPU reference unresolved: worker exit ' + str(process.returncode)
                             + (': ' + errors.decode('utf-8', 'replace')[:2048] if errors else ''))
        if errors:
            raise ValueError('Unexpected analysis worker stderr')
        value = json.loads(output)
        if type(value) is not dict or set(value) != {'result', 'resources', 'capture_sha256'}:
            raise ValueError('Exact analysis worker result envelope required')
        result, resources = value['result'], value['resources']
        _assert_result(result, manifest)
        if (value['capture_sha256'] != capture_digest or type(resources) is not dict
                or resources.get('address_space_limit_bytes') != MAX_RSS_BYTES
                or resources.get('memory_enforcement') != 'linux_hard_RLIMIT_AS_bounds_process_RSS'
                or resources.get('cpu_seconds_limit') != seconds
                or resources.get('reference_source_sha256') != specification['source_hashes'][
                    'numerical_reference/prefill_attention_reference.py']
                or resources.get('reference_source_origin') != str(root / 'numerical_reference/prefill_attention_reference.py')
                or resources.get('rss_measurement') != 'linux_proc_self_status_VmHWM_post_exec_mm'
                or type(resources.get('lifetime_including_pre_exec_peak_rss_bytes')) is not int
                or resources['lifetime_including_pre_exec_peak_rss_bytes'] < 0
                or resources.get('lifetime_rss_measurement') != 'linux_getrusage_RUSAGE_SELF_ru_maxrss_may_include_pre_exec'
                or type(resources.get('peak_process_rss_bytes')) is not int
                or not 0 < resources['peak_process_rss_bytes'] <= MAX_RSS_BYTES):
            raise ValueError('Owned analysis resource/source binding differs')
        # Reject stale archive swaps even if a worker computed internally
        # consistent results. Current validated request/storage are caller inputs.
        current = read_capture(evidence.directory, specification,
            expected_request_sha256=expected_request_sha256,
            expected_storage_binding_sha256=expected_storage_binding_sha256, source_root=root)
        if (plan.digest(current['capture']) != capture_digest
                or current['capture']['raw_manifest'] != result['input_manifest']):
            raise ValueError('Capture/raw manifest changed during CPU analysis')
        if time.monotonic() >= stop:
            raise TimeoutError('Analysis final validation exceeded shared/analysis wall deadline')
    except BaseException as error:
        primary = error
        raise
    finally:
        if process is not None:
            # The pinned helper signals only PID/start-time identities through
            # pidfds. No broad process-group kill and no GPU query are used.
            cleanup_errors = []
            def cleanup_problem(error, operation):
                message = operation + ': ' + type(error).__name__ + ': ' + str(error)
                if primary is not None:
                    module.preserve_primary(primary, message)
                else:
                    cleanup_errors.append(error)
            try:
                cleanup = owned.cleanup(lambda: [], process.poll, term_seconds=1, kill_seconds=1)
                if not cleanup['cleanup_complete']:
                    cleanup_problem(RuntimeError('Owned CPU analysis cleanup incomplete'), 'owned cleanup')
            except BaseException as error:
                cleanup_problem(error, 'owned cleanup raised')
                # If cleanup itself was interrupted, one identity-safe kill is
                # still attempted; its errors cannot erase the primary either.
                try:
                    owned.stop(module.signal.SIGKILL)
                    process.wait(timeout=1)
                except BaseException as stop_error:
                    cleanup_problem(stop_error, 'owned cleanup fallback')
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        if not stream.closed:
                            stream.close()
                    except BaseException as error:
                        cleanup_problem(error, 'analysis stream close')
            if primary is None and cleanup_errors:
                first = cleanup_errors[0]
                for extra in cleanup_errors[1:]:
                    module.preserve_primary(first, 'Additional analysis cleanup error: ' + repr(extra))
                raise first
    minimum_host_available = min(minimum_host_available, _require_host_free())
    host_samples += 1
    if time.monotonic() >= stop:
        raise TimeoutError('Owned analysis cleanup exhausted the shared/analysis wall deadline')
    resources.update({'wall_seconds': time.monotonic() - started,
        'wall_seconds_limit': seconds, 'shared_deadline_monotonic': deadline,
        'host_available_policy': 'bounded_proc_meminfo_before_launch_and_each_owned_loop',
        'host_free_floor_bytes': HOST_FREE_FLOOR_BYTES,
        'minimum_sampled_host_available_bytes': minimum_host_available,
        'host_available_samples': host_samples,
        'ownership': cleanup, 'child_pid': identity.pid, 'child_start_ticks': identity.start_ticks})
    return {'result': result, 'resources': resources,
            'capture_sha256': capture_digest, 'request_identity_sha256': expected_request_sha256,
            'inherited_storage_binding_sha256': expected_storage_binding_sha256,
            'input_manifest_matches_validated_raw_manifest': True}


def worker(seconds, deadline):
    """Called only by the isolated worker after kernel limits are installed."""
    import importlib.util
    import resource
    if (sys.platform != 'linux' or type(seconds) is not int or not 0 < seconds <= MAX_SECONDS
            or resource.getrlimit(resource.RLIMIT_AS) != (MAX_RSS_BYTES, MAX_RSS_BYTES)
            or resource.getrlimit(resource.RLIMIT_CPU) != (seconds, seconds)):
        raise ValueError('Required hard worker resource limits are not in effect')
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError('Analysis invocation overflow')
    value = json.loads(raw)
    expected = {'directory', 'specification', 'request_sha256', 'storage_sha256',
                'capture_sha256', 'source_root'}
    if type(value) is not dict or set(value) != expected:
        raise ValueError('Exact analysis invocation required')
    root = Path(value['source_root']).absolute()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError('Worker source root differs from executing analysis module')
    packet = read_capture(value['directory'], value['specification'],
        expected_request_sha256=value['request_sha256'],
        expected_storage_binding_sha256=value['storage_sha256'], source_root=root)
    if plan.digest(packet['capture']) != value['capture_sha256']:
        raise ValueError('Worker capture differs from caller-validated capture')
    path = root / 'numerical_reference/prefill_attention_reference.py'
    sha = value['specification']['source_hashes']['numerical_reference/prefill_attention_reference.py']
    if plan._source_hash(path) != sha:
        raise ValueError('Frozen independent oracle changed')
    spec = importlib.util.spec_from_file_location('megartx_independent_attention_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if (Path(reference.__file__).absolute() != path or reference.__spec__.origin != str(path)
            or plan._source_hash(path) != sha):
        raise ValueError('Executing independent oracle origin/source changed')
    remaining, _ = _allowance(deadline, time.monotonic())
    result = reference.analyze_arrays(packet['arrays'], max_seconds=remaining)
    _assert_result(result, packet['capture']['raw_manifest'])
    resources = {'memory_enforcement': 'linux_hard_RLIMIT_AS_bounds_process_RSS',
        'address_space_limit_bytes': MAX_RSS_BYTES, 'cpu_seconds_limit': seconds,
        'peak_process_rss_bytes': _worker_peak_rss_bytes(),
        'rss_measurement': 'linux_proc_self_status_VmHWM_post_exec_mm',
        'lifetime_including_pre_exec_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'lifetime_rss_measurement': 'linux_getrusage_RUSAGE_SELF_ru_maxrss_may_include_pre_exec',
        'reference_source_sha256': sha, 'reference_source_origin': str(path)}
    output = json.dumps({'result': result, 'resources': resources,
                         'capture_sha256': value['capture_sha256']},
                        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(output) > MAX_RESULT_BYTES or time.monotonic() >= deadline:
        raise ValueError('Analysis result exceeded its output or wall bound')
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()

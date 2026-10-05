"""Owned isolated CPU-reference controls; all operands are synthetic zeros."""
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from megartx.prefill_attention_analysis import (
    _allowance, _assert_result, _helper, _host_available_bytes, _require_host_free,
    _worker_peak_rss_bytes,
    HOST_FREE_FLOOR_BYTES, MAX_RSS_BYTES, run_analysis,
)
from megartx.prefill_attention_evidence import StreamingEvidence
from test_prefill_attention_cpu import fixture, REQUEST, STORAGE

ROOT = Path(__file__).resolve().parents[1]


def helper():
    path = ROOT / 'scripts/prefill_owned_processes.py'
    spec = importlib.util.spec_from_file_location('test_attention_owned_processes', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalysisPolicyTests(unittest.TestCase):
    def setUp(self):
        # CI hosts need not meet the native-host floor. Only this host signal is
        # synthetic; full worker arithmetic, kernel limits and cleanup are real.
        self.host = patch('megartx.prefill_attention_analysis._host_available_bytes',
                          return_value=HOST_FREE_FLOOR_BYTES + (1 << 30))
        self.host.start()
        self.addCleanup(self.host.stop)

    def test_bounded_memavailable_parser_and_missing_duplicate_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'meminfo'
            path.write_bytes(b'MemTotal: 10485760 kB\nMemAvailable: 8388608 kB\n')
            self.assertEqual(_host_available_bytes(path), HOST_FREE_FLOOR_BYTES)
            for data in (b'MemFree: 8388608 kB\n', b'MemAvailable: -1 kB\n',
                         b'MemAvailable: 8388608 MB\n', b'MemAvailable: 8388608 kB\n' * 2,
                         b'X' * 65537):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    _host_available_bytes(path)

    def test_postexec_peak_uses_bounded_vmhwm_and_rejects_ambiguous_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status'
            path.write_bytes(b'VmRSS: 8192 kB\nVmHWM: 16384 kB\n')
            self.assertEqual(_worker_peak_rss_bytes(path), 16384 * 1024)
            for data in (b'VmRSS: 8192 kB\n', b'VmHWM: -1 kB\n',
                         b'VmHWM: 16384 MB\n', b'VmHWM: 16384 kB\n' * 2,
                         b'X' * 65537):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    _worker_peak_rss_bytes(path)

    def test_exec_discards_parent_vmhwm_but_getrusage_retains_it(self):
        # 64MiB suffices to distinguish inherited highwater from the post-exec
        # worker footprint, without allocating anywhere near the 512MiB cap.
        child = ('import resource,sys,json; '
                 'resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20)); '
                 'sys.path.insert(0,' + repr(str(ROOT / 'src')) + '); '
                 'from megartx.prefill_attention_analysis import _worker_peak_rss_bytes; '
                 'print(json.dumps({"current_mm_peak":_worker_peak_rss_bytes(),'
                 '"lifetime_peak":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))')
        stage = ('import os,sys; data=bytearray(64<<20); '
                 'os.execv(sys.executable,[sys.executable,"-I","-S","-c",' + repr(child) + '])')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', stage],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        measured = json.loads(result.stdout)
        self.assertGreaterEqual(measured['lifetime_peak'], 64 << 20)
        self.assertLess(measured['current_mm_peak'], 64 << 20)
        self.assertGreater(measured['lifetime_peak'], measured['current_mm_peak'])

    def test_host_floor_is_exact_and_low_host_prevents_spawn(self):
        with patch('megartx.prefill_attention_analysis._host_available_bytes',
                   return_value=HOST_FREE_FLOOR_BYTES):
            self.assertEqual(_require_host_free(), HOST_FREE_FLOOR_BYTES)
        module = helper()
        owner = module.OwnedProcesses(module.read_process(__import__('os').getpid()))
        with tempfile.TemporaryDirectory() as directory:
            specification, _ = fixture(Path(directory))
            with patch('megartx.prefill_attention_analysis._host_available_bytes',
                       return_value=HOST_FREE_FLOOR_BYTES - 1024), \
                 patch('megartx.prefill_attention_analysis.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'host MemAvailable below'):
                    run_analysis(StreamingEvidence(directory), specification,
                        expected_request_sha256=REQUEST, expected_storage_binding_sha256=STORAGE,
                        deadline=time.monotonic() + 300, ownership=owner, source_root=ROOT)
                spawn.assert_not_called()

    def test_host_floor_drop_during_worker_cleans_owned_child(self):
        module = helper()
        owner = module.OwnedProcesses(module.read_process(__import__('os').getpid()))
        spawned = []
        real_popen = subprocess.Popen
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            spawned.append(child)
            return child
        with tempfile.TemporaryDirectory() as directory:
            specification, _ = fixture(Path(directory))
            samples = [HOST_FREE_FLOOR_BYTES, HOST_FREE_FLOOR_BYTES,
                       HOST_FREE_FLOOR_BYTES - 1024]
            with patch('megartx.prefill_attention_analysis._host_available_bytes', side_effect=samples), \
                 patch('megartx.prefill_attention_analysis.subprocess.Popen', side_effect=spawn):
                with self.assertRaisesRegex(ValueError, 'host MemAvailable below'):
                    run_analysis(StreamingEvidence(directory), specification,
                        expected_request_sha256=REQUEST, expected_storage_binding_sha256=STORAGE,
                        deadline=time.monotonic() + 300, ownership=owner, source_root=ROOT)
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())

    def test_remaining_shared_deadline_is_not_reset(self):
        self.assertEqual(_allowance(1500.9, 1400), (100, 1500))
        self.assertEqual(_allowance(2500, 1400), (300, 1700))
        for value in (1400, 1400.9, float('inf'), True):
            with self.assertRaises(ValueError):
                _allowance(value, 1400)

    def test_fake_ownership_object_is_not_an_imported_helper(self):
        with self.assertRaisesRegex(ValueError, 'executing prefill ownership'):
            _helper(object(), ROOT)

    def test_independent_manifest_equality_is_mandatory(self):
        fake = {'schema': 'megartx-prefill-attention-errors-v1',
            'status': 'independent_attention_errors_computed', 'input_manifest': {},
            'native_arithmetic_acceptance': None, 'numerical_qualified': False,
            'quality_qualified': False, 'performance_qualified': False,
            'native_execution_attested': False, 'sampled_repeatability_qualified': False,
            'external_native_storage_frontier_validation_required': True,
            'cases': [{}] * 60, 'aggregate': {'coordinates': 480}}
        _assert_result(fake, {})
        with self.assertRaisesRegex(ValueError, 'manifest'):
            _assert_result(fake, {'stale': {'sha256': 'a' * 64}})
        fake['native_arithmetic_acceptance'] = True
        with self.assertRaises(ValueError):
            _assert_result(fake, {})

    def test_worker_hard_as_limit_is_effective_not_linux_rlimit_rss(self):
        code = ('import resource; resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20)); '
                '\ntry: x=bytearray(600<<20)\nexcept MemoryError: print("bounded")\n')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'bounded')

    def test_current_storage_identity_rejected_before_child_spawn(self):
        module = helper()
        owner = module.OwnedProcesses(module.read_process(__import__('os').getpid()))
        with tempfile.TemporaryDirectory() as directory:
            specification, _ = fixture(Path(directory))
            with patch('megartx.prefill_attention_analysis.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'storage binding mismatch'):
                    run_analysis(StreamingEvidence(directory), specification,
                        expected_request_sha256=REQUEST, expected_storage_binding_sha256='e' * 64,
                        deadline=time.monotonic() + 300, ownership=owner, source_root=ROOT)
                spawn.assert_not_called()

    def test_primary_survives_secondary_cleanup_and_stream_close_failure(self):
        module = helper()
        owner = module.OwnedProcesses(module.read_process(__import__('os').getpid()))
        spawned = []
        real_popen = subprocess.Popen
        original_cleanup = module.OwnedProcesses.cleanup
        class CloseFailure:
            def __init__(self, stream):
                self.stream = stream
            @property
            def closed(self):
                return self.stream.closed
            def fileno(self):
                return self.stream.fileno()
            def close(self):
                self.stream.close()
                raise OSError('injected close failure')
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            child.stderr = CloseFailure(child.stderr)
            spawned.append(child)
            return child
        def broken_cleanup(instance, *args, **kwargs):
            original_cleanup(instance, *args, **kwargs)
            raise OSError('injected cleanup failure')
        with tempfile.TemporaryDirectory() as directory:
            specification, _ = fixture(Path(directory))
            samples = [HOST_FREE_FLOOR_BYTES, HOST_FREE_FLOOR_BYTES,
                       HOST_FREE_FLOOR_BYTES - 1024]
            with patch('megartx.prefill_attention_analysis._host_available_bytes', side_effect=samples), \
                 patch('megartx.prefill_attention_analysis.subprocess.Popen', side_effect=spawn), \
                 patch.object(module.OwnedProcesses, 'cleanup', broken_cleanup):
                with self.assertRaisesRegex(ValueError, 'host MemAvailable below') as caught:
                    run_analysis(StreamingEvidence(directory), specification,
                        expected_request_sha256=REQUEST, expected_storage_binding_sha256=STORAGE,
                        deadline=time.monotonic() + 300, ownership=owner, source_root=ROOT)
            notes = '\n'.join(caught.exception.__notes__)
            self.assertIn('injected cleanup failure', notes)
            self.assertIn('injected close failure', notes)
            self.assertIsNotNone(spawned[0].poll())

    def test_timeout_cleans_exact_owned_child_and_returns_no_metrics(self):
        module = helper()
        owner = module.OwnedProcesses(module.read_process(__import__('os').getpid()))
        spawned = []
        real_popen = subprocess.Popen
        def stalled(*args, **kwargs):
            command = [sys.executable, '-I', '-S', '-c',
                       'import sys,time; sys.stdin.buffer.read(); time.sleep(20)']
            child = real_popen(command, **kwargs)
            spawned.append(child)
            return child
        with tempfile.TemporaryDirectory() as directory:
            specification, _ = fixture(Path(directory))
            with patch('megartx.prefill_attention_analysis.subprocess.Popen', side_effect=stalled):
                with self.assertRaisesRegex(TimeoutError, 'wall deadline'):
                    run_analysis(StreamingEvidence(directory), specification,
                        expected_request_sha256=REQUEST, expected_storage_binding_sha256=STORAGE,
                        deadline=time.monotonic() + 2, ownership=owner, source_root=ROOT)
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())

    def test_owned_full_synthetic_oracle_under_hard_limit(self):
        module = helper()
        owner = module.OwnedProcesses(module.read_process(__import__('os').getpid()))
        with tempfile.TemporaryDirectory() as directory:
            specification, capture = fixture(Path(directory))
            result = run_analysis(StreamingEvidence(directory), specification,
                expected_request_sha256=REQUEST, expected_storage_binding_sha256=STORAGE,
                deadline=time.monotonic() + 300, ownership=owner, source_root=ROOT)
            self.assertEqual(result['result']['input_manifest'], capture['raw_manifest'])
            self.assertEqual(result['result']['aggregate']['coordinates'], 480)
            self.assertIsNone(result['result']['native_arithmetic_acceptance'])
            self.assertEqual(result['resources']['address_space_limit_bytes'], MAX_RSS_BYTES)
            self.assertLess(result['resources']['peak_process_rss_bytes'], MAX_RSS_BYTES)
            self.assertEqual(result['resources']['rss_measurement'], 'linux_proc_self_status_VmHWM_post_exec_mm')
            self.assertEqual(result['resources']['lifetime_rss_measurement'],
                             'linux_getrusage_RUSAGE_SELF_ru_maxrss_may_include_pre_exec')
            self.assertTrue(result['resources']['ownership']['cleanup_complete'])
            self.assertEqual(result['resources']['host_free_floor_bytes'], HOST_FREE_FLOOR_BYTES)
            self.assertGreaterEqual(result['resources']['minimum_sampled_host_available_bytes'], HOST_FREE_FLOOR_BYTES)
            self.assertGreater(result['resources']['host_available_samples'], 3)
            self.assertEqual(owner.remembered, {})


if __name__ == '__main__':
    unittest.main()

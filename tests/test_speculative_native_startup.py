"""Real CPU subprocess startup/cleanup regressions; no compiler or model imports."""
import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from megartx.speculative_native_evidence import PrivateEvidence, bounded_json_chunks, error_detail
from megartx.speculative_native_plan import LIMITS
from megartx.speculative_native_probe import ProbeError

ROOT = Path(__file__).resolve().parents[1]
# Both processes run the production child/supervisor logic. Only source admission,
# GPU telemetry and the runtime factory are CPU fixtures. Ownership is real Linux
# subreaper/PID-start-time/pidfd; the factory's child writes an actual 9 MiB file.
HARNESS = r'''
import errno, importlib.util, json, os
from pathlib import Path
import subprocess, sys
from types import SimpleNamespace as NS
from unittest.mock import patch
root, work, fault = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root/'src'), str(root/'scripts')]
spec=importlib.util.spec_from_file_location('startup_client',root/'scripts/speculative_native_receipt_client.py')
client=importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
from megartx import speculative_native_client as runtime
from megartx.speculative_native_probe import ProbeError
from megartx.speculative_native_v2_plan import PURPOSE
from m1_owned_processes import OwnedProcesses
paths={name:str(work/'runtime'/name) for name in ('cache','tmp','home','config')}
paths['root']=str(work/'runtime')
plan={'python':sys.executable,'runner_lane':'v2','purpose':PURPOSE,'plan_sha256':'1'*64,
      'client_mode':'sync','runtime_binding':{'paths':paths,'project_root':str(root),
      'entrypoint':{'root':str(work/'adapter')}},
      'source_sha256':{'src/megartx/speculative_native_evidence.py':'2'*64}}
original_read=client.read_json
client.read_json=lambda path,cap: plan if Path(path).name=='plan.json' else ({} if Path(path).name=='auth.json' else original_read(path,cap))
client.validate_plan=lambda value,project:value
client.validate_authorization=lambda *a:None
client.installed_preflight=lambda *a,**k:{}
client.checkpoint_preflight=lambda *a:{}
client.runtime_preflight=lambda *a,**k:None
client.validate_entrypoint_discovery=lambda *a,**k:None
client.gpu_processes=lambda:[]
client.headroom=lambda:{'gpu_free_bytes':2<<30,'host_free_bytes':8<<30}
client.receipt_admission=lambda *a,**k:{}
def create(*a):
    for path in paths.values(): Path(path).mkdir(parents=True,exist_ok=True,mode=0o700)
client.create_runtime=create
original_write=client.PrivateEvidence.write
def write(self,name,value,**kw):
    if ((fault=='child_failure_write' and name=='native-client-failure.private.json') or
        (fault in ('cleanup_write','cleanup_and_fallback_write','cleanup_without_primary') and name=='native-owned-cleanup.private.json') or
        (fault=='cleanup_and_fallback_write' and name=='native-owned-cleanup-failure.private.json')):
        error=OSError(errno.ENOSPC,'fixture disk full','/private/fixture/location')
        error.megartx_operation='fixture_evidence_write'
        raise error
    return original_write(self,name,value,**kw)
client.PrivateEvidence.write=write
args=NS(plan=work/'plan.json',authorization=work/'auth.json',private_directory=work/'evidence',
        checkpoint_manifest=work/'checkpoint.json',control_fd=int(sys.argv[5]) if len(sys.argv)>5 else None)
if len(sys.argv)>4 and sys.argv[4]=='child':
    def startup(*a):
        program="from pathlib import Path; import json,os,resource; p=Path(os.environ['TMPDIR'])/'compiler-artifact.o'; p.write_bytes(b'x'*(9<<20)); print(json.dumps({'artifact_bytes':p.stat().st_size,'file_limit':resource.getrlimit(resource.RLIMIT_FSIZE)}))"
        subprocess.run([sys.executable,'-S','-c',program],check=True)
        assert not any(name in sys.modules for name in ('torch','vllm','flashinfer','triton'))
        if fault=='cleanup_without_primary': return NS()
        raise RuntimeError('fixture startup failure')
    if fault=='cleanup_without_primary':
        runtime.ReceiptSession=lambda *a,**kw:NS(events=[],release_attempted=False,shutdown_attempted=False,run_sync=lambda admission:{})
    runtime.make_actual_client=startup
    try:
        client.child_entry(args)
    except BaseException as error:
        # Python 3.10 does not render exception notes in its traceback. Verify
        # the retained diagnostic independently of the interpreter renderer.
        print(json.dumps({'retained_notes':getattr(error,'__notes__',[])}),flush=True)
        raise
else:
    # The production pin is independently tested elsewhere. This process is a
    # portable CPU failure-path fixture, never an admissible GPU authorization.
    client.sys.version_info=(3,12,3)
    real_popen=subprocess.Popen
    def spawn(argv,**kw):
        control=argv[argv.index('--control-fd')+1]
        return real_popen([sys.executable,str(Path(__file__).resolve()),str(root),str(work),fault,'child',control],**kw)
    client.subprocess.Popen=spawn
    original_cleanup=OwnedProcesses.cleanup
    def cleanup(self,*a,**kw):
        row=original_cleanup(self,*a,**kw)
        if fault=='cleanup_oversize': row['fixture_oversize']=['x'*4096]*65
        if fault=='cleanup_exception':
            try: raise OSError(errno.EIO,'fixture cleanup source failed','/private/fixture/source')
            except OSError as error: raise ProbeError('fixture cleanup wrapper failed') from error
        return row
    OwnedProcesses.cleanup=cleanup
    try:
        client.supervise(args)
        raise AssertionError('startup fixture must fail')
    except BaseException as error:
        assert not any(name in sys.modules for name in ('torch','vllm','flashinfer','triton'))
        print(json.dumps({'error_type':type(error).__name__,'message':str(error),
                          'notes':getattr(error,'__notes__',[])}))
'''


class StartupSubprocessControls(unittest.TestCase):
    def run_fixture(self, fault='none'):
        temporary = tempfile.TemporaryDirectory(prefix='ds-startup-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        harness = root / 'fixture.py'
        harness.write_text(HARNESS)
        result = subprocess.run([sys.executable, '-S', str(harness), str(ROOT), str(root), fault],
            capture_output=True, text=True, timeout=40,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        return root, json.loads(result.stdout)

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux real ownership required')
    def test_compiler_artifact_exceeds_receipt_cap_then_failed_startup_has_cleanup_record(self):
        root, result = self.run_fixture()
        self.assertEqual((root/'runtime/tmp/compiler-artifact.o').stat().st_size, 9 << 20)
        self.assertEqual(result['message'], 'Owned child failed; private logs retained')
        cleanup = json.loads((root/'evidence/native-owned-cleanup.private.json').read_text())
        self.assertTrue(cleanup['cleanup_complete'])
        self.assertEqual(cleanup['owned_identities_remaining'], [])
        self.assertGreaterEqual(len(cleanup['remembered_identities']), 1)
        self.assertEqual(cleanup['compiler_rss_limit_bytes'], 2 << 30)
        self.assertEqual(cleanup['shared_compiler_seconds_limit'], 300)
        failed = json.loads((root/'evidence/native-client-failure.private.json').read_text())
        self.assertEqual(failed['failure']['stage'], 'owned_client_startup')
        self.assertEqual(failed['failure']['message'], 'fixture startup failure')
        self.assertEqual(LIMITS['receipt_host_metadata_bytes'], 8 << 20)
        self.assertEqual(LIMITS['private_log_bytes'], 8 << 20)
        # The actual compiler-like temporary file is outside private evidence.
        self.assertFalse((root/'evidence/compiler-artifact.o').exists())
        with PrivateEvidence(root/'evidence') as store:
            with self.assertRaisesRegex(ProbeError, 'exact byte bound'):
                store.write('oversized-receipt.json', ['x'*4096]*2048)
        self.assertFalse((root/'evidence/oversized-receipt.json').exists())

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux real ownership required')
    def test_cleanup_write_failure_preserves_primary_and_reports_uncertainty(self):
        for fault in ('cleanup_write', 'cleanup_oversize', 'cleanup_exception', 'cleanup_and_fallback_write'):
            with self.subTest(fault=fault):
                root, result = self.run_fixture(fault)
                self.assertEqual(result['message'], 'Owned child failed; private logs retained')
                notes = '\n'.join(result['notes'])
                self.assertIn('cleanup', notes.lower())
                if fault == 'cleanup_exception':
                    full = json.loads((root/'evidence/native-owned-cleanup.private.json').read_text())
                    self.assertFalse(full['cleanup_complete'])
                    self.assertEqual(full['failure_detail']['causes'][0]['errno'], errno.EIO)
                else:
                    self.assertIn('cleanup uncertainty retained', notes)
                    self.assertFalse((root/'evidence/native-owned-cleanup.private.json').exists())
                    if fault == 'cleanup_and_fallback_write':
                        self.assertIn('Owned cleanup diagnostic evidence failed', notes)
                        self.assertIn('fixture disk full', notes)
                    else:
                        fallback = json.loads((root/'evidence/native-owned-cleanup-failure.private.json').read_text())
                        self.assertFalse(fallback['cleanup_complete'])
                        self.assertFalse(fallback['internal_identity_graph_verified'])
                        self.assertTrue(fallback['process_cleanup_reported_complete'])
                        if fault == 'cleanup_write':
                            self.assertEqual(fallback['failure']['errno'], errno.ENOSPC)
                            self.assertEqual(fallback['failure']['operation'], 'fixture_evidence_write')
                        else:
                            self.assertIn('exact byte bound', fallback['failure']['message'])
                            self.assertEqual(fallback['failure']['operation'], 'evidence_serialization_preflight')
                self.assertNotIn('/private/fixture', notes)
                self.assertFalse((root/'evidence/native-receipt-comparison.scalars.json').exists())

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux real ownership required')
    def test_cleanup_evidence_failure_without_prior_primary_still_fails_run(self):
        root, result = self.run_fixture('cleanup_without_primary')
        self.assertEqual(result['error_type'], 'OSError')
        self.assertIn('fixture disk full', result['message'])
        self.assertIn('cleanup uncertainty retained', '\n'.join(result['notes']))
        self.assertFalse((root/'evidence/native-receipt-comparison.scalars.json').exists())
        fallback = json.loads((root/'evidence/native-owned-cleanup-failure.private.json').read_text())
        self.assertFalse(fallback['cleanup_complete'])
        self.assertFalse(fallback['internal_identity_graph_verified'])

    def test_actual_log_limit_remains_eight_mib_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            program = '''
import importlib.util, io, json, pathlib, sys
spec=importlib.util.spec_from_file_location('bounded_log',sys.argv[1])
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
failures=[]
client.bounded_log(io.BytesIO(b'x'*((8<<20)+1)),pathlib.Path(sys.argv[2]),failures.append)
print(json.dumps(failures))
'''
            path = Path(directory)/'log'
            result = subprocess.run([sys.executable,'-S','-c',program,
                str(ROOT/'scripts/speculative_native_receipt_client.py'),str(path)],
                capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),['private_log_limit'])
            self.assertEqual(path.stat().st_size,8<<20)

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux real ownership required')
    def test_startup_error_survives_secondary_child_failure_record_write(self):
        root, result = self.run_fixture('child_failure_write')
        log = (root/'evidence/runtime.private.log').read_text()
        self.assertIn('RuntimeError: fixture startup failure', log)
        self.assertIn('Owned client failure evidence failed', log)
        self.assertIn('fixture_evidence_write', log)
        self.assertEqual(result['message'], 'Owned child failed; private logs retained')
        self.assertTrue(json.loads((root/'evidence/native-owned-cleanup.private.json').read_text())['cleanup_complete'])


class EvidenceFailureControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.directory.chmod(0o700)

    def test_exact_near_cap_and_one_byte_over_before_file_acquisition(self):
        row = ['x'*4096]*63 + ['x'*3896]
        exact = len(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
        self.assertLess(exact, LIMITS['plan_bytes'])
        with PrivateEvidence(self.directory) as store:
            result = store.write('near-cap.json', row, cap=exact)
            self.assertEqual(result['bytes'], exact)
            with self.assertRaisesRegex(ProbeError, 'exact byte bound'):
                store.write('too-big.json', row, cap=exact-1)
        self.assertEqual([p.name for p in self.directory.iterdir()], ['near-cap.json'])

    def test_repeated_compiler_graph_exact_bytes_and_long_argv_stay_bounded(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        from m1_owned_processes import OwnedProcesses, Process
        owner = OwnedProcesses(Process(10, 10, 1))
        root = Process(20, 20, 10)
        owner.register(root)
        processes = {20:root}
        for n in range(8):
            process = Process(30+n, 30+n, 20, executable='gcc', compiler_argv=('gcc','x'*1000))
            processes[process.pid] = process
        owner.observe(processes, 100)
        report = {**owner.report(), 'cleanup_complete':True}
        raw = json.dumps(report, sort_keys=True, separators=(',', ':')).encode()
        self.assertGreater(sum(len(p['compiler_argv'][1]) for p in report['sampled_compiler_invocations'])*12*4,
                           LIMITS['plan_bytes'])
        with PrivateEvidence(self.directory) as store:
            result = store.write('complete-graph.json', report, cap=LIMITS['plan_bytes'])
            self.assertEqual(result['bytes'], len(raw))
            self.assertEqual(json.loads((self.directory/'complete-graph.json').read_text()), json.loads(raw))
            report['sampled_compiler_invocations'][0]['compiler_argv'] = ('gcc', 'x'*4097)
            with self.assertRaisesRegex(ProbeError, 'string exceeds'):
                store.write('long-argv.json', report, cap=LIMITS['plan_bytes'])
        self.assertFalse((self.directory/'long-argv.json').exists())

    def test_byte_domain_cannot_be_increased_by_caller(self):
        with self.assertRaisesRegex(ProbeError, 'fixed serialized limit'):
            list(bounded_json_chunks({}, cap=(8 << 20)+1))

    def test_publish_primary_survives_temporary_cleanup_error(self):
        primary = OSError(errno.EIO, 'fixture publish failed', '/private/primary')
        secondary = OSError(errno.EPERM, 'fixture unlink failed', '/private/secondary')
        with PrivateEvidence(self.directory) as store, \
                patch('megartx.speculative_native_evidence.os.link', side_effect=primary), \
                patch('megartx.speculative_native_evidence.os.unlink', side_effect=secondary):
            with self.assertRaises(OSError) as raised:
                store.write('fail.json', {})
        self.assertIs(raised.exception, primary)
        self.assertEqual(primary.megartx_operation, 'evidence_publish_no_replace')
        self.assertIn('evidence_temporary_unlink', primary.__notes__[0])
        self.assertIn('fixture unlink failed', primary.__notes__[0])
        self.assertNotIn('/private/secondary', primary.__notes__[0])

    def test_directory_close_error_cannot_replace_body_error(self):
        primary = ProbeError('fixture body failure')
        store = PrivateEvidence(self.directory)
        fd = store.fd
        try:
            with patch('megartx.speculative_native_evidence.os.close', side_effect=OSError(errno.EIO,'fixture close')):
                with self.assertRaises(ProbeError) as raised:
                    with store:
                        raise primary
            self.assertIs(raised.exception, primary)
            self.assertIn('evidence_directory_close', primary.__notes__[0])
        finally:
            os.close(fd)

    def test_error_detail_retains_bounded_cause_and_redacts_paths(self):
        try:
            try:
                raise OSError(errno.ENOSPC, 'disk full', '/private/secret')
            except OSError as cause:
                raise ProbeError('wrapper failed') from cause
        except ProbeError as error:
            detail = error_detail(error, 'cleanup', 'fixture_write')
        self.assertEqual(detail['causes'][0]['errno'], errno.ENOSPC)
        self.assertIn('disk full', detail['causes'][0]['message'])
        self.assertNotIn('/private/secret', json.dumps(detail))


if __name__ == '__main__':
    unittest.main()

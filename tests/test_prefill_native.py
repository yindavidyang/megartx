"""CPU-only negative and accounting tests. Fakes never establish native fit."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from megartx.prefill_diagnostic_plan import (Evidence, freeze_plan, load_plan, require_clearance,
                                           publish_fit, BOUNDS, BASE, server_args, digest)
from megartx.loaded_engine_access import owned_page_ranges, reject_page_alias
from megartx.prefill_native import RequestLedger
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prefill_diagnostic_client import StreamLedger, payload, validate_observation
from megartx.prefill_diagnostic_plan import native_request_identity
from prefill_identity_fixture import run_case
import asyncio

ROOT = Path(__file__).resolve().parents[1]
TOKENS = list(range(2048))


class NativeDiagnosticTests(unittest.TestCase):
    def plan(self):
        from megartx.controlled_kv_capture import CONFIG_SHA256
        return freeze_plan(TOKENS, BASE, ROOT, {'config_sha256': CONFIG_SHA256,
            'index_sha256': 'a'*64, 'shard_stats': {'synthetic.safetensors': {'size':1,'mtime_ns':1}}})

    def test_imports_never_load_runtime_or_device(self):
        code = "import megartx.prefill_native, megartx.loaded_engine_access; import sys; assert not any(n=='torch' or n=='vllm' or n.startswith(('torch.','vllm.')) for n in sys.modules)"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpoint_index_above_four_mib_is_bounded_cpu_metadata(self):
        from megartx.prefill_diagnostic_plan import checkpoint_identity, CHECKPOINT_INDEX_MAX_BYTES
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=b'{}';(root/'config.json').write_bytes(config)
            (root/'synthetic.safetensors').write_bytes(b'x')
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'synthetic':'synthetic.safetensors'},
                                         'cpu_fixture_padding':'x'*(4<<20)}))
            self.assertGreater(index.stat().st_size,4<<20)
            with patch('megartx.controlled_kv_capture.CONFIG_SHA256',hashlib.sha256(config).hexdigest()):
                identity=checkpoint_identity(root)
                self.assertEqual(identity['index_sha256'],hashlib.sha256(index.read_bytes()).hexdigest())
                self.assertEqual(set(identity['shard_stats']),{'synthetic.safetensors'})
                with index.open('wb') as stream:stream.truncate(CHECKPOINT_INDEX_MAX_BYTES+1)
                with patch('megartx.prefill_diagnostic_plan.json.loads',side_effect=AssertionError('oversized index parsed')):
                    with self.assertRaisesRegex(ValueError,'bounded identity'):
                        checkpoint_identity(root)

    def test_frozen_exact_plan_bounds_drift_and_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'plan.json'
            plan = self.plan(); p.write_text(json.dumps(plan))
            self.assertEqual(load_plan(p, ROOT), plan)
            for key, value in [('bounds', {**BOUNDS, 'max_requests': 2}), ('tokens', TOKENS[:-1]),
                               ('source_hashes', {**plan['source_hashes'], 'src/megartx/m1_live.py': '0'*64})]:
                bad = copy.deepcopy(plan); bad[key] = value
                bad['plan_sha256'] = digest({k:v for k,v in bad.items() if k != 'plan_sha256'})
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load_plan(p, ROOT)
            clearance = {'schema': 'megartx-prefill-native-clearance-v1', 'plan_sha256': plan['plan_sha256'],
                         'source_head': BASE, 'cpu_review_passed': True, 'parent_gpu_slot_clearance': True}
            p.write_text(json.dumps(clearance)); require_clearance(p, plan)
            for key in ('cpu_review_passed', 'parent_gpu_slot_clearance'):
                bad = {**clearance, key: 1}; p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): require_clearance(p, plan)

    def test_pre_overflow_append_and_publication_leave_file_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            writer = Evidence(d, 80)
            writer.write('frames.jsonl', {'v': 'a'*30}, append=True)
            old = (Path(d)/'frames.jsonl').read_bytes()
            with self.assertRaisesRegex(ValueError, 'before write'):
                writer.write('frames.jsonl', {'v': 'a'*60}, append=True)
            self.assertEqual((Path(d)/'frames.jsonl').read_bytes(), old)
            with self.assertRaises(ValueError): writer.write('complete.json', {'v':'a'*60})
            self.assertFalse((Path(d)/'complete.json').exists())

    def test_group_overlay_disjoint_owned_pages_but_overlap_rejected(self):
        # Historical HND geometry only, not a current fit assumption.
        local = (1, 1000, 8*131072, 0, (8,8,16,512), (65536,512,4096,1), 'cuda:0')
        global_ = (2, 1000, 8*131072, 0, (8,2,32,1024), (65536,1024,2048,1), 'cuda:0')
        reject_page_alias({0: owned_page_ranges(local,[1,2]), 5: owned_page_ranges(global_,[3,4])})
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            reject_page_alias({0: owned_page_ranges(local,[1,2]),5:owned_page_ranges(global_,[2,3])})
        for bad in (-1,8):
            with self.assertRaises(ValueError): owned_page_ranges(local,[bad])
        malformed = list(local); malformed[5] = (65536,500,4096,1)
        with self.assertRaises(ValueError): owned_page_ranges(tuple(malformed),[1])

    def test_actual_prompt_sample_plus_255_decode_inputs_accounting(self):
        ledger = RequestLedger(TOKENS)
        identities = {i: ('identity',i) for i in range(30)}
        for start in range(0,2303):
            if start < 2048 and start % 256: continue
            count = 256 if start < 2048 else 1
            tokens = TOKENS[start:start+count] if start < 2048 else ledger.outputs[-1:]
            slots = {i:list(range(start+32,start+count+32)) for i in range(30)}
            ledger.begin(tokens,list(range(start,start+count)),slots,identities,'request')
            ledger.complete(); ledger.sampled(17,ledger.end<2048)
        self.assertTrue(ledger.complete_request)
        self.assertEqual((ledger.frames,len(ledger.outputs),ledger.end),(263,256,2303))
        with self.assertRaises(ValueError): ledger.begin([17],[2303],{i:[2335] for i in range(30)},identities,'request')
        ledger.abort(); self.assertFalse(ledger.complete_request)

    def test_slots_owner_anchor_and_sample_boundaries_fail_closed(self):
        identities = {i: ('identity',i) for i in range(30)}
        for mutation in ('positions','tokens','slots','request','discard','duplicate'):
            ledger = RequestLedger(TOKENS)
            slots = {i:list(range(32,288)) for i in range(30)}
            if mutation in ('positions','tokens','slots'):
                pos = list(range(256)); tokens = TOKENS[:256]
                if mutation == 'positions': pos[0]=1
                if mutation == 'tokens': tokens=[9]*256
                if mutation == 'slots': slots[0][1]=slots[0][0]
                with self.assertRaises(ValueError): ledger.begin(tokens,pos,slots,identities,'r')
                continue
            ledger.begin(TOKENS[:256],list(range(256)),slots,identities,'r'); ledger.complete()
            if mutation == 'discard':
                with self.assertRaises(ValueError): ledger.sampled(17,False)
            else:
                ledger.sampled(17,True)
                if mutation == 'duplicate':
                    with self.assertRaises(ValueError): ledger.sampled(17,True)
                else:
                    with self.assertRaises(ValueError): ledger.begin(TOKENS[256:512],list(range(256,512)),{i:list(range(288,544)) for i in range(30)},identities,'other')

    def test_sse_tokens_usage_native_chain_and_no_qualification(self):
        plan = self.plan(); stream = StreamLedger(plan)
        for i in range(256):
            stream.consume('data: '+json.dumps({'id':stream.response_id,'choices':[{'index':0,'token_ids':[17], 'finish_reason':'length' if i==255 else None}]}))
        stream.consume('data: '+json.dumps({'id':stream.response_id,'choices':[], 'usage':{'prompt_tokens':2048,'completion_tokens':256,'total_tokens':2304}}))
        stream.consume('data: [DONE]')
        engine_id = asyncio.run(run_case(base_id=payload(plan)['request_id']))['engine_ids'][0]
        observer = {'schema':'megartx-prefill-native-observation-v1','status':'request_observed',
                    'plan_sha256':plan['plan_sha256'],'engine_request_id':engine_id,
                    'request_identity':native_request_identity(plan, engine_id),
                    'output_ids_sha256':digest([17]*256),'prompt_frames':8,'decode_input_rows':255,
                    'emitted_outputs':256,'committed_length':2303,'numerical_qualified':False,'performance_qualified':False}
        validate_observation(plan,stream,observer)
        for k,v in [('output_ids_sha256','0'*64),('engine_request_id','wrong'),('numerical_qualified',True)]:
            with self.assertRaises(ValueError): validate_observation(plan,stream,{**observer,k:v})
        self.assertEqual(payload(plan)['max_tokens'],256)
        self.assertIn('--no-async-scheduling',server_args())
        with self.assertRaises(ValueError): StreamLedger(plan).consume('data: [DONE]')
        with self.assertRaises(ValueError): stream.consume('data: [DONE]')

    def test_launcher_rejects_missing_plan_before_environment_or_resources(self):
        result = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_scale_validation.py'),'--label','cpu','--mode','native','--client','prefill-native','--trials','1'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('exact default-off',result.stderr)
        self.assertNotIn('MEGARTX_BASE',result.stderr)

    def test_admitted_cli_http_import_errors_acquire_no_resources(self):
        # Synthetic CPU plan/clearance only. Execution stops at HTTP import,
        # before any checkpoint, environment, process, device or owned output.
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            plan = self.plan(); plan['source_head'] = head
            plan['plan_sha256'] = digest({k:v for k,v in plan.items() if k != 'plan_sha256'})
            (directory/'plan.json').write_text(json.dumps(plan))
            (directory/'clearance.json').write_text(json.dumps({'schema':'megartx-prefill-native-clearance-v1',
                'plan_sha256':plan['plan_sha256'],'source_head':head,
                'cpu_review_passed':True,'parent_gpu_slot_clearance':True}))
            command = [sys.executable,'-S',str(ROOT/'scripts/run_scale_validation.py'),
                       '--label','cpu','--mode','native','--client','prefill-native','--trials','1',
                       '--prefill-native-plan',str(directory/'plan.json'),
                       '--prefill-native-clearance',str(directory/'clearance.json')]
            env = {**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(directory),
                   'MEGARTX_BASE':str(directory/'untouched-base'),
                   'MEGARTX_WORK':str(directory/'untouched-work')}
            for stub, error in ((None,"ModuleNotFoundError: No module named 'requests'"),
                                ("raise RuntimeError('http import primary')\n",'RuntimeError: http import primary')):
                if stub is not None:(directory/'requests.py').write_text(stub)
                result = subprocess.run(command,cwd=directory,env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(error,result.stderr)
                self.assertNotIn('exact default-off',result.stderr)
                self.assertFalse((directory/'untouched-base').exists())
                self.assertFalse((directory/'untouched-work').exists())
            result = subprocess.run(command,cwd=directory,env={**env,'VLLM_USE_V2_MODEL_RUNNER':'0'},
                                    capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('inherited override',result.stderr)
            self.assertNotIn('http import primary',result.stderr)
            self.assertFalse((directory/'untouched-work').exists())


if __name__ == '__main__': unittest.main()

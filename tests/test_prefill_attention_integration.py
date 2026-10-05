"""CPU-only default policy, complete storage gates and exclusive new selection."""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from megartx import prefill_attention_native_plan as native
from megartx import prefill_attention_publication as publication
from megartx import prefill_storage_plan as storage
from megartx import prefill_diagnostic_plan as legacy
from megartx.prefill_storage_native import StorageProvider
from megartx.controlled_kv_capture import CONFIG_SHA256
from test_prefill_storage_plan import receipt_fixture

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/prefill-attention-base'
MANIFEST=json.loads((FIXTURE/'manifest.json').read_text())


def prior(path):
    item=MANIFEST['files'][path]; data=(FIXTURE/item['fixture']).read_bytes()
    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
        raise ValueError('Exact main8060 source fixture changed')
    return data.decode()


def dump(node):
    return ast.dump(node,include_attributes=False)


class WithoutAttention(ast.NodeTransformer):
    """Remove only the explicit new-purpose selection, preserving all old AST."""
    def known(self,node):
        if isinstance(node,ast.Name) and node.id=='prefill_attention':return False
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='attention_capture':
            return False
        if isinstance(node,ast.BoolOp) and isinstance(node.op,ast.And):
            values=[self.known(x) for x in node.values]
            if False in values:return False
        return None
    def visit_Assign(self,node):
        names={x.id for x in node.targets if isinstance(x,ast.Name)}
        if names & {'prefill_attention','attention_capture'}:return None
        if names=={'prefill_storage'}:
            expected=ast.parse('args.client in {"prefill-storage", "prefill-attention"}',mode='eval').body
            if dump(node.value)!=dump(expected):raise AssertionError('Unknown compact lifecycle selector')
            node.value=ast.parse('args.client == "prefill-storage"',mode='eval').body
            return node
        return self.generic_visit(node)
    def visit_If(self,node):
        known=self.known(node.test)
        if known is not None:
            return [self.visit(x) for x in (node.body if known else node.orelse)]
        return self.generic_visit(node)
    def visit_IfExp(self,node):
        known=self.known(node.test)
        if known is not None:return self.visit(node.body if known else node.orelse)
        return self.generic_visit(node)
    def visit_Tuple(self,node):
        node=self.generic_visit(node)
        node.elts=[x for x in node.elts if not (isinstance(x,ast.Constant) and x.value in
                   ('prefill-attention','MEGARTX_PREFILL_ATTENTION_CAPTURE'))]
        return node
    def visit_Set(self,node):
        node=self.generic_visit(node)
        node.elts=[x for x in node.elts if not (isinstance(x,ast.Constant) and x.value=='prefill-attention')]
        return node


def fixture_plan():
    # Explicit CPU synthetic plan: no catalog admission, native run or clearance.
    value=legacy.freeze_plan(list(range(2048)),'3'*40,ROOT,{
        'config_sha256':CONFIG_SHA256,'index_sha256':'a'*64,
        'shard_stats':{'synthetic.safetensors':{'size':1,'mtime_ns':1}}})
    value['source_hashes'][storage.CHECKER_SOURCE]=storage.file_sha(ROOT/storage.CHECKER_SOURCE)
    value['plan_sha256']=native.digest({k:v for k,v in value.items() if k!='plan_sha256'})
    return value


def attention_control(old):
    value=copy.deepcopy(old)
    value.update(schema='megartx-prefill-attention-storage-control-v1',
        sample_combined_kv_rows=0,raw_head_payload_rows=0,attention_raw_files=8,
        attention_raw_bytes=5395952,attention_callbacks_restored=True,
        observed_head_hashes={str(p):hashlib.sha256(bytes(524288)).hexdigest() for p in (2047,2048)})
    domains=value['transfer']['domains']
    domains.update(attention_cache_entry=73388032,attention_q_output=102400,attention_metadata=65536)
    value['transfer']['transferred_bytes']=sum(domains.values())
    return value


class IntegrationTests(unittest.TestCase):
    def test_entire_plugin_and_launcher_old_selection_ast_unchanged(self):
        for path in ('src/megartx/vllm_scale_plugin.py','scripts/run_scale_validation.py'):
            current=WithoutAttention().visit(ast.parse((ROOT/path).read_text()))
            self.assertEqual(dump(current),dump(ast.parse(prior(path))),path)

    def test_selector_is_default_off_exclusive_and_rejected_before_native_import(self):
        from megartx.vllm_scale_plugin import install
        with patch.dict(os.environ,{},clear=True):self.assertIsNone(install())
        complete={'MEGARTX_SCALE_MODE':'native','MEGARTX_PREFILL_NATIVE_PLAN':'plan',
                  'MEGARTX_PREFILL_NATIVE_DIR':'dir','MEGARTX_PREFILL_NATIVE_DEADLINE':'1',
                  'MEGARTX_PREFILL_NATIVE_SOURCE_ROOT':'root'}
        for extra in ({native.SELECTOR:'0'}, {native.SELECTOR:'1','MEGARTX_PREFILL_STORAGE_CONTROL':'1'},
                      {native.SELECTOR:'1','MEGARTX_SCALE_MODE':'reference'}):
            with patch.dict(os.environ,{**complete,**extra},clear=True),self.assertRaisesRegex(RuntimeError,'Attention capture'):
                install()
        for missing in complete:
            env={**complete,native.SELECTOR:'1'};env.pop(missing)
            with patch.dict(os.environ,env,clear=True),self.assertRaisesRegex(RuntimeError,'Attention capture'):
                install()

    def test_unknown_installed_closure_cannot_be_cleared(self):
        plan={'installed_sources':{k:None for k in native.INSTALLED_CONTRACT}}
        with self.assertRaisesRegex(ValueError,'unknown'):
            native.require_clearance('/no/clearance/read/is/allowed',plan)
        self.assertEqual(set(native.INSTALLED_CONTRACT),{'vllm.utils.flashinfer','flashinfer.api_logging','flashinfer.trace.template','flashinfer.utils'})

    def test_exact_new_clearance_is_not_old_consumed_storage_clearance(self):
        plan={'installed_sources':dict(native.INSTALLED_CONTRACT),'plan_sha256':'a'*64,'source_head':'b'*40}
        good={'schema':'megartx-prefill-attention-native-clearance-v1','plan_sha256':'a'*64,
              'source_head':'b'*40,'cpu_review_passed':True,'parent_gpu_slot_clearance':True,
              'installed_source_vector_sha256':native.digest(plan['installed_sources']),
              'max_contexts':1,'max_requests':1,'max_retries':0}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'clearance.json';path.write_text(json.dumps(good))
            self.assertEqual(native.require_clearance(path,plan),good)
            for key in good:
                changed=dict(good);changed[key]=None;path.write_text(json.dumps(changed))
                with self.subTest(field=key),self.assertRaises(ValueError):native.require_clearance(path,plan)

    def test_all_original_storage_and_frontier_domains_are_mandatory(self):
        plan=fixture_plan()
        with tempfile.TemporaryDirectory() as d:
            evidence,old,_=receipt_fixture(Path(d),plan)
            value=attention_control(old)
            self.assertEqual(publication.validate_control(plan,value),value)
            publication.validate_transcripts(plan,Path(d),value)
            for key in (*storage.CONTROL_COUNTS,'attention_raw_bytes','attention_callbacks_restored','raw_head_payload_rows'):
                changed=copy.deepcopy(value);changed[key]=None
                with self.subTest(field=key),self.assertRaises(ValueError):publication.validate_control(plan,changed)
            for key in value['transfer']['domains']:
                changed=copy.deepcopy(value);changed['transfer']['domains'][key]-=1
                changed['transfer']['transferred_bytes']-=1
                # Manager copies allow the exact source-derived interval; choose zero.
                if key=='manager_table_metadata':changed['transfer']['domains'][key]=0
                with self.subTest(domain=key),self.assertRaises(ValueError):publication.validate_control(plan,changed)
            changed=copy.deepcopy(value);changed['observed_head_hashes']['2048']='a'*64
            with self.assertRaisesRegex(ValueError,'head'):publication.validate_transcripts(plan,Path(d),changed)
            changed=copy.deepcopy(value);changed['manager']['manager_to_kernel_ratios']=[2,2]
            changed['manager']['manager_block_tokens']=[32,32]
            with self.assertRaises(ValueError):publication.validate_control(plan,changed)

    def test_original_storage_engine_only_has_explicit_policy_seams(self):
        path='src/megartx/prefill_storage_native.py'
        original=prior(path); current=(ROOT/path).read_text()
        old_cls=next(x for x in ast.parse(original).body if isinstance(x,ast.ClassDef))
        new_cls=next(x for x in ast.parse(current).body if isinstance(x,ast.ClassDef))
        old_methods={x.name:ast.get_source_segment(original,x) for x in old_cls.body if isinstance(x,ast.FunctionDef)}
        new_methods={x.name:ast.get_source_segment(current,x) for x in new_cls.body if isinstance(x,ast.FunctionDef)}
        seams={'on_processed_row','retain_processed_row','retain_head_row','sample_retention_receipt','write_control_receipt'}
        self.assertEqual(set(new_methods)-set(old_methods),seams)
        for name,body in old_methods.items():
            actual=new_methods[name]
            if name=='__init__':
                actual=actual.replace("'purpose': self.storage_binding_purpose,", "'purpose': 'first-nine-frame-storage-frontier-control',")
            elif name=='before_writer':
                actual=actual.replace("                    self.on_processed_row(layer, position, slots[layer][row], k, v)\n                    self.retain_processed_row(layer, position, k, v)",
                    "                    if position in (15, 16, 1023, 1024, 2047, 2048):\n                        self.evidence.raw(f'kv-layer-{layer:02d}-position-{position:04d}.bf16', k+v)\n                        self.sample_count += 1")
            elif name=='head':
                actual=actual.replace('self.retain_head_row(position, raw)', "self.evidence.raw(f'head-position-{position}.bf16', raw)")
            elif name=='sampled':
                actual=actual.replace("or self.storage_frames != 9\n", "or self.storage_frames != 9\n                    or self.sample_count != 180 or self.raw_heads != 2\n")
                actual=actual.replace('            retention = self.sample_retention_receipt()\n','')
                actual=actual.replace("self.write_control_receipt({", "self.evidence.write('control.json', {")
                actual=actual.replace('                **retention,', "                'sample_combined_kv_rows': self.sample_count, 'raw_head_rows': self.raw_heads,")
                actual=actual.replace("                'frame_records_sha256':", "                'raw_samples_sha256': digest(raw_manifest(self.directory)),\n                'frame_records_sha256':")
            self.assertEqual(actual,body,name)

    def test_default_processed_callback_does_not_touch_bytes_or_evidence(self):
        p=StorageProvider.__new__(StorageProvider)
        self.assertIsNone(p.on_processed_row(0,0,0,b'k',b'v'))
        class Evidence:
            def __init__(self):self.raws=[]
            def raw(self,*args):self.raws.append(args)
        p.evidence=Evidence();p.sample_count=0
        p.retain_processed_row(0,0,b'k',b'v')
        p.retain_processed_row(0,15,b'k',b'v')
        p.retain_head_row(2047,b'head')
        self.assertEqual(p.evidence.raws,[('kv-layer-00-position-0015.bf16',b'kv'),('head-position-2047.bf16',b'head')])
        self.assertEqual(p.sample_count,1)

    def test_actual_launcher_uses_prefill_helper_compact_evidence_and_analysis_after_cleanup(self):
        from test_shared_launcher_composition import LauncherHarness
        from megartx.prefill_attention_evidence import StreamingEvidence
        harness=LauncherHarness(self,'storage')
        harness.args.client='prefill-attention'
        harness.env['prefill_attention']=True
        harness.evidence=StreamingEvidence(harness.output/'prefill-attention')
        harness.env['prefill_evidence']=harness.evidence
        calls=[]
        def published(evidence,plan,report,*,deadline,ownership):
            self.assertIs(evidence,harness.evidence)
            self.assertIs(ownership,harness.env['ownership'])
            self.assertEqual(deadline,harness.env['prefill_deadline'])
            self.assertTrue(report['cleanup_complete'])
            self.assertEqual(harness.server.returncode,0)
            self.assertTrue((evidence.directory/'cleanup.json').is_file())
            calls.append('analysis_after_owned_cleanup')
        with patch('megartx.prefill_attention_native_plan.load_plan',return_value=harness.env['prefill_plan']), \
             patch('megartx.prefill_attention_native_plan.validate_binding',return_value={'runner_policy':'cpu'}), \
             patch('megartx.prefill_attention_publication.publish_attention',side_effect=published):
            harness.execute()
        self.assertEqual(calls,['analysis_after_owned_cleanup'])
        self.assertIn('prefill_owned_processes',harness.imports)
        self.assertNotIn('m1_owned_processes',harness.imports)
        self.assertEqual(Path(harness.client_calls[0][0][1]).name,'prefill_attention_client.py')
        harness.publish.assert_not_called()
        harness.summary.assert_not_called()
        self.assertTrue(harness.cleanup()['cleanup_complete'])
        for name in ('client.log','benchmark.exit','gpu-telemetry.jsonl','server-phases.jsonl','status.json'):
            self.assertFalse((harness.output/name).exists(),name)

    def test_publication_failure_is_terminal_and_preserves_primary(self):
        from megartx.prefill_attention_evidence import StreamingEvidence
        with tempfile.TemporaryDirectory() as directory:
            evidence=StreamingEvidence(directory)
            primary=KeyboardInterrupt('synthetic interrupted oracle')
            with patch.object(publication,'_publish_attention',side_effect=primary),self.assertRaises(KeyboardInterrupt) as caught:
                publication.publish_attention(evidence,{'plan_sha256':'a'*64},{},deadline=0,ownership=None)
            self.assertIs(caught.exception,primary)
            value=json.loads((Path(directory)/'storage-failure.json').read_text())
            self.assertEqual(value['exception_type'],'KeyboardInterrupt')
            self.assertIsNone(value['native_arithmetic_acceptance'])
            self.assertFalse((Path(directory)/'attention.json').exists())

    def test_imports_remain_cpu_only(self):
        result=subprocess.run([sys.executable,'-S','-c',
            "import megartx.prefill_attention_native_plan, megartx.prefill_attention_publication; "
            "import sys; sys.path.insert(0,'scripts'); import prefill_attention_client; "
            "assert not any(n.split('.')[0] in {'torch','vllm','requests','numpy'} for n in sys.modules)"],
            cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

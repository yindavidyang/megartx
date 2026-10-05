"""Full synthetic storage-to-attention publication; no native execution/clearance.

Only a declared host-availability signal is mocked. Exact current terminal
source-catalog admission executes normally. Metadata/operands are synthetic. Complete streaming contributions,
current storage/frontier/client/cleanup gates, request/frame crossbinding,
read_capture, isolated owned reference worker, hard process limits, and final
publication all execute normally. Fsync is elided only during fixture setup;
the evidence suite covers real durability calls and failure handling.
"""
import copy, hashlib, importlib.util, json, os, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
from megartx import prefill_attention_native_plan as n, prefill_attention_plan as p
from megartx import prefill_attention_validation as v, prefill_attention_publication as pub
from megartx import prefill_diagnostic_plan as legacy, prefill_storage_plan as old
from megartx.prefill_attention_evidence import StreamingEvidence, RAW_FILES, _geometry
from megartx.controlled_kv_capture import CONFIG_SHA256
from test_prefill_storage_plan import receipt_fixture
from test_prefill_attention_integration import attention_control

def write(path,obj):path.write_text(json.dumps(obj))
def setup(directory):
    plan=legacy.freeze_plan(list(range(2048)),'8060d8b6a9e335042925f9666394e19c0b50b7cc',ROOT,{
      'config_sha256':CONFIG_SHA256,'index_sha256':'a'*64,'shard_stats':{'synthetic.safetensors':{'size':1,'mtime_ns':1}}})
    plan.update(schema=n.SCHEMA,purpose=n.PURPOSE,control_spec=copy.deepcopy(n.CONTROL_SPEC),
      source_hashes={f:n.file_sha(ROOT/f) for f in n.SOURCES},installed_sources=dict(n.INSTALLED_CONTRACT))
    plan['plan_sha256']=n.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    oldev,ctl,ownership=receipt_fixture(directory,plan)
    for f in old.RAW_FILES:(directory/f).unlink()
    (directory/'storage-client.json').unlink()
    ev=StreamingEvidence(directory); spec=n.cpu_capture_plan(plan,ROOT)
    ev.create_capture(spec['plan_sha256'])
    with patch('megartx.prefill_attention_evidence.os.fsync'):
      for f in RAW_FILES:
        rows,width=_geometry(f)
        for row in range(rows):ev.contribute(f,row*width,bytes(width))
    manifest=ev.seal_raw()
    ctl=attention_control(ctl);ctl['raw_samples_sha256']=n.digest(manifest)
    write(directory/'control.json',ctl)
    observer=json.loads((directory/'observer.json').read_text()); client=json.loads((directory/'client.json').read_text())
    binding=json.loads((directory/'storage-binding.json').read_text());binding['purpose']=n.PURPOSE;write(directory/'storage-binding.json',binding)
    ev.write('attention-binding.json',{'schema':'megartx-prefill-attention-binding-v1','plan_sha256':plan['plan_sha256'],
      'capture_plan_sha256':spec['plan_sha256'],'owner_pid':1,'owner_start_ticks':1,'hooks_bound':True,
      'installed_sources':dict(n.INSTALLED_CONTRACT),'completion_policy':n.CONTROL_SPEC['completion_policy']})
    ev.write('cleanup.json',ownership)
    ev.write('attention-client.json',{'schema':'megartx-prefill-attention-client-v1','purpose':n.PURPOSE,
      'plan_sha256':plan['plan_sha256'],'status':'complete','control_sha256':n.digest(ctl),
      'output_ids_sha256':client['output_ids_sha256'],'sample_hashes_sha256':ctl['sample_hashes_sha256'],
      'storage_capture_end':2049,'metadata_only_decode_inputs':254,'numerical_qualified':False,'performance_qualified':False})
    arrays={f:bytes(size) for f,size in RAW_FILES.items()}; records=[]
    request=n.digest(n.native_request_identity(plan,observer['engine_request_id']))
    for start,end in p.FRAMES:
      inputs=n.digest(plan['tokens'][start:end] if start<2048 else [17])
      for layer in (0,5):
        route='paged_prefill' if start<2048 else ('xqa_decode' if layer==0 else 'paged_decode')
        records.append({'layer':layer,'start':start,'end':end,'frame_input_ids_sha256':inputs,
          'frame_identity_sha256':v.frame_identity(plan['plan_sha256'],request,start,end,inputs),
          'operator_binding_sha256':'e'*64,'cache_mapping_sha256':'f'*64,'semantics':v.semantic_contract(layer),
          **v.record_roots(arrays,layer,start,end),'dispatch':{'provider':v.ENTRYPOINTS[route][0],'entrypoint':route,
          'backend_source_sha256':v.BACKEND_SHA,'wrapper_source_sha256':v.ENTRYPOINTS[route][1],
          'split_plan_sha256':'1'*64,'kernel_binary_sha256':None,'kernel_profile_sha256':None,'native_rounding_contract':None}})
    ev.write('attention-records.json',{'schema':'megartx-prefill-attention-native-records-v1','native_plan_sha256':plan['plan_sha256'],
      'capture_plan_sha256':spec['plan_sha256'],'request_identity_sha256':request,'raw_manifest':manifest,
      'callbacks_restored':True,'records':records})
    path=ROOT/'scripts/prefill_owned_processes.py';sp=importlib.util.spec_from_file_location('review_attention_owned',path)
    module=importlib.util.module_from_spec(sp);sys.modules[sp.name]=module;sp.loader.exec_module(module)
    owner=module.OwnedProcesses(module.read_process(os.getpid()))
    return ev,plan,ownership,owner

class AttentionPublicationTests(unittest.TestCase):
    def test_full_actual_owned_worker_publication(self):
      with tempfile.TemporaryDirectory() as d:
        ev,plan,report,owner=setup(Path(d))
        with patch('megartx.prefill_attention_analysis._host_available_bytes',return_value=16<<30):
          receipt=pub.publish_attention(ev,plan,report,deadline=time.monotonic()+300,ownership=owner)
        self.assertEqual(receipt['status'],'independent_attention_error_observed')
        self.assertIsNone(receipt['native_arithmetic_acceptance'])
        self.assertEqual(len(receipt['raw_manifest']),8)
        self.assertEqual(sum(v['bytes'] for v in receipt['raw_manifest'].values()),5395952)
        self.assertFalse(receipt['numerical_qualified'])
        errors=json.loads((ev.directory/'attention-errors.json').read_text())
        self.assertTrue(errors['input_manifest_matches_validated_raw_manifest'])
        self.assertEqual(errors['result']['input_manifest'],receipt['raw_manifest'])
        resources=errors['resources']
        self.assertEqual(resources['address_space_limit_bytes'],512<<20)
        self.assertLessEqual(resources['peak_process_rss_bytes'],512<<20)
        self.assertLessEqual(resources['wall_seconds'],resources['wall_seconds_limit'])
        self.assertLessEqual(resources['wall_seconds_limit'],300)
        self.assertTrue(resources['ownership']['cleanup_complete'])
        self.assertEqual(resources['host_free_floor_bytes'],8<<30)
        self.assertEqual(resources['minimum_sampled_host_available_bytes'],16<<30)
        self.assertLessEqual(ev.sizes()['metadata_bytes'],2<<20)
        self.assertLessEqual(ev.sizes()['total_bytes'],8<<20)
    def test_broken_frontier_prevents_analysis(self):
      with tempfile.TemporaryDirectory() as d:
        ev,plan,report,owner=setup(Path(d));ctl=json.loads((ev.directory/'control.json').read_text())
        ctl['frontier']['checked_heads']=262;write(ev.directory/'control.json',ctl)
        with patch('megartx.prefill_attention_analysis.run_analysis') as oracle:
          with self.assertRaisesRegex(ValueError,'frontier'):pub.publish_attention(ev,plan,report,deadline=time.monotonic()+300,ownership=owner)
          oracle.assert_not_called()
        self.assertFalse((ev.directory/'attention-errors.json').exists());self.assertFalse((ev.directory/'attention.json').exists())
        self.assertTrue((ev.directory/'storage-failure.json').exists())
    def test_stale_decode_record_prevents_analysis(self):
      with tempfile.TemporaryDirectory() as d:
        ev,plan,report,owner=setup(Path(d));record=json.loads((ev.directory/'attention-records.json').read_text())
        record['records'][-1]['frame_input_ids_sha256']='a'*64;write(ev.directory/'attention-records.json',record)
        with patch('megartx.prefill_attention_analysis.run_analysis') as oracle:
          with self.assertRaisesRegex(ValueError,'stale'):pub.publish_attention(ev,plan,report,deadline=time.monotonic()+300,ownership=owner)
          oracle.assert_not_called()
        self.assertFalse((ev.directory/'attention-errors.json').exists());self.assertFalse((ev.directory/'attention.json').exists())
        self.assertTrue((ev.directory/'storage-failure.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)

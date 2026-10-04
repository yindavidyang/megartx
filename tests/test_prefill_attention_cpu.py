"""Synthetic CPU packet tests; none of these artifacts came from a GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from megartx import prefill_attention_plan as p
from megartx import prefill_attention_validation as v

ROOT=Path(__file__).resolve().parents[1]
REQUEST='b'*64
STORAGE='c'*64


def fixture(directory):
    spec=p.make_plan('a'*64,'d'*64,p.BASE_HEAD,ROOT)
    arrays={name:bytes(layout['bytes']) for name,layout in p.raw_layouts().items()}
    for name,data in arrays.items():(directory/name).write_bytes(data)
    records=[]
    for start,end in p.FRAMES:
        for layer in (0,5):
            inputs=p.digest(['synthetic_cpu_tokens',start,end])
            provider='xqa' if start==2048 else 'fa2'
            records.append({'layer':layer,'start':start,'end':end,'frame_input_ids_sha256':inputs,
                'frame_identity_sha256':v.frame_identity(spec['native_plan_sha256'],REQUEST,start,end,inputs),
                'operator_binding_sha256':'e'*64,'cache_mapping_sha256':'f'*64,
                'semantics':v.semantic_contract(layer),**v.record_roots(arrays,layer,start,end),
                'dispatch':{'provider':provider,'entrypoint':'xqa_decode' if start==2048 else 'paged_prefill',
                    'backend_source_sha256':v.BACKEND_SHA,
                    'wrapper_source_sha256':v.WRAPPER_SHA[provider],'split_plan_sha256':'1'*64,
                    'kernel_binary_sha256':None,'kernel_profile_sha256':None,'native_rounding_contract':None}})
    capture={'schema':v.CAPTURE_SCHEMA,'plan_sha256':spec['plan_sha256'],
        'native_plan_sha256':spec['native_plan_sha256'],'prompt_sha256':spec['prompt_sha256'],
        'request_identity_sha256':REQUEST,'inherited_storage_binding_sha256':STORAGE,
        'raw_manifest':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                        for name,data in arrays.items()},'records':records}
    save(directory,capture)
    return spec,capture


def save(directory,value):
    (directory/v.CAPTURE_FILE).write_text(json.dumps(value))


def read(directory,spec,**kwargs):
    return v.read_capture(directory,spec,expected_request_sha256=kwargs.get('request',REQUEST),
                          expected_storage_binding_sha256=kwargs.get('storage',STORAGE),source_root=ROOT)


class AttentionCPUPlanTests(unittest.TestCase):
    def test_new_imports_cannot_import_native_runtime(self):
        result=subprocess.run([sys.executable,'-S','-c',
            "import megartx.prefill_attention_plan, megartx.prefill_attention_validation; "
            "import sys; assert not {'torch','vllm','numpy','requests'} & set(sys.modules)"],
            cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_exact_layout_budget_and_storage_obligations(self):
        b=p.budget();self.assertEqual(b['raw_bytes'],5395952)
        self.assertEqual(b['retained_bound_bytes'],7493104)
        self.assertEqual(b['output_coordinates'],480)
        self.assertEqual(b['additional_cache_entry_d2h_bytes'],73388032)
        self.assertEqual(b['additional_q_and_output_head_d2h_bytes'],102400)
        self.assertEqual(b['reused_writer_d2h_additional_charge_bytes'],0)
        from megartx.prefill_storage_plan import CONTROL_SPEC
        for key in ('capture_frames','capture_end','checked_layer_frames','pre_rows','processed_rows','post_rows'):
            self.assertEqual(p.STORAGE_OBLIGATIONS[key],CONTROL_SPEC[key])
        self.assertEqual(p.raw_layouts()['attention-layer-05-k.bf16']['bytes'],4196352)

    def test_plan_is_cpu_only_and_rehashed_scope_drift_rejected(self):
        spec=p.make_plan('a'*64,'d'*64,p.BASE_HEAD,ROOT)
        p.validate_plan(spec,ROOT)
        for field,value in (('cpu_only',False),('native_integration_ready',True),('base_head','0'*40),
                            ('native_plan_sha256','a'*63),('source_hashes',{})):
            altered=copy.deepcopy(spec);altered[field]=value
            altered['plan_sha256']=p.digest({k:v for k,v in altered.items() if k!='plan_sha256'})
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_plan(altered,ROOT)
        for field,value in (('numerical_qualified',True),('native_arithmetic_acceptance',True),
                            ('prompt_tokens',2048.0),('query_positions',[0]),('local_cross_group_coverage',True)):
            altered=copy.deepcopy(spec);altered['specification'][field]=value
            altered['plan_sha256']=p.digest({k:v for k,v in altered.items() if k!='plan_sha256'})
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_plan(altered,ROOT)

    def test_writer_selection_uses_original_full_head_and_sparse_value_order(self):
        for layer in (0,5):
            s=p.LAYERS[layer];n=s['kv_heads']*s['dim']
            values=[0x3f80+i%64 for i in range(n)]
            raw=struct.pack('<'+'H'*n,*values)
            k,vv=p.select_writer_row(layer,2048,raw,raw)
            expected_k=[values[h*s['dim']+d] for h in s['selected_kv_heads'] for d in range(s['dim'])]
            expected_v=[values[h*s['dim']+d] for h in s['selected_kv_heads'] for d in p.coordinates(layer)]
            self.assertEqual(k[2],struct.pack('<'+'H'*len(expected_k),*expected_k))
            self.assertEqual(vv[2],struct.pack('<'+'H'*len(expected_v),*expected_v))
            self.assertEqual(k[1]+len(k[2]),p.raw_layouts()[k[0]]['bytes'])
        self.assertEqual(p.select_writer_row(1,0,b'',b''),())

    def test_head_selection_offsets_and_nonfinite_extent_guards(self):
        for layer in (0,5):
            dim=p.LAYERS[layer]['dim'];raw=struct.pack('<'+'H'*dim,*[0x3f80+i%64 for i in range(dim)])
            for role in ('q','o'):
                name,offset,data=p.select_head_row(layer,2048,p.LAYERS[layer]['q_heads'][-1],role,raw)
                self.assertEqual(offset+len(data),p.raw_layouts()[name]['bytes'])
        for args in ((0,2048,2,'q',bytes(512)),(0,2049,0,'q',bytes(512)),
                     (0,0,0,'q',bytes(1024)),(0,0,0,'o',b'\x80\x7f'+bytes(510))):
            with self.assertRaises(ValueError):p.select_head_row(*args)


class AttentionPacketTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.directory=Path(t.name);self.spec,self.capture=fixture(self.directory)

    def test_complete_packet_validates_only_new_domain(self):
        result=read(self.directory,self.spec)
        self.assertEqual(len(result['arrays']),8)
        self.assertFalse(result['native_execution_attested'])
        self.assertTrue(result['external_native_storage_frontier_validation_required'])
        self.assertLess(result['evidence_bytes'],8<<20)

    def test_external_identity_binding_is_mandatory(self):
        for key in ('request','storage'):
            with self.assertRaises(ValueError):read(self.directory,self.spec,**{key:'0'*64})

    def test_rehashed_wrong_geometry_and_mask_scale_are_rejected(self):
        for field,value in (('q_heads',8),('head_dim',256),('window_left',1024),('causal',False),
                            ('sm_scale',0.5),('q_scale',2.0),('pos_encoding_mode','ROPE_LLAMA'),
                            ('custom_mask',True),('sinks',True),('kv_sharing',True),('q_dtype','float32')):
            c=copy.deepcopy(self.capture);c['records'][1]['semantics'][field]=value;save(self.directory,c)
            with self.subTest(field=field),self.assertRaises(ValueError):read(self.directory,self.spec)

    def test_call_order_types_missing_records_and_stale_anchor_rejected(self):
        mutants=[]
        c=copy.deepcopy(self.capture);c['records'].pop();mutants.append(c)
        c=copy.deepcopy(self.capture);c['records'][0],c['records'][1]=c['records'][1],c['records'][0];mutants.append(c)
        c=copy.deepcopy(self.capture);c['records'][0]['layer']=False;mutants.append(c)
        c=copy.deepcopy(self.capture);c['records'][-1]['start']=2047;mutants.append(c)
        c=copy.deepcopy(self.capture);c['records'][-1]['frame_input_ids_sha256']='0'*64;mutants.append(c)
        for c in mutants:
            save(self.directory,c)
            with self.assertRaises(ValueError):read(self.directory,self.spec)

    def test_unknown_dispatch_source_and_acceptance_claims_rejected(self):
        for field,value in (('provider','auto'),('provider','fa3'),('wrapper_source_sha256','0'*64),
                            ('native_rounding_contract','passes'),('kernel_binary_sha256','short')):
            c=copy.deepcopy(self.capture);c['records'][0]['dispatch'][field]=value;save(self.directory,c)
            with self.subTest(field=field),self.assertRaises(ValueError):read(self.directory,self.spec)

    def test_fa2_decode_binds_decode_entrypoint_not_prefill_wrapper(self):
        c=copy.deepcopy(self.capture)
        for record in c['records'][-2:]:
            record['dispatch'].update(provider='fa2',entrypoint='paged_decode',wrapper_source_sha256=v.WRAPPER_SHA['xqa'])
        save(self.directory,c);read(self.directory,self.spec)
        c['records'][-1]['dispatch']['wrapper_source_sha256']=v.WRAPPER_SHA['fa2']
        save(self.directory,c)
        with self.assertRaises(ValueError):read(self.directory,self.spec)

    def test_empty_metadata_files_cannot_exhaust_unbounded_directory_memory(self):
        for i in range(v.MAX_DIRECTORY_ENTRIES):
            (self.directory/f'empty-{i}.json').touch()
        with self.assertRaisesRegex(ValueError,'entry limit'):read(self.directory,self.spec)

    def test_fifo_open_is_nonblocking_in_evidence_and_source_readers(self):
        code="""
import os,tempfile
from pathlib import Path
from megartx import prefill_attention_validation as v,prefill_attention_plan as p
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'fifo';os.mkfifo(path);fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY)
    try:
        for fn in (lambda:v._read_regular(fd,'fifo',32),lambda:p._source_hash(path)):
            try:fn()
            except ValueError:pass
            else:raise AssertionError('FIFO was accepted')
    finally:os.close(fd)
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True,timeout=3)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_rehashed_payload_cannot_escape_operator_root(self):
        for role in ('k','v','q','o'):
            name=f'attention-layer-00-{role}.bf16';path=self.directory/name
            original=path.read_bytes();data=b'\x80\x3f'+original[2:];path.write_bytes(data)
            c=copy.deepcopy(self.capture);c['raw_manifest'][name]['sha256']=hashlib.sha256(data).hexdigest();save(self.directory,c)
            with self.subTest(role=role),self.assertRaises(ValueError):read(self.directory,self.spec)
            path.write_bytes(original)

    def test_rehashed_nonfinite_and_truncated_raw_rejected(self):
        name='attention-layer-05-k.bf16';path=self.directory/name;original=path.read_bytes()
        for data in (original[:-2],b'\x80\x7f'+original[2:]):
            path.write_bytes(data);c=copy.deepcopy(self.capture)
            c['raw_manifest'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};save(self.directory,c)
            with self.assertRaises(ValueError):read(self.directory,self.spec)

    def test_extra_old_raw_failure_record_and_metadata_overflow_rejected(self):
        for name,data in (('head-position-2047.bf16',b'\0\0'),('storage-failure.json',b'{}'),
                          ('temporary.json',b'x'*(2<<20))):
            path=self.directory/name;path.write_bytes(data)
            with self.subTest(name=name),self.assertRaises(ValueError):read(self.directory,self.spec)
            path.unlink()

    def test_symlink_hardlink_directory_and_duplicate_keys_rejected(self):
        for kind in ('symlink','hardlink','directory'):
            path=self.directory/'extra.json'
            if kind=='symlink':path.symlink_to(self.directory/v.CAPTURE_FILE)
            elif kind=='hardlink':os.link(self.directory/v.CAPTURE_FILE,path)
            else:path.mkdir()
            with self.subTest(kind=kind),self.assertRaises(ValueError):read(self.directory,self.spec)
            path.rmdir() if kind=='directory' else path.unlink()
        path=self.directory/v.CAPTURE_FILE
        path.write_text('{"schema":"x",'+path.read_text()[1:])
        with self.assertRaises(ValueError):read(self.directory,self.spec)


if __name__=='__main__':unittest.main()

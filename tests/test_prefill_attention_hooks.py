"""CPU execution of exact source-extracted forward/run/lazy/decorator bodies.

No Torch/vLLM/native module imports, device access, model weights or kernel
replay. Host uint16 arrays model bit-preserving views and delayed completion.
"""
import ast
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from megartx import prefill_attention_hooks as hooks
from megartx import prefill_attention_plan as spec
from megartx.prefill_attention_validation import record_roots
from megartx.prefill_native import GpuScratch
from megartx.prefill_storage import ManagerInputs, TransferBudget
from test_prefill_storage_native import Tensor, CUDA, CPU, BF16, I32, U8, fake_torch

FIXTURES=Path(__file__).parent/'fixtures/prefill-attention'
NEW_SOURCES={'vllm.utils.flashinfer','flashinfer.api_logging','flashinfer.trace.template','flashinfer.utils'}


class RuntimeTensor(Tensor):
    def _view(self,array,dtype=None):
        return RuntimeTensor(array,dtype or self.dtype,self.device,backing=self.backing,
                             events=self.events,label=self.label)
    def __getitem__(self,index):return self._view(self.array[index])
    def stride(self,axis=None):
        values=super().stride()
        return values if axis is None else values[axis]
    def size(self,axis=None):return self.shape if axis is None else self.shape[axis]
    def dim(self):return len(self.shape)
    @property
    def ndim(self):return len(self.shape)
    def permute(self,*axes):return self._view(self.array.transpose(axes))
    def split(self,width,dim=-1):
        bounds=range(width,self.shape[dim],width)
        return tuple(self._view(v) for v in np.split(self.array,list(bounds),axis=dim))
    def contiguous(self):
        if self.device is CUDA:
            if not self.array.flags.c_contiguous:raise AssertionError('Unexpected GPU copy')
            return self
        return super().contiguous()
    def view(self,*shape):
        if shape==(U8,):return self._view(self.array.view(np.uint8),U8)
        return self._view(self.array.reshape(shape))
    def unsqueeze(self,axis):return self._view(np.expand_dims(self.array,axis))
    def fill_(self,value):self.array.fill(value);return self
    def copy_(self,value):self.array[:]=value.array;return self


class Evidence:
    def __init__(self):
        self.raw={name:bytearray(layout['bytes']) for name,layout in spec.raw_layouts().items()}
        self.writes=set()
    def writer_row(self,layer,position,key,value):
        for name,offset,data in spec.select_writer_row(layer,position,key,value):
            self.raw[name][offset:offset+len(data)]=data
    def head_row(self,layer,position,head,role,raw):
        name,offset,data=spec.select_head_row(layer,position,head,role,raw)
        if (name,offset) in self.writes:raise RuntimeError('Duplicate evidence contribution')
        self.writes.add((name,offset));self.raw[name][offset:offset+len(data)]=data
    def read_selection(self,layer,position):
        s=spec.LAYERS[layer];widths=(len(s['selected_kv_heads'])*s['dim']*2,len(s['selected_kv_heads'])*16)
        return tuple(bytes(self.raw[f'attention-layer-{layer:02d}-{role}.bf16'][position*w:(position+1)*w])
                     for role,w in zip(('k','v'),widths))


class SourceRuntime:
    """Load only audited extracted bodies into isolated fake source modules."""
    def __init__(self):
        self.stack=ExitStack();self.directory=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.events=[];self.pending=[];self.error=None;self.native_mutation=None
        self.torch=fake_torch(self.events);self.torch.Tensor=RuntimeTensor
        self.torch.float16='float16';self.torch.float8_e4m3fn='fp8';self.torch.float8_e5m2='fp8-e5';self.torch.float32='float32'
        self.torch.empty_like=lambda t:RuntimeTensor(np.zeros_like(t.array),events=self.events)
        self.torch.Size=tuple
        self.torch.device=lambda **kw:CUDA
        self.torch.cuda.get_device_properties=lambda device:NS(major=9,minor=0,multi_processor_count=1)
        original_sync=self.torch.cuda.synchronize
        def synchronize():
            original_sync()
            while self.pending:self.pending.pop(0)()
        self.torch.cuda.synchronize=synchronize
        self.modules={};self.manifest=json.loads((FIXTURES/'manifest.json').read_text())
        self.hashes={entry['module']:entry['fixture_sha256'] for entry in self.manifest.values()}
        self.stack.enter_context(patch.dict(sys.modules,{}))
        self.stack.enter_context(patch.dict(os.environ,{'FLASHINFER_LOGLEVEL':'0','FLASHINFER_TRACE_DUMP':'0'}))
        self.stack.enter_context(patch.object(hooks,'SOURCE_HASHES',self.hashes))
        self.trace=self.load('trace',{'os':os})
        class TraceTemplate:
            name_prefix='gqa';op_type='attention';init=None
            def build_fi_trace_fn(self,api):return lambda **kw:None
        self.trace.TraceTemplate=TraceTemplate
        self.logging=self.load('logging',{'functools':functools,'inspect':inspect,'_API_LOG_LEVEL':0,'_TRACE_REGISTRY':[]})
        trace=TraceTemplate()
        native_leaf=hooks.SourceGuard.native_leaf
        def cpu_leaf(guard,callback):
            if callback in (self.kernel,self.xqa_native):return {'CPU_TEST_OPAQUE_LEAF':True}
            return native_leaf(guard,callback)
        self.stack.enter_context(patch.object(hooks.SourceGuard,'native_leaf',cpu_leaf))
        self.utils=self.load('utils',{'torch':self.torch,'functools':functools,'Enum':Enum,'math':math,
                                   '_cache_buf':{('alibi_slopes_16',CUDA):self.tensor((16,))}})
        common={'torch':self.torch,'flashinfer_api':self.logging.flashinfer_api,
                'gqa_paged_prefill_trace':trace,'gqa_paged_decode_trace':trace,'xqa_batch_decode_trace':trace,
                'device_support_pdl':lambda d:False,'_unpack_paged_kv_cache':lambda v,l:v,
                '_check_cached_qkv_data_type':lambda *a:None,'_check_pos_encoding_mode':lambda *a:None,
                'check_trtllm_gen_sm107_only_feature':lambda *a:None,'check_shape_dtype_device':lambda *a:None,
                'is_float8':lambda t:False,'_get_cache_alibi_slopes_buf':lambda *a:None,
                'MaskMode':NS(CAUSAL=NS(value=1),NON_CAUSAL=NS(value=0),CUSTOM=NS(value=2),MULTIITEMSCORING=NS(value=3)),
                'TensorLayout':{'HND':NS(value=1),'NHD':NS(value=0)},'math':math,
                'get_device_sm_count':lambda d:1,'xqa':self.xqa,'xqa_trace':trace,
                'functools':functools,'SimpleNamespace':NS,'register_custom_op':lambda *a,**k:lambda f:f,
                'register_fake_op':lambda *a,**k:lambda f:f,
                'get_batch_prefill_uri':lambda *a:'cpu-test-native',
                'gen_batch_prefill_module':lambda *a:NS(build_and_load=lambda:NS(plan=lambda *a:None,paged_run=self.kernel,ragged_run=lambda *a:None)),
                'gen_xqa_module':lambda *a:NS(name='cpu-test-xqa',build_and_load=lambda:NS(xqa_wrapper=self.xqa_native))}
        common.update({name:getattr(self.utils,name) for name in ('_unpack_paged_kv_cache',
            '_check_cached_qkv_data_type','device_support_pdl','is_float8','_get_cache_alibi_slopes_buf',
            'check_shape_dtype_device','MaskMode','TensorLayout','check_trtllm_gen_sm107_only_feature',
            '_check_pos_encoding_mode','get_device_sm_count','get_compute_capability')})
        self.xqa_module=self.load('xqa',common)
        common['xqa']=self.xqa_module.xqa
        self.prefill=self.load('prefill',common)
        self.decode=self.load('decode',common)
        self.shim=self.load('shim',{'functools':functools,'_missing':lambda *a,**k:None,
                                  'has_flashinfer':lambda:True,'_get_submodule':self.modules.get})
        # Natural prior call resolves only the pre-existing lazy cache in tests.
        getter=hooks._closure(self.shim.flashinfer_xqa_batch_decode_with_kv_cache)['_get_impl']
        getter()
        self.backend=self.load('backend',{'torch':self.torch,'dataclass':dataclass,'Enum':Enum,
            'is_quantized_kv_cache':lambda dtype:False,'FP8_DTYPE':'fp8','FP4_DTYPE':'fp4',
            'canonicalize_singleton_dim_strides':lambda t:t,'logger':NS(debug=lambda *a:None),
            'BatchPrefillWithPagedKVCacheWrapper':self.prefill.BatchPrefillWithPagedKVCacheWrapper,
            'BatchDecodeWithPagedKVCacheWrapper':self.decode.BatchDecodeWithPagedKVCacheWrapper,
            'BatchAttentionWithAttentionSinkWrapper':type('UnusedSink',(),{}),
            'flashinfer_xqa_batch_decode_with_kv_cache':self.shim.flashinfer_xqa_batch_decode_with_kv_cache,
            'get_flashinfer_layout_string':lambda layout:layout.name,
            '_get_trtllm_workspace_buffer':lambda:self.tensor((8*1024*1024+1024,),dtype=U8),
            'is_strictly_contiguous':lambda t:True})
        self.gemma=self.load('gemma',{})
    def load(self,name,extra):
        entry=self.manifest[name+'.source'];module_name=entry['module']
        path=self.directory/(name+'.py');path.write_bytes((FIXTURES/(name+'.source')).read_bytes())
        module=ModuleType(module_name);module.__file__=str(path);module.__spec__=importlib.util.spec_from_file_location(module_name,path)
        module.__dict__.update(extra);sys.modules[module_name]=module;self.modules[module_name]=module
        exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),module.__dict__)
        return module
    def tensor(self,shape,word=0,dtype=BF16,label='tensor'):
        dt=np.uint8 if dtype==U8 else np.int32 if dtype==I32 else np.uint16
        return RuntimeTensor(np.full(shape,word,dtype=dt),dtype,events=self.events,label=label)
    def kernel(self,*args):
        q,k,v,out=args[3],args[4],args[5],args[10]
        self.events.append(('native',q,k,v,out))
        if self.error:raise self.error
        if self.native_mutation:self.native_mutation()
        self.pending.append(lambda:out.array.fill(0x3f80))
    def xqa_native(self,*args):
        self.events.append(('xqa-native',args))
        out=args[6]
        if self.error:raise self.error
        self.pending.append(lambda:out.array.fill(0x3f80))
    def xqa(self,q,k,v,table,lengths,out,*args,**kwargs):
        self.events.append(('xqa',q,k,v,out,kwargs))
        if self.error:raise self.error
        self.pending.append(lambda:out.array.fill(0x3f80))
    def close(self):self.stack.close()


class Harness:
    def __init__(self,start=0,end=256):
        self.r=SourceRuntime();r=self.r
        p=self.p=NS(torch=r.torch,started=True,completed=False,failed=False,frame=None,
            plan={'plan_sha256':'a'*64},deadline=float('inf'),layers={},descriptors={},writer_seen={0,5},
            transfer=TransferBudget(),evidence=Evidence(),scratch=GpuScratch(r.torch,8<<20))
        p.transfer.reserve(65536,'attention_metadata')
        p.manager=ManagerInputs([16],[16],[1]);ids=list(range(1,145));p.manager.append(0,(ids,),True)
        p.access=NS(groups={f'layer-{i}':(0,None) for i in (0,5)})
        p.ledger=NS(pending=(start,end,{}),frames=0)
        p.frame=NS(owner=p,sequence=0,request_id='cmpl-megartx-prefill-'+('a'*20)+'-0-12345678',
            input_ids=r.tensor((end-start,),dtype=I32),identities={},context=NS(attn_metadata={},no_compile_layers={}))
        for layer in (0,5):
            s=spec.LAYERS[layer];heads,dim=s['kv_heads'],s['dim']
            backing=np.zeros((145,16,heads,2*dim),dtype=np.uint16)
            cache=RuntimeTensor(backing.transpose(0,2,1,3),backing=backing,events=r.events,label='cache')
            attn=NS(kv_cache=cache,kv_sharing_target_layer_name=None,_q_scale_float=1.0,_k_scale_float=1.0,
                    _v_scale_float=1.0,_q_scale=r.tensor((1,)),kv_cache_dtype='auto')
            impl=r.backend.FlashInferImpl();impl.__dict__.update(num_heads=16,num_kv_heads=heads,head_size=dim,
                dcp_world_size=1,sinks=None,is_kvcache_nvfp4=False,kv_sharing_target_layer_name=None,
                scale=1.0,logits_soft_cap=0.0,window_left=1023 if layer==0 else -1,kv_cache_dtype='auto',
                cache_dtype='auto',kv_cache_layout=NS(layer_view_order=(0,1,2,3),name='HND'),bmm1_scale=None,bmm2_scale=None,o_sf_scale=None)
            attn.impl=impl;parent=r.gemma.Gemma4Attention();parent.attn=attn;parent.is_kv_shared_layer=False
            p.layers[layer]=(f'layer-{layer}',parent,attn,impl)
            identity=hooks.tensor_identity(cache)
            p.frame.identities[layer]=(identity[0],identity[1],identity[2],identity[3],identity[4],identity[5],identity[7])
            p.frame.context.no_compile_layers[f'layer-{layer}']=attn
        self.h=hooks.AttentionHooks(p,installed_sources={k:r.hashes[k] for k in NEW_SOURCES})
        def poison(primary):
            p.failed=True
            self.h.restore(primary)
        p.poison_storage=poison
        # Populate all written prefix rows, including current frame. Runtime
        # cache and writer evidence are independently produced from same inputs.
        for layer in (0,5):
            s=spec.LAYERS[layer];width=s['kv_heads']*s['dim']*2
            for position in range(end):
                key=bytes(width);value=bytes(width)
                p.evidence.writer_row(layer,position,key,value)
                self.h.processed_row(layer,position,position+16,key,value)
        self.configure(start,end)
    def configure(self,start,end,routes=None):
        r,p=self.r,self.p;p.ledger.pending=(start,end,{})
        for layer in (0,5):
            route=(routes or {}).get(layer,'paged_prefill' if end-start==256 else 'paged_decode')
            if route=='xqa_decode':
                p.layers[layer][3].kv_cache_layout=NS(layer_view_order=(0,2,1,3),name='NHD')
                decode=r.backend.FlashInferTrtllmAPIDecode(r.backend.FlashInferDecodeKernel.XQA,
                    RuntimeTensor(np.array([p.manager.expanded(0)],dtype=np.int32),I32,events=r.events),
                    r.tensor((1,),word=end,dtype=I32),end)
                prefill=None
            else:
                cls=r.prefill.BatchPrefillWithPagedKVCacheWrapper if route=='paged_prefill' else r.decode.BatchDecodeWithPagedKVCacheWrapper
                w=cls()
                for node in ast.walk(ast.parse((FIXTURES/('prefill.source' if route=='paged_prefill' else 'decode.source')).read_text())):
                    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self':
                        if node.attr.startswith('_'):setattr(w,node.attr,None)
                w.__dict__.update(_backend='fa2',_jit_module=None,_use_cuda_graph=False,_kv_layout='HND',
                    _pos_encoding_mode='NONE',_sm_scale=1.0,_logits_soft_cap=0.0,_window_left=1023 if layer==0 else -1,
                    _cached_q_data_type=BF16,_cached_kv_data_type=BF16,_cached_o_data_type=BF16,
                    _num_qo_heads=16,_num_kv_heads=spec.LAYERS[layer]['kv_heads'],_causal=True,
                    _use_fp16_qk_reduction=False,_token_pos_in_items_len=0,_qo_indptr_last=end-start,
                    _q_len_per_req=1,_window_right=-1,_plan_info=[end,start,16],_use_tensor_cores=True,
                    _cached_module=r.prefill.get_batch_prefill_module('fa2'),_workspace_size=0,
                    _float_workspace_buffer=None,_int_workspace_buffer=None)
                pages=(end+15)//16
                w._paged_kv_indptr_buf=RuntimeTensor(np.array([0,pages],dtype=np.int32),I32,events=r.events)
                w._paged_kv_indices_buf=RuntimeTensor(np.array(p.manager.expanded(0)[:pages],dtype=np.int32),I32,events=r.events)
                w._paged_kv_last_page_len_buf=r.tensor((1,),word=(end-1)%16+1,dtype=I32)
                w._qo_indptr_buf=RuntimeTensor(np.array([0,end-start],dtype=np.int32),I32,events=r.events)
                prefill=r.backend.FIPrefill(w) if route=='paged_prefill' else None
                decode=r.backend.FIDecode(w) if route=='paged_decode' else None
            meta=r.backend.FlashInferMetadata(end-start,None,BF16,BF16,
                0 if end-start==256 else 1,0 if end-start==256 else 1,
                1 if end-start==256 else 0,256 if end-start==256 else 0,True,prefill,decode,False,None)
            p.frame.context.attn_metadata[f'layer-{layer}']=meta
    def call(self,layer=0,**changed):
        p,r=self.p,self.r;start,end,_=p.ledger.pending;dim=spec.LAYERS[layer]['dim']
        _,_,attn,impl=p.layers[layer]
        args=dict(layer=attn,query=r.tensor((end-start,16,dim),label='q'),
                  key=r.tensor((end-start,spec.LAYERS[layer]['kv_heads'],dim)),
                  value=r.tensor((end-start,spec.LAYERS[layer]['kv_heads'],dim)),kv_cache=attn.kv_cache,
                  attn_metadata=p.frame.context.attn_metadata[f'layer-{layer}'],
                  output=r.tensor((end-start,16,dim),label='o'))
        args.update(changed);self.last_args=args
        result=impl.forward(**args)
        return result
    def close(self):
        self.h.restore();self.r.close()


class AttentionHooksTests(unittest.TestCase):
    def harness(self,*args,**kwargs):
        h=Harness(*args,**kwargs);self.addCleanup(h.r.close);self.addCleanup(h.h.restore);return h

    def test_fixture_hashes_and_exact_fragment_provenance(self):
        manifest=json.loads((FIXTURES/'manifest.json').read_text())
        for name,entry in manifest.items():
            raw=(FIXTURES/name).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),entry['fixture_sha256'])
            self.assertEqual(entry['source_sha256'],hooks.SOURCE_HASHES[entry['module']])
            lines=raw.decode().splitlines(keepends=True);tree=ast.parse(raw)
            owners={}
            for node in tree.body:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                    owners[node.name]=node
                    if isinstance(node,ast.ClassDef):
                        owners.update({node.name+'.'+child.name:child for child in node.body if isinstance(child,ast.FunctionDef)})
            for item in entry['fragments']:
                node=owners[item['owner']];start=min([node.lineno,*[n.lineno for n in node.decorator_list]])
                part=''.join(lines[start-1:node.end_lineno])
                self.assertEqual(hashlib.sha256(part.encode()).hexdigest(),item['sha256'])

    def test_actual_prefill_forward_and_run_bodies_bind_shared_layer_and_sync_before_o(self):
        h=self.harness()
        a=h.call(0);b=h.call(5)
        self.assertEqual(a.array[0,0,0],0x3f80);self.assertEqual(b.array[0,0,0],0x3f80)
        self.assertEqual([r['layer'] for r in h.h.records],[0,5])
        h.h.require_frame_complete(h.p.frame)
        for layer in (0,5):
            r=h.h.records[layer==5]
            self.assertEqual(record_roots(h.p.evidence.raw,layer,0,256),{k:r[k] for k in ('cache_inputs_sha256','query_sha256','output_sha256')})
            self.assertNotIn('run',vars(h.p.frame.context.attn_metadata[f'layer-{layer}'].prefill.wrapper))
        events=h.r.events
        native=next(i for i,e in enumerate(events) if isinstance(e,tuple) and e[0]=='native')
        output=next(i for i,e in enumerate(events) if isinstance(e,tuple) and e[:2]==('cpu','o'))
        self.assertTrue(any(isinstance(e,tuple) and e[0]=='synchronize' for e in events[native:output]))

    def test_all_twelve_source_native_calls_exact_roots_and_transfer_domains(self):
        h=self.harness()
        for sequence,(start,end) in enumerate(spec.FRAMES):
            for layer in (0,5):
                s=spec.LAYERS[layer];width=s['kv_heads']*s['dim']*2
                for position in range(end):
                    if position not in h.h.writer_rows[layer]:
                        h.p.evidence.writer_row(layer,position,bytes(width),bytes(width))
                        h.h.processed_row(layer,position,position+16,bytes(width),bytes(width))
            h.p.frame.sequence=h.p.ledger.frames=sequence
            h.p.frame.input_ids=h.r.tensor((end-start,),dtype=I32)
            h.configure(start,end)
            h.call(0)
            if end-start==256:
                # The real source may share wrapper owners across layer calls;
                # association must come from the active enclosing layer ticket.
                first=h.p.frame.context.attn_metadata['layer-0'].prefill.wrapper
                second=h.p.frame.context.attn_metadata['layer-5'].prefill.wrapper
                first.__dict__.update(second.__dict__)
                h.p.frame.context.attn_metadata['layer-5'].prefill.wrapper=first
            h.call(5)
            h.h.require_frame_complete(h.p.frame)
        self.assertEqual(len(h.h.require_complete()),12)
        for record in h.h.records:
            roots=record_roots(h.p.evidence.raw,record['layer'],record['start'],record['end'])
            self.assertEqual(roots,{key:record[key] for key in roots})
        self.assertEqual(h.p.transfer.kinds,{'attention_metadata':65536,
            'attention_cache_entry':73388032,'attention_q_output':102400})

    def test_consumed_helper_semantic_enum_and_source_producer_substitutions_rejected(self):
        for mode in ('unpack','nested_expand','enum','enum_value','producer','native_leaf','backend',
                     'maybe_quant','parent','helper_code','split_scale','xqa'):
            with self.subTest(mode=mode):
                h=self.harness();w=h.p.frame.context.attn_metadata['layer-0'].prefill.wrapper
                if mode=='unpack':h.r.prefill._unpack_paged_kv_cache=lambda kv,layout:(h.r.tensor(kv[0].shape),h.r.tensor(kv[1].shape))
                elif mode=='nested_expand':h.r.utils._expand_4d=lambda tensor,layout:h.r.tensor(tensor.shape)
                elif mode=='enum':h.r.prefill.MaskMode=NS(CAUSAL=NS(value=0))
                elif mode=='enum_value':h.r.utils.MaskMode.CAUSAL._value_=0
                elif mode=='producer':w._cached_module.paged_run=lambda *args:None
                elif mode in ('native_leaf','backend'):
                    name='paged_run_func' if mode=='native_leaf' else 'backend'
                    callback=w._cached_module.paged_run
                    for variable,cell in zip(callback.__code__.co_freevars,callback.__closure__):
                        if variable==name:cell.cell_contents=(lambda *a:None) if mode=='native_leaf' else 'fa3'
                elif mode=='maybe_quant':h.p.layers[0][3].maybe_quant_query=lambda q,*a:q
                elif mode=='parent':h.p.layers[0][1].forward=lambda *a:None
                elif mode=='helper_code':h.r.utils._expand_4d.__code__=(lambda x,y:x).__code__
                elif mode=='split_scale':h.r.prefill._split_scale_param=lambda scale:(None,2.0)
                elif mode=='xqa':h.r.decode.xqa=lambda *a,**kw:None
                with self.assertRaises(RuntimeError):h.call()
                self.assertEqual(h.h.records,[])
                self.assertFalse(any(e[0]=='native' for e in h.r.events if isinstance(e,tuple)))

    def test_preexisting_helper_replacement_and_midcall_mutation_poison(self):
        r=SourceRuntime();self.addCleanup(r.close)
        r.prefill._unpack_paged_kv_cache=lambda kv,layout:kv
        with self.assertRaisesRegex(RuntimeError,'helper alias'):
            hooks.AttentionHooks(NS(torch=r.torch),installed_sources={k:r.hashes[k] for k in NEW_SOURCES})
        h=self.harness();h.r.native_mutation=lambda:setattr(h.r.utils,'_expand_4d',lambda x,y:x)
        with self.assertRaisesRegex(RuntimeError,'helper replaced'):h.call()
        self.assertTrue(h.h.failed);self.assertEqual(h.h.records,[])

    def test_semantic_table_and_plan_mutation_during_native_call_cannot_seal(self):
        for mode in ('scale','window','causal','dtype','layout','table_owner','table_value','plan_owner','quant_method'):
            with self.subTest(mode=mode):
                h=self.harness();w=h.p.frame.context.attn_metadata['layer-0'].prefill.wrapper
                def mutate():
                    if mode=='scale':w._sm_scale=0.5
                    elif mode=='window':w._window_left=7
                    elif mode=='causal':w._causal=False
                    elif mode=='dtype':w._cached_q_data_type='float16'
                    elif mode=='layout':w._kv_layout='NHD'
                    elif mode=='table_owner':w._paged_kv_indices_buf=h.r.tensor(w._paged_kv_indices_buf.shape,dtype=I32)
                    elif mode=='table_value':w._paged_kv_indices_buf.array[0]=99
                    elif mode=='plan_owner':w._plan_info=list(w._plan_info)
                    elif mode=='quant_method':h.p.layers[0][3].maybe_quant_query=lambda q,*a:q
                h.r.native_mutation=mutate
                with self.assertRaises(RuntimeError):h.call()
                self.assertTrue(h.h.failed);self.assertEqual(h.h.records,[])

    def test_interrupted_second_xqa_install_restores_first_resolver_hook(self):
        h=self.harness(2048,2049);h.configure(2048,2049,{0:'xqa_decode'});h.h.records=[{}]*10
        resolver=h.r.xqa_module.get_xqa_module
        original=hooks.ObservedCallHook
        primary=KeyboardInterrupt('second installation interrupted')
        def install(owner,name,*args,**kwargs):
            if name=='flashinfer_xqa_batch_decode_with_kv_cache':raise primary
            return original(owner,name,*args,**kwargs)
        with patch.object(hooks,'ObservedCallHook',side_effect=install):
            with self.assertRaises(KeyboardInterrupt) as caught:h.call()
        self.assertIs(caught.exception,primary)
        self.assertIs(h.r.xqa_module.get_xqa_module,resolver)
        self.assertIsNone(h.h.active_ticket)
        self.assertTrue(h.h.failed)

    def test_singleton_stride_canonicalization_keeps_original_element_addresses(self):
        h=self.harness(2048,2049);h.configure(2048,2049,{0:'xqa_decode'});h.h.records=[{}]*10
        def canonical(tensor):
            if len(tensor.shape)==3 and tensor.shape[0]==1:
                array=np.lib.stride_tricks.as_strided(tensor.array,shape=tensor.shape,
                    strides=(1234*tensor.array.itemsize,*tensor.array.strides[1:]))
                return tensor._view(array)
            return tensor
        h.r.backend.canonicalize_singleton_dim_strides=canonical
        self.assertEqual(h.call(0).array[0,0,0],0x3f80)

    def test_source_size_is_bounded_before_read(self):
        h=self.harness();path=Path(h.r.trace.__file__)
        with path.open('wb') as stream:stream.truncate((1<<20)+1)
        with patch.object(hooks.os,'read',side_effect=AssertionError('Source was read before size check')):
            with self.assertRaisesRegex(RuntimeError,'Bounded regular'):
                h.h.guard.module('flashinfer.trace.template')

    def test_d512_paged_decode_route_executes_real_body_without_dispatch_forcing(self):
        h=self.harness(2048,2049)
        # Prior ten selected records are represented only for ordering in this
        # focused single-frame test; full twelve-call test below makes them live.
        h.h.records=[{}]*10
        h.call(0);h.call(5)
        self.assertEqual(h.h.records[-1]['dispatch']['entrypoint'],'paged_decode')
        self.assertEqual(h.last_args['query'].shape,(1,16,512))

    def test_local_xqa_cached_lazy_alias_and_actual_inner_source_body(self):
        h=self.harness(2048,2049);h.configure(2048,2049,{0:'xqa_decode'})
        h.h.records=[{}]*10;original=h.r.backend.flashinfer_xqa_batch_decode_with_kv_cache
        output=h.call(0)
        self.assertIs(h.r.backend.flashinfer_xqa_batch_decode_with_kv_cache,original)
        event=next(e for e in h.r.events if isinstance(e,tuple) and e[0]=='xqa-native')
        self.assertEqual(event[1][3],1024)
        self.assertEqual(event[1][4],16.0)
        self.assertEqual(output.array[0,0,0],0x3f80)
        self.assertEqual(h.h.records[-1]['dispatch']['entrypoint'],'xqa_decode')

    def test_missing_new_installed_closure_or_wrong_origin_fails_before_install(self):
        with self.assertRaisesRegex(RuntimeError,'not admitted'):hooks.SourceGuard({})
        h=self.harness();h.r.trace.__spec__.origin='/different.py'
        with self.assertRaisesRegex(RuntimeError,'origin'):h.call()

    def test_decorator_forged_wrapped_or_trace_enable_rejected(self):
        for mode in ('wrapped','trace','logging'):
            with self.subTest(mode=mode):
                h=self.harness();w=h.p.frame.context.attn_metadata['layer-0'].prefill.wrapper
                if mode=='wrapped':type(w).run.__wrapped__=lambda *a:None
                elif mode=='trace':os.environ['FLASHINFER_TRACE_DUMP']='1'
                else:h.r.logging._API_LOG_LEVEL=10
                with self.assertRaises(RuntimeError):h.call()
                self.assertFalse(any(e[0]=='native' for e in h.r.events if isinstance(e,tuple)))

    def test_actual_cache_addresses_and_retained_archive_independently_checked(self):
        for mode in ('cache','archive','manager'):
            with self.subTest(mode=mode):
                h=self.harness()
                if mode=='cache':h.p.layers[0][2].kv_cache.array[1,0,0,0]=0x3f80
                elif mode=='archive':h.p.evidence.raw['attention-layer-00-k.bf16'][0]=1
                else:h.p.manager.tables=(tuple(range(2,146)),)
                with self.assertRaises(RuntimeError):h.call()
                self.assertFalse(any(e[0]=='native' for e in h.r.events if isinstance(e,tuple)))

    def test_wrong_semantics_tables_or_dtype_fails_closed(self):
        for key,value in (('_sm_scale',0.5),('_causal',False),('_custom_mask_buf',object()),
                          ('_backend','auto'),('_cached_q_data_type','float16')):
            with self.subTest(key=key):
                h=self.harness();w=h.p.frame.context.attn_metadata['layer-0'].prefill.wrapper
                setattr(w,key,value)
                with self.assertRaises((RuntimeError,AssertionError)):h.call()
                self.assertTrue(h.h.failed)
        h=self.harness();w=h.p.frame.context.attn_metadata['layer-0'].prefill.wrapper
        w._paged_kv_indices_buf.array[0]=99
        with self.assertRaisesRegex(RuntimeError,'table/length'):h.call()

    def test_native_exception_and_cleanup_failure_keep_primary_exception(self):
        h=self.harness();primary=KeyboardInterrupt('native interruption');h.r.error=primary
        h.r.torch.cuda.failures[5]=RuntimeError('drain failed')
        h.r.torch.cuda.failures[6]=RuntimeError('drain failed')
        with self.assertRaises(KeyboardInterrupt) as caught:h.call()
        self.assertIs(caught.exception,primary);self.assertTrue(h.h.failed)
        self.assertTrue(h.h.drain_failed);self.assertTrue(h.h.hooks[0].active)
        self.assertEqual(h.h.records,[])
        self.assertTrue(any('drain failed' in note for note in getattr(primary,'__notes__',())))

    def test_reentry_duplicate_missing_call_and_owner_mutation(self):
        h=self.harness();h.r.native_mutation=lambda:h.call(5)
        with self.assertRaisesRegex(RuntimeError,'reentrant'):h.call()
        h=self.harness();h.call()
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):h.call()
        h=self.harness()
        with self.assertRaisesRegex(RuntimeError,'omitted'):h.h.require_frame_complete(h.p.frame)
        h=self.harness();h.p.layers[0][3].forward=lambda *a,**k:None
        with self.assertRaisesRegex(RuntimeError,'replaced'):h.h.require_current()
        # Cleanup must retain foreign replacement rather than clobber it.
        h.h.restored=True

    def test_cold_lazy_cache_foreign_cached_callee_and_global_xqa_refused(self):
        for mode in ('cold','foreign','global'):
            with self.subTest(mode=mode):
                h=self.harness(2048,2049);h.configure(2048,2049,{0:'xqa_decode',5:'xqa_decode'})
                h.h.records=[{}]*10
                getter=hooks._closure(h.r.shim.flashinfer_xqa_batch_decode_with_kv_cache)['_get_impl']
                if mode=='cold':getter.cache_clear()
                if mode=='foreign':h.r.decode.xqa_batch_decode_with_kv_cache=lambda **kw:None
                with self.assertRaises(RuntimeError):h.call(5 if mode=='global' else 0)

    def test_observed_hook_preserves_positional_keywords_results_and_exception(self):
        class Owner:
            def call(self,first,second=2,*,third=3):
                captured.append((first,second,third));return result
        owner=Owner();captured=[];result=object();seen=[]
        hook=hooks.ObservedCallHook(owner,'call',lambda args:seen.append(dict(args)),lambda ticket,r:self.assertIs(r,result),lambda e:None)
        self.assertIs(owner.call('x',5,third=9),result)
        self.assertEqual(captured,[('x',5,9)]);self.assertEqual(seen,[{'first':'x','second':5,'third':9}])
        hook.restore();self.assertNotIn('call',vars(owner))


if __name__=='__main__':unittest.main()

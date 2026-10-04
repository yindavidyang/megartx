"""Actual-code CPU adversarial checks. Synthetic boundaries never establish native fit.
Adapted from independent review reproductions; no real Torch/vLLM import.
"""
import ast
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from megartx.loaded_engine_access import LoadedEngineAccess, owned_page_ranges, reject_page_alias
from contextlib import nullcontext
from megartx.prefill_native import NativeProvider, RequestLedger, GpuScratch, retained_host_bytes, install_native_observer
from megartx.prefill_diagnostic_plan import Evidence

class Tensor:
    def __init__(self, values, pointer=1234): self.values, self.pointer = values, pointer
    def cpu(self): return self
    def tolist(self): return self.values
    def data_ptr(self): return self.pointer
    def item(self): return self.values
    def __len__(self): return len(self.values)
    def __getitem__(self, key):
        value = self.values[key[0]][key[1]] if isinstance(key, tuple) else self.values[key]
        return Tensor(value, self.pointer)

TORCH = NS(equal=lambda a,b: a.values == b.values)

def frame_fixture(start=0, end=256):
    access = LoadedEngineAccess.__new__(LoadedEngineAccess)
    access.torch, access.model, access.registry = TORCH, object(), {}
    access.common, access.groups = {}, {}
    owner, tables, placements, metas, slots, identities = NS(layers={}), {}, [], {}, {}, {}
    for layer in range(30):
        name = f"layer-{layer}"
        ids = list(range(1, 73))
        actual = Tensor([ids], pointer=10000+layer)
        table = NS(get_device_tensor=lambda n,a=actual:a, block_size=32, num_blocks_per_row=[72])
        tables[layer] = table
        slots[layer] = [ids[p//32]*32+p%32 for p in range(start,end)]
        meta = NS(slot_mapping=Tensor(slots[layer]))
        common = NS(num_reqs=1,num_actual_tokens=end-start, query_start_loc=Tensor([0,end-start]),
            query_start_loc_cpu=Tensor([0,end-start]),seq_lens=Tensor([end]),causal=True,
            block_table_tensor=actual, slot_mapping=meta.slot_mapping)
        access.common[id(meta)] = (meta,common,layer,(name,))
        metas[name] = meta
        owner.layers[layer] = (name,)
        page = 8*32*512*2
        size = 256*page
        identities[layer] = (layer+1, 100000000+layer*size, size,0,(256,8,32,512),(8*32*512,32*512,512,1),'cuda:0')
        placements.append(NS(layers=[name],size=size,offset=0,layer_stride=size,block_stride=page,host_resident=False))
        access.groups[name] = (layer, NS(block_size=32,page_size_bytes=page))
    batch = NS(num_reqs=1,req_ids=['request'],block_table=tables)
    access._runner = NS(get_model=lambda:access.model,input_batch=batch,_kernel_block_sizes={i:32 for i in range(30)},
                       kv_cache_config=NS(kv_cache_tensors=placements))
    context = NS(no_compile_layers=access.registry,attn_metadata=metas)
    return access,owner,context,Tensor(list(range(start,end))),slots,identities

class FrameReview(unittest.TestCase):
    def test_actual_bind_rejects_query_and_slot_drift_before_write(self):
        for kind in ('length','slot','placement','alias'):
            with self.subTest(kind=kind):
                a,o,c,p,s,i=frame_fixture()
                if kind=='length': a.common[id(c.attn_metadata['layer-0'])][1].seq_lens=Tensor([255])
                if kind=='slot': s[0][0]=4000
                if kind=='placement': a._runner.kv_cache_config.kv_cache_tensors[0].offset=2
                if kind=='alias': i[1]=(i[1][0],i[0][1],*i[1][2:])
                with self.assertRaises((RuntimeError,ValueError)):
                    a.bind_frame(o,0,c,Tensor(list(range(256))),p,s,i)
    def test_retained_query_prefix_remapping_rejected_before_model_work(self):
        ledger=RequestLedger(list(range(2048)))
        a,o,c,p,s,i=frame_fixture()
        first=a.bind_frame(o,0,c,Tensor(list(range(256))),p,s,i)
        ledger.begin(list(range(256)),p.tolist(),s,i,'request');ledger.complete();ledger.sampled(17,True)
        a,o,c,p,s,i=frame_fixture(256,512)
        # Same backing allocations and new writer rows. Only a retained query page changes.
        a._runner.input_batch.block_table[5].get_device_tensor(1).values[0][0]=100
        with self.assertRaisesRegex(RuntimeError,'Retained native query block'):
            a.bind_frame(o,1,c,Tensor(list(range(256,512))),p,s,i,ledger.positions)
        self.assertEqual(first.block_tables[5][0],1)
        self.assertIsNone(ledger.pending)

    def test_every_required_local_position_and_global_prefix_is_bound(self):
        for layer,block in ((0,8),(5,0),(5,17)):
            a,o,c,p,s,i=frame_fixture(1280,1536)
            retained={l:{pos:(pos//32+1)*32+pos%32 for pos in range(1280)} for l in range(30)}
            a._runner.input_batch.block_table[layer].get_device_tensor(1).values[0][block]=100
            with self.subTest(layer=layer,block=block),self.assertRaisesRegex(RuntimeError,'Retained native query block'):
                a.bind_frame(o,5,c,Tensor(list(range(1280,1536))),p,s,i,retained)
        # Expired local rows are outside the query union; full-context storage
        # still owns disjoint physical pages, checked independently.
        a,o,c,p,s,i=frame_fixture(1280,1536)
        a._runner.input_batch.block_table[0].get_device_tensor(1).values[0][0]=100
        a.bind_frame(o,5,c,Tensor(list(range(1280,1536))),p,s,i,retained)

    def test_historical_group_overlays_use_owned_page_intervals(self):
        local=(1,1000,3276*131072,0,(3276,8,16,512),(65536,512,4096,1),'cuda:0')
        full=(2,1000,3276*131072,0,(3276,2,32,1024),(65536,1024,2048,1),'cuda:0')
        reject_page_alias({0:owned_page_ranges(local,[1,2]),5:owned_page_ranges(full,[3,4])})
        with self.assertRaises(ValueError):
            reject_page_alias({0:owned_page_ranges(local,[1,2]),5:owned_page_ranges(full,[2,3])})

class Provider:
    def __init__(self,runner,*unused):
        self.access=NS(model=runner.model,builders={id(runner.builder):1})
        self.failed=False;self.log=[];self.scratch=NS(scope=lambda phase:nullcontext())
    def prepare_inputs(self,result): self.log.append('prepare')
    def begin(self,*args): self.log.append('begin');return object()
    def finish(self,*args): self.log.append('finish')
    def head(self,*args): self.log.append('head')
    def sampled(self,*args): self.log.append('sample')
    def abort(self): self.failed=True;self.log.append('abort')

def hook_fixture():
    class Builder:
        def build(self,*a,**kw):
            if getattr(self,'error',None): raise self.error
            return object()
    class Model:
        def forward(self,*a,**kw):
            if getattr(self,'error',None): raise self.error
            return object()
        def compute_logits(self,*a,**kw): return object()
    class Runner:
        def __init__(self): self.model,self.builder=Model(),Builder()
        def initialize_kv_cache(self,*a,**kw): return 'initialized'
        def _prepare_inputs(self,*a,**kw):
            if getattr(self,'error',None): raise self.error
            return (None,None,1)
        def _sample(self,*a,**kw): return object()
    modules={}
    for name in ('vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu_model_runner',
                 'vllm.v1.attention','vllm.v1.attention.backends','vllm.v1.attention.backends.flashinfer'):
        modules[name]=ModuleType(name)
    modules['vllm.v1.worker.gpu_model_runner'].GPUModelRunner=Runner
    modules['vllm.v1.attention.backends.flashinfer'].FlashInferMetadataBuilder=Builder
    return modules,Runner,Model

class HookReview(unittest.TestCase):
    def exercise(self,callback):
        modules,Runner,Model=hook_fixture()
        env={'MEGARTX_PREFILL_NATIVE_PLAN':'synthetic-plan','MEGARTX_PREFILL_NATIVE_DIR':'synthetic-dir','MEGARTX_SCALE_MODE':'native'}
        with patch.dict(sys.modules,modules),patch.dict(os.environ,env,clear=True), \
             patch('megartx.prefill_native.load_plan',return_value={}),patch('megartx.prefill_native.NativeProvider',Provider),patch('megartx.prefill_native.verify_adapter_sources',return_value={}):
            install_native_observer(TORCH,Model)
            runner=Runner();runner.initialize_kv_cache()
            provider=Runner._prepare_inputs.__closure__
            providers=next(cell.cell_contents for cell in provider if isinstance(cell.cell_contents,dict))
            callback(runner,providers[id(runner)])
    def test_forward_interrupt_poison_preserves_same_primary(self):
        def callback(r,p):
            error=KeyboardInterrupt('forward interrupt');r.model.error=error
            with self.assertRaises(KeyboardInterrupt) as caught:r.model.forward([],[])
            self.assertIs(caught.exception,error);self.assertTrue(p.failed)
        self.exercise(callback)
    def test_prepare_interrupt_poison_preserves_primary_and_rejects_reuse(self):
        def callback(r,p):
            error=KeyboardInterrupt('prepare interrupt');r.error=error
            with self.assertRaises(KeyboardInterrupt) as caught:r._prepare_inputs()
            self.assertIs(caught.exception,error);self.assertTrue(p.failed)
            with self.assertRaisesRegex(RuntimeError,'Poisoned'):r._prepare_inputs()
        self.exercise(callback)
    def test_builder_interrupt_poison_preserves_primary_and_rejects_reuse(self):
        def callback(r,p):
            error=KeyboardInterrupt('builder interrupt');r.builder.error=error
            with self.assertRaises(KeyboardInterrupt) as caught:r.builder.build(0,object())
            self.assertIs(caught.exception,error);self.assertTrue(p.failed)
            with self.assertRaisesRegex(RuntimeError,'Poisoned'):r._prepare_inputs()
        self.exercise(callback)
    def test_abort_failure_does_not_replace_primary_interrupt(self):
        def callback(r,p):
            error=SystemExit('prepare primary');r.error=error
            def failed_abort(): raise RuntimeError('secondary abort')
            p.abort=failed_abort
            with self.assertRaises(SystemExit) as caught:r._prepare_inputs()
            self.assertIs(caught.exception,error);self.assertTrue(p.failed)
            if hasattr(error,'__notes__'):self.assertIn('secondary abort',error.__notes__[0])
            with self.assertRaisesRegex(RuntimeError,'Poisoned'):r.builder.build(0,object())
        self.exercise(callback)

    def test_default_off_installer_performs_no_imports(self):
        with patch.dict(os.environ,{},clear=True):install_native_observer(TORCH,object())
    def test_stale_ticket_rejected_by_actual_finish(self):
        provider=NativeProvider.__new__(NativeProvider)
        ticket=NS(owner=provider,sequence=0)
        provider.frame=ticket;provider.ledger=NS(frames=1)
        with self.assertRaisesRegex(RuntimeError,'Stale/reused'):provider.finish(ticket,object())

class ShapeAndSourceReview(unittest.TestCase):
    def test_actual_head_rejects_shape_and_dtype(self):
        for shape,dtype in (((1,262143),'fp32'),((1,262144),'bf16')):
            with self.subTest(shape=shape,dtype=dtype):
                provider=NativeProvider.__new__(NativeProvider)
                model=object()
                provider.started,provider.completed,provider.failed=True,False,False
                provider.access=NS(model=model)
                provider.frame,provider.hidden,provider.logit_seen=None,object(),False
                provider.torch=NS(float32='fp32')
                with self.assertRaisesRegex(RuntimeError,'head shape'):
                    provider.head(model,NS(shape=(1,2816)),NS(shape=shape,dtype=dtype))
    def test_actual_access_rejects_changed_loaded_source(self):
        from megartx.controlled_kv_capture import SOURCE_HASHES
        Runner=type('GPUModelRunner',(),{'__module__':'vllm.v1.worker.gpu_model_runner'})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'drift.py';path.write_text('# changed loaded implementation bytes\n')
            module=ModuleType(next(iter(SOURCE_HASHES)));module.__file__=str(path)
            with patch.dict(sys.modules,{module.__name__:module}):
                with self.assertRaisesRegex(RuntimeError,'Installed observation source drift'):
                    LoadedEngineAccess(Runner(),object(),TORCH)
    def test_actual_finish_rejects_final_hidden_shape(self):
        a,o,c,p,s,i=frame_fixture()
        provider=NativeProvider.__new__(NativeProvider)
        provider.torch=NS(Tensor=Tensor,bfloat16='bf16',cuda=NS(synchronize=lambda:None))
        provider.ledger=NS(frames=0,pending=(0,256,s),positions={i:{} for i in range(30)})
        provider.read_frame=lambda positions,tokens:(s,{},i,[])
        provider.access=NS(bind_frame=lambda *args:NS(block_tables={}),model=object())
        provider.boundaries={};provider.hash_rows=lambda *args:{}
        ticket=NS(owner=provider,sequence=0,positions=p,input_ids=Tensor(list(range(256))),
                  context=c,identities=i,slots=s,block_tables={})
        provider.frame=ticket
        module=ModuleType('vllm.forward_context');module.get_forward_context=lambda:c
        result=Tensor([]);result.shape=(255,2816);result.dtype='bf16'
        with patch.dict(sys.modules,{'vllm':ModuleType('vllm'),'vllm.forward_context':module}):
            with self.assertRaisesRegex(RuntimeError,'final hidden rows'):
                provider.finish(ticket,result)
    def test_host_ledger_separately_charged_from_gpu_scratch(self):
        ledger=RequestLedger(list(range(2048)))
        identities={i:('identity',i) for i in range(30)}
        for start in range(2303):
            if start<2048 and start%256:continue
            count=256 if start<2048 else 1
            tokens=list(ledger.tokens[start:start+count]) if start<2048 else ledger.outputs[-1:]
            slots={i:[int(str(n)) for n in range(start+32,start+count+32)] for i in range(30)}
            ledger.begin(tokens,list(range(start,start+count)),slots,identities,'r')
            ledger.complete();ledger.sampled(17,ledger.end<2048)
        def size(value,seen):
            if id(value) in seen:return 0
            seen.add(id(value));total=sys.getsizeof(value)
            if isinstance(value,dict):total+=sum(size(k,seen)+size(v,seen) for k,v in value.items())
            elif isinstance(value,(list,tuple)):total+=sum(size(v,seen) for v in value)
            return total
        self.assertEqual(retained_host_bytes(ledger.positions),size(ledger.positions,set()))
        self.assertGreater(retained_host_bytes(ledger.positions),4<<20)
        receipt=GpuScratch(NS(cuda=CudaCounters()),8<<20).receipt()
        self.assertEqual(receipt['domain'],'incremental_gpu_allocator_bytes')
        self.assertTrue(receipt['host_heap_excluded'])

class ExecutedSourceReview(unittest.TestCase):
    def test_stale_adapter_copy_and_frozen_origin_rejected(self):
        import hashlib,json,shutil
        import megartx.prefill_native as executing
        from megartx.prefill_diagnostic_plan import SOURCES,freeze_plan,load_plan,verify_adapter_sources,BASE
        from megartx.controlled_kv_capture import CONFIG_SHA256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'planned-source'
            for relative in SOURCES:
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target)
            target=root/'src/megartx/prefill_native.py'
            target.write_bytes(target.read_bytes()+b'\n# different planned source bytes\n')
            checkpoint={'config_sha256':CONFIG_SHA256,'index_sha256':'a'*64,
                        'shard_stats':{'synthetic.safetensors':{'size':1,'mtime_ns':1}}}
            plan=freeze_plan(list(range(2048)),BASE,root,checkpoint)
            path=Path(d)/'plan.json';path.write_text(json.dumps(plan))
            self.assertEqual(load_plan(path,root),plan)
            self.assertNotEqual(hashlib.sha256(Path(executing.__file__).read_bytes()).hexdigest(),
                                plan['source_hashes']['src/megartx/prefill_native.py'])
            with self.assertRaisesRegex(ValueError,'Executing adapter'):
                verify_adapter_sources(plan)
            # Matching origins but stale helper bytes also fail.
            plan=freeze_plan(list(range(2048)),BASE,root,checkpoint,ROOT/'src')
            with self.assertRaisesRegex(ValueError,'Executing adapter'):
                verify_adapter_sources(plan)
            good=freeze_plan(list(range(2048)),BASE,ROOT,checkpoint)
            self.assertTrue(verify_adapter_sources(good))
            for name in ('megartx.prefill_native','megartx.loaded_engine_access','megartx.controlled_kv_capture','megartx.nvfp4_runtime'):
                module=ModuleType(name);module.__file__=str(root/'src/megartx'/ (name.split('.')[-1]+'.py'))
                module.__spec__=NS(origin=module.__file__)
                with patch.dict(sys.modules,{name:module}),self.assertRaisesRegex(ValueError,'source/origin drift'):
                    verify_adapter_sources(good)

class CudaCounters:
    def __init__(self):self.current,self.peak,self.reserved,self.reserved_peak=10<<20,10<<20,10<<20,10<<20
    def synchronize(self):pass
    def memory_allocated(self):return self.current
    def memory_reserved(self):return self.reserved
    def reset_peak_memory_stats(self):self.peak,self.reserved_peak=self.current,self.reserved
    def max_memory_allocated(self):return self.peak
    def max_memory_reserved(self):return self.reserved_peak
    def allocate(self,size):
        self.current+=size;self.reserved+=size
        self.peak=max(self.peak,self.current);self.reserved_peak=max(self.reserved_peak,self.reserved)
    def release(self,size):self.current-=size;self.reserved-=size

class AggregateGpuScratchTests(unittest.TestCase):
    def test_prior_model_allocated_and_reserved_peaks_survive_repeated_callbacks_and_error(self):
        cuda=CudaCounters();cuda.peak,cuda.reserved_peak=100<<20,128<<20
        scratch=GpuScratch(NS(cuda=cuda),8<<20)
        for phase in ('finish','head','begin'):
            with scratch.scope(phase):pass
            receipt=scratch.receipt()
            self.assertEqual(receipt['runwide_allocator_allocated_peak_bytes'],100<<20)
            self.assertEqual(receipt['runwide_allocator_reserved_peak_bytes'],128<<20)
            self.assertEqual(receipt['measured_phase_allocator_increment_peak_bytes'],0)
        cuda.peak,cuda.reserved_peak=140<<20,160<<20  # Intervening incumbent work.
        error=KeyboardInterrupt('observer primary')
        with self.assertRaises(KeyboardInterrupt) as caught:
            with scratch.scope('finish'):raise error
        self.assertIs(caught.exception,error)
        receipt=scratch.receipt()
        self.assertEqual(receipt['runwide_allocator_allocated_peak_bytes'],140<<20)
        self.assertEqual(receipt['runwide_allocator_reserved_peak_bytes'],160<<20)
        self.assertEqual(receipt['measured_phases'],4)

    def test_reserved_pool_growth_is_charged_and_rejected_even_without_tensor_growth(self):
        cuda=CudaCounters();scratch=GpuScratch(NS(cuda=cuda),8<<20)
        with self.assertRaisesRegex(RuntimeError,'scratch cap'):
            with scratch.scope('reserved_pool'):
                cuda.reserved_peak += (8<<20)+1
        receipt=scratch.receipt()
        self.assertEqual(receipt['measured_phase_allocator_increment_peak_bytes'],0)
        self.assertGreater(receipt['measured_phase_reserved_increment_peak_bytes'],8<<20)

    def test_concurrent_reservations_reject_before_second_allocation(self):
        cuda=CudaCounters();scratch=GpuScratch(NS(cuda=cuda),8<<20)
        with scratch.scope('rows'):
            with scratch.allocation(5<<20):
                cuda.allocate(5<<20)
                with self.assertRaisesRegex(RuntimeError,'before allocation'):
                    with scratch.allocation(4<<20):cuda.allocate(4<<20)
                with self.assertRaisesRegex(RuntimeError,'Concurrent'):
                    with scratch.scope('nested'):pass
                cuda.release(5<<20)
        receipt=scratch.receipt()
        self.assertEqual(receipt['managed_tensor_simultaneous_peak_bytes'],5<<20)
        self.assertEqual(receipt['measured_phase_allocator_increment_peak_bytes'],5<<20)
        self.assertEqual(receipt['measured_phases'],1)
        self.assertEqual(cuda.current,10<<20)  # Incumbent baseline is excluded.

    def test_framework_peak_overflow_rejects_fit_and_preserves_primary(self):
        for primary in (None,KeyboardInterrupt('primary')):
            cuda=CudaCounters();scratch=GpuScratch(NS(cuda=cuda),8<<20)
            with self.assertRaises(RuntimeError if primary is None else KeyboardInterrupt) as caught:
                with scratch.scope('framework'):
                    cuda.allocate((8<<20)+1);cuda.release((8<<20)+1)
                    if primary is not None:raise primary
            if primary is not None:self.assertIs(caught.exception,primary)
            self.assertGreater(scratch.receipt()['measured_phase_allocator_increment_peak_bytes'],8<<20)

if __name__=='__main__':unittest.main()

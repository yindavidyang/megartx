import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np

from megartx.m1_normal_capture import NormalCapture
from megartx.m1_normal_plan import CASES,ORIGIN,REVISION,digest,validate_plan
from compare_m1_normal import expected_records,arrays_for
from compare_m1_live import correlate_trace
from test_m1_live_comparison import trace


def plan():
    p = {**ORIGIN,"schema":"megartx-m1-normal-plan-v1","checkpoint_revision":REVISION,
         "outputs":4,"prefill_chunk":256,"continuation_token_id":7,"eos_token_ids":[1,2],
         "expected_live_calls":210,"expected_fallback_calls":150,"expected_model_forwards":12,
         "cases":[{"id":name,"prompt_token_ids":[3]*n,"prompt_sha256":digest([3]*n)} for name,n in CASES]}
    p["plan_sha256"] = digest(p)
    return validate_plan(p)


class TestNormalCapture(unittest.TestCase):
    def test_partial_prefill_heads_are_excluded_but_out_of_plan_rows_fail(self):
        obj=NormalCapture.__new__(NormalCapture);obj.plan=plan();obj.case=obj.plan["cases"][0];obj.logit_counter=0
        obj.logits(object(),[255],[3],"verified_storage_view",1)
        self.assertEqual(obj.logit_counter,0)
        with self.assertRaisesRegex(RuntimeError,"declared prediction"):
            obj.logits(object(),[260],[3],"verified_storage_view",1)

    def test_startup_unmarked_output_hook_remains_callable_without_cuda(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"plan.json";source.write_text(json.dumps(plan()))
            layers=[SimpleNamespace(layer_name=f"language_model.model.layers.{i}.moe.experts",_megartx={"ordinal":i})for i in range(30)]
            env={"MEGARTX_M1_NORMAL_PLAN":str(source),"MEGARTX_M1_NORMAL_DIR":str(root/"normal")}
            with patch.dict("os.environ",env),patch("megartx.m1_normal_capture.NormalKV"):
                obj=NormalCapture(object(),layers,"native")
            value=object()
            self.assertIs(obj.output(object(),value),value)
            self.assertFalse(obj.active)

    def test_begin_fences_and_copies_actual_rows_before_persistent_input_reuse(self):
        import json
        from megartx.m1_normal_plan import request
        class Tensor:
            ndim=1
            def __init__(self,value): self.value=value
            def numel(self): return self.value.size
            def long(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"plan.json";source.write_text(json.dumps(plan()))
            layers=[SimpleNamespace(layer_name=f"language_model.model.layers.{i}.moe.experts",_megartx={"ordinal":i})for i in range(30)]
            env={"MEGARTX_M1_NORMAL_PLAN":str(source),"MEGARTX_M1_NORMAL_DIR":str(root/"normal")}
            profile=SimpleNamespace(__enter__=lambda:None)
            events=[]
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append("fence")),
                profiler=SimpleNamespace(profile=lambda **kw:profile,ProfilerActivity=SimpleNamespace(CPU=1,CUDA=2)))
            with patch.dict("os.environ",env),patch("megartx.m1_normal_capture.NormalKV"):
                obj=NormalCapture(object(),layers,"native")
            obj.marker.write_text(json.dumps(request(obj.plan,obj.plan["cases"][0])))
            tokens=np.asarray([3]*256,dtype=np.int64);positions=np.arange(256,dtype=np.int64)
            with patch.dict("sys.modules",{"torch":torch}):obj.begin(Tensor(tokens),Tensor(positions))
            tokens[0]=7;positions[0]=256
            self.assertEqual(events,["fence"])
            self.assertEqual(obj.context["tokens"][0],3)
            self.assertEqual(obj.context["positions"][0],0)

    def test_fixed_expected_counts_separate_tail_from_decode(self):
        live,fallback,routes,kv = expected_records(plan())
        self.assertEqual((len(live),len(fallback),len(routes),len(kv)),(210,150,360,360))
        self.assertEqual(sum(v[-1]=="single_row_prefill_tail" for v in live),30)
        self.assertEqual(sum(v[-1]=="cached_decode" for v in live),180)
        self.assertEqual({v[3][0] for v in live},{256,257,258,259,1023,1024,1025})

    def test_route_observer_returns_original_objects_and_rejects_duplicate(self):
        class Tensor:
            def __init__(self,array,dtype): self.array,self.dtype,self.shape=array,dtype,array.shape
            def int(self): return self
            def cpu(self): return self
            def numpy(self): return self.array
        with tempfile.TemporaryDirectory() as directory:
            obj=NormalCapture.__new__(NormalCapture)
            layer=SimpleNamespace(_megartx={"ordinal":0})
            obj.layers={0:layer};obj.directory=Path(directory);obj.records=[]
            obj.total_forwards,obj.case_index,obj.forward_counter=1,0,1
            obj.context={"positions":np.asarray([256],dtype=np.int64),"tokens":np.asarray([3],dtype=np.int64),"seen":set()}
            x=Tensor(np.zeros((1,2816),dtype=np.uint16),"bf16")
            ids=Tensor(np.asarray([[4,9,15,28,37,60,80,117]],dtype=np.int32),"i32")
            weights=Tensor(np.ones((1,8),dtype=np.float32),"f32")
            with patch.dict("sys.modules",{"torch":SimpleNamespace(bfloat16="bf16",float32="f32")}),patch("megartx.m1_normal_capture.bits",side_effect=lambda t:t.array):
                a,b=obj.routes(layer,x,ids,weights)
                self.assertIs(a,ids);self.assertIs(b,weights)
                with self.assertRaisesRegex(RuntimeError,"owner"): obj.routes(layer,x,ids,weights)
            self.assertEqual(obj.records[0]["classification"],"single_row_prefill_tail")

    def test_normal_trace_requires_own_positive_correlations_and_no_artificial_scope(self):
        events=trace("fused")
        positive=correlate_trace(events,"fused",["normal_live_request"]*32,
            request_scope="normal_live_request",request_count=32,artificial_count=0)
        self.assertEqual(positive["positive_request_fused_launches"],32)
        self.assertEqual(positive["positive_artificial_fused_launches"],0)
        for changed in ("zero_hit","artificial","count"):
            altered=copy.deepcopy(events);scopes=["normal_live_request"]*32
            if changed=="zero_hit":
                for e in altered:
                    if e.get("cat")=="kernel" and e["name"]=="m1_maps_expand": e["args"]["correlation"]+=1000000
            elif changed=="artificial": scopes[0]="artificial_route_control"
            else: scopes.pop()
            with self.assertRaises(ValueError): correlate_trace(altered,"fused",scopes,
                request_scope="normal_live_request",request_count=32,artificial_count=0)

    def test_numeric_reader_rejects_changed_hash_object_and_unknown_fields(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"output-00-00.npz"
            for values in ({"routed_bits":np.zeros((1,2816),dtype=np.uint16)},
                           {"routed_bits":np.asarray([object()],dtype=object)},
                           {"unexpected":np.asarray([1])}):
                np.savez_compressed(p,**values)
                record={"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                if set(values)=={"routed_bits"} and values["routed_bits"].dtype.kind!="O":
                    self.assertEqual(len(list(arrays_for(Path(directory),[record]))),1)
                    record["sha256"]="0"*64
                with self.assertRaises(ValueError): list(arrays_for(Path(directory),[record]))

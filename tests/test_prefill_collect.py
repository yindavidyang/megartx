"""Synthetic CPU observers exercise the live collector; no device qualification."""

import copy
import ast
from contextlib import nullcontext
import hashlib
import inspect
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

from megartx import prefill_collect as c
from megartx import prefill_observe as o
from megartx import prefill_kv as kv
from megartx import prefill_runner as r
from megartx import controlled_kv_capture as historical
from test_prefill_runner import inputs, synthetic_records, DIGEST


def job_for(prompt=2048, chunk=256, mode="profile", region="attention"):
    manifest, plan, prompts = inputs(True)
    if prompt == 8192:
        prompts["token_ids"] = {"p8192": [i % 262144 for i in range(8192)]}
        plan["workloads"][1]["prompt_token_ids_sha256"] = r.digest(prompts["token_ids"]["p8192"])
    protocol = r.compile_protocol(manifest, plan, "p" + str(prompt), chunk)
    job = next(j for j in protocol["jobs"] if j["mode"] == mode and j["region"] == region)
    return job, prompts["token_ids"][job["workload"]], protocol, prompts


def observation(start, end, layer):
    low, high = c.required_kv(start, end, layer)
    return {"layer": layer, "kind": "global" if layer in c.GLOBAL_LAYERS else "local",
            "kv_dtype": "bfloat16", "required_start": low, "required_end": high,
            "retained_start": low, "retained_end": high, "k_storage_id": f"layer-{layer}-k-region",
            "v_storage_id": f"layer-{layer}-v-region", "processed_k_sha256": DIGEST,
            "processed_v_sha256": "b" * 64, "layout_sha256": DIGEST, "queries_complete": True}


def bootstrap(job, tokens):
    p = job["prompt_tokens"]
    return {"request_id": job["request_id"], "logit_position": p - 1, "predicts_position": p,
            "input_token_sha256": r.digest([tokens[-1]]), "logit_rows": 1, "vocab_size": 262144,
            "logit_bits_sha256": DIGEST, "source_site": "Gemma4ForCausalLM.compute_logits",
            "source_sha256": c.LOGIT_SITE_SHA256, "comparison_receipt_sha256": DIGEST,
            "sampling_policy": "greedy_seed_1234", "emitted_tokens": 1, "pending_anchor_sha256": r.digest([17]),
            "pending_anchor_position": p, "anchor_kv_written": False, "cached_length": p,
            "remaining_output_capacity": 255}


def finish_prompt(collector, routes=False):
    for start, end in collector.job["spans"]:
        collector.begin_forward(list(collector.tokens[start:end]), list(range(start, end)))
        for i in range(30):
            if routes:
                m = end - start
                collector.record_routes(i, [list(range(8)) for _ in range(m)], [[1.0] * 8 for _ in range(m)],
                    [m] * 8 + [0] * 120, 0, 0)
            collector.layer_complete(i, observation(start, end, i))
        collector.end_forward()


class FakeEvents:
    def __init__(self):
        self.calls = []
    def begin(self):
        self.calls.append("begin")
        return object()
    def end(self, ticket):
        self.calls.append("end")
        return ticket
    def synchronize(self):
        self.calls.append("sync")
    def resolve(self, ticket):
        if "sync" not in self.calls:
            raise ValueError("Not synchronized")
        self.calls.append("resolve")
        return 17.0, "synthetic-stream"


class CacheTests(unittest.TestCase):
    def test_required_union_1279_and_recycle_after_all_layers(self):
        ledger = c.CacheLedger(2048, 256, DIGEST)
        for start, end in ledger.spans[:4]:
            ledger.begin(start, end)
            for i in range(30):
                ledger.observe(observation(start, end, i))
            ledger.end()
        ledger.begin(1024, 1280)
        self.assertEqual(c.required_kv(1024, 1280, 0), (1, 1280))
        self.assertEqual(c.required_kv(1024, 1280, 5), (0, 1280))
        for i in range(30):
            ledger.observe(observation(1024, 1280, i))
        release = ledger.end()
        self.assertEqual(release[0], 256)
        self.assertEqual(release[5], 0)
        self.assertEqual(ledger.frames[-1]["layers"][0]["retained_start"], 1)

    def test_final_window_recycled_before_query_completion_poisoned(self):
        ledger = c.CacheLedger(2048, 2048, DIGEST)
        ledger.begin(0, 2048)
        value = observation(0, 2048, 0)
        value["retained_start"] = 1024
        with self.assertRaisesRegex(ValueError, "retained start"):
            ledger.observe(value)
        self.assertTrue(ledger.poisoned)
        with self.assertRaises(ValueError):
            ledger.begin(0, 2048)

    def test_cache_failures_do_not_publish_partial_frame(self):
        for field, value in (("kv_dtype", "fp8"), ("layer", True), ("queries_complete", False),
                             ("required_end", 255), ("layout_sha256", "b" * 64),
                             ("v_storage_id", "layer-0-k-region"), ("processed_v_sha256", "bad")):
            with self.subTest(field=field):
                ledger = c.CacheLedger(2048, 256, DIGEST)
                ledger.begin(0, 256)
                item = observation(0, 256, 0)
                item[field] = value
                with self.assertRaises(ValueError):
                    ledger.observe(item)
                self.assertTrue(ledger.poisoned)
                self.assertEqual(ledger.frames, [])
        ledger = c.CacheLedger(2048, 256, DIGEST)
        ledger.begin(0, 256)
        for i in range(29):
            ledger.observe(observation(0, 256, i))
        with self.assertRaisesRegex(ValueError, "thirty"):
            ledger.end()

    def test_cross_layer_alias_and_changed_owner_reject(self):
        ledger = c.CacheLedger(2048, 256, DIGEST)
        ledger.begin(0, 256)
        ledger.observe(observation(0, 256, 0))
        value = observation(0, 256, 1)
        value["k_storage_id"] = "layer-0-k-region"
        with self.assertRaisesRegex(ValueError, "Cross-layer"):
            ledger.observe(value)
        ledger = c.CacheLedger(2048, 256, DIGEST)
        ledger.begin(0, 256)
        for i in range(30):
            ledger.observe(observation(0, 256, i))
        ledger.end()
        ledger.begin(256, 512)
        value = observation(256, 512, 0)
        value["k_storage_id"] = "foreign-k-region"
        with self.assertRaisesRegex(ValueError, "owner changed"):
            ledger.observe(value)

    def test_entire_sweep_tails_and_final_cache_handoff(self):
        for prompt, chunks in c.CHUNKS.items():
            for chunk in chunks:
                with self.subTest(prompt=prompt, chunk=chunk):
                    job, tokens, protocol, prompts = job_for(prompt, chunk)
                    collector = c.PrefillCollector(job, tokens, DIGEST)
                    finish_prompt(collector)
                    self.assertEqual(sum(x["rows"] for x in collector.forwards), prompt)
                    self.assertEqual(collector.forwards[-1]["rows"], prompt % chunk or chunk)
                    if chunk == 255:
                        self.assertEqual(collector.forwards[-1]["rows"], 8 if prompt == 2048 else 32)
                    record = next(x for x in synthetic_records(protocol, prompts)["runs"] if x["run_id"] == job["run_id"])
                    collector.handoff(record["handoff"])
                    self.assertEqual(collector.handoff_record["capacity"], prompt + 256)


class WriterTests(unittest.TestCase):
    def test_actual_slot_union_rejects_early_final_window_reuse(self):
        ledger = kv.WriterLedger()
        capacities = {i: 2304 for i in range(30)}
        for start in range(0, 1024, 256):
            ledger.prepare(start, start+256, {i:list(range(start,start+256)) for i in range(30)}, capacities)
            ledger.complete()
        # A naive 1024-slot ring starts overwriting slots 0..255, including
        # still-required position1; it is rejected before any model dispatch.
        slots = {i: list(range(1024,1280)) for i in range(30)}
        slots[0] = list(range(256))
        with self.assertRaisesRegex(ValueError, "still needed"):
            ledger.prepare(1024,1280,slots,capacities)
        self.assertTrue(ledger.poisoned)
        self.assertEqual(ledger.end,1024)

    def test_full_context_partial_tails_and_global_growth(self):
        ledger = kv.WriterLedger()
        capacity = {i: 2304 for i in range(30)}
        for start,end in r.plan_contract.chunk_spans(2048,255):
            ledger.prepare(start,end,{i:list(range(start,end)) for i in range(30)},capacity)
            self.assertEqual(len(ledger.pending["positions"][5]),end)
            self.assertEqual(set(ledger.pending["positions"][0]),set(range(max(0,start-1023),end)))
            ledger.complete()
        self.assertEqual(set(ledger.positions[0]),set(range(1024,2048)))
        self.assertEqual(set(ledger.positions[5]),set(range(2048)))

    def test_alias_missing_layer_and_insufficient_capacity_fail_atomically(self):
        for what in ("alias","missing","capacity"):
            ledger = kv.WriterLedger()
            slots = {i:list(range(256)) for i in range(30)}
            capacity = {i:2304 for i in range(30)}
            if what == "alias": slots[0][-1] = 0
            if what == "missing": del slots[29]
            if what == "capacity": capacity[0] = 255
            with self.subTest(what=what), self.assertRaises(ValueError): ledger.prepare(0,256,slots,capacity)
            self.assertTrue(ledger.poisoned)
            self.assertEqual(ledger.positions,{i:{} for i in range(30)})


class CollectorTests(unittest.TestCase):
    def test_wrong_schedule_tokens_output_reserve_and_unsupported_workload_reject(self):
        for prompt, chunk in ((32768, 256), (2048, 1023), (8192, 32768)):
            with self.assertRaises(ValueError):
                c.CacheLedger(prompt, chunk, DIGEST)
        job, tokens, _, _ = job_for()
        for name, value in (("capacity_tokens", 2048), ("spans", [[0, 2048]]), ("observer_enabled", False)):
            changed = copy.deepcopy(job)
            changed[name] = value
            with self.assertRaises(ValueError):
                c.PrefillCollector(changed, tokens, DIGEST)
        for token_change, pos_change in ((True, False), (False, True)):
            collector = c.PrefillCollector(job, tokens, DIGEST)
            actual, positions = tokens[:256], list(range(256))
            if token_change:
                actual[3] += 1
            if pos_change:
                positions[-1] -= 1
            with self.assertRaisesRegex(ValueError, "scheduler"):
                collector.begin_forward(actual, positions)
            self.assertTrue(collector.failed)

    def test_incomplete_duplicate_and_nested_frames_poison(self):
        job, tokens, _, _ = job_for()
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:256], list(range(256)))
        with self.assertRaises(ValueError):
            collector.begin_forward(tokens[:256], list(range(256)))
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:256], list(range(256)))
        collector.layer_complete(0, observation(0, 256, 0))
        with self.assertRaises(ValueError):
            collector.layer_complete(0, observation(0, 256, 0))
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:256], list(range(256)))
        with self.assertRaises(ValueError):
            collector.end_forward()
        self.assertTrue(collector.cache.poisoned)

    def test_routes_actual_selected_positive_scheduled_and_zero_weights(self):
        job, tokens, _, _ = job_for(chunk=255, region="expert")
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:255], list(range(255)))
        ids = [list(range(8)) for _ in range(255)]
        weights = [[0.0] + [1.0] * 7 for _ in range(255)]
        collector.record_routes(0, ids, weights, [0] + [256] * 7 + [0] * 120, 2, 2)
        row = collector.routes[0]
        self.assertEqual(sum(row["selected_m"]), 255 * 8)
        self.assertEqual(row["positive_m"][0], 0)
        self.assertEqual(row["scheduled_m"][1], 256)
        self.assertEqual(row["route_sha256"], r.digest({"ids": ids, "weights": weights}))
        self.assertEqual(ids[0], list(range(8)))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            collector.record_routes(0, ids, weights, [255] * 8 + [0] * 120, 0, 0)

    def test_route_semantic_and_geometry_drift_reject(self):
        for what in ("duplicate", "negative", "nonfinite", "id_bool", "scheduled", "correction"):
            job, tokens, _, _ = job_for(region="expert")
            collector = c.PrefillCollector(job, tokens, DIGEST)
            collector.begin_forward(tokens[:256], list(range(256)))
            ids, weights = [list(range(8)) for _ in range(256)], [[1.0] * 8 for _ in range(256)]
            scheduled, correction = [256] * 8 + [0] * 120, 0
            if what == "duplicate": ids[0][0] = 1
            if what == "negative": weights[0][0] = -1.0
            if what == "nonfinite": weights[0][0] = float("nan")
            if what == "id_bool": ids[0][0] = False
            if what == "scheduled": scheduled[0] = 255
            if what == "correction": correction = 256 * 8 + 1
            with self.subTest(what=what), self.assertRaises(ValueError):
                collector.record_routes(0, ids, weights, scheduled, correction, 0)
            self.assertTrue(collector.failed)

    def test_timing_cannot_capture_routes_or_stage_events(self):
        job, tokens, _, _ = job_for(mode="timing", region=None)
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:256], list(range(256)))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            collector.record_routes(0, [], [], [], 0, 0)
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:256], list(range(256)))
        events = FakeEvents()
        with self.assertRaises(ValueError):
            with collector.stage("attention", 0, "site", DIGEST, (256, 256, 256), events):
                pass
        self.assertEqual(events.calls, [])
        with self.assertRaises(ValueError):
            o.OwnedHooks(collector, events, {})

    def test_bootstrap_and_decode_off_by_one(self):
        job, tokens, protocol, prompts = job_for(chunk=2048)
        collector = c.PrefillCollector(job, tokens, DIGEST)
        finish_prompt(collector)
        record = next(x for x in synthetic_records(protocol, prompts)["runs"] if x["run_id"] == job["run_id"])
        collector.handoff(record["handoff"])
        collector.bootstrap(bootstrap(job, tokens))
        self.assertEqual(collector.bootstrap_record["cached_length"], 2048)
        self.assertEqual(collector.bootstrap_record["pending_anchor_position"], 2048)
        for i in range(255):
            collector.begin_forward([17], [2048 + i], "decode")
            for layer in range(30): collector.layer_complete(layer)
            collector.end_forward()
        self.assertEqual(collector.index, 1)
        self.assertEqual(collector.decode_rows, 255)
        with self.assertRaisesRegex(ValueError, "budget"):
            collector.begin_forward([17], [2303], "decode")

    def test_bootstrap_bad_final_row_or_written_anchor_poisoned(self):
        for key, value in (("logit_position", 2048), ("predicts_position", 2049), ("anchor_kv_written", True),
                           ("cached_length", 2049), ("emitted_tokens", 0), ("remaining_output_capacity", 256),
                           ("input_token_sha256", "b" * 64), ("source_sha256", "b" * 64)):
            job, tokens, protocol, prompts = job_for(chunk=2048)
            collector = c.PrefillCollector(job, tokens, DIGEST)
            finish_prompt(collector)
            record = next(x for x in synthetic_records(protocol, prompts)["runs"] if x["run_id"] == job["run_id"])
            collector.handoff(record["handoff"])
            value_dict = bootstrap(job, tokens)
            value_dict[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): collector.bootstrap(value_dict)
            self.assertTrue(collector.failed)

    def test_timing_whole_prompt_order_and_backward_clock_reject(self):
        job, tokens, _, _ = job_for()
        collector = c.PrefillCollector(job, tokens, DIGEST, lambda: 10)
        for boundary in ("request_accept", "input_ready", "prompt_begin"): collector.mark(boundary)
        with self.assertRaisesRegex(ValueError, "Whole prompt"):
            collector.mark("prompt_complete")
        ticks = iter([10, 5])
        collector = c.PrefillCollector(job, tokens, DIGEST, lambda: next(ticks))
        collector.mark("request_accept")
        with self.assertRaisesRegex(ValueError, "backwards"): collector.mark("input_ready")

    def test_device_events_resolve_only_after_stream_completion(self):
        job, tokens, _, _ = job_for(chunk=2048)
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens, list(range(2048)))
        events = FakeEvents()
        for i in range(30):
            with collector.stage("attention", i, "site", DIGEST, (2048, 2048, 256), events): pass
        for i in range(30): collector.layer_complete(i, observation(0, 2048, i))
        collector.end_forward()
        result = collector.finish_events(events)
        self.assertEqual(events.calls, ["begin", "end"] * 30 + ["sync"] + ["resolve"] * 30)
        self.assertEqual(result[0]["device_elapsed_ns"], 17.0)
        self.assertFalse(result[0]["cupti_correlated"])
        self.assertEqual(result[0]["scope"], "inclusive_event_range")

    def test_head_only_after_prompt_complete_last_row_one(self):
        job, tokens, _, _ = job_for(chunk=2048, region="head")
        collector = c.PrefillCollector(job, tokens, DIGEST)
        finish_prompt(collector)
        events = FakeEvents()
        with collector.stage("head", 29, "site", DIGEST, (1, 262144, 2816), events): pass
        result = collector.finish_events(events)
        self.assertEqual(result[0]["forward_index"], 0)
        with self.assertRaises(ValueError):
            with collector.stage("head", 29, "site", DIGEST, (2048, 262144, 2816), events): pass


class CallableOwner:
    def forward(self, value):
        return value


class HookTests(unittest.TestCase):
    def make(self):
        job, tokens, _, _ = job_for(chunk=2048)
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens, list(range(2048)))
        fn = CallableOwner.forward
        bindings = {"site": {"module": fn.__module__, "qualname": fn.__qualname__,
            "file_sha256": hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest(), "site_sha256": DIGEST}}
        return collector, o.OwnedHooks(collector, FakeEvents(), bindings)

    def test_actual_instance_callable_retains_object_and_restores_class_dispatch(self):
        collector, hooks = self.make()
        owner, tensor = CallableOwner(), object()
        hooks.wrap(owner, "forward", "site", 0, "attention", lambda a, k: (2048, 2048, 256))
        self.assertIs(owner.forward(tensor), tensor)
        self.assertEqual(len(collector.events), 1)
        hooks.close()
        self.assertNotIn("forward", vars(owner))
        self.assertIs(owner.forward.__func__, CallableOwner.forward)

    def test_local_override_restore_and_foreign_hook_not_clobbered(self):
        _, hooks = self.make()
        owner = CallableOwner()
        saved = owner.forward
        owner.forward = saved
        hooks.wrap(owner, "forward", "site", 0, "attention", lambda a, k: (2048, 2048, 256))
        hooks.close()
        self.assertIs(owner.forward, saved)
        collector, hooks = self.make()
        owner = CallableOwner()
        hooks.wrap(owner, "forward", "site", 0, "attention", lambda a, k: (2048, 2048, 256))
        foreign = lambda x: x
        owner.forward = foreign
        with self.assertRaisesRegex(ValueError, "foreign replacement"): hooks.close()
        self.assertIs(owner.forward, foreign)
        self.assertTrue(collector.failed)

    def test_unknown_qualname_and_source_bytes_reject_before_hook(self):
        for key, value in (("qualname", "unknown"), ("file_sha256", "c" * 64)):
            collector, hooks = self.make()
            owner = CallableOwner()
            hooks.bindings["site"][key] = value
            with self.assertRaises(ValueError):
                hooks.wrap(owner, "forward", "site", 0, "attention", lambda a, k: (2048, 2048, 256))
            self.assertNotIn("forward", vars(owner))


class Gemma4DecoderLayer:
    pass

Gemma4DecoderLayer.__module__ = "vllm.model_executor.models.gemma4"


class NativeInterfaceTests(unittest.TestCase):
    def model(self):
        layers = []
        for i in range(30):
            layer = Gemma4DecoderLayer()
            layer.layer_idx = i
            d, h = (512, 2) if i in r.GLOBAL_LAYERS else (256, 8)
            qkv, out = CallableOwner(), CallableOwner()
            qkv.input_size, qkv.output_size = 2816, 16*d+2*h*d
            out.input_size, out.output_size = 16*d, 2816
            norms = [CallableOwner() for _ in range(3)]
            for n in norms: n._forward_method = n.forward
            layer.self_attn = SimpleNamespace(is_kv_shared_layer=False,is_sliding=i not in r.GLOBAL_LAYERS,
                head_dim=d,num_kv_heads=h,num_heads=16,attn=CallableOwner(),qkv_proj=qkv,o_proj=out,
                q_norm=norms[0],k_norm=norms[1],v_norm=norms[2],rotary_emb=CallableOwner())
            layer.mlp, layer.router, layer.moe = CallableOwner(), CallableOwner(), CallableOwner()
            layers.append(layer)
        return SimpleNamespace(named_modules=lambda: [(f"language_model.model.layers.{i}",x) for i,x in enumerate(layers)]), layers

    def sites(self):
        fn = CallableOwner.forward
        binding = {"module":fn.__module__,"qualname":fn.__qualname__,
            "file_sha256":hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest(),"site_sha256":DIGEST}
        return {name:{"binding":dict(binding),"component":component,"scope":"synthetic_native_interface",
            "attribute":"_forward_method" if name.endswith("norm") else "forward"} for name,component in
            {"qkv":"qkv","o_projection":"o_projection","shared_mlp":"shared_mlp","router":"router",
             "dispatch":"dispatch","attention":"attention","q_norm":"qk_norm","k_norm":"qk_norm","v_norm":"qk_norm"}.items()}

    def test_gemma_observer_wraps_actual_instances_and_projection_shapes(self):
        model,layers = self.model()
        job,tokens,_,_ = job_for(chunk=2048,region="dense")
        collector = c.PrefillCollector(job,tokens,DIGEST)
        collector.begin_forward(tokens,list(range(2048)))
        tensor = SimpleNamespace(shape=(2048,2816))
        with o.GemmaCallableObserver(model,collector,FakeEvents(),self.sites()):
            for layer in layers:
                self.assertIs(layer.self_attn.qkv_proj.forward(tensor),tensor)
                layer.self_attn.o_proj.forward(SimpleNamespace(shape=(2048,layer.self_attn.o_proj.input_size)))
                layer.mlp.forward(tensor)
        events=collector.events
        self.assertEqual(len(events),90)
        self.assertEqual(events[0]["shape"],[2048,8192,2816])
        self.assertEqual(events[15]["shape"],[2048,10240,2816])
        self.assertNotIn("forward",vars(layers[0].self_attn.qkv_proj))

    def test_gemma_registry_geometry_and_missing_layer_fail_before_hooks(self):
        for what in ("geometry","ordinal","missing"):
            model,layers = self.model()
            if what == "geometry": layers[5].self_attn.num_kv_heads=8
            if what == "ordinal": layers[2].layer_idx=3
            if what == "missing": model.named_modules=lambda: [(f"layers.{i}",x) for i,x in enumerate(layers[:-1])]
            with self.subTest(what=what), self.assertRaises(ValueError): o.bind_gemma_modules(model,self.sites())
            self.assertNotIn("forward",vars(layers[0].self_attn.qkv_proj))

    def test_cuda_event_interface_records_and_resolves_the_same_stream(self):
        calls=[]
        stream=SimpleNamespace(cuda_stream=37,synchronize=lambda:calls.append("stream_sync"))
        class Event:
            def __init__(self,enable_timing): calls.append(("event",enable_timing))
            def record(self,s): calls.append(("record",s.cuda_stream))
            def elapsed_time(self,end): calls.append("elapsed");return 0.25
        torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda:stream,Event=Event))
        clock=c.CudaEventClock(torch)
        ticket=clock.end(clock.begin())
        clock.synchronize()
        self.assertEqual(clock.resolve(ticket),(250000.0,"37"))
        self.assertEqual(calls[-2:],["stream_sync","elapsed"])
        ticket=clock.begin()
        torch.cuda.current_stream=lambda:SimpleNamespace(cuda_stream=38)
        with self.assertRaisesRegex(ValueError,"changed streams"): clock.end(ticket)

    def test_native_cache_prepare_checks_actual_tensors_and_two_row_hash_bound(self):
        import numpy as np
        job,tokens,_,_=job_for(chunk=2048)
        collector=c.PrefillCollector(job,tokens,DIGEST)
        collector.begin_forward(tokens,list(range(2048)))
        observer=object.__new__(kv.PrefillKVObserver)
        observer.collector,observer.writer=collector,kv.WriterLedger()
        observer.storage_identities=None
        observer.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        observer.layers={i:(None,None,SimpleNamespace(kv_cache="synthetic-cache"),None) for i in range(30)}
        observer.descriptors=[{"head_dim":512 if i in r.GLOBAL_LAYERS else 256} for i in range(30)]
        slots={i:list(range(2048)) for i in range(30)}
        caps={i:2304 for i in range(30)}
        ids={i:(i,i+1,2**20,0,(144,8,16,512),(65536,8192,512,1),"synthetic-cuda") for i in range(30)}
        observer.read_frame=lambda p,t:(slots,caps,ids,[])
        positions=SimpleNamespace(tolist=lambda:list(range(2048)))
        input_ids=SimpleNamespace(tolist=lambda:tokens)
        observer.prepare(positions,input_ids)
        batches=[]
        def rows(cache,selected,dim,torch):
            batches.append(len(selected))
            return np.zeros((len(selected),1,1),dtype=np.uint16),np.ones((len(selected),1,1),dtype=np.uint16)
        with patch("megartx.prefill_kv.gather_writer_rows",rows):
            observer.finish()
            collector.end_forward()
            result=observer.final_handoff(DIGEST)
        self.assertEqual(max(batches),2)
        self.assertEqual(result["committed_length"],2048)
        self.assertNotEqual(result["layers"][5]["processed_k_sha256"],result["layers"][5]["processed_v_sha256"])
        self.assertEqual(collector.handoff_record,result)


class TorchDTypeFixture:
    """CPU-only dtype identity with Torch's canonical string representation."""
    def __init__(self, name): self.name = name
    def __str__(self): return "torch." + self.name


class TensorFixture:
    def __init__(self, values, dtype):
        self.values, self.dtype = values, dtype
        self.ndim, self.shape = 1, (len(values),)
    def tolist(self): return self.values.copy()
    def detach(self): return self
    def cpu(self): return self


class NativeDTypeTests(unittest.TestCase):
    def frame(self):
        job, tokens, _, _ = job_for()
        collector = c.PrefillCollector(job, tokens, DIGEST)
        collector.begin_forward(tokens[:256], list(range(256)))
        int64, int32, boolean, bf16 = [TorchDTypeFixture(n) for n in ("int64", "int32", "bool", "bfloat16")]
        torch = SimpleNamespace(Tensor=TensorFixture, int64=int64, int32=int32, bool=boolean,
            bfloat16=bf16, equal=lambda a,b:a.tolist()==b.tolist())
        observer = object.__new__(kv.PrefillKVObserver)
        observer.collector, observer.writer, observer.torch = collector, kv.WriterLedger(), torch
        collector.attach_writer(observer.writer)
        observer.storage_identities = None
        observer.layers, observer.descriptors = {}, historical.nominal_layers()
        context = SimpleNamespace(slot_mapping={},attn_metadata={},no_compile_layers={},ubatch_slices=None,
            cudagraph_runtime_mode=SimpleNamespace(name="NONE"),is_padding=None)
        for i, d in enumerate(observer.descriptors):
            name = f"layer-{i}"
            mapping = TensorFixture(list(range(256)),int64)
            h, dim = d["kv_heads"], d["head_dim"]
            shape = (144,h,16,2*dim)
            strides = (h*16*2*dim,16*2*dim,2*dim,1)
            cache = TensorFixture([],bf16)
            cache.shape, cache.device = shape, SimpleNamespace(type="cuda")
            cache.stride=lambda s=strides:s
            cache.storage_offset=lambda:0
            cache.untyped_storage=lambda i=i,s=shape:SimpleNamespace(data_ptr=lambda:i+1,
                nbytes=lambda:s[0]*s[1]*s[2]*s[3]*2)
            impl=SimpleNamespace(num_kv_heads=h,head_size=dim,window_left=1023 if d["window_size"] else -1,
                cache_dtype="bfloat16",is_kvcache_nvfp4=False,kv_cache_layout=SimpleNamespace(name="BHNC"))
            attn=SimpleNamespace(impl=impl,kv_cache=cache,num_kv_heads=h,head_size=dim,head_size_v=dim,
                sliding_window=d["window_size"],kv_cache_dtype="bfloat16")
            observer.layers[i]=(name,SimpleNamespace(attn=attn),attn,impl)
            context.slot_mapping[name]=mapping
            context.no_compile_layers[name]=attn
            context.attn_metadata[name]=SimpleNamespace(num_actual_tokens=256,num_decodes=0,num_prefills=1,
                num_decode_tokens=0,num_prefill_tokens=256,use_cascade=False,causal=True,slot_mapping=mapping)
        module=ModuleType("vllm.forward_context")
        module.get_forward_context=lambda:context
        return observer, TensorFixture(list(range(256)),int64), TensorFixture(tokens[:256],int64), module, context

    def test_canonical_torch_int64_objects_reach_all_cache_checks(self):
        observer,positions,tokens,module,_=self.frame()
        self.assertEqual(str(positions.dtype),"torch.int64")
        with patch.dict(sys.modules,{"vllm.forward_context":module}), \
             patch.object(kv,"_digest",return_value=historical.SOURCE_HASHES["vllm.forward_context"]):
            observer.prepare(positions,tokens)
        self.assertEqual(set(observer.writer.pending["positions"]),set(range(30)))
        self.assertEqual(len(observer.native_bindings),30)
        self.assertFalse(observer.collector.failed)

    def test_int32_bool_string_alias_and_non_tensor_inputs_reject(self):
        for value in ("int32","bool","int64_alias","torch_int64_alias","non_tensor"):
            observer,positions,tokens,module,_=self.frame()
            if value=="non_tensor": positions=SimpleNamespace(dtype=observer.torch.int64,ndim=1,tolist=positions.tolist)
            elif value=="int64_alias": positions.dtype="int64"
            elif value=="torch_int64_alias": positions.dtype="torch.int64"
            else: tokens.dtype=getattr(observer.torch,value)
            with self.subTest(value=value), patch.dict(sys.modules,{"vllm.forward_context":module}), \
                 patch.object(kv,"_digest",return_value=historical.SOURCE_HASHES["vllm.forward_context"]), \
                 self.assertRaisesRegex(RuntimeError,"I64 row identity"):
                observer.prepare(positions,tokens)
            self.assertTrue(observer.collector.failed)
            self.assertTrue(observer.writer.poisoned)

    def test_non_tensor_writer_metadata_rejects(self):
        observer,positions,tokens,module,context=self.frame()
        mapping=context.slot_mapping["layer-0"]
        context.slot_mapping["layer-0"]=SimpleNamespace(dtype=observer.torch.int64,shape=mapping.shape)
        with patch.dict(sys.modules,{"vllm.forward_context":module}), \
             patch.object(kv,"_digest",return_value=historical.SOURCE_HASHES["vllm.forward_context"]), \
             self.assertRaisesRegex(RuntimeError,"writer slot map"):
            observer.prepare(positions,tokens)
        self.assertTrue(observer.writer.poisoned)

    def test_local_override_preserves_all_other_historical_frame_checks(self):
        old=ast.parse(textwrap.dedent(inspect.getsource(historical.ControlledKV.read_frame))).body[0]
        new=ast.parse(textwrap.dedent(inspect.getsource(kv.PrefillKVObserver.read_frame))).body[0]
        self.assertEqual([ast.unparse(d) for d in new.decorator_list],["_poison_on_failure"])
        new.decorator_list=[]
        new.body[0],new.body[1]=copy.deepcopy(old.body[0]),copy.deepcopy(old.body[1])
        # The only functional deltas are real Tensor/dtype checks. All source,
        # metadata, registry, shape/stride/extent, geometry and slot reads remain.
        old_dtype=next(n for n in old.body if isinstance(n,ast.If) and "positions.dtype" in ast.unparse(n.test))
        new_dtype=next(n for n in new.body if isinstance(n,ast.If) and "positions.dtype" in ast.unparse(n.test))
        expected=ast.parse("not isinstance(positions, torch.Tensor) or not isinstance(tokens, torch.Tensor) or positions.dtype != torch.int64 or tokens.dtype != torch.int64 or positions.ndim != 1 or tokens.ndim != 1",mode="eval").body
        self.assertEqual(ast.dump(new_dtype.test),ast.dump(expected))
        new_dtype.test=copy.deepcopy(old_dtype.test)
        mapping=next(n for n in ast.walk(new) if isinstance(n,ast.If) and "mapping.dtype" in ast.unparse(n.test))
        self.assertEqual(ast.unparse(mapping.test.values.pop(0)),"not isinstance(mapping, torch.Tensor)")
        self.assertEqual(ast.dump(new),ast.dump(old))


class FailureLifetimeTests(unittest.TestCase):
    def events(self):
        job,tokens,_,_=job_for(chunk=2048)
        collector=c.PrefillCollector(job,tokens,DIGEST)
        writer=kv.WriterLedger()
        collector.attach_writer(writer)
        collector.begin_forward(tokens,list(range(2048)))
        events=FakeEvents()
        for i in range(30):
            with collector.stage("attention",i,"site",DIGEST,(2048,2048,256),events): pass
        for i in range(30): collector.layer_complete(i,observation(0,2048,i))
        collector.end_forward()
        return collector,writer,events

    def test_hook_context_interruption_poisoned_and_owned_hook_restored(self):
        collector,hooks=HookTests().make()
        writer=kv.WriterLedger();collector.attach_writer(writer)
        owner=CallableOwner()
        with self.assertRaisesRegex(KeyboardInterrupt,"between callables"):
            with hooks:
                hooks.wrap(owner,"forward","site",0,"attention",lambda a,k:(2048,2048,256))
                owner.forward(object())
                raise KeyboardInterrupt("between callables")
        self.assertNotIn("forward",vars(owner))
        self.assertTrue(collector.failed and collector.cache.poisoned and writer.poisoned)
        with self.assertRaises(ValueError): collector.end_forward()

    def test_hook_cleanup_failure_retains_primary_and_restores_other_owned_hook(self):
        class FailingRestore(CallableOwner):
            def __delattr__(self,name):
                if name=="forward": raise RuntimeError("restoration secondary")
                super().__delattr__(name)
        collector,hooks=HookTests().make()
        writer=kv.WriterLedger();collector.attach_writer(writer)
        healthy,failing=CallableOwner(),FailingRestore()
        with self.assertRaisesRegex(KeyboardInterrupt,"primary interruption"):
            with hooks:
                for owner in (healthy,failing):
                    hooks.wrap(owner,"forward","site",0,"attention",lambda a,k:(2048,2048,256))
                raise KeyboardInterrupt("primary interruption")
        self.assertTrue(hooks.closed)
        self.assertNotIn("forward",vars(healthy))
        self.assertTrue(collector.failed and collector.cache.poisoned and writer.poisoned)

    def test_event_resolution_failures_poison_and_retry_never_resolves(self):
        for method,error in (("synchronize",KeyboardInterrupt("sync")),("resolve",RuntimeError("resolve"))):
            collector,writer,events=self.events()
            with self.subTest(method=method),patch.object(events,method,side_effect=error),self.assertRaises(type(error)):
                collector.finish_events(events)
            calls=events.calls.copy()
            self.assertTrue(collector.failed and collector.cache.poisoned and writer.poisoned)
            with self.assertRaisesRegex(ValueError,"poisoned"): collector.finish_events(events)
            self.assertEqual(events.calls,calls)

    def test_invalid_event_identity_and_evidence_bound_poison(self):
        for mode in ("identity","bound"):
            collector,writer,events=self.events()
            if mode=="identity": events.resolve=lambda ticket:(17.0,"")
            context=patch.object(c,"_bounded",side_effect=ValueError("bounded")) if mode=="bound" else nullcontext()
            with self.subTest(mode=mode),context:
                with self.assertRaises(ValueError): collector.finish_events(events)
            self.assertTrue(collector.failed and writer.poisoned)
            with self.assertRaises(ValueError): collector.finish_events(events)

    def test_final_handoff_hash_interrupt_poison_and_retry_rejects(self):
        job,tokens,_,_=job_for(chunk=2048)
        collector=c.PrefillCollector(job,tokens,DIGEST);finish_prompt(collector)
        observer=object.__new__(kv.PrefillKVObserver)
        observer.collector,observer.writer=collector,kv.WriterLedger()
        collector.attach_writer(observer.writer)
        observer.writer.end=2048
        with patch.object(observer,"_hash_positions",side_effect=KeyboardInterrupt("handoff hash")) as hashing:
            with self.assertRaisesRegex(KeyboardInterrupt,"handoff hash"): observer.final_handoff(DIGEST)
            with self.assertRaisesRegex(ValueError,"complete committed"): observer.final_handoff(DIGEST)
        self.assertEqual(hashing.call_count,1)
        self.assertTrue(collector.failed and collector.cache.poisoned and observer.writer.poisoned)
        self.assertIsNone(collector.handoff_record)

    def test_clock_interruption_after_layer_completion_burns_writer(self):
        job,tokens,_,_=job_for(chunk=2048)
        clock=lambda:1
        collector=c.PrefillCollector(job,tokens,DIGEST,clock)
        writer=kv.WriterLedger();collector.attach_writer(writer)
        collector.begin_forward(tokens,list(range(2048)))
        for i in range(30): collector.layer_complete(i,observation(0,2048,i))
        collector.clock=lambda:(_ for _ in ()).throw(KeyboardInterrupt("clock interrupted"))
        with self.assertRaises(KeyboardInterrupt): collector.end_forward()
        self.assertTrue(collector.failed and collector.cache.poisoned and writer.poisoned)
        with self.assertRaises(ValueError): collector.end_forward()


if __name__ == "__main__":
    unittest.main()

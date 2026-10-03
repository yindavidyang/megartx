"""Negative controls for request-bound dispatch qualification (CPU only)."""
import copy
import hashlib
from pathlib import Path
import json
import tempfile
import unittest

from compare_m1_live import correlate_trace, check_call, check_run, EXTENTS, load_build


SCOPES = ["controlled_live_request", "artificial_route_control", "artificial_route_control"] + ["controlled_live_request"]*29


def trace(lane):
    events = []
    for i in range(32):
        start = i*100
        common = {"ph": "X", "pid": 5, "tid": 7}
        events.extend([dict(common, cat="user_annotation", name="megartx::m1_routed_"+lane, ts=start, dur=30),
                       dict(common, cat="user_annotation", name="megartx::m1_preparation_"+lane, ts=start+1, dur=20)])
        kernels = ["MainloopSm120ArrayTmaWarpSpecializedBlockScaled_GEMM"]*2 + ["incumbent_tma", "incumbent_activation", "incumbent_finalize"]
        kernels += ["m1_maps_expand"] if lane == "fused" else ["fusedBuildExpertMapsSortFirstTokenKernel", "expandInputRowsKernel<fp4>"]
        for j,name in enumerate(kernels):
            correlation = i*100+j
            events.append(dict(common, cat="cuda_runtime", name="cudaLaunchKernel", ts=start+2+j, dur=.2, args={"correlation": correlation}))
            # GPU execution may occur much later than CPU enqueue.
            events.append({"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 19,
                           "ts": 100000+start+j, "dur": 1, "args": {"correlation": correlation, "stream": 19}})
        # GPU annotation copies do not count as request CPU dispatch spans.
        events.append({"ph": "X", "cat": "gpu_user_annotation", "name": "megartx::m1_preparation_"+lane,
                       "pid": 0, "tid": 19, "ts": 100000+start, "dur": 20})
    return events


class TestLiveComparison(unittest.TestCase):
    def test_unmocked_load_build_accepts_pr14_lifecycle_source_contract(self):
        repo = Path(__file__).resolve().parents[1]
        controller_sources = (
            "m1_live.py", "vllm_scale_plugin.py", "m1_execution.py",
            "controlled_capture.py", "controlled_kv_capture.py",
            "m1_normal_plan.py", "m1_normal_capture.py",
            "m1_external_observer.py", "m1_process_lifecycle.py")
        source_paths = ["probes/m1_live_bridge.cu"] + [
            "src/megartx/" + name for name in controller_sources]
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            source_hashes = {}
            for relative in source_paths:
                source = repo / relative
                target = build / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                source_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
            contract = {
                "abi_version": 2,
                "view_count": 15,
                "view_bytes": 32,
                "execution_modes": ["captured", "capture-free"],
                "capture_free_begin": "megartx_m1_begin_capture_free_v2",
                "external_observer": {
                    "registration": "megartx_m1_set_external_observer_v1",
                    "callback_abi_version": 1,
                },
                "native_source_sha256": source_hashes["probes/m1_live_bridge.cu"],
                "controller_source_hashes": {
                    name: source_hashes["src/megartx/" + name]
                    for name in controller_sources
                },
            }
            bridge = build / "m1_live_bridge.so"
            bridge.write_bytes(b"PR14-shaped source-bound bridge fixture")
            report = {
                "returncode": 0,
                "reason": None,
                "compiled_lease_controls_returncode": 0,
                "compiled_binding_controls_returncode": 0,
                "required_exports_present": True,
                "installed_pins_unchanged": True,
                "binary_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
                "source_hashes": source_hashes,
                "live_contract": contract,
            }
            (build / "build.json").write_text(json.dumps(report))
            declaration = json.dumps(json.dumps(contract, sort_keys=True))
            (build / "m1_live_symbols.h").write_text(
                "#define M1_LIVE_CONTRACT_JSON " + declaration + "\n")
            lease = {
                "execution_modes_tested": contract["execution_modes"],
                "cross_mode_nested_rejection_preserves_outer": True,
                "captured_null_directory_rejected": True,
                "external_observer_export": "megartx_m1_set_external_observer_v1",
                "external_observer_registration_guarded_by_live_leases": True,
                "external_observer_capture_free_begin_end_events": [
                    ["lease_begin", "fused", 0, 7, 0],
                    ["lease_end", "fused", 0, 7, -1]],
                "live_contract": contract,
                "invalid_framing_before_dereference": True,
                "historical_begin_symbol_absent": True,
                "active_after": 0,
            }
            binding = {
                "torch_runtime": "2.13.0+cu130",
                "relocations_bound_to_bridge": 4,
                "active_after": 0,
            }
            (build / "lease-controls.json").write_text(json.dumps(lease))
            (build / "binding-controls.json").write_text(json.dumps(binding))

            loaded = load_build(build)

            self.assertEqual(loaded["live_contract"], contract)
            self.assertIn("m1_process_lifecycle.py",
                          loaded["live_contract"]["controller_source_hashes"])

    def test_capture_free_launch_cannot_be_presented_as_correctness_evidence(self):
        for declaration in ({"m1_execution_requested": "capture-free"},
                            {"environment_overrides": {"MEGARTX_M1_EXECUTION": "capture-free"}}):
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/"launch-manifest.json").write_text(json.dumps(declaration))
                with self.assertRaisesRegex(ValueError, "Capture-free execution"):
                    check_run(root, "fused")
    def test_positive_correlated_dispatch_and_matching_incumbent_kernels(self):
        stock = correlate_trace(trace("stock"), "stock", SCOPES)
        fused = correlate_trace(trace("fused"), "fused", SCOPES)
        self.assertEqual(fused["positive_request_fused_launches"], 30)
        self.assertEqual(fused["positive_artificial_fused_launches"], 2)
        self.assertEqual(stock["positive_request_fused_launches"], 0)
        self.assertEqual(stock["incumbent_kernel_counts"], fused["incumbent_kernel_counts"])

    def test_global_fused_kernel_names_without_request_launch_correlation_are_zero_hits(self):
        events = trace("fused")
        for event in events:
            if event.get("cat") == "kernel" and event["name"] == "m1_maps_expand":
                event["args"]["correlation"] += 1000000
        with self.assertRaisesRegex(ValueError, "positive preparation|bounded call set"):
            correlate_trace(events, "fused", SCOPES)

    def test_missing_fused_launch_or_extra_incumbent_map_rejected(self):
        for mutation in ("missing", "extra_stock"):
            events = trace("fused")
            candidate = next(e for e in events if e.get("cat") == "kernel" and e["name"] == "m1_maps_expand")
            if mutation == "missing": events.remove(candidate)
            else:
                extra = copy.deepcopy(candidate);extra["name"] = "fusedBuildExpertMapsSortFirstTokenKernel";events.append(extra)
            with self.assertRaisesRegex(ValueError, "positive preparation|bounded call set"):
                correlate_trace(events, "fused", SCOPES)

    def test_ambiguous_correlations_and_wrong_stream_rejected(self):
        for mutation in ("correlation", "default_stream", "multiple_streams"):
            events = trace("fused")
            if mutation == "correlation":
                event = next(e for e in events if e.get("cat") == "cuda_runtime")
                events.append(copy.deepcopy(event))
            elif mutation == "default_stream":
                for e in events:
                    if e.get("cat") == "kernel": e["args"]["stream"] = 0
            else: next(e for e in events if e.get("cat") == "kernel")["args"]["stream"] = 23
            with self.assertRaises(ValueError): correlate_trace(events, "fused", SCOPES)

    def test_missing_request_parent_or_artificial_scope_rejected(self):
        events = [e for e in trace("fused") if e.get("name") != "megartx::m1_routed_fused"]
        with self.assertRaisesRegex(ValueError, "routed parent"): correlate_trace(events, "fused", SCOPES)
        with self.assertRaisesRegex(ValueError, "request spans"): correlate_trace(trace("fused"), "fused", ["controlled_live_request"]*32)

    def test_incomplete_or_failed_receipts_rejected_before_payload_read(self):
        record = {"schema": "megartx-m1-live-v2", "lane": "fused", "actual_backend": "fused", "execution_mode": "eager",
                  "pdl": False, "finalize_fusion": False, "producer_wait_inserted": True, "consumer_wait_inserted": True,
                  "lease_released": True, "routed_call_failed": False, "owner_extents": EXTENTS, "workspace_bytes": EXTENTS[5],
                  "forward_index": 1, "positions": [32], "tokens": [11], "live_contract": {"abi_version": 2, "view_count": 15, "view_bytes": 32}}
        for key,value in (("actual_backend", "stock"), ("pdl", True), ("lease_released", False),
                          ("consumer_wait_inserted", False), ("routed_call_failed", True), ("execution_mode", "graph"),
                          ("diagnostics_mode", "capture-free"),
                          ("owner_extents", EXTENTS[:7]), ("positions", [31]), ("live_contract", {"abi_version": 1})):
            bad = dict(record);bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): check_call(Path("/unused-evidence"), bad, "fused")


if __name__ == "__main__": unittest.main()

import json
import ctypes
import hashlib
from contextlib import contextmanager, nullcontext
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from megartx import m1_live
from megartx.m1_live import LivePreparation, load_controller, View, digest, CONTROLLER_SOURCES


class Tensor:
    is_cuda = True
    dtype = "u8"
    shape = (1, 1408)
    def __init__(self, address, size=1408, dtype="u8", shape=(1, 1408)):
        self.address, self.size, self.dtype, self.shape = address, size, dtype, shape
    def is_contiguous(self): return True
    def data_ptr(self): return self.address
    def numel(self): return self.size
    def element_size(self): return 1
    def untyped_storage(self): return self
    def nbytes(self): return self.size
    def record_stream(self, stream): pass


class Native:
    def __init__(self, reject=False):
        self.reject, self.active, self.ends = reject, False, 0
        self.begins = []
    def megartx_m1_begin_v2(self, *args):
        self.begins.append(("captured", args))
        if self.reject: return -1
        self.active = True;return 0
    def megartx_m1_begin_capture_free_v2(self, *args):
        self.begins.append(("capture-free", args))
        if self.reject: return -1
        self.active = True;return 0
    def megartx_m1_end(self):
        self.ends += 1;self.active = False;return 1
    def megartx_m1_active(self): return int(self.active)
    def megartx_m1_metadata(self): return b"source-bound test metadata"
    def megartx_m1_error(self): return b"nested native lease"


class TestLiveLease(unittest.TestCase):
    def test_default_does_not_construct_controller(self):
        with patch.dict("os.environ", {}, clear=True), patch("megartx.m1_live.LivePreparation") as constructor:
            self.assertIsNone(load_controller("native"))
            constructor.assert_not_called()

    def test_unqualified_modes_cannot_enable(self):
        for lane, mode in (("1", "native"), ("fused", "reference"), ("stock", "control")):
            with patch.dict("os.environ", {"MEGARTX_M1_PREPARATION": lane}, clear=True):
                with self.assertRaises(RuntimeError): load_controller(mode)

    def test_bounded_sha256_is_compatible_without_python311_file_digest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(hashlib, "file_digest", None, create=True):
            path = Path(directory)/"hash-input"
            for data in (b"", b"abc", bytes(range(256)) * 8192 + b"final partial block"):
                path.write_bytes(data)
                self.assertEqual(digest(path), hashlib.sha256(data).hexdigest())

    def admission_fixture(self, directory):
        library = Path(directory) / "bridge.so"
        library.write_bytes(b"CPU mock binary")
        module = Path(directory) / "fused_moe_120.so"
        module.write_bytes(b"installed mock")
        cupti = Path(directory) / "libcupti.so.13"
        cupti.write_bytes(b"CPU mock CUPTI provider")
        provider = {"distribution": "nvidia-cuda-cupti", "version": "13.0.85",
                    "library_name": cupti.name, "library_sha256": digest(cupti)}
        contract = {"abi_version": 2, "view_count": 15, "view_bytes": ctypes.sizeof(View),
                    "controller_source_hashes": {name: digest(Path(m1_live.__file__).with_name(name))
                                                 for name in CONTROLLER_SOURCES},
                    "native_source_sha256": "b" * 64,
                    "cupti_stream_id_provider": provider}
        contract.update(execution_modes=list(m1_live.EXECUTION_MODES), capture_free_begin=m1_live.CAPTURE_FREE_BEGIN)
        contract["external_observer"] = {"registration": m1_live.EXTERNAL_OBSERVER_SETTER,
                                          "callback_abi_version": 1}
        report = {"returncode": 0, "reason": None, "compiled_lease_controls_returncode": 0,
                  "compiled_binding_controls_returncode": 0, "required_exports_present": True,
                  "binary_sha256": digest(library),
                  "installed_pins": {str(module): "d" * 64, str(cupti): provider["library_sha256"]},
                  "installed_package_versions": {"nvidia-cuda-cupti": provider["version"]},
                  "cupti_stream_id_provider": provider,
                  "source_hashes": {"probes/m1_live_bridge.cu": "b" * 64}, "live_contract": contract}
        receipt = Path(directory) / "build.json"
        env = {"MEGARTX_M1_BRIDGE": str(library), "MEGARTX_M1_BUILD_RECEIPT": str(receipt),
               "MEGARTX_M1_STOCK_MODULE": str(module), "MEGARTX_M1_CAPTURE_DIR": str(Path(directory)/"unused")}
        return report, receipt, env

    def test_historical_receipt_and_changed_controller_rejected_before_loading(self):
        for change in ("historical", "controller", "version", "count", "struct"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                report, receipt, env = self.admission_fixture(directory)
                if change == "historical": del report["live_contract"]
                elif change == "controller": report["live_contract"]["controller_source_hashes"]["m1_live.py"] = "c"*64
                elif change == "version": report["live_contract"]["abi_version"] = 1
                elif change == "count": report["live_contract"]["view_count"] = 7
                else: report["live_contract"]["view_bytes"] = 16
                receipt.write_text(json.dumps(report))
                with patch.dict("os.environ", env, clear=True), patch.dict("sys.modules", {"torch": SimpleNamespace()}), \
                     patch("ctypes.CDLL") as loader, self.assertRaisesRegex(RuntimeError, "source contract"):
                    LivePreparation("fused")
                loader.assert_not_called()
                self.assertFalse((Path(directory)/"unused").exists())

    def test_historical_native_and_forged_contract_cannot_enter_lease(self):
        class Function:
            def __init__(self, value): self.value, self.calls = value, 0
            def __call__(self, *args): self.calls += 1;return self.value
        for change in ("historical_exports", "version", "count", "controller", "native_source"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                report, receipt, env = self.admission_fixture(directory)
                module = env["MEGARTX_M1_STOCK_MODULE"]
                pinned = "dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9"
                report["installed_pins"][module] = pinned
                receipt.write_text(json.dumps(report))
                compiled = json.loads(json.dumps(report["live_contract"]))
                if change == "version": compiled["abi_version"] = 1
                elif change == "count": compiled["view_count"] = 7
                elif change == "controller": compiled["controller_source_hashes"]["m1_live.py"] = "c"*64
                elif change == "native_source": compiled["native_source_sha256"] = "c"*64
                begin = Function(0)
                native = SimpleNamespace(megartx_m1_begin=begin) if change == "historical_exports" else SimpleNamespace(
                    megartx_m1_contract_v2=Function(json.dumps(compiled).encode()), megartx_m1_begin_v2=begin)
                def fake_digest(path): return pinned if str(path) == module else digest(path)
                with patch.dict("os.environ", env, clear=True), patch.dict("sys.modules", {"torch": SimpleNamespace()}), \
                     patch("megartx.m1_live.digest", side_effect=fake_digest), patch("ctypes.CDLL", return_value=native), \
                     self.assertRaisesRegex(RuntimeError, "versioned ABI|differs from controller"):
                    LivePreparation("fused")
                self.assertEqual(begin.calls, 0)
                self.assertFalse((Path(directory)/"unused").exists())

    def controller(self, directory, native):
        obj = LivePreparation.__new__(LivePreparation)
        obj.native, obj.lane, obj.directory = native, "fused", Path(directory)
        obj.failed = False
        obj.execution_mode, obj.diagnostics = "captured", True
        obj.forward = {"request": {"id": "bounded"}, "forward_index": 0, "tokens": [11], "positions": [32]}
        obj.layer = SimpleNamespace(layer_name="model.layers.0")
        obj.workspace = Tensor(0x50000, 3185408, shape=(3185408,))
        obj.shadow = Tensor(0x500000, 32, shape=(32,))
        obj.stream = SimpleNamespace(cuda_stream=7)
        obj.binary_sha256, obj.call_index = "a" * 64, 0
        obj.live_contract = {"abi_version": 2}
        obj.call_limit, obj.external_observer = 64, None
        obj.normal_plan = None
        obj.route_controls, obj.route_controls_done = False, False
        return obj

    def kwargs(self):
        return {"input": Tensor(0x10000), "input_sf": Tensor(0x20000, 128*176),
                "token_selected_experts": Tensor(0x30000, 8, "i32", (1, 8)),
                "token_final_scales": Tensor(0x40000, 8, "f32", (1, 8)),
                "output": Tensor(0x800000, 5632),
                "fc1_expert_weights": Tensor(0x900000), "fc2_expert_weights": Tensor(0xa00000),
                "quant_scales": [Tensor(0xb00000+i*0x10000, 128, "i32" if i in (1, 4) else "f32", (128,))
                                 for i in range(6)]}

    def modules(self):
        return {"torch": SimpleNamespace(Tensor=Tensor, uint8="u8", int32="i32", float32="f32",
                                          profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
                                          cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)),
                "flashinfer.fused_moe": SimpleNamespace(cutlass_fused_moe_workspace_size=lambda *a, **kw: 3185408)}

    def test_submission_exception_releases_lease_without_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", self.modules()):
            native = Native();obj = self.controller(directory, native)
            calls = []
            error = RuntimeError("candidate submission failed")
            def incumbent(**kwargs):
                calls.append(kwargs);raise error
            with self.assertRaises(RuntimeError) as raised:
                obj.invoke(incumbent, (), self.kwargs())
            self.assertIs(raised.exception, error)
            self.assertEqual(len(calls), 1)
            self.assertEqual(native.ends, 1)
            self.assertFalse(native.active)
            receipt = json.loads((Path(directory)/"call-0000/receipt.json").read_text())
            self.assertTrue(receipt["lease_released"])
            self.assertEqual(receipt["error"], "RuntimeError")

    def test_nested_begin_rejection_does_not_end_outer_lease(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", self.modules()):
            native = Native(reject=True);native.active = True
            obj = self.controller(directory, native);calls = []
            with self.assertRaisesRegex(RuntimeError, "nested native"):
                obj.invoke(lambda **kw: calls.append(kw), (), self.kwargs())
            self.assertEqual(calls, [])
            self.assertEqual(native.ends, 0)
            self.assertTrue(native.active)

    def test_outside_request_uses_complete_original_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", self.modules()):
            native = Native();obj = self.controller(directory, native);obj.forward = None
            args, kwargs = ("unsupported",), {"unknown": "preserve"}
            calls = []
            def incumbent(*a, **kw): calls.append((a, kw));return 9
            self.assertEqual(obj.invoke(incumbent, args, kwargs), 9)
            self.assertEqual(calls, [(args, kwargs)])
            self.assertEqual(native.ends, 0)

    def test_nested_routed_context_fails_before_incumbent(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", self.modules()):
            obj = self.controller(directory, Native());calls = []
            with self.assertRaisesRegex(RuntimeError, "nested corrected"):
                obj.routed(lambda *a: calls.append(a), object())
            self.assertEqual(calls, [])

    def test_forward_cleanup_releases_detected_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native();native.active = True;obj = self.controller(directory, native)
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                obj.end_forward()
            self.assertIsNone(obj.forward)
            self.assertIsNone(obj.layer)
            self.assertFalse(native.active)

    def routed_modules(self, events, fail_wait=False):
        @contextmanager
        def enter(stream):
            events.append("enter bridge")
            try: yield
            finally: events.append("exit bridge")
        def caller_wait(stream):
            events.append("caller wait bridge")
            if fail_wait: raise RuntimeError("caller dependency failed")
        producer = SimpleNamespace(wait_stream=caller_wait)
        owned = SimpleNamespace(cuda_stream=7, wait_stream=lambda stream: events.append("bridge wait caller"))
        modules = self.modules()
        modules["torch"].cuda.current_stream = lambda: producer
        modules["torch"].cuda.stream = enter
        return modules, owned

    def test_late_submission_and_capture_errors_insert_caller_dependency(self):
        for phase in ("candidate submission", "preparation capture", "routed output capture"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                events = [];modules, stream = self.routed_modules(events)
                native = Native();obj = self.controller(directory, native);obj.layer = None;obj.stream = stream
                error = RuntimeError(phase + " failed")
                def native_call(**kwargs):
                    events.append("queued native work");raise error
                def routed(layer, *args): return obj.invoke(native_call, (), self.kwargs())
                with patch.dict("sys.modules", modules), self.assertRaises(RuntimeError) as raised:
                    obj.routed(routed, SimpleNamespace(layer_name="model.layers.0"))
                self.assertIs(raised.exception, error)
                self.assertEqual(events, ["bridge wait caller", "enter bridge", "queued native work",
                                          "exit bridge", "caller wait bridge"])
                self.assertFalse(native.active);self.assertEqual(native.ends, 1)
                self.assertIsNone(obj.layer);self.assertTrue(obj.failed)
                receipt = json.loads((Path(directory)/"call-0000/receipt.json").read_text())
                self.assertTrue(receipt["consumer_wait_inserted"])
                self.assertTrue(receipt["routed_call_failed"])

    def test_dependency_failure_preserves_primary_and_blocks_context_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            events = [];modules, stream = self.routed_modules(events, fail_wait=True)
            native = Native();obj = self.controller(directory, native);obj.layer = None;obj.stream = stream
            error = RuntimeError("candidate submission failed")
            def native_call(**kwargs): events.append("queued native work");raise error
            with patch.dict("sys.modules", modules), self.assertRaises(RuntimeError) as raised:
                obj.routed(lambda layer: obj.invoke(native_call, (), self.kwargs()),
                           SimpleNamespace(layer_name="model.layers.0"))
            self.assertIs(raised.exception, error)
            self.assertEqual(events.count("queued native work"), 1)
            self.assertEqual(events.count("caller wait bridge"), 1)
            self.assertFalse(native.active);self.assertEqual(native.ends, 1)
            self.assertTrue(obj.failed)
            with self.assertRaisesRegex(RuntimeError, "process restart"):
                obj.begin_forward(None, None)
            receipt = json.loads((Path(directory)/"call-0000/receipt.json").read_text())
            self.assertFalse(receipt["consumer_wait_inserted"])
            self.assertEqual(receipt["consumer_wait_error"], "RuntimeError")

    def test_receipt_failure_preserves_primary_after_native_release(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", self.modules()):
            native = Native();obj = self.controller(directory, native);calls = []
            error = RuntimeError("preparation capture failed")
            def native_call(**kwargs): calls.append(1);raise error
            with patch.object(Path, "write_text", side_effect=OSError("receipt storage failed")), self.assertRaises(RuntimeError) as raised:
                obj.invoke(native_call, (), self.kwargs())
            self.assertIs(raised.exception, error)
            self.assertEqual(calls, [1]);self.assertEqual(native.ends, 1)
            self.assertFalse(native.active);self.assertTrue(obj.failed)

    def capture_free(self, obj):
        obj.execution_mode, obj.diagnostics, obj.directory = "capture-free", False, None
        return obj

    def test_modes_preserve_operator_arguments_and_exact_owner_views(self):
        for lane in ("stock", "fused"):
            invocations = []
            for diagnostics in (True, False):
                with self.subTest(lane=lane, diagnostics=diagnostics), tempfile.TemporaryDirectory() as directory:
                    modules = self.modules()
                    native = Native();obj = self.controller(directory, native);obj.lane = lane
                    if not diagnostics:
                        self.capture_free(obj)
                        modules["torch"].profiler.record_function = lambda *a: self.fail("disabled profiler hook")
                        native.megartx_m1_metadata = lambda: self.fail("disabled metadata read")
                    kwargs = self.kwargs();output = object();calls = []
                    def operator(**kw): calls.append(kw);return output
                    with patch.dict("sys.modules", modules):
                        if diagnostics:
                            self.assertIs(obj.invoke(operator, (), kwargs), output)
                        else:
                            with patch.object(Path, "mkdir", side_effect=AssertionError("diagnostic mkdir")), \
                                 patch.object(Path, "open", side_effect=AssertionError("diagnostic file I/O")), \
                                 patch.object(Path, "write_text", side_effect=AssertionError("diagnostic receipt")):
                                self.assertIs(obj.invoke(operator, (), kwargs), output)
                    self.assertEqual(len(calls), 1)
                    for key, value in kwargs.items(): self.assertIs(calls[0][key], value)
                    self.assertIs(calls[0]["workspace_buffer"], obj.workspace)
                    self.assertIs(calls[0]["enable_pdl"], False)
                    self.assertIs(calls[0]["use_fused_finalize"], False)
                    mode, begin = native.begins[0]
                    self.assertEqual(mode, "captured" if diagnostics else "capture-free")
                    self.assertEqual((begin[0], begin[2], begin[3], begin[4], begin[5]), (2, 15, 32, 7, int(lane == "fused")))
                    self.assertEqual(len(begin), 7 if diagnostics else 6)
                    pointers = [view.pointer for view in begin[1]]
                    tensors = (kwargs["input"], kwargs["input_sf"], kwargs["token_selected_experts"],
                               kwargs["token_final_scales"], kwargs["output"], obj.workspace, obj.shadow,
                               kwargs["fc1_expert_weights"], kwargs["fc2_expert_weights"], *kwargs["quant_scales"])
                    self.assertEqual(pointers, [tensor.data_ptr() for tensor in tensors])
                    invocations.append(pointers)
                    self.assertEqual(native.ends, 1);self.assertFalse(native.active)
                    self.assertEqual(bool(list(Path(directory).iterdir())), diagnostics)
            self.assertEqual(invocations[0], invocations[1])

    def test_capture_free_submission_failure_keeps_waits_and_primary_error(self):
        for fail_wait in (False, True):
            with self.subTest(fail_wait=fail_wait), tempfile.TemporaryDirectory() as directory:
                events = [];modules, stream = self.routed_modules(events, fail_wait=fail_wait)
                modules["torch"].profiler.record_function = lambda *a: self.fail("disabled profiler hook")
                native = Native();obj = self.capture_free(self.controller(directory, native))
                obj.layer = None;obj.stream = stream
                error = RuntimeError("candidate submission failed")
                def operator(**kw): events.append("queued native work");raise error
                with patch.dict("sys.modules", modules), \
                     patch.object(Path, "open", side_effect=AssertionError("disabled diagnostic I/O")), \
                     self.assertRaises(RuntimeError) as raised:
                    obj.routed(lambda layer: obj.invoke(operator, (), self.kwargs()),
                               SimpleNamespace(layer_name="model.layers.0"))
                self.assertIs(raised.exception, error)
                self.assertEqual(events, ["bridge wait caller", "enter bridge", "queued native work",
                                          "exit bridge", "caller wait bridge"])
                self.assertFalse(native.active);self.assertEqual(native.ends, 1)
                self.assertIsNone(obj.layer);self.assertTrue(obj.failed)
                with patch.dict("sys.modules", modules), self.assertRaisesRegex(RuntimeError, "process restart"):
                    obj.invoke(lambda **kw: self.fail("failed context reused"), (), self.kwargs())

    def test_capture_free_unsupported_geometry_and_capture_keep_full_stock_call(self):
        for capturing in (False, True):
            with self.subTest(capturing=capturing), tempfile.TemporaryDirectory() as directory:
                modules = self.modules();modules["torch"].cuda.is_current_stream_capturing = lambda: capturing
                native = Native();obj = self.capture_free(self.controller(directory, native))
                kwargs = self.kwargs()
                if not capturing: kwargs["input"].shape = (32, 1408)
                calls = []
                with patch.dict("sys.modules", modules), patch.object(Path, "open", side_effect=AssertionError("fallback trace")):
                    result = obj.invoke(lambda **kw: calls.append(kw) or 9, (), kwargs)
                self.assertEqual(result, 9);self.assertEqual(calls, [kwargs])
                self.assertEqual(native.begins, []);self.assertEqual(native.ends, 0)

    def test_capture_free_unbound_or_unreleased_runner_is_fatal_without_diagnostics(self):
        for unbound in (False, True):
            with self.subTest(unbound=unbound), tempfile.TemporaryDirectory() as directory:
                native = Native();obj = self.capture_free(self.controller(directory, native))
                if unbound:
                    def end(): native.active = False;return -1
                else:
                    def end(): return 1
                native.megartx_m1_end = end
                with patch.dict("sys.modules", self.modules()), self.assertRaisesRegex(RuntimeError, "bind/release"):
                    obj.invoke(lambda **kw: 9, (), self.kwargs())
                self.assertTrue(obj.failed)

    def test_every_execution_source_is_bound_before_loading_in_both_modes(self):
        for execution in ("captured", "capture-free"):
            for source in CONTROLLER_SOURCES:
                with self.subTest(execution=execution, source=source), tempfile.TemporaryDirectory() as directory:
                    report, receipt, env = self.admission_fixture(directory)
                    report["live_contract"]["controller_source_hashes"][source] = "c" * 64
                    receipt.write_text(json.dumps(report))
                    if execution == "capture-free":
                        env.pop("MEGARTX_M1_CAPTURE_DIR")
                        env.update(MEGARTX_M1_EXECUTION=execution, MEGARTX_M1_PREPARATION="fused",
                                   MEGARTX_SCALE_MODE="native", MEGARTX_CONTROLLED_DIR="controlled",
                                   MEGARTX_CONTROLLED_PLAN="plan", MEGARTX_LOGITS_DIR="logits")
                    with patch.dict("os.environ", env, clear=True), patch.dict("sys.modules", {"torch": SimpleNamespace()}), \
                         patch("ctypes.CDLL") as loader, self.assertRaisesRegex(RuntimeError, "source contract"):
                        LivePreparation("fused")
                    loader.assert_not_called()

    def test_compiled_mode_capability_is_required_before_binding_and_allocation(self):
        class Function:
            def __init__(self, value): self.value, self.calls = value, []
            def __call__(self, *args): self.calls.append(args);return self.value
        for execution in ("captured", "capture-free"):
            for missing in (False, True):
                with self.subTest(execution=execution, missing=missing), tempfile.TemporaryDirectory() as directory:
                    report, receipt, env = self.admission_fixture(directory)
                    module = env["MEGARTX_M1_STOCK_MODULE"]
                    pinned = "dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9"
                    report["installed_pins"][module] = pinned
                    receipt.write_text(json.dumps(report))
                    env.update(MEGARTX_M1_EXECUTION=execution, MEGARTX_M1_PREPARATION="stock",
                               MEGARTX_SCALE_MODE="native", MEGARTX_CONTROLLED_DIR="controlled",
                               MEGARTX_CONTROLLED_PLAN="plan", MEGARTX_LOGITS_DIR="logits")
                    if execution == "capture-free": env.pop("MEGARTX_M1_CAPTURE_DIR")
                    native = SimpleNamespace(**{name: Function(value) for name, value in {
                        "megartx_m1_contract_v2": json.dumps(report["live_contract"]).encode(),
                        "megartx_m1_begin_v2": 0, "megartx_m1_begin_capture_free_v2": 0,
                        "megartx_m1_end": 1, "megartx_m1_error": b"", "megartx_m1_metadata": b"",
                        "megartx_m1_active": 0, "megartx_m1_verify_bindings": 0}.items()})
                    if missing: delattr(native, "megartx_m1_begin_capture_free_v2")
                    def fake_digest(path): return pinned if str(path) == module else digest(path)
                    with patch.dict("os.environ", env, clear=True), patch.dict("sys.modules", {"torch": SimpleNamespace()}), \
                         patch("megartx.m1_live.digest", side_effect=fake_digest), patch("ctypes.CDLL", return_value=native):
                        if missing:
                            with self.assertRaisesRegex(RuntimeError, "versioned ABI"): LivePreparation("stock")
                            self.assertEqual(native.megartx_m1_verify_bindings.calls, [])
                        else:
                            obj = LivePreparation("stock")
                            self.assertEqual(obj.execution_mode, execution)
                            self.assertEqual(obj.diagnostics, execution == "captured")
                            self.assertIsNone(obj.workspace);self.assertIsNone(obj.stream)
                            self.assertEqual(len(native.megartx_m1_verify_bindings.calls), 1)
                    self.assertEqual(native.megartx_m1_begin_v2.calls, [])
                    self.assertFalse((Path(directory)/"unused").exists())
    def test_failed_controller_blocks_direct_routed_and_invoke_reentry(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules",self.modules()):
            obj = self.controller(directory,Native());obj.failed = True
            calls = []
            for action in (lambda: obj.invoke(lambda **kw: calls.append(kw),(),self.kwargs()),
                           lambda: obj.routed(lambda *a: calls.append(a),object())):
                with self.assertRaisesRegex(RuntimeError,"process restart"): action()
            self.assertEqual(calls,[])
            self.assertEqual(obj.call_index,0)
            self.assertEqual(obj.native.ends,0)

    def test_producer_wait_failure_clears_layer_without_submission_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            events = [];modules,stream = self.routed_modules(events)
            error = RuntimeError("real producer stream query failed")
            def failed_wait(other): raise error
            stream.wait_stream = failed_wait
            obj = self.controller(directory,Native());obj.layer = None;obj.stream = stream
            calls = []
            with patch.dict("sys.modules",modules),self.assertRaises(RuntimeError) as raised:
                obj.routed(lambda *a: calls.append(a),object())
            self.assertIs(raised.exception,error)
            self.assertIsNone(obj.layer);self.assertTrue(obj.failed)
            self.assertEqual(calls,[])
            self.assertEqual(events,["caller wait bridge"])

    def test_total_call_cap_is_checked_before_receipt_directory_or_submission(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict("sys.modules",self.modules()):
            obj = self.controller(directory,Native());obj.call_limit = 210;obj.call_index = 210
            with self.assertRaisesRegex(RuntimeError,"call count"): obj.invoke(lambda **kw: None,(),self.kwargs())
            self.assertFalse((Path(directory)/"call-0210").exists())
            self.assertEqual(obj.native.ends,0);self.assertTrue(obj.failed)


if __name__ == "__main__": unittest.main()

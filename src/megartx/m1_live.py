"""Default-off, pinned, request-bound live preparation bridge.

The Python layer retains every owner and establishes both stream dependencies.
Native scope is entered only inside the verified corrected routed adapter.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

from .m1_execution import (CAPTURE_FREE_BEGIN, CONTROLLER_SOURCES,
                           EXECUTION_MODES, execution_mode, profile_scope)


LIVE_ABI_VERSION = 2
LIVE_VIEW_COUNT = 15


class View(ctypes.Structure):
    _fields_ = [("pointer", ctypes.c_void_p), ("bytes", ctypes.c_uint64),
                ("storage", ctypes.c_void_p), ("storage_bytes", ctypes.c_uint64)]


def tensor_view(tensor):
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise RuntimeError("live preparation requires contiguous CUDA views")
    storage = tensor.untyped_storage()
    return View(tensor.data_ptr(), tensor.numel() * tensor.element_size(),
                storage.data_ptr(), storage.nbytes())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_controller(mode):
    execution_mode()  # Reject contradictory diagnostics even without a live opt-in.
    lane = os.environ.get("MEGARTX_M1_PREPARATION")
    if lane is None:
        return None
    if lane not in {"stock", "fused"} or mode != "native":
        raise RuntimeError("live preparation requires explicit stock/fused and native correction")
    return LivePreparation(lane)


class LivePreparation:
    def __init__(self, lane):
        import torch
        self.lane = lane
        self.execution_mode = execution_mode()
        self.diagnostics = self.execution_mode == "captured"
        library = Path(os.environ["MEGARTX_M1_BRIDGE"]).resolve()
        receipt = json.loads(Path(os.environ["MEGARTX_M1_BUILD_RECEIPT"]).read_text())
        if (receipt.get("returncode") != 0 or receipt.get("reason")
                or receipt.get("compiled_lease_controls_returncode") != 0
                or receipt.get("compiled_binding_controls_returncode") != 0
                or not receipt.get("required_exports_present")
                or digest(library) != receipt.get("binary_sha256")):
            raise RuntimeError("live bridge has no matching successful bounded build")
        expected_contract = {
            "abi_version": LIVE_ABI_VERSION, "view_count": LIVE_VIEW_COUNT,
            "view_bytes": ctypes.sizeof(View),
            "controller_source_hashes": {name: digest(Path(__file__).with_name(name))
                                         for name in CONTROLLER_SOURCES},
            "native_source_sha256": receipt.get("source_hashes", {}).get("probes/m1_live_bridge.cu"),
            "execution_modes": list(EXECUTION_MODES), "capture_free_begin": CAPTURE_FREE_BEGIN,
        }
        if (receipt.get("live_contract") != expected_contract
                or not expected_contract["native_source_sha256"]):
            raise RuntimeError("live bridge ABI/controller/native source contract mismatch")
        if any(digest(p) != value for p, value in receipt["installed_pins"].items()):
            raise RuntimeError("installed live preparation identities changed")
        modules = [Path(p).resolve() for p, value in receipt["installed_pins"].items()
                   if Path(p).name == "fused_moe_120.so" and
                   value == "dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9"]
        if len(modules) != 1 or Path(os.environ["MEGARTX_M1_STOCK_MODULE"]).resolve() != modules[0]:
            raise RuntimeError("live bridge must bind the exact validated incumbent module")
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("live bridge requires ordinary Torch initialization without preload")
        self.native = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        try:
            contract = self.native.megartx_m1_contract_v2
            contract.argtypes, contract.restype = [], ctypes.c_char_p
            if json.loads(contract().decode()) != expected_contract:
                raise RuntimeError("compiled live bridge contract differs from controller/receipt")
            begin = self.native.megartx_m1_begin_v2
            capture_free_begin = getattr(self.native, CAPTURE_FREE_BEGIN)
        except (AttributeError, ValueError, UnicodeError) as error:
            raise RuntimeError("live bridge lacks the required versioned ABI") from error
        begin.argtypes = [ctypes.c_uint32, ctypes.POINTER(View), ctypes.c_uint32,
                          ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p]
        begin.restype = ctypes.c_int
        capture_free_begin.argtypes = begin.argtypes[:-1]
        capture_free_begin.restype = ctypes.c_int
        self.native.megartx_m1_end.restype = ctypes.c_int
        self.native.megartx_m1_error.restype = ctypes.c_char_p
        self.native.megartx_m1_metadata.restype = ctypes.c_char_p
        self.native.megartx_m1_active.restype = ctypes.c_int
        self.native.megartx_m1_verify_bindings.argtypes = [ctypes.c_char_p]
        self.native.megartx_m1_verify_bindings.restype = ctypes.c_int
        if self.native.megartx_m1_verify_bindings(os.fsencode(library)):
            raise RuntimeError(self.native.megartx_m1_error().decode())
        self.directory = Path(os.environ["MEGARTX_M1_CAPTURE_DIR"]) if self.diagnostics else None
        # Plugin registration also runs in the API process. Allocate/capture
        # only inside the verified model's first marked request.
        self.stream = None
        self.workspace = None
        self.shadow = None
        self.forward = None
        self.layer = None
        self.failed = False
        self.forward_index = self.call_index = 0
        self.binary_sha256 = receipt["binary_sha256"]
        self.live_contract = expected_contract
        self.route_controls = os.environ.get("MEGARTX_M1_ROUTE_CONTROLS") == "1"
        self.route_controls_done = False

    def begin_forward(self, input_ids, positions):
        if self.failed:
            raise RuntimeError("failed live preparation context requires an owned process restart")
        if self.forward is not None:
            raise RuntimeError("nested model preparation context")
        marker = Path(os.environ["MEGARTX_LOGITS_DIR"]).parent / "capture-request.json"
        if not marker.exists():
            return
        if input_ids is None or positions.ndim != 1 or input_ids.numel() != positions.numel():
            raise RuntimeError("live preparation needs actual model token/position rows")
        request = json.loads(marker.read_text())
        self.forward = {"request": request, "forward_index": self.forward_index,
                        "tokens": input_ids.detach().cpu().tolist(),
                        "positions": positions.detach().cpu().tolist()}
        self.forward_index += 1

    def end_forward(self):
        self.forward = None
        self.layer = None
        if self.native.megartx_m1_active():
            self.native.megartx_m1_end()
            raise RuntimeError("native lease escaped the routed call")

    def routed(self, incumbent, layer, *args):
        import torch
        if self.forward is None:
            return incumbent(layer, *args)
        if self.layer is not None:
            raise RuntimeError("nested corrected routed preparation context")
        # binding(layer, routed_adapter) has just checked the actual registry,
        # owner, method, kernel and callable. Keep these exact objects alive.
        self.layer = layer
        first_call = self.call_index
        if self.stream is None:
            if self.diagnostics:
                self.directory.mkdir(parents=True, exist_ok=False)
            self.stream = torch.cuda.Stream()
            self.shadow = torch.empty(32, dtype=torch.uint8, device="cuda")
        producer = torch.cuda.current_stream()
        self.stream.wait_stream(producer)
        for tensor in args:
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(self.stream)
        primary_error = None
        dependency_inserted = False
        try:
            with torch.cuda.stream(self.stream), profile_scope("megartx::m1_routed_" + self.lane, self.diagnostics):
                result = incumbent(layer, *args)
        except BaseException as error:
            primary_error = error
            self.failed = True
            raise
        finally:
            self.layer = None
            # record_stream protects allocator lifetime; the caller wait also
            # protects mutation/reuse after a late submission/capture error.
            cleanup_error = None
            try:
                producer.wait_stream(self.stream)
                dependency_inserted = True
            except BaseException as error:
                cleanup_error = error
                self.failed = True
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note("Caller stream dependency failed: " + str(error))
            try:
                for index in range(first_call, self.call_index) if self.diagnostics else ():
                    path = self.directory / f"call-{index:04d}" / "receipt.json"
                    if not path.exists():
                        continue
                    record = json.loads(path.read_text())
                    record["consumer_wait_inserted"] = dependency_inserted
                    record["routed_call_failed"] = primary_error is not None or cleanup_error is not None
                    if cleanup_error is not None:
                        record["consumer_wait_error"] = type(cleanup_error).__name__
                    path.write_text(json.dumps(record, indent=2))
            except BaseException as error:
                self.failed = True
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note("Dependency receipt write failed: " + str(error))
        def retain(value):
            if isinstance(value, torch.Tensor):
                value.record_stream(producer)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    retain(child)
        retain(result)
        return result

    def invoke(self, incumbent, args, kwargs):
        import torch
        if self.failed:
            raise RuntimeError("failed live preparation context requires an owned process restart")
        if self.layer is None or self.forward is None or args:
            return incumbent(*args, **kwargs)
        x = kwargs.get("input")
        if x is None or x.shape[0] != 1 or torch.cuda.is_current_stream_capturing():
            if self.diagnostics:
                with (self.directory / "stock-fallbacks.jsonl").open("a") as stream:
                    stream.write(json.dumps({"layer_name": self.layer.layer_name,
                                            "forward_index": self.forward["forward_index"],
                                            "request_id": self.forward["request"]["id"],
                                            "rows": None if x is None else x.shape[0],
                                            "reason": "unsupported_geometry_or_capture"}) + "\n")
            return incumbent(**kwargs)
        # Unsupported Python states keep the entire ordinary call untouched.
        sf = kwargs.get("input_sf")
        ids, weights = kwargs.get("token_selected_experts"), kwargs.get("token_final_scales")
        if (x.dtype != torch.uint8 or tuple(x.shape) != (1, 1408) or sf is None
                or sf.numel() != 128 * 176 or ids is None or weights is None
                or ids.dtype != torch.int32
                or tuple(ids.shape) != (1, 8) or weights.dtype != torch.float32
                or tuple(weights.shape) != (1, 8) or kwargs.get("enable_alltoall", False)
                or kwargs.get("tp_size", 1) != 1 or kwargs.get("ep_size", 1) != 1
                or kwargs.get("use_w4_group_scaling", False)
                or kwargs.get("use_deepseek_fp8_block_scale", False)
                or kwargs.get("use_mxfp8_act_scaling", False)
                or kwargs.get("enable_pdl") not in (None, False)):
            return incumbent(**kwargs)
        quant = kwargs.get("quant_scales")
        if (not isinstance(quant, (tuple, list)) or len(quant) != 6
                or any(not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous() for t in quant)
                or any(quant[i].dtype != torch.float32 or tuple(quant[i].shape) != (128,) for i in (0, 2, 3, 5))
                or any(quant[i].dtype != torch.int32 for i in (1, 4))):
            return incumbent(**kwargs)
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
        if self.workspace is None:
            size = cutlass_fused_moe_workspace_size(
                1, 2816, 704, 128, 8, x_dtype=x.dtype,
                weight_dtype=kwargs["fc1_expert_weights"].dtype,
                output_dtype=kwargs["output_dtype"], activation_type=kwargs["activation_type"],
                use_fused_finalize=False, device=x.device)
            if size + 32 > 8 << 20:
                raise RuntimeError(f"live workspace needs {size + 32} bytes; 8 MiB scope retained")
            self.workspace = torch.zeros(size, dtype=torch.uint8, device=x.device)
        kwargs = dict(kwargs, workspace_buffer=self.workspace, enable_pdl=False,
                      use_fused_finalize=False)
        tensors = (x, sf, ids, weights, kwargs["output"], self.workspace, self.shadow,
                   kwargs["fc1_expert_weights"], kwargs["fc2_expert_weights"], *quant)
        views = (View * LIVE_VIEW_COUNT)(*(tensor_view(t) for t in tensors))
        for tensor in tensors:
            tensor.record_stream(self.stream)
        if self.call_index >= 64:
            raise RuntimeError("bounded live preparation call count exceeded")
        directory = self.directory / f"call-{self.call_index:04d}" if self.diagnostics else None
        if self.diagnostics:
            directory.mkdir(exist_ok=False)
        self.call_index += 1
        record = {"schema": "megartx-m1-live-v2", "scope": "controlled_live_request", "lane": self.lane,
                  "diagnostics_mode": self.execution_mode,
                  "layer_name": self.layer.layer_name, **self.forward,
                  "binary_sha256": self.binary_sha256, "live_contract": self.live_contract,
                  "workspace_bytes": self.workspace.numel(),
                  "owner_extents": [v.bytes for v in views], "execution_mode": "eager",
                  "quantization_owners": [{"role": role, "dtype": str(t.dtype),
                                           "shape": list(t.shape), "view_bytes": v.bytes,
                                           "storage_bytes": v.storage_bytes}
                                          for role, t, v in zip(("fc1_weight", "fc2_weight", "fc1_act_global",
                                              "fc1_weight_sf", "fc1_global", "fc2_act_global", "fc2_weight_sf", "fc2_global"),
                                              tensors[7:], views[7:])],
                  "producer_wait_inserted": True, "consumer_wait_required": True,
                  "pdl": False, "finalize_fusion": False} if self.diagnostics else None
        framing = (LIVE_ABI_VERSION, views, len(views), ctypes.sizeof(View),
                   self.stream.cuda_stream, int(self.lane == "fused"))
        if self.diagnostics:
            started = self.native.megartx_m1_begin_v2(*framing, os.fsencode(directory))
        else:
            started = getattr(self.native, CAPTURE_FREE_BEGIN)(*framing)
        if started:
            self.failed = True
            raise RuntimeError(self.native.megartx_m1_error().decode())
        primary_error = None
        try:
            with profile_scope("megartx::m1_preparation_" + self.lane, self.diagnostics):
                result = incumbent(**kwargs)
        except BaseException as error:
            primary_error = error
            self.failed = True
            if self.diagnostics:
                record["error"] = type(error).__name__
            raise
        finally:
            try:
                status = self.native.megartx_m1_end()
                released = self.native.megartx_m1_active() == 0
                if self.diagnostics:
                    record["actual_backend"] = "fused" if status == 1 else "stock" if status == 0 else "unbound"
                    record["native_metadata"] = self.native.megartx_m1_metadata().decode()
                    record["lease_released"] = released
                    (directory / "receipt.json").write_text(json.dumps(record, indent=2))
            except BaseException as error:
                self.failed = True
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note("Native lease/receipt cleanup failed: " + str(error))
        if status < 0 or not released:
            self.failed = True
            raise RuntimeError("request did not bind/release the actual native runner")
        if self.route_controls and not self.route_controls_done:
            self.run_route_controls(incumbent, kwargs)
        return result

    def run_route_controls(self, incumbent, kwargs):
        """Two explicitly artificial calls; controlled request routing/output is unchanged."""
        import torch
        self.route_controls_done = True  # Prevent recursion through invoke.
        saved_workspace, saved_forward = self.workspace, self.forward
        needed = (2 * saved_workspace.untyped_storage().nbytes() + self.shadow.untyped_storage().nbytes()
                  + kwargs["output"].numel() * kwargs["output"].element_size() + 2 * 8 * 4)
        if needed > 8 << 20:
            raise RuntimeError(f"route-control scratch needs {needed} bytes; 8 MiB scope retained")
        routes = [torch.tensor([list(range(11, 19))], dtype=torch.int32, device=kwargs["input"].device),
                  torch.tensor([list(range(65, 73))], dtype=torch.int32, device=kwargs["input"].device)]
        output = torch.empty_like(kwargs["output"])
        probe_workspace = torch.zeros_like(saved_workspace)
        # Include both workspaces, retained shadow, output and both route arrays.
        extra_bytes = sum(t.untyped_storage().nbytes()
                          for t in (saved_workspace, probe_workspace, self.shadow, output, *routes))
        if extra_bytes > 8 << 20:
            raise RuntimeError(f"route-control scratch needs {extra_bytes} bytes; 8 MiB scope retained")
        try:
            self.workspace = probe_workspace
            for index, ids in enumerate(routes):
                self.forward = dict(saved_forward, scope="artificial_route_control",
                                    route_control_case="first_use_omits_zero" if index == 0 else "reuse_disjoint_omits_zero",
                                    route_control_scratch_bytes=extra_bytes, route_control_workspace_reused=index == 1)
                with torch.profiler.record_function("megartx::m1_artificial_route_control"):
                    self.invoke(incumbent, (), dict(kwargs, output=output, token_selected_experts=ids))
        finally:
            self.workspace, self.forward = saved_workspace, saved_forward

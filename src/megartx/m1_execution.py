"""Diagnostic policy for the eager, bounded M1 controlled request.

This switch changes evidence collection, never arithmetic or admission. It
does not enable benchmarks, natural routing, repeated requests or CUDA graphs.
"""
from contextlib import nullcontext
import os


EXECUTION_MODES = ("captured", "capture-free")
CAPTURE_FREE_BEGIN = "megartx_m1_begin_capture_free_v2"
CONTROLLER_SOURCES = ("m1_live.py", "vllm_scale_plugin.py", "m1_execution.py",
                      "controlled_capture.py", "controlled_kv_capture.py")


def execution_mode():
    value = os.environ.get("MEGARTX_M1_EXECUTION", "captured")
    if value not in EXECUTION_MODES:
        raise RuntimeError("unknown explicit M1 execution mode")
    if value == "capture-free":
        if (os.environ.get("MEGARTX_M1_PREPARATION") not in {"stock", "fused"}
                or os.environ.get("MEGARTX_SCALE_MODE") != "native"
                or not os.environ.get("MEGARTX_CONTROLLED_DIR")
                or not os.environ.get("MEGARTX_CONTROLLED_PLAN")
                or not os.environ.get("MEGARTX_LOGITS_DIR")):
            raise RuntimeError("capture-free M1 requires explicit native controlled stock/fused execution")
        if (os.environ.get("MEGARTX_M1_ROUTE_CONTROLS") == "1"
                or os.environ.get("MEGARTX_LAYER0_BOUNDARIES") == "1"
                or any(os.environ.get(key) for key in (
                    "MEGARTX_ROUTE_AUDIT_PATH", "MEGARTX_ROUTING_COVERAGE_PATH",
                    "MEGARTX_ROUTER_SCORE_DIR", "MEGARTX_M1_CAPTURE_DIR"))):
            raise RuntimeError("capture-free M1 cannot mix tensor/trace diagnostic destinations or route controls")
    return value


def profile_scope(name, enabled=True):
    if not enabled:
        return nullcontext()
    import torch
    return torch.profiler.record_function(name)

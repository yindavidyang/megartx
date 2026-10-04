"""Diagnostic policy for bounded eager M1 requests.

The capture switch changes evidence collection, never arithmetic. Repeated
natural timing requests need their separate explicit source-bound benchmark
plan; neither policy enables CUDA graphs or admits quality qualification.
"""
from contextlib import nullcontext
import os


EXECUTION_MODES = ("captured", "capture-free")
CAPTURE_FREE_BEGIN = "megartx_m1_begin_capture_free_v2"
EXTERNAL_OBSERVER_SETTER = "megartx_m1_set_external_observer_v1"
CONTROLLER_SOURCES = (
    "m1_live.py", "vllm_scale_plugin.py", "m1_execution.py",
    "controlled_capture.py", "controlled_kv_capture.py",
    "m1_normal_plan.py", "m1_normal_capture.py", "m1_external_observer.py",
    "m1_process_lifecycle.py",
    "m1_eager_benchmark.py",
)


def synchronous_scheduler_args(args):
    """Return the pinned vLLM CLI option required by every explicit M1 run."""
    if args.client == "normal" or args.m1_preparation in {"stock", "fused"}:
        return ["--no-async-scheduling"]
    return []


def execution_mode():
    value = os.environ.get("MEGARTX_M1_EXECUTION", "captured")
    if value not in EXECUTION_MODES:
        raise RuntimeError("unknown explicit M1 execution mode")
    benchmark = os.environ.get("MEGARTX_M1_EAGER_BENCHMARK_PLAN")
    if os.environ.get("MEGARTX_M1_DECODE_PROFILE_DIR") and not benchmark:
        raise RuntimeError("decode attribution requires an explicit source-bound eager pilot")
    if benchmark:
        forbidden = ("MEGARTX_CONTROLLED_DIR", "MEGARTX_CONTROLLED_PLAN", "MEGARTX_LOGITS_DIR",
                     "MEGARTX_M1_NORMAL_PLAN", "MEGARTX_M1_NORMAL_DIR", "MEGARTX_M1_CAPTURE_DIR",
                     "MEGARTX_M1_EXTERNAL_OBSERVER_DIR", "MEGARTX_M1_PROCESS_EVIDENCE_DIR",
                     "MEGARTX_ROUTE_AUDIT_PATH", "MEGARTX_ROUTING_COVERAGE_PATH", "MEGARTX_ROUTER_SCORE_DIR")
        if (value != "capture-free" or os.environ.get("MEGARTX_SCALE_MODE") != "native"
                or os.environ.get("MEGARTX_M1_PREPARATION") not in {"stock", "fused"}
                or not os.environ.get("MEGARTX_M1_EAGER_BENCHMARK_DIR")
                or any(os.environ.get(k) for k in forbidden)
                or os.environ.get("MEGARTX_M1_ROUTE_CONTROLS") == "1"
                or os.environ.get("MEGARTX_LAYER0_BOUNDARIES") == "1"):
            raise RuntimeError("eager benchmark requires isolated capture-free native observer-off execution")
        return value
    if os.environ.get("MEGARTX_M1_EAGER_BENCHMARK_DIR"):
        raise RuntimeError("eager benchmark destination requires explicit plan")
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
        if os.environ.get("MEGARTX_M1_NORMAL_PLAN") or os.environ.get("MEGARTX_M1_NORMAL_DIR"):
            raise RuntimeError("capture-free M1 is controlled-only; normal-plan validation requires captured diagnostics")
        if os.environ.get("MEGARTX_M1_EXTERNAL_OBSERVER_DIR") and os.environ.get("MEGARTX_M1_ROUTE_CONTROLS") == "1":
            raise RuntimeError("external M1 observer cannot run with artificial route controls")
    elif os.environ.get("MEGARTX_M1_EXTERNAL_OBSERVER_DIR"):
        raise RuntimeError("external M1 observer is available only for explicit capture-free validation")
    return value


def profile_scope(name, enabled=True):
    if not enabled:
        return nullcontext()
    import torch
    return torch.profiler.record_function(name)

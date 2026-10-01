"""Fail-closed binding and bounded fixtures for the actual registered MoE runner."""
import hashlib
import json
import os
import re
from pathlib import Path


def identity(obj):
    cls = type(obj)
    return cls.__module__ + "." + cls.__name__


def binding(layer, adapter):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import FusedMoEModularMethod
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import FlashInferExperts
    data = layer._megartx
    registry = get_forward_context().no_compile_layers
    if registry is not data["registry"]:
        raise RuntimeError("Forward context changed the captured runner registry")
    runner = registry.get(layer.layer_name)
    method = layer.quant_method
    owner = data["owner"]
    if runner is None or runner.routed_experts is not layer or runner._quant_method is not method:
        raise RuntimeError("Registered MoE runner/routed-layer/method identity mismatch")
    if method is not owner and not (isinstance(method, FusedMoEModularMethod) and method.old_quant_method is owner):
        raise RuntimeError("Unsupported replacement of captured quantization method")
    kernel = method.moe_kernel
    if kernel is not data["kernel"] or method.is_monolithic:
        raise RuntimeError("MoE kernel changed or is monolithic")
    if type(kernel.impl) is not FusedMoEKernelModularImpl or type(kernel.fused_experts) is not FlashInferExperts:
        raise RuntimeError("Expected modular FlashInfer CUTLASS experts; observed " + identity(kernel.impl) + " / " + identity(kernel.fused_experts))
    if layer.forward_modular.__func__ is not adapter:
        raise RuntimeError("Registered routed-layer callable bypasses the adapter")
    return runner


def prepare(model, adapter):
    import torch
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    layers = [module for module in model.modules() if isinstance(module, RoutedExperts)]
    if len(layers) != 30 or any(not hasattr(layer, "_megartx") for layer in layers):
        raise RuntimeError("Missing original-scale bindings for the 30 Gemma4 routed layers")
    records = []
    for layer in layers:
        runner = binding(layer, adapter)
        records.append({"layer_name": layer.layer_name, "registered_runner": identity(runner), "quant_method": identity(layer.quant_method), "captured_owner": identity(layer._megartx["owner"]), "kernel_impl": identity(layer.quant_method.moe_kernel.impl), "experts_impl": identity(layer.quant_method.moe_kernel.fused_experts), "routed_callable": layer.forward_modular.__func__.__qualname__})
    affected = [(layer, expert) for layer in layers for expert in layer._megartx["experts"]]
    if len(affected) != 6:
        raise RuntimeError("Frozen integration fixture requires exactly six affected experts")
    from safetensors import safe_open
    checkpoint = Path(os.environ["MEGARTX_CHECKPOINT_PATH"])
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    captures = []
    for layer, expert in affected:
        match = re.search(r"layers\.(\d+)\.", layer.layer_name)
        if match is None:
            raise RuntimeError("Cannot bind routed-layer name to the frozen checkpoint")
        for name in ("gate", "up", "down"):
            projection = getattr(expert, name)
            for suffix, value in (("weight", projection.packed), ("weight_scale", projection.scales), ("weight_scale_2", projection.global_scale)):
                key = f"model.language_model.layers.{match.group(1)}.experts.{expert.index}.{name}_proj.{suffix}"
                with safe_open(str(checkpoint / weight_map[key]), framework="pt", device="cpu") as archive:
                    original = archive.get_tensor(key).contiguous().reshape(-1).view(torch.uint8)
                retained = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
                if not torch.equal(original, retained):
                    raise RuntimeError("Loaded original-scale capture differs from immutable tensor " + key)
                captures.append({"tensor": key, "bytes": original.numel(), "sha256": hashlib.sha256(original.numpy().tobytes()).hexdigest(), "original_bytes_equal": True})
    metrics = []
    trace = Path(os.environ["MEGARTX_ACTIVATION_TRACE_PATH"])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for layer, expert in affected:
            runner = binding(layer, adapter)
            x = torch.zeros((1, 2816), dtype=torch.bfloat16, device="cuda")
            x[0, expert.index % 2816] = 32
            weights = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
            weights[0, 0] = 1
            ids = (torch.arange(8, device="cuda", dtype=torch.int32) + expert.index).remainder(128).reshape(1, 8)
            router = runner.router
            had_override = "select_experts" in router.__dict__
            saved = router.__dict__.get("select_experts")

            def forced_selection(**kwargs):
                return weights, ids.to(kwargs["topk_indices_dtype"])

            router.select_experts = forced_selection
            outputs = {}
            before = layer._megartx["forced_hits"].get(expert.index, 0)
            try:
                for mode in ("native", "reference"):
                    layer._megartx["fixture_mode"] = mode
                    with torch.profiler.record_function("megartx::forced_registered_runner_" + mode):
                        result = runner._apply_quant_method(hidden_states=x, router_logits=torch.zeros((1, 128), device="cuda"), shared_experts_input=x)
                    routed = result[1]
                    if isinstance(routed, tuple):
                        routed = routed[1]
                    outputs[mode] = routed.detach().clone()
            finally:
                layer._megartx.pop("fixture_mode", None)
                if had_override:
                    router.select_experts = saved
                else:
                    del router.select_experts
            native, reference = outputs["native"], outputs["reference"]
            if not torch.isfinite(native).all() or not torch.equal(native, reference):
                raise RuntimeError("Forced registered-runner native/reference fixture differs")
            hits = layer._megartx["forced_hits"].get(expert.index, 0) - before
            if hits < 2:
                raise RuntimeError("Forced registered-runner fixture did not execute both corrections")
            metrics.append({"layer_name": layer.layer_name, "expert": expert.index, "forced_correction_rows": hits, "observed_bf16_value_equal": True, "max_absolute_difference": (native.float() - reference.float()).abs().max().item(), "nonzero_output_elements": int((native != 0).sum().item()), "original_router_factor": 1.0})
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace))
    if not all(m["nonzero_output_elements"] > 0 for m in metrics):
        raise RuntimeError("Forced correction fixture produced only zero outputs")
    report = {"qualification": "Bounded forced integration through registered runner; artificial routing is separate from natural model routing and G1", "registered_layers": records, "original_tensor_captures": captures, "forced_fixtures": metrics, "forced_all_six_executed": True, "natural_model_forward_verified": False, "forced_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
    return layers, report


def write_proof(report):
    path = Path(os.environ["MEGARTX_ACTIVATION_PROOF_PATH"])
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)

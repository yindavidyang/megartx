"""Opt-in experimental plugin; never edits the installed vLLM or checkpoint.

Enable only with VLLM_PLUGINS=megartx_scale_adapter and an explicit mode. This
adapter is intentionally eager, single GPU/TP1/EP1, BF16, text-only Gemma4.
"""

import importlib.metadata
import json
import os
from pathlib import Path

from .m1_execution import profile_scope


def install():
    mode = os.environ.get("MEGARTX_SCALE_MODE")
    if mode is None:
        return  # Ordinary installations remain inactive unless explicitly enabled.
    controlled = os.environ.get("MEGARTX_CONTROLLED_DIR")
    if mode not in {"native", "reference", "control", "paired_reference", "gate_only_negative_control"}:
        raise RuntimeError("Unknown explicit adapter arithmetic lane")
    if controlled and mode not in {"native", "paired_reference", "gate_only_negative_control"}:
        raise RuntimeError("Controlled routing requires a declared paired arithmetic lane")
    if not controlled and mode in {"paired_reference", "gate_only_negative_control"}:
        raise RuntimeError("Paired arithmetic lanes require explicit controlled routing")
    if os.environ.get("VLLM_PLUGINS") != "megartx_scale_adapter":
        raise RuntimeError("Experimental adapter requires its explicit, exclusive VLLM_PLUGINS allowlist")
    quantizer_flags = {k: v for k, v in os.environ.items() if k.startswith("FLASHINFER_NVFP4_") or k == "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"}
    if quantizer_flags:
        raise RuntimeError("Frozen quantizer lane requires no NVFP4 override/fast-math environment flags")
    expected = {"vllm": "0.30.0", "flashinfer-python": "0.6.18.post1", "torch": "2.13.0"}
    for name, version in expected.items():
        if importlib.metadata.version(name).split("+")[0] != version:
            raise RuntimeError(f"Unsupported adapter package {name}; required {version}")
    import torch
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe
    from .nvfp4_runtime import capture_experts, run_expert
    from .nvfp4_integration import binding, prepare, write_proof

    if getattr(ModelOptNvFp4FusedMoE, "_megartx_installed", False):
        return
    old_load = ModelOptNvFp4FusedMoE.process_weights_after_loading
    old_routed = RoutedExperts.forward_modular
    old_fused = flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe
    load_ordinal = 0
    route_audit = os.environ.get("MEGARTX_ROUTE_AUDIT_PATH")
    capture = os.environ.get("MEGARTX_LOGITS_DIR")
    router_score = os.environ.get("MEGARTX_ROUTER_SCORE_DIR")
    router_observer = None
    controlled_observer = None
    normal = os.environ.get("MEGARTX_M1_NORMAL_DIR")
    normal_observer = None
    if not os.environ.get("MEGARTX_ACTIVATION_PROOF_PATH") or not os.environ.get("MEGARTX_ACTIVATION_TRACE_PATH"):
        raise RuntimeError("Experimental adapter requires explicit bounded integration-proof destinations")
    if route_audit and not capture:
        raise RuntimeError("Route audit requires the bounded quality-capture client")
    if router_score and (not capture or mode != "native"):
        raise RuntimeError("Router score diagnostic requires native mode and explicit request capture")
    if controlled and (not capture or route_audit or router_score or os.environ.get("MEGARTX_ROUTING_COVERAGE_PATH") or not os.environ.get("MEGARTX_CONTROLLED_PLAN")):
        raise RuntimeError("Controlled routing requires its isolated capture/plan without natural audit destinations")
    from .m1_live import load_controller
    m1 = load_controller(mode)
    if normal and (controlled or router_score or route_audit or not capture or not os.environ.get("MEGARTX_M1_NORMAL_PLAN")):
        raise RuntimeError("normal M1 collector requires its isolated exact plan")
    if m1 is not None and not (controlled or normal):
        raise RuntimeError("live preparation requires a bounded controlled or normal request collector")
    if normal and m1 is None:
        raise RuntimeError("normal M1 collector requires explicit preparation opt-in")
    if m1 is not None and normal and (not m1.diagnostics or m1.normal_plan is None):
        raise RuntimeError("normal M1 validation requires its captured exact-plan controller")
    if m1 is not None and controlled and m1.normal_plan is not None:
        raise RuntimeError("controlled M1 cannot carry a natural normal plan")
    if m1 is not None and m1.external_observer is not None and (not controlled or normal):
        raise RuntimeError("external M1 observer is controlled-only")
    diagnostics = m1 is None or m1.diagnostics
    validation_profile = diagnostics or (m1 is not None and m1.external_observer is not None)

    def deterministic_fused(*args, **kwargs):
        kwargs["use_fused_finalize"] = False
        if m1 is not None:
            return m1.invoke(old_fused, args, kwargs)
        return old_fused(*args, **kwargs)

    def incumbent_routed(layer, *args):
        return old_routed(layer, *args) if m1 is None else m1.routed(old_routed, layer, *args)

    def load(self, layer):
        nonlocal load_ordinal
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        pc = config.parallel_config
        for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"):
            if getattr(pc, key, 1) != 1:
                raise RuntimeError(f"Adapter requires {key}=1")
        if config.lora_config is not None or not config.model_config.enforce_eager:
            raise RuntimeError("Adapter requires eager execution without LoRA")
        if config.model_config.dtype != torch.bfloat16 or config.cache_config.cache_dtype != "bfloat16":
            raise RuntimeError("Adapter requires BF16 compute and KV cache")
        if config.scheduler_config.max_num_seqs != 1:
            raise RuntimeError("Adapter supports one active sequence")
        if config.model_config.hf_config.architectures != ["Gemma4ForConditionalGeneration"]:
            raise RuntimeError("Adapter scope is the selected Gemma4 architecture")
        if self.nvfp4_backend.value != "FLASHINFER_CUTLASS" or self.use_a16:
            raise RuntimeError("Separate-projection adapter requires FlashInfer CUTLASS W4A4")
        if tuple(torch.cuda.get_device_capability()) != (12, 0):
            raise RuntimeError("Adapter is qualified for the selected SM120 device only")
        parallel = layer.moe_config.moe_parallel_config
        if getattr(parallel, "tp_size", 1) != 1 or getattr(parallel, "ep_size", 1) != 1 or getattr(parallel, "enable_eplb", False):
            raise RuntimeError("Adapter requires TP1/EP1, without expert balancing")
        if layer.activation.value != "gelu_tanh" or layer.apply_router_weight_on_input or layer.expert_map is not None:
            raise RuntimeError("Adapter requires GELU-tanh, output routing, and local experts")
        if tuple(layer.w13_weight.shape) != (128, 1408, 1408) or tuple(layer.w2_weight.shape) != (128, 2816, 352):
            raise RuntimeError("Adapter requires all 128 local experts with the frozen Gemma4 shapes")
        # Mutate the callable only after verifying this model's complete scope.
        flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe = deterministic_fused
        experts = capture_experts(layer)
        self._megartx_experts = experts
        self._megartx_load_ordinal = load_ordinal
        load_ordinal += 1
        old_load(self, layer)
        layer._megartx = {"owner": self, "registry": config.compilation_config.static_forward_context, "kernel": self.moe_kernel, "experts": experts, "ordinal": self._megartx_load_ordinal, "dispatch_calls": 0, "forced_hits": {}, "natural_hits": {}, "controlled_hits": {}, "unmarked_hits": {}}
        manifest = os.environ.get("MEGARTX_SCALE_MANIFEST")
        if manifest and experts:
            with Path(manifest).open("a") as f:
                f.write(json.dumps({"mode": mode, "correction_active": mode != "control", "layer": layer.layer_name, "loader_ordinal": self._megartx_load_ordinal, "affected_experts": [{"expert": e.index, "gate_global": e.gate.global_scale.item(), "up_global": e.up.global_scale.item(), "down_global": e.down.global_scale.item(), "a1_shared_dequant": e.a1.item(), "a2_shared_dequant": e.a2.item()} for e in experts], "weights_requantized": False, "fused_finalize": False}) + "\n")

    def routed_adapter(layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        binding(layer, routed_adapter)
        if flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe is not deterministic_fused:
            raise RuntimeError("FlashInfer finalization callable changed")
        data = layer._megartx
        data["dispatch_calls"] += 1
        experts = data["experts"]
        controlled_request = controlled_observer is not None and controlled_observer.active and "fixture_mode" not in data
        if controlled_request:
            topk_ids, topk_weights = controlled_observer.routes(layer, x, topk_ids, topk_weights)
        if normal_observer is not None:
            topk_ids, topk_weights = normal_observer.routes(layer, x, topk_ids, topk_weights)
        normal_stages = normal_observer is not None and normal_observer.active and x.shape[0] == 1
        if router_observer is not None:
            router_observer.consume(layer, topk_ids, topk_weights)
        # Diagnostic CPU histograms are confined to quality runs and marked
        # requests. They prove natural IDs/counters independently of rare-hit
        # branches, and are absent from timed clients.
        coverage_path = os.environ.get("MEGARTX_ROUTING_COVERAGE_PATH")
        if coverage_path and experts and "fixture_mode" not in data:
            marker = Path(capture).parent / "capture-request.json"
            if marker.exists():
                request = json.loads(marker.read_text())
                ids = topk_ids.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
                weights = topk_weights.detach().float().cpu().reshape(-1)
                if torch.any((ids < 0) | (ids >= 128)):
                    raise RuntimeError("Natural router IDs are outside frozen local expert scope")
                all_counts = torch.bincount(ids, minlength=128).tolist()
                nonzero_counts = torch.bincount(ids[weights != 0], minlength=128).tolist()
                with Path(coverage_path).open("a") as f:
                    f.write(json.dumps({"case_id": request["id"], "prompt_sha256": request["prompt_sha256"], "mode": mode, "loader_ordinal": data["ordinal"], "layer_name": layer.layer_name, "input_rows": x.shape[0], "selected_slots": ids.numel(), "counts": all_counts, "nonzero_weight_counts": nonzero_counts}) + "\n")
        execution_mode = data.get("fixture_mode", mode)
        if not experts or execution_mode == "control":
            result = incumbent_routed(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
            return result if normal_observer is None else normal_observer.output(layer,result)
        if x.dtype != torch.bfloat16:
            raise RuntimeError("Adapter input must be BF16")
        # Data-dependent gather intentionally runs eagerly. Original router
        # weights are retained; no renormalization or top-k replacement occurs.
        affected = torch.zeros_like(topk_ids, dtype=torch.bool)
        for e in experts:
            affected |= topk_ids == e.index
        ordinary_weights = topk_weights.masked_fill(affected, 0)
        output = incumbent_routed(layer, x, ordinary_weights, topk_ids, shared_experts, shared_experts_input)
        pair = isinstance(output, tuple)
        routed = output[1] if pair else output
        accumulator = routed.float()
        captured = []
        for e in experts:
            locations = torch.nonzero(topk_ids == e.index)
            if locations.shape[0] == 0:
                continue
            rows, slots = locations[:, 0], locations[:, 1]
            with profile_scope("megartx::corrected_expert_" + execution_mode,
                               validation_profile or "fixture_mode" in data):
                if controlled_request:
                    with profile_scope("megartx::controlled_expert_" + execution_mode, diagnostics):
                        y, stages = run_expert(x[rows], e, mode=execution_mode, return_stages=True)
                    captured.append((e, rows, slots, topk_weights[rows, slots], stages))
                elif normal_stages:
                    y, stages = run_expert(x[rows], e, mode=execution_mode, return_stages=True)
                    captured.append((e, rows, slots, topk_weights[rows, slots], stages))
                else:
                    y = run_expert(x[rows], e, mode=execution_mode)
            kind = "forced_hits" if "fixture_mode" in data else ("controlled_hits" if controlled_request else ("unmarked_hits" if controlled else "natural_hits"))
            data[kind][e.index] = data[kind].get(e.index, 0) + rows.numel()
            if route_audit and "fixture_mode" not in data:
                marker = Path(capture).parent / "capture-request.json"
                if marker.exists():
                    request = json.loads(marker.read_text())
                    with Path(route_audit).open("a") as f:
                        f.write(json.dumps({"case_id": request["id"], "prompt_sha256": request["prompt_sha256"], "mode": execution_mode, "loader_ordinal": data["ordinal"], "layer_name": layer.layer_name, "expert": e.index, "routed_rows": rows.numel(), "nonzero_route_weights": int((topk_weights[rows, slots] != 0).sum().item())}) + "\n")
            accumulator.index_add_(0, rows, y.float() * topk_weights[rows, slots].float().unsqueeze(1))
        corrected = accumulator.to(routed.dtype)
        for e, rows, slots, weights, stages in captured:
            observer = controlled_observer if controlled_request else normal_observer
            observer.expert(layer, e, rows, slots, weights, stages, routed, corrected)
        result = (output[0], corrected) if pair else corrected
        return result if normal_observer is None else normal_observer.output(layer,result)

    ModelOptNvFp4FusedMoE.process_weights_after_loading = load
    RoutedExperts.forward_modular = routed_adapter
    ModelOptNvFp4FusedMoE._megartx_installed = True
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
    old_forward = Gemma4ForConditionalGeneration.forward
    old_logits = Gemma4ForConditionalGeneration.compute_logits
    counter = 0
    forward_counter = 0
    context = None
    integration_layers = None
    integration_report = None

    def model_forward(self, input_ids, positions, *args, **kwargs):
        nonlocal context, forward_counter, integration_layers, integration_report, router_observer, controlled_observer, normal_observer
        if integration_layers is None:
            integration_layers, integration_report = prepare(self, routed_adapter)
            if router_score:
                from .router_score_capture import RouterScoreCapture
                router_observer = RouterScoreCapture(self, integration_layers)
            if controlled:
                from .controlled_capture import ControlledCapture
                controlled_observer = ControlledCapture(self, integration_layers, mode,
                    execution_mode="captured" if m1 is None else m1.execution_mode,
                    external_observer=None if m1 is None else m1.external_observer)
            if normal:
                from .m1_normal_capture import NormalCapture
                normal_observer = NormalCapture(self, integration_layers, mode)
        before = [layer._megartx["dispatch_calls"] for layer in integration_layers]
        primary_error = None
        try:
            if router_observer is not None:
                router_observer.begin(input_ids, positions)
            if controlled_observer is not None:
                controlled_observer.begin(input_ids, positions)
            if normal_observer is not None:
                normal_observer.begin(input_ids, positions)
            if m1 is not None:
                m1.begin_forward(input_ids, positions)
            result = old_forward(self, input_ids, positions, *args, **kwargs)
            if router_observer is not None:
                router_observer.end()
            if any(layer._megartx["dispatch_calls"] <= count for layer, count in zip(integration_layers, before)):
                raise RuntimeError("Model forward bypassed one or more registered routed adapters")
            if not integration_report["natural_model_forward_verified"]:
                # This historical field verifies dispatch, not natural coverage.
                integration_report["natural_model_forward_verified"] = True
                write_proof(integration_report)
            if controlled_observer is not None:
                controlled_observer.end()
            if normal_observer is not None:
                normal_observer.end()
        except BaseException as error:
            primary_error = error
            if m1 is not None:
                m1.failed = True
            for observer in (controlled_observer,normal_observer):
                if observer is not None:
                    try:
                        observer.abort(error)
                    except BaseException as cleanup:
                        if hasattr(error,"add_note"):
                            error.add_note("Capture cleanup failed: "+str(cleanup))
            context = None
            raise
        finally:
            if m1 is not None:
                try:
                    m1.end_forward()
                except BaseException as cleanup:
                    m1.failed = True
                    if primary_error is None:
                        raise
                    if hasattr(primary_error,"add_note"):
                        primary_error.add_note("Forward lease cleanup failed: "+str(cleanup))
        if not capture:
            return result
        if not isinstance(result, torch.Tensor) or result.ndim != 2 or result.shape[1] != 2816:
            raise RuntimeError("Capture requires final Gemma4 hidden-state rows")
        if positions.ndim != 1 or positions.numel() != result.shape[0]:
            raise RuntimeError("Capture requires one verified position per hidden row")
        context = {"hidden": result.detach(), "positions": positions.detach(), "input_ids": None if input_ids is None else input_ids.detach(), "forward_id": forward_counter}
        forward_counter += 1
        return result

    def logits_forward(self, hidden_states, *args, **kwargs):
        nonlocal counter
        result = old_logits(self, hidden_states, *args, **kwargs)
        if not capture:
            return result
        marker = Path(capture).parent / "capture-request.json"
        # Startup/dummy runs are excluded. The local serial client creates
        # this marker before POST and removes it after response completion.
        if result is not None and marker.exists():
            request = json.loads(marker.read_text())
            directory = Path(capture)
            directory.mkdir(parents=True, exist_ok=True)
            # User-generated bounded corpus only. Raw logits stay local.
            if counter >= 2048 or result.numel() > 262144 * 256:
                raise RuntimeError("Bounded logit capture limit exceeded")
            path = directory / f"logits-{counter:06d}.pt"
            if path.exists():
                raise RuntimeError("Refusing to overwrite prior logit capture")
            rows = min(8, result.shape[0])
            if context is None:
                raise RuntimeError("No current forward context for logit correspondence")
            selected, source = hidden_states[-rows:].detach(), context["hidden"]
            if selected.dtype != source.dtype or selected.shape[1] != source.shape[1]:
                raise RuntimeError("Logit rows differ from final model hidden-state contract")
            # Prompt logprobs normally use a storage view, while sampled
            # rows are gathered. First prove view correspondence; otherwise
            # require unique equality of ALL raw hidden-state bytes.
            row_bytes = source.shape[1] * source.element_size()
            offset = selected.data_ptr() - source.data_ptr()
            aliases = selected.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
            if aliases and source.is_contiguous() and selected.is_contiguous() and offset >= 0 and offset % row_bytes == 0 and offset // row_bytes + rows <= source.shape[0]:
                indices = torch.arange(offset // row_bytes, offset // row_bytes + rows, device=source.device)
                method = "verified_storage_view"
            else:
                selected_bits = selected.contiguous().view(torch.uint8)
                source_bits = source.contiguous().view(torch.uint8)
                matches = (selected_bits[:, None, :] == source_bits[None, :, :]).all(-1)
                if not torch.all(matches.sum(-1) == 1):
                    raise RuntimeError("Logit-to-token mapping is missing or ambiguous")
                indices = matches.to(torch.int32).argmax(-1)
                method = "unique_full_hidden_bit_match"
            if not torch.equal(selected.contiguous().view(torch.uint8), source[indices].contiguous().view(torch.uint8)):
                raise RuntimeError("Logit hidden-row identity check failed")
            pos = context["positions"][indices].long().cpu().tolist()
            token_ids = None if context["input_ids"] is None else context["input_ids"][indices].long().cpu().tolist()
            if request["teacher_forced"]:
                if token_ids is None or any(p < 0 or p >= len(request["prompt_token_ids"]) or t != request["prompt_token_ids"][p] for p, t in zip(pos, token_ids)):
                    raise RuntimeError("Captured row does not match the supplied teacher-forced prefix")
            if normal_observer is not None:
                if token_ids is None:
                    raise RuntimeError("normal M1 logits require actual input IDs")
                normal_observer.logits(result[-rows:], pos, token_ids, method, result.shape[0])
            elif controlled_observer is not None:
                if token_ids is None:
                    raise RuntimeError("Controlled logits require actual input IDs")
                controlled_observer.logits(result[-rows:], pos, token_ids, method, result.shape[0])
            else:
                torch.save({"logits": result[-rows:].detach().float().cpu(), "source_rows": result.shape[0], "selected_last_rows": rows, "forward_id": context["forward_id"], "input_positions": pos, "prediction_positions": [p + 1 for p in pos], "input_token_ids": token_ids, "row_correspondence": method, "case_id": request["id"], "prompt_sha256": request["prompt_sha256"]}, path)
            counter += 1
        return result

    Gemma4ForConditionalGeneration.forward = model_forward
    Gemma4ForConditionalGeneration.compute_logits = logits_forward

"""Bounded GPU fixture for original projection globals and native SM120 GEMMs.

Run only on an idle approved GPU, with the immutable checkpoint already local.
No model download, checkpoint write, training, system changes or secret access.
"""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
jobs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=5)
if jobs.strip():
    raise SystemExit("Another GPU job is present; fixture not started")

import torch
import numpy as np
from safetensors import safe_open
import nvfp4_reference as independent
from megartx.nvfp4_runtime import Expert, make_projection, run_expert, decode_units, unswizzle, mm_native, mm_reference

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(name):
    with safe_open(str(args.model / index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda()


def projection(layer, expert, which):
    key = f"model.language_model.layers.{layer}.experts.{expert}.{which}_proj."
    return make_projection(tensor(key + "weight"), tensor(key + "weight_scale"), tensor(key + "weight_scale_2"))


def shared_activation(layer, which):
    keys = [k for k in index if k.startswith(f"model.language_model.layers.{layer}.experts.") and k.endswith(f".{which}_proj.input_scale")]
    return torch.stack([tensor(k).reshape(()) for k in keys]).max().reshape(1)


def comparison(native, reference):
    a, b = native.double(), reference.double()
    d = a - b
    return {"finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "max_abs": d.abs().max().item(), "relative_l2": (torch.linalg.vector_norm(d) / torch.linalg.vector_norm(b).clamp_min(1e-300)).item(), "equal_fraction": (a == b).double().mean().item(), "reference_rms": b.square().mean().sqrt().item()}


def gemm_bound(q, sf, p, ag, native):
    a = decode_units(q, unswizzle(sf, q.shape[0], q.shape[1] * 2 // 16))
    b = decode_units(p.packed, p.scales)
    alpha = (p.global_scale * ag).float().double()
    exact = a @ b.T * alpha
    n = a.shape[-1]
    # Standard F32 FMA accumulation bound, plus scalar multiply and one BF16
    # output rounding. This envelope is analytical, frozen before candidate
    # measurements, and per-element (cancellation does not loosen relative RMS).
    u = 2 ** -24
    gamma = ((n + 3) * u) / (1 - (n + 3) * u)
    accumulation = gamma * (a.abs() @ b.abs().T * alpha.abs())
    rounding = (exact.abs() + accumulation) * (2 ** -8)
    bound = accumulation + rounding + 2 ** -133
    error = (native.double() - exact).abs()
    return {"f32_gamma": gamma, "elements": native.numel(), "outside_analytical_envelope": (error > bound).sum().item(), "max_error_over_bound": (error / bound).max().item()}


report = {"qualification": "Packing/global preservation and bounded operator evidence; not a full-model quality gate", "checkpoint_revision": "a19cfe00be84568a6867111c9a68c9c44fdcffe6", "vllm": "0.30.0", "flashinfer": "0.6.18.post1", "native_backend": "cutlass", "activation_quantizer": "Pinned runtime layerwide maximum calibrated global", "f32_bound": "Conditional diagnostic: assumes RNE F32 accumulation; gamma_(K+3)*sum(abs(products))*abs(alpha) + u_BF16*(abs(exact)+bound). PTX does not guarantee MMA rounding order/RNE.", "cases": []}
report["quantizer_environment_overrides"] = {k: v for k, v in __import__("os").environ.items() if k.startswith("FLASHINFER_NVFP4_") or k == "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"}
if report["quantizer_environment_overrides"]:
    raise RuntimeError("Unexpected frozen quantizer override")
generator = torch.Generator(device="cuda").manual_seed(21491)
for layer, expert in [(0, 0), (0, 42), (0, 82), (1, 126), (2, 89), (3, 7), (5, 12)]:
    e = Expert(expert, projection(layer, expert, "gate"), projection(layer, expert, "up"), projection(layer, expert, "down"), shared_activation(layer, "gate"), shared_activation(layer, "down"))
    for rows, magnitude in [(1, 0.0), (1, 1.0), (8, .125), (32, 2.0)]:
        x = (torch.randn(rows, 2816, generator=generator, device="cuda") * magnitude).to(torch.bfloat16)
        native, stages = run_expert(x, e, return_stages=True)
        oracle, reference = run_expert(x, e, mode="reference", return_stages=True)
        checks = {}
        for name, q, sf, p, ag in [("gate", stages["q1"], stages["sf1"], e.gate, e.a1), ("up", stages["q1"], stages["sf1"], e.up, e.a1), ("down", stages["q2"], stages["sf2"], e.down, e.a2)]:
            n = stages[name]
            r = mm_reference(q, sf, p, ag)
            to_bytes = lambda t: t.contiguous().view(torch.uint8).cpu().numpy()
            a_linear = unswizzle(sf, q.shape[0], q.shape[1] * 2 // 16)
            numpy_reference = independent.gemm_reference(to_bytes(q), to_bytes(a_linear), to_bytes(p.packed), to_bytes(p.scales), alpha_fp32=(p.global_scale * ag).float().item())
            independent_check = independent.compare_gemm(n.float().cpu().numpy(), numpy_reference, output_dtype="bf16")
            decode_agreement = np.array_equal(decode_units(q, a_linear).cpu().numpy(), independent.decode_qsf(to_bytes(q), to_bytes(a_linear)))
            layout_agreement = np.array_equal(to_bytes(p.scales), independent.unswizzle_128x4(to_bytes(p.swizzled), p.scales.shape[0], p.scales.shape[1]))
            checks[name] = {**comparison(n, r), **gemm_bound(q, sf, p, ag, n), "independent_numpy": independent_check, "independent_activation_decode_exact": decode_agreement, "independent_weight_layout_exact": layout_agreement}
        # Check activation against separately implemented NumPy mathematical
        # semantics, not the adapter's own helper. Approximate-tanh and FMA
        # rounding are recorded, rather than asserting bit-exact libtanh.
        g = stages["gate"].float().cpu().numpy()
        u = stages["up"].float().cpu().numpy()
        semantic = independent.round_bf16((independent.gelu_tanh_semantic_fp64(g).astype(np.float32) * u).astype(np.float32))
        actual_h = stages["activation"].float().cpu().numpy()
        semantic_delta = actual_h.astype(np.float64) - semantic.astype(np.float64)
        activation_check = {"max_abs": float(np.max(np.abs(semantic_delta))), "equal_fraction": float(np.mean(actual_h == semantic)), "relative_l2": float(np.linalg.norm(semantic_delta) / max(np.linalg.norm(semantic.astype(np.float64)), 1e-300)), "qualification": "Independent mathematical semantics; CUDA approximate tanh can differ at BF16/FP4 boundaries"}
        # Correlated bug control is deliberately WRONG: use the gate global
        # for the up projection, as the pinned stock loader does.
        correct_up = e.up.global_scale.clone()
        e.up.global_scale = e.gate.global_scale
        wrong, _ = run_expert(x, e, return_stages=True)
        e.up.global_scale = correct_up
        native_a2 = decode_units(stages["q2"], unswizzle(stages["sf2"], rows, 44))
        reference_a2 = decode_units(reference["q2"], unswizzle(reference["sf2"], rows, 44))
        case = {"layer": layer, "expert": expert, "rows": rows, "input_rms_scale": magnitude, "gate_global": e.gate.global_scale.item(), "up_global": e.up.global_scale.item(), "stock_gate_over_up": (e.gate.global_scale / e.up.global_scale).item(), "gemm_checks": checks, "independent_activation_semantic": activation_check, "end_to_end_reference": comparison(native, oracle), "stock_bug_control": comparison(wrong, oracle), "fp4_intermediate_payload_equal": bool(torch.equal(stages["q2"], reference["q2"])), "fp4_intermediate_scale_equal": bool(torch.equal(stages["sf2"].view(torch.uint8), reference["sf2"].view(torch.uint8))), "fp4_intermediate_decoded_values_equal": bool(torch.equal(native_a2, reference_a2))}
        report["cases"].append(case)
        (args.output / "progress.json").write_text(json.dumps(case, indent=2))
        print(json.dumps({"layer": layer, "expert": expert, "rows": rows, "end_to_end_relative_l2": case["end_to_end_reference"]["relative_l2"], "stock_bug_relative_l2": case["stock_bug_control"]["relative_l2"], "outside_bounds": sum(c["outside_analytical_envelope"] for c in checks.values())}), flush=True)
    del e
    torch.cuda.empty_cache()
report["native_gemm_conditional_diagnostic_passed"] = all(c["finite"] and c["outside_analytical_envelope"] == 0 for case in report["cases"] for c in case["gemm_checks"].values())
report["independent_operand_and_gemm_checks_passed"] = all(c["independent_numpy"]["conditional_envelope_pass"] and c["independent_activation_decode_exact"] and c["independent_weight_layout_exact"] for case in report["cases"] for c in case["gemm_checks"].values())
report["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
report["max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
(args.output / "operator-report.json").write_text(json.dumps(report, indent=2))
raise SystemExit(0 if report["native_gemm_conditional_diagnostic_passed"] and report["independent_operand_and_gemm_checks_passed"] else 2)

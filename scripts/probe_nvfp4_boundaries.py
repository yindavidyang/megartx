"""Idle-GPU quantizer boundaries and compiled activation instruction evidence."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
flags = {k: v for k, v in os.environ.items() if k.startswith("FLASHINFER_NVFP4_") or k == "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"}
if flags:
    raise RuntimeError("Frozen quantizer lane requires no override environment flags")
if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=5).strip():
    raise SystemExit("Another GPU job is present")
import numpy as np
import torch
import nvfp4_reference as ref
from megartx.nvfp4_runtime import quantize, unswizzle
from megartx.nvfp4_activation import cutlass_gelu_product

report = {"qualification": "Named mathematical/IEEE profiles compared with pinned CUDA fast math; differences are not silently accepted as frozen-quantizer equivalence", "quantizer_cases": []}
thresholds = np.asarray([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -6], dtype=np.float32)
for label, values, global_ in [("zero", np.zeros(16, dtype=np.float32), 537.), ("ties", thresholds, 256.), ("small", thresholds * np.float32(2 ** -20), 1.), ("ordinary", thresholds * np.float32(.17), 537.), ("saturation", thresholds * np.float32(1024), 537.)]:
    x = torch.tensor(np.tile(values, 2).reshape(1, 32), device="cuda").to(torch.bfloat16)
    # Preserve the exact pre-rounded F32 quant multiplier through an F32
    # dequant scalar, just like the selected runtime.
    ag = torch.tensor([1 / global_], dtype=torch.float32, device="cuda")
    g = (1 / ag).float().item()
    q, sf = quantize(x, ag)
    raw = q.cpu().numpy()
    linear = unswizzle(sf, 1, 2).view(torch.uint8).cpu().numpy()
    comparisons = {}
    for profile in ("vllm_python_rn", "flashinfer_strict_rn"):
        expected = ref.quantize_activation_rn(x.float().cpu().numpy(), g, profile=profile)
        native_decoded = ref.decode_qsf(raw, linear) / g
        expected_decoded = ref.decode_qsf(expected.packed, expected.linear_sf) / g
        comparisons[profile] = {"payload_byte_mismatches": int(np.count_nonzero(raw != expected.packed)), "scale_byte_mismatches": int(np.count_nonzero(linear != expected.linear_sf)), "decoded_values_equal": bool(np.array_equal(native_decoded, expected_decoded)), "max_decoded_abs_difference": float(np.max(np.abs(native_decoded - expected_decoded)))}
    report["quantizer_cases"].append({"id": label, "quant_global_fp32": g, "input_bf16_values": x.float().cpu().numpy().tolist(), "native_payload": raw.tolist(), "native_linear_scale_bytes": linear.tolist(), "comparisons": comparisons})

# Symmetric values, signed zeros, very small finite values, and BF16-rounded
# random points exercise independent semantics without duplicating PTX code.
generator = torch.Generator(device="cuda").manual_seed(50117)
gate = (torch.randn((8, 704), device="cuda", generator=generator) * 8).to(torch.bfloat16)
up = (torch.randn((8, 704), device="cuda", generator=generator) * 3).to(torch.bfloat16)
gate[0, :16] = torch.tensor([-64, -16, -6, -3, -1, -.5, -0., 0., 2 ** -20, .5, 1, 3, 6, 16, 32, 64], device="cuda", dtype=torch.bfloat16)
native_tensor, compiled = cutlass_gelu_product(gate, up, return_compiled=True)
native = native_tensor.float().cpu().numpy()
semantic = ref.round_bf16(ref.gelu_tanh_semantic_fp64(gate.float().cpu().numpy()).astype(np.float32) * up.float().cpu().numpy())
delta = native.astype(np.float64) - semantic.astype(np.float64)
report["activation_semantics"] = {"elements": native.size, "max_abs": float(np.max(np.abs(delta))), "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(semantic.astype(np.float64))), "equal_fraction": float(np.mean(native == semantic)), "finite": bool(np.isfinite(native).all()), "scope": "Independent mathematical GELU reference; source-matched fast-tanh implementation is not declared bit-exact to libtanh"}
data = compiled.asm["ptx"]
(args.output / "owned-activation.ptx").write_text(data)
expected_sequence = ["mul.rn.f32", "fma.rn.f32", "mul.rn.f32", "tanh.approx.f32", "fma.rn.f32", "mul.rn.f32", "mul.rn.f32"]


def operand_bits(operand, preceding_ptx):
    operand = operand.strip()
    if operand.startswith("%"):
        definitions = re.findall(r"mov\.(?:b32|f32)\s+" + re.escape(operand) + r"\s*,\s*([^;]+);", preceding_ptx)
        if not definitions:
            raise RuntimeError("No verified coefficient definition for " + operand)
        operand = definitions[-1].strip()
    if operand.lower().startswith("0f"):
        return int(operand[2:], 16)
    return int(operand, 0) & 0xffffffff


blocks = []
for match in re.finditer(r"MEGARTX_GELU_PINNED_BEGIN(.*?)MEGARTX_GELU_PINNED_END", data, re.S):
    body = match.group(1)
    instructions = re.findall(r"(?:mul|fma)\.rn\.f32|tanh\.approx\.f32", body)
    k1 = re.search(r"mul\.rn\.f32\s+v0\s*,\s*([^,]+),", body)
    k0 = re.search(r"fma\.rn\.f32\s+v1\s*,\s*v0\s*,\s*[^,]+,\s*([^;]+);", body)
    half = re.search(r"mul\.rn\.f32\s+v5\s*,\s*v4\s*,\s*([^;]+);", body)
    if k0 is None or k1 is None or half is None:
        raise RuntimeError("Marked GELU coefficient operands missing")
    bits = {"k0": operand_bits(k0.group(1), data[:match.start()]), "k1": operand_bits(k1.group(1), data[:match.start()]), "half": operand_bits(half.group(1), data[:match.start()])}
    blocks.append({"source_order_instructions": instructions, "source_order_matches": instructions == expected_sequence, "coefficient_bits": bits, "coefficient_bits_match": bits == {"k0": 0x3f4c422a, "k1": 0x3d122279, "half": 0x3f000000}})
report["compiled_ptx"] = {"sha256": hashlib.sha256(data.encode()).hexdigest(), "source": "CompiledKernel returned by this invocation", "marked_blocks": blocks, "bf16_rne_output_conversion": "cvt.rn.bf16" in data}
report["compiled_operation_sequence_verified"] = bool(blocks) and all(b["source_order_matches"] and b["coefficient_bits_match"] for b in blocks) and "cvt.rn.bf16" in data
(args.output / "boundary-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"activation": report["activation_semantics"], "compiled_operation_sequence_verified": report["compiled_operation_sequence_verified"], "quantizer_cases": report["quantizer_cases"]}), flush=True)
raise SystemExit(0 if report["activation_semantics"]["finite"] and report["compiled_operation_sequence_verified"] else 2)

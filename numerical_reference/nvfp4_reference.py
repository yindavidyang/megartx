"""Independent, CPU-only NVFP4 format and bounded GEMM reference.

This module imports neither vLLM nor FlashInfer nor CUDA. Production qualification
requires captured quantized operands and an independently verified cast profile.
The supplied RN activation quantizer is a named reference, not an assertion that
CUDA approximate reciprocal instructions are bit-identical to IEEE division.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GROUP_SIZE = 16
F32_U = 2.0**-24
F64_U = 2.0**-53


def decode_e2m1(code: int) -> float:
    if not 0 <= code <= 15:
        raise ValueError("An E2M1 code must fit in four bits")
    return math.copysign(FP4_MAGNITUDES[code & 7], -1 if code & 8 else 1)


def decode_e4m3fn(code: int) -> float:
    if not 0 <= code <= 255:
        raise ValueError("An E4M3 code must fit in one byte")
    exponent, fraction = (code >> 3) & 15, code & 7
    if exponent == 15 and fraction == 7:
        return math.nan
    magnitude = (
        fraction * 2.0**-9
        if exponent == 0
        else (1.0 + fraction / 8.0) * 2.0 ** (exponent - 7)
    )
    return math.copysign(magnitude, -1 if code & 128 else 1)


E2M1_TABLE = np.asarray([decode_e2m1(i) for i in range(16)], dtype=np.float64)
E4M3_TABLE = np.asarray([decode_e4m3fn(i) for i in range(256)], dtype=np.float64)
E4M3_POSITIVE = tuple(float(x) for x in E4M3_TABLE[:127])


def _nearest_even_code(value: float, magnitudes: tuple[float, ...]) -> int:
    upper = bisect.bisect_left(magnitudes, value)
    if upper == 0:
        return 0
    if upper == len(magnitudes):
        return len(magnitudes) - 1
    lower = upper - 1
    dlow, dhigh = value - magnitudes[lower], magnitudes[upper] - value
    if dlow < dhigh:
        return lower
    if dhigh < dlow:
        return upper
    return lower if lower % 2 == 0 else upper


def encode_e2m1_rne(value: float) -> int:
    if math.isnan(value):
        raise ValueError("NaN has no finite E2M1 encoding")
    sign = 8 if math.copysign(1.0, value) < 0 else 0
    return sign | _nearest_even_code(abs(value), FP4_MAGNITUDES)


def encode_e4m3fn_rne_satfinite(value: float) -> int:
    if math.isnan(value):
        raise ValueError("NaN is invalid for an NVFP4 scale")
    sign = 128 if math.copysign(1.0, value) < 0 else 0
    return sign | _nearest_even_code(abs(value), E4M3_POSITIVE)


def _u8_matrix(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value)
    if out.dtype != np.uint8 or out.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional uint8 array")
    return out


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    packed = _u8_matrix(packed, "packed")
    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2], out[:, 1::2] = packed & 15, packed >> 4
    return out


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    codes = _u8_matrix(codes, "codes")
    if codes.shape[1] % 2 or np.any(codes > 15):
        raise ValueError("Expected an even width and codes in [0,15]")
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def decode_qsf(packed: np.ndarray, linear_sf: np.ndarray) -> np.ndarray:
    """Decode q*sf without the tensor global, exactly representable in BF16."""
    codes = unpack_nibbles(packed)
    sf = _u8_matrix(linear_sf, "linear_sf")
    if codes.shape[1] % GROUP_SIZE:
        raise ValueError("Logical K must be divisible by 16")
    if sf.shape != (codes.shape[0], codes.shape[1] // GROUP_SIZE):
        raise ValueError("Scale shape must be [rows, logical_K/16]")
    values = E4M3_TABLE[sf]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("NVFP4 scales must be finite and nonnegative")
    decoded = E2M1_TABLE[codes].reshape(codes.shape[0], -1, GROUP_SIZE)
    return (decoded * values[:, :, None]).reshape(codes.shape)


def decode_weight(
    packed: np.ndarray, linear_sf: np.ndarray, weight_global: float
) -> np.ndarray:
    """Canonical FP64 weight values; no intermediate BF16 scale fold."""
    _positive_f32(weight_global, "weight_global")
    return decode_qsf(packed, linear_sf) * float(np.float32(weight_global))


def sf_offset_128x4(row: int, block: int, padded_blocks: int) -> int:
    """Independent coordinate formula for the pinned 128x4 scale layout."""
    if row < 0 or block < 0 or padded_blocks % 4 or block >= padded_blocks:
        raise ValueError("Invalid padded coordinate")
    return (
        (((row // 128) * (padded_blocks // 4) + block // 4) * 32 + row % 32)
        * 4
        + (row % 128) // 32
    ) * 4 + block % 4


def swizzle_128x4(linear: np.ndarray) -> np.ndarray:
    """Fixture helper using coordinates, rather than the production transpose."""
    linear = np.asarray(linear)
    if linear.ndim != 2:
        raise ValueError("Expected a two-dimensional scale matrix")
    rows, blocks = linear.shape
    padded_rows, padded_blocks = (rows + 127) // 128 * 128, (blocks + 3) // 4 * 4
    out = np.zeros(padded_rows * padded_blocks, dtype=linear.dtype)
    for row in range(rows):
        for block in range(blocks):
            out[sf_offset_128x4(row, block, padded_blocks)] = linear[row, block]
    return out.reshape(padded_rows, padded_blocks)


def unswizzle_128x4(physical: np.ndarray, rows: int, blocks: int) -> np.ndarray:
    physical = np.asarray(physical)
    padded_rows, padded_blocks = (rows + 127) // 128 * 128, (blocks + 3) // 4 * 4
    if rows <= 0 or blocks <= 0 or physical.size != padded_rows * padded_blocks:
        raise ValueError("The physical buffer must match the exact padded extent")
    flat = physical.reshape(-1)
    out = np.empty((rows, blocks), dtype=physical.dtype)
    for row in range(rows):
        for block in range(blocks):
            out[row, block] = flat[sf_offset_128x4(row, block, padded_blocks)]
    return out


def round_bf16(value: np.ndarray) -> np.ndarray:
    """IEEE RNE BF16 values returned as float32, without Torch conversion."""
    x = np.asarray(value, dtype=np.float32).copy()
    bits = x.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    upper = rounded >> np.uint32(16)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    upper = np.where(nan, (bits >> np.uint32(16)) | np.uint32(0x40), upper)
    return np.asarray(upper << np.uint32(16), dtype=np.uint32).view(np.float32)


def _positive_f32(value: float, name: str) -> np.float32:
    out = np.float32(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a finite positive FP32 scalar")
    return out


@dataclass(frozen=True)
class QuantizedActivation:
    packed: np.ndarray
    linear_sf: np.ndarray
    quant_global: float
    profile: str


def quantize_activation_rn(
    value: np.ndarray,
    quant_global: float,
    *,
    profile: Literal["vllm_python_rn", "flashinfer_strict_rn"] = "vllm_python_rn",
) -> QuantizedActivation:
    """Named IEEE quantizer profiles; CUDA approximate reciprocals are unresolved.

    quant_global is the large multiplier, e.g. FP32(1/source_input_scale).
    vllm_python_rn multiplies amax by FP32(1/6), then by the global.
    flashinfer_strict_rn multiplies the global by FP32(1/6), then by amax.
    These orders can produce different E4M3 codes at rounding boundaries.
    Neither profile silently substitutes for the default CUDA fast-math path.
    """
    x = np.asarray(value, dtype=np.float32)
    g = _positive_f32(quant_global, "quant_global")
    if x.ndim != 2 or x.shape[1] % GROUP_SIZE or not np.isfinite(x).all():
        raise ValueError("Finite [rows,K] activations with K divisible by 16 required")
    if profile not in ("vllm_python_rn", "flashinfer_strict_rn"):
        raise ValueError("Unknown quantizer profile")
    codes = np.empty_like(x, dtype=np.uint8)
    sf = np.empty((x.shape[0], x.shape[1] // GROUP_SIZE), dtype=np.uint8)
    one_over_g = np.float32(np.float32(1.0) / g)
    for row in range(x.shape[0]):
        for block in range(sf.shape[1]):
            part = x[row, block * GROUP_SIZE : (block + 1) * GROUP_SIZE]
            mx = np.float32(np.max(np.abs(part)))
            with np.errstate(over="ignore", under="ignore"):
                if profile == "vllm_python_rn":
                    unscaled = np.float32(mx * np.float32(1.0 / 6.0))
                    scale = np.float32(g * unscaled)
                else:
                    global_over_six = np.float32(g * np.float32(1.0 / 6.0))
                    scale = np.float32(mx * global_over_six)
            sfcode = encode_e4m3fn_rne_satfinite(float(scale))
            sf[row, block] = sfcode
            scale_value = np.float32(decode_e4m3fn(sfcode))
            if profile == "flashinfer_strict_rn" and mx == 0:
                output_scale = np.float32(0.0)
            else:
                denominator = np.float32(scale_value * one_over_g)
                # The vLLM Python source uses 1/(x+(x==0)*1e8), whereas
                # FlashInfer strict mode uses reciprocal+finite clamp and
                # selects zero only when block_amax is zero. A nonzero block
                # whose SF underflows can retain saturated FP4 codes.
                if profile == "vllm_python_rn" and denominator == 0:
                    denominator = np.float32(1e8)
                with np.errstate(over="ignore", divide="ignore"):
                    output_scale = np.float32(np.float32(1.0) / denominator)
                if profile == "flashinfer_strict_rn":
                    output_scale = min(output_scale, np.finfo(np.float32).max)
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.clip(part * output_scale, -6.0, 6.0)
            if not np.isfinite(normalized).all():
                raise ValueError("Quantizer produced invalid normalized values")
            codes[row, block * GROUP_SIZE : (block + 1) * GROUP_SIZE] = [
                encode_e2m1_rne(float(v)) for v in normalized
            ]
    return QuantizedActivation(pack_nibbles(codes), sf, float(g), profile)


def gamma(count: int, unit_roundoff: float) -> float:
    if count < 0 or count * unit_roundoff >= 1:
        raise ValueError("Invalid rounding bound")
    return count * unit_roundoff / (1.0 - count * unit_roundoff)


@dataclass(frozen=True)
class GemmReference:
    accumulator_fp64: np.ndarray
    sum_absolute_products: np.ndarray
    alpha_fp32: float
    ideal_scaled_fp64: np.ndarray
    fp32_error_bound: np.ndarray
    expected_fp32: np.ndarray
    expected_bf16: np.ndarray
    bf16_lower: np.ndarray
    bf16_upper: np.ndarray


def gemm_reference(
    a_packed: np.ndarray,
    a_linear_sf: np.ndarray,
    w_packed: np.ndarray,
    w_linear_sf: np.ndarray,
    *,
    alpha_fp32: float,
    output_chunk_rows: int = 64,
) -> GemmReference:
    """FP64 dot oracle plus a conditional IEEE FP32 diagnostic envelope.

    Weight decoding is limited to output_chunk_rows at a time. alpha_fp32 MUST
    be the actual FP32 epilogue scalar, not a reconstructed reciprocal of gA.
    q*sf products are exact in BF16; their pairwise products are exact in FP32.
    The gamma_K bound assumes standard IEEE FP32 accumulation with RNE,
    no FTZ/overflow, and one FP32 multiply by alpha before the output cast.
    PTX E2M1 MMA does NOT guarantee accumulation rounding or subnormal handling.
    The bound is a diagnostic until kernel-specific evidence supports its
    assumptions; being inside it cannot independently qualify a native kernel.
    """
    a = decode_qsf(a_packed, a_linear_sf)
    w_packed = _u8_matrix(w_packed, "w_packed")
    w_linear_sf = _u8_matrix(w_linear_sf, "w_linear_sf")
    if a.shape[1] != w_packed.shape[1] * 2 or output_chunk_rows <= 0:
        raise ValueError("GEMM dimensions or output chunk size are invalid")
    alpha = _positive_f32(alpha_fp32, "alpha_fp32")
    result_shape = (a.shape[0], w_packed.shape[0])
    acc, sum_abs = np.empty(result_shape), np.empty(result_shape)
    for start in range(0, w_packed.shape[0], output_chunk_rows):
        end = min(start + output_chunk_rows, w_packed.shape[0])
        w = decode_qsf(w_packed[start:end], w_linear_sf[start:end])
        acc[:, start:end] = a @ w.T
        sum_abs[:, start:end] = np.abs(a) @ np.abs(w).T
    if not np.isfinite(acc).all() or not np.isfinite(sum_abs).all():
        raise ValueError("GEMM overflow or invalid operands")
    # Add FP64 oracle reduction uncertainty independently of candidate results.
    dot_bound = (gamma(a.shape[1], F32_U) + gamma(a.shape[1], F64_U)) * sum_abs
    ideal = acc * float(alpha)
    error = abs(float(alpha)) * dot_bound + F32_U * (
        np.abs(ideal) + abs(float(alpha)) * dot_bound
    )
    expected_fp32 = (acc.astype(np.float32) * alpha).astype(np.float32)
    if not np.isfinite(expected_fp32).all():
        raise ValueError("FP32 epilogue overflow")
    return GemmReference(
        acc,
        sum_abs,
        float(alpha),
        ideal,
        error,
        expected_fp32,
        round_bf16(expected_fp32),
        round_bf16(ideal - error),
        round_bf16(ideal + error),
    )


def compare_gemm(
    candidate: np.ndarray, reference: GemmReference, *, output_dtype: Literal["fp32", "bf16"]
) -> dict:
    candidate = np.asarray(candidate, dtype=np.float32)
    if candidate.shape != reference.ideal_scaled_fp64.shape:
        raise ValueError("Candidate shape differs from the oracle")
    finite = bool(np.isfinite(candidate).all())
    if output_dtype == "bf16":
        representable = bool(np.array_equal(candidate, round_bf16(candidate)))
        low, high = reference.bf16_lower, reference.bf16_upper
        expected = reference.expected_bf16.astype(np.float64)
    elif output_dtype == "fp32":
        representable = True
        low = reference.ideal_scaled_fp64 - reference.fp32_error_bound
        high = reference.ideal_scaled_fp64 + reference.fp32_error_bound
        expected = reference.expected_fp32.astype(np.float64)
    else:
        raise ValueError("Expected fp32 or bf16 output dtype")
    outside = (candidate < low) | (candidate > high) | ~np.isfinite(candidate)
    delta = candidate.astype(np.float64) - expected
    rmse = float(np.sqrt(np.mean(delta**2))) if finite else None
    scale = float(np.sqrt(np.mean(expected**2)))
    # BF16 values use exactly representable F32 storage here. Comparing those
    # storage bits retains the BF16 sign of zero; floating == does not.
    expected_f32 = expected.astype(np.float32)
    output_bits_equal = candidate.view(np.uint32) == expected_f32.view(np.uint32)
    output_bit_equal_fraction = float(np.mean(output_bits_equal))
    return {
        "conditional_envelope_pass": finite and representable and not bool(outside.any()),
        "native_kernel_qualified": False,
        "finite": finite,
        "declared_dtype_representable": representable,
        "outside_analytical_envelope": int(outside.sum()),
        "total_elements": candidate.size,
        "max_absolute_error": float(np.max(np.abs(delta))) if finite else None,
        "rmse": rmse,
        "normalized_rmse": rmse / scale if finite and scale else (0.0 if rmse == 0 else None),
        "output_bit_equal_fraction": output_bit_equal_fraction,
        "bf16_bit_equal_fraction": output_bit_equal_fraction if output_dtype == "bf16" else None,
        "value_equal_fraction": float(np.mean(candidate == expected)),
        "signed_zero_differences": int(np.count_nonzero((candidate == 0) & (expected == 0) & ~output_bits_equal)),
        "bound_contract": "conditional IEEE RNE FP32 accumulate/alpha/output cast diagnostic",
        "qualification_blocker": "PTX E2M1 MMA rounding/subnormal behavior is unspecified; kernel evidence required",
    }


def projection_alpha(weight_global: float, input_dequant_global: float) -> float:
    w = _positive_f32(weight_global, "weight_global")
    a = _positive_f32(input_dequant_global, "input_dequant_global")
    return float(np.float32(w * a))


@dataclass(frozen=True)
class ExpertCastProfile:
    """Explicit experimental cast profile. Verify actual CUDA stages before gating."""
    fc1_output: Literal["fp32", "bf16"]
    gelu_output: Literal["fp32", "bf16"]
    product_output: Literal["fp32", "bf16"]
    fc2_output: Literal["fp32", "bf16"]
    source_evidence: str


def _cast(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return round_bf16(value)
    if dtype == "fp32":
        return np.asarray(value, dtype=np.float32)
    raise ValueError("Unknown reference cast dtype")


def gelu_tanh_semantic_fp64(value: np.ndarray) -> np.ndarray:
    """Mathematical semantic reference; libtanh/FMA error needs separate qualification."""
    x = np.asarray(value, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def expert_semantic_reference(
    input_activation: QuantizedActivation,
    gate: tuple[np.ndarray, np.ndarray, float],
    up: tuple[np.ndarray, np.ndarray, float],
    down: tuple[np.ndarray, np.ndarray, float],
    *,
    input_dequant_global: float,
    down_input_dequant_global: float,
    cast_profile: ExpertCastProfile,
    quantizer_profile: Literal["vllm_python_rn", "flashinfer_strict_rn"],
) -> dict:
    """Bounded expert stage traces. This alone does not qualify CUDA GELU or QDQ."""
    if not cast_profile.source_evidence:
        raise ValueError("Record the explicit experimental cast profile provenance")
    stages = {}
    for name, projection in (("gate", gate), ("up", up)):
        packed, sf, global_w = projection
        ref = gemm_reference(
            input_activation.packed,
            input_activation.linear_sf,
            packed,
            sf,
            alpha_fp32=projection_alpha(global_w, input_dequant_global),
        )
        stages[name + "_reference"] = ref
        stages[name] = _cast(ref.expected_fp32, cast_profile.fc1_output)
    stages["gelu_math_fp64"] = gelu_tanh_semantic_fp64(stages["gate"])
    stages["gelu"] = _cast(stages["gelu_math_fp64"], cast_profile.gelu_output)
    stages["product"] = _cast(
        stages["gelu"].astype(np.float32) * stages["up"].astype(np.float32),
        cast_profile.product_output,
    )
    a2 = quantize_activation_rn(
        stages["product"],
        float(np.float32(1.0 / _positive_f32(down_input_dequant_global, "down_input_scale"))),
        profile=quantizer_profile,
    )
    stages["a2"] = a2
    stages["down_reference"] = gemm_reference(
        a2.packed,
        a2.linear_sf,
        down[0],
        down[1],
        alpha_fp32=projection_alpha(down[2], down_input_dequant_global),
    )
    stages["down"] = _cast(stages["down_reference"].expected_fp32, cast_profile.fc2_output)
    stages["qualification"] = "experimental cast profile; CUDA GELU and quantizer unqualified"
    return stages


class CheckpointReader:
    """Read-only, bounded safetensors access without whole-model materialization."""

    def __init__(self, checkpoint_dir: Path):
        self.entries = {}
        for path in sorted(Path(checkpoint_dir).glob("*.safetensors")):
            with path.open("rb") as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError("Short safetensors header")
                size = struct.unpack("<Q", raw)[0]
                if size > 64 << 20:
                    raise ValueError("Header exceeds the bounded 64 MiB budget")
                header = json.loads(stream.read(size))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if name in self.entries:
                    raise ValueError("Duplicate tensor name across shards")
                self.entries[name] = (path, 8 + size, entry)

    def tensor(self, name: str, *, max_bytes: int = 32 << 20) -> tuple[np.ndarray, str]:
        path, base, entry = self.entries[name]
        start, end = entry["data_offsets"]
        size = end - start
        dtype = entry["dtype"]
        if start < 0 or end < start or size > max_bytes:
            raise ValueError("Tensor offset or requested memory budget is invalid")
        itemsize = 4 if dtype == "F32" else 1
        if dtype not in ("U8", "F8_E4M3", "F32"):
            raise ValueError("This bounded reader only accepts NVFP4 expert tensors")
        if math.prod(entry["shape"]) * itemsize != size or base + end > path.stat().st_size:
            raise ValueError("Tensor extent does not match its dtype/shape or file size")
        with path.open("rb") as stream:
            stream.seek(base + start)
            raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("Short safetensors payload")
        out = np.frombuffer(raw, dtype="<f4" if dtype == "F32" else np.uint8)
        return out.reshape(entry["shape"]), hashlib.sha256(raw).hexdigest()

    def projection(self, prefix: str) -> tuple[tuple[np.ndarray, np.ndarray, float], dict]:
        for suffix, expected_dtype in (("weight", "U8"), ("weight_scale", "F8_E4M3"), ("weight_scale_2", "F32")):
            if self.entries[prefix + "." + suffix][2]["dtype"] != expected_dtype:
                raise ValueError("Projection tensor dtype differs from the frozen NVFP4 format")
        packed, ph = self.tensor(prefix + ".weight")
        scales, sh = self.tensor(prefix + ".weight_scale")
        global_w, gh = self.tensor(prefix + ".weight_scale_2")
        if global_w.shape != ():
            raise ValueError("Expected a scalar projection global")
        # Validate shapes/scales without expanding the full matrix.
        decode_qsf(packed[:1], scales[:1])
        return (packed, scales, float(global_w)), {
            "weight_sha256": ph,
            "weight_scale_sha256": sh,
            "weight_scale_2_sha256": gh,
        }


def exhaustive_qsf_bf16_summary() -> dict:
    products = E2M1_TABLE[:, None] * E4M3_TABLE[None, :]
    valid = np.isfinite(products)
    values = products[valid]
    exact = values == round_bf16(values).astype(np.float64)
    return {
        "finite_products": int(valid.sum()),
        "products_not_exact_in_bf16": int((~exact).sum()),
        "largest_absolute_product": float(np.max(np.abs(values))),
        "smallest_nonzero_absolute_product": float(np.min(np.abs(values[values != 0]))),
        "scope": "format algebra only; no CUDA or checkpoint/model qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-summary", action="store_true")
    parser.add_argument("--gemm-case", type=Path, help="NPZ with packed A/W, linear SF, alpha, candidate")
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.self_test_summary:
        print(json.dumps(exhaustive_qsf_bf16_summary(), indent=2))
    elif args.gemm_case:
        if args.gemm_case.stat().st_size > 256 << 20:
            raise ValueError("Case exceeds the 256 MiB input budget")
        with np.load(args.gemm_case, allow_pickle=False) as case:
            ref = gemm_reference(
                case["a_packed"], case["a_sf"], case["w_packed"], case["w_sf"],
                alpha_fp32=float(case["alpha_fp32"]),
            )
            report = compare_gemm(case["candidate"], ref, output_dtype=args.output_dtype)
            print(json.dumps(report, indent=2))
    else:
        parser.error("Choose --self-test-summary or --gemm-case")


if __name__ == "__main__":
    main()

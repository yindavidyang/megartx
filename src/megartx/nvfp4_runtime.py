"""Experimental, version-pinned separate-projection NVFP4 expert adapter.

The original packed weights, E4M3 block scales and F32 projection globals are
retained. Native FlashInfer CUTLASS GEMMs use separate gate/up alphas. This is
not a checkpoint conversion, a bit-exact fused reduction, or an acceptance
claim. It uses the selected runtime's layerwide activation quantizer.
"""

from dataclasses import dataclass


@dataclass
class Projection:
    packed: object
    scales: object
    swizzled: object
    global_scale: object


@dataclass
class Expert:
    index: int
    gate: Projection
    up: Projection
    down: Projection
    a1: object
    a2: object


def make_projection(packed, scales, global_scale):
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    # Clone before the stock loader reorders [gate; up] in-place. Swizzle each
    # projection separately: a 704-row split lies inside a 128-row scale tile.
    p, s, g = packed.detach().clone().contiguous(), scales.detach().clone().contiguous(), global_scale.detach().clone().reshape(1)
    return Projection(p, s, swizzle_blockscale(s), g)


def capture_experts(layer):
    import torch

    if layer.w13_weight_scale_2.ndim != 2 or layer.w13_weight_scale_2.shape[1] != 2:
        raise RuntimeError("Separate-projection adapter requires original gate/up globals")
    globals_ = layer.w13_weight_scale_2
    indices = torch.nonzero(globals_[:, 0] != globals_[:, 1]).flatten().tolist()
    if not indices:
        return []
    if tuple(layer.w13_weight.shape[1:]) != (1408, 1408) or tuple(layer.w2_weight.shape[1:]) != (2816, 352):
        raise RuntimeError("Adapter shape contract is Gemma-4-26B-A4B, hidden=2816, expert=704")
    if torch.any(globals_ <= 0) or not torch.isfinite(globals_).all():
        raise RuntimeError("NVFP4 global scales must be positive and finite")
    # Match the official pinned runtime's amax reduction; do not silently switch
    # from shared calibration to the checkpoint's per-expert activation globals.
    a1 = layer.w13_input_scale.max().detach().clone().reshape(1)
    a2 = layer.w2_input_scale.max().detach().clone().reshape(1)
    result = []
    for e in indices:
        result.append(Expert(e,
            make_projection(layer.w13_weight[e, :704], layer.w13_weight_scale[e, :704], globals_[e, 0]),
            make_projection(layer.w13_weight[e, 704:], layer.w13_weight_scale[e, 704:], globals_[e, 1]),
            make_projection(layer.w2_weight[e], layer.w2_weight_scale[e], layer.w2_weight_scale_2[e]),
            a1, a2))
    return result


def quantize(x, dequant_global):
    import flashinfer

    return flashinfer.nvfp4_quantize(x.contiguous(), 1.0 / dequant_global,
        sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False, backend="cuda")


def mm_native(q, sf, projection, activation_global):
    import torch
    import flashinfer

    return flashinfer.mm_fp4(q, projection.packed.T, sf, projection.swizzled.T,
        (projection.global_scale * activation_global).reshape(1),
        out_dtype=torch.bfloat16, backend="cutlass", block_size=16, use_nvfp4=True)


def unswizzle(scales, rows, columns):
    """Inverse of the documented 128x4 scale permutation, including tails."""
    padded_rows, padded_cols = ((rows + 127) // 128) * 128, ((columns + 3) // 4) * 4
    return scales.reshape(padded_rows // 128, padded_cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(padded_rows, padded_cols)[:rows, :columns].contiguous()


def decode_units(packed, linear_scales):
    """E2M1 * E4M3 only; globals are applied after the dot product.

    These products fit BF16 exactly. Folding a small F32 global into BF16
    weights first would introduce an avoidable, different rounding lane.
    """
    import torch

    if linear_scales.dtype == torch.uint8:
        linear_scales = linear_scales.view(torch.float8_e4m3fn)
    if linear_scales.dtype != torch.float8_e4m3fn:
        raise RuntimeError("NVFP4 scale storage must be E4M3 or its raw uint8 bytes")
    code = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device=packed.device)
    return lut[code] * linear_scales.double().repeat_interleave(16, dim=1)


def mm_reference(q, sf, projection, activation_global):
    import torch

    # Float64 accumulation on a bounded expert fixture, independently decoded
    # original weights. No global-scale folding, re-quantization or FP4 GEMM.
    aq = decode_units(q, unswizzle(sf, q.shape[0], q.shape[1] * 2 // 16))
    wq = decode_units(projection.packed, projection.scales)
    alpha = (projection.global_scale * activation_global).float()
    # Native accumulation/epilogue is F32. Make both rounding boundaries
    # explicit; a direct FP64-alpha/BF16 cast can differ at a BF16 tie.
    return ((aq @ wq.T).float() * alpha).to(torch.bfloat16)


def activation_semantic(gate, up):
    import torch

    # CUTLASS GELU_taylor<float>, followed by F32 up multiplication, then ONE
    # BF16 cast. BF16 F.gelu followed by BF16 multiply has another cast.
    g = gate.float()
    h = .5 * g * (1 + torch.tanh(0.7978845608028654 * (g + .044715 * g * g * g)))
    return (h * up.float()).to(torch.bfloat16)


def run_expert(x, expert, mode="native", return_stages=False):
    mm = {"native": mm_native, "reference": mm_reference}[mode]
    q1, sf1 = quantize(x, expert.a1)
    gate = mm(q1, sf1, expert.gate, expert.a1)
    up = mm(q1, sf1, expert.up, expert.a1)
    from .nvfp4_activation import cutlass_gelu_product
    h = cutlass_gelu_product(gate, up) if mode == "native" else activation_semantic(gate, up)
    q2, sf2 = quantize(h, expert.a2)
    down = mm(q2, sf2, expert.down, expert.a2)
    if return_stages:
        return down, {"q1": q1, "sf1": sf1, "gate": gate, "up": up, "activation": h, "q2": q2, "sf2": sf2, "down": down}
    return down

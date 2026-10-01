"""Small opt-in adapter for the pinned CUTLASS GELU-tanh operation sequence.

Source order is reviewed in docs/nvfp4-scale-correction.md. This does not define
a new approximation: it uses the same F32 instructions and constants as the
selected fused expert path, with one final BF16 product cast.
"""
import triton
import triton.language as tl


@triton.jit
def _gelu_product(gate, up, output, count: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    g = tl.load(gate + idx, idx < count, 0).to(tl.float32)
    u = tl.load(up + idx, idx < count, 0).to(tl.float32)
    h = tl.inline_asm_elementwise(
        """{
        // MEGARTX_GELU_PINNED_BEGIN
        .reg .f32 v0, v1, v2, v3, v4, v5;
        mul.rn.f32 v0, $3, $1;
        fma.rn.f32 v1, v0, $1, $4;
        mul.rn.f32 v2, $1, v1;
        tanh.approx.f32 v3, v2;
        fma.rn.f32 v4, $1, v3, $1;
        mul.rn.f32 v5, v4, 0f3F000000;
        mul.rn.f32 $0, v5, $2;
        // MEGARTX_GELU_PINNED_END
        }""",
        constraints="=f,f,f,f,f",
        args=[g, u, 0.035677406936883926, 0.7978845834732056],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(output + idx, h, idx < count)


def cutlass_gelu_product(gate, up, *, return_compiled=False):
    import torch

    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16 or gate.shape != up.shape:
        raise ValueError("Matched contiguous BF16 gate/up operands required")
    if not gate.is_contiguous() or not up.is_contiguous() or not gate.is_cuda or not up.is_cuda:
        raise ValueError("Matched contiguous CUDA operands required")
    out = torch.empty_like(gate)
    compiled = _gelu_product[(triton.cdiv(gate.numel(), 256),)](gate, up, out, gate.numel(), 256)
    return (out, compiled) if return_compiled else out

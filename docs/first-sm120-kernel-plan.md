# Conditional first SM120 kernel milestone

Prepare a single-token NVFP4 MoE preparation fusion: expert maps, byte-exact
packed-input/scale expansion and grouped-GEMM descriptor setup. Retain the
activation quantizer, expert GEMMs, original separate-global corrections,
GELU, expert combine, attention, shared branch, head and sampler. This is a
bounded integer/byte integration exercise, pending the gates below; no kernel
implementation or GPU benchmark is part of this plan.

The source planning task used repository snapshot
`fd088a32ef1314acd9143e8e247111fc01e35969`, before merged main's signed-zero
metric clarification. Its [sanitized CPU audit](evidence/first-sm120-kernel-planning.json)
preserves the original plan, trace and audit hashes. The current
[layer-0 diagnosis](layer0-arithmetic-diagnosis.md) advances localization but
does not open native, full/cached, quantizer, natural-quality or timing gates.
The [master plan](master-plan.md) and [measurement protocol](protocols/measurement.md)
remain authoritative for subsequent qualification.

## Exploratory budget and priority

The CPU audit separates eight prefill annotations from 31 decode annotations
in one old 2048-prompt / 32-output profiling request. It includes the head and
sampling suffix in the decode denominator. These are summed profiled kernel
durations, not ITL or recoverable critical-path time:

| Region | us / decode | Whole-decode kernel-sum share |
| --- | ---: | ---: |
| BF16 QKV, inferred from source/order/grids | 1004.553 | 17.77% |
| BF16 output projections, similarly inferred | 580.558 | 10.27% |
| Vocabulary head | 933.212 | 16.51% |
| Selected maps / expansion / descriptors | 227.909 | 4.03% |
| All decode kernels, including head / suffix | 5653.994 | 100% |

Completely eliminating the selected preparation region would yield about
1.042x on this kernel-sum ledger. The replacement has a cost, the old trace
uses graphs, and kernel durations overlap by about 82.092 us/decode. Thus this
starter cannot itself supply the proposed 15% end-to-end improvement. Maps
plus expansion alone occupy 160.655 us/decode, about 2.84%, as a narrower
future ablation.

The old loader discarded six up globals and the old graph trace is not a
corrected timing baseline. The corrected adapter currently requires eager
execution. Do not compare corrected eager A against uncorrected graph B or
credit graph enablement to this preparation kernel. BF16 QKV/output projection
qualification is the next profiling priority, followed by the vocabulary head
and shared BF16 branch. Runtime shapes were not recorded in the old profile;
its role assignments remain explicit inferences.

## Proposed semantic boundary

Version one would support Gemma's TP1/EP1 `M=1, H=2816, E=128, top_k=8,
F=704` geometry with eight distinct expert IDs. Preserve their router slot
order, exact packed E2M1 bytes, E4M3 scale bytes and FP32 route-weight bits.
Use caller-owned, address-stable storage. Prepare both incumbent 128-problem
lists, with eight M=1 problems and 120 M=0 problems, preserving actual FC1/FC2
`swap_ab`, scale layouts, strides, alpha pointers and tactics.

For input IDs `d[j]`, define rank `r[j]=count_i(d[i]<d[j])` and expert offsets
`o[e]=count_j(d[j]<e)`. The two maps must be exact inverses; offsets must start
at 0, end at 8 and increment by 0 or 1. Every expanded packed row must equal
the original row byte for byte. Valid scale coordinates require the existing
independent `sf_offset_128x4` oracle and the actual grouped expert-base address;
do not infer physical contiguity from a gate/up split or invent binary structs.
No floating-point cast, scale folding, requantization or extra weight padding
belongs inside this proposed operation.

The reference ABI's common prequantized activation lane requires swizzled
input SF, no per-expert activation quantization and no min-latency specialization.
If numerical qualification changes that lane, reject this candidate instead
of claiming that copying one activation row represents per-expert calibration.
The correction path for all six unequal globals must remain active.

The planner resolved upstream FlashInfer `v0.6.18.post1` to
[`8bc3b578027791336c6ae87db5c9d76f82cef8bc`](https://github.com/flashinfer-ai/flashinfer/commit/8bc3b578027791336c6ae87db5c9d76f82cef8bc).
Its [public interface](https://github.com/flashinfer-ai/flashinfer/blob/8bc3b578027791336c6ae87db5c9d76f82cef8bc/flashinfer/fused_moe/core.py)
and [C++ preparation runner](https://github.com/flashinfer-ai/flashinfer/blob/8bc3b578027791336c6ae87db5c9d76f82cef8bc/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh)
are reference inputs. Equality with installed C++ sources, generated tactics,
CUTLASS submodule and binary layouts is unverified. A package version alone
does not establish that identity or a supported insertion point.

## Gates before implementation

| Gate | Concrete evidence | Current status |
| --- | --- | --- |
| Installed ABI / source | Source and generated-module hashes, compiler flags, tactic IDs, typed descriptor sizes/offsets, actual pointers/strides, SF extents, scratch owners/lifetimes and consumer-read masks | Missing |
| Numerical baseline | Actual QKV/norm/attention contract, full/cached disposition, activation lane, natural coverage and G0/G1 decision | Open |
| Execution mode | Corrected graph A/B qualification or separately authorized exploratory operator scope; narrow C++ build/insertion point | Missing |
| Exact preparation fixture | Captured incumbent inputs/outputs with independently computed maps, offsets, byte copies, scales and active descriptors | Future |
| Implementation scope | Approval of the bounded kernel/build work after the preceding evidence | Not supplied by this plan |

Only after those gates, a possible initial prototype uses eight ordinary CTAs,
128 threads each. Each CTA computes ranks from all eight IDs, owns one expanded
row and a disjoint descriptor range; one CTA writes the final offset. No CTA
reads another CTA's output, and incumbent FC1 follows normal same-stream
completion. No grid barrier, global spin loop, atomics, persistent scheduler,
TMEM or speculative next-layer staging is proposed. Freeze actual compiled
resources and matched PDL policy before testing. Test maps-plus-expansion as
the narrower alternative if typed descriptor fusion is costly.

Retain the incumbent path until exact consumer-visible state, sanitizers,
resource usage and matched integrated gains justify a replacement. No gate,
implementation authorization, benchmark or broader WP3 scope is granted by
this planning record.

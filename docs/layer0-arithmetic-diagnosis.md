# Layer-0 full/cached arithmetic diagnosis

The frozen 33-input fixture first differs at the **raw layer-0 QKV projection
of position 32**, after identical incoming and input-RMS rows. An isolated
original-weight operator replay reproduces both outputs exactly. M1 decode and
M33 full execution select different cuBLAS kernels and retain the same 18
adjacent-BF16 differences across three repeats per shape. This supports a
repeatable shape-dependent arithmetic explanation for this boundary. It does
not establish a healthy whole-model tolerance or an unconditional native
accumulation contract.

The complete common 32-row layer-0 cache prefix agrees exactly between full
and cached execution. Each path's actual writer inputs agree with stored rows;
the cached decode write preserves its entire 32-row prefix. The final row's
three K and three V differences already exist before the writer. These checks
exclude a writer-copy or prefix-retention mismatch for this bounded fixture.
They do not qualify the independent scheduler, window wrapping, other layers'
entire caches or longer contexts.

This follow-on starts from merged main
`8b81aa2c65ec971159b8e19e8a23b2eacf7a74e8`. Both exact-main CPU workflows passed:
[reference](https://github.com/yindavidyang/megartx/actions/runs/36838820294) and
[scaffold](https://github.com/yindavidyang/megartx/actions/runs/36838820281).
The original [controlled integration evidence](controlled-scale-integration.md)
and its 90 retained original-weight projection matches remain unchanged.
New [sanitized scalar evidence](evidence/layer0-arithmetic-live.json) separates
this diagnosis from those historical runs.

## Fixed scope and observed boundaries

All new model requests use the existing token/route fixture, original checkpoint
revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, pinned runtime, BF16
compute/KV, one sequence, eager execution and no prefix caching or speculation.
There are five new owned lifecycles: one M33 full and four M32-prefill/M1-decode
cached captures. Two cached captures form a fixed repeat pair; two use the same
latest observer source with synchronous versus deferred downstream copying.
No prompts, contexts, training, kernels or benchmarks were added.

The observer retains positions 31/32 through input RMS, QKV, head RMS, RoPE,
Attention, output projection, post-attention RMS, BF16 residual addition and
the norm feeding the corrected experts. It additionally retains only layer
0's frozen 33-row writer/cache prefix and one bounded loaded K partition.
Loaded Q/K/V hashes match their distinct original checkpoint tensors. Local
layer-0 K/V are separate weights despite the global projection-sharing option.

Position 31 agrees at every retained boundary. Position 32 differs as follows:

| Boundary | Different BF16 values | Maximum absolute difference |
| --- | ---: | ---: |
| Incoming / input-RMS output / QKV input | 0 / 2816 | 0 |
| Raw Q | 11 / 4096 | 8 |
| Raw K | 4 / 2048 | 0.03125 |
| Raw V | 3 / 2048 | 0.0078125 |
| Q after head RMS | 252 / 4096 | 0.0625 |
| K after RoPE / writer / stored row | 3 / 2048 | 0.000091552734375 |
| V after head RMS / writer / stored row | 3 / 2048 | 0.00048828125 |
| Attention output | 406 / 4096 | See scalar evidence |
| Output projection | 826 / 2816 | See scalar evidence |
| Post-attention RMS | 740 / 2816 | See scalar evidence |
| Residual entering the MoE input norm | 408 / 2816 | See scalar evidence |
| MoE input norm / actual expert-82 input | 393 / 2816 | 0.0078125 |

RoPE cache bits agree. The independent CPU reader checks raw associations,
including output-projection boundaries, BF16 residual addition and the actual
corrected-expert input. It rejects rehashed stale expert rows, altered residuals,
cache-prefix corruption and aliased addresses rather than trusting capture
booleans.

## Operator replay and defensible contract

The operator-only probe reads the original 33 embedding rows, input-RMS weight
and bounded original Q/K/V tensors. It reconstructs the source's BF16 embedding
scalar multiplication and pinned RMS CUDA call, requiring retained incoming
and normalized rows to match before proceeding. It uses actual M1/M33
geometry; three repetitions and all-zero / exact-one-term controls are fixed
before execution. Both controls pass. Original M1 and M33 results reproduce
the corresponding captured rows in raw bits.

Named observer/replay spans bind exactly one CPU `aten::mm` to its CUDA launches
through the profiler's external ID. Decode QKV uses `internal::gemvx` with
grid `[2048,1,1]`; full QKV uses the CUTLASS BF16 `64x64_32x10` tensor-op
family with grid `[8,16,1]`. M32 prefill uses the BF16 WMMA `32x32_128x2`
family. Output-projection prefill/full also includes an explicit split-K
reduction; M1 uses GEMV. Kernel symbols establish dispatch association, not
all arithmetic details.

A separate process-local probe disables BF16 reduced-precision reduction.
Kernel selection and all retained outputs remain unchanged, including the 18
M1/M33 differences. The original flag is restored; the model/runtime setting
was never changed. This experiment does not identify reduced-precision
partials as the cause.

The NumPy-only oracle computes FP64 original-weight dots and sum-absolute
products. Its interval is derived analytically from gamma-K for F32 RNE,
including FP64 dot/sum uncertainty and outward F32 then BF16 endpoint rounding.
All retained Q/K/V outputs fit that **conditional** interval. It assumes full
BF16 products, F32 RNE accumulation without BF16 partial sums, no overflow and
no FTZ-sensitive intermediates. No observed error sets the interval.

An additional CPU audit sums the 18 differing dots as exact dyadic rational
numbers. M1 matches correctly rounded exact dots at 17 coordinates, M33 at one.
All candidate pairs are adjacent BF16 values. Distances from their midpoint
range up to about 0.252 BF16 spacing; some cancellation-sensitive cells are
farther from a midpoint. This is a description of observed coordinates, not
an acceptance threshold, and a native F32 reduction need not produce the
correctly rounded exact dot.

The [NVIDIA PTX contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions)
does not fix BF16 tensor-MMA accumulation order, rounding or subnormal handling.
General [PyTorch batch/slice guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations)
also does not promise bit identity; that documentation is for 2.14, while the
unchanged installed runtime is 2.13. These sources explain the limits of the
conditional diagnostic. They do not certify this installed kernel's RNE
assumptions. Norm/RoPE rounding and end-to-end sensitivity remain separate.

The current contract therefore requires exact provenance, row/route identity,
original payloads, copy/address associations and the declared operator profile.
Unfitted conditional intervals can diagnose arithmetic but cannot promote
native-MMA, full/cached, scheduler, quantizer, natural quality or timing gates.
Bit-stable repetitions of one fixture do not establish a universal tolerance.

## Cached repeatability and the remaining anomaly

All four new cached captures agree in every retained boundary, all six retained
expert records, all 30 selected K/V records and all three matched head-call
variants. They reproduce the original v14 retained evidence exactly. The two
capture-placement modes also agree, including the entire layer-0 prefix.
Deferred mode preserves tensor objects and copies added downstream/prefix
observations only after decoder 0 finishes. Both modes alter lifetimes or
synchronization and provide no timing or observer-free repeatability claim.

The historical v3 cached observer run remains different: its layer-0 expert-82
input differs in 156 values and its decode head in 258,043 raw F32 values,
despite exact retained layer-0 K/V and position-31 controls. That older run did
not retain attention/output-projection outputs or the complete prefix. The new
capture-placement comparison **did not reproduce or explain it**. Keep it as
an unresolved anomaly, not an exact replica of v14 or these fresh runs. No
tolerance or native defect conclusion follows from it.

The next work is to qualify the actual M1 GEMV and M33 MMA reduction/cast
contract from the pinned provider implementation, then source-match RMS,
RoPE and attention sensitivity against an independent arithmetic model. A
future, explicitly bounded replay intended to reproduce the historical cached
anomaly must retain the attention/output-projection boundary and actual
algorithm/workspace metadata. Do not broaden prompts or use timing to accept
the current results. The [conditional first-kernel plan](first-sm120-kernel-plan.md)
remains planning only.

## Verification and private replay

The reference suite passed 335 tests locally and in the target's pinned
NumPy 2.3.5 environment before the last GPU pair; the scaffold has 32 tests.
Final exact-head CI is reported with the draft PR. All five newly owned process
groups are gone, all runs exited 0 and reached `cleanup_complete`, minimum free
GPU memory was 10,795 MiB, and final state was 41 MiB used / 32,101 MiB free
with no compute PIDs. Both complete original checkpoint shard hashes still
match. Installed runtime versions and pinned Python sources were rechecked.

Subsequent CPU review found that the full/cached checker omitted the cached
Q-weight hash. The [provenance guard revalidation](evidence/layer0-provenance-guard.md)
records the narrow repair, six added regression tests and an independent reread
of the immutable retained pair. Both paths still bind to all three original
Q/K/V tensors, and the first discrepancy remains the same 18 raw QKV values.
The repaired suite passes 341 reference tests; acceptance gates remain closed.

The observer is opt-in through `--layer0-boundaries` on the native controlled
full/cached lifecycle; other paths are rejected. Its optional
`--layer0-capture-policy deferred_downstream` requires that flag. CPU commands
read immutable completed evidence and exclusively create a new report outside
every input directory:

```bash
python numerical_reference/layer0_reference.py --plan "$PLAN" \
  --full "$FULL" --cached "$CACHED" --checkpoint "$CHECKPOINT" \
  --output "$NEW_PRIVATE_REPORT"

python numerical_reference/layer0_reference.py --plan "$PLAN" \
  --cached "$CACHED_A" --cached-repeat "$CACHED_B" \
  --output "$NEW_PRIVATE_REPEAT_REPORT"
```

Use `--observation-ablation` only for an intentional comparison of different
capture-placement policies. CLI completion reports diagnostic execution; the
explicit acceptance gates remain false. The fixed
[`replay_layer0_qkv.py`](../scripts/replay_layer0_qkv.py) operator probe requires
the recorded GPU/runtime and a separate owned, bounded invocation. Raw tensors,
tokens, exact coordinate values, traces and logs remain in ignored private
artifacts. No checkpoint or raw activation payload is committed.

# Bounded NVFP4 numerical reference

The independent CPU reference and 25 golden tests establish format algebra and
a reproducible global-scale counterexample. Another 17 binding-guard tests and
49 artifact-guard tests reject hook bypass, invalidated evidence, missing native
trace events and zero natural coverage; their API stubs do not establish live
CUDA execution. These checks alone do not establish
CUDA, expert, full-model or quality correctness. The module reads bounded
tensors and runs CPU arithmetic without changing the checkpoint. Target GPU
experiments and their narrower scope are documented in the
[scale-correction report](../docs/nvfp4-scale-correction.md).

## Run the independent CPU checks

    python3 -m unittest discover -s numerical_reference -v
    python3 numerical_reference/nvfp4_reference.py --self-test-summary

All 25 tests also passed in the isolated target environment with NumPy 2.3.5
and Python 3.12.3. The dedicated
[CPU workflow](../.github/workflows/numerical-reference.yml) pins NumPy 2.3.5 on
Python 3.12, separately from the scaffold's Python 3.10 job. Initial local
reference development used existing NumPy 2.4.4.
The module imports neither Torch, vLLM, nor FlashInfer. The
safetensors reader reads individual original expert tensors and hashes their
payloads. It limits a tensor read to 32 MiB, and never expands all experts.

`compare_gemm` reports output storage-bit equality separately from numerical
value equality and counts differing signs of zero. Its BF16-specific bit metric
is null for a declared F32 output. Earlier standalone reports used floating
value equality under the `bf16_bit_equal_fraction` name, so that historical
metric does not prove signed-zero equality. The controlled replay's raw uint16
comparisons and its 90 retained projection matches are unaffected. Conditional
arithmetic intervals remain separate from exact-bit diagnostics and native
qualification.

`layer0_reference.py` adds NumPy-only validation of the frozen full/cached
boundary, bounded original BF16 Q/K/V dots, copy/address associations and
cached repeats. Its gamma-K interval assumes F32 RNE accumulation and is never
an unconditional native-MMA or whole-model gate. Schema 2 checks the frozen
33-row layer-0 prefix, residual addition and actual corrected-expert input;
historical schema 1 remains readable with its narrower provenance. See the
[diagnosis, private replay commands and remaining limits](../docs/layer0-arithmetic-diagnosis.md).

## One-request router diagnostic

The independent `router_reference.py` and `check_router_capture.py` add 41
CPU tests for BF16 rounding/products, signed-bit top-eight ordering, bounded
FP64 projection diagnostics, score/weight profiles and original-payload
provenance. Fifteen additional observer guard tests reject changed prefixes,
request/row bounds and missing per-forward layers using small tensor API stubs.
The initial router/reference guard suite had 147 tests; the scaffold has 32
more. The controlled preparation below adds separate CPU checks. These counts
are CPU checks, not CUDA qualification.

Given the private output from the explicitly bounded score client, run:

```bash
python numerical_reference/check_router_capture.py "$PRIVATE_ROUTER_SCORE_DIR" \
  --output "$NEW_PRIVATE_CPU_REPORT"
```

The checker verifies file hashes, original tensor-payload hashes, actual token
positions and unchanged selected IDs before its arithmetic. It separately
reports all-row score/rank evidence and retained-input projection checks; it
never applies the eight-row matmul evidence to every score row. The inline root
product is reconstructed with separate BF16 stores. Ideal RMS and named F32
weight profiles remain semantic diagnostics; enabled BF16 partial/split-K
preferences keep the FP32 dot envelope conditional. No observed error is used
to fit a tolerance. Raw NPZ files contain weights/inputs and stay private. See
[the one-prefix evidence](../docs/evidence/nvfp4-router-scores.json) and the
[qualification limits](../docs/nvfp4-scale-correction.md#one-unchanged-prefix-router-score-diagnostic).

## Controlled integration checks

`controlled_reference.py` prepares a fixed 33-input, all-30-layer route table
and verifies complete actual ID/F32-weight correspondence by absolute position.
Its captured-operand replay uses original bounded checkpoint projections and
the independent NumPy format/GEMM oracle. It compares raw full-vocabulary
logits and selected logical BF16 K/V with explicit token/table/row provenance.
Synthetic tests reject missing decode rows, altered ordinary routes, duplicate
positions, wrong alpha and incomplete execution records. These are preparation
and artifact checks; they do not establish live execution by themselves.

The live collector and independent `compare_controlled_capture.py` now read
hash-bound actual routes/stages, dedicated-span trace correlations, original
projection payloads and calibration scalars, weighted/routed outputs, raw
full-vocabulary head-call variants and populated logical K/V rows. The separate
`compare_controlled_paths.py` binds fresh per-path run directories to their own
proofs, rejects changed metadata and unknown pinned layout enums, and reports
incomplete negative matrices explicitly. Both checkers are CPU-only and never
launch a server. Reports are exclusively created outside all input directories.
Exit 0 refers to the retained operator gate, while the explicit handoff,
matrix-completeness and quality fields remain separate acceptance conditions.

The completed five-pass fixture matches 90 observed projection replays and
positive within-path stages/logits/KV. Full/cache handoff fails upstream of the
position-32 correction. Raw head variants remain separate; no tolerance is
fitted, and native MMA, frozen quantizer, independent attention/scheduler,
natural quality and timing acceptance remain false. The final local suite has
312 reference/guard tests plus 32 scaffold tests. See the
[live evidence and stopping point](../docs/controlled-scale-integration.md).

Natural coverage additionally rejects a `CONTROLLED-ROUTING.json` marker or
explicit non-natural request, manifest or hit scope before report creation.
Positive artificial counters cannot satisfy the natural gate. Existing timing
refusals and numerical acceptance limits remain intact. The reviewed minimal
experiment, paired-reference limitations and direct-KV prerequisites are in
the [controlled integration design](../docs/controlled-scale-integration.md).

## Keep three numerical contracts distinct

| Contract | Packed weights and weight globals | Activation globals |
| --- | --- | --- |
| Frozen checkpoint oracle | Original separate gate/up/down bytes and scalars | Original scalar for each expert/projection |
| Selected-runtime oracle | Original separate gate/up/down bytes and scalars | vLLM/FlashInfer layer maximum for FC1 and FC2 |
| Old exploratory control | Original bytes; gate global used for both FC1 halves | vLLM/FlashInfer layer maximum |

The corrected adapter must first agree with the selected-runtime oracle. That
agreement isolates the six weight-scale fixes and their integration. It does not
prove that sharing activation globals preserves the frozen checkpoint quantizer.
Use the frozen checkpoint oracle for the second comparison.

The [vLLM 0.30.0 conversion helper](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py#L331-L340)
collapses original FC1 and FC2 activation globals across the layer and repeats
each maximum across experts. Its
[maximum helper](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/quant_utils.py#L65-L82)
uses FP32 maximum, with an additional rank reduction only when EPLB is enabled.
Local scale-audit data has between 42 and 95 different FC1 input globals per
layer. The smallest source/global-maximum ratio reaches 0.02923 in layer 16.
All six mismatched experts happen to have FC1 globals equal to their layer
maximum. FC2 source-versus-runtime globals still require an independent audit.

## Canonical packing and GEMM

The low nibble represents logical even K; the high nibble represents odd K.
E2M1 sign is bit 3; magnitude codes 0 through 7 represent 0, 0.5, 1, 1.5,
2, 3, 4, and 6. Decode E4M3fn scales independently from exponent/fraction bits;
reject NaN or negative scale values. Weight interpretation is

    W[n,k] = E2M1(q[n,k]) * E4M3(sf[n,k//16]) * original_weight_global

These conventions are corroborated by the
[pinned vLLM emulation source](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py#L492-L569).
Our decoder and rounder use their own tables/bit arithmetic and import no
production helper. Coordinate offset fixtures independently check the
[128x4 scale layout](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py#L12-L49),
including both real projection shapes and padded rows.

Define U_A=q_A*sf_A and U_W=q_W*sf_W, without folding either global into a BF16
operand. Exhaustive enumeration of all 4,064 finite nibble/scale combinations
proves these operands are exactly representable in BF16. Their nonzero
magnitudes range from 2^-10 to 2688; operand-product precision fits FP32.
Thus a simple BF16 GEMM with FP32 accumulation on these unglobalized operands
can cross-check native FP4 multiplication without the added global-fold error.
The primary independent oracle computes the dot in FP64, by output-row chunks.

For the selected runtime, capture rather than reconstruct the exact alpha:

    quant_global_A1 = FP32(1 / original_layer_max_input_scale)
    alpha_gate = FP32(original_gate_global * layer_max_input_scale)
    alpha_up   = FP32(original_up_global   * layer_max_input_scale)
    alpha_down = FP32(original_down_global * layer_max_down_input_scale)

The inverse global used by the quantizer is already rounded in FP32. Its
reconstructed inverse need not recover the original small input scalar exactly.
Use the original small scalar when reproducing the epilogue alpha. Apply alpha
before the native GEMM output cast. Rescaling a previously rounded BF16 up
output introduces another rounding and cannot be called exact.

## GPU stage qualification sequence for the owner

1. Pin hashes of the actually installed Python/CUDA sources. The inspected
   official tags are useful evidence but do not substitute for the installed
   FlashInfer 0.6.18.post1 source hash.
2. Check original tensor bytes, low/high nibble roundtrip, scale swizzle
   roundtrip, separate globals, gate/up row order, padding and output slicing.
   Cover all six affected experts plus equal-global controls from the same
   layers and representative later layers.
3. Capture A1 packed bytes and linear block-scale bytes. Compare these to an
   independently implemented quantizer on zero, extreme, saturation, ordinary,
   and rounding-boundary fixtures. The supplied vllm_python_rn and
   flashinfer_strict_rn profiles implement different documented IEEE operation
   orders: amax*FP32(1/6) then global versus global*FP32(1/6) then amax.
   FlashInfer strict mode also distinguishes zero amax from a nonzero block
   with an underflowed scale byte. Neither pretends to implement CUDA
   approximate reciprocal/FTZ.
   Any unexplained mismatch remains a quantizer blocker.
4. Compare separate native gate and up GEMMs against captured-operand FP64
   references using separate original alphas. Use M=1,2,8 and the actual
   K=2816,N=704 geometry. Include exact single-term/power-of-two cases and
   cancellation, as well as realistic saved expert inputs.
5. Capture pre-GELU gate/up, GELU output, product, A2 codes/scales, unweighted
   FC2 output, weighted contribution, and combined routed result. Establish
   actual cast points from installed C++/CUDA and direct traces. The explicit
   ExpertCastProfile is experimental until that evidence exists. Mathematical
   tanh GELU in this module cannot by itself qualify CUDA fast tanh or FMA.
6. Run native FC2 against the same captured A2 operands and original down
   global, K=704,N=2816. Compare weighting and reduction before the routed
   branch norm. Adding corrected contributions after a BF16 fused reduction
   changes order/cast boundaries and needs its own bounded check.
7. Validate live routing and full-model teacher-forced traces only after these
   stages pass; then rerun performance with all adapter costs included.

The strict profile follows the explicitly rounded operations in
[FlashInfer's pinned FP4 helpers](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/cute_dsl/fp4_common.py#L1928-L1948)
and its [output-scale helper](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/cute_dsl/fp4_common.py#L943-L975).
Fixtures include the E4M3 midpoint where reversing the multiplication order
changes the scale byte, and a nonzero group with zero scale byte but retained
saturated FP4 codes. Matching tags remains conditional on installed-source
verification.

FlashInfer documents that
[default fused finalize uses non-associative atomics](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/fused_moe/core.py#L1276-L1280).
Use deterministic finalize consistently for stage fixtures when possible;
separately quantify repeatability of the actual performance path. Do not
expand tolerances to absorb an unexamined change in reduction semantics.

## Conditional diagnostic envelopes, frozen before comparisons

Packing, addressing, source weight bytes, original scale bytes/scalars, IDs in
non-tie routing fixtures, and quantizer codes/scales under the exact same
quantizer profile require exact agreement. No weight-rounding tolerance permits
E4M3 scale reconciliation to inherit the frozen-model lane.

For a fixed captured-operand GEMM, let S=sum(abs(U_A[k]*U_W[k])) and
gamma_K=K*u/(1-K*u), with u=2^-24 for FP32. The oracle's accumulator bound uses
gamma_K*S, plus FP64 oracle uncertainty. One alpha multiply adds its FP32
rounding bound. This bound is fixed analytically and does not inspect candidate
errors. The BF16 diagnostic interval is obtained by rounding its endpoints to
BF16; a declared BF16 output must also be BF16-representable. It assumes FP32
accumulation with IEEE round-to-nearest behavior, finite operands/epilogue and
no FTZ-sensitive intermediates. The
[PTX E2M1 MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
provides at least single precision accumulation while leaving order, rounding
and subnormal handling unspecified. Therefore this interval is conditional
diagnostic evidence, not a source-guaranteed SM120 correctness tolerance.
Do not promote an in-envelope result to native qualification without stronger
pinned kernel/epilogue evidence and independent stage controls. The report
explicitly sets native_kernel_qualified=false. Any changed bound needs a
prior numerical justification; candidate errors alone cannot justify widening.

Record maximum absolute error, normalized RMSE, exact BF16 fraction, rejected
coordinates, and systematic projection-scale ratios. Relative error near zero
is an unsuitable standalone gate. Exact toy fixtures should remain bit-exact.
The current synthetic tests accept all six correct up alphas and reject all six
gate-only alphas. This is a useful negative control, not an affected-weight test.

To compare one captured GEMM without invoking GPU code, provide an NPZ file
with a_packed, a_sf, w_packed, w_sf, alpha_fp32 and candidate arrays:

    python3 numerical_reference/nvfp4_reference.py --gemm-case case.npz --output-dtype bf16

## Bounded full-model comparisons

Use one resident quantized model and one owned process at a time. A reference
override should read original packed weights directly and evaluate experts one
at a time, or by a fixed small cache. Avoid expanding the whole model.

- A differential integration oracle can replace all selected experts in the
  five affected layers (0,1,2,3,5), using the selected-runtime A1/A2 contract and
  independently decoded weights. Replacing all selected contributions avoids
  a subtract-old/add-new cancellation and exposes the combine ordering. This
  validates the scale fix in the existing runtime, while unchanged attention,
  dense branches and other layers remain a shared dependency.
- A stronger quantizer comparison uses the same simple expert oracle at all
  30 layers, once with layer maxima and once with each original projection's
  input global. Shared runtime components stay fixed. This isolates the
  activation-quantizer change without downloading the original BF16 model.
- A six-expert-only override is useful as an early bounded differential check.
  It cannot establish full-model checkpoint-quantizer equivalence or independent
  correctness of untouched MoE operators.

Per expert, gate/up/down total 5,947,392 scalar values: 11.34375 MiB BF16,
22.6875 MiB FP32 or 45.375 MiB FP64. One full expert layer expanded in FP32 is
about 2.84 GiB; avoid retaining it unnecessarily. This reference decodes only
64 output rows at a time. Small M blocks and one expert keep transient storage
well below 256 MiB on either CPU or GPU in addition to the resident model.

For teacher forcing, both runs receive identical token IDs and prefixes.
Start with four 128-token sequences, then a fixed held-out set of at least
4,096 scored tokens if the early stage checks pass. Use a small fixed set of
2K and 8K prefixes with 32 to 64 forced suffix tokens; include positions
1023/1024/1025 and repeated window wraps. Trace the first divergence in norms,
router inputs/IDs/weights, selected expert outputs, combined branch, residuals
and final logits. Do not compare independent autoregressive continuations as
if their later inputs were identical.

The vocabulary has 262,144 entries: one FP32 logit row is 1 MiB. Compute/store
full distributions only for chosen positions or chunks of at most 64 rows.
Score ground-truth NLL using the complete softmax normalization, and compute
paired perplexity delta, per-position max logit difference, top-1 agreement,
top-2 margins and distribution divergence from full rows. Top-five sampled
log-probabilities cannot establish full-distribution agreement or perplexity.
Use held-out documents as paired uncertainty units rather than treating every
correlated token as an independent sample.

Project protocol thresholds are still proposed: 1% relative held-out
perplexity increase and one percentage point task-score loss. Resolve and
record those before acceptance measurements. A small test that cannot decide
those margins is preliminary. Original BF16 model quality is an additional
reference limitation; absence of that large model should not be disguised as
an equivalence claim.

## Existing alternatives and stopping points

The installed lock pins humming-kernels 0.1.12. Its
[weight converter](https://github.com/inclusionAI/humming/blob/v0.1.12/humming/schema/modelopt.py#L101-L118)
folds per-half globals into parameter-dtype block scales when they differ.
Its [default input schema](https://github.com/inclusionAI/humming/blob/v0.1.12/humming/schema/humming.py#L289-L310)
uses 16-bit activations, and vLLM derives the Humming input schema from an
environment config or an empty config. Humming therefore supplies a separately
labeled arithmetic/quality control, requiring its own operand checks, rather
than an automatic native W4A4 oracle.

Default vLLM EMULATION inherits the gate-only FC1 global from the
[ModelOpt loader](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/modelopt.py#L914-L943),
so it must preserve separate globals before it becomes an independent scale
control. Its fake activation QDQ also uses the layer maximum. A BF16 dequantized
expert control can test semantic plausibility but carries extra representation
rounding when globals are multiplied into weights before the BF16 cast.

Stop qualification at the first unresolved stage: source/hash disagreement;
packing/global loss; quantizer code mismatch; unverified GELU/cast point;
out-of-envelope FC1/FC2; combine mismatch; unexplained routing divergence; or
insufficient full-model quality evidence. Keep a corrected runtime result
explicitly preliminary if original-quantizer fidelity remains unestablished.

# Separate original-scale NVFP4 expert correction

This experiment addresses six unequal gate/up weight globals in the immutable NVIDIA Gemma-4-26B-A4B-NVFP4 checkpoint. The pinned vLLM loader previously discarded each distinct up global before FlashInfer CUTLASS packing. The adapter retains the original FP4 bytes, E4M3 block-scale bytes and all original F32 weight globals, and computes gate and up in separate native SM120 W4A4 GEMMs. It never rounds checkpoint scales into a common E4M3 global.

## Current evidence and blocker

The [sanitized scalar evidence](evidence/nvfp4-scale-fixtures.json) records 28 bounded expert cases and 84 projections. All 84 projection outputs agreed in BF16 value between native, Torch FP64/F32-epilogue and independent NumPy references; the wrong gate-only global produced expert relative L2 error up to 0.3140022. Seven intermediate payloads differed under zero block scales while their decoded effective values agreed. This is observed fixture agreement, not universal or signed-zero bit equality. The independent mathematical activation comparison differs in 12 of the 28 cases (maximum absolute difference 0.015625 and relative L2 5.6336e-05); GEMM agreement does not establish GELU equivalence. The separate 5,632-element boundary fixture has maximum mathematical activation difference 0.0001220703125; it is not a bound on every input.

The actual registered `MoERunner` binds the original `ModelOptNvFp4FusedMoE` method and `FlashInferExperts`. The adapter intercepts `RoutedExperts.forward_modular`, verifies its callable identity against the live forward registry captured during loading, and fails every model forward if any of the 30 adapters is bypassed. No modular-method replacement was observed. At startup, all 54 retained projection tensors match immutable checkpoint bytes. Six forced fixtures through the registered runner execute native and reference corrections, produce 2,816 nonzero output elements each and agree in BF16 value. The private forced trace contains 18 dense SM120 blockscaled GEMMs, six adapter GELU launches and six CPU correction spans in each mode. Forced sparse inputs and router factor 1 do not validate mixed-route accumulation.

Natural coverage is separate. A bounded diagnostic issued 20 requests across six languages, technical/creative domains and 64-token generation: 2,227 prompt tokens and 85 output tokens. The affected layers each recorded 91 client-bound calls, 2,294 input rows and 18,352 positive routing slots. Between 119 and 124 experts per layer were selected, but none of the six affected experts was selected. The test therefore **fails closed**, and no corrected 2K/8K baseline is reported. This finite corpus does not prove that those experts are permanently inactive. The [bounded original router audit](evidence/nvfp4-router-audit.json) found rows and factors finite/nonzero, with no explicit six-expert mask in the inspected source.

### One unchanged-prefix router-score diagnostic

The [sanitized score evidence](evidence/nvfp4-router-scores.json) records one replay of the already captured 1,025-token prefix, with an eight-output-token cap. Routing was unchanged. Each of the five affected layers recorded five prefill calls with 1,025 rows and seven decode calls with seven rows; the eighth output requires no further router input. All 660,480 scores were finite. An independent source-matched bit-order check reproduced all 41,280 selected IDs exactly, and all selected weights were nonzero. The 15 original router-weight/dimension-factor/expert-factor tensors were byte-equal to loaded tensors and their private payload hashes matched the reference manifests. Literal checkpoint layer/expert IDs, TP1/EP1 and the live registered runners agree; there is no observed offset, shard or reindex ambiguity.

The scores explain nonselection for this prefix. Rank 1 is highest; selection requires rank at most 8. The gap is the eighth score minus the target score, minimized over all captured rows.

| Layer, expert | Prefill rank range | Decode rank range | Minimum gap below eighth score |
| --- | --- | --- | --- |
| 0, 42 | 102–128 | 108–127 | 2.070488 |
| 0, 82 | 79–126 | 80–122 | 1.733698 |
| 1, 126 | 121–128 | 124–128 | 2.135818 |
| 2, 89 | 27–128 | 126–128 | 0.367060 |
| 3, 7 | 24–128 | 46–127 | 0.394349 |
| 5, 12 | 84–128 | 107–128 | 0.838470 |

Eight actual residual, normalized and projection-input rows per layer were retained at positions `0,128,255,1024,1025,1026,1027,1028`. The independent NumPy checker reconstructed all 40 BF16 norm/root/dimension inputs exactly. Its ideal RMS-to-BF16 profile agreed on all 112,640 retained values. The 5,120 retained F32 logits differed from independent FP64 dots by at most `2.884449713747017e-6` and passed the conditional gamma-2816 diagnostic. Local NumPy 2.4.4 and isolated NumPy 2.3.5 passed the same checks. This is sampled agreement, not qualification of all 1,032 projection rows; several closest-gap positions have no retained input.

Both BF16 reduced-precision partial-reduction and split-K preferences were enabled. F32 GEMM output therefore does not prove full F32 internal accumulation. Native RMS reduction/rsqrt and Triton exp2/division also remain unqualified. The selected-weight differences were at most `9.2372e-8` against F64 semantics and `1.1920928955078125e-7` against the named F32 profile; these are descriptive values with no fitted acceptance tolerance. The inline root product is reconstructed in the CPU reference, not separately captured.

The ranked diagnosis is observed low target scores in this prefix, followed by unresolved broader selection/calibration or earlier hidden-state semantics. No mapping/logging defect, expert-factor suppression of IDs, or sampled preprocessing discrepancy was found. No production router correction is justified. This finite replay does not establish permanent inactivity or rarity across other prompts. The owned server was stopped after capture; no additional GPU request or benchmark followed. Positive natural correction coverage and the corrected full-model/baseline gates remain blocked. The safest next input would be a trusted, preidentified calibration/usage prefix known to naturally select affected experts, inspected before one bounded replay. Forced fixtures remain separate and cannot supply that prerequisite.

Earlier full-model self-equality and supposedly corrected timings lacked verified hook identity and positive correction coverage. They are **invalidated**, with immutable raw artifacts and explicit private provenance/markers. Their zero-hit metrics cannot establish an active correction or diagnose hook bypass; the newer verified hook also observes zero natural selections. A guarded startup/dummy forward proves dispatch only; positive marked client routing is required independently. Artifact checks reject invalidation, duplicate/incomplete scope, altered or span-only traces, inactive/mixed modes and unmatched client prefixes/layers. Both benchmark entry points currently refuse timing.

The next prerequisite is a bounded natural corpus that positively exercises the six experts, followed by matched active teacher-forced checks and a combined-route reference. If calibration/export examples or an upstream corrected checkpoint become available, inspect their provenance first. Layer-maximum activation calibration, whole-model/cache semantics and accepted held-out quality margins remain separate blockers. Do not broaden GPU sweeps to substitute for those checks.

## Frozen artifacts and scope

- Checkpoint: `nvidia/Gemma-4-26B-A4B-NVFP4`, revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`.
- Runtime: vLLM 0.30.0, FlashInfer 0.6.18.post1, Torch 2.13.0 with CUDA 13.0 runtime, Transformers 5.18.0, Triton 3.7.1, NumPy 2.3.5.
- One RTX 5090, native SM120; BF16 compute and KV; text only; no speculation; TP1/EP1, one active sequence; eager execution, no LoRA.
- Original checkpoint, original runtime files, and prior exploratory results stay immutable. Project plugin builds into a separate target directory. Source hashes accompany each private run.

The six affected `(layer, expert)` pairs are `(0,42)`, `(0,82)`, `(1,126)`, `(2,89)`, `(3,7)`, `(5,12)`. Their discarded gate/up ratios range from 0.6839081 to 1.2500001. The complete source tensor audit found six mismatches among 3,840 pairs. Both gate and up input globals agree within each expert.

## Numerical contract

For each projection, `W[n,k] = E2M1(q[n,k]) * E4M3(sf[n,k//16]) * original_global`. Packing places even K in the low nibble. Each original projection is independently swizzled into 128x4 scale layout; a 704-row gate/up split crosses a tile, so slicing the already fused scale buffer is invalid.

The runtime quantizes an activation once using the maximum calibrated input global across each layer. It also replaces the FC2 activation globals with their layer maximum. This was already the selected pinned FlashInfer path. It differs from using every expert's original calibration. Keep the original-checkpoint oracle, selected-runtime oracle, and defective gate-only control distinct. Neither fixing weight globals nor numerical agreement within this runtime proves whole-checkpoint activation-quantizer equivalence.

Reuse quantized FC1 A for both native GEMMs, with separate F32 epilogue alphas `gate_global*A1_dequant` and `up_global*A1_dequant`. The FP64 oracle explicitly rounds its dot to F32 and applies the F32 alpha in F32 before BF16 conversion. FC1 outputs BF16. Pinned CUTLASS loads these values into F32, performs GELU-tanh and up multiplication, then casts the product once to BF16 before frozen FC2 quantization. A small opt-in Triton adapter reproduces the reviewed F32 instruction sequence, including explicit `fma.rn.f32` and `tanh.approx.f32`; it implements no wider fusion.

For the six experts, original router IDs/weights remain unchanged. The defective contributions are suppressed by zeroing their weights without renormalization. Corrected native expert outputs are multiplied by the original router weights and added in F32. All other experts retain the fused implementation. This still performs the suppressed computations and changes summation grouping. It is not a bit-exact replacement or a claimed speed optimization.

The experiment sets FlashInfer `use_fused_finalize=False`. That lane casts down GEMM outputs to BF16 before F32 weighting/reduction. Default fused finalization instead can weight in F32 before BF16 conversion and uses non-associative atomic reduction. Controls and references must use the same finalization setting. The original CUDA-graph exploratory timings are therefore not a matched performance comparator.

## Independent references and stopping rule

The independent NumPy reference imports no production runtime. It separately decodes all E2M1/E4M3 codes, checks coordinate offsets and padding, reads bounded original tensor slices, performs chunked FP64 dots and software BF16 rounding, and supplies two explicitly named IEEE activation-quantizer profiles. Exhaustive enumeration proves all 4,064 finite `q*sf` products fit BF16 exactly, so the oracle does not introduce an extra global-fold rounding lane.

The gamma-K FP32 RNE error envelope is a conditional diagnostic. PTX leaves E2M1 MMA rounding, order and subnormal handling unspecified; an envelope pass alone cannot certify a native kernel. Preserve direct observed equality, exact packing/layout checks, negative controls, compiled-operation evidence and stage metrics separately. Do not fit an absolute tolerance to observed errors.

Source-matched approximate tanh can differ from mathematical libtanh near BF16 boundaries. CUDA fast reciprocal can also differ from named IEEE quantizer profiles. Capture those differences and stop a frozen-quantizer claim at the first unresolved stage. Bounded teacher-forced comparisons replace only the six affected experts with the FP64 reference; untouched model components remain shared. Every captured row is matched to a verified forward position and full hidden-state bytes, and its input token is checked against the supplied fixed prefix. Sampled synthetic token NLL is reported once per unique position. Synthetic regression and sampled full-vocabulary rows are not held-out quality evaluation or independent attention/cache validation.

G0/G1 require more than a scale fix. Keep qualification incomplete until cache/semantics fixtures and enough paired quality data resolve the chosen margins. Record any corrected timing as a separate exploratory lane if those gates are still pending.

## Reproduce in an isolated approved workspace

Use an already provisioned idle GPU host and the immutable checkpoint. Set `MEGARTX_BASE` to the prior isolated runtime/model workspace and `MEGARTX_WORK` to a new working directory. No script changes drivers, clocks, power, network or security settings. No script downloads weights or accepts remote code.

```bash
export PATH="$MEGARTX_BASE/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export XDG_CACHE_HOME="$MEGARTX_BASE/cache" TMPDIR="$MEGARTX_BASE/tmp"
export FLASHINFER_WORKSPACE_BASE="$MEGARTX_BASE/cache/flashinfer-workspace"
export TRITON_CACHE_DIR="$MEGARTX_BASE/cache/triton"
export CUDA_CACHE_PATH="$MEGARTX_BASE/cache/cuda"
export TORCH_EXTENSIONS_DIR="$MEGARTX_BASE/cache/torch-extensions"
export MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1
# If Python development headers were isolated rather than system-installed,
# also set CPATH to their recorded include directories.
python -m pip install --no-index --no-deps --no-build-isolation \
  --target "$MEGARTX_WORK/adapter-site" .
PYTHONPATH=src:numerical_reference python scripts/probe_nvfp4_scales.py \
  --model "$MEGARTX_BASE/models/gemma4-nvfp4" --output "$MEGARTX_WORK/operator-fixture"
PYTHONPATH=src:numerical_reference python scripts/probe_nvfp4_boundaries.py \
  --output "$MEGARTX_WORK/boundary-fixture"
python scripts/run_scale_validation.py --mode native --client quality --activation-only \
  --routing-diagnostic --label native-routing-diagnostic
# A zero-hit corpus exits nonzero and preserves natural-coverage.json.
# Continue to a paired run only after the guarded natural coverage passes.
python scripts/run_scale_validation.py --mode native --client quality --label native-quality
python scripts/run_scale_validation.py --mode reference --client quality --label reference-quality
python scripts/compare_quality.py "$MEGARTX_WORK/results/native-quality" \
  "$MEGARTX_WORK/results/reference-quality" --output "$MEGARTX_WORK/quality-comparison.json"
```

For the separate one-request score diagnostic, reuse a private recorded request manifest containing exactly one `context-1025` entry; do not generate another corpus. The score client sends at most eight output tokens, preserves the original router, retains at most eight input rows per affected layer, and leaves quality/timing qualification false even when capture succeeds.

```bash
python scripts/run_scale_validation.py --mode native --client quality \
  --router-score-only --router-prefix-manifest "$RECORDED_REQUEST_MANIFEST" \
  --label one-prefix-router-score --trials 0
python numerical_reference/check_router_capture.py \
  "$MEGARTX_WORK/results/one-prefix-router-score/router-scores" \
  --output "$MEGARTX_WORK/router-score-cpu-report.json"
```

`RouterScoreCapture` installs only observational hooks returning `None`. It checks actual decoder/router/runner identity and byte-equal original router tensors. Missing per-forward observations fail the test. Raw NPZ files include private router weights and actual inputs; keep them outside the repository. The independent checker imports NumPy and its own numerical reference, never Torch/vLLM/FlashInfer, and reads hashes, shapes and token positions before arithmetic. It rejects report destinations inside captures or equal to the request manifest and exclusively creates a new report. CPU request-bound tests use small stubs; they do not prove live CUDA invocation.

The lifecycle runner records exact flags, phases and telemetry, bounds compilation/request lifetime, reserves 2 GiB free GPU memory and 8 GiB host RAM, and cleans up only its own server process group. Do not run the oracle while another compute job is present. CPU checks require the pinned NumPy dependency in `numerical_reference/requirements-cpu.txt`. Large raw logits, weights, traces and private environment manifests stay outside the public repository.

## Source evidence

- [Pinned MoE runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/runner/moe_runner.py) and [forward context](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/forward_context.py): actual registered runner lookup and routed-layer dispatch.
- [Pinned vLLM ModelOpt loader](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/modelopt.py): original separate globals are collapsed before format conversion.
- [Pinned activation/scale packing helper](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py): `[gate;up]` reorder, layer calibration maxima, independent scale swizzle.
- [FlashInfer dense FP4 API](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/gemm/gemm_base.py): explicit CUTLASS SM120 dispatch and 128x4 operands.
- [FlashInfer fused MoE implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh): GEMM, gated activation and finalization boundaries. Installed wheel source hashes are retained privately because bundled source can differ from navigation tags.
- [Pinned Gemma4 router](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/gemma4.py) and [GateLinear](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/router/gate_linear.py): router residual, separate BF16 input products, F32 logits, finite-score bit ordering and post-selection expert factors.
- [Installed Torch commit CUDA BLAS](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cuda/CUDABlas.cpp): BF16 reduction preferences can apply with F32 output. Actual flags are in the sanitized evidence.
- [PTX MMA precision contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma): at least single precision for E2M1 accumulation; rounding/order/subnormal behavior unspecified.

The upstream scale-reconciliation proposal was read as background only. Its code was not executed. Common-global E4M3 reconciliation and BF16 scale folding are different rounded models; the original-scale adapter avoids both.

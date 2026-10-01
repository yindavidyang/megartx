# Independent bounded router diagnostic

The independent CPU check explains nonselection in this single observed request:
all six target experts rank below the top eight in the captured scores. It finds
no ID-sort discrepancy or retained-input scale/cast discrepancy. It does not
establish permanent inactivity, whole-model correctness or natural correction
coverage.

The checker imports only NumPy and the standard library. It does not import
production helpers, Torch, Triton, vLLM or FlashInfer, and runs no GPU code. The
source contract comes from the separately reviewed installed vLLM 0.30.0 files;
the checkpoint revision is `a19cfe00be84568a6867111c9a68c9c44fdcffe6`.

## Observed result

The one request used 1,025 unchanged prefix tokens and eight output tokens. Each
of the five affected layers retained all 1,032 natural input rows: 1,025 prefill
and seven decode inputs. The last sampled output needs no subsequent router
input. All 60 forward files and all 15 original tensor payload hashes match
their manifests. Supplied and returned token IDs match every captured position.

All 660,480 F32 scores are finite. All 41,280 selected expert IDs match the
independent finite-bit sorting rule exactly. None of the six target IDs is
selected, and none has a numeric score tie in this request.

| Layer/expert | Observed one-based rank range | Smallest deficit below eighth score |
| --- | ---: | ---: |
| 0 / 42 | 102–128 | 2.070488 |
| 0 / 82 | 79–126 | 1.733698 |
| 1 / 126 | 121–128 | 2.135818 |
| 2 / 89 | 27–128 | 0.367060 |
| 3 / 7 | 24–128 | 0.394349 |
| 5 / 12 | 84–128 | 0.838470 |

Only eight input rows per layer were retained for independent arithmetic, at
positions `0,128,255,1024,1025,1026,1027,1028`. All 40 reconstructed gate inputs
match the captured BF16 bits after the two separate root/dimension product
stores. All 112,640 normalized BF16 values match the rounded F64 RMS semantic
profile. The root storage/cast is BF16 with value `0.018798828125`.

Across 5,120 checked projection outputs, the largest native F32 logit difference
from the F64 dot is `2.884449713747017e-6`. Every checked value passes the
conditional gamma-2816 diagnostic; the largest error/envelope ratio is
`0.0020841131321597444`. All 48 checked target/rank pairs remain outside the top
eight under that conditional envelope. Some closest full-score positions are
not among the eight retained inputs; this is not independent matmul validation
of all 1,032 rows.

Selected-weight differences are descriptive: maximum absolute difference
`9.237195830458234e-8` from F64 softmax semantics and
`1.1920928955078125e-7` from the named F32 profile. No tolerance was selected
from those errors and no native weight-kernel acceptance is inferred.

## Preserved math and cast contract

The router receives the post-attention residual; the expert MLP receives a
separately normalized tensor. Replacing the router input with the expert input
would test a different calculation. The weightless RMS operation conceptually
uses F32 square/mean/reciprocal square root and stores BF16. The checker also
reports a rounded F64 semantic profile without claiming the same native
reduction or reciprocal-square-root instructions.

Production performs two separate products with BF16 stores: normalized input
times `root_size.to(BF16)`, then that stored result times the original learned
dimension factor. The root intermediate was not captured. It is reconstructed
explicitly, and a golden test demonstrates that combining both products can
change the final BF16 code. Original router weights are BF16 `[128,2816]`;
projection output is F32, with no intermediate BF16 logit cast.

The finite F32 selection rule packs transformed score bits and the original
expert ID into a signed I64 key, then sorts ascending. Identical bits tie by
ascending expert ID; positive zero precedes negative zero. Numeric tie ranges
and this exact bit rank are reported separately. Nonfinite scores are rejected.

Learned expert factors multiply only after top-eight selection and selected
softmax normalization. There is no second normalization after applying those
factors. Weight profiles are F64 exponential semantics and an explicitly named
F32 subtract/log2e multiply, F64-exp2-to-F32, balanced F32 selected sum, RN divide
and two F32 products. This does not presume native approximate exp2/div behavior
or the selected kernel's reduction order.

## Conditional bound and residual scope

For K=2816, let `u32=2^-24`, `u64=2^-53` and
`gamma(K,u)=K*u/(1-K*u)`. BF16 products have at most 16 significant bits and are
exactly representable as normal F32 products in the checked range. Let `S` be
the exact sum of absolute products and let `S_upper` conservatively inflate the
F64-computed sum by `1/(1-gamma(K,u64))`, with outward rounding. The diagnostic is

`abs(native_f32_dot - computed_f64_dot) <= (gamma(K,u32) + gamma(K,u64))*S_upper`.

The F64 reference error is included. This bound assumes full F32 RNE
accumulation with no partial BF16 reduction, overflow, unmodeled cast or FTZ
effect. Checking normal products alone does not prove every partial sum avoids
subnormal/FTZ effects. The actual invocation enables both BF16 reduced-precision
reduction and split-K preferences. F32 output alone therefore does not establish
the bound's native-accumulation assumptions. A passing conditional envelope is
not a kernel certification.

The native RMS provider and invocation-specific exp2/div error contract remain
unqualified. Original tensor payloads are checked against the producer's source
hashes and loaded-byte-equality assertions; this CPU checker did not reread the
checkpoint files. Earlier hidden states are shared model outputs. No quality,
performance, cache, whole-checkpoint or G0/G1 acceptance follows. Natural
six-expert coverage remains zero, so the corrected baseline stays blocked.

## Reuse and tests

`router_reference.py`, `check_router_capture.py` and
`test_router_reference.py` are supplied in this directory. The CLI reads captures without modifying them. It resolves report
paths, rejects destinations inside the capture directory or equal to the request
file (including symlink aliases), and exclusively creates a new external report
so an earlier report cannot be overwritten. It verifies file hashes, exact array scope,
original tensor provenance, cast metadata, row/token identities and one-request
limits before numerical analysis. NumPy object/pickle arrays are refused.

```bash
python -m unittest discover -s numerical_reference -p test_router_reference.py -v
python numerical_reference/check_router_capture.py CAPTURE_DIRECTORY \
  --requests QUALITY_REQUESTS_JSON --output CPU_REPORT_JSON
```

All 41 CPU tests pass on the existing local NumPy 2.4.4 runtime. Tests cover
every finite BF16 code, midpoint rounding, separate-store counterexample,
signed-zero/expert-ID ties, selection-before-scaling, cancellation, an unfitted
negative dot control, range assumptions, uncertain ranks, provenance and bounded
input rejection, protected output paths and exclusive report creation. The mathematical checker also ran in the pinned NumPy 2.3.5 environment
with CUDA masked; it reported the same maximum dot error and incomplete
qualification. Original private audit snapshots remain preserved.

Primary source locations reviewed separately: installed `gemma4.py:113–162,
278–304,754–756`, `gate_linear.py:111–119,165–201`, and
`ir/ops/layernorm.py:14–21`. Installed SHA256 values respectively:
`16ac0a67dcf5dd695a59edb177f20e1e69d9c3ec45c5883744470c7c06c516a3`,
`faa43362c2e7c568498a923c682ca18cae75e87f48d00882770821cb19c21198`,
`65d33dcb96404ddde273acf84ef901151a8155a2cffc144bdd0c49fe1d576a22`.
The pinned PyTorch reduction-preference paths are in
[CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cuda/CUDABlas.cpp),
including its BF16-to-F32 specialization.

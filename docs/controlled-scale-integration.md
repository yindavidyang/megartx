# Controlled scale integration

The unchanged one-prefix router diagnostic placed all six affected experts
below top eight. No numerical or source-ID correction is justified by those
observations. This next experiment deliberately changes routing to test scale
correction propagation through the model. It cannot establish natural expert
coverage, held-out quality, frozen-checkpoint quantizer equivalence or a timing
baseline. The existing quality and timing gates stay closed.

Five original live GPU requests exercise the frozen 33-input fixture: full-prefill
native, paired reference and gate-only negative control; cached-decode native
and paired reference. Both positive paths match their independent original-weight
projection replays and each other's captured stages, raw head-call rows and
selected K/V rows. Full-versus-cached handoff equivalence fails earlier than the
scale correction. GPU expansion stopped there; no negative cached, chunked or
2K/8K timing request was added. The
[sanitized live evidence](evidence/nvfp4-controlled-live.json) records the bounded
results and closed gates. The subsequent
[layer-0 arithmetic diagnosis](layer0-arithmetic-diagnosis.md) localizes the
first final-row difference to raw QKV after identical input RMS, reproduces
the M1/M33 outputs in an isolated operator probe, and verifies the entire common
32-row layer-0 cache prefix. Native arithmetic and whole-model gates remain closed.

The [CPU preparation evidence](evidence/nvfp4-controlled-preparation.json)
records 52 new controlled-fixture tests and seven new artifact-boundary
regressions, with all 206 reference/guard tests and 32 scaffold tests passing
locally. That historical preparation used Python 3.12.8 and NumPy 2.4.4; the CI workflow pins
NumPy 2.3.5 separately. Synthetic execution records do not prove live execution.

## Recorded live results

Every request used BF16 compute/KV, one active sequence, eager execution, no
speculation or prefix caching, and the same unchanged runtime source. The
checkpoint revision and stack match the scale-correction record. All five
owned lifecycles exited successfully and cleaned up; the minimum observed
free GPU memory was 10,685 MiB. Raw artifacts remain private and immutable.

| Path and lane | Actual inputs | Model forwards | Completed corrections | Controlled native SM120 dense launches |
| --- | --- | --- | --- | --- |
| Full, native | 33 | 1 | 6 | 18 |
| Full, paired reference | 33 | 1 | 6 | 0 |
| Full, gate-only negative | 33 | 1 | 6 | 18 |
| Cached, native | 32 prefill + 1 decode | 2 | 6 | 18 |
| Cached, paired reference | 32 prefill + 1 decode | 2 | 6 | 0 |

Each lane has six dedicated controlled spans and six adapter activation
launches. CPU launch events and GPU correlation IDs bind those launches to the
six spans, excluding ordinary experts and startup fixtures. Original 54-tensor
hashes match bounded independent checkpoint reads. Shared A1/A2 bits match the
original affected-layer calibration maxima. Each positive native pass's 18
gate/up/down BF16 projections matches NumPy replay exactly. Paired reference
uses independently decoded FP64 dots with explicit F32 epilogues; its actual
stages, ordinary/weighted/routed rows, matched raw logits and all 30 selected
K/V pairs agree in raw bits with native within each path.

The first negative row, layer 0 expert 42 at position 31, has exactly common
actual input, Q1/logical SF1 and gate bits. Changing only the diagnostic up
alpha changes 553 of 704 up BF16 values, maximum absolute difference
0.00390625. It matches independent wrong-alpha replay and differs from the
correct-alpha replay. Later negative activations legitimately differ. This
demonstrates scale sensitivity on the fixture; it is not held-out quality.

Two LM-head calls retain position 32 in the full pass: sampler `M=1` and
prompt-logprob `M=33`. Their outputs differ in 135 of 262,144 raw F32 values,
maximum 0.125, despite verified hidden-row identity. Preserve both. Native and
paired agree exactly when actual head batch sizes match. The checker retains
variants by `(position, source_rows)` and never collapses differing batches.

## Unresolved full/cache boundary

The cached and full passes supply the same 33 actual token/position rows and
the same frozen route/weight bits. Layer 0 position 31 K/V and its expert-42
stages agree exactly. At position 32, before that row's scale correction,
layer 0 K differs in 3 of 2,048 BF16 values (maximum 0.000091552734375) and V
also differs in 3 (maximum 0.00048828125). The actual expert-82 input already
differs in 393 of 2,816 values, maximum 0.0078125. Differences propagate to
all 60 selected layer K/V fields. The same `M=1` sampler row differs in
251,741 of 262,144 F32 values, maximum 3.125 and RMSE 0.6173275557. Both
positive lanes reproduce this result. An unchanged top token does not qualify
handoff equivalence.

The leading source-supported hypothesis is batch/path-dependent upstream
input RMS, QKV projection or K/V RMS arithmetic. V receives no RoPE in the
inspected Gemma4 source, so RoPE alone cannot explain both differences.
Token/layer identities, cloned routes, actual writer slots, stable cache owners
and source hashes were checked. Independent scheduler reconstruction and the
actual pre-QKV/normalization intermediates were not captured, so the cause and
an acceptable arithmetic contract remain unproved. No tolerance is fitted.

The narrow next experiment is two replays of this same fixture, retaining only
layer 0 positions 31/32: input-RMS input/output, raw QKV projection outputs,
K/V before and after head RMS, and actual cache-write associations. Compare
those stages against bounded original-weight CPU dots and source-matched
normalization before proposing a numerical change. Do not expand contexts,
prompts or timing while this boundary remains unresolved.

## Fixed minimal case

Freeze one 32-token prefix from the existing recorded input and one non-EOS
token `d`, giving exactly 33 declared model inputs `P + [d]`. Keep actual API
prompt IDs, expected input IDs and emitted IDs separate in the manifest.

Freeze a complete `30 x 33 x 8` table by literal layer and absolute position.
Ordinary rows use expert IDs `[0,1,2,3,4,5,6,8]` and eight exact F32 weights
of `1/8`. These IDs exclude every affected expert. Replace slot 7 only at:

| Layer | Expert | Input position | First cached request phase |
| --- | --- | --- | --- |
| 0 | 42 | 31 | Prefill |
| 0 | 82 | 32 | Decode |
| 1 | 126 | 32 | Decode |
| 2 | 89 | 32 | Decode |
| 3 | 7 | 32 | Decode |
| 5 | 12 | 32 | Decode |

Retain all eight `1/8` weights, without normalization or a second learned-factor
multiply. Both IDs and F32 weight bits are synthetic and identical across
arithmetic modes and execution paths. This exercises mixed-route accumulation,
with six planned positive corrections per pass. It does not reuse the original
learned router distribution. Never reconstruct ordinary routes from each
candidate's natural scores.

After verifying the installed protocol, a singleton `allowed_token_ids=[d]`
and `max_tokens=2` can keep the cached request's two emitted tokens fixed. The
[pinned completion protocol](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/openai/completion/protocol.py)
exposes that field. This produces 32 prefill inputs plus one decode input; the
second emitted token is not forwarded. The matching monolithic request uses
`P + [d]` as its 33-token prompt and at most one output. Require actual observed
token IDs, positions and chunk boundaries to match the manifest. A future chunk
comparison must use these same 33 inputs and table. Window-boundary extensions
require a separately frozen case after the minimal checks pass.

## Hook and evidence contract

Verify the actual registered runner/kernel/callable binding first. Establish
the request's token/absolute-position context before model forward, then select
the table rows at the registered routed-expert hook. Clone route tensors;
leave actual expert input `x`, router scores, retained original weights and
checkpoint files untouched. Clear the request context in `finally`.

Every controlled request needs a `CONTROLLED-ROUTING.json` marker, explicit
`route_origin=controlled` and independent controlled counters/trace files.
It must never write natural hit counts. The natural-coverage checker rejects
the marker and explicit non-natural request, manifest or hit scope even if all
six counters are positive. Older immutable natural records omit the new fields;
their previously recorded scope remains historical, with zero coverage.

Save actual applied ID/weight bits for every layer/position, the original 54
projection-tensor hashes, registered identities, and six completed expert
execution events with nonzero weights. Save actual BF16 inputs, Q1/SF1,
gate/up, activation, Q2/SF2, down, alpha bits, weighted contributions and routed
BF16 rows at the declared positions. Controlled adapter trace events must prove
the corresponding native dense SM120 projections and activation; startup
forced fixtures are separate evidence. Histogram selection alone is insufficient.

## Arithmetic and first-divergence checks

Use the same adapter, quantizer, activation and reduction path for a causal
comparison:

| Live lane | Corrected projection dot | Up alpha |
| --- | --- | --- |
| Native | Native FlashInfer CUTLASS W4A4 | Original up global times A1 |
| Paired reference | Independent decode and FP64 dot, then explicit F32 dot/alpha/BF16 casts | Original up global times A1 |
| Gate-only negative control | Same native adapter | Gate global times A1 |

Both positive lanes share the source-matched native activation and selected
CUDA quantizer. The existing reference mode's mathematical GELU is a separate
diagnostic; the existing control mode also changes fused path/reduction and is
not this single-change negative control. Create an immutable diagnostic
projection descriptor for the wrong alpha, preserving original up bytes/scalar.

Before comparing downstream logits, reconstruct captured packed operands and
separate original globals with the CPU NumPy reader and bounded GEMM oracle.
Compare exact BF16 bits first. Report signed zero and all nonexact stages.
Conditional F32 accumulation envelopes do not certify native MMA rounding;
named IEEE quantizers and mathematical tanh do not certify CUDA fast math.
Do not fit a whole-model tolerance to observed errors. The negative control
must show sensitivity on a common actual input and agree with the independent
wrong-alpha replay, or the test is inconclusive. Later negative-control hidden
states may legitimately differ.

Only after route, execution, provenance and operator checks pass, compare raw
full-vocabulary logits at absolute input positions 31 and 32 across paired
modes and full/cached paths. Copy logits before sampling: the
[pinned sampler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/sample/sampler.py)
masks the allowed-token whitelist in place. Preserve raw row identity and
token/prediction alignment; constrained sampled argmax is not a quality check.

## Direct KV capture and limits

The retained checkpoint config SHA256 is
`4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98`,
from revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`. Its nominal
geometry is 25 local layers (8 KV heads, head dimension 256, window 1024) and
5 global layers (2 KV heads, dimension 512), with `num_kv_shared_layers=0` and
`attention_k_eq_v=true`. This config is not a live layout/ownership proof,
and the projection-sharing flag does not imply cached K equals cached V.

The collector verifies installed class/source/registry identities and the
FlashInfer writer's actual slot map. BF16 live views use logical
`[block, head, token, 2*head_dim]` with resolved LBNHC strides; local kernel
block size is 16 and global size is 32. Index each actual view's token axis,
retain stable ownership across forwards, and read only populated K and V rows
31/32 after committed length 33. The ledger rejects padding, negative/reused
slots and changed tokens/owners. It does not independently reconstruct the
scheduler block table. Physical allocation IDs may differ between requests. The correction
follows its layer's attention write, so do not require it to alter K/V that were
already written at that layer. Logit agreement alone does not prove every cache
byte, long-window eviction, rollback or independent cache correctness.

Stop at the first unresolved mismatch or unavailable prerequisite. Preserve raw
private evidence and publish only code/tests, provenance hashes and scalar
summaries. No broad prompt search, whole BF16 checkpoint download or timing
suite is part of this bounded experiment. See the existing
[scale-correction evidence](nvfp4-scale-correction.md) and the independent
[CPU reference](../numerical_reference/README.md).

## Reproduce the bounded checks

Use the already provisioned, approved isolated stack and private frozen plan.
`MEGARTX_ADAPTER_SITE` points to a fresh project-only package target; the
lifecycle sets the opt-in plugin, BF16 cache and all proof/capture destinations.
Each label is new and exclusively created.

```bash
python scripts/run_scale_validation.py --mode native --client controlled \
  --controlled-plan "$PRIVATE_CONTROLLED_PLAN" --controlled-path full \
  --label native-controlled-full
# Paired and matched negative lanes use the same plan and new labels.
python numerical_reference/compare_controlled_paths.py \
  --plan "$PRIVATE_CONTROLLED_PLAN" --run-map "$PRIVATE_RUN_MAP" \
  --checkpoint "$MEGARTX_BASE/models/gemma4-nvfp4" \
  --paths full cached --output "$PRIVATE_NEW_REPORT"
```

The private run map declares controlled origin, token/schedule hashes and
per-path directories for `native`, optionally `paired_reference`, and optionally
`gate_only_negative_control`. Fresh servers have distinct activation-proof
hashes; never combine them under one fabricated parent proof. The five-pass
map contains full three-lane evidence and cached positive-only evidence,
explicitly leaving the cached negative lane absent. CPU reports preserve
those limits and keep full-model/quantizer/quality acceptance false. The runner
still refuses timing clients, and natural coverage rejects controlled artifacts.
Checker exit 0 means only that retained operator conditions passed. Inspect
the explicit handoff, complete-matrix and quality flags separately; this
recorded five-pass report has handoff and quality false.

# Controlled scale integration: CPU preparation

The unchanged one-prefix router diagnostic placed all six affected experts
below top eight. No numerical or source-ID correction is justified by those
observations. This next experiment deliberately changes routing to test scale
correction propagation through the model. It cannot establish natural expert
coverage, held-out quality, frozen-checkpoint quantizer equivalence or a timing
baseline. The existing quality and timing gates stay closed.

The route/replay helpers and synthetic CPU tests are prepared. Live controlled
intervention, its paired arithmetic modes, direct KV capture and new GPU results
are pending. The existing plugin still supports only its previous `native`,
`reference` and `control` modes; this document does not authorize an unsupported
environment setting or imply a live runner has been changed.

The [CPU preparation evidence](evidence/nvfp4-controlled-preparation.json)
records 52 new controlled-fixture tests and seven new artifact-boundary
regressions, with all 206 reference/guard tests and 32 scaffold tests passing
locally. Those checks use Python 3.12.8 and NumPy 2.4.4; the CI workflow pins
NumPy 2.3.5 separately. Synthetic execution records do not prove live execution.

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

| Proposed live lane | Corrected projection dot | Up alpha |
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

## Direct KV prerequisites and limits

The retained checkpoint config SHA256 is
`4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98`,
from revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`. Its nominal
geometry is 25 local layers (8 KV heads, head dimension 256, window 1024) and
5 global layers (2 KV heads, dimension 512), with `num_kv_shared_layers=0` and
`attention_k_eq_v=true`. This config is not a live layout/ownership proof,
and the projection-sharing flag does not imply cached K equals cached V.

Inspect actual installed cache owners, strides/layout, block tables, slot
mapping and committed lengths before direct capture. Bind the live contract to
source hashes. Copy only predeclared populated logical K and V rows at positions
31/32, with separate keys and values; exclude padded, stale or uninitialized
memory. Physical allocation IDs may differ between requests. The correction
follows its layer's attention write, so do not require it to alter K/V that were
already written at that layer. Logit agreement alone does not prove every cache
byte, long-window eviction, rollback or independent cache correctness.

Stop at the first unresolved mismatch or unavailable prerequisite. Preserve raw
private evidence and publish only code/tests, provenance hashes and scalar
summaries. No broad prompt search, whole BF16 checkpoint download or timing
suite is part of this minimal preparation. See the existing
[scale-correction evidence](nvfp4-scale-correction.md) and the independent
[CPU reference](../numerical_reference/README.md).

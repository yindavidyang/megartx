# Short speculative verifier contract and independent CPU reference

Local proposal, 3 October 2026. Base: `09341f2ad3f6e4b03e8f264b9e3c59fe5ddb52ed`.
This packet adds a reference and tests under `numerical_reference/`; it changes
no runtime, prefill interface, master roadmap, GPU program or dependency.
Publication, runtime integration, model acquisition and training are separate
decisions. The local clone is isolated from shared Desktop and peer checkouts.
No applicable `AGENTS.md` or `.agents` instructions were present in this clone
or its workspace ancestors. Read contracts: [WP5](../action-plans/wp5.md),
[I03/I06](../master-plan.md), [numerical attention](../layer0-norm-attention-contract.md)
and [source comparison](reference-comparison.md).

The reported target decode status is bounded correctness validation. It does
not establish full G1 quality, end-to-end speedup or graph qualification. These
CPU witnesses establish neither Gemma numerical correctness nor native cache
write, memory ordering or GPU execution. WP5 still enters after WP1 and an
accepted stable WP3/WP4 target path.

## Workload boundary and ownership

One request supplies a short causal continuation block against an existing
prefix: `k` proposed continuations plus one already-emitted anchor input.
Test `k = 0, 1, 2, 4, 7`; `k=0` is the target-only control. These are candidate
lengths, not an eight-row implementation requirement. A verifier must advertise
its supported row counts, output capacity, scratch budget and dispatch lane.
Padding rows must neither receive positions nor write KV or select outputs.

This is distinct from full-prompt prefill and dynamic request batching. Each
row uses an absolute sequence position and a causal mask; the old prompt is
read from committed cache. Short-row dispatch, attention arithmetic and MoE
expert union may differ from both M1 decode and long prefill. An accepted M1
fixture does not qualify these shapes. The prefill CPU owner retains
`src/megartx/prefill_plan.py`, its test, the prefill reference and prefill docs.
The decode owner remains the sole GPU owner; the WP7 owner retains roadmap
docs. This packet proposes shared fields below before any existing-code edit.
It creates no third runtime implementation.

## Freeze the target, draft and verifier interfaces

Each future run must bind the exact checkpoint and quantizer revision, target
numerical lane, tokenizer/vocabulary map, template, EOS set, sampler processors,
cache geometry and source/runtime revisions. A vocabulary-size match alone
does not establish token-ID identity. Reject a stale request/epoch or unknown
mapping before dispatch. Current proposed `configs/experiment.json` still has
unset checkpoint/tokenizer and sampling freeze fields; do not treat them as
qualified WP5 inputs. The bounded normal-decode plan has its own checkpoint
pin; it is not a draft-compatibility certificate.

| Proposed shared field | Meaning and required check |
| --- | --- |
| `request_id`, `epoch`, `transaction_id` | Bind output, every cache layer, auxiliary features and draft state to one request and one attempt |
| `logical_sequence_length` | Tokens already committed to the request history, including its pending anchor |
| `cached_input_length = c` | Contiguous target inputs already materialized; active/terminal convention here has logical length `c+1` |
| `anchor_token_id`, `anchor_absolute_position = c` | Previously emitted target token; it has no target KV yet and must not be emitted again |
| `proposed_token_ids[0:k]` | Scheduled prefix in target vocabulary, predicting positions `c+1` through `c+k` |
| `draft_generated_length`, `scheduled_length = k` | Distinguish all draft work from the prefix actually verified |
| `draft_actual_q[j]` | For sampling, the normalized conditional law that actually produced proposal `j`, after maps/truncation/head biases |
| `target_p[j]` or processed greedy logits | Row `j` predicts position `c+j+1` at the corresponding proposal-conditioned prefix |
| `kv_layer_descriptors` | Per-layer processed K/V dtype/layout/strides, page/ring addresses, capacity, absolute tags, sliding/global policy and aliases |
| `auxiliary_features` | Target layer IDs, normalization/cast convention, absolute positions and accepted-prefix ownership for the selected drafter |
| `rng_snapshot`, emission cursor, stop state | Join KV, sequence, draft hidden/context state and pending token in one publication boundary |

The pure CPU `VerifyBlock`, `Decision`, `KVRecord`, `CacheGeometry` and
`CacheSnapshot` are specification objects, not a stable runtime ABI. The future
runtime owner should agree the field mapping with prefill and decode owners
before implementation. Gemma MTP and a DSpark drafter need different auxiliary
feature adapters; neither may infer feature positions from physical cache slots.

## Anchor, acceptance and cache frontiers

Use one anchor convention throughout; adapters must explicitly translate
sources that count their anchor or draft rows differently. Let the logical
history be `prefix[0:c] + anchor`, with target KV only for `prefix[0:c]`.
Verify inputs `[anchor, y0, ..., y(k-1)]` at positions `[c, ..., c+k]`.
Row `j` predicts `c+j+1`, so row zero checks `y0`, and row `k` supplies the
all-accepted bonus. There is no extra row for the pending bonus itself.

| Outcome | Newly emitted tokens | Target rows committed | Next pending anchor / cached length |
| --- | --- | --- | --- |
| Zero acceptance, `k>0` | Target fallback from row 0 | Anchor only | Fallback at `c+1`; cache `c+1` |
| First rejection at proposal `a` | `y[0:a] + target_fallback(row a)` | Anchor and the `a` accepted draft inputs | Fallback at `c+a+1`; cache `c+a+1` |
| Full acceptance of `k` proposals | `y[0:k] + target_bonus(row k)` | Anchor and all `k` accepted draft inputs | Bonus at `c+k+1`; cache `c+k+1` |
| Accepted EOS or output budget stop | Accepted emissions through the stopping token | Anchor plus all newly emitted tokens except the final one | Final token stays pending; cache advances by number emitted |
| Cancel/failure before publication | Nothing new | Nothing | Original pending anchor and all committed bytes retained |

Example: `c=1023`, anchor `17`, proposals `[23,24]`, target row choices
`[23,99,42]`. Emit `[23,99]`, commit anchor at 1023 and proposal 23 at 1024,
discard proposal 24 at 1025, retain 99 as pending at 1025. On full acceptance,
row 2 supplies the bonus at 1026. Appending either the old anchor or the bonus
KV prematurely gives a different sequence/cache frontier.

Greedy verification compares each proposal to the processed target argmax at
its prefix and stops at the first mismatch. The reference chooses the lowest
ID on exact ties; qualify and pin the real target sampler's tie policy. Logit
processors with history, minimum-length EOS masks or repetition penalties
must be evaluated at each row's hypothetical prefix. Never apply the final
block history to every row. Reject NaN/+inf or an all-masked row.

EOS IDs belong to the target's frozen policy. Accepted EOS stops emission and
discards later rows; a rejected draft EOS does not stop the request. Fallback
and bonus EOS also stop. A pending anchor that is already EOS skips the next
verifier entirely. Truncate at the remaining output budget, without producing
an extra bonus. This proposal leaves the terminal token unmaterialized; if a
future adapter requires terminal EOS KV, it must perform an explicit separately
accounted materialization rather than changing the commit rule silently.

## Transactional KV, pages and absolute positions

1. Reserve bounded staging and output capacity before execution. Snapshot all
   committed layers, page tables, sequence state, RNG and draft context. No
   speculative row may write the authoritative committed storage.
2. Stage processed K and V under absolute positions. They are distinct payloads
   with their own casts/strides; global projection sharing does not make stored
   K equal V. Target row `q` sees only positions at most `q`. A sliding layer
   with window `W` sees `max(0,q-W+1)..q`; a global layer sees `0..q`.
3. Complete verifier execution and selection. Retain only the consumed prefix
   described above. No logits, route/feature row or rejected KV suffix may enter
   the next cycle's history.
4. Prepare every layer's candidate mapping before publishing any. Promote the
   consumed rows, increment the generation, and publish sequence/KV/RNG/draft
   context and pending anchor together. Release staging only after all native
   readers/writers have completed. Stream only the published output, with an
   emission cursor that prevents duplicate delivery on retry.
5. On cancellation or failure before publication, drain the native work and
   abort the entire transaction. Preserve every old byte and restore snapshot
   counters/state. Retry the pending anchor or use the already-qualified
   target-only fallback; do not reuse a partially updated draft context.

Native commit and cancellation need a single linearization point. Cancellation
observed before it aborts; after it retains the published prefix and stops the
next cycle. The single-thread CPU `commit_group` prepares all candidates before
installing any; it proves neither multi-thread atomicity nor CUDA fences. Native
implementation must also prevent old async writers from touching recycled
staging and bind page/slot generations to their request and absolute position.

For a ring, `slot = absolute_position % capacity` is an address, not a RoPE
coordinate. RoPE tables and causal/window origins use the absolute query/key
positions, including 1023/1024/1025 and repeated wraps. Physical pages may be
reused only after last use. A partly occupied committed page needs staging,
copy-on-write preserving its committed fragment, or a proven undo buffer.
Logical/physical page remapping and K/V alias lifetimes must be qualified by
the runtime owner; this CPU reference uses a simple fixed physical page layout.

Position tags and rejected-suffix masks cannot recover destroyed bytes.
For `W=capacity=8`, a four-row speculative write at 8..11 replaces slots
previously holding 0..3. Row 8 still needs position 1, and cancellation needs
the original prefix. Resetting the length or hiding tags 8..11 loses that
needed entry. Stage all verifier rows separately, including blocks longer than
the ring, and overwrite committed slots only after the consumed prefix is known.
Budget peak staging/COW/undo memory rather than assuming rollback is free.

## Stochastic extension and its separate gates

Use the rejection sampler in [Leviathan et al., pinned v2](https://arxiv.org/html/2211.17192v2).
At accepted-prefix position `j`, let `p_j` be the actual processed target law
and `q_j` the actual proposal law over target token IDs. For a proposal sampled
from `q_j`, accept with `min(1, p_j(y_j)/q_j(y_j))`. At first rejection draw
from `r_j(v) = max(p_j(v)-q_j(v),0) / sum_u max(p_j(u)-q_j(u),0)`.
After full acceptance draw the bonus from `p_k`.

Proposal, acceptance and correction/bonus RNG draws must be conditionally
independent given history. Each acceptance uniform must remain uniform given
the proposed tokens and earlier decisions; the correction/bonus uniform must
remain uniform given the proposal/acceptance trace. Use fresh draws at distinct
RNG coordinates; proposal tokens can still depend on earlier proposal tokens.
Reusing a proposal uniform for acceptance with `p=[3/4,1/4]`, `q=[1/2,1/2]`
produces `[1,0]`. This is invalid caller RNG use, not a selector defect; valid
ranges or coincidentally equal draw values cannot establish independence.

Apply temperature, top-k/top-p, penalties, constraints and token maps before
computing these laws. A deterministic greedy draft is one-hot `q`, even when
its network has nontrivial softmax scores. `q(y)=0` is an invalid sampled
proposal; `p(y)=0,q(y)>0` must reject; equal laws always accept and never need
a residual. Approximate shortlist probabilities must describe the sampler
actually used, not an untruncated network distribution. The target law must
cover the target-only sampler's true support; candidate truncation must not
quietly alter that law.

The reference uses `Fraction` arithmetic on small normalized laws. Its exact
three-token joint-law test enumerates proposal/acceptance/target uniform grids
over repeated cycles and compares with an independent serial target product.
It also covers one-hot drafts, zeros, equal laws, residual support, EOS and
output limits. An intentionally wrong independent-token-agreement algorithm
returns `[21/32,11/32]` instead of `[3/4,1/4]`; token agreement is no proof of
distribution preservation. Fixed-uniform replay checks reproducibility, not
same-seed equality with target-only generation or real floating-point sampling.

For this stochastic extension choose `k` before sampling proposals, from frozen
history/features or an independent policy. Selecting which traces to verify
using the sampled tokens can change the conditional draft law. A counterexample
that verifies draft 0 but discards draft 1 gives `[7/8,1/8]` rather than
`[3/4,1/4]`. Adaptive confidence scheduling under stochastic proposals needs a
separate proof of its selection law, or correct conditional probabilities.
The greedy contract can schedule any prefix without that distribution issue.
Draft, acceptance and target RNG streams/counters must be explicit and replay
the same pre-publication snapshot on retry. Real finite-precision sampler and
processor qualification remains separate from this exact-rational oracle.

## Generic speculation, DSpark and the proposed Gemma path

Generic speculation supplies a block, performs target verification, accepts a
prefix and corrects rejection. A block verifier alone is not DSpark. The
[DSpark paper v1](https://arxiv.org/html/2607.05147v1) adds a parallel backbone,
a lightweight sequential Markov/RNN head that conditions later proposals, and
confidence scheduling tied to an engine cost profile. Its draft input/prediction
alignment includes predicting from the anchor row. This paper mechanism must
not be confused with target verifier row count.

The [pinned Inferact `kimi/dspark.py`](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/dspark.py)
provides a parallel mask-token backbone, sequential low-rank Markov selection,
and fused target-logit acceptance with the next anchor at first position plus
accepted count plus one. It includes both a top-M draft-candidate path and a
full-vocabulary fused path. Its draft context cache uses absolute tags and a
ring mask. Its stochastic helper treats deterministic proposals as one-hot,
removes the rejected proposal from residual support, and explicitly qualifies
its target distribution by gathered-candidate truncation. Those are source
mechanisms, not a complete cache safety or target-sampler certificate for Gemma.

The [Inferact report](https://inferact.ai/blog/tpu-megakernels) names
`RedHatAI/Kimi-K3-speculator.dspark` and uses one anchor plus seven continuations
for Kimi on its TPU system. Its throughput/acceptance and TPU-versus-GB200
comparison do not predict RTX 5090 results. Current card metadata for that named
draft is separately pinned here; it is not automatically the revision used by
the report or the pinned source. The report does not establish who trained
that draft; report attribution and checkpoint provenance are separate.
Nor does this review establish that Inferact's fixed fused path implements the
paper's adaptive scheduler end to end.

Existing Gemma candidate review, metadata only:

| Option and frozen metadata revision | Supported observation | Remaining gate |
| --- | --- | --- |
| [Google 26B-A4B assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant/tree/6e5aaaf4c42b98394530b8fda2e95cadd65c151c) | Matching model family; Apache-2.0 card; vocabulary 262144, backbone width 2816, four 1024-wide assistant layers | Pin tokenizer/weights, target-feature boundary, KV sharing map, actual installed support, memory and latency; metadata is insufficient |
| [RedHatAI Gemma 31B DSpark](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark/tree/0026c7d1899651ca3c45ede471712f04849723ac) | Card/config name dense 31B target, auxiliary layers `[1,17,29,47,58]`, Markov/confidence heads and reduced draft vocabulary | Different target architecture/features; not an A4B-compatible checkpoint or a drop-in head |
| Independent smaller autoregressive assistant | Generic interface could support it after exact token-ID mapping | No particular compatible profitable checkpoint established here |
| New/adapted Gemma DSpark drafter | Conditional research path | Architecture, compatible features, data, recipe, checkpoints and compute require separate approval |

[Google's MTP overview](https://ai.google.dev/gemma/docs/mtp/overview) says the
assistant consumes target activations and shared embeddings, and warns that
batch-one MoE verification may load extra experts. The repository-pinned
[vLLM MTP source](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/gemma4_mtp.py)
also specifies Q-only assistant attention reading target KV and a projected
hidden-state feedback interface. This establishes an integration candidate,
not successful loading or compatibility with this installed NVFP4 target and
its cache override. Gemma's published MTP is not the paper's DSpark parallel
backbone/sequential-head architecture. Proposed first Gemma path: qualify that
matching existing MTP assistant using this generic verifier contract, then
evaluate whether a separately approved DSpark adaptation is worthwhile.

No weights, tokenizer payloads or datasets were acquired. Small public configs,
cards and pinned source text were read; their hashes and retrieval bounds are in
[source pins](../evidence/speculative-verifier-source-pins.json). Unknowns include
exact target/draft tokenizer hashes and weight revisions, actual shared KV
mapping under the installed lane, feature normalization, candidate support,
VRAM fit, acceptance survival, stochastic confidence policy and total cost.

## CPU evidence and native handoff

The [local CPU result](../evidence/speculative-verifier-cpu-reference.json)
records exact commands, environment, counts, source-file hashes and retained
log hashes. Run the standalone checks with the existing Python 3.10+ standard library:

```sh
python3 -m unittest discover -s numerical_reference -p test_speculative_verifier_reference.py -v
```

The tests independently rebuild all processed-byte witnesses and physical
slots from a serial sequence, compare full synthetic logits for each causal
row, then retry the fallback/bonus against fresh serial state. Ten prefix
lengths include page/window edges and repeated wraps; five proposal lengths
exercise every acceptance count. Negative controls expose suffix visibility,
committed ring destruction, shifted rows, missing/duplicate bonus/anchor work,
and a correct absolute tag paired with a wrong positional payload. Group
prepare failure, cancellation, stale generations and unequal layer frontiers
must leave original storage intact. Existing CPU discovery includes the new
test file without CI or package changes.

These tests must later be adapted to captured native bytes/addresses/logits by
the runtime owner. Required new native evidence: actual processed K/V and masks
for every supported row shape, target-only greedy equivalence with frozen
numerical lane/near-tie controls, fault/cancellation injection, multi-wrap replay,
staging lifecycle and synchronization checks. Existing FA2/XQA diagnosis already
shows shape-dependent arithmetic; do not turn its conditional envelopes into
a relaxed speculative argmax rule. Any mismatch blocks speculation regardless
of acceptance or speed. There is no mock GPU pass in this packet.

## Cost decision and conditional next milestones

Use `total_request_wall_time / newly_emitted_tokens` with identical prompt,
target lane, output/EOS policy and cold/warm/cache conditions. Include draft
backbone and sequential head, confidence selection, target verification, output
head/sampling, transaction staging/copies/rollback/fences and host streaming.
Record critical wall time and component intervals without double-counting
overlap. Rejected and unscheduled draft work remain inside the denominator's
time boundary; a short verifier does not make the longer draft free.

Record generated/scheduled/proposed/accepted/emitted counts, target fallback
versus bonus, verifier rows/calls, acceptance survival distribution, p50/p95,
memory peaks, and per-layer selected-expert union including rejected rows.
Measure actual expert weight bytes and reuse; extra rows can broaden MoE weight
traffic even when few tokens commit. For no EOS/cap, expected emissions are
`1 + sum_i Pr(accepted_count >= i)`; real stop policies require measured counts.
Acceptance alone does not determine a profitable `k` or a speed claim.

Compare four arms: candidate target-only, incumbent target-only, candidate with
draft, incumbent with the **same draft** and proposal/selection/sampling policy.
Use the [paired measurement protocol](../protocols/measurement.md). A different
draft is a separate draft-quality experiment. Include unprofiled full-model
timing, graph qualification if used, actual backend dispatch and quality gates.
No Kimi acceptance assumption or TPU speed claim enters the RTX scorecard.

| Bounded next milestone | Entry and result; no execution authorized here |
| --- | --- |
| Interface review | Decode/prefill/correctness owners agree anchor/frontier/feature mapping and failure publication; keep runtime files unchanged until coordination |
| Existing assistant feasibility | Stable accepted target path and separate model/GPU approval; freeze revisions/maps, supported lengths, assistant features, VRAM and same-draft pilot budget |
| Native correctness integration | Sole decode GPU owner runs separately approved bounded short-row capture/rollback/sampler tests; no output-cost claim before correctness |
| Conditional training proposal | Only if existing candidates fail compatibility or cost: propose architecture/loss, target-feature traces, data rights and provenance, held-out split/leakage checks, acquisition/storage/compute budget |
| Separately approved pilot and evaluation | Reproducible recipe/software/seeds; bounded pilot, checkpoint hashes, held-out per-position survival/quality and complete latency/memory/expert-union costs; review before widening training |
| G5 decision | Full paired same-draft evidence and target semantics; retain target-only by workload if total cost, tail latency or memory loses |

Training/data/checkpoint paths are future deliverables. This packet chooses no
dataset, generates no target training traces, executes no training and spends
no GPU resources. Publication of this new local task also requires separate
approval; the parent coordinates publication and review.

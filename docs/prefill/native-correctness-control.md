# Smallest independent native cache and handoff control

Status: **CPU contract and proposed source interfaces only**. No runtime hook,
launcher selection, native import, device work, new tolerance, numerical gate,
quality gate or timing result is supplied by this change. This addendum starts
from public PR27 head `6b7530dc1b809e8d49ad990a2dfa7ecc51f279e2`, tree
`d37d5e0799519dc2343493c31e11dcb5ebd7fe77`; historical receipts stay unchanged.

## What the fit result changes

The parent reports one successful native P2048/chunk256, BF16, capacity2304,
256-output observation: 263 heads/frames (7 discarded prompt heads, one final
prompt head and 255 decode heads); all 30 writers; disjoint full-context owned
pages; strict request identity; ordered native/client output hash
`ee72064e6c0b9d5df0e7060f05d1bb48b136bf814fd62fce4aa5171f7b5d9d00`;
usage2048/256/2304; committed2303; completed stream and cleanup. Reported minimum
free GPU memory was10855MiB, observer managed scratch32KiB, measured allocation
increment16KiB/reservation increment0 and native evidence approximately 2.93MB.
The actual heads were BF16, hidden `(1,2816)`, logits `(1,262144)`, int64 selected
indices, and loaded suppression mask empty. These are **parent-supplied facts**,
not a cloud reinspection of private artifacts or independent arithmetic proof.

The later sanitized target receipt (SHA256
`c41c733fb55ee479012b5089947d853e86ae79ac01e82b05dbe12bf201ffea01`)
resolves the page question: **both local and global kernel pages are 16 tokens**,
LBNHC. Local actual BHNC view is`[595,8,16,512]`,131072bytes/page;
global is`[595,2,16,1024]`,65536bytes/page. The checked-live manager size and
subdivision ratio were not serialized, so their actual values remain unknown.
The receipt confirms no pre-write processed inputs or paired raw head/KV
reference was retained, and no natural affected-expert coverage was measured.

It preserves final-prompt head hash
`c976e7c99fd19d109fb0c1b8adc9d9e43b154bcf2d71b97ed592717e817d6024`
and first-decode head hash
`db769837d7723d8bdb6b2cd1d55e012410243ada47805ad638126f0ef753a70c`.
These permit a scoped future byte-repeatability test if all matching inputs are
established, but do not recover raw rows or numerical differences. Checkpoint
config/index/shard-stat identity was
`26b0d3094b99c1a32ffcd25586452616ad83ad213d589c8a6cc9351a242c574b`.
Full model-weight byte identity, serialized loaded runtime config, tokenizer
instance and complete actual SamplingParams digest were not recorded. Recorded
request settings were temperature0, seed1234, ignore_eos=true, n1 and max_tokens256;
the launch used generation-config vllm. Frozen file hashes do not fill those
missing runtime identities.

The existing observer hashes cache contents after writing and compares retained
boundary hashes. Its proof is valuable but does not independently establish
that stored bits came from the actual normalized/rotated writer inputs. Sampling
the same cache twice cannot close that gap. Same emitted IDs likewise cannot
certify logits, all cache bytes or quality.

The master I03/G7 protocol, WP7 A7.1/A7.3/A7.5, existing native protocol,
`prefill_cache_reference.py`, `controlled_kv_capture.py`, controlled-scale
integration and layer 0 norm/attention diagnosis were read before this proposal.
No repository AGENTS.md or local skill directory exists in this exact tree;
the task's supplied AGENTS instructions apply.

## Recommendation and deliberately separate gates

Start with **one unchanged P2048/chunk256 native control**, retaining capacity2304
and the full 256-output continuation. No prompt search, 8K, chunk sweep, new
projection path, custom kernel, DSpark, graphs or TTFT measurement belongs here.

1. Independently bind actual processed K and V to their actual writer call and
   exact stored bytes, for every written row in every one of 30 layers through
   the first consumed anchor: eight prompt chunks plus input 2048. Check all
   required prior rows before each of these writers runs and all required rows
   after query completion. This is the missing prefill/handoff storage control.
2. Bind the final prompt head at absolute2047 to the uncached emitted anchor2048,
   its actual next input and every subsequent input/emission through2303. Reuse
   the already accepted frozen token reference only if its private manifest
   exactly matches; compare the complete ordered output hash, never token text.
3. Compare all-layer selected processed/cache rows and two full native head rows
   with an **independently retained, exact matched native reference**. Such a
   reference has not been supplied to this cloud task. If unavailable, a fresh
   same-path reference capture and fresh check capture require two serialized
   owned contexts, one request each, and a newly reviewed combined budget. They
   are repeatability controls; agreement still is not an independent arithmetic
   oracle. Do not represent the old fit's hashes as missing raw numeric values.

A fresh continuous native P2048 comparison remains the next **separate**
arithmetic-path control named by the existing protocol. Chunk256 fit does not
admit continuous P2048 fit. First bind and fit that exact path, then compare
selected full/continuous and chunked rows under a predeclared independent
operator contract. The known shape-dependent QKV and FA2/XQA behavior makes
bit-equality an invalid default acceptance rule. This packet does not quietly
replace that pending requirement with same-path repeatability.

## Exact identity and independence

Freeze exact private prompt IDs, tokenizer/template hashes, checkpoint revision
and file identity, original packed weights and distinct gate/up globals,
activation-quantizer calibration and code, correction mode, model arithmetic
source vector, stack/binary/driver/device identity, natural routing, sampler
settings and RNG, EOS policy, total capacity, full-context physical policy and
actual dispatch. Keep BF16 compute/KV, TP1/EP1, eager V2, one active request,
prefix caching off, async scheduling off, hybrid manager disabled and M1 custom
preparation/controlled routing off. A changed dispatch or GEMM M is a changed
arithmetic path even with identical source files. Runtime loaded origins and
method owners need the existing exact admission checks.

`match_native_identity()` requires the complete frozen vector and identical
P2048/chunk256/output256/cap2304. Its hashes are bindings supplied by the native
adapter, not cryptographic attestation that code ran. Distinct PIDs, request
UUIDs and physical page IDs are expected across fresh contexts; logical
positions, roles, metadata and contents are the comparison keys. Never compare
raw addresses from different owned lifecycles as though they should match.

Three independent sides are necessary:

- Expected **value**: actual post-K-normalization/RoPE K and post-V-normalization V
  passed to the native cache writer, copied before that writer can mutate them
- Expected **address**: separately reconstruct the request's manager table and
  manager/kernel subdivision using installed source and actual gathered tables;
  reconcile these with actual writer slots, without using those slots as the
  reconstruction's input or reusing the observer's mapping helpers
- Observed **value/address**: read the populated cache via its actual storage
  offset, shape and strides after the native write/query dependency completes,
  independently of the existing `gather_writer_rows` / `hash_rows` code path

The CPU checker imports none of those runtime helpers. Native wrappers still
need their own source-extracted CPU tests for positional/keyword calls, exact
loaded class/instance ownership, stream dependencies, no mutation, interruption
and restoration. A second Python function reading the same observer's output
does not establish this independence.

## Proposed source interfaces and ordering

Use a new default-off standalone native control adapter plus this stdlib-only
offline module. The proposed adapter is **not implemented or registered here**.

1. `begin(Frame, independently_reconstructed_tables, actual_writer_slots)`:
   source-bound actual request/token/absolute-position identity, all 30 geometries,
   owned byte ranges, capacities and table/slot continuity must already be
   verified. The CPU `Frame` models eight 256-row prompt chunks followed by
   255 one-row inputs, but the storage control accepts only the first nine frames
   through committed 2049. Every required table entry must be supplied.
2. `retained(..., phase="pre")`: before any layer writer, stream every still
   required historical row, at most two rows at once, against the original
   processed-input digest. For local queries `[s,e)`, the union is
   `[max(0,s-1023),e)`, not merely the final1024 rows. Global layers5/11/17/23/29
   require `[0,e)`. Every prior required row must be checked before writes begin.
3. `processed(layer, absolute_position, writer_slot, K_bytes, V_bytes)`: attach at
   the actual selected FlashInfer writer call, preserving ordinary arithmetic,
   source objects and slot maps. Require exact finite BF16 logical shapes
   K/V `(8,256)` local or `(2,512)` global. Borrow/slice the actual inputs; never
   reconstruct them from the just-written cache. Copy only one/two rows, hash on
   CPU with role/layer/absolute-position domain separation and immediately release
   the temporary. A full 256-row global/all-layer GPU clone is not permitted.
4. After source-bound completion of all relevant writer/query streams, call
   `retained(..., phase="post")` on **every row of every required union**, including
   new writes, then `finish(queries_complete=True)`. Only all 30 complete layers
   commit. Missing evidence, stale owners, aliasing, a source/slot mismatch or an
   interruption poisons the context. Stop and rebuild through the owned native
   lifecycle; never continue from partial state.
5. Copy final-prompt and first-decode processed model-head rows before in-place
   sampler transforms. Bind actual selected hidden bits and int64 index0/255,
   loaded BF16 head policy/empty suppression mask, softcap, head ordinal,
   prediction position and actual sampler inputs/outputs/counts. Preserve all 263
   scalar head events to distinguish discarded intermediate samples. Independently
   check `verify_frontier()` and the strict native/client request-token ledger.

For this full-context control, physical slots are never recycled within a
request. The union contract is nevertheless required so later ring/page designs
cannot prematurely recycle rows needed by early chunk queries. This control
does not qualify a recycling implementation. The CPU ledger stores bounded row
digests in host RAM; it is not an allocator, attention mask implementation or
GPU collector. Caller-provided source/stream assertions are not native proof.

## All-layer samples and exact versus numerical policy

Every written and every required retained row through input 2048 is checked
in-stream for exact writer-to-cache equality; no tolerance applies to storage, ownership, position,
K/V role, masks, source identity, token transport or commit frontier. Signed zero
is preserved and a wrong bit is a mismatch. Processed K and V may legitimately
have equal values; different ownership and correct provenance remain necessary.

Privately retain at most eight rows for **each of all 30 layers, both K and V**:
the positions immediately before/at each actual first kernel-page boundary,
1023/1024, 2047/2048. The sampler is derived from all 30 actual page sizes, not the
historical16/32 assumption. Two distinct page sizes dividing256 fit this
contract; other geometry needs re-review, never truncation. Streaming exact
checks cover later prompt page boundaries, all chunk rows and the first consumed
anchor. The remaining 254 decode inputs retain exact metadata/token/frontier
checks and the matched output digest, but their K/V contents are not independently
qualified by this storage control. The raw numerical sample set is narrower
than the first-nine-frame exhaustive bit checks and is reported as such.

Also retain full `(1,262144)` BF16 final-prompt and first-decode head rows per
context, before sampling. Only their private hashes and scalar comparison
results can be public. An unchanged top token cannot substitute for either row.

- **Same path, same actual inputs and dispatch:** test raw bit repeatability.
  A difference is unresolved, not automatically evidence of a storage defect.
  Do not loosen a threshold until it passes. Exact agreement establishes only
  matched sampled native repeatability in addition to the in-run storage proof.
- **Changed chunk/GEMM/attention path:** report BF16 word mismatch count, maximum
  absolute error and RMSE as observations, with no automatic acceptance. Existing
  M33/M1 QKV and FA2/XQA differences demonstrate why cross-path equality cannot
  be presumed. A tolerance must be derived and frozen per operator/domain from
  a named independent oracle, formats/casts/reduction/dispatch assumptions and
  sensitivity controls **before** judging those comparisons. Missing contract
  means numerical acceptance remains pending; no fallback `atol/rtol` exists.
- **Previous references:** the ideal retained-token/cache model supplies mask,
  position, local-union and frontier semantics. Controlled33-input same-path
  positive/negative scale replays and layer 0 operator evidence support only
  their exact operands/source/forced-route domains. They cannot be relabeled as
  natural2K quality, all-layer arithmetic, current-shape attention or continuous
  P2048 acceptance. Original-scale source identity does not equate runtime
  layer-max activation calibration to checkpoint per-expert calibration.

`compare_bf16_samples()` therefore always leaves independent arithmetic,
whole-model quality and performance false. This is intentional, including when
all selected samples agree. Natural positive correction coverage needs a
separate actual execution receipt; unavailable coverage remains unavailable.

## Finite resources and evidence layout

One shared wall deadline1800s starts before the first owned startup and includes
any admitted second fresh context, drain and publication; existing separately
bounded owned TERM/KILL cleanup remains available. Maximum two serialized model
contexts, one request each, zero retries or sweeps. The first context alone can
establish the exact in-run storage/frontier result; a missing matched reference
does not authorize an extra run. A fresh pair needs explicit parent clearance.

Preserve incremental observer GPU cap8MiB (including measured allocation and
reserved-pool growth), free GPU floor2GiB, host-available floor8GiB, aggregate
compiler RSS2GiB and **shared** build wall 300s. The same process-global peak-reset
preservation requirement applies. Observer tensors and framework temporaries
must be accounted before/after every scope; separately ledger incumbent model,
KV and startup/reference workspace. The previous FP64 startup projection
workspace is not hidden inside the observer allowance. No new compiler work is
planned by this CPU change.

Transfer work is bounded separately: exhaustive source/pre/post checks across
the first nine frames require4,024,934,400bytes of transient D2H payload per
context (3.749GiB),547650 single-row-equivalent reads, or bounded two-row batches.
Freeze a4GiB transfer cap per context and the shared1800s wall deadline; overhead
may still fail that deadline and is not yet measured. Rechecking every retained
row before/after all 263 frames would cost approximately124.06GiB per context
and19,080,760 single-row-equivalent reads. That larger control is intentionally
excluded, even though its concurrent scratch could still look tiny. Transferred
bytes are immediately hashed/discarded except the selected rows; they are not
authorization to retain a multi-GiB private tensor capture.

With the historical alternative page sizes16/32, the worst-case combined
two-context private evidence budget is:

- Eight selected processed/cache-equivalent K/V rows per layer:3,604,480bytes
- Two full BF16 head rows per context:2,097,152bytes
- All provenance, compact all-row digest/count roots, frame/head/sample/client
  summaries, telemetry, comparisons and temporary publication files combined:
  at most2,097,152bytes
- Total7,798,784bytes, below8,388,608bytes; remaining589,824bytes is not a separate
allowance. The cross-process budget must enforce actual bytes before writes

For the now-reported actual16/16pages the sample positions are
15/16/1023/1024/2047/2048 (six rows), making selected K/V 2,703,360bytes and the
combined two-context bound 6,897,664bytes. This is capacity planning from the
sanitized geometry, not a native allocation or permission to add other captures.

Keep only one identical copy of each selected processed/cache row after its
exact comparison succeeds. On failure, retain a bounded first-difference scalar
receipt or replace an unused sample within the same allocation; do not save an
unbudgeted second cache tensor. In-memory row digests and decoding heap count
against measured host memory, not GPU scratch. A complete host tensor capture,
whole-cache GPU clone, all-row logits or full prompt dump is not part of this
plan. No raw tokens/tensors/checkpoint bytes go into public source/evidence.

The old approximately 2.93MB full diagnostic transcript **cannot simply be
duplicated beside** these samples under the metadata budget. Use an independently
reviewed compact control receipt (all 263 frame/head identities and checked
all-row counts/root hashes, no repeated tables), or stop before capture. Never
silently omit required evidence or raise the8MiB cap. Historical artifacts stay
immutable outside the new experiment; referencing their hashes does not count
as new copies and does not make them independent numeric inputs.

## Source ownership, overlap and admission still needed

This patch adds exactly three files: this protocol, the independent CPU module
and its tests. Existing plugin, launcher, loaded-engine access, sampler, native
observer, source pins and historical receipts are untouched. No executable GPU
entry point exists in these files.

A future native adapter would touch the shared boundaries in
`vllm_scale_plugin.py`, `run_scale_validation.py`, `prefill_native.py`, runner
bindings and the selected FlashInfer cache-writer boundary. Coordinate with the
parent before modifying those files: decode/DSpark work shares plugin/lifecycle
paths, and any byte change requires additive source reconciliation, independent
review and a new exact-head clearance. CPU tests here do not authorize that
integration or a GPU slot. Keep the existing fit proof byte-identical.

Requested compact target receipt, with no private token/tensor disclosure:
successful exact HEAD/tree/plan/source/environment hashes; checkpoint/tokenizer/
sampler identity hashes; local/global actual kernel pages and manager subdivision;
whether pre-write processed K/V were retained; whether an accepted same-chunk
native reference retains the selected rows and two raw heads; actual natural
affected-expert positive execution coverage. Unknown fields remain null/pending.

Stop on missing provenance/reference, mismatched source or same-path identity,
incorrect local/global union, any processed/cache bit mismatch, partial all-layer
commit, wrong final-head/anchor/input chain, telemetry loss, resource overrun,
foreign GPU work or uncertain cleanup. No8K/full2K change/timing expansion follows
automatically. Separate outcomes must remain visible: storage/provenance,
frontier, sampled same-path repeatability, cross-path arithmetic, quality and
performance. Only the first three are addressed by this proposed small control.

## CPU verification

From this checkout, using existing Python/NumPy:

```sh
python -m unittest discover -s numerical_reference -p 'test_prefill_native_control.py' -v
python -m unittest discover -s numerical_reference -v
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m megartx validate configs
python -m compileall -q numerical_reference src tests
```

The new controls exercise all 263 frame unions against the retained independent
semantic reference, first-chunk and anchor all-layer row checking, wrong K/V
role/value/address, omitted required prefixes, premature final-window retention,
unsynchronized/partial commits, mutable caller tables, changed identity, budget
overflow admission, head/emission shifts, uncached frontier and the absence of
an invented numerical tolerance. Synthetic row tags and substituted inputs are
CPU contract evidence only.

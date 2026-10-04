# Native storage/frontier control implementation

Status: **CPU/source implementation awaiting independent review and separate
exact-head GPU admission**. This does not contain a run, matched native reference,
quality result, numerical tolerance, performance result, or clearance. The public
parent is PR34 head `3e9f50f509c5f79da8505dd04b251cf950502283`, tree
`7f0543aaad4999a5a8ed698c58524cfb0a32ee06`; its parent is PR27
`6b7530dc1b809e8d49ad990a2dfa7ecc51f279e2`. The earlier
[native correctness protocol](native-correctness-control.md) and every historical
catalog/receipt are unchanged.

## Purpose and exact scope

The new explicit client is `prefill-storage`. It selects
`MEGARTX_PREFILL_STORAGE_CONTROL=1`; the existing `prefill-native` client selects
its previous provider and plan schema. Storage and fit plans/clearances cannot
substitute for one another. Neither mode is active by default. The existing
launcher plan/clearance argument names are reused: `--prefill-native-plan` and
`--prefill-native-clearance`. Their storage schema binds the new purpose, complete
source catalog, adapter origins, checker bytes and unchanged native runner policy.

There is exactly **one context, one P2048 request, chunk256, BF16, capacity2304,
256 emitted outputs**, with zero retries. All30 layers receive exhaustive
processed-to-storage checks for eight prompt chunks and consumed input2048:
9 frames, committed2049,61470 processed layer-rows,212355 pre-retained layer-rows,
273825 post-retained layer-rows. All263 frame/head events remain bound to the
request. The remaining254 decode inputs have metadata/frontier/token checks,
not independent K/V content qualification. The final emitted token at2303 remains
uncached. No optional pair, continuous2K,8K, benchmark, prompt search, new kernel,
model download, source edit on the experiment host or automatic follow-on exists.

## Three separate authorities

1. **Expected values:** an instance-owned wrapper on each actual selected
   `FlashInferImpl.do_kv_cache_update(self, layer, key, value, kv_cache,
   slot_mapping)` copies the original processed K and V before calling the
   original method once with unchanged arguments. The pinned Gemma4 path has
   already applied K norm/RoPE and V norm; the pinned Attention reshapes those
   values. Positional, keyword and mixed calls use the original bound signature.
   Inputs must have exact `[rows,8,256]` local or `[rows,2,512]` global BF16 shape,
   distinct logical K/V starting addresses, and no cache-allocation alias.
2. **Expected addresses:** an instance wrapper captures immutable copies of the
   real `BlockTables.append_block_ids` manager inputs before native subdivision.
   It commits the receipt only after native append succeeds. Capture must begin
   with the actual new-request overwrite and bind the sole request-state index
   and request ID. Current manager size B, kernel size K, ratio r satisfy B=K*r;
   K must be16 for every group. For absolute p, manager ID m gives kernel ID
   `m*r + (p%B)//K` and slot `kernel*K + p%K`. No division/back-solving of observed
   kernel IDs supplies expected manager IDs. Both persistent and gathered kernel
   tables, then the actual writer slot tensor values at the call boundary, must
   agree. Missing capture, reordering, prefix replacement, reuse and alias fail.
3. **Observed values:** `stored_row` constructs borrowed physical views from the
   actual cache storage offset and strides. It never imports/calls the old
   `gather_writer_rows` or `hash_rows`. It reads the K and V halves independently
   as raw BF16 bits after explicit device completion. The independent stdlib CPU
   checker is loaded from its source-bound file, not reconstructed from old
   observer output.

Every required historical row is checked before any writer starts. Local queries
[s,e) require the union [max(0,s-1023),e), including the early-query prefix;
global layers5/11/17/23/29 require [0,e). After model completion and whole-device
synchronization, the complete union including new rows is checked. All30 complete
layers are required for commit. Full-context slots are never recycled. The
observer is perturbing; these synchronizations invalidate timing claims.

## Ownership, interruption and restoration

Existing loaded model/cache/group/builder/ForwardContext checks remain active.
The new hooks additionally bind actual class functions, instance wrappers,
receivers, cache views, manager owners, sampler owners and source files. The
actual manager num_blocks GPU tensor legitimately rotates among the fixed
source-defined UVA pool. The control freezes num_blocks/CPU/NumPy/pool/list/buffer
owners, requires the active view to point into the selected unchanged pool slot,
and rejects a foreign equal-content replacement. It does not incorrectly freeze
a transient UVA view object. Pool growth/replacement is not admitted.

The selected source-pinned no-op KV connector and absent kv_transfer_config are
required; connector cache transfers would enlarge the provenance boundary. The
actual SamplingParams struct fields are serialized into a private identity hash
at sampler.add_request. All request settings still require greedy temperature0,
seed1234, ignore_eos, n1 and max_tokens256. Unsupported identity types fail rather
than fall back to repr. Native BF16 head dtype, empty loaded suppression mask,
softcap30, actual int64 index255/0 and selected hidden bits are checked. The
actual sampler must consume the same head object and return the same output
object to the runner. The independent frontier checker verifies all263 heads,
256 emissions and255 consumed decode inputs; the strict client transport ledger
separately checks exact request identity, complete ordered token hash, usage and
SSE completion.

Any exception, interruption, ownership loss, synchronization loss, mismatch or
resource breach poisons the context. It cannot continue. Before restoring the
new callbacks, drain the device. Drain failure retains the hooks for owned
process teardown; a foreign callback replacement is never overwritten during
restoration. Primary failures retain cleanup errors as notes. A valid control
receipt requires successful own-hook restoration. The inherited runner/model
observer hooks remain until the owned process is removed. Final publication also
requires the existing owned lifecycle/cleanup receipt and no remaining owned
process or GPU identity.

## Bounded data and resource domains

Only borrowed row views exist on GPU, at most one K/V row pair at a time. D2H
copies complete synchronously before the writer runs; contiguous conversion is
on CPU. Raw inputs are hashed/discarded immediately, except six selected positions
15/16/1023/1024/2047/2048 in each of all30 layers, both K and V. Exactly180 combined
K/V files contain1351680bytes. Two full native BF16 head rows at input positions
2047/2048 contain1048576bytes. Each selected K/V has one copy only, and publication
requires the exhaustive comparison and exact raw manifest root. A bounded
scalar mismatch exception is retained; no second cache dump is added.

The other261 heads use a finite scalar reduction plus hidden/index identity;
there is no all-head raw-logit download or retention. The boolean reduction
scratch is262145bytes explicitly reserved before allocation. The existing
GpuScratch ledger separately measures allocator growth and reserved-pool growth,
including framework temporaries, and preserves process-global peaks across
resets. Incremental observer cap is8MiB; host free floor8GiB, GPU free floor2GiB,
compiler RSS2GiB and shared build wall300s are unchanged. Model/KV/incumbent hidden
storage and startup/reference projection workspace are baseline resources,
separately reported, never hidden as observer scratch.

The exhaustive storage payload is4024934400bytes (3.749GiB). Exact extra head,
writer-slot and selection payloads plus the conservative inherited metadata
allowance total4071312461bytes before actual independent manager-table reads.
The inherited allowance is1MiB per prompt frame and128KiB per decode frame,
charged before entering existing metadata helpers; it is an upper bound, not
measured traffic. All remaining explicit copies charge the shared4GiB transfer
ledger before copying. Thus the transfer receipt combines exact payload domains
and an explicitly named conservative domain. No claim of exact total PCIe
traffic is made. One common1800s deadline starts before startup, includes request,
drain and publication, with inherited separately bounded owned cleanup.

All new control evidence uses one cross-process flock budget:8MiB total and2MiB
metadata, including temporary publication files. Writers reserve before writing,
reject links/nonregular entries, and never silently truncate. The directory
reserves the two-byte external run.exit lifecycle signal before its first write.
Raw sample files are exact-name/exact-size whitelisted. The old2.93MB diagnostic
transcript is not duplicated. Compact records preserve all frame/head identities,
all-row counts/root hashes, client facts, resource observations and cleanup.

GPU telemetry keeps the unchanged200ms wait/polling cadence. A single field header
and lossless timestamp/value/exit arrays retain every observation (max130bytes
per record,9000records); no downsampling or peak erasure occurs. Duplicate outer
phase, status and telemetry files are disabled for this purpose. Launch manifest,
owned process/cleanup receipts and bounded client console are budgeted once.
Inherited server console and fixed activation/forced-startup artifacts are listed
in the separate startup/reference domain; they cannot serve as storage evidence.
Six artificial startup fixtures do not establish natural positive correction
coverage, which remains explicitly null.

## Receipts and qualification

`storage-binding.json` is required before POST, in addition to the existing exact
runner binding. `control-frames.jsonl` and `heads.jsonl` contain263 compact ordered
records; `samples.jsonl` retains256 scalar output bindings. `control.json` binds
counts, roots, actual manager geometry, actual sampler identity, transfer domains,
raw manifest and independent frontier. The storage-specific client receipt
requires exact transport agreement. `storage.json` is written only after complete
owned cleanup and revalidation; **fit.json is forbidden in this purpose**.

All sampled-repeatability, independent arithmetic, quality, numerical and
performance flags remain false. There is no accepted raw native comparison
reference, so this cannot report sampled repeatability. The checker proves an
in-run storage/frontier contract when actual hooks run; CPU fake success does
not prove that they ran. The known old model/tokenizer/head hashes and six forced
fixtures do not fill missing native operands or natural correction coverage.

## Source and CPU verification

`native-storage-source-reconciliation.json` binds the complete new-purpose vector
and rejects mixed or unknown sources. Historical catalogs are unchanged. Old
packet tests use explicit preserved PR34 source fixtures, while current-source
compatibility tests independently prove full plugin/launcher/default-provider
AST equality after evaluating only the absent storage selector and declared
optional-provider defaults. The legacy plan refactor preserves the complete
validation AST. Existing current-source runner ownership and fit/client tests
also continue to run, so historical fixture tests do not substitute for current
compatibility coverage.

CPU tests execute the actual extracted source bodies for selected FlashInfer
writer and manager append calls, test original argument identity and source
ordering, ratios1/2/4, no initial capture, prefix/tail retention, mutable caller
inputs, partial append, wrong receiver, class/instance replacement, source-defined
UVA rotation, stream failure, poisoned continuation, safe restoration, signed zero,
K/V swaps, first-frame all30-layer storage checking and budget pre-overflow.
Fake native tensors contain raw BF16 words without importing Torch or vLLM.

Run only these CPU commands before independent review:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m unittest discover -s numerical_reference -v
PYTHONPATH=src python -m megartx validate configs
python -m compileall -q numerical_reference src tests scripts
git diff --check
```

Freeze runtime and protocol first; freeze catalog metadata separately afterward.
Independent review must inspect actual source/hook dispatch, UVA rotation and
replacements, current default-fit equivalence, primary-error/cleanup ordering,
and complete resource domains. Only a later, explicit exact-head review and
parent slot clearance may admit one experiment. No merge, GPU experiment or
second context is authorized by this source packet.

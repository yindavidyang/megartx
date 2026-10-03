# CPU prefill profiling runner and launcher handoff

This local implementation extends the merged PR16/17 preparation on main
`5a32504b5ccb209a779b8a110eb2ebc56d72e373`. It compiles a serialized slot, builds
exact-token client payloads and validates normalized recorded results. It uses
only the Python standard library. It has no network, process, CUDA, model-load,
native-build, plugin-registration or GPU execution path. `--execute-gpu` always
fails before any device work. Draft publication is approved; independent review
is tracked in [PR19](https://github.com/yindavidyang/megartx/pull/19).

No existing file is modified. In particular, the stock prefill fallback,
`m1_live.py`, `vllm_scale_plugin.py`, quantizer and shared launcher are untouched.
[WP7](../action-plans/wp7.md) and the [merged semantics](README.md) remain the
source contract. Nothing here accepts G0/G1/G7, numerical equivalence, peak fit,
graphs, asynchronous scheduling, performance or broader public safety claims.

## CPU commands

From the repository root, with an existing Python 3.10+ installation:

```sh
PYTHONPATH=src python3 -m megartx.prefill_runner \
  --manifest docs/prefill/runner-plan.json \
  --plan docs/prefill/profile-plan.json \
  --source-manifest docs/prefill/source-binding.json \
  --runner-binding docs/prefill/runner-binding.json --root . \
  --workload p2048 --chunk 2048

PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_prefill_runner.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_prefill_plan.py' -v
python3 -m unittest discover -s numerical_reference -p 'test_prefill_cache_reference.py' -v
```

Omitting the cell selection lists the full reviewed matrix: 2K chunks
255/256/512/1024/2048 and 8K chunks 255/256/512/1024/2048/8192. A 255-row upper
bound gives a final M of 8 at 2K and 32 at 8K. These are scheduler rows, not
request concurrency or per-expert M. 32K and WP5 short DSpark verification are
rejected; they need separate fit/workload/cache-transaction review. Single-cell
selection lets the first future slot stay small. `--output PATH` creates a new
local file exclusively and never overwrites an earlier packet.

For private offline normalized records, append
`--records PRIVATE_RECORDS.json --prompts PRIVATE_PROMPTS.json`. Both inputs are
required, regular non-symlink JSON, at most 8 MiB each; select fewer cells when
normalized data would exceed this cap. Duplicate keys and nonfinite constants
are rejected. The output contains scalar summaries and hashes, not prompt or
output token IDs. The executable does not ingest or download raw GPU traces.
`records_consistent` is a parser/accounting result; admission blockers and every
qualification flag remain explicit. It never turns fixture data into a gate.

## Admission inputs

[runner-plan.json](runner-plan.json) is disabled and deliberately incomplete.
The original [profile plan](profile-plan.json) is unchanged. Its exact bytes,
historical source-binding bytes and the seven new/additional source pins in
[runner-binding.json](runner-binding.json) are verified locally. The runner
binding records the implementation base, not an assertion that the future target
is running this code. Freeze the final Git head/tree and patch independently.

Before any future live integration, fill and independently review:

- Exact checkpoint, unchanged runtime layer-max W4A4 quantizer lane, BF16 compute
  and KV, causal/local window, absolute RoPE, original scale/cast/reduction
  contracts and last-prompt-row head. None is replaceable through this runner.
- Exact prompt IDs and canonical ID hashes; tokenizer/template, prompt set,
  oracle, target environment/source/binary/package/driver/device identities;
  accepted fixed decode-control commit and source-bound observer registries.
- Explicit physical local KV storage policy (`full_context` or a source-reviewed
  `bounded_window`), power-of-two page size and cache-layout contract hash. A local
  attention mask alone never authorizes ring/window allocation or its fit estimate.
- G0/G1, native numerical/cache comparison and positive natural correction
  coverage, supported actual dispatch and fit for every selected chunk, output
  capacity and frozen reserve. Forced routes and small CPU fixtures cannot fill
  those receipts. Fit for 2K does not admit 8K; neither admits 32K.
- Separate GPU scope approval, the sole owner's explicit handoff, prior cleanup,
  launcher/source review, frozen thresholds and finite WP7 build RSS/time,
  scratch, requests, wall time and trace bounds. Existing M1 limits are not defaults.
- A unique `run_namespace`, fixed `greedy_seed_1234` sampling and
  `ignore_eos_256` output policy in the reviewed prefill plan. The client has no
  natural-EOS or alternative sampler implementation; such lanes fail intake.

The runner computes `scope_sha256` over executable identities, source binding,
controls, resource bounds and the selected cell schedule. Acceptance pointers and
the containing plan's byte digest are excluded to avoid receipt self-hashing.
The entire plan's actual byte digest is checked separately. Fill identities and
bounds first, compile the selected scope, then collect review receipts. Changing
an executable input changes the scope and invalidates those receipts.

Each `receipts` value is null or the SHA256 of an existing bounded JSON artifact
named `<digest>.json` under `--receipt-root`. Its exact schema is:

```json
{
  "schema": "megartx-prefill-receipt-v1",
  "kind": "numerical",
  "scope_sha256": "<selected executable scope, 64 lowercase hex>",
  "decision": "accepted",
  "evidence_sha256": "<reviewed evidence artifact, 64 lowercase hex>",
  "reviewer_reference": "<bounded independent-review reference>"
}
```

Kinds: `source_review`, `launcher_review`, `gpu_scope`, `ownership`, `resource`,
`cleanup`, `g0`, `g1`, `numerical`, `natural_correction_coverage`, `dispatch`,
`decode_control`, and `fit_p2048` / `fit_p8192` for the selected prompts. The
original workload `fit_receipt_sha256` must match its declared runner receipt.
Each receipt must also resolve a bounded regular JSON evidence manifest named
`<evidence_sha256>.evidence.json` in that directory, with matching bytes. This
manifest can point to private larger numerical/trace artifacts; reviewers must
verify those artifacts independently. Missing receipt/evidence files, nulls and
unaccepted decisions block admission; changed bytes, wrong kind or scope fail
parsing. Reviews must establish that the evidence actually
covers all selected chunks. Hashes and self-declared reference strings are
integrity checks, not signatures or independent attestation. This CPU code cannot
authenticate an approver, physical device, resource lease, target binary or
numerical evidence. A structurally complete packet still cannot enable GPU work.

## Required final PR15 integration and file ownership

`LauncherAdapter` is the proposed narrow data interface:

```python
describe_contract() -> dict
execute_serialized(protocol: dict, private_prompts: dict) -> dict
```

There is no adapter implementation or module-import option. The eventual shared
launcher must consume a reviewed protocol, describe the pinned contract below,
refuse unsupported controls before startup, preserve verified stock fallback,
and return records with owned startup/cleanup evidence. The prefill side owns
schedule/client payloads/record validation; the shared launcher owns process
guards, leases, AOT/provider loading, exact stream-ID translation, synchronization,
timeout/drain/cleanup and poisoning uncertain cache state. Do not copy these
mechanisms into this runner or terminate another owner's process.

The handshake is `megartx-prefill-launcher-adapter-v1` plus exact
`pr15_final_commit`, `source_manifest_sha256`, `stream_provider_sha256` and
`supported_chunk_tokens`. All identities are null and the support list is empty
in the checked-in manifest. No final PR15 version is accepted yet. Integration
requires the **final independently reviewed PR15 commit**, its launcher/collector
source manifest and AOT/CUPTI provider pin, following the parent's approved
prefill source reconciliation. A moving PR head or current peer checkout cannot
substitute for those pins. Record the final commit in this manifest only after
that review, and rebind changed source contracts through the prefill owner.

The PR15 eager benchmark collector's 256-row fallback guard is incompatible with
the 255/512/1024/2048/8192 sweep. Declaring `[256]` blocks the other cells. The
256-row case must also distinguish prompt forwards from the one-row fixed decode
continuation calls, which the reviewed collector must explicitly support. The
support list declares upper chunk sizes;
the receipt scope binds actual spans, tails, 30-layer forwards and decode rows.
No collector is reused or independently patched here. `m1_live.py` and
`vllm_scale_plugin.py` prefill pins are known to drift in PR15: the historical
PR17 binding is preserved, not silently rehashed. Changed local bytes fail the
original source check before protocol generation. Existing-file touch needs for
future integration must be reported to the parent before implementation.

Until this integration is reviewed and the GPU owner hands off an approved slot,
the CPU executable remains disabled regardless of receipt completeness. Sole
GPU ownership stays with `01a10079-c8d1-752e-991a-4e76b523704e`. Decode PR15 and
DSpark PR18 review remain independent workstreams.

## Serialized jobs and timing boundaries

Each selected cell compiles nine distinct run IDs in order:

| Run | Requests | Observer | Purpose |
| --- | ---: | --- | --- |
| initialization | 0 | lifecycle only | Load/import/repack/compile/allocation/warmup, build RSS and wall time |
| correctness | 1 | on | Continuous/chunked native oracle, all-layer KV and fixed decode comparison |
| profile attention | 1 | on | Norm/RoPE/cache and actual local/global attention |
| profile dense | 1 | on | QKV/O/shared dense projections and norms |
| profile expert | 1 | on | Router, quantization/packing/dispatch, FC1/activation/FC2/combine and correction work |
| profile head | 1 | on | Final norm/last row/tied head/soft cap/sampling |
| memory | 1 | on | Lifecycle and chunk/handoff/head/continuation memory ledger |
| timing cold | 1 | off | Declared cold request boundary, separately labeled |
| timing warm | 1 | off | Preallocated, warmed/reset sequence and clean allocator state |

A cell requires eight requests. The complete 11-cell matrix is 99 runs / 88
requests; this is a protocol size, not approval to run it. A smaller selected
slot receives a different scope and must have its own receipts and finite bounds.
Initialization has no prompt forwards/client token stream and is never averaged
into warm timing. Lifecycle stage durations are disjoint; compiler wall time and
RSS remain separate from those stage totals. Build wall time cannot exceed its
initialization interval. A cold timing process must start
fresh: after initialization/profile/memory processes are cleaned up, the launcher
starts its cold instance, then establishes a warmed/reset instance for warm time.
The client request clock determines whether load is inside cold TTFT; do not add
an unrelated lifecycle duration to TTFT. These lifecycle transitions are the
future launcher's responsibility, not evidence generated by a CPU schedule.

Warm timing uses the existing six monotonic host boundaries from PR17:
`request_accept`, `input_ready`, `prompt_begin`, `prompt_complete`, `kv_ready`,
`first_token`. Whole-prompt latency includes every chunk and interchunk gaps and
ends before first-token head/sampling. TTFT includes input work, head, sampling
and client serialization; handoff is separate. If the target cannot expose these
exact boundaries, refuse this record schema and review a separately labeled
combined scope. No server-forward duration is relabeled as pure prefill.

The payload uses exact token IDs, no extra special tokens, one streamed request,
greedy fixed seed, 256 outputs with EOS ignored and reported usage. No routes,
activations or reusable prompt K/V are precomputed; no forced expert or allowed
output-token control is inserted. The client records each SSE data payload with
a local monotonic receive time. It requires a single stable response ID, choice
index zero, valid token IDs, exactly 256 outputs, a length finish, exact prompt /
completion / total usage, ordered events and a final `[DONE]`. Every timing-run
SSE event, including tokenless metadata, must occur at or after request acceptance.
Multitoken chunks
are flagged and cannot supply per-token ITL. The explicit run/request ID joins
server forwards; server response ID joins streaming packets. Network transport
and endpoint selection are intentionally left to the future reviewed adapter.

## Normalized result contract

The source code and generated synthetic CPU tests are executable schema examples.
Private prompts use `schema: megartx-prefill-private-prompts-v1` and
`token_ids: {p2048: [...], p8192: [...]}` containing exactly the selected prompts.
ID hashes use ASCII JSON with sorted keys and compact separators (for lists this
is `json.dumps(ids, separators=(",", ":"))`), then SHA256. Values must be integer
vocabulary IDs, with booleans rejected.

Results use `schema: megartx-prefill-records-v1`, selected `scope_sha256`, one
`host_clock_id`, explicit provenance (`synthetic_cpu` or `recorded_target`,
sanitized artifact hash and origin reference) and exactly one ordered record per
job. Preserve private raw artifact provenance outside the public repository.
Synthetic test records have no observed target source or measurement provenance.

| Record | Checked content |
| --- | --- |
| Run identity | Exact run/request/mode/region/state/observer, complete status, no stop reason; serial host begin/end; declared retained trace bytes |
| Launcher lifecycle | Per-run startup/cleanup artifact digests, exact launcher source and ownership scope, owned-only and cleanup-complete assertions |
| Forward | Exact count, chunk index/start/end/M, original per-span token hash, no output rows, 30 layers, completion |
| Correctness | Pinned oracle/decode identities, finite intermediates, unchanged natural routing, cache/logit comparison assertions and comparison receipt |
| Handoff | Complete/nonpoisoned BF16 state, committed length/next absolute position, P+256 capacity; exact 25 local/5 global layer intervals and processed K/V/layout digests |
| Timing | PR17 whole-prompt coverage and ordered boundaries; first-token clock matches client; warm preallocation/warmup; no profiler/memory/correctness capture or trace |
| Profile | Explicit runtime-stream to CUPTI-stream mapping with pinned provider; unique launch and kernel IDs, source-bound registry/site, layer/chunk/component and M/N/K; kernel names, grid/block, device interval and bytes |
| Experts | Every layer/chunk has 128 selected/positive/scheduled-M counts, route digest, correction hits and suppressed-stock rows; selected sum is M*8, each selected count is <= M, positive does not exceed selected, scheduled covers positive and may exceed M for padding |
| Memory | Load/import/compile-warmup/every chunk/handoff/head/fixed-continuation allocated/reserved/device-used peaks; uniquely owned physical allocation IDs/categories; scratch/reserve and BF16 capacity floors |

Attention/dense/expert profile records must cover all selected layer/chunk pairs;
head records cover only the last forward/layer 29. Fusion can combine several
operators in one source-bound kernel. The schema validates correlation and scalar
shape/count fields; it does not infer kernel arithmetic, tile/calibration domains,
route authenticity or unrecorded work from a name. The exporter/reviewer must
retain actual selected/positive/scheduled expert M, padded N/K, projection and
gate/up scale segments, quantizer globals, scratch last use and adapter redundant
work in private source artifacts. Unreported detailed attribution cannot support
a tuning decision. Stage summaries remain partial whenever the exporter omits
those artifacts. They are not G7 acceptance or a substitute for an oracle.

CPU-launch correlation IDs are independent of device timing. GPU intervals may
occur after CPU annotations. Runtime stream keys are **not** assumed equal to
profiler/CUPTI IDs. Kernel sums by component and merged busy intervals per stream
are reported separately. The critical path is explicitly unknown: dependency
events, overlaps, scheduling gaps and handoff/head allocation must be reviewed
before any critical-path cost or Amdahl claim. Host/device clocks are never
subtracted or merged. Client and host timing clocks must be the same target-local
monotonic clock identity; an SSH or remote client clock cannot substitute.

Lifecycle artifact digests are required even for initialization. Their contents
and actual owned cleanup must be verified by the future shared launcher before
the next run; this offline parser only verifies their declared identity and
completion fields. A missing/false cleanup cannot parse as a complete run.

Memory categories are model/scales, retained originals, local/global KV,
activations, route/packing, scratch, conversion and allocator other. Count tied
or aliased storage once per physical allocation. The ledger may leave measured
allocated bytes unattributed; that slack cannot be claimed as recovered memory.
Allocated <= reserved <= device-used <= device-total is required for these
normalized peak samples. BF16 payload lower bounds after preallocation are
25*8192 bytes per local stored position and 5*4096 bytes per global position.
`full_context` local allocation covers P+256. Only an explicitly reviewed
`bounded_window` policy uses the largest chunk-required union
`end - max(0,start-1023)` and a 1024-position continuation window. Global capacity
always covers P+256. Both payload floors round up to the pinned page size.
Logical payload floors do not prove allocator layout/page-padding,
staged K/V coexistence, conversion, resident fit or native storage semantics.
Measured allocations and fit review must cover those separately. Scratch must be
bounded and device-total minus device-used must retain the frozen reserve at
every phase. Trace bytes and suite host wall/request counts are bounded in total.

## Stops and remaining work

Stop before launch on missing or mismatched sources, final launcher support,
numerical/fit/dispatch/ownership/resource/cleanup receipts or unresolved output
policy. During the future slot, stop on first source/dispatch drift, missing
forward/token/stream/route coverage, nonfinite/oracle failure, partial cache,
position/capacity error, reserve/scratch/build/request/wall/trace breach,
ownership ambiguity or failed cleanup. A failed run retains its failure receipt;
it cannot become a successful normalized record by dropping the offending row.
Only owned work may drain. Poison uncertain cache and rebuild through the
verified library fallback using the committed token log. No next job starts
until owned cleanup is verified by the shared launcher.

This task delivers CPU protocol and record validation, not GPU observers,
projections/attention/expert/head kernels, model numerical results or a live
launcher. The parent's source/interface review, the prefill owner's PR15 pin
reconciliation, final PR15 independent review, complete admission evidence and a
future sole-owner handoff remain blockers. Further native instrumentation or
launcher integration requires a reviewed scope and file-ownership plan first.

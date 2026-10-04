# Exclusive native receipt integration contract

This CPU implementation composes reviewed PR24 `9fc367d6ce0fc4e442a612231752a66344460085`
and PR22 `cf656d6632b9f1b08019a527a9269a9ed0fb0a26` in merge `dc37465`.
It has not loaded a model or run CUDA. Shared plugin/launcher integration remains
with the prefill owner. The source manifest is unchanged; the newly inspected
DLPack ABI/helper hashes are installed-source observations, not loaded-binary
equivalence claims. The [zero-forward protocol](speculative-native-zero-forward-protocol.json)
is the review boundary.

## Exact shared-owner callback and arguments

The general plugin calls the following before its installed-idempotence return:

```python
from .speculative_native_lifecycle import install_native_diagnostic
install_native_diagnostic()
```

No environment flag returns `False` before runtime imports or mutation. The
explicit branch requires `MEGARTX_NATIVE_DIAGNOSTIC=1`, native corrected scale
mode, the exclusive existing plugin allowlist, and M1 preparation off. Validate
mutually exclusive prefill/native diagnostic modes before this callback. Do not
combine their capabilities. Use:

```text
--worker-extension-cls megartx.speculative_native_lifecycle.NativeDiagnosticWorkerExtension
--no-async-scheduling
```

Pinned `EngineCore.__init__` calls general plugins at its beginning, before the
executor or scheduler exists. Registration validates the 22 source hashes and
all class/duplicate checks before patches. It observes that exact active
constructor frame for the first core; future cores enter the installed init
wrapper. A late call cannot adopt a loaded core. First utility dispatch completes
the observed birth ledger. Requests are counted from preprocessing onward.

Registration also compares every replaced method with code compiled from pinned
bytes without executing imports, including callable globals, qualified name,
defaults, closure and inherited guard identity. A matching filename cannot admit
a partial constructor replacement. Active-core birth identity/state is prepared
before any hook write. Installation failures restore only owned hook writes and
preserve the original error. These are the narrow corrections to the two
independent registration findings at `acb9ce3`; runtime ownership/resource scope
and shared-owner APIs are unchanged.

The actual synchronous utility client sequence is:

```python
client.call_utility("pause_scheduler", "keep", False)  # await completed Future
summary = client.call_utility("megartx_owned_native_receipt", receipt_admission)
client.call_utility("megartx_owned_native_release")
client.shutdown()  # engine remains paused throughout
```

Only one fresh lease is allowed. Empty scheduler, input queue, pending adds and
batch queue are checked under the ingress lock. Prefix caching, transfers,
speculation, DBO, v2 runner and asynchronous scheduling are rejected. The engine
never resumes or serves after receipt entry. Idle busy-loop steps return without
entering the original runner; request ingress and unrelated utilities are sealed.
On a protocol violation, owned launcher teardown remains mandatory.

## Admission, ownership and cleanup

`check_receipt_admission` requires the exact implementation binding (three new or
corrected modules), zero protocol hash, current four adapter hashes, immutable
checkpoint revision/manifest/tokenizer evidence, independent review, parent
source/protocol acceptance, sole-owner slot, bounded monotonic deadline and
verified owned compiler/startup/cleanup controls. These fields convey prior
authorization and evidence; generating them does not create authorization.
`scripts/run_speculative_native_probe.py` freezes hashes on CPU and offers no
GPU execution flag. The sibling launcher must enforce the existing process,
compiler and wall/resource controls before model startup.

EngineCore obtains actual refcount-one objects using `BlockPool.get_new_blocks`,
reserving each group's `ceil((2048+4)/B)+1` capacity. Object references remain
held across receipt and any later admitted probe. The worker receives a nonce,
PID/start-time and purpose-bound ticket. It binds that ticket to its actual
`Worker.model_runner`; an observation `OwnedFrame` cannot grant this lease.

Release RPC first synchronizes the bound worker device and revokes the worker
capability. Only the confirmed `{drained: true, revoked: true}` result permits
`BlockPool.free_blocks`. A failed/uncertain RPC, drain, revocation or ownership
check retains the objects and poisons the engine for owned teardown. A second
release cannot free again. Collection failures preserve the primary error and
attach any cleanup uncertainty. No cache clearing or allocator caching changes
are used by the receipt.

## Zero-forward receipt and precise remaining fit blocker

The collector binds real cache placements, raw storage pointer/extent, layer and
block strides, manager/kernel/builder block sizes, and every reserved page's
physical bytes. Whole views may overlap. Owned dense page byte intervals must
be disjoint in the actual placement. Manager/kernel splitting and sparse strided
page holes are rejected pending separate adapters. Shapes alone grant no alias
authorization. Historical LBNHC pages are supported by host addressing tests;
the historical 4,995,152-byte lower bound is not a current fit receipt.

Existing initialized attention/wrapper/XQA fields, FlashInfer buffer cache and
vLLM workspace manager are read without lazy workspace getters, planning,
autotuning or preparation reruns. Backing storage is deduplicated by device and
pointer. CPU and GPU baseline extents are separately ledgered. Allocated,
reserved, cached slack and pre-existing peaks come from one selected-device
`memory_stats_as_nested_dict` query, without enabling history or resetting peaks.
The global allocator snapshot is not requested: segment count/hash are `null`
with `not_collected_unbounded_global_snapshot`, and coverage explicitly covers
aggregate counters only. Missing aggregate counters are an error, not zero.

The narrow correction after `95f3109` replaces whole-receipt `json.dumps`/encode
materialization with plain-primitive structure and exact byte-count preflight,
then streaming canonical SHA256. Sorted keys, compact separators, ASCII escaping
and finite-number policy preserve the former digest for accepted metadata.
Preflight limits depth to 16, values to 131072, container entries to 8192, scalar
strings to 4096 characters and integers to 256 bits. It counts bounded scalar
tokens and container punctuation, rejecting before hash encoding or full worker
RPC return if serialized evidence exceeds 8 MiB. Excess is never truncated.

Targeted acquisition checks thirty cache owners and at most thirty groups,
placements and attention builders before inspecting tensor storage. The reviewed
16/32/64 manager sizes permit at most 130 pages per owner, 3900 page records total.
Workspace acquisition limits records to 512, visited objects to 256, fields and
workspace/cache containers to 256, labels to 4096 characters, tensor dimensions
to 64 and the process-map read to 1 MiB. The existing nested-owner scan depth of
three remains its scope. These limits can reject a legitimate topology; the
decision requires measured resource review, not silently enlarged limits.

The **8 MiB serialized evidence cap is not an 8 MiB Python acquisition-heap
cap**. Object graphs, sorting and individual bounded encoding tokens have host
overhead. The **8 GiB host-free floor is a separate reserve**, checked before
targeted collection and before/after the allocator query. It is not an allocation
ceiling or a measurement of transient peak host use.

Read-only inspection of installed `torch/cuda/memory.py` binds SHA256
`7dc1d9d2a00b571977e6ecc6997d7ccfa4f13b620c11faf1f3f82317c548ec99`.
Its public scalar getters call the flattened `memory_stats` route. Using one
nested query avoids those repeated queries and flattened private-pool copies,
while retaining useful aggregate counters. Source controls require a fresh,
empty, eager single-worker lease, preventing serving-history accumulation and
normal model graph capture. They do not establish zero private pools from other
startup/library paths. The native `_cuda_memoryStats` result still materializes
private-pool records before Python can guard their count. The 64-pool guard is
explicitly **after acquisition**; the receipt records observed count (or `null`
when absent), host readings and `native_query_preallocation_bound_bytes: null`.
No fixed native query host-peak bound is exposed or inferred. This concrete
limitation remains for independent and owned startup/resource review before any
execution. Neither reduced snapshot coverage nor these host readings can admit
the native probe or resolve its FFI/temporary/external allocation blockers.

The separate correction after `b9f8e95` distinguishes installed distribution
metadata (`torch` version `2.13.0`) from the loaded runtime build
(`torch.__version__` literal `2.13.0+cu130`). Read-only source inspection matches
the client owner's proof: `torch/version.py` SHA256
`d7662da37d4b8b037c81e7ae20381a43893b379facf17988a6d4f17a93265140`,
CUDA build `13.0`, Git revision `cf30153c4c131c8164ee7798e5022d810682e2cb`.
The shared `check_torch_build` guard hashes that source and requires all of these
identities exactly in both the receipt and existing gated probe. It compares the
base literal string of the `TorchVersion` subclass, preserving the full CUDA
suffix without normalized equality or prefix acceptance. The private receipt
records both identities; utility/admission arguments and the scalar allowlist
are unchanged. The [version correction evidence](../evidence/speculative-native-torch-identity-cpu-validation.json)
is a source/CPU freeze, not a runtime-build measurement or GPU qualification.

Existing startup model frames come from the 30 routed dispatch counters and must
fit the existing eleven-frame allowance. Six forced expert fixture pairs and
their measured correction rows are recorded separately. The receipt invokes no
model/head/cache writer. Its `diagnostic_target_forwards` is zero. Existing model,
cache and startup workspaces remain a separate baseline; private pages, native
head/hidden, metadata, temporary work and conservative cached slack are charged
to the unchanged 8 MiB incremental verifier cap.

The pinned DLPack 1.3 capsule table is read without invoking its allocator. The
callback address and executable mapping's file digest are captured when present.
That observation does not prove the loaded argument exchange route or transitive
Torch allocation coverage. The original CUTLASS MoE call omits `workspace_buffer`;
its FFI binding allocates workspace per call. Actual M1/M2/M256 required-size
queries can be device/JIT dependent and are deliberately outside this zero-call
phase. FP4 fallback allocations and CUDA module/library/static/workspace paths
also lack an enforced residual external bound.

Therefore this collector's live decision remains **unadmitted**: original-lane
temporary bound, FFI argument-route coverage and residual external CUDA bound
are explicit blockers, not fabricated zeros. It still produces useful actual
geometry, owner, baseline and known-slack rejection evidence. If cached slack
alone exceeds the cap with known buffers, the scalar decision reports the exact
known excess for resource review; it does not raise the cap or disable caching.

The later utility `megartx_owned_native_probe(prompt_ids, probe_admission)` exists
but requires the retained live receipt digest and a separately admitted live
decision. No supplied boolean or substituted receipt can override unknown fit.
Concrete source-bound bound/enforcement work must be reviewed before enabling
the existing 85 diagnostic calls plus at most eleven original startup frames,
total at most 96. The original protocol, 2 GiB GPU reserve, 8 GiB host floor,
shared 2 GiB/300-second compiler limit and 1800-second wall bound remain intact.

## Publication and claims

EngineCore retains the full receipt privately. Its public result is an explicit
scalar allowlist plus receipt digest; no page tables, pointers, prompts, tensors,
snapshot or logs are returned. Native correctness, allocation coverage and fit
remain unexecuted. CPU fixtures exercise lifecycle failures and host metadata;
they are never accepted as native EngineCore, worker, runner or cache owners.
No benchmark, quality or performance claim follows from this implementation.

Client task `01a10651-18d3-7268-86df-bee5b2989041` keeps the same callback,
worker class, utility arguments, admission keys and scalar allowlist. Its bounded
writer must bind the corrected committed `implementation_binding` and new
`zero_protocol_sha256`; old admission hashes must fail. Do not import sibling
uncommitted files. Receipt errors use the existing confirmed-drain cleanup;
unknown drain still retains references for owned teardown. The separate
[bounded-receipt CPU evidence](../evidence/speculative-native-receipt-bounded-cpu-validation.json)
records this correction for independent review without granting a GPU slot.

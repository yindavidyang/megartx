# Explicit V2 zero-forward receipt client

This source-only change extends the existing bounded client with an explicitly
selected V2 lane, based on reviewed ownership head
`95f7b20d91b08ed83e81fd3d9af82b91f3f6a31e` (tree
`d6e5157e8cb9347a7276d35b021b7d118fdf358f`). The later shared composition preserves this reviewed owner ancestry and the
explicit client, then adds only purpose selection, plugin registration and
launcher delegation. It does not run a target host, import a model runtime,
or authorize GPU work. Exact-composition independent review and CI remain
mandatory before a parent can authorize a GPU slot.
Historical source pins and V1 receipts remain unchanged.

## Selection and source binding

Freeze requires an explicit `--runner-lane v2` or `--runner-lane v1-legacy`.
The V2 client plan and authorization have distinct schemas and purposes.
Validation reconstructs the clean committed plan and compares canonical values,
including its exact source head/tree, owner-plan digest, engine kwargs/argv,
installed source manifest, collector guard, limits and implementation hashes.
The supervisor and owned child independently validate authorization before any
runtime import, GPU query or acquisition. EngineCore and the worker independently
reconstruct the V2 client/owner binding before reserving or collecting.

`validate_actual_config` requires resolved `use_v2_model_runner is True` before
`EngineCoreClient.make_client`, plus the V2 worker extension and the reviewed
target-only synchronous/eager configuration. It never sets
`VLLM_USE_V2_MODEL_RUNNER`. The actual loaded runner, getter code, config, device,
request state, block tables, metadata builders and allocation owners remain bound
by the reviewed V2 owner implementation. A frontend selection observation alone
is not evidence of a loaded runner. The V1 legacy path keeps its resolved-False
guard and its original utility names.

## Shared integration contract

The shared owner must compose the exact committed DSpark head, then select only
one lifecycle/evidence pair. For V2:

1. `install_native_v2_diagnostic()` from `speculative_native_v2_lifecycle`
2. `install_native_v2_receipt_evidence()` from `speculative_native_evidence`

The opt-ins are `MEGARTX_NATIVE_V2_DIAGNOSTIC=1` and
`MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE=1`. Both V1 flags, M1 preparation and the runner
selector override must be absent. The shared plugin calls this pair before its adapter-idempotence return.
`run_scale_validation.py --client native-v2-receipt` delegates directly to
`speculative_native_receipt_client.py` before HTTP imports, output creation or
server acquisition. The V2 frozen plan, schema and
[composition protocol](native-diagnostic-composition-protocol.json) bind both
public source parents, the exact flag pair, worker extension, launcher purpose
and source vector. V2 `preflight_blockers` is empty only in this source
composition. Authorization cannot override a nonempty blocker list or replace
the dedicated client’s source, review, CI, slot and lifecycle checks.

Prefill read-only observation, legacy V1 receipts, V2 receipts and native M1
preparation are mutually exclusive. Unknown, partial, mixed or inherited
diagnostic flags fail closed. Prefill never registers a mutation utility or
grants a lease. The legacy V1 plan keeps its historical blockers; its presence
is not V2 authorization. The actual loaded V2 runner remains checked separately
from frontend selection.

## Owned session and evidence

The one-use sync/async session snapshots admission before pause, awaits
`pause_scheduler("keep", False)`, calls only the V2 receipt utility, validates the
V2 allowlisted summary, calls only the V2 release utility, then shuts down.
It never resumes, prepares a batch, serves a request, or calls a verifier/drafter.
Fresh EMPTY EngineCore ownership and real BlockPool references remain enforced
by the lifecycle. A receipt/release timeout or cancellation does not trigger a
second uncertain RPC. A known post-receipt writer failure uses the existing
retained V2 lease's drain path; uncertain drain retains original references.

The evidence hook wraps the already registered V2 utility only. It obtains the
raw receipt from the existing live lease, writes immutable mode-0400 files in an
owned mode-0700 directory, and returns the allowlisted scalar map. The V2 evidence
identity has a separate schema/purpose and a V2 runner-policy binding. Offline
comparison verifies receipt bytes/digest, worker identity, real cache-page
geometry, source/build identities, metadata-owner identities and allocator
observations. V1 scalar/raw/identity substitution fails closed.

The canonical receipt cap is still 8 MiB. The one native allocator-counter query
has an explicitly unknown native pre-acquisition host peak. The monitored 8 GiB
host reserve before and after that query is preserved; it is not a hard native
heap bound. The comparator rejects a fabricated bound, wrong query count,
unmatched device, or missing host-boundary evidence. No global snapshot is added.

All fit/probe/drafter decisions remain false. FFI exchange coverage, native
M1/M2/M256 temporary bounds, external CUDA allocations and the future V2 verifier
interface remain unresolved. The selected future Makora draft's approximately
2.41 GB weights and draft KV need a separate persistent budget. They are not
charged to, or admitted by, the unchanged 8 MiB candidate/observer scratch cap.

## CPU checks and later source-only freeze

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m megartx validate configs
python -m unittest discover -s numerical_reference -v
python numerical_reference/nvfp4_reference.py --self-test-summary
python -m compileall -q src tests scripts numerical_reference
# After committing a clean exact head; no device query or runtime import:
python -S scripts/speculative_native_receipt_preflight.py --freeze --runner-lane v2 --client-mode async
python -S scripts/speculative_native_receipt_preflight.py --freeze --runner-lane v1-legacy --client-mode async
```

CPU fixtures check dispatch, source/schema/admission separation, sync/async pause
ordering, cancellation and no retry, immutable evidence, V2 comparison, monitored
counter-query semantics, and retained V1 behavior. They do not establish native
ownership, native fit, numerical correctness, speed, or runtime success.

## Shared launcher entry

After freezing the exact clean composition and obtaining independent review,
exact-head CI and an explicit parent-owned GPU slot, the V2-only entry is:

```sh
python scripts/run_scale_validation.py --label native-v2-receipt --mode native \
  --client native-v2-receipt --trials 1 \
  --native-receipt-plan /private/exact-v2-plan.json \
  --native-receipt-authorization /private/exact-v2-authorization.json \
  --native-receipt-directory /private/fresh-v2-receipt
```

The `--label` does not create a second output directory. The dedicated client
owns the supplied fresh private directory and its existing checkpoint-manifest
default; an explicit `--native-receipt-checkpoint-manifest` is passed unchanged.
No shared launcher switch can waive the client’s checks or select a target probe.

The latest prefill experiment on the inherited `418e1ec` source failed after
live V2 binding and geometry checks, with one observed frame and no fit evidence.
This CPU/source composition does not reinterpret that failed experiment or claim
its diagnosis resolved. A prefill runtime fix requires a separate exact delta.

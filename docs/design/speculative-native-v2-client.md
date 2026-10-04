# Explicit V2 zero-forward receipt client

This source-only change extends the existing bounded client with an explicitly
selected V2 lane, based on reviewed ownership head
`95f7b20d91b08ed83e81fd3d9af82b91f3f6a31e` (tree
`d6e5157e8cb9347a7276d35b021b7d118fdf358f`). It does not compose the shared plugin
or launcher, run a target host, import a model runtime, or authorize GPU work.
The source blocker remains until the parent composes and reviews exact heads.
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
selector override must be absent. The shared plugin and `run_scale_validation.py`
are not edited in this head. The frozen V2 `preflight_blockers` and corresponding
schema must be changed only as part of the exact shared composition and fresh
review/CI. Authorization cannot override a nonempty blocker list. Removing this
source blocker does not itself provide GPU-slot authorization.

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

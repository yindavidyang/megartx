# CPU-only route observation contract

## Status and boundary

`megartx.m1_route_receipt` is an additive, unimported, standard-library-only state
machine for testing a possible future provenance observer. It installs no hooks,
reads no environment switches, imports no Torch/Triton/CUDA package, changes no
runtime callsite, and supplies no native admission capability.

Every `RouteObservationReceipt.route_readback_required` is unconditionally true.
The property has no constructor switch or setter; Boolean conversion raises.
The existing route D2H, synchronization, dynamic range/uniqueness check, descriptor
qualification, diagnostic/external callbacks, fallback and failure paths remain
unchanged. Successful consumption means **supplied metadata matched**, not that
GPU production completed or that the IDs were validated.

No installed source hashes, compiler artifacts, CUDA functions or package
qualification are supplied or guessed by this module. The test keys are explicitly
CPU fixture labels. This change neither activates a producer contract nor changes
an asynchronous-error boundary.

## Why this boundary is useful

The exact Gemma4 CUDA producer in vLLM commit
[`ced6857afa0ea7b2e3f0846a62e1394e90f15607`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/gemma4.py#L98-L187)
packs a unique lane ID into every integer sorting key. For E=BLOCK_E=128 and K=8,
a correct completed execution selects eight distinct IDs in [0,128), even when
score bits tie or are nonfinite. That is an ID invariant only, not a route-weight
or model-numerics qualification.

It is insufficient to trust the model name, generic top-k, or a previous valid
route. The upstream
[BaseRouter callback](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/fused_moe/router/base_router.py#L287-L305)
receives mutable IDs;
[router selection](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/fused_moe/router/router_factory.py#L121-L235)
can choose simulation instead; custom routing or an instance override can return
arbitrary IDs; and modular prepare/finalize can substitute a new ID tensor.
Existing project binding checks do not authenticate all those boundaries.

Official Triton 3.7.1
[`JITFunction.run`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/triton/runtime/jit.py#L708-L763)
returns the compiled object it launches, but that object and its cache are mutable.
A future observer would need actual snapshots at both launch boundaries, including
initialized module/function/launcher identities, callable and cache selection,
arguments, device/context/stream, specialization and hooks. A post-return object
or hash comparison alone is not an immutable launch receipt. The official tag is
reference evidence, not an assertion that installed bytes or artifacts match it.

## Minimal interface

- `BindingSnapshot` holds retained, opaque expected objects for the registered
  runner/layer/router, selection/custom routing/closure, prepare/finalize, expert
  implementation, JIT entrypoint and initialized compiled launch. Object fields
  are matched with `is`, never user-defined equality. Source/cache keys are
  compared as exact plain strings; the module does not validate their provenance
- `GuardSnapshot` defaults every field to `UNKNOWN`. Capture state, callbacks,
  replay, EPLB, overrides, hooks, simulation, diagnostics and artificial routing
  must all be explicitly supplied in their supported state. Unknown never means
  absent or false
- `StreamSnapshot` retains an opaque context identity plus device and handle. A
  zero producer handle is permitted for a default stream; the consumer's handle
  must be nonzero to match the current native M1 scope
- `OutputSnapshot` retains the exact tensor owner and storage owner, address,
  offset, extent, shape, stride and type metadata. It performs no pointer query,
  memory access, alias discovery or tensor-content read
- `ProducerSnapshot` combines the above with T=1/E128/K8/BLOCK128/one-warp
  specialization metadata. These values must be observed by a future qualified
  extractor, not inferred from an arbitrary tensor

The current code supplies **no live extractor**. Supplying fabricated but matching
snapshots can produce a matching audit record. Tests deliberately mutate owner
contents to invalid IDs while retaining metadata, then confirm readback remains
required. This is why this interface cannot authorize a bypass.

## Lifecycle

A `RouteObservationLedger` retains expected identities for its lifetime. Each
`invocation(...)` opens a fresh opaque token with an increasing generation and
retains the originating `threading.Thread` object, frame and producer stream.
Only one scope is active in a ledger; nested and concurrent entry are rejected.
A numerical thread ID, tensor address or old generation cannot substitute for
the current token/Thread/owner identity.

The only successful sequence is:

1. OPEN: no producer has been observed
2. PRODUCED: `observe_production(token, before, after)` matches both supplied
   boundary snapshots and the same output owner/view
3. DEPENDENCY_OBSERVED: `observe_dependency(...)` records matching stream/context
   metadata after a future actual wait would have succeeded. It performs no wait
4. CONSUMED: `consume(...)` rechecks current bindings, guards, output/view,
   consumer stream and exact frame, then returns one audit record
5. CLOSED: originating-thread cleanup releases retained per-call owners

Missing, duplicate or reordered operations, stale/forged/foreign tokens, wrong
threads, identity/layout changes, nested invocation and exceptions poison the
ledger. Poison is terminal and survives cleanup; there is no reset API. Closing
an incomplete scope also poisons it. A foreign token/thread cannot consume or
clear another active scope. Owner cleanup is idempotent after release, including
poisoned release.

Unsupported or unknown provenance **before opening** rejects observation without
poisoning the otherwise unused ledger. This differs from a broken active
transaction. A future caller must retain the ordinary checked path for unsupported
input and cannot turn partial submission or a broken active transaction into a
fallback/retry decision.

Primary exceptions, including BaseException subclasses, propagate unchanged.
Cleanup failure poisons the ledger, releases only its own originating-thread
scope, and adds a note to a primary exception when supported instead of replacing
it. Strong output/frame references are retained until cleanup; audit records do
not retain them afterward.

## Future work remains separately gated

Before any live observation integration, independently review the actual extractor
and source/callable/closure/compiled-kernel/loaded-function pins, both launch
boundaries, exact stream waits, ownership, all alternative routes, and current
native lease matching. The current module is not a replacement for those proofs.
No prefill purpose/provider integration or shared validator change is included.

Before any route-copy/fence bypass, independently review the complete qualified
producer lane and a concrete final-completion/poisoning protocol. Removing the
current fence changes the point where asynchronous CUDA failures are surfaced.
Metadata matching does not preserve today's pre-map fault timing. Diagnostic,
external, forced, controlled and unproven paths must retain the complete current
checks. Do not infer graph replay eligibility or performance improvement from
these CPU tests.

## CPU checks

Run `PYTHONPATH=src python -m unittest discover -s tests -p test_m1_route_receipt.py -v`.
The tests exercise successful lifecycle, strict unknown handling, each identity
and guard, pre/post launch changes, shape/layout/extent, same-address owner
substitution, generation reuse, ordering, thread/context/stream/frame mismatch,
primary and cleanup exceptions, terminal poison, owner release and source-level
runtime disconnection. The unchanged native preparation header remains covered
by the existing CPU fault harness and additional negative-ID/ID=128 fixture cases.

# Disconnected actual-binding adapter and shared route dispatcher

## Boundary and base

This additive CPU-only proposal starts at exact main
`3bd3e1c0db92388efb048530bc96c81c2f2c8acb`, tree
`d136dbbe47469b4bac1d42c495fea401f9b821cb` (merged PR40).
The original receipt ledger, observer prototype, their tests/fixtures/docs,
production modules, launcher, native source, package configuration and all
existing source checks are unchanged. Only new probe, test, fixture and document
files are added. Nothing installs automatically or runs on import.

Every returned audit/receipt permanently requires the existing route D2H,
synchronization and native checks. No receipt is Boolean dispatch eligibility.
No route check is removed, bypassed or replaced; there is no graph, asynchronous
fault-timing, GPU correctness, completion or performance claim. No runtime
integration, device query, CUDA/Torch/Triton import, warmup, real compilation,
package change or model access occurs in this work.

## Actual-object extraction

`ActualBindingReaders` / `make_readers` take retained actual layer, runner,
routed-adapter, controller and producer owners, explicit metadata interfaces,
a shared dispatcher, and already observed boundary data. They return the merged
`ObservationReaders` callbacks. They do not find substitutes by names, import a
vendor package, rerun routing, replay a binder, scan a cache or initialize handles.

The readers check these ownership edges against current objects:

- Current forward-context registry is the captured registry; the layer-name
  entry is the supplied runner; runner → routed layer → quant method and
  captured owner/kernel identities satisfy the existing integration contract
- The routed adapter has both the expected function and the exact layer as its
  bound self. The actual router/custom callable and Gemma `self` closure cell
  agree; that closure owner owns this runner and the same scale object
- The closure's actual routing-function global points to the observed function,
  whose actual JIT global points to the retained shared JIT. Callable code,
  defaults, closure cells and referenced globals are retained and rechecked
- Actual modular implementation, experts and prepare/finalize ownership and
  current callable identities agree. No prepare/finalize function is executed
  merely to fill an observation
- Actual plugin adapter closure cells expose controlled, diagnostic and
  validation state, all three observer callbacks, route-audit state and the
  same controller. Missing cells remain incomplete. Controller diagnostic,
  external-observer and artificial-route-control state, plus a current forced
  fixture marker, cannot be hidden by a supplied false boundary flag
- Capture/replay/EPLB, actual instance overrides, pre-run hooks, exact mutable
  HookChain type/call/direction/list, separate compile hooks and instrumentation
  must all be explicitly absent. Missing runtime or plugin observations are
  unknown; absent environment variables do not establish absence
- Exact `controller.forward` identity and its generation are retained for the
  invocation. A refreshed boundary cannot legitimize a changed generation in
  the same frame object. Boundary records are tied to the originating Thread

The tensor reader accepts an explicit finite set of exact classes (the CPU
fixture models separate Tensor and Parameter owners), never arbitrary
subclasses. It reads the actual dtype/device/shape/stride, data pointer,
storage pointer/extent, offset, element size and element count through the
source interfaces also used by `m1_live.tensor_view`. Metadata method/type
replacement is rejected. It retains the actual storage object and requires
stable ownership across reads, validates extent relationships, and retains and
compares storage/view metadata for all four producer tensor arguments across
the launch. It never reads tensor contents or synchronizes.

The merged CPU profile still explicitly requires f32 score/scale/weights and
int32 IDs for `(T,E,K,BLOCK_E)=(1,128,8,128)`. The reader can describe an explicitly
supplied bf16 dtype, but this does not admit it to that unchanged profile. An
unseen installed scale dtype is not assumed f32. Another specialization needs
source-qualified tests and a separate profile review.

Stream extraction resolves the exact inner launch handle against already
observed, unique device/context records. It does not consult an outer/default
current stream. Dependency notification occurs only after the existing wait
has succeeded; the observer never adds that wait.

## One shared dispatcher

`SharedRouteDispatcher.register(observer)` returns runner/select/JIT/launcher
wrappers without setting attributes. The caller would require a separately
reviewed integration before installing any wrapper outside CPU fake objects.
The dispatcher owns one exact original/wrapper pair per shared JIT `run` and
already initialized kernel `_run`. Thirty runner/closure profiles therefore
share two producer wrappers, rather than stacking thirty wrappers.

An active owner-thread registered runner selects its profile. Unrelated global
calls and foreign-thread calls with no matching runner scope delegate to the
original once and remain unobserved. Explicit stale/foreign dependency or
consumer events cannot validate an active scope. Nested/concurrent registered
runners terminally poison the shared observation lifecycle. All in-flight calls
are tracked until they drain, including when the first owner returns early;
a third runner cannot acquire a complete observation during that interval.
There is no reset API. Ordinary delegation continues after observation poison.

`dependency_observed(scope, producer, consumer)` and
`checked_consumer(scope, actual_ids, frame, consumer_stream)` dispatch to the
active profile. The consumer facade contains extraction failures inside the
observation. Its metadata return must never gate the caller's original
readback/fence/native checks, incumbent call or cleanup. No receipt is passed
to native admission.

Wrapped original arguments, results and primary exception objects are preserved
and delegated exactly once. Observation failure never retries or substitutes
execution. Owner-thread observation cleanup releases the same ledger context
even if an audit cleanup callback raises before entering ledger cleanup. It
never cleans another thread's context. Storage references are in the merged
observer's finally-cleared retained-owner list.

## Honest cache-selection candidate

`build_cache_selection_candidate` in `probes/m1_route_cache_seam.py` returns an
**uninstalled reference-only** plain function. It accepts exactly the reviewed
Triton `JITFunction.run` slice and hard-pinned SHA-256
`8686893714b9aef17074011b857f5a40eab9a545a35976d7eea895b7d352221d`.

Its sole executable AST delta wraps the result of the original
`kernel_cache.get(key, None)` expression. Python evaluates that original lookup
once, before the helper reports actual cache/key/selected/target/specialization/
options locals. The original binder and key calculation execute once in the
original body. No post-return scan, second lookup, recomputed binder or
`kernel.hash` masquerades as selection evidence.

An independent restoration function removes exactly that permitted expression
delta and compares the complete original AST, including signature/defaults,
control flow, exception paths and returns. The source-byte pin is checked
separately. Any different upstream source requires another review. A lookup
exception bypasses the callback; original compile/launch errors preserve their
identity and context. A callback exception is swallowed solely for ordinary
execution and recorded as a scalar failure count/constant marker, never an
exception/traceback that retains invocation owners. Such a failure invalidates
any evidence using the candidate. Callbacks must be read-only; raw Python object
references cannot sandbox a malicious callback.

Cold/new/unrecognized selections follow the original body's path once and stay
incomplete, even if that body normally compiles/initializes and returns a kernel.
The candidate is not installed in Triton and is not proof that any loaded JIT
matches its source. The CPU integration control uses an ordinary counting cache
with no observation callback, so only the candidate's actual lookup delta can
supply selection in that test.

## Three distinct kinds of evidence

1. Source-file evidence: the new 25 literal slices come from seven exact pinned
   vLLM files. Full-file and slice hashes/line ranges are recorded; recovered
   bytes match the saved source-intake digests. The unchanged observer fixture
   supplies the actual extracted Triton JIT/launcher/HookChain bodies
2. Current Python owner evidence: the readers retain/check supplied actual
   objects and callable identity/code/global/closure relationships. These are
   controlled-worker observations, not a security boundary against arbitrary
   Python mutation or a claim that a file hash attests loaded bytecode
3. Live selected/loaded native evidence: actual installed selection, native
   launcher annotations/signature, loaded CUDA module/function, stream/context
   and launch success remain unobserved here. Cached file/cubin hashes cannot
   establish them. The existing observer intentionally rejects opaque native
   PyKernelArg/signature structures; no native memory is inspected

`SourceFileAttestation` cannot fill any loaded binding. The already materialized
JIT source key is read from `jit.__dict__['hash']`; its lazy `cache_key` property
is never touched, and this source key is never used as actual selection.
`bindings().cache_key` always remains UNKNOWN. Only the source-bound lookup
notification can fill an invocation's selection key.

A workspace replacement during development required reconstructing unpublished
files and cloning the exact public base. Source bytes were reread at their
pinned revisions and matched all retained source hashes. The manifest clearly
identifies the restored intake artifact rather than claiming the unavailable
original whole-artifact digest. Final checks must run on the recovered frozen
revision; pre-recovery results are not final evidence.

## Remaining shared integration seams

Prefill owns shared runtime integration. This proposal changes none of it.
A later separately reviewed installation would still need:

- Qualified loaded-object/source evidence and native launcher metadata
- An explicit observational purpose and lifecycle owned jointly with prefill
- Actual context/boundary/stream records and plugin closure access for the
  reviewed worker, without new CUDA queries to manufacture missing evidence
- Source-bound cache-lookup observation admitted for the actual loaded JIT
- Notifications at the existing successful wait and checked-consumer boundaries
- Default-off/source-catalog/lifecycle controls reconciled with prefill's final
  public revision, preserving every original route and native check

The current source-defined callbacks and CPU owners do not resolve those live
seams. Runtime wiring remains absent.

## Validation commands

```sh
PYTHONPATH=src:probes:tests python -m unittest test_m1_route_binding_adapter test_m1_route_cache_seam -v
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src:numerical_reference python -m unittest discover -s numerical_reference -v
PYTHONPATH=src python -m megartx validate configs
python -m compileall -q src tests probes
```

Controls cover thirty shared layers, interleaving, nesting, threads and overlap
drain; exact-source cache restoration and cold/new paths; registry/callable/
closure/prepare/scale/HookChain/cache/kernel substitution; actual inner stream;
frame generation; invalid dtype/shape/stride/storage/arguments; primary and
cleanup exceptions; strong owner release; native opaque metadata rejection and
runtime disconnection. All existing source checks remain unchanged.

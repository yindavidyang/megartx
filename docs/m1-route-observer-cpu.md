# Disconnected route-observer CPU prototype

## Status and boundary

This is a **reference-only CPU prototype**, based on main
`e4d82e5c3348c71dd31a76fac9ba4c2f25ab12eb` (tree
`e09f6ca40d8269ea61f3ddc135613cba6b9d7676`). It reuses the merged
`m1_route_receipt` ledger unchanged. It is not installed in the runtime package:
`probes/m1_route_observer.py` is imported explicitly by CPU tests only. Keeping it
outside `src` also preserves the ledger's existing zero-runtime-import test.

There is no plugin, launcher, `m1_live`, native, package entry point, environment
switch, global Triton hook, activation or runtime integration change. Factories
return wrappers; they do not set attributes or install anything. Tests install
returned wrappers on CPU fake instances only. Shared integration remains owned
by the prefill workstream and requires a separately coordinated review.

Every audit and every ledger receipt requires the existing route D2H readback,
synchronization and native checks. Neither is a Boolean eligibility token. The
observer adds no CUDA query, wait, allocation, compilation or fallback and does
not change the existing producer/consumer exception or cleanup behavior. A CPU
success establishes neither installed provenance nor GPU execution/completion,
correctness, performance, graph eligibility or a new asynchronous-fault contract.

## Interfaces

`RouteObserver(expected, reference, readers)` retains an explicit reference
profile and source-bound metadata readers. These are test/adapter inputs, not an
authentication boundary against arbitrary Python mutation. A controlled worker,
truthful inspected readers and a reviewed source-selection seam are necessary.
There is intentionally no helper that guesses an installed source key, tensor
owner, cache key, CUDA handle, target or stream context.

- `wrap_runner(runner, original)` opens a fresh observer generation around the
  registered runner's exact `_apply_quant_method` callable
- `wrap_select(router, original)` brackets the exact `select_experts` callable
  and retains its actual returned weights/IDs tuple
- `wrap_jit(jit, original)` brackets one exact Gemma JIT object's `run`, retaining
  all incoming owner arguments and verifying the returned compiled object
- `wrap_launcher(kernel, original)` accepts only an already-initialized `_run`
  callable. It captures actual inner grid, stream, function, packed metadata,
  enter/exit hooks and bound owners before and after delegation. It does not
  access the lazy `CompiledKernel.run` property to initialize a kernel
- `cache_selection_observed(scope, cache, key, selected, target,
  specialization, options)` records only the actual selection boundary
- `dependency_observed(scope, producer, consumer)` records the existing wait
  after it succeeded; it performs no wait itself
- `checked_consumer(scope, actual_ids, current_bindings, frame,
  consumer_stream)` compares the final actual IDs owner/view and consumes the
  unchanged ledger for audit. The existing D2H/fence/native check still follows
  regardless of its return value; no receipt enters native admission

Each wrapper invokes its exact original exactly once with the original argument
objects and returns the original result unchanged. Original exceptions propagate
with their identity. Observer errors invalidate/poison the audit; they cannot
cause fallback, a retry, a second binder call or suppress a producer error.
Unexpected observation or cleanup failures are terminal for that observer.
Unknown/unsupported metadata can remain incomplete without changing the checked
path. After a poisoned audit, the wrappers continue ordinary delegation but can
never complete another audit with that observer.

The registry strongly retains owner/original/wrapper pairs and recognizes only
its own exact wrapper objects. `__wrapped__` attributes confer no ownership.
Foreign instance overrides are preserved as mismatches. Callable code,
defaults/closure owners, prepare/finalize extras, initialized kernel/module/
function, launch targets and scalar launcher configuration are rechecked.
Per-invocation arguments, returned IDs, frame and output/storage owners remain
strongly retained until owner-thread cleanup and are then released.

The source-bound readers must supply current bindings, all guards, frame,
tensor metadata and extra callable bindings (including finalize). They must
report missing fields as UNKNOWN, not assume absent callbacks or overrides.
The current small CPU profile covers `(T,E,K,BLOCK_E)=(1,128,8,128)`, one warp,
f32 score/scale/weights and i32 IDs. A different dtype/layout remains incomplete;
this does **not** assert the installed model's scale dtype. Native annotation/
signature structures not represented by reviewed scalar reference fixtures are
not qualified by this prototype. Matching public source is not sufficient to
create a production profile.

## Honest cache selection and stream provenance

The wrappers do not observe the key passed to `kernel_cache.get(key)` by merely
seeing the returned compiled object. They leave it UNKNOWN unless the explicit
selection seam is called at that actual boundary. They never infer it from a
unique cache entry, `kernel.hash`, a post-return scan or a recomputed binder.
The source-extracted CPU fixture uses an instrumented fake cache's actual `get`
operation to report its selected key/object once. That seam is not installed in
Triton or implemented as production source rewriting.

A runner observer scope exists for every wrapped invocation. The unchanged
ledger transaction begins only once a real selection and inner launch have
supplied all mandatory fields; it remains bounded by the outer runner lifetime.
Missing selection, cold cache, async compilation, new specialization, unknown
hooks, lazy initialization, unqualified kernel, callable grid, warmup, missing
launch or duplicate launch cannot complete an observation. Ordinary JIT code
may still compile as it normally would; the observer never compiles or warms to
obtain evidence. A cold selection cannot be retrospectively made warm by the
returned object.

The producer stream is the actual integer passed to the inner launcher. Its
already-known device/context description must match that same handle. An outer
'current stream' is not used as a substitute. The actual passed function,
metadata and output argument must match the retained selected compilation and
JIT owners. Pre/post snapshots and the consumer snapshot preserve view identity.
Snapshots cannot prove tensor contents or detect a transient malicious mutation
restored between observations; the old checked route path remains authoritative.

## HookChain and callbacks

The reference Triton version uses `HookChain`, not None, for launch enter/exit
and kernel load start/end. The observer requires its exact expected class and
callable, retained chain objects, exact list-valued `calls`, empty copied calls
and correct forward/reverse direction at each boundary. Separate JIT cache,
post-compile, pipeline inspection and launch metadata callbacks must be absent;
instrumentation mode must be the source-defined empty string. Nonzero scratch
sizes remain incomplete because they would invoke allocator callbacks.

Nested/concurrent runner invocations invalidate the active observation. Unrelated
calls through shared JIT/launcher wrappers on another thread delegate without
borrowing its scope. Explicit stale/foreign scope events are rejected. Thread
ownership uses the Thread object, not a recycled integer ID.

## Exact-source fixtures

`tests/fixtures/route_observer_reference.py` preserves 22 original UTF-8 source
slices from nine vLLM/Triton reference files. Its manifest records source URL,
revision, full-source SHA-256, original line interval, slice SHA-256/length and
retrieval verification status. Loading uses AST-only annotation/decorator
removal and CPU fake globals; executable bodies are unchanged.

- vLLM reference revision: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
- Official Triton 3.7.1 reference revision:
  `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`
- Every fixture explicitly has `installed_bytes_verified=false`

Historical source associations and this reference are not a current installed
pin. Installed file/RECORD hashes, native extensions/compiler artifacts, actual
current knobs, selected compilation, loaded handles and worker process evidence
remain separate intake requirements. Missing cache artifacts do not prevent
these CPU tests, but prevent claims about an installed compilation.

Run the isolated suite with:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_m1_route_observer.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_m1_route_receipt.py -v
```

Run the repository's unchanged full CPU scaffold and independent numerical
reference checks before freezing. No device imports, installation, GPU/model
work, runtime publication or activation are part of this prototype.

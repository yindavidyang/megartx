# Natural route-kernel bootstrap: bounded wiring proposal

Base: merged PR42, commit `0c00ddd8e81fe9c1af335edf7ce57001dd31ddd0`, tree
`c0bb8407a154abac28228bbcf41ce4ecd2f93850`. This proposal changes no existing
file. It installs nothing and admits no runtime experiment. The accompanying
JSON freezes exact shared-file blobs/hashes and the relevant saved source vector.

## The concrete gap and addition

PR42 needs a known compiled kernel before `BindingRouteObserver` can register.
Its dispatcher deliberately ignores lookup notifications without an active
registered profile. The first naturally selected object therefore cannot be
discovered through that API alone.

`probes/m1_route_bootstrap.py` adds a small pre-registration listener. It reuses
the exact existing cache-lookup callback and original Python launcher call. One
instance covers the 30 natural runners and returns one shared JIT wrapper and
one shared `CudaLauncher.__call__` wrapper. It creates no receipt, admits no
native pointer, scans no cache, reruns no binder and never reads the lazy
`CompiledKernel.run` property. Its default-off factory does not inspect owners.

The ordered operation is:

1. Surround an original natural runner with `bootstrap.invocation(actual_runner)`
2. The shared JIT wrapper retains this invocation's actual argument identities
3. The existing source-bound lookup helper reports the original lookup result
   to `bootstrap.cache_selection_observed(...)`, before profile registration
4. The original `CompiledKernel.run` property may naturally initialize handles
5. The shared launcher wrapper matches its actual `self` to the selected
   kernel's existing `_run`, and matches the actual bound tensor objects to the
   JIT inputs. It calls the exact original launcher once
6. Only after that call returns does it attach the observed kernel/launcher,
   module/function/packed-metadata identities and actual inner stream handle
7. The original runner completes. Namespace and callback-failure checks run
   again; accepted overlap and terminal publication use one lock
8. `take_captures()` drains the captured stable owners once, after all runner
   scopes retire. These inputs are for a separately qualified later profile

No tensor contents, IDs, output pointers or invocation argument containers are
exported. Original exceptions and returns are preserved. No lock spans original
execution. A cold lookup remains incomplete even if ordinary compilation and
launch succeed; a later ordinary warm lookup can be observed without repeating
or initiating any work. Callback failure, wrong owners, namespace changes,
nested/concurrent registered runners and early close invalidate evidence.
Unrelated foreign-thread calls delegate once without joining the owner scope.

This listener and the admitted `SharedRouteDispatcher` are distinct phases,
not stacked per-layer producer wrappers. After bootstrap drains, installation
coordination must restore only its own two producer slots before one shared
qualified dispatcher owns all 30 profiles. The helper does not implement that
installation or admission decision.

## Facts available from the original path

| Fact | Existing origin | Treatment |
| --- | --- | --- |
| Actual cache, key, selected object, target, specialization, options | Original `kernel_cache.get(key, None)` and its surrounding locals | Capture at that lookup, never a later scan |
| Actual JIT argument owners | Entry to the original shared JIT call | Identity-match against original launcher arguments; drop at exit |
| Actual initialized launcher | Launcher wrapper's `self`, matched to `vars(selected)['_run']` | No lazy property read |
| CUDA function and packed metadata objects | Original launcher arguments 4 and 5 | Identity-match before/after original call |
| CUDA module object | Selected kernel's existing `__dict__`, after natural initialization | Retain observed value; missing remains UNKNOWN |
| Inner launch stream handle | Original launcher argument 3 | Retain exact integer, including a possible default-stream handle; no context inference |
| Host launcher returned normally | Return from that exact original call | Host-return provenance only; not GPU completion |
| Python module/function/global/builtin ownership | Actual supplied modules, `sys.modules`, function globals and effective builtins | Retain/recheck identities and presence, including absent globals |

The imported `_FunctionSeal` retains PR42's absent-global and effective-builtin
guards. Actual function globals must be the actual module dictionary; a cloned
namespace alone is rejected. This establishes current Python ownership
consistency, not source-to-loaded-code or native authenticity.

## Facts still missing, with no substitute in the launch return

- Producer-time CUDA context and capture state are not arguments or results of
  this Python launcher call. Its returned kernel/module/function/stream objects
  cannot manufacture them. The later native eligibility queries and invoke-time
  capture check cannot retroactively validate the routing producer boundary
- A stream handle does not establish device/context ownership. The existing
  producer/owned-stream objects can supply their observed handles, but the
  qualified context records required by PR42 still need an explicit source seam
- Opaque `PyKernelArg`/launcher-signature structures are not converted to trusted
  scalars. The existing observer's rejection remains unchanged. No native memory
  inspection is proposed
- Installed loaded-source correspondence, full guards/frame ownership, tensor
  dtype/storage/layout and specialization admission remain separate checks
- The supported receipt profile still requires f32 score/scale/weights and
  int32 IDs. Installed `per_expert_scale` dtype remains unobserved

`CapturedKernel.qualification_missing` lists these open gates. A capture cannot
be used as a Boolean, and neither the existing ledger nor any native interface
accepts it. It contains no reusable unchecked route/output cache.

## Exact coordination interfaces; shared files remain untouched

Prefill owns the shared plugin/controller/launcher. Proposed edits must be
coordinated against its eventual exact revision, using the hashes in
`route-observation-wiring-proposal.json` as the current baseline:

1. `vllm_scale_plugin.py:69–84`: add an explicit, exclusive, untimed observation
   purpose. Current admission allows controlled/normal/benchmark purposes only.
   Exclude controlled routing, normal diagnostics, external observers, profiling,
   prefill diagnostics/storage control and artificial route controls
2. `vllm_scale_plugin.py:230–244`: construct the single bootstrap only after the
   existing `prepare()` succeeds. Use its actual 30 layers and `binding()`'s
   actual registered runners. Do not rerun preparation or search by names.
   Preparation temporarily overrides routing and marks artificial fixtures
3. At the actual runner/JIT/launcher owner slots, retain original functions and
   provide the returned wrappers. Check slot ownership before assignment and
   before restoration. Do not overlap bootstrap and qualified dispatcher leases.
   No wrapper assignment is implemented by this proposal
4. The Triton lookup retains the reviewed single-expression delta and exact
   slice hash `8686893714b9aef17074011b857f5a40eab9a545a35976d7eea895b7d352221d`.
   The existing candidate's copied globals are CPU/reference-only: installation
   needs a separately reviewed actual-module binding, original/candidate source
   comparison, helper ownership and callback-failure counter. Do not silently
   rehome its code or install its copied dictionary as live evidence
5. For a later qualified dispatcher, publish the exact controller frame after
   `m1.begin_forward()` and keep it through runner cleanup. Notify dependency
   only after `m1_live.py:395`'s existing incoming wait succeeds. The reverse wait
   at :413 remains untouched and is a different dependency
6. Notify checked-consumer entry immediately before the existing incumbent call
   at `m1_live.py:562`, inside its native-lease try/finally. Ignore the metadata
   return for dispatch. This Python entry does not claim native checks succeeded
7. `m1_execution.py` and `run_scale_validation.py`: coordinate default-off purpose,
   bounded plan/source-catalog admission, inherited-variable clearing, owned
   process cleanup and packaging. Adding an ad hoc probes path is not a reviewed
   runtime installation

Every new notification must be failure-contained. Preserve the original native
begin/end, both stream dependencies, record_stream calls, incumbent execution,
route D2H, synchronization, range/uniqueness and native checks. Publication
permission does not authorize this wiring or another experiment.

## CPU validation

The tests use the exact reviewed JIT body and actual lookup seam with CPU owners.
They cover warm discovery, ordinary cold compilation and later warm selection,
natural lazy initialization, 30 owners/one wrapper pair, original errors,
callback partial-failure, module/global/effective-builtin drift, owner swaps,
thread/reentry/close behavior, default-off and argument release. No native
runtime, Mac, SSH, model or GPU is touched. Existing PR42 files remain unchanged.

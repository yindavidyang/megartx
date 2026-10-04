# Scoped borrowing of the fresh invocation workspace map

CPU-only change based on exact public aa36de98af05fff7e1e3862fe35ec30e2418f8db,
tree 2b9165ad156503850f0ee81a38288ae180dc0ab2. No native CUDA build, GPU execution,
performance, quality or graph qualification is claimed.

The internal `Invocation` borrows a const reference to the current `runMoe`
stack frame's workspace-region map instead of copying that map. The map is still
computed freshly for every call by the existing installed geometry routine.
It is a named const local, never a temporary, persistent cache, or device object.

Declaration order is `regions`, `current`, then `ClearInvocation`. The cleanup
guard clears the thread-local invocation pointer before `current` and `regions`
are destroyed on success or exception. All map/expand/TMA host callbacks complete
inside that synchronous scope; device kernels receive values and device pointers,
not a pointer to the host map. Nested admission is still rejected before replacing
the existing invocation. Captured/observer metadata still iterates the original
map, and every use through the invocation is read-only.

This removes one source-level map copy per qualified preparation: 30 per decoded
token or 7,650 for 255 decode steps in a 256-output request. The fresh map's own
construction remains. No map-node count, allocator saving, latency improvement or
performance acceptance follows from this source-level change.

All fifteen current owner views, actual allocation bounds, route range and
uniqueness, stream/context checks, producer/consumer waits, allocator lifetime,
request ledger, C ABI exports and descriptor contracts remain unchanged. The
route readback/fence remains one per preparation in both observer-off lanes,
with ten pointer-attribute and ten allocation-range queries. Kernels, GEMMs,
profiler counts, diagnostics, warmup/window/resource/cleanup rules and failure
semantics are unchanged. The redundant map copy's own possible allocation failure
is removed; no validation/provider exception is suppressed.

## CPU coverage and limits

The new CPU harness extracts the actual Invocation and synchronous runner scope,
using host stubs rather than importing CUDA or compiling the native bridge.
It checks reference identity and copy elision, fresh maps including reused owners,
normal/exception lifetime, reentrant rejection, and diagnostic workspace metadata.
The existing exact TMA harness continues to test complete diagnostic descriptor
readbacks and host/provider/device-entry negatives.

The inverse source normalizer undoes exactly the map member and local-const
changes and recovers the full aa36 bridge hash. The previous descriptor/layout
normalizations and hashes remain chained underneath it. Mutation controls reject
missing/duplicated borrow edits and unrelated geometry, route, owner/stream or
cleanup changes. A new additive source ledger binds the changed files to aa36;
all previous source ledgers and evidence retain their original bytes.

CPU results are listed in the new CPU proof receipt. Host mock/sanitizer evidence
is not a native CUDA ABI or runtime claim. A later exact-source native build and
bounded correctness/operation run need separate review and resources. This work
does not authorize another GPU run or change the preserved aa36 experiment source.

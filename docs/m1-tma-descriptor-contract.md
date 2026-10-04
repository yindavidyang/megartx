# Source-bound TMA descriptor production

CPU candidate based on public head `a135a11dc1669734059fd90d6bcb83b556d77e8d`,
tree `7e699ebcf6412496fd14c8bd8b742cd30651a34b`. The local review-history
commit `81964c5479ab50de04a0268f0e5ed32df7528f76` has that identical tree.
No native CUDA compilation, CUDA runtime import, GPU run, model execution or
performance acceptance is part of this change.

The narrow change is to trust the existing pinned native producer's internal
device descriptor entries during observer-off, capture-free execution, after
fresh host input/owner/workspace/table checks. Captured and external-observer
execution retain full device-entry readback and physical-layout validation.
This changes a proof boundary; it is not a descriptor cache or a claim to detect
arbitrary device memory corruption or malicious native code.

## Interfaces and retained state

The C ABI stays v2 with the same 15 retained views and existing begin/end APIs.
No plugin, controller, launcher, model registration, quantizer, GEMM or preparation
kernel changes are required. `setupTmaWarpSpecializedInputs` still calls the exact
installed implementation and returns its unmodified descriptors. All work stays
on the lease's owned nondefault stream. The eight routes are still read and
validated freshly before preparation, including uniqueness and range.

Within that same setup call, two temporary host `Desc` values are constructed by
the pinned `configureWorkspace` routine from the current stage and GEMM workspace
regions. That routine only partitions host addresses. It performs no device read,
write, allocation or synchronization. Its implementation is newly added to the
installed source pins. Temporary values are destroyed on return, with no route,
pointer, weight, model generation or observed descriptor persisted across calls.

Before calling the producer, the bridge checks:

- Caller origin, expansion ordering, current stream, exact one-row/eight-expanded
  geometry, Geglu, TP1/EP1/cluster1, zero ranks, and unsupported mode/bias flags
- Both current tactic strings, the original input/SF/output/weight pointers,
  six quant pointers and both per-expert flags against this invocation's owners
- No competing MXFP8/MXFP4 or MXFP8/MXFP8 block-scale pointers. Those provider
  branches could otherwise overwrite NVFP4 SF metadata despite its host type tag
- Current packed-input, shared-output, expert-offset and both SF member bindings
- Stage workspace capacity before address construction, and each member descriptor's
  exact typed table identities against a freshly configured stage descriptor,
  including weight/SF/alpha tables, C/D fields, GEMM workspace and absent scheduler

After the unchanged producer returns, the bridge checks the same exact host table
bindings, null C fields, 128 groups, eight routed tokens, NVFP4, no swap/fusion/PDL
or groupwise mode. A nonblocking `cudaPeekAtLastError` propagates an observed CUDA
error without clearing the thread's last-error state. Only then may production
return without copying device entries. Diagnostic execution continues into the
unchanged fourteen-copy/two-fence validation block.

## What is proven where

| Check or guarantee | Classification | This candidate |
| --- | --- | --- |
| Source/package/module/ABI identity, relocation bindings, registered corrected runner | Static source/model admission | Retained; workspace-partition implementation additionally pinned |
| Source-derived dense SF layout proof | Static source invariant | Retained once per loaded bridge |
| Request markers, rows/positions, call caps and failure state | Dynamic request | Unchanged |
| Tensor dtype/shape/storage/owner retention, aliases, stream recording and waits | Dynamic input/owner/lifetime | Unchanged |
| Fifteen lease owner views, exact subview bounds and dedicated workspace cap | Dynamic owner/lifetime | Unchanged |
| Current runner dimensions, tactics, original weight/quant pointers and workspace ledger | Dynamic model/input binding | Unchanged, with fresh setup-boundary checks added |
| Capture/context/device/allocation queries and fresh route ID readback | Dynamic route/device | Unchanged, including fallback and primary error behavior |
| Returned host descriptor lane and table bindings | Internal native host output | Checked every call, before consumer return; templates also checked before producer submission |
| Per-expert M/N/K, SF layouts/bases, AQ/output bases and packed strides | Internal native device output | Production trusts exact pinned producer plus qualified current inputs; diagnostics retain fresh checks |
| Actual physical SF masks, per-entry metadata and payload artifacts | Diagnostic introspection | Retained in captured/observer calls |
| Detection of arbitrary native/device corruption before GEMM | Outside trusted-provider contract | Not claimed |

No weight descriptor or route-dependent value is moved to initialization. The
removed fourteen transfers were output introspection of the internal descriptor
kernel, not immutable weight payload reads. Caching them by pointer would be both
unnecessary and incorrect as routes, allocation owners and workspace contents
change.

## Exact audited provider

The upstream files were retrieved from FlashInfer commit
`69ff11fc4954396d98326656dc85debd2223f637` (v0.6.18). The producer and type header
bytes independently match the existing installed-source pins for the retained
0.6.18.post1 environment:

- [cutlass_fused_moe_kernels.cuh](https://github.com/flashinfer-ai/flashinfer/blob/69ff11fc4954396d98326656dc85debd2223f637/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh), SHA256 `fd9e2e976496ab318bda6d133d2b68f45b3451a6482978f452acd7a86e029841`
- [moe_gemm_kernels.h](https://github.com/flashinfer-ai/flashinfer/blob/69ff11fc4954396d98326656dc85debd2223f637/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h), SHA256 `eca60a5a7f30b70b7a4f426fb833085ee2ae320445fb217fdd8642c9480b23ab`
- [moe_gemm_tma_warp_specialized_input.cu](https://github.com/flashinfer-ai/flashinfer/blob/69ff11fc4954396d98326656dc85debd2223f637/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu), prospective installed pin SHA256 `647bde0277d7b93f3316d3a45748ad520f1774e056f08d4b2b34b999afd1c128`

The last pin is source-audited here; actual target installed-file equality remains
a prerequisite of a future bounded build. A public source match does not prove
which transitive headers built a historical cached binary. Existing exact module,
installed-source and compiled binding guards remain required.

The producer writes all 128 shapes from consecutive fresh expert offsets. For
active expert `e` at sorted rank `r`, M=1, FC1 (N,K)=(1408,2816), FC2=(2816,704).
It constructs nontransposed SF layouts through the pinned CuTE configuration,
packed AQ at `r*K/2`, BF16 output at `r*N*2`, and SF at
`ceil((r+127*e)/128)*128*(K/16)` from their currently bound bases. Source-selected
weight/alpha pointers remain tied to original quantization owners. Independent
CPU enumeration covers every 968 feasible active (expert,rank) pair per stage,
all 1,080 feasible inactive (expert,prefix) combinations, disjoint route changes,
and one-byte-short owner negatives. This is a source-equation proof, not native
execution. Existing real typed CuTE host proof remains separate.

Zero-row groups receive shapes but the producer returns before writing their
payload pointer, stride and SF entries. Those entries may be stale. This patch
does not require them to be zero, valid or newly initialized. Source-bound group
scheduling has no zero-row payload tile; metadata may still be read. Historical
inactive-metadata limitations are preserved.

## Error and ordering boundary

Every host contract error is raised before returning to GEMM. Member-table/input
errors are checked before descriptor submission. An observed pending CUDA error
propagates before GEMM, without clearing it or attempting stock fallback. A
provider exception propagates unchanged. The installed producer currently ignores
the direct `cudaLaunchKernelEx` return; the new peek observes last-error state in
the linked CUDA runtime and does not strengthen that into a completion guarantee.

Removing host fences intentionally changes asynchronous fault-detection timing.
A later device execution failure may first be observed at an ordinary later
runtime/synchronization boundary. Neither `cudaPeekAtLastError` nor the retained
caller `wait_stream` proves host-visible completion; the latter establishes GPU
ordering. The existing failed-controller/no-retry behavior, retained tensor
lifetimes and primary-error-preserving cleanup remain unchanged. Diagnostics
still synchronize and may surface completion errors earlier.

## Expected operations and evidence limits

Per production preparation, descriptor transfers decrease 14 to 0 and descriptor
fences 2 to 0. The route transfer/fence stay 1/1; ten pointer-attribute queries and
all kernel launches remain. Per 120-call lane this predicts:

- D2H copies: 1,800 to 120, of which 1,680 descriptor copies are removed
- Explicit preparation fences: 360 to 120
- Pointer-attribute queries: unchanged at 1,200
- Stock/fused full-window GPU operations: 6,440/6,320 to 4,760/4,640

These are source counts, not a measured speedup. The clean six-pair results at
the parent source remain preliminary and below the 30-pair acceptance floor.
Their lane differences cannot be attributed to this unexecuted candidate.

The operation validator retains the frozen historical inventory digests and
legacy `validate_trace` behavior. The new source-bound run calls the explicit
candidate contract. It verifies actual reduced copies/fences and all original
frame/layer/stream/correlation/kernel checks, then compares the complete GPU
inventory after adding only the fourteen source-specified removed pageable-D2H
signatures per preparation to the historical digest. These added signatures are
digest arithmetic, never fabricated observed events or execution evidence. If
the route and historical descriptor transfer signatures differ, admission fails;
no new digest is learned from a failed run. Legacy traces cannot satisfy candidate
counts, nor candidate traces satisfy legacy counts.

CPU controls compile the byte-identical setup body with mocked CUDA/provider
interfaces: eight test methods run 195 cases (97 pre-provider, 54 returned-host,
30 device-entry, five error, four repeated-call, two unsupported and three positive
mode controls). All 349 scaffold and 532 numerical-reference tests pass, as do
eight config validations, Python compilation, the 4,064-product format self-check
and whitespace validation. These tests do not compile the native CUDA bridge or
verify installed ABI.
Native build, captured/observer byte-and-layout diagnostics, unchanged operation
inventory outside the declared delta, replayed outputs/startup fixtures and
bounded paired measurements require separate exact-source review and GPU approval.
Keep all historical ledgers and raw artifacts. No graph eligibility, package
change, resource expansion or wider fusion is implied.

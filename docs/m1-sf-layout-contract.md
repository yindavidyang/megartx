# Source-derived decode SF layout proof

This CPU-only candidate starts from exact green main
`e886b98c1db30b1d77fb08471f1d6b33d162bbf1`. It replaces repeated physical
scale-factor layout enumeration with comparison to two immutable, source-derived
layouts. It keeps actual descriptor readback, stream fences, routes, allocation
queries, buffer ownership, arithmetic, quantization and dispatch unchanged.
Native bridge compilation and GPU validation remain pending independent review
and parent clearance. No latency, speedup, quality or graph result is claimed.

## Actual hot-path check inventory

The inventory covers `LivePreparation.routed/invoke`, the native lease admission,
`runMoe`, map/expand hooks, `candidate_eligible` and TMA setup. It distinguishes
source constants from per-call operands and from device-produced descriptors.

| Boundary and checks | What can change | Treatment in this patch |
| --- | --- | --- |
| Initialization: successful build and compiled lease/binding controls; compiled ABI, native/controller hashes, installed package/ELF/header identities; CUPTI identity, exact original-scale startup fixtures; registered adapter/callable identity | Source/package/module generation and loaded model | Retained. No weight, tactic or allocation cache is introduced. |
| Python frame/request: marker, tokens, positions, lane progression, failed-state/nesting guard, call cap and backend/frame ledger | Every frame/request | Retained. Token/position host inspection remains. |
| Python operands: one-row FP4 input, SF count, route/weight dtype and shape, six contiguous CUDA quantization owners and typed global-scale shapes, unsupported flags, capture status | Every call; inputs/routes/owners may change | Retained; unsupported states call the whole incumbent. |
| Python ownership: contiguous CUDA tensor views, storage base/bytes, retained owners, workspace size and reuse, `record_stream`, producer-to-owned and owned-to-caller waits, native release and primary-preserving cleanup | Tensor/storage/stream lifetime and workspace replacement | Retained; pointer equality alone never establishes identity. |
| Native lease: inactive/nonrecursive admission, ABI/version/view framing, nonnull stream/owners, each subview within its storage; whole dedicated workspace; combined workspace/shadow cap; seven input/output/scratch aliases and eight immutable-owner versus mutable-view aliases; observer registration serialization | Every retained lease | Retained. The new source proof is initialized within the existing caught C ABI boundary, before device dispatch. |
| Native runner: pinned caller/module, exactly one runner, geometry/activation, parallelism, bias/fusion/PDL/LoRA/groupwise flags, actual FC1/FC2 tactic strings and config presence, stream, input/output/workspace and original typed weight/quantization pointers, fifteen declared operand extents | Model bindings/tactics may be stable after load, but current values are inspected on every call | Retained. Nothing is moved to a pointer-keyed cache. |
| Workspace: installed size/offset ledger, each requested region size and containment, exact source-map FFI partition, no recursive invocation | Dimensions, tactics, workspace generation, owner extent | Retained. Fixed dimensions do not imply a fixed allocation. |
| Map eligibility: exact map arguments and once-only hook; capture/context/stream identity; ten pointer/device/alignment/address-range queries, allocation extents and pairwise aliases; eight route IDs in range and unique | Routes, allocations, streams and context on every call | Retained. The prior one-check immediate-dispatch helper remains byte-identical. |
| Expansion: pinned caller, once-only hook, exact AQ/SF/maps/offset pointers and dimensions, prequantized FP4 typed lane, per-expert flag, globals/block-scale identity, no additional epilogue/finalize/PDL, stream | Native arguments and route-dependent preparation outputs | Retained; unsupported map qualification still delegates stock. Launch/runtime errors never retry after candidate mutation. |
| Actual TMA descriptor: pinned caller and preparation ordering; NVFP4/no fusion/no PDL; seven device tables per stage; four AQ/output table extents, two fences; 128 fresh shapes, layouts and SF/AQ/output pointers/strides per stage | Native outputs and expert route prefixes | Retained, including existing table validation coverage. The shape/SF-layout/SF-pointer tables are copied under the pinned native producer contract; this patch adds no claim of a new allocation receipt for those tables. |
| Each active expert: token rows in {0,1}, no swap, fixed M/N/K, SF domain dimensions/cosize, full physical carrier bounds; dense physical SF layout; AQ/output packed strides and owner containment | Active expert set, rank, native pointers/strides/layouts | Retained. Only the dense-layout enumeration is replaced in production by exact semantic layout comparison plus the immutable proof. |
| Inactive expert groups | Shape/metadata depend on routing; no payload tile for zero rows | All tables/shapes stay fresh. Existing zero-row skip remains before active-layout proof and AQ/output checks; no payload read is inferred from an unused cosize. |
| Diagnostics: capture or external observer; complete physical enumeration, detailed shapes/masks/payload readback; source-bound trace identity/counts/correlation/stream checks | Explicit diagnostic intent and actual native outputs | Retained. Both diagnostics now also require exact semantic layout identity. Trace validation behavior changes only by adding the new header to the exact source inventory. |

## Why this proof can be reused

The pinned FlashInfer producer `setupFP4BlockScalingFactors` constructs the
nontransposed activation layout with
`Desc::NVFP4BlockScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1))`.
The type is the pinned CUTLASS `Sm1xxBlockScaledConfig<16>::LayoutSF`, a CuTE
layout whose address function is entirely specified by its nested shape and
stride tuples. FC1 uses `(1,1408,2816,1)`; FC2 uses `(1,2816,704,1)`.
CuTE tuple comparison compares every semantic leaf. Static leaves are also
enforced by the typed layout; object padding is irrelevant. The build now pins
the layout and tuple comparison source files as well as the existing producer,
type and block-layout source pins.

`M1SfLayoutContract` constructs those two layouts from the actual installed
configuration and exhaustively verifies their dense physical carrier domains
once. Every active native layout must still match both full tuples before the
bridge uses that conclusion. A mismatch raises the existing native contract
error before GEMM consumption; the controller retains its failed-state,
no-retry and stream-cleanup behavior. Captured and external-observer calls also
enumerate every actual layout, including physical padding.

This stores only source-derived values, with no observed descriptor, route ID,
prefix, weight, allocation or owner. Its lifetime is the loaded bridge's static
object. A new compiled/loaded bridge initializes its own proof and retains all
source/ABI guards. There is no mutable model/workspace generation in this proof
to invalidate. Model/workspace replacement is still checked afresh by the old
owner/extent/dispatch checks; it cannot authorize a different layout.

Per active expert, FC1 enumerated `128*(2816/16)=22,528` coordinates and FC2
enumerated `128*(704/16)=5,632`. Eight unique routes therefore previously cost
`8*(22,528+5,632)=225,280` coordinate evaluations and sixteen temporary seen
vectors per qualified preparation. Production now performs sixteen tuple
comparisons, after one `28,160`-coordinate proof per loaded bridge. Across a
120-call lane window, the repeated work removed is `27,033,600` coordinate
evaluations. The one-time proof is reported separately, not subtracted from a
measured timing. Descriptor copies, fences and device launch counts are expected
to remain identical: fourteen descriptor copies, one route copy, three explicit
preparation fences and ten pointer queries per call; 1,800 copies, 360 fences
and 1,200 pointer queries per 120-call lane.

## CPU evidence and bounds

The [CPU proof](evidence/m1-sf-layout-cpu-proof.json) binds the helper, host probe
and verifier sources, copied installed header hashes, binary and semantic offset
output hashes. A host C++ compiler builds the real CUTLASS configuration without
CUDA calls, library loading, CUDA imports or model access. Apple Clang requires
`-Wno-invalid-specialization` for an upstream CUTLASS/libc++ specialization; this
host-only flag does not alter native build flags. All 28,160 physical offsets
match the independent `sf_coordinate` oracle, including 127 padding rows. All
968 physically possible selected (expert, rank) pairs per stage have the exact
source-derived carrier base and fit the shared SF owner. Independent extent
controls reject a one-byte-short carrier, while the byte-identical bridge
`contains` function rejects both FC1/FC2 shortened/replaced ranges and endpoints.
Actual zero-row layouts are inspected separately and not admitted as payloads.
Twenty dynamic-leaf and six stage drift controls reject in the real typed probe.

CPU CI compiles the byte-identical helper against counted mock layout APIs and
proves that 120 production preparations perform zero coordinate evaluations,
that diagnostic calls do enumerate, and that source-shape/stride drift, mutation
at the same object address, bad constructor density and bounds fail. Existing
allocation/route/capture/API-error, no-fallback-on-error, owner/lifetime, release
and primary-error tests remain. The new exact current-source ledger extends the
historical chain without editing any old ledger or reviewed prefill overlay;
wrong prior/current hashes, unknown files, mixed vectors and added-header drift
remain failures. CPU evidence does not compile or execute the native bridge.

Candidate increment: zero device allocations, no execution scratch change and
two small immutable host layouts. Initialization temporarily allocates at most
22,528 host bytes for the dense proof (the two vectors are sequential). Existing
descriptor host buffers remain; the sixteen repeated seen vectors are absent
from production. Existing model and six-fixture startup reference workspaces are
incumbent allocations, not candidate increments. The existing private workspace
plus 32-byte shadow stays under 8 MiB; compiler aggregate <=2 GiB/shared <=300 s,
host available >=8 GiB and GPU free >=2 GiB remain unchanged.

## Proposed hardware proof, awaiting clearance

Freeze exact candidate head/tree, full tracked source vector, focused/all CPU
results, copied installed/header/package pins and private original packet hashes
for independent review. Do not build/load a CUDA bridge or start model/device
work until the parent clears both source and a serialized GPU slot.

After clearance, use one isolated adapter/runtime/cache/AOT/bridge from the frozen
head with the pinned vLLM/FlashInfer/Torch/checkpoint. Run the same bounded
source-bound operation diagnostic as `cf656d6`: seed 9471, one warmup and one
measurement per stock/fused 2K/8K context, eight private 256-output requests,
then one separate one-token drain. Profile the same four 2K decode frames per
lane. No corpus, token, package, model, kernel math or quantizer change. The
existing trace validator must admit the exact unchanged operation inventories
and all frame/layer/dispatch/stream/correlation/count identities. Compare all
eight output sequences to the immutable accepted original and stock versus
fused/warmup versus measurement. All six forced startup BF16 fixtures must pass;
report natural rare correction coverage separately. No latency comparison or
clean-timing admission is part of this run.

Before the model diagnostic, compiled native framing/binding controls must pass.
A bounded captured or external-observer correctness call must prove that full
enumeration and detailed masks remain available; this additional device work
requires its own exact source-bound plan and parent-approved inclusion, rather
than being inferred from the operation diagnostic. CPU evidence already checks
the exact diagnostic helper. Retain existing source/AOT/binary/package drift
guards, compiler-resource accounting, headroom/watchdogs, owned cleanup and
postrun identity checks. Any real blocker stops the run; no automatic sweep.

## Graph blockers left for later work

Route D2H readback/fence, fourteen native descriptor D2H copies/two fences,
Python token/position host inspection, explicit capture-status rejection and
unqualified producer/consumer wait behavior remain. Allocation/context queries,
dynamic workspace ownership and source-bound native descriptor construction also
need a separately justified capture contract. Removing CPU enumeration supplies
no graph eligibility, and default-off native verifier code on this base remains
outside this patch's validation.

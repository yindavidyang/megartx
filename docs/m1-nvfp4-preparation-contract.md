# CPU contract for M1 NVFP4 dispatch preparation

This change supplies an independent CPU byte oracle and an unavailable dispatch
boundary for the proposed Gemma preparation milestone. It starts from merged
main `8b81aa2c65ec971159b8e19e8a23b2eacf7a74e8`. It implements no CUDA kernel,
changes no installed runtime, and makes no performance or project-gate claim.

The [source pins](evidence/m1-preparation-source-pins.json) bind the planner inputs,
existing numerical evidence, saved installed Python adapter, and three public
FlashInfer headers. The installed C++/CUTLASS ABI remains unsupported. Missing
fields are explicit `null` values, never inferred from the package version.

## Boundary and exact CPU outputs

The [independent oracle](../numerical_reference/m1_preparation_reference.py) uses
only Python's standard library and imports no candidate, Torch, vLLM, FlashInfer,
or NumPy. The contract is one token, hidden size 2816, 128 local experts, top 8,
expert intermediate size 704, TP1/EP1. It consumes an already quantized common
activation row. Quantization, expert GEMMs, GELU, correction, weighting/reduction,
attention, shared branch, head, and sampling remain outside this boundary.

| Input | CPU contract |
| --- | --- |
| IDs | Eight distinct integer IDs in `[0,128)`, preserving route-slot order |
| Route weights | 32 raw bytes, eight finite FP32 encodings; signed values, subnormals and signed zero are copied without casts |
| Packed activation | 1,408 raw bytes, logical E2M1 width 2,816; even K is the low nibble |
| Activation SF | Full 22,528-byte 128x4 physical extent; only 176 row-zero coordinates are valid |
| Stage context | Explicit independent FC1/FC2 `swap_ab`, fusion modes, alpha bytes, activation global bytes and scalar provenance |
| Original weight globals | Separate 128-entry FP32 gate/up tables for FC1 and down table for FC2; no folding, multiplication or reciprocal reconstruction |

All inputs are immutable. The valid SF bytes must represent finite nonnegative
E4M3fn values; `0x80` is permitted signed zero, matching the committed format
oracle. Unused source padding is opaque and may contain poison bytes.

Stable ascending-ID sorting generates the inverse route maps. A separate
histogram/prefix calculation generates all 129 expert offsets. In notation with
input IDs `d[j]`, sorted rank is `count_i(d[i] < d[j])`, and offset `o[e]` is
`count_j(d[j] < e)`. Both maps are little-endian `int32[8]` fixtures; offsets are
little-endian `int64[129]`. Each sorted AQ row is a byte-exact copy of the input.
Route-weight bytes move with their IDs, yielding 11,264 AQ bytes and 32 weight
bytes without any floating-point arithmetic.

The oracle emits sparse writes for exactly 1,408 useful expanded SF bytes. For
block coordinate `b` and row `m`, with padded block width `B`, the offset is:

```text
((m // 128) * (B // 4) + b // 4) * 512
    + (m % 32) * 16 + ((m % 128) // 32) * 4 + b % 4
```

The standalone coordinate decomposition is exhaustively checked against the
existing independent `sf_offset_128x4` oracle for 256 rows at both real block
widths, 176 and 44. Under the explicitly selected upstream reference contract,
the grouped base for expert `e` and prefix `o[e]` is:

```text
round_up(o[e] + 127 * e, 128) * round_up(K, 64) / 16
```

The reference capacity endpoint `(e=128, prefix=8)` is 2,883,584 bytes for FC1
and 720,896 for FC2. These are symbolic extents, not new GPU allocations or
measured resident memory. Scale addressing uses the expert ID, not compact
sorted rank. Padding/guards retain their supplied initial bytes; the complete
storage checker rejects any undeclared write. FC2 scales are future activation
outputs: preparation describes their views but does not invent their contents.

## Symbolic descriptor contract and unsupported ABI

Every pointer is represented as `(owner, byte_offset, extent, capacity)` and
checked for bounds. Owners are canonical fixture names with no assumed scratch
aliases. The oracle never reads weights or allocates an expert weight matrix.

Both problem lists contain 128 freshly initialized triples. Logical FC1
`(M,1408,2816)` and FC2 `(M,2816,704)` have `M=1` for selected experts and `M=0`
for the remaining 120. Each explicit `swap_ab` independently exchanges M/N.
The fixture encodes triples as little-endian int64s; these bytes are **not** a
C++ struct layout or an installed descriptor size assertion.

For active experts the oracle checks packed activation and weight views, grouped
activation-SF and expert weight-SF views, logical element strides, alpha/global
bindings and output/finalization references. Alpha/global bits and their symbolic
owner identity are preserved. Logical A `[M,K]` strides are `(K,1)` and B `[K,N]`
strides `(1,K)`; these are not fabricated binary CuTe stride objects.

The narrow reference supports FC1 fusion `none`, and explicitly declared FC2
`none` or `finalize`. With no fusion the output view is BF16 `[M,N]`. Finalize
describes the sorted-map and route-weight views; the typed final-output layout,
reduction and other epilogue state remain unresolved. Other fusion modes are
rejected. Inactive pointer/stride fields are omitted, rather than assumed zero;
only every inactive problem shape is prescribed. Native consumer-read masks are
still required before those omissions can be accepted for an installed kernel.

The reference identity is `flashinfer-8bc3b578-nvfp4-128x4-semantic-v1`. Any other
identity, including `installed` or `0.6.18.post1`, is rejected. Public headers at
[the pinned commit](https://github.com/flashinfer-ai/flashinfer/blob/8bc3b578027791336c6ae87db5c9d76f82cef8bc/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh)
justify the reference views; equality to the loaded installed sources is unknown.
No opaque CuTe object, alignment, binary field offset, output alias or lifetime
is reconstructed from this document. Weight SF is referenced as complete expert
storage; gate/up at row 704 is not sliced as contiguous swizzled scale storage.

## Fail-closed integration interface

[`select_preparation`](../src/megartx/m1_preparation.py) is a pure decision
function. It accepts a typed request and optional CPU fixture IDs, returning
`backend="incumbent"` with explicit reasons. Its `cpu_contract_eligible` field
only describes shape/lane eligibility. Even opt-in on a fully eligible request
cannot select a candidate: none is compiled or registered. `run_candidate`
unconditionally raises `BackendUnavailable`, including with fabricated positive
receipts, before touching supplied state. Existing runtime/plugin code does not
call this scaffolding, and there is no environment switch or graph hook.

Prefill, other geometries, TP/EP, LoRA, all-to-all, groupwise modes, missing or
unswizzled scales, per-expert activation quantization, unknown swap flags and
an inactive correction runner return unsupported reasons. Invalid IDs in a
declared M1 CPU route are validation failures. Device validation/integration
remains future work. The current corrected eager lane must stay active; graph
mode remains unqualified. A future supported branch must choose fallback before
mutating live preparation state and preserve the correction runner and tactics.

## CPU verification and GPU dependencies

Run from the repository root with the existing CPU environment:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m unittest discover -s numerical_reference -v
PYTHONPATH=src python3 -m megartx validate configs
python3 numerical_reference/nvfp4_reference.py --self-test-summary
python3 -m compileall -q src tests numerical_reference
```

New tests cover every expert in every route slot (including the six affected
experts), all packed byte/nibble patterns, all permitted SF bytes, signed-zero
and equal weights, unsorted/low/high IDs, repeated/disjoint routes, independently
swapped problems, bounds, source immutability and padding/guard preservation.
Negative controls swap a weight without its ID, reverse a nibble pair, alter an
SF byte, use compact SF bases, retain stale nonzero-M state, and substitute the
gate global for up. Every control must fail exact comparison; the last also
fails the existing bounded original-projection oracle's alpha provenance check.
These synthetic checks establish CPU algebra and rejection boundaries only.

Local validation on Python 3.12.8 / existing NumPy 2.4.4 passed 332 numerical
reference tests and 38 scaffold/dispatch tests (25 new test methods). Config
validation, the 4,064 finite-product format self-check and Python compilation
also passed. The existing CI discovers these files and separately uses Python
3.10/3.12 for scaffold checks and NumPy 2.3.5 for numerical checks; this local
record does not claim those remote jobs passed.

Before CUDA implementation or selection, the GPU owner still needs installed
FlashInfer/CUTLASS/generated source identities, compiler/module/tactic pins, both
swap flags and epilogues, a typed C++ descriptor/layout probe with `sizeof`,
`alignof`, field offsets and consumer masks, actual workspace ownership/alignment/
aliases/lifetimes, and matched incumbent input/output fixtures linked to a
corrected layer/token/route. No such probe was run for this CPU change.

After those ABI dependencies, native equality, source/write bounds, sanitizer
and repeated-route stress must pass under the same corrected numerical lane.
Numerical baseline qualification, activation-calibration choice, corrected graph
support and any matched measurement are separate prerequisites owned by the
validation effort. CPU success cannot unlock them or advance G0/G1/G2. No
GPU/SSH activity, package/runtime change, shared-checkout edit, planner edit or
numerical-investigation edit belongs to this task.

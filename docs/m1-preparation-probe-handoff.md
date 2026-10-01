# M1 preparation ABI probe and capture handoff

This CPU-only harness starts at merged main
`52b69ea10566e78ef514a7959f3f986ed117f0c3`. It supplies a typed host probe,
bounded artifact intake, and independent validation using the merged
[M1 byte oracle](m1-nvfp4-preparation-contract.md). No target connection, native
probe build, GPU execution, runtime modification or candidate selection occurred.

## Tools and evidence boundaries

[`capture_m1_preparation.py`](../scripts/capture_m1_preparation.py) invokes the
standard-library [checker](../numerical_reference/m1_abi_probe.py). It never
imports a GPU library, launches a subprocess, compiles, loads or tunes anything.
All writes are exclusive new files/directories; capture copies only explicitly
named, hash-bound artifacts after validation, never an entire producer directory.
Traversal, symlinks, oversized/nonregular files, unknown fields, duplicate keys,
floating-point JSON and bool-as-integer values are rejected. Scalar encodings
use exact bits; FIFOs are rejected without waiting for a writer.

The four source-audit inputs are the three FlashInfer headers already pinned in
[the source manifest](evidence/m1-preparation-source-pins.json) plus CUTLASS's
`include/cutlass/detail/sm100_blockscaled_layout.hpp`, SHA256
`598e054bef21edf94b1fd6bb1447cfa9cfcf5a5907ab370128102448dbb6d530`.
FlashInfer reference commit `8bc3b578027791336c6ae87db5c9d76f82cef8bc` resolves its
[CUTLASS submodule](https://github.com/flashinfer-ai/flashinfer/tree/8bc3b578027791336c6ae87db5c9d76f82cef8bc/3rdparty/cutlass)
to `b46b16d003484063bca4ed365e44095c4c6ed633`. These reference identities do not
establish installed identity. Missing or changed files stop this source profile;
matching four files still leaves the transitive include graph, generated code,
compiler/flags, loaded module and actual tactic bindings pending.

[`m1_host_abi_probe.cpp`](../probes/m1_host_abi_probe.cpp) is a host-only program
that includes the *supplied installed* typed header. It emits measured type
sizes/alignments and offsets of 33 descriptor/nested-finalization fields without
printing addresses or using guessed `offsetof` on opaque types. It constructs
the supplied CUTLASS SFA/SFB layouts for each FC1/FC2 swap flag and emits row-zero,
all 128 tile rows, and the final logical weight row's coordinates. The checker
compares every reported coordinate to the independent CPU formula. The program
takes no arguments and describes M1 only; it calls no CUDA API and contains no
kernel or allocation. Runtime workspace layouts and consumer masks remain null.

The header spelling `moe_gemm_kernels.h` follows the reference runner's
[JIT include roots](https://github.com/flashinfer-ai/flashinfer/blob/8bc3b578027791336c6ae87db5c9d76f82cef8bc/flashinfer/jit/fused_moe.py#L226).
That public file's SHA256 is
`b41f213cba1be67d7367e0f01b5612992f38ca06e29de1d9e97c8194fa9bc58d`.
Its `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include` search root
contains the pinned header. A CPU preprocessor regression checks this lookup
using a sentinel header in that source topology; it does not compile native types.

The standalone [C++ wire helper](../probes/m1_probe_wire.hpp) is compiled and
tested locally against ordinary CPU types. **The typed target program has not
been compiled against installed FlashInfer/CUTLASS.** Its build and the missing
runner exporter are target prerequisites. A host report proves typed layout
observations only; it cannot prove equivalence to a loaded generated module.

## Prepare and collect without target access

From the repo root, in a fresh private output directory:

```sh
python3 scripts/capture_m1_preparation.py request --output /tmp/m1-request.json
python3 scripts/capture_m1_preparation.py source-audit \
  --flashinfer-root "$TASK_FI_SOURCE_ROOT" --cutlass-root "$TASK_CUTLASS_SOURCE_ROOT" \
  --output /tmp/m1-source-audit.json
python3 scripts/capture_m1_preparation.py capture \
  --input "$TASK_PRODUCER_DIR" --output "$TASK_NEW_CAPTURE_DIR"
python3 scripts/capture_m1_preparation.py validate "$TASK_NEW_CAPTURE_DIR" --cpu-only
```

`request` writes a specification with exact byte extents and null ABI/ownership
bindings. It neither creates fake observed outputs nor authorizes a GPU run.
`source-audit` reads existing roots only and returns 2 for a changed profile.
`capture` accepts an existing native producer packet or clearly labeled synthetic
fixture, validates and copies the immutable bytes once. Its success concerns
artifact intake only. `validate --cpu-only` returns 0 for exact CPU agreement.
Default validation returns 2 because installed execution remains unsupported.
Malformed/mismatching inputs return 1. Every result keeps
`installed_abi_verified=false` and `candidate_selectable=false`; relabeling origin
or providing digest assertions cannot unlock either. The runtime dispatch guard
is unchanged and does not consume these receipts.

The packet is `capture.json`, schema `megartx.m1-preparation-capture.v1`, with
explicit geometry, origin, source audit, optional hash-bound `host-abi.json`,
null-or-digest binary bindings, owner registry, and one to three cases. Case
contexts encode separate original gate/up/down and alpha/global FP32 tables as
exact hex, explicit fusion and independent swap flags. Cases use one configured
layer with identical contexts. `descriptors.fc1/fc2` are the oracle's semantic
JSON records for eight active experts; native opaque object bytes are excluded.
The [synthetic builder/tests](../numerical_reference/test_m1_abi_probe.py) provide
a runnable packet example without checkpoint data.

Each case's `files` maps the following roles to exact
`case0/<role>.bin` (then `case1`, `case2`), SHA256 and byte length:

| Roles | Bytes per role |
| --- | ---: |
| `ids`, `weights`, both route maps, `permuted_weight_bits` | 32 |
| `aq`, `sf` | 1,408 / 22,528 |
| `input_after` (IDs + weights + AQ + SF) | 24,000 |
| `expert_offsets`, `expanded_aq` | 1,032 / 11,264 |
| `fc1_shapes`, `fc2_shapes` (all 128 int64 triples) | 3,072 |
| `sf_before`, `sf_after` (full FC1 scale extent + two 32-byte guards) | 2,883,648 |

The raw total is 5,833,832 bytes per case, 17,501,496 for three. Shape/map
fixtures use little-endian integers; all FP4, SF and FP32 encodings stay raw.
`input_after` must exactly equal the pre-launch sources. Successive scale
snapshots must chain: each `sf_before` equals the previous `sf_after`. Only 1,408 valid SF
bytes may change; every padding and guard byte must retain its declared initial
value. FC2 activation/SF *contents* are future outputs and are not fabricated.

## Smallest coordinated target fixture

The first target handoff is CPU metadata only: locate installed sources and
include dependencies, record their hashes and the existing runner's compiler,
defines and flags, then build the host probe with those headers in an isolated
directory. Record the complete dependency file and binary digest. Run only that
host executable and import its bounded JSON. Do not substitute an upstream
checkout or generic compile flags to claim installed ABI equality. Compilation
failure or missing headers is an explicit stop, not a package-install request.

Before requesting the GPU, audit one source-matched C++ insertion point after
workspace configuration and after the three stock preparation launches, before
FC1. **A native fixture exporter/insertion point is not implemented here.**
The public Python fused-MoE call is not an acceptable substitute because it
enters expert computation. If the narrow hook cannot be verified, stop with
source/layout evidence and leave GPU access with the numerical owner.

After that hook and GPU handoff are separately agreed, the minimum fixture is
one M1 preparation observation with one configured layer, eight distinct IDs,
existing prequantized AQ/SF and unchanged original scalar bindings. For stale
state coverage, use three preparations in the same scratch: unsorted IDs
`[127,0,82,42,126,7,89,12]`, repeat, then disjoint `[20,21,22,23,24,25,26,27]`.
This is an artificial routing fixture, not natural model coverage. Use the same
corrected runner context and stock tactics; no quantization, GEMM, GELU, model
forward, graph capture or benchmark belongs to this fixture. Synchronize the
owned stream at each capture boundary before D2H copies; do not read another
stream's unpublished metadata. No weight payload is copied or repacked.

Use fixture-owned guarded scale scratch; never read before/after an allocation
to manufacture guards. Log canonical role views and anonymous `alloc_0` IDs,
actual allocation extent/alignment, relative origin and inclusive lifetime phases
0=input, 1=maps, 2=expansion, 3=FC1 consumption, 4=activation, 5=FC2 consumption,
6=finalization. All views must fit their owners. Both activation-scale roles
also require `offset >= 32` and `offset + capacity + 32 <= allocation_extent`,
so guard reads stay inside the allocation. These guard bounds do
not apply to resident weight scales. The referenced stock runner
aliases FC1/FC2 scale storage; permit that only with disjoint declared lifetimes
such as `[2,3]` and `[4,5]`. V1 rejects every overlapping live alias, including
unproven read-only aliases. Declared lifetimes are consistency evidence, not an
independent proof of actual runtime scheduling. Opaque workspace/consumer masks
remain pending until the target exporter and typed consumer audit supply them.

Hard artifact caps are 1 MiB JSON, 4 MiB per source file and 32 MiB total intake,
enforced by the reader. Propose at most 8 MiB incremental device scratch and
512 MiB host staging/RSS allowance; these are run guards, not measured peaks or
assumed workspace sizes. Reuse resident weights and inspect actual typed workspace
requirements first. Abort before allocation if the 8 MiB cap cannot cover them.
Allow one compile process, a 300-second compile limit, a 30-second host probe
limit, and a 60-second GPU capture-slot limit with at most three preparations
(nine incumbent launches). Limits are owner-enforced; no scheduler or target
process is started by these tools. Do no tuning or retries that widen this slot.

On success or failure, stop the fixture, synchronize its owned stream, restore
any temporarily borrowed runner arguments before FC1, release only its buffers,
and return the GPU to the validation owner. Keep raw captures private in ignored
artifact storage; review only sanitized summaries for publication. Remove only
the request-owned build/capture directory when no longer needed. No global cache
deletion, process killing, package change or shared-checkout reset is part of
cleanup. A partial capture is never reused as a completed native receipt.

## Verification and future small-M boundary

CPU tests cover exact bytes, input mutation, malformed/hash-consistent corruptions,
stale shapes, descriptor/global changes, unknown qualification keys, path/file
bounds, repeated/disjoint routes with scratch continuity, typed layout samples,
and lifetime conflicts. Even populated host/owner/hash evidence remains unqualified.
Existing CPU CI discovers the new tests; the generic C++ helper test uses the
system compiler when available. Native target compilation, actual descriptor
export, execution trace correlation and installed build equivalence remain open.

Local validation passed 382 reference tests (22 new probe tests), 38 scaffold
tests, all eight config contracts, Python compilation and the existing exhaustive
4,064 finite-product BF16 self-test. It used existing Python 3.12.8, NumPy 2.4.4
and Apple Clang 21.0.0, without changing packages. The repository's independent
CPU workflow pins NumPy 2.3.5. CLI request/source-audit smoke checks used copies
of all four pinned public sources, explicitly without claiming installed identity.

Geometry is a versioned discriminator, separate from bounded file intake and
anonymous owner/lifetime records. Future short verification shapes can reuse
those intake helpers with a new shape oracle/schema and target layout probe.
Present M=2/4/8 inputs are rejected before reading case payloads. This extension
point makes no small-M support or performance claim. Numerical/activation-lane,
graph and project-gate qualification remain with their separate investigations.

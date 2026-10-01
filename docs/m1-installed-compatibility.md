# Installed M1 preparation compatibility and native opt-in adapter

The installed FlashInfer maps and NVFP4-to-NVFP4 expansion launchers now have a
compiled typed bridge. Their three reused-scratch captures agree byte-for-byte
with the standalone fused kernel and the independent CPU oracle. The explicit
[native adapter](../kernels/m1_installed_preparation.cuh) also passed that comparison,
stock fallback checks, 22 rejection checks and four Compute Sanitizer tools.
After review, 40 `prepare()` control-flow cases also verify that CUDA query errors
propagate before output mutation or fallback. Fresh captures and sanitizer runs
bind the repaired adapter.
The [sanitized receipt](evidence/m1-installed-compatibility.json) binds the actual
tested source, binary, installed module, flags, dependencies and raw captures.

This milestone starts from main `86b17fa9958b72b58757c886d138d7b300728394` in the
clean, isolated `probe/m1-installed-preparation-capture` checkout. Shared Desktop
sources, installed packages, the cached FlashInfer module and checkpoint remain
unchanged. The previous installed attempt and PR8 raw evidence are preserved.

## Typed binding and installed execution

`QuantParams` is defined in the installed `moe_kernels.h`, SHA256
`8289e5d92e4a8fd04963d58286db5a550cd6fac0cae94919ac4cb34b883152af`.
The [bridge](../probes/m1_installed_bridge.cuh) includes that header and uses its
`QuantParams::FP4` constructor. Function pointers use the actual typed reference
and FP4 types; no guessed opaque object offsets or fabricated byte struct is
passed to the module. Address-free measurements report:

| Installed type/field | Bytes / alignment / offset |
| --- | --- |
| `QuantParams` | 344 / 8 |
| `fp4` | 64 / 8 / 192 |
| `fp4.fc1`, `fp4.fc2` | 32 / 8 / 192, 224 |
| FC1 act-global, weight-block, global pointers | offsets 200, 208, 216 |
| FC2 act-global, weight-block, global pointers | offsets 232, 240, 248 |
| Grouped TMA descriptor | 360 / 8 |
| Underlying problem shape | 24 / 8 |
| Typed SF layout | 20 / 4 |

All 33 descriptor/finalization fields are measured. Four independent stage/swap
layouts agree with the oracle at all 57,200 exported coordinates, including each
activation row-zero coordinate, all 128 weight-tile rows and the last weight row.
These are typed host layout observations; the descriptor is never passed to a
GPU kernel in this experiment.

The executable resolves both launchers from the same loaded cached
`fused_moe_120.so`, SHA256
`dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9`.
It calls those existing implementations. It does not rebuild stock preparation
kernels or invoke the public fused-MoE entrypoint. Compilation uses the exact
recorded Ninja CUDA flags: NVCC 13.3.33, C++17, SM120f, one compiler thread,
C++11 ABI=1, the installed FP4/BF16 enables and include roots. Link additions are
the existing module, existing `libtvm_ffi.so`, `libdl` and the CUDA driver library.
The [832-file dependency manifest](evidence/m1-installed-typed-dependencies.json)
records this probe's transitive include graph. It does not prove the historical
cached module was built from every currently installed transitive/generated file.

The first preflight stopped because Torch wheel metadata is `2.13.0`; a separately
bounded import verifies runtime `2.13.0+cu130`. Two small links failed while
resolving installed TensorRT-LLM and TVM FFI support symbols. Both failure reports
are preserved. The successful typed build then stopped while recording relative
dependency paths. Its unchanged, digest-verified binary was resumed after fixing
that bookkeeping. No resource limit was widened. The final native adapter was
compiled separately after the original three-case installed comparison passed.

Review of `f8abeaeb87af688b859a3d202faccae043dde4a9` found that failed CUDA
preflight queries were treated as unsupported inputs, allowing `prepare()` to
invoke the incumbent callback. Those query APIs can report earlier asynchronous
errors. The adapter now propagates every non-success runtime/driver query result;
only successful queries that report unsupported metadata select fallback.
The [original receipt](evidence/m1-installed-compatibility-f8abeae-superseded.json)
and [original dependency manifest](evidence/m1-installed-typed-dependencies-f8abeae-superseded.json)
are preserved byte-for-byte. The original receipt's dependency filename refers to
the manifest in [that original commit](https://github.com/yindavidyang/megartx/blob/f8abeaeb87af688b859a3d202faccae043dde4a9/docs/evidence/m1-installed-typed-dependencies.json).
Its positive byte comparison remains evidence for those sources; its error-path
coverage is superseded by the repaired source-bound receipt and control-flow tests.

## Exact bytes, padding, ownership and consumer coverage

The fixtures retain M=1, H=2816, E=128, top8 and the prior standalone routes:
unsorted low/high experts, the same IDs with changed payloads, then disjoint IDs.
Both implementations reuse their original scratch throughout each three-case
slot. The producer synchronizes one owned nonblocking stream for uploads,
capture boundaries and host-buffer lifetime. The checker independently rebuilds
every expectation instead of trusting an expected-output file.

| Observed region, per case and implementation | Exact comparison |
| --- | --- |
| Both route maps, 32 bytes each | pass |
| All 129 int64 offsets, 1,032 bytes | pass |
| Expanded packed AQ, 11,264 bytes | pass |
| Permuted FP32 route encodings, 32 bytes | pass, including signed zero/subnormals |
| Full guarded SF snapshot, 2,883,648 bytes | pass |
| Unchanged input IDs/weights/AQ/SF, 24,000 bytes | pass |
| Stock SF before/previous-after continuity | pass for all three cases |

Exactly 1,408 useful SF bytes are assigned per preparation. Every unused byte
and both 32-byte guards retain their previous contents, including stale storage
from the prior route. The producer compares both SF snapshots before each launch;
the stock before-snapshot is retained as raw evidence. Guard reads stay inside
fixture-owned allocations.

Seventeen dedicated `cudaMalloc` requests contain all input/output views and
the fixture scalar. Requested guarded storage totals **5,817,044 bytes**, checked
before each allocation against 8 MiB. The anonymous registry records exact
requested extents, origins, capacities and verified alignment. All fixture views
have the full slot lifetime; their allocations are disjoint. The native adapter
additionally checks CUDA pointer/device attributes, driver allocation bounds and
pairwise live view overlap. A driver backing allocation alone cannot establish
the reserved extent or other live aliases of a pooled runtime subview. Live
runner workspace binding remains a prerequisite for model hookup.

The typed layouts reproduce the logical FC1 consumer mask of 176 SF bytes per
active expert for both swap choices. Exact comparison also covers a conservative
mask of **every byte of the 2,883,584-byte SF capacity**, plus guards, so any bounded
consumer sees the same captured bytes under either implementation. Physical TMA/MMA
load/predicate masks for the actual live GEMM tactics were not executed or verified.
This conservative comparison does not relabel them as qualified, and does not
establish inactive descriptor reads or full runner ownership/lifetimes.

All scalars/routes are synthetic. The stock wrapper requires a nonnull weight-SF
pointer to choose the FP4 lane; this fixture points it at owned input SF storage,
which this expansion specialization does not dereference as weights. Activation
global scale is an owned FP32 one. No weight payload or checkpoint scalar table
is read or fabricated as installed model evidence. The adapter preserves its
caller's typed `QuantParams`; the fixture checks its complete local representation
for mutation without publishing opaque bytes or addresses.

## Explicit native integration boundary

`prepare(call, incumbent_callback, opt_in=false)` is a C++ insertion API for the
source-audited maps/expansion boundary, before the existing TMA setup and GEMMs.
Default calls invoke the supplied incumbent callback. Supported opt-in calls
launch the fused byte kernel; TMA descriptors, tactics, expert GEMMs, activation,
original scale correction and finalization remain the caller's incumbent work.
No installed file is patched and no native entrypoint is interposed.

Candidate eligibility requires the exact geometry/TP1/EP1, prequantized FP4 typed
call, common swizzled SF, distinct valid IDs, a caller-declared active corrected
context, and absence of PDL/min-latency/LoRA/groupwise/all-to-all modes. Device
views must fit the current device's allocations, be aligned and disjoint, and
the supplied nondefault stream must belong to the current CUDA context.
Graph capture always declines before host ID reads. IDs are read on the supplied
stream and synchronized **before any output mutation**; this host fence is part
of the adapter's current execution cost. No latency improvement is claimed.

Unsupported requests invoke the caller's complete incumbent callback, preserving
its own modes and three-step fallback. Failed capture, context, device, pointer or
allocation queries raise before invoking either backend. Failed ID copy or stream
synchronization also raises before any output mutation. Driver diagnostic lookup
failure preserves the primary operation and error code. Launch/runtime failures
after candidate submission raise; they do not attempt fallback over partially
published output.
Tests observe the default-disabled callback, an unsupported correction context,
and actual stock PDL fallback, plus 22 rejected geometry/lane/pointer/route cases
with unchanged outputs. Fixture correction flags exercise selection logic; they
do not establish a live corrected model runner.

The [Python selector](../src/megartx/m1_preparation.py), vLLM registration and model
dispatch remain disabled. This header is exercised by the manual native producer,
not by a model forward. Connecting it to the live installed runner still needs
source-bound workspace subviews/aliases, tactics/epilogues, corrected-runner
identity and consumer masks. Existing [quality/cache blockers](controlled-scale-integration.md)
and graph qualification remain separate. There is no qualified full-model baseline,
full-model timing sweep, installed-incumbent timing result or end-to-end claim.
PR8's 2.17919 versus 3.17081 microseconds remains its development control only.

## Focused validation and resource limits

Native capture and memcheck, initcheck, racecheck and synccheck all pass, with
zero errors and zero race hazards/warnings. Each sanitizer's stock/fused raw
captures pass an independent target recheck and another recheck after download.
All tested native source digests still match the published tree.

The fresh repaired build took **6.039880 seconds**, with sampled compiler-group plus
wrapper RSS of **708,329,472 bytes**, under 2 GiB / 300 seconds. Racecheck's sampled
aggregate RSS was 1,302,564,864 bytes. The smallest sampled host availability was
61,935,874,048 bytes; smallest sampled GPU free memory was 33,658,241,024 bytes.
The executable also checks CUDA/host headroom before and after allocation. These
periodic GPU samples are not a subsecond peak-memory measurement. All capture
and sanitizer slots were below one second and retained the 60-second slot cap.

CPU checks pass **420 numerical-reference tests**, **38 scaffold tests**, all eight
configs, Python compilation and the existing 4,064-product BF16 format self-check.
New CPU corruption tests reject changed stock/candidate data, SF continuity,
typed nested bounds, owner extents, links, truncation, duplicate JSON keys and
invented qualification fields. Synthetic CPU agreement never attests its producer
or unlocks model dispatch.

Five new CPU test methods compile a byte-identical adapter header with mocked
CUDA APIs and execute `prepare()` itself in 40 cases: 29 preflight API failures,
one submission failure and ten supported/unsupported/callback controls. Launch
failure codes are injected at each query, including all ten pointer and allocation
query positions; invalid-value results also propagate. Every preflight error
throws with zero incumbent calls, zero candidate launches and unchanged output.
Successful unsupported states call the incumbent exactly once; submission errors
never fall back. The test also checks failed diagnostic lookup and incumbent
exception propagation. The original header reproduces the review failure and the
repaired header passes the same case. These are CPU control-flow observations,
not real GPU fault injection or installed ABI evidence. The bounded coordinator
runs them before native compilation and hashes both test sources in its receipt.

Reproduce only in the existing pinned environment, in a new output directory:

```sh
python3 -B scripts/run_m1_installed_probe.py \
  --flashinfer-root "$TASK_FLASHINFER_ROOT" \
  --cache "$TASK_FUSED_MOE_CACHE" \
  --output "$TASK_NEW_PROBE_DIRECTORY" \
  --base-head "$(git rev-parse HEAD)" --sanitizers
```

The coordinator copies only named repo sources, imports Torch solely to check
its runtime version, enforces the caps, terminates only an owned process group
on failure, and preserves raw reports/captures. `--resume-compiled` is limited to
an unchanged successful binary/source/command identity stopped before ABI/capture;
it preserves the prior report. It does not authorize runtime or package changes.

Raw artifacts were downloaded and independently rechecked before the owned remote
temporary directory was removed. Cleanup verified all installed pins unchanged,
no compute PIDs and 41 MiB GPU memory used / 32,101 MiB free. Only that owned
directory was removed; prior raw evidence remains intact.

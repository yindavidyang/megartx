# Guarded matched eager M1 preparation comparison

This milestone adds a default-off, exploratory timing path from merged main
`09341f2ad3f6e4b03e8f264b9e3c59fe5ddb52ed`. It compares **guarded stock eager**
and **guarded fused eager** preparation in the same corrected native runtime.
It does not pass G1, qualify full-model quality, admit CUDA graphs, or reopen
the old `--client benchmark` / `host_benchmark.py` gate. Historical graph and
uncorrected baselines are not comparison inputs. CPU success is not GPU evidence.

The fresh source-bound pilot and six-pair comparison at runtime `eecfc13` passed
on 2026-10-03. The exploratory decode result establishes **no speedup**; both
95% paired bootstrap intervals cross zero. See the [results below](#fresh-eecfc13-gpu-result)
and [sanitized scalar receipt](evidence/m1-eager-eecfc13-gpu-scalars.json).

## Scope and retained work

Explicit `--client m1-eager-benchmark --m1-eager-benchmark-plan PATH` is required,
along with native correction, capture-free execution and a freshly source-bound
bridge. There are no controlled routes, normal-capture collector, logits/KV
serializer, request profiler, external observer or process probe in this lane.
The existing controlled and captured normal correctness paths retain their
validation and caps. Their fixture-specific KV/stage checks are not claimed for
natural benchmark requests. The benchmark verifies actual prompt rows, exact
synchronous position progression, all thirty routed adapters, native backend
counts, and actual input/SSE token correspondence. Broader cache semantics,
activation-quantizer quality and unconstrained quality remain open.

No native guard or CUDA error path is removed. Stock retains one route-ID
readback/fence; eligible fused retains two. Each lane retains seven descriptor
table readbacks and one stream fence for each of FC1 and FC2, dense SF carrier
enumeration and host allocations, context/allocation/extent/alias checks, fifteen
owner views and six quantization owners, native lease checks, producer-to-owned
and owned-to-caller stream dependencies, allocator lifetime recording, and
correction `torch.nonzero` dispatch. Controller marker reads and token/position
CPU copies remain measured. Unsupported successful native queries can still
take stock safely, but their counts invalidate a purported fused timing cell.
Query, submission, consumer, lease or dependency errors poison the process;
there is no stock retry after candidate mutation.

The new plan and both driver scripts are bound to the same build source
snapshot as the adapter. Admission checks exact revision, source hashes,
private prompt hashes, request order, matched seed, eager/synchronous BF16 KV,
single sequence, 256-token prefill chunks, unchanged tactics/PDL/finalization,
and observer-off diagnostics. Inherited diagnostic destinations are cleared.
Each plan has an explicit global live-call cap, independent of successful
request counters. A request can change stock/fused selection only after the
previous request has its full frame and backend ledger; it cannot reset the
global invoke cap. The same owned stream and workspace survive lane switches.
The native lane flag is passed for each lease, with its ordinary error cleanup.

Backend and natural correction selection counters remain in memory throughout
the timed requests. A separate, final one-token drain request exports scalar
receipts after the response timestamps. It is outside timing. Receipts hash
the actual input transcript, which the client checks against the prompt and
first 255 generated token IDs. Paired 256-token output IDs and usage must match.
Completed transcript lists are retained by reference at ordinary request
transitions; all transcript hashing/JSON encoding occurs only in the untimed
drain. Prior frame/backend completion checks still precede each lane switch.
Natural selected-row correction hits are recorded separately from the six
forced startup fixtures. Zero natural hits are disclosed; they cannot qualify
coverage or quality. Selected rows do not independently prove nonzero weights.

## Frozen initial protocol

- Checkpoint `nvidia/Gemma-4-26B-A4B-NVFP4`, revision
  `a19cfe00be84568a6867111c9a68c9c44fdcffe6`; pinned Python 3.12.3,
  vLLM 0.30.0, FlashInfer 0.6.18.post1, Torch 2.13.0+cu130 on SM120.
- Exact 2,048 and 8,192 chat input IDs from one private plan; greedy 256 outputs,
  `ignore_eos=True`, identical seed/output policy for each pair. Prefix caching
  is disabled, and output capacity is 256 beyond the 8K prompt.
- One user, concurrency one, eager and synchronous both lanes. One warm server
  runs request-bound lanes in randomized adjacent pairs, with the same compiler,
  model, quantizer, BF16 KV, power policy and stream/workspace ownership.
- Two complete warmup requests per context **per lane**, followed by six matched
  pairs per context, three stock-first and three fused-first. Seed 9471 freezes
  schedule generation. All warmups precede measured requests.
- Initial plan: 32 requests, 244,800 native live calls, 19,200 stock prefill
  fallback calls; then exactly one untimed drain request. A separate one-pair,
  one-warmup-per-lane pilot has 8 requests / 61,200 live calls, and makes no
  statistical performance claim. No automatic expansion beyond six pairs.
- Client has a 3,600-second total cap; each response has a 900-second read
  timeout. Compilation uses aggregate RSS <=2 GiB and <=300 seconds; maintain
  host availability >=8 GiB, GPU free >=2 GiB and extra device scratch <=8 MiB.
  Stop only the owned server on failure and verify its group/GPU PIDs are gone.

Primary timing is host-local HTTP `perf_counter_ns` immediately before POST,
through delivered SSE tokens, DONE, and response close. Encoding, tokenization,
marker writes and subsequent checks are outside the interval; guarded model,
prefill, correction, head, sampler, KV and local delivery are included.
TTFT, amortized decode ITL, last-token latency, DONE latency and complete response
latency are distinct fields. No GPU-resident event timing is claimed in this
first lane; adding caller-fork/owned-join/cache/head/sampler events needs a
separate reviewed scope. No extra per-token fence is introduced for timing.

Raw timestamps/tokens, order, errors and telemetry stay task-local. Multi-token
SSE chunks make individual ITL unobservable: retain their timestamps and report
amortized interval rate, with individual ITL summaries disabled for the affected
context. The scalar summary reports paired request-level bootstrap 95% intervals
for six pairs, with no token-independence assumption. It is preliminary; the
measurement protocol's 30-pair minimum remains unmet. A later 30-pair extension
needs a new reviewed budget and must report inconclusive intervals as such.

VRAM comes from roughly 200-ms `nvidia-smi` samples throughout startup, warmup,
measurement and cleanup, with request intervals used to separate lane samples.
These are sampled device peaks, not exact transient or Torch allocated/reserved
peaks. Telemetry and lifecycle sampling costs remain in the guarded lane.

## Startup ownership and private cache admission

The eager lane now also requires `--m1-private-aot PATH` and a source-bound,
CUDA-hidden CPU loader dry-run. The failed pilot at runtime `77af241` reached
startup JIT after relocating a cache whose Ninja outputs were absolute paths.
The sampled aggregate compiler RSS was 3,805,532,160 bytes, above the existing
2 GiB bound. No client request or startup correction fixture ran. The bridge
build passed separately; this failed pilot supplies no timing result. Sanitized
receipts and final cleanup state are in
[m1-eager-startup-failure.json](evidence/m1-eager-startup-failure.json).

The task-local launcher becomes a Linux child subreaper before launch. It scans
owned descendants about every 50 ms, retains PID plus `/proc` start-time
identities across parent exit and `setsid`, and uses verified pidfds for signals.
It sums RSS over all concurrently owned Ninja/NVCC/compiler trees and retains
one 300-second elapsed budget from the earliest compiler start across jobs and
idle gaps. Bounds are sampled, so a brief overshoot can occur between samples;
the observed peak and triggering identities are recorded and invalidate the run.
Host available RAM stays at least 8 GiB and GPU headroom at least 2 GiB.
Any guard failure blocks dispatch and stops verified owned identities. Cleanup
uses bounded TERM/KILL, reaps adopted children, checks remembered identities and
owned GPU PIDs across groups, and preserves the primary failure if cleanup fails.
Dispatch, post-client and final admission read the retained ownership failure
directly; watchdog notification is not the authority. A compiler breach first
detected during successful cleanup invalidates the run while cleanup remains
separately reportable as complete.
Ownership and resource supervision costs remain in both measured lanes.

The next pilot at runtime `6d8f166` passed its fresh bridge controls, six startup
fixtures, and all eight stock/fused requests plus untimed drain. Matched outputs,
usage, actual input transcripts and native backend counts passed. Final cleanup
retained an unlocalized `EINVAL` error and correctly rejected the run, even
though a separate readback found all 180 retained identities absent and the GPU
idle. There is no accepted timing summary. Its compiler classifier also omitted
three `tileiras` identities; zero counters therefore did not establish
quiescence. Executable names cannot distinguish compilation from metadata
probes, and pinned FlashInfer contains a `tileiras --help` architecture probe.
No retrospective acceptance is made. Sanitized failure and CPU reproduction
evidence are in
[m1-eager-pilot-cleanup-failure.json](evidence/m1-eager-pilot-cleanup-failure.json).

A CPU-only owned child exit/reaping test reproduced `pidfd_open(pid, 0)` returning
`EINVAL`, followed by an absent `/proc` identity. Linux 6.8's
[pidfd_prepare implementation](https://github.com/torvalds/linux/blob/v6.8/kernel/fork.c#L2178-L2183)
can return that error after the TGID task disappears. This proves a compatible
mechanism; the original pilot did not record its exact failing operation.
The narrow cleanup fix accepts that `pidfd_open` error only when a fresh read
proves the retained PID/start identity absent or replaced. Same-identity errors,
unreadable identity checks, failed signals and other unknown errors still block
cleanup, even if final identities disappear. Receipts now identify the operation,
cleanup stage, identity and errno. Primary errors remain authoritative.
The compiler classifier includes `tileiras`, reads argv only for compiler
executables, and retains sampled compiler invocations after exit. Exact
`tileiras --help` / `--version` argv is labeled as a metadata probe; extra or
unavailable arguments stay unknown. This classification supplies evidence and
does not exempt probes from resource accounting or timing admission. The pinned
FlashInfer `cutile/cutile_common.py` caller SHA-256 is
`8b80053b84ad68fee19cc66f2b9a8f3e55df2c10be77a71e892da6aa22e35bae`.
Without the separately reviewed opt-in below, final summary admission enforces
the existing stricter all-server-lifetime gate, including zero-RSS zombie
compiler observations. The descriptive `--version` label supplies no exemption.

### One pinned metadata command, default off

`--m1-timing-metadata-help` must be supplied to both plan preparation and the
eager launcher. Its Boolean value is part of the source-bound plan digest;
omission retains strict compiler rejection. Other clients reject this flag.
It distinguishes only argv `[/usr/local/cuda/bin/tileiras, --help]`, resolved
executable `/usr/local/cuda-13.3/bin/tileiras`, and binary SHA-256
`88737a8be5c56bf73fb885a567a950f247a8cfa1d146dbc7a65eff77e7d62bf0`.
The pinned [official FlashInfer caller](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18.post1/flashinfer/cutile/cutile_common.py)
uses that command to inspect architecture support. The reviewed proposal is
SHA-256 `fecc1f67f9bbac374235ffbf82e0ce83d362ce3269237bfa2a8761e5ef027d1a`.
Its separate CPU syscall receipt corroborates only this pinned command; it is
not a universal proof of proprietary tool behavior or acceptance of pilot 2.

Before launch and after owned cleanup, untimed stable-file hashing verifies the
binary and caller against their pinned hashes. Between those hashes, file
versions include device, inode, size, mtime_ns and ctime_ns; same-inode content
drift invalidates admission. Compiler samples recheck actual `/proc/PID/exe`
path and file version, exact cmdline, PID and start ticks. Missing or unstable
evidence, path aliases, extra args, `--version` and initial zombies fail closed.
A verified terminal zombie can retain earlier help evidence only for the same
PID/start identity. Every observed unknown/work transition stays permanently
blocking, even if the same identity later returns to help. Descendants receive
no exemption. Successful final file verification is required before summary.

This distinction affects timing admission only. All tool and descendant RSS,
the shared 300-second budget, ownership identities, error handling and cleanup
remain accounted for. File/version and process sampling costs remain in both
guarded lanes; command execution during request intervals stays inside latency.
Nothing is subtracted or shifted beyond the existing interval boundaries.
Sampling can miss an entire brief process or a transition between samples;
these receipts establish sampled classification, not an instruction trace.
The lifetime gate remains strict for every nonmetadata or unknown compiler,
including startup activity. There is no new warmup-relative boundary.

Private mount isolation was unavailable on the assigned host. Instead, eight
exact prebuilt module hashes are promoted through FlashInfer's supported private
`flashinfer_jit_cache` provider. All four installed loader files byte-match the
official `v0.6.18.post1` release. Private AOT symlinks resolve to the same private
incumbent `.so` bound by the bridge, retaining its native relocation checks.
Installed package files, shared caches and checkpoint are untouched. A private,
source-bound Python startup hook rejects cache admission errors and forbids
`JitSpecNvcc.build` only for those eight promoted names; other specs retain their
ordinary loading policy under the shared resource guard. No global JIT-disable
or version-bypass flag is enabled. The CPU dry-run loads all eight through the
unmodified AOT fast path with rebuilding forbidden, checks module/Ninja hashes,
and verifies CUDA stayed uninitialized. This checks loader reuse, not kernels,
startup correction, quality, or GPU timing.

For a CUDA-hidden dry-run on this pinned loader, set
`FLASHINFER_CUDA_ARCH_LIST=12.0f` only for that CPU check. With no visible device
and no explicit architecture, `CompilationContext` selects the version-only JIT
directory rather than `120f`, and the strict private workspace guard rejects it.
The fresh run preserved that initial exit-78 setup failure. Both path probes kept
CUDA uninitialized; the corrected CPU invocation loaded all eight modules with
zero build calls. The live launcher clears this architecture override and uses
actual GPU discovery. No loader guard, version policy or source changed.

## Source review and serialized GPU gates

Freeze a clean commit/tree, CPU results and this scope with the parent and
independent reviewer **before** any GPU execution. Rebuild the bridge from that
exact source; do not reuse prior PR14 binaries or promote their old results.
Keep packages, checkpoint, system, Tailscale and SSH settings untouched. Inspect
idle/owned processes and bounds before each launch; run one server at a time.

1. Fresh exact-source build: installed pins, four relocations, versioned contract,
   both ABI entry points, compiled framing/owner/lease/binding controls, binary
   and controller/driver hashes. Recheck all six native/reference startup fixtures.
2. Fresh observer-off 2K/8K pilot with both lanes, one complete warmup per
   context/lane and one matched pair per context. Require actual request/frame
   and backend counts, transcript hashes, identical 256-token outputs/usage,
   natural correction counters, repeated lane switches/workspace reuse, and
   successful owned identity/GPU cleanup. Require no observer/probe or diagnostic
   capture destination or preparation/NPZ/request-trace/call-receipt output.
   Its scalar checks are not a CUDA trace, tensor oracle or quality proof.
   Timing receipts must establish compiler quiescence after warmups. For the
   initial bounded comparison, reject timing if the ownership receipt shows any
   sampled compiler activity during the server lifetime, except the exact
   metadata command under the opt-in and evidence requirements above. Ordinary
   other specs may compile, but those runs supply no accepted timing result under
   this plan. This is enforced before summary publication, separately from
   resource bounds and successful cleanup. Any additional exemption or narrower
   warmup-relative scope needs new source-bound evidence and review.
3. Only after the affected source/build and pilot gates pass, prepare the six-pair plan
   and execute one fresh observer-off warm server. Any failed gate blocks timing.
   Publish only sanitized scalar results and source/test hashes, never prompts,
   tensors, full profiler data or host logs. Draft PR authorized; merging remains
   outside this milestone.

The independent review accepts this minimum scope because native safety and
arithmetic sources are unchanged. Do not repeat the complete historical
captured-controlled, external-observer or normal suites by default. A pilot
mismatch or further relevant source changes require affected diagnostics and
fresh review before timing proceeds. Historical numerical results retain their
own source scope and do not become new full-model quality evidence.

Exact next timing invocations, after source approval and the above gates:

```sh
python scripts/prepare_m1_eager_benchmark.py --model-path "$MODEL" --output "$PRIVATE_PLAN" --trials 6 --warmups 2 --seed 9471
python scripts/run_scale_validation.py --mode native --client m1-eager-benchmark --m1-eager-benchmark-plan "$PRIVATE_PLAN" --m1-private-aot "$PRIVATE_AOT" --m1-preparation stock --m1-execution capture-free --m1-bridge "$BUILD/m1_live_bridge.so" --m1-build-receipt "$BUILD/build.json" --label m1-guarded-eager-pairs6
```

For the reviewed single-command distinction, append
`--m1-timing-metadata-help` to **both** commands. A mismatch fails before launch.

Here `--m1-preparation stock` supplies explicit initial M1 admission; the exact
plan selects both lanes at verified request boundaries. All subsequent lane
choices are recorded in the plan, raw client records and dispatch ledger.

## Fresh eecfc13 GPU result

Measured source was `eecfc13e1be9aaf4e361888d5f4c65ad22fb41c4`, tree
`3c9e248405d2f97677223759e86174f09f068d08`, including integrated main
`5a32504b5ccb209a779b8a110eb2ebc56d72e373`. All 21 eager/native runtime inputs
were byte-identical to independently reviewed `dfd77d8`; all 214 source files
and private adapter copies were rechecked after both runs. This result update
changes documentation and scalar evidence only. Prefill GPU work was not run.

The fresh bridge build took 6.112 seconds with 731,148,288-byte aggregate compiler
RSS. Compiled lease and binding controls passed, installed pins stayed unchanged,
and binary SHA-256 was
`f5a11f7d1e95de40abc0b58e017403a60c7789226e954c2bece10945b0389371`.
Each fresh server passed six native/reference startup fixtures: 30 registered
adapters, 54 equal original tensor captures, zero BF16 maximum absolute
difference, 18 native SM120 dense launches and six GELU launches. Their startup
profiles precede the client and are outside the timed request phase.

The separate eight-request pilot passed before the fresh 32-request comparison.
Both used eager synchronous corrected native execution, BF16 KV and concurrency
one. The comparison used two warmups per context/lane, six adjacent randomized
pairs per context and three of each first-lane order. Every 256-token pair had
identical IDs and usage, with actual input transcripts and native backend counts
verified. Comparison totals were 122,400 stock decode calls, 122,400 fused decode
calls and 19,200 stock prefill fallback calls; the final drain was untimed.
Observer, process probe, request profiler, preparation NPZ and call-capture
artifacts were absent. All 40 requests across both runs, including warmups, had
zero natural correction selected-row counts. The forced fixtures do not turn
that result into natural correction coverage or full-model quality acceptance.

All table latencies are request medians. Decode ITL is the amortized interval
from first to last delivered token over 255 intervals. Positive reduction means
fused is faster; uncertainty resamples matched request pairs, using 2,000
bootstrap resamples with seed 9471.

| Input context | Stock / fused TTFT (ms) | Stock / fused decode ITL (ms) | Fused ITL reduction, 95% interval | Stock / fused complete response (s) |
|---|---:|---:|---:|---:|
| 2,048 | 259.110 / 259.287 | 43.592 / 43.960 | -0.844%; [-1.230%, +0.154%] | 11.375 / 11.472 |
| 8,192 | 1040.345 / 1031.006 | 43.695 / 43.908 | -0.489%; [-1.205%, +0.308%] | 12.175 / 12.235 |

Both contexts had single-token SSE chunks, permitting the receipt's separate
individual-ITL tail summaries. The measured critical path includes all retained
route/descriptor copies, synchronization, owner/lease checks and stream handoffs,
along with prefill, head, sampler, KV, supervision and local streaming. It does
not isolate kernel time. Six pairs remain below the unchanged 30-pair minimum;
neither a speedup nor a statistically established regression is claimed.

Both comparison lanes reached the same sampled device peak of 21,377 MiB,
with minimum sampled free memory 10,765 MiB. Pilot peak/free values were
21,719/10,423 MiB. The host 8-GiB guard never failed; no exact host minimum or
transient device peak was recorded. The 8-MiB additional scratch contract stayed
unchanged. The RTX 5090 used driver 610.43.02 and a 575-W power limit.

The pilot observed five compiler-tool identities and the comparison one. All
had actual PID/start, exact help argv, executable path and pinned file-version
evidence, positive metadata samples and no unknown/work history. Final binary
and caller hashes matched preflight. Resource accounting included every tool:
pilot peak aggregate compiler RSS/shared elapsed budget were 34,603,008 bytes /
29.355 seconds; comparison values were 19,660,800 bytes / 0.030 seconds. The
50-ms process scan can miss brief processes; this is sampled evidence. No new
timing exception or warmup-relative compiler boundary was introduced.

Driver, client and run exits were zero for both fresh runs. Automated cleanup
passed without errors. Independent final readback checked all 604 retained
identities as absent or reused, found no vLLM/compiler/GPU compute processes,
and confirmed port 18000 closed and GPU idle at 41 MiB used / 32,101 MiB free.
The exact-source CPU CI passed 164 scaffold tests on Python 3.10 and 3.12,
470 independent numerical tests and 4,064 finite representation products.

The [scalar receipt](evidence/m1-eager-eecfc13-gpu-scalars.json) retains source,
plan, binary, audit and result hashes, latency distributions and scope limits.
Raw prompts, tokens, tensors, profiles and host logs remain private. Earlier
startup/cleanup failures remain rejected and preserved. G1, broader quality,
the legacy benchmark gate and CUDA graph admission remain open.

### Source-based next measurement hypothesis

The first attribution target is retained descriptor validation:
[`setupTmaWarpSpecializedInputs`](../probes/m1_live_bridge.cu) copies seven
128-expert tables to host and fences for each of FC1 and FC2. This is fourteen
copies and two stream fences per eligible layer call, plus host allocation and
SF-carrier enumeration. Across thirty eligible decode layers, source counting
predicts 420 descriptor copies and sixty fences per token. Route eligibility in
[`candidate_eligible`](../kernels/m1_installed_preparation.cuh) also copies the
eight route IDs and fences. The map hook checks once in either lane; fused
`prepare` checks again, adding one copy/fence per eligible fused layer call.

These are likely important costs at concurrency one, but this run did not
measure their duration or fraction of decode latency. Other candidates include
[`LivePreparation`](../src/megartx/m1_live.py) token/position CPU copies,
producer-to-owned and owned-to-caller waits, owner/lease checks and allocator
stream recording, plus the correction adapter's data-dependent
[`torch.nonzero`](../src/megartx/vllm_scale_plugin.py) even on zero-hit requests.
The candidate point estimate being slightly higher does not show that the fused
kernel itself is slower. The next useful step is separately reviewed, untimed
host/stream attribution on the same eager scope. Any proposed validation
amortization needs fresh evidence for dynamic routes, descriptors, lifetimes
and unchanged failure behavior; deleting checks is not a supported conclusion.

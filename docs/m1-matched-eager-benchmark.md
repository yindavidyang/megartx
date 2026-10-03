# Guarded matched eager M1 preparation comparison

This milestone adds a default-off, exploratory timing path from merged main
`09341f2ad3f6e4b03e8f264b9e3c59fe5ddb52ed`. It compares **guarded stock eager**
and **guarded fused eager** preparation in the same corrected native runtime.
It does not pass G1, qualify full-model quality, admit CUDA graphs, or reopen
the old `--client benchmark` / `host_benchmark.py` gate. Historical graph and
uncorrected baselines are not comparison inputs. CPU success is not GPU evidence.

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
   sampled compiler activity during the server lifetime; this stricter condition
   avoids accepting an unlocalized compilation interval. Ordinary other specs
   may compile, but those runs supply no accepted timing result under this plan.
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

Here `--m1-preparation stock` supplies explicit initial M1 admission; the exact
plan selects both lanes at verified request boundaries. All subsequent lane
choices are recorded in the plan, raw client records and dispatch ledger.

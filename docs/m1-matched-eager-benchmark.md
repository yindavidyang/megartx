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

## Source review and serialized GPU gates

Freeze a clean commit/tree, CPU results and this scope with the parent and
independent reviewer **before** any GPU execution. Rebuild the bridge from that
exact source; do not reuse prior PR14 binaries or promote their old results.
Keep packages, checkpoint, system, Tailscale and SSH settings untouched. Inspect
idle/owned processes and bounds before each launch; run one server at a time.

1. Fresh exact-source build: installed pins, four relocations, versioned contract,
   both ABI entry points, compiled framing/owner/lease/binding controls, binary
   and controller/driver hashes. Recheck all six native/reference startup fixtures.
2. Exact-source captured controlled stock/fused pair with separately labeled
   artificial first-use/disjoint-reuse route controls; strict `compare_m1_live`.
   Fresh observer-enabled controlled stock/fused runs, pre-dispatch EngineCore
   ownership and strict external comparison; instrumentation is never timed.
3. Fresh observer-off stock and fused controlled cached preflights: response
   usage, no observer/probe import or destination, no preparation/NPZ/trace/call
   receipt, successful owned cleanup. These establish the off contract separately.
4. Exact-source captured normal stock/fused two-request plan and strict normal
   comparison establish new-request/workspace-reuse fidelity and dispatch at
   their bounded natural prompts. Their constrained continuations do not qualify
   timing or G1. Then the separate 2K/8K observer-off pilot exercises the actual
   benchmark plan, natural output matching, lane switching, counts/transcripts
   and full cleanup. Its scalar status is not a CUDA trace or tensor oracle.
5. Only after the affected source/fidelity gates pass, prepare the six-pair plan
   and execute one fresh observer-off warm server. Any failed gate blocks timing.
   Publish only sanitized scalar results and source/test hashes, never prompts,
   tensors, full profiler data or host logs. Draft PR authorized; merging remains
   outside this milestone.

Exact next timing invocations, after source approval and the above gates:

```sh
python scripts/prepare_m1_eager_benchmark.py --model-path "$MODEL" --output "$PRIVATE_PLAN" --trials 6 --warmups 2 --seed 9471
python scripts/run_scale_validation.py --mode native --client m1-eager-benchmark --m1-eager-benchmark-plan "$PRIVATE_PLAN" --m1-preparation stock --m1-execution capture-free --m1-bridge "$BUILD/m1_live_bridge.so" --m1-build-receipt "$BUILD/build.json" --label m1-guarded-eager-pairs6
```

Here `--m1-preparation stock` supplies explicit initial M1 admission; the exact
plan selects both lanes at verified request boundaries. All subsequent lane
choices are recorded in the plan, raw client records and dispatch ledger.

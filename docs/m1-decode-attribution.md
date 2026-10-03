# Bounded decode attribution and lean execution

Base: verified green main `0581f09259d18044ffb5f9e1363d0760f37ddd4f`.
This milestone measures the retained eager preparation path before changing it.
Historical six-pair results retain their metadata-admission uncertainty and
show no established speedup. The old graph/uncorrected result is not a comparator.
The [historical hold](evidence/m1-eager-metadata-admission-review.json) remains;
new evidence cannot retroactively admit those runs or pass G1/quality/graphs.

## Check inventory

| Check | Frequency and reason | Disposition |
| --- | --- | --- |
| Checkpoint, package, installed ELF/header/relocation identities, compiled ABI and controller/driver hashes; static architecture/configuration and thirty registered adapters | Initialization; qualify exact source and installed lane | Retain; immutable source facts can be initialized once |
| Original typed quantization owners and original-byte fixture comparison | Initialization fixtures and actual retained owner pointers per call | Retain; no quantizer/math change |
| Request marker/plan, actual tokens/positions, synchronous progression, lane transitions, global call cap, frame/backend ledger | Every actual frame/call/request; input and request state change | Production checks; retain |
| Callable registry, runner callsite, exact operands, fifteen owner extents, six quantization owners, workspace partition and alias checks | Every invocation; owner/lifetime/state may change | Production checks; retain |
| Capture status, device/context/stream identity, actual allocation extents, route range/uniqueness | Every eligible invocation; dynamic routes, allocations and stream state | Production checks; retain; repeated evaluation in the same fused map invocation is the narrow reuse candidate |
| Actual FC1/FC2 descriptor shapes/layouts/pointers/strides, full physical SF carrier bounds, dense carrier enumeration, AQ/output extents | Every eligible invocation; actual descriptors are produced dynamically | Production checks; retain in first lean change; separately measure copies/fences and enumeration |
| Native lease framing and exactly-once map/expansion/runner; CUDA/driver/submission/consumer errors | Every call; safe fallback only for successful unsupported queries | Retain; errors poison the context and never retry stock after candidate mutation |
| Producer-to-owned and owned-to-caller waits; allocator `record_stream`; owners kept through native call | Every routed call; preserve dependency and lifetime | Production checks; retain |
| Six forced startup fixtures, captured original bytes/stages, masks, external observer, artificial routes | Explicit diagnostics/startup proof | Keep separate from natural request coverage; no optional diagnostics in clean timing |
| Owner PID/start identity, compiler aggregate RSS/shared elapsed budget, host/GPU headroom, exact metadata argv/executable/file identity, final cleanup | Entire owned server lifecycle | Retain unchanged, including metadata correction merged on main |

## Frozen first profiling protocol

The instrumentation commit is measured **before** the eligibility reuse commit.
No eligibility or descriptor check is removed in the original-path profile.
Parent review and explicit GPU clearance are required before the following work.

- One isolated source/adapter/runtime/cache/AOT/bridge under a new task directory.
  Preserve baseline packages, checkpoint, system settings, shared Mac repo and
  sibling worktrees. Verify current host state before any GPU operation.
- Checkpoint `nvidia/Gemma-4-26B-A4B-NVFP4` revision
  `a19cfe00be84568a6867111c9a68c9c44fdcffe6`; Python 3.12.3, vLLM 0.30.0,
  FlashInfer 0.6.18.post1, Torch 2.13.0+cu130, RTX 5090/SM120.
- Exact existing eager pilot: seed 9471, trials=1, warmups=1, 2,048/8,192
  private exact input IDs, greedy 256 outputs, ignore EOS, BF16 KV, 256-token
  prefill chunks, one sequence, synchronous eager, prefix caching off.
  Eight requests / 61,200 live decode calls / 4,800 prefill fallback calls,
  then one separate one-token drain. The 8K requests remain because the current
  source-bound pilot requires both contexts; they are not profiled.
- All four whole-request warmups precede the measurement requests. No separate
  profiler warmup or instrumented performance trial is implied. CUPTI/profiler
  initialization can perturb the first active frame and this is disclosed.
- Start Torch CPU+CUDA profiler before marker/token/position checks for the
  first single-row forward of each 2K measurement lane. Collect four complete
  forwards, heads, sampling and inter-frame scheduler gaps. Stop at the next
  forward boundary, before its checks. Exactly eight full decode steps total.
  Stop/export fences are instrumented overhead, outside subsequent active work.
  No tensor shapes, stacks, memory payloads or tokens are recorded by the profiler.
- Source-bound plan and native contract identify the request and lane. Python
  record-function names identify each layer and preparation lane; CUDA events
  carry profiler correlation IDs and actual stream IDs. Original producer/owned
  waits and lifetime recording remain. Native thread-local host timers accumulate
  runner, first eligibility, dispatch (including repeated fused eligibility),
  descriptor readback/fence, and descriptor validation/enumeration durations.
  Inclusive runner time contains the other phases; totals must not be added.
  Readback/fence host time includes preceding stream work and owner/submission
  checks; it is not isolated device-copy time. Correction-selection layer labels
  require their enclosing routed range and ordered preparation correspondence.
  Evidence review must verify four model/head/sampling frames, CUDA launch/kernel
  correlation and actual streams. Expected native phase counts per lane are
  120 runner/eligibility/dispatch and 240 readback/validation; trace existence
  and source markers alone do not establish these attribution relationships.
- Read native timers once per window after stop. They do not introduce per-call
  D2H copies or fences. Profile failures propagate and abort releases native
  profiling state. Inactive leases are required to toggle profiling.
  Frame-ledger and logits-head failures also stop/reset attribution and mark the
  controller failed; cleanup failures preserve the primary error and never retry.
- Fresh bounded bridge build and compiled framing/binding controls, source-bound
  CUDA-hidden AOT dry-run and six native/reference startup fixtures are required.
  Metadata help opt-in remains exact-command/identity bound, with corrected
  retained terminal evidence and trailing empty argv checks. No broader exemption.
- Compiler aggregate <=2 GiB/shared <=300 seconds; host available >=8 GiB;
  GPU free >=2 GiB; extra execution scratch <=8 MiB. No new execution tensors or
  scratch allocation. Existing sampled guards include profiler resource costs.
  Keep existing 20-minute startup and 3,600-second client total caps; no sweep.
- Raw traces/tokens/prompts/logs stay private. Only project code, Markdown and
  reviewed sanitized scalar counts/durations/hashes can be published. Instrumented
  runs refuse eager-summary generation, including direct summary replay.
  Check exact sources/adapter copies after the run and retained owned cleanup.
  Report CPU phase times, actual GPU copy/fence/wait counts and durations, GPU
  activity/gaps, preparation and major compute/attention/head kernels with their
  attribution/correlation limits. No clean timing or speedup inference.

Exact commands, with task-local paths and frozen head supplied in the review:

```sh
python scripts/prepare_m1_eager_benchmark.py --model-path "$MODEL" --output "$PRIVATE_PLAN" --trials 1 --warmups 1 --seed 9471 --m1-timing-metadata-help
python scripts/run_scale_validation.py --mode native --client m1-eager-benchmark --m1-eager-benchmark-plan "$PRIVATE_PLAN" --m1-private-aot "$PRIVATE_AOT" --m1-preparation stock --m1-execution capture-free --m1-bridge "$BUILD/m1_live_bridge.so" --m1-build-receipt "$BUILD/build.json" --m1-timing-metadata-help --m1-decode-profile --label m1-decode-original-profile
```

After attribution review, freeze any lean implementation separately, prove that
the checked inputs/stream/state stay unchanged through its immediate use, test
unsupported routes and error/cleanup behavior, then request clearance for a
fresh clean one-pair pilot with profiling disabled. Graph restoration and broader
fusion remain later work. Draft publication does not authorize merging.

# RTX 5090 Gemma 4 decode master plan

*30 September 2026 | Contract 1.0 | Markdown initialization*

Objective: outperform a tuned, compatible FlashInfer backed runtime on the owned RTX 5090 for end to end single request Gemma 4 FP4 decode latency, while preserving model behavior.

The core hypothesis is that a model specific schedule can coordinate dependencies, activation lifetimes and weight transfers across operator boundaries better than separately optimized kernels. FlashInfer already has specialized kernels. The opportunity must therefore be measured against tuned FlashInfer plus CUDA Graphs, not an untuned or unsupported configuration. A whole decoder megakernel is one possible outcome, not a prerequisite for success.

### How the two plans fit together

This master plan governs scope, requirements, interfaces, evidence and decisions. The companion [RTX 5090 Gemma 4 Decode Action Plans](action-plans/README.md) describes execution tasks A0.1 through A6.4. Work package IDs WP0 through WP6 and gate IDs G0 through G6 are shared. If task steps conflict with this document, resolve the conflict before execution and update both plans under the same decision record.

### Decision register

| ID | Status | Decision |
| --- | --- | --- |
| D01 | Confirmed | Start on the owned RTX 5090 SM120; four RTX PRO 6000 GPUs remain future work |
| D02 | Confirmed | One active user request; no dynamic request batching |
| D03 | Confirmed | Aim to outperform FlashInfer on the RTX 5090 |
| D04 | Proposed | Gemma 4 26B A4B instruction model; NVIDIA NVFP4 checkpoint revision to freeze |
| D05 | Proposed | Text only; 8K initial cached context target; BF16 KV; output capacity reserved separately |
| D06 | Proposed | Native expert NVFP4 W4A4 candidate; W4A16 is a separate numerical lane |
| D07 | Proposed | 15% median ITL improvement, 20% stretch, quality/tail guards and 2 GiB initial reserve |
| D08 | Partially authorized | Repository initialization in Markdown and a draft WP0–WP1 compatibility/benchmark scaffold are authorized; target RTX host access and hardware execution remain pending |

D04 through D07 are planning defaults, not user mandates. The model candidate and native FP4 direction motivate this investigation; the exact checkpoint, numerical path, context and acceptance margins still require a documented freeze.

### Scope and authority

The authorized initial scope is repository initialization with this master plan, the action plans and README, followed by a draft PR for the WP0–WP1 compatibility and FlashInfer benchmark scaffold. This authorization does not freeze D04–D07, authorize GPU experiments or custom kernels, or imply any gate has passed. Host setup and target-hardware execution remain pending. Proceed through evidence gates, retain a correct fallback and stop at the smallest design that meets the goal. No hardware performance result has yet been established. See the [decision ledger](decision_ledger.md) for the current scope.

## Requirements and research hypotheses

| ID | Requirement | Evidence gate |
| --- | --- | --- |
| R01 | Use the actual RTX 5090 and SM120 capabilities | G0 |
| R02 | Optimize one active request without dynamic request batching | G1 and G4 |
| R03 | Make tuned compatible FlashInfer backed end to end decode the primary incumbent | G1 and G4 |
| R04 | Freeze checkpoint, quantization coverage, layouts and workload | G0 and G1 |
| R05 | Preserve model semantics; validate each numerical lane against its own oracle | G1 through G5 |
| R06 | Prove progress, memory safety and bounded kernel lifetime | G2 through G5 |
| R07 | Measure complete memory and token cost including output head and KV | G0 and G1 |
| R08 | Pin artifacts and use paired same checkpoint comparisons | All gates |
| R09 | Keep separate prefill, exact cache handoff and a graph based fallback | G1 through G4 |
| R10 | Require compatible drafting, exact acceptance and safe rollback | G5 only |
| R11 | Measure future topology; assume neither NVLink nor scaling benefit | G6 only |
| R12 | Obtain implementation authorization and host/repository access before execution | Before WP0 execution |

### What the references establish

Inferact fused 92 Kimi K3 layers using Pallas on 16 TPU v7 chips with 64 MiB VMEM per TensorCore. Its DSpark verifier uses one anchor and seven proposals; the reported result exceeds 700 output tokens per second at acceptance length six versus 452 on 16 GB200 GPUs. Explicit lifetimes and asynchronous staging are relevant ideas. That different model, memory system, topology and acceptance rate cannot predict RTX performance. [1, 2]

DeepGEMM Mega MoE fuses dispatch, FC1, SwiGLU, FC2 and combine with NVLink overlap on SM90/SM100. Its FP8 by FP4 path is not an NVFP4 SM120 backend. CUTLASS instead documents SM120 mma.sync block scaled GEMMs and GeForce grouped examples; do not import SM100 tcgen05 or TMEM assumptions. [3, 4, 5]

### Hypotheses to accept or reject

H01: cross operator scheduling hides memory bubbles left by FlashInfer plus graphs. H02: early routing overlaps selected expert tile fetches with the shared branch. H03: native W4A4 can beat W4A16 at these low M shapes after quantization and packing costs. H04: bounded persistence retains that gain without spills, idle workers or synchronization cost. Existing SM120 libraries already cover small batch and persistent MoE; generic format support alone does not prove Gemma compatibility. [10, 11, 12]

## Model contract and memory envelope

### Selected model dimensions

| Contract item | Google reference configuration |
| --- | --- |
| Decoder and experts | 30 layers, hidden 2816; 128 experts, top 8; routed width 704; shared width 2112 |
| Attention pattern | 25 sliding layers with window 1024 and 5 global layers in a five plus one pattern |
| Head geometry | 16 query heads; local head width 256 with 8 KV heads; global width 512 with 2 KV heads |
| Output and activation | Tied 262144 vocabulary; logit soft cap 30; gated GELU tanh, not SwiGLU |
| Special options | No per layer input embedding path; global K/V projection sharing does not alias cached K and V |

Reconcile this contract with the pinned NVIDIA checkpoint before packing. The NVIDIA card describes experts only weight and activation NVFP4 quantization and B200 evaluation; its repository is about 18.8 GB. Disk size does not prove runtime fit on the RTX 5090. [6, 7, 9]

### Complete allocation ledger

Record stored tensors, scales, runtime packing/padding, retained original copies, cache capacity, scratch, graph pools, allocator reservations and peak prefill/decode allocations. Measure actual free memory and display use. Avoid expanding all experts to BF16. The proposed initial reserve is at least 2 GiB; any reduction needs measured peak evidence. Gate G0 requires a resident workload without CPU weight offload.

For BF16 K and V with bounded local caches, logical KV bytes at cached length L are 2 × 2 × [25 × 8 × 256 × min(L,1024) + 5 × 2 × 512 × L]. Padding, paging, scratch and allocator overhead are excluded. A sliding attention mask alone does not establish bounded allocation. [6]

| Initial cached length | Bounded local KV | All local history retained |
| --- | --- | --- |
| 2048 | 240 MiB | 0.430 GiB |
| 8192 | 360 MiB | 1.719 GiB |
| 32768 | 840 MiB | 6.875 GiB |

### The full token cost matters

Build a byte/time ledger for selected expert weights, dense/shared weights, attention projections, output head, scales, KV reads and intermediate writes. Tied embeddings save allocation, not the vocabulary projection’s per token work: a BF16 262144 by 2816 matrix is 1.375 GiB before caching or quantization. Profile it and growing global attention alongside MoE. [6]

Use measured traffic and attainable bandwidth for comparable accesses, including scattered experts. For each stage compare time with max(bytes/bandwidth, operations/compute), then account for dependencies, valid overlap, launches, barriers, sampling and host streaming. Amdahl’s bound determines whether the selected hotspot can plausibly meet G4; eliminating a fraction f gives at most 1/(1−f) speedup.

## Execution architecture and core interfaces

### Preferred progression

Start with a proven prefill and graph captured decode path. Replace only measured hotspots. Move from small fusions to persistent MoE/block regions, then optionally cross layer or full token persistence. Fixed model shapes and context buckets permit preallocated scratch and GPU resident metadata. Keep one bounded invocation per token initially; an indefinitely running device server is outside the MVP.

The reference dependencies suggest early routing alongside the shared dense branch after attention. Once top 8 IDs exist, stage only selected expert tiles. Deterministic next layer projection tiles can be lookahead candidates; next layer routes are not yet known. Measure contention because overlapping work can reduce effective bandwidth or compute utilization. [9]

Registers hold short lived fragments; shared memory holds bounded tiles; explicit global buffers carry persistent cross block state. L2 residency is opportunistic, never a correctness guarantee. Monitor register allocation, spills, occupancy, instruction cache pressure and live ranges. NVIDIA lists SM120 limits of 48 resident warps per SM, 64K registers, 128 KB shared memory per SM and 99 KB per block. Query the actual device and compiled kernel. [13]

### I01  Checkpoint and packed tensor manifest

Owner: model and correctness lead. Each tensor record has source revision/hash, logical shape, dtype, quantization coverage, block/global scales, layout, padding, alignment, packed checksum and tied storage identity. Packing is a versioned transform with round trip fixtures. No backend may silently reinterpret NVFP4 as another four bit format or retain unbudgeted copies. Consumers are all operator and runtime implementations; changes reopen G0 and relevant G1 tests.

### I02  Operator numerical contract

Owner: model and correctness lead with kernel engineer. Define input/output shapes and strides, input/scaling provenance, accumulator/output types, exact activation, rounding boundaries, reduction ordering, scratch ownership and supported M/context ranges. Each implementation declares its numerical lane: W4A4, W4A16 or BF16. Dispatch must expose the selected implementation and fallback reason. Unsupported semantics are an explicit error or verified fallback, never an approximate substitution.

### I03  Prefill and KV state handoff

Owner: runtime engineer. State includes layer/cache geometry, page/ring layout, K and V dtype, absolute position, committed length, capacity and sliding window origin. Prefill and decode must agree on masks, RoPE positions and normalization. A conversion is tested against a continuous reference sequence, including chunk and window boundaries. Global projection sharing does not mean cached K equals V. Allocate output token capacity beyond each initial prompt length. [9]

Invariant: the cache presents exactly the committed sequence under each layer’s mask. A decode failure must leave recoverable state or restart through the verified fallback; stale or partially written cache state is never reused silently.

## Safety and numerical invariants

### I04  Decode schedule and synchronization

Owner: runtime engineer; independent reviewer approves the progress argument. The first cross block prototype uses cudaLaunchCooperativeKernel and this_grid().sync() only after cooperativeLaunch support is checked. Bound the grid by occupancy for the actual compiled kernel and dynamic shared memory size. Every CTA, including idle ones, reaches every required grid barrier. Resource pressure in one phase can depress occupancy throughout the kernel. [14]

Never spin on work from potentially nonresident blocks. Later queues require bounded capacity, a progress invariant, release/acquire publication and generation tagged completion flags. PDL overlap is opportunistic and cannot be needed for correctness. Keep cancellation between token steps and a graph based fallback. Require sanitizer and repeated stress evidence before widening fusion. [16]

### Three numerical references

Use BF16 for authoritative model semantics and quality context; a simple W4A4 oracle for the selected weights, scales and exact activation quantizer; and the tuned compatible FlashInfer backed runtime for performance comparison. W4A16 gets its own oracle and scorecard. It consumes higher precision activations, so a speed/quality difference cannot be attributed solely to scheduling.

For W4A4, freeze clipping, scale calculation, global and block scale conventions, rounding/saturation, packed layout, accumulation and output cast points. Compare both quantized operands and resulting outputs. Packed values and scale tensors should match exactly where the contract demands it; mathematical equivalence without the same quantization boundaries is insufficient.

### Required semantic invariants

- Router: no scale RMSNorm, learned input scale divided by square root of hidden width, FP32 softmax, top 8 renormalization, then learned per expert scales. Compare selected IDs and weights; investigate near ties explicitly.

- Feed forward: gated GELU tanh; separate shared/routed normalizations, common post normalization and residual. Preserve the reference’s operand paths and reduction contract.

- Attention and output: Q/K/V normalization, head geometry, rotary dimensions, causal/sliding masks, exact positions and logit soft cap. Keep stored K and V distinct. [6, 9]

### Evidence required before a numerical gate passes

Record maximum absolute/relative error, normalized RMSE, cosine similarity where meaningful, router ID mismatch rate, selected weight error, top logit agreement and first teacher forced divergence. Freeze tolerances from the oracle’s rounding envelope before tuning. Never loosen a bound to make a candidate pass. Any NaN, infinity, bounds violation, stale cache observation or unexplained systematic routing mismatch blocks promotion.

Use adversarial scale boundaries, near tied routers, repeated expert hits, 1023/1024/1025 window lengths, prompt chunk boundaries and capacity limits. Teacher forced traces localize error before autoregressive divergence. Long generation and held out task quality complement these checks; token equality alone neither proves nor disproves numerical correctness.

## FlashInfer baseline and acceptance contract

### I05  Reproducible benchmark record

Owner: benchmark owner. Pin host runtime, FlashInfer, CUDA, driver, PyTorch, attention/MoE backends, checkpoint/tokenizer revisions, graph buckets, power configuration and dispatched kernels. FlashInfer is a library; name its runtime and every non FlashInfer operator. Verify full Gemma semantics and report unsupported paths precisely. An unavailable or incorrect FlashInfer configuration cannot count as a performance win. [8, 10, 11, 12]

Tune the compatible FlashInfer incumbent before comparison. Keep non target components fixed where possible. Retain default optimized selection, CUTLASS and eligible b12x configurations as controls. A compatible llama.cpp local user control is useful, with its distinct quantization/quality labeled separately. W4A4 and W4A16 are separate comparisons, not interchangeable baseline settings.

| Measure | Required comparison boundary |
| --- | --- |
| Headline | End to end non speculative batch one decode ITL at 8K; GPU timing reported separately |
| Contexts and outputs | 2K and 8K initial cached prompts; 32K only if fit; reserve output capacity; at least 256 output tokens plus long run |
| Fairness | Identical checkpoint, tokenizer, prompt, cache dtype, sampling, EOS and output policy in each numerical lane |
| Procedure | Warm shapes and graphs; randomized paired trials; initially at least 30 trials per cell, more if uncertainty remains |
| Report | Per request p50/p95 ITL, TTFT, total response time, cold load/compile separately, VRAM peaks and thermal/power state |
| Attribution | Separate profiler runs; kernel/DRAM/L2/spill/barrier/host evidence; live routing in end to end timing |

### Proposed project acceptance thresholds

G4 performance: at least 15% lower median end to end 8K decode ITL versus the tuned compatible FlashInfer backed runtime; 20% is a stretch objective. Proposed guards: p95 ITL no worse by more than 5%, and no unexplained greater than 5% TTFT or total response regression. Report 2K/32K tradeoffs and stronger control results. Kernel microbenchmarks are explanatory evidence, not the headline outcome.

G4 quality: no more than 1% relative held out perplexity increase and 1 percentage point task score decrease versus the selected quantized reference, plus BF16 context where practical. Use paired confidence intervals and enough samples to resolve these margins. Small noisy tests are inconclusive. The project owner must freeze these proposed margins before tuning; strict semantics and safety invariants are not negotiable performance tradeoffs.

Store raw per request/token results, seeds, warmup policy and excluded runs with reasons. Bootstrap paired comparisons, preserve thermal stability and continue trials when uncertainty cannot decide the gate. A fresh upstream FlashInfer release requires a separately pinned rebaseline; never silently change the incumbent during an experiment.

## Core work packages and decision gates

Dependency chain: authorization → WP0 → WP1 → WP2 → WP3 → WP4. Interface and test preparation can overlap where inputs are stable, but no package may claim its gate before dependencies pass. A successful partial fusion path may be the final single GPU result; WP4 is an optional increase in fusion scope, not a requirement to discard a better simpler design.

### WP0  Compatibility and fit

Accountable role: runtime engineer. Action plans A0.1–A0.4. Outcome: actual RTX/SM120 capability record, immutable environment and checkpoint manifests, exact FlashInfer path classification, smoke outputs and allocation timeline. G0: correct resident prefill/decode at 2K and 8K under the chosen reserve, with unsupported paths visible. If FlashInfer compatibility is unresolved, independent reference work may continue, but the headline speed comparison remains blocked.

### WP1  Reference correctness and baseline profile

Accountable roles: model and correctness lead; benchmark owner. Action plans A1.1–A1.4. Outcome: I01–I03 contracts, BF16/W4A4 oracles and fixtures, tuned FlashInfer incumbent/controls, per token byte/time ledger and ranked hypotheses. G1: semantic and numerical tests pass, measurements reproduce, and recoverable time justifies a selected hotspot. If Amdahl’s bound is too small, retarget the bottleneck before writing a new scheduler.

### WP2  SM120 FP4 expert execution

Accountable role: kernel engineer. Action plans A2.1–A2.4. Outcome: verified packing/scales, actual shape FC1/FC2, gated GELU and reduction, native MMA versus unpack/GEMV and W4A16 controls. Top 8 at batch one is eight independent M=1 expert problems, not one shared weight M=8 GEMM. G2: oracle/router checks pass and integrated MoE latency improves over compatible tuned FlashInfer in the same numerical lane, including live routing, quantization, scratch and reduction.

### WP3  Fused decoder blocks

Accountable roles: kernel and runtime engineers. Action plans A3.1–A3.4. Outcome: winning partial/block fusions, early router overlap experiment, liveness and occupancy maps, cache interoperability and fallback. G3: full model regression passes, an integrated gain survives, progress is safe and p95 does not regress without explanation. Preserve the simplest winning boundary; reject variants that merely move work outside the timed region.

### WP4  Cross layer persistent decode

Accountable role: runtime engineer; independent reviewer signs off safety. Action plans A4.1–A4.4. Outcome: a bounded cross layer or full token schedule, graph/persistent ablations, register/spill evidence and sustained stability. G4: the proposed project performance/quality guards in the [acceptance contract](#flashinfer-baseline-and-acceptance-contract) pass with complete resource accounting. If wider persistence loses, promote the best WP3 path if it meets the same project gate, or record the remaining shortfall.

### Required gate record

Every gate record names requirements covered, candidate and baseline hashes, evidence artifacts, unresolved deviations, measured outcome and the next decision. The accountable role proposes pass, revise or stop; the independent reviewer checks evidence, and the project owner approves material scope or acceptance changes. No dates or people are assigned until actual access and capacity are known.

## Optional work packages and state transactions

### WP5  Compatible speculation

Accountable roles: runtime engineer and model/correctness lead. Action plans A5.1–A5.4. Depends on WP1 and an accepted stable WP3 or WP4 path; it does not replace the non speculative evidence. Use a verified compatible Gemma assistant/draft. Kimi’s DSpark model and acceptance results do not transfer. Compare equally enabled FlashInfer and candidate paths using the same draft and settings. [1, 2, 8, 15]

The output is a separate latency/quality decision based on total draft, verify, sample and cache cost per emitted token. Count proposals, accepted draft tokens, bonus target tokens and verifier calls consistently. Measure the union of experts touched and rejected work: additional rows can increase MoE weight traffic enough to erase a batch one gain. G5 passes only with lower wall clock cost and preserved target decoding semantics.

### I06  Draft acceptance and cache transaction

Owner: runtime engineer; correctness lead validates sampling. Keep committed sequence length separate from tentative positions. For greedy decoding, require exact target only agreement. For stochastic decoding, use the target/draft probability acceptance and residual correction procedure consistent with the actual sampler; matching sampled tokens alone is insufficient. Test distribution preservation on tractable cases before model evaluation.

Rejected suffixes never become visible to later attention. A ring cache must not destroy committed entries through speculative overwrites: stage, copy on write or retain a proven rollback buffer. Validate zero/full acceptance, EOS in proposals, page and window wrap boundaries, RoPE positions and cancellation. G5 is blocked by any stale state or acceptance bug regardless of speed. Drafting may remain disabled on workloads where acceptance or p95 is poor.

### WP6  Future four GPU topology

Accountable roles: runtime engineer and benchmark owner. Action plans A6.1–A6.4. Depends on an accepted single GPU baseline plus actual future hardware and authorization. This is a separate capacity/latency experiment, not an MVP dependency. Inventory exact RTX PRO 6000 SKU, PCIe generation/width, CPU roots, NUMA, peer access, IOMMU/ACS constraints and NCCL transport. Do not assume NVLink or uniform peer latency.

Measure small message collectives and peer copies before choosing TP, pipeline or expert placement. Compare TP1 with TP2/TP4 where supported, including all communication and remote expert costs. G6 passes only if the measured result provides the agreed latency or capacity benefit over one GPU. Otherwise retain TP1. More aggregate VRAM can improve capacity without making a single token faster; no fourfold speed claim is justified.

### Dependency and scope rules

WP5 branches from the stable single GPU path. WP6 waits for hardware and may be deferred indefinitely without blocking WP0–WP4. Changes to quantization or checkpoint reopen I01/I02 and applicable G0/G1 evidence. Changes to cache layout reopen I03 and rollback tests. Any new persistent schedule reopens I04 safety review. The action playbook uses these same dependencies and gates.

## Risk register and plan governance

Roles describe responsibilities, not assigned people. The project owner settles workload and quality tradeoffs; technical roles propose evidence based choices; the independent reviewer verifies correctness, fairness and progress arguments before promotion.

| Risk | Owner role | Trigger and required response |
| --- | --- | --- |
| K01 Unsupported FlashInfer path | Runtime engineer | Missing Gemma activation/routing/SM120 path: record exact gap, validate alternatives, defer headline claim |
| K02 Numerical drift | Correctness lead | Packing, routes or logits exceed frozen bounds: localize first divergence before optimization proceeds |
| K03 False FP4 gain | Benchmark owner | W4A4/A16, cache or checkpoint differs: separate lanes and rerun matched comparison |
| K04 Memory expansion | Runtime engineer | Peak load/repack/graph/prefill exceeds budget: remove duplicate buffers or revise proposed workload |
| K05 Deadlock or stale state | Runtime engineer | Residency/progress unproven or sanitizer failure: stop promotion and restore graph fallback |
| K06 Register or occupancy loss | Kernel engineer | Spills or idle worker time erase gain: reduce live scope or split persistent regions |
| K07 Wrong bottleneck | Benchmark owner | Output head/global attention dominates: retarget from expert microbenchmarks using full token ledger |
| K08 Upstream catches up | Benchmark owner | Pinned new FlashInfer/control improves: publish separate rebaseline and narrow custom scope |
| K09 Speculation cost or bugs | Correctness lead | Poor expert reuse, rejection or rollback issue: disable WP5 until its independent gate passes |
| K10 Communication dominates | Runtime engineer | Future PCIe collectives outweigh local savings: keep TP1 or choose capacity-only deployment |

### Decisions required before execution

Resolve D04–D08: exact checkpoint revision and quantization coverage; workload, response lengths, context and KV dtype; primary numerical lane; proposed performance, quality and memory margins; host/repository access and initial execution authorization. Choose the implementation stack after WP1 identifies the bottleneck and needed SM120 primitives. Do not lock into a kernel language for novelty.

### Change control and handoff

Keep decision records with status, rationale, evidence, owner role, affected R/I/WP/G IDs and date. A proposed default becomes frozen only when approved and recorded. Preserve previous benchmark configurations rather than overwriting them. The master plan defines what must hold; the action playbook defines how to produce the evidence. Updates to IDs, dependencies or gates must be reflected in both.

Suggested repository destinations are docs for contracts/decisions, configs for pinned manifests, reference for oracles, kernels/sm120 for implementations, runtime for cache/schedules, tests for invariants and bench/results for raw evidence. These are planned destinations; only artifacts actually present in the repository are implemented. Paths named in action tasks remain proposed until delivered and reviewed. Large weights and traces remain external with hashes. Review licenses before code reuse.

## Primary references

Checked 30 September 2026. These sources establish architecture and ecosystem context, not performance on the target card. Replace mutable main branches and documentation with immutable revisions in the implementation manifest.

[1] [Inferact 700 TPS on Kimi K3 A Case for TPU Megakernels](https://inferact.ai/blog/tpu-megakernels)

23 September 2026. TPU design, VMEM, 16 chip experiment and DSpark results.

[2] [Inferact TPU megakernels source](https://github.com/Inferact/tpu-megakernels)

Implementation and reproduction starting point for scheduling ideas.

[3] [DeepSeek DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe)

Supported architectures, fused MoE scope, formats and NVLink overlap.

[4] [NVIDIA CUTLASS Blackwell SM120 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)

Instruction family, layouts, tile schedules and GeForce cluster restrictions.

[5] [NVIDIA CUTLASS GeForce NVFP4 grouped GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu)

Concrete SM120 grouped GEMM reference; not a complete Gemma backend.

[6] [Google Gemma 4 26B A4B configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)

Layer dimensions, attention pattern, vocabulary and model options.

[7] [NVIDIA Gemma 4 NVFP4 model card and files](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)

Experts only weight/activation quantization, B200 evaluation and file listing.

[8] [vLLM Gemma 4 26B A4B recipe](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it)

Desktop validation context and assistant/MTP build requirements.

[9] [Transformers Gemma 4 reference implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py)

Routing, branch normalization, attention and cache semantics.

[10] [b12x local inference library](https://github.com/local-inference-lab/b12x)

SM120/SM121 kernels and small batch/persistent execution candidates.

[11] [vLLM 0 29 0 b12x backend documentation](https://docs.vllm.ai/en/v0.29.0/features/quantization/b12x/)

Backend selection, FP4 activation choices and stated support boundaries.

[12] [FlashInfer release 0 6 18](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.18)

W4A16/GeGLU related work and version specific backend changes.

[13] [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

Compute capability 12.0 resource and occupancy constraints.

[14] [NVIDIA CUDA cooperative grid synchronization](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#grid-synchronization-cg)

Cooperative launch requirements and occupancy bounded grid synchronization.

[15] [Google Gemma 4 multi token prediction overview](https://ai.google.dev/gemma/docs/mtp/overview)

Compatible assistants and the batch one MoE expert bandwidth caveat.

[16] [NVIDIA programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

Opportunistic overlap and synchronization safety requirements.

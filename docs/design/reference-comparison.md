# What transfers from Mega MoE and Kimi K3

*Source comparison after the first [SM120 design draft](megakernel-design.md), 30 September 2026. Engineering conclusions are proposals, not RTX 5090 measurements.*

## Bottom line

Adopt explicit data lifetimes, bounded asynchronous staging and a dependency-aware execution schedule. Replace the hardware mechanisms. The 5090 has no TMEM; its design must use register accumulators, CTA-private shared-memory tiles and global inter-CTA state. Neither DeepSeek's SM100 MoE pipeline nor Inferact's TPU VMEM allocator is an implementation template for SM120.

Keep a graph-backed partial-fusion path as a first-class outcome. The question is whether wider visibility removes enough measured idle time to outweigh the 5090's register, shared-memory, barrier and code-size costs. Full-model ITL against tuned compatible FlashInfer decides; fused scope alone does not.

## Sources and evidence boundaries

The review used these immutable source revisions:

- DeepGEMM `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`: [Mega MoE API](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/mega_moe.hpp) and [SM100 FP8/FP4 implementation](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)
- Inferact `4048f0820aa4ff8787f707ca9d99b2bada9751aa`: [Kimi decoder](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/decode_megakernel.py) and [repository documentation](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/README.md)

Hardware support comes from [CUTLASS SM120 documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms), not inference from the word Blackwell. Model semantics remain governed by the Gemma references in the [design](megakernel-design.md#2-numerical-dag-before-scheduling) and [master](../master-plan.md#primary-references).

No benchmark was reproduced during this review. The [Inferact report](https://inferact.ai/blog/tpu-megakernels) describes fusion of 92 Kimi MoE layers across 16 TPU v7 chips/32 TensorCores, each with 64 MiB VMEM, and a speculative result above 700 emitted tokens/s at acceptance length six against 16 GB200 GPUs. Its DSpark verifier uses an anchor plus seven proposals. The [pinned K3 configuration](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/__init__.py#L16-L51) has 93 total layers including an initial dense layer; exact fused scope depends on entry path. Those model, hardware, parallelism and acceptance conditions do not predict Gemma batch-one latency on a 5090. The named draft is `RedHatAI/Kimi-K3-speculator.dspark`; the report does not establish that Inferact trained it.

## 1. DeepGEMM Mega MoE: keep the operation-level idea

### Direct-port blockers verified in source

The reviewed [API dispatch](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/mega_moe.hpp#L255-L277) selects the SM100 implementation. General DeepGEMM support for SM90 does not establish an SM90 Mega MoE backend. The implementation allocates through `cute::TMEM::Allocator2Sm` and uses the `tcgen05` family, which is not the SM120 execution contract.

The API also constrains [scaled widths to multiples of 128](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/mega_moe.hpp#L87-L92) and [requires SwiGLU](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/mega_moe.hpp#L175-L180). Gemma's routed intermediate width is 704 and activation is gated GELU tanh. The reviewed FP8×FP4 path is not the selected native NVFP4 W4A4 lane. Passing shape checks through padding or substituting an activation would not establish a correct Gemma backend.

### Scheduler mechanisms worth learning

The [Mega MoE scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) does more than retain CTAs:

- [Warmup waves/ring capacity](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L15-L79) constrain FC1/FC2 scheduling so consumers do not occupy all workers before needed producers are issued
- [Generation-parity readiness](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L141-L165) uses release/acquire communication; [TMA stores complete before publication](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1237-L1250)
- [Double-buffered task slots](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L454-L465) coordinate ownership and metadata-read completion before reuse

Adapt these invariants only after the bulk-synchronous SM120 baseline. A resident grid can still deadlock if every worker waits on FC2 while FC1 remains unissued. Per-K-block FC2 readiness is legal only when the Gemma activation/quantizer dependencies for that block are complete; any full-vector scale reduction postpones publication. Do not copy spin loops, parity lifetimes or two-CTA geometry without a new proof.

### Concrete substitutions

| Mechanism / idea | Decision | Proposed SM120 replacement and test |
| --- | --- | --- |
| Fuse dispatch, expert projections, activation and combine | Adopt the boundary as a candidate | T2 persistent feed-forward region, with actual Gemma branch norms and live top-8 routing; E03/E06 measure integration |
| SM100 TMEM accumulators and two-SM machinery | Replace | Thread-register FP32 accumulators; CTA-local operand/scale tiles; compiled resource and spill reports for every candidate |
| FP8×FP4 operand path | Replace | Native NVFP4 W4A4 quantizer/scale contract and its oracle; separate W4A16 control; E02 |
| SwiGLU epilogue | Reject for this model | GELU tanh, up multiplication, projection-specific FC2 quantization and reference casts |
| Width divisibility constraint | Adapt only with explicit accounting | Inspect 704-channel gate/up boundaries and FC2 K tails; validate zero pads/scales and report wasted work |
| Multi-GPU expert dispatch/communication | Defer | Single-GPU selected-ID table and bounded task range; no NVLink transport in WP0–WP4 |
| Persistent worker specialization | Experiment | Static resident CTA roles first; queue-based overlap only after independent progress review |

These replacements change utilization and resource balance. A different instruction family plus different activation and numerical operands is a new kernel design, not a flag change. Preserve upstream attribution and review licensing before reusing source.

## 2. Inferact K3: transfer lifetime discipline, not VMEM capacity

The Kimi source makes several mechanisms concrete:

- [Selected-expert staging](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/decode_megakernel.py#L1580-L1656) uses explicit slot bounds during routing
- [Projection/expert pools](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/decode_megakernel.py#L3811-L3838) share reusable storage under known lifetimes
- [Next-layer preparation](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/decode_megakernel.py#L4724-L4730) stages deterministic metadata/weights before use
- [Dense DMA slabs](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/decode_megakernel.py#L4811-L4881) use a bounded four-slot start/wait/recycle scheme
- [Decoder invocation](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/kimi/decode_megakernel.py#L5133-L5155) explicitly provisions its Pallas/VMEM setting

The transferable lesson is an auditable producer→consumer→last-use schedule. The particular buffer count, allocation size and execution topology are not transferable constants.

| K3 idea | Decision | Gemma/5090 adaptation |
| --- | --- | --- |
| Issue copies before their consumer phase | Adopt | Start selected FC1 tiles only after current route IDs; E05 counts producer cost and contention |
| Recycle a small asynchronous staging pool | Adopt | Per-CTA shared slots with completion and generation checks; derive slot count from compiled resources |
| Alias projection and expert temporary pools | Adapt | Global scratch aliases only after B11 or an equivalent proven last-use boundary; CTA shared aliases wait for all async users |
| Keep all activations in large managed VMEM | Replace | Retain short fragments in registers, current tiles in shared, cross-CTA live vectors in global memory |
| Next-layer weight lookahead | Adapt | Known QKV/norm/router addresses only; E09 compares hints or explicitly owned staged tiles; next routes are unavailable |
| One sequential grid-less decoder program | Replace | Occupancy-bounded cooperative CUDA grid, uniform phases/barriers and a progress proof; E06 compares region widths |
| TP4×EP8, topology-specific collectives | Defer | One GPU has neither those collectives nor their overlap opportunity |
| KDA/MLA state and equations | Reject for Gemma | Gemma sliding/global GQA, correct processed K/V, rotary and masks |

Absence of TMEM does **not** imply absence of TMA. SM120 CUTLASS offers TMA-oriented schedules. Use it where packed-byte layouts, descriptors and alignment are valid; tiny/gathered data may prefer vector loads or `cp.async`. L2 prefetch can be useful but cannot replace program-owned storage for correctness. Large speculative staging that steals occupancy may be worse than separate kernels with a fresh resource budget.

## 3. What the review changed

The first design draft already proposed staged partial/block persistence. Independent review then checked numerical dependencies, source compatibility and failure semantics before publication.

| Review item | Applied design change | Validation / rejection evidence |
| --- | --- | --- |
| Hidden producer/consumer dependencies | Separate attention merge from output projection; explicitly repeat small input norms/shared GELU inside each consuming CTA | Two-layer named-intermediate trace; repeated work included in latency |
| Asynchronous work at later barriers | Every phase drains its operand/output async work and required proxy fences before publishing or reusing storage | Sanitizers, delayed-producer stress and slot-generation checks |
| Divergent abort risk | Atomic first-error record; common phase control broadcast; idle/failed CTAs still traverse every barrier | Fault injection at each phase; no early-return strand |
| Retry after stochastic sampling | RNG counter and sampled result join tentative KV state; commit publishes all, failed pre-commit replay uses the same snapshot | Replay fixture checks RNG advancement and duplicate-emission avoidance |
| Model entry/exit ambiguity | Explicit BF16-cast embedding scale and unscaled tied output matrix after final norm | Embedding and logits oracle checkpoints |
| Memory-capacity documentation conflict | Record 128 KiB versus 100 KiB/SM documentation discrepancy; query actual caps and compiled occupancy | WP0 device record and per-candidate resource evidence |
| Literal implementation transfer risk | Make TMEM/VMEM substitutions and Mega MoE activation/width/format blockers explicit | No candidate promoted by analogy alone |

The full [execution design](megakernel-design.md) contains the B0–B11 barrier schedule, buffer ownership table, cache transaction, numerical oracle requirements and E01–E10 A/B matrix. This comparison does not waive any of those contracts.

## 4. Plain decode first, DSpark and draft training later

The roadmap retains a separate WP5 draft-training workstream after a stable ordinary decoder. An existing compatible Gemma assistant is a baseline to evaluate, not a reason to erase training from the plan. Training requires an approved data/provenance/licensing plan, architecture/loss and target-trace choices, budget, reproducible checkpoint, held-out acceptance/cost evaluation, then verifier/cache integration under I06 and G5. No training execution is authorized here.

A verifier with multiple proposal rows changes the workload. Its rows may route to different expert sets, increasing the union of weights read; the native single-token case remains eight distinct M=1 expert problems. Re-evaluate tile choices, memory, cache rollback and latency per **emitted** token. Measure draft, verify, acceptance, rejection and sampling together. The Kimi draft and its acceptance length cannot be imported as a Gemma performance assumption.

## Decision

Proceed with the smallest correct region that has measurable headroom beyond tuned FlashInfer plus graphs. Preserve T0 fallback and independently tuned attention/head paths. Adopt the scheduling discipline of both projects; adapt the storage and instruction mechanisms to SM120; reject transplanted semantics and hardware assumptions. A negative E06 result is useful: it says to keep the winning smaller boundary, not to force a whole-decoder megakernel.

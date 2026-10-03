# Gemma 4 single-token execution on SM120

*Design proposal, 30 September 2026. No target-GPU measurements or kernel implementation claimed.*

[Master contract](../master-plan.md) · [WP2](../action-plans/wp2.md) · [WP3](../action-plans/wp3.md) · [WP4](../action-plans/wp4.md) · [WP7 prefill](../action-plans/wp7.md)

## 1. Recommendation and scope

Build a model-specific, bounded single-token execution engine with interchangeable fusion regions. Begin with CUDA Graphs, retain the incumbent attention and output-head paths, and test a persistent post-attention feed-forward region. Expand to two decoder layers only after integrated gains survive. Thirty-layer persistence is an experiment, not the definition of success.

The target is one RTX 5090, one active text request, no dynamic request batching, and native expert NVFP4 W4A4. The proposed initial workload is 8K cached tokens plus separately reserved output capacity; resident fit remains unproved. W4A16 has a separate oracle and scorecard. The project succeeds on full-model, non-speculative end-to-end ITL against a tuned compatible FlashInfer-backed runtime, under the master contract's quality, memory and tail guards.

**The architectural constraint is absence of TMEM on SM120.** Accumulators live in registers, weight/scale tiles compete for CTA shared memory, and inter-CTA state lives in explicit global buffers. There is no SM100 `tcgen05`/TMEM schedule to transplant and no TPU-sized VMEM to hold a layer. Fusion must pay for its worst-phase register allocation, shared-memory reservation and barriers. A graph of smaller specialized kernels can win because each kernel reacquires its own resource budget. The [SM120 CUTLASS contract](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms) specifies `mma.sync` narrow-precision paths, TN operands and GeForce 1×1×1 clusters without multicast.

Documentation and the WP0–WP1 scaffold are authorized. This proposal does not authorize custom kernel execution or imply G0–G4 passed. The subsequent [Mega MoE/K3 source comparison](reference-comparison.md) records adopted ideas, SM120 substitutions and review changes.

The active milestone remains decode-first. The current bounded M1 preparation/lifecycle checks do not establish a speedup, broad G1 quality or complete-model graph admission; graph-backed execution below is a target architecture. The appended [full-prompt prefill design](#11-full-prompt-prefill-is-a-separate-optimization-regime) and WP7 action plan are proposals under D11; their draft PR publication is separately approved, with no GPU execution or merge approval.

## 2. Numerical DAG, before scheduling

### Dimensions and notation

The [Google configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json) and [NVIDIA configuration](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/main/config.json) agree on the following text geometry. Validate it again against the frozen checkpoint.

| Item | Contract |
| --- | --- |
| Hidden / layers | H=2816, 30 layers, no per-layer embedding input |
| Routed / shared MLP | 128 experts, selected K=8; expert F=704; shared F=2112 |
| Local attention | 25 layers; 16 Q heads, 8 KV heads, D=256, window=1024 |
| Global attention | Zero-based layers 5,11,17,23,29; 16 Q heads, 2 KV heads, D=512 |
| Output | Tied embedding/head, vocabulary=262144, final logit soft cap=30 |
| Norm / activation | epsilon=1e-6; learned multiplicative RMS weights; GELU tanh |

Use column vectors below. `N_w(v)=cast_BF16(w·v_fp32·(mean(v_fp32²)+epsilon)^(-1/2))`; `N_1` omits learned weight. Matrix products and pointwise operations retain the pinned reference's cast boundaries. These mathematical equations do not authorize reassociation across BF16 casts.

The initial hidden vector is `x0 = embedding[token] * cast_BF16(sqrt(H))`, retaining the reference cast-before-multiply order; the head uses the tied unscaled matrix after final RMS normalization. For input `x_l`, the required graph is:

```text
u      = N_input(x_l)
q      = RoPE_l,p(N_q(Wq u))
k_raw  = Wk u
v_raw  = Wv u                         [local]
       = k_raw                       [global projection sharing]
k_new  = RoPE_l,p(N_k(k_raw))
v_new  = N_1(v_raw)
a      = Wo Attention_l(q, K_history ∪ k_new, V_history ∪ v_new)
h      = x_l + N_post_attention(a)

s_in   = N_pre_ff(h)
s      = N_post_ff_1(Ws_down [GELU_tanh(Ws_gate s_in) · (Ws_up s_in)])

r_in   = N_1(h) · router_input_scale / sqrt(H)
P      = softmax_FP32(Wrouter r_in)
ids    = top8(P)
alpha_j= (P[ids_j] / sum_j P[ids_j]) · learned_expert_scale[ids_j]
e_in   = N_pre_ff_2(h)
e_j    = We_down[j] [GELU_tanh(We_gate[j] e_in) · (We_up[j] e_in)]
e_sum  = ReferenceReduce_j(alpha_j · e_j)
e      = N_post_ff_2(e_sum)
x_l+1  = layer_scalar_l · (h + N_post_ff(s + e))
```

The [Gemma implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py) makes `h` the fork point: routing does not consume the shared MLP output or its normalized input. `s` and `e` are normalized separately before the common normalization. Preserve `layer_scalar`, even if its inspected value is one. The reference casts weighted expert contributions back to the output dtype before accumulation; capture its actual expert visitation/reduction order. An FP32 unordered atomic combine is not automatically equivalent. Do not fold `alpha` into expert inputs across nonlinearities or normalization.

GELU tanh means `0.5 z (1+tanh(sqrt(2/pi)(z+0.044715 z³)))`, followed by multiplication by the up branch. It is not SwiGLU. The selected PyTorch activation and cast sequence, rather than an algebraic approximation alone, define fixture outputs. Norm reductions use FP32; replacing the reference's power operation with approximate reciprocal-square-root requires an explicitly bounded oracle comparison.

### Attention details that affect interfaces

For Q head h, KV head index is `floor(h/G)` with G=2 locally and G=8 globally. Attention forms `scores_j=q_h·k_j+mask_j`, applies FP32 softmax, casts probabilities to the value dtype, then reduces `sum_j probability_j·v_j` under the reference accumulation/cast contract. The score scale is one: do not insert `1/sqrt(D)` again. Q/K are normalized; V has unscaled RMS normalization. Global K and V share the raw projection but not the post-normalization/rotary values or cache storage.

Local rotary uses theta=10000 over the head. Global proportional rotary uses theta=1000000 and fraction 0.25. The [proportional RoPE implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_rope_utils.py) produces full-head frequencies, with 64 nonzero and 192 zero frequencies for D=512; the exponent denominator stays 512. With half-rotation pairing, active coordinate pairs are `(i,i+256)` for `i<64`, not a contiguous 128-element prefix. Reuse frozen reference-generated cos/sin fixtures before implementing an optimized rotary path.

At absolute input position `p`, text-only global visibility is `0 <= j <= p`; local visibility is `max(0,p-1023) <= j <= p`. Padding and prompt offsets are explicit. A ring index is an address, never the position used for RoPE. This workload has no cross-layer KV-sharing layers; global K/V projection sharing is a different mechanism.

## 3. Checkpoint and packing contract: I01/I02

### Verified metadata versus unresolved bytes

The NVIDIA [quantization metadata](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/main/hf_quant_config.json) identifies ModelOpt `0.43.0rc2.dev91+gc79ebc014`, NVFP4 and group size 16, excluding dense/shared MLPs, router, attention, vision and head. It also declares **FP8 KV quantization**. Therefore BF16 KV in D05 is an explicit runtime choice needing confirmation and matched-baseline validation; neither filename nor model dtype proves actual cache dtype. The main config describes static activation calibration metadata. It does not establish that all runtime block scales are constant or identical among experts.

The [file listing](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/tree/main) has two weight shards and a tensor index. Full safetensors headers/bytes have not been inspected in this design pass. Exported tensor names, scale broadcast shapes, nibble ordering, all padding, and global-scale sharing remain **import blockers**, not facts inferred from the BF16 module shape. Current [ModelOpt export code](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/export/unified_export_hf.py) is a guide to `weight`, `weight_scale`, `weight_scale_2`, and `input_scale`, not proof of an older checkpoint's representation. Inspect the producer revision as well as the pinned runtime loader.

### Proposed canonical representation

Each packed projection has logical `[expert, output, input]` coordinates. FC1 logically concatenates gate then up: `[128,1408,2816]`; FC2 is `[128,2816,704]`. Separate exported projections can be joined only after verifying activation/global-scale compatibility. Preserve different scales with segmented descriptors if necessary; silently taking a maximum changes quantization.

Represent decoded values canonically as:

```text
W_hat[e,n,k] = FP4_E2M1(q[e,n,k]) * SFw[e,n,floor(k/16)] * Gw[e,projection]
A_hat[k]     = FP4_E2M1(aq[k])    * SFa[floor(k/16)]     * Ga[quantizer]
```

The importer maps the checkpoint's multiply/divide conventions into `Gw/Ga`; it records inversions rather than guessing from suffixes. Weight q/SFw are preserved exactly. Activation `aq/SFa` come from the pinned oracle with its calibration scalar, live block reduction, clipping, E4M3 scale conversion, FP4 rounding, zero handling and saturation. If FC1 activation scales differ by expert or gate/up projection, quantize separately; eight experts sharing a real-valued input does not prove one quantized A buffer is legal. FC2 quantizes each expert's post-GELU product with that projection's contract.

For `C=A·B`, use A row-major `[M,K]` and B column-major `[K,N]`, logically the transpose of output-major weights. The [SM120 grouped example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu) provides a concrete 128×128×128 NVFP4 starting point, FP32 accumulation and 32-element input alignment. Its example output quantizer is not the Gemma epilogue. Use the pinned CUTLASS scale-layout helper; raw row-major scale arrays are not its swizzled SFA/SFB storage. Record helper type, revision, physical extents, strides and checksums.

The versioned manifest must carry source shard/name/hash, logical and padded extents, nibble-to-coordinate map, scale dtype/shape/index function, global-scale semantics, gate/up order, alignment, tied-storage identity and destination checksum. Padding is explicit zero-valued operand data with valid finite scale storage; tails cannot read absent scale blocks. At K=704, tile padding must not turn 704 into a new quantization domain. Round-trip every non-padding nibble and scale, then compare dequantized coordinates. Stream repacking by tensor/expert and release source copies after verification; budget coexistence at peak. No whole-model BF16 expert expansion. For text-only loading, explicitly list any omitted vision tensors and compare identical text behavior; never treat the full repository file size as the text-resident allocation.

## 4. Host preparation and bounded device step

**Preparation outside the token loop:** freeze checkpoint/tokenizer/runtime/compiler hashes; inspect device attributes; build/import I01 packs; preallocate cache/scratch/graphs; produce tensor-map descriptors where valid; compile candidates and query their actual resources; select a context bucket; run proven prefill and verify I03 handoff. Compilation, load and conversion time are reported separately, never confused with TTFT.

**Per request:** tokenize and apply the pinned chat template; reserve prompt plus output capacity; initialize request epoch, absolute positions and sampler RNG; prefill; establish the first pending decode input token. The scaffold's adapter is currently fail-closed and does not implement this execution path.

**Per token:** one stream-ordered submission evaluates that input token, yields next-token logits, samples, commits cache state and copies only status/token data to pinned host memory. A host event controls reading/streaming. No CPU round-trip for routes, no device malloc, and no indefinite GPU server loop. Do not launch the next token until the previous state transaction is committed. The sampled token enters KV only when it becomes the next step's input.

Proposed implementation interfaces extend, rather than silently replace, I01–I05:

| Interface | Required fields / outcome |
| --- | --- |
| `PreparedModel` | Manifest hash, lane, layer descriptors, packed weights, tensor maps, dispatch and resource records |
| `DecodeState` | Request epoch, committed length, input token/position, layer cache views, capacity, RNG state, tentative state |
| `RegionPlan` | Layer range, topology, compiled entry point, cooperative grid/block sizes, scratch layout, fixed phase schedule |
| `decode_step` | State + plan → status, next token/logits handle, elapsed GPU events, commit epoch |
| `CacheView` | K/V pointers, dtype/scales, physical strides, head geometry, page table/ring map, absolute origin and validity |
| `TraceSink` | Optional named DAG intermediates and generations; disabled in headline timing |

Reject incompatible descriptors before launch. A kernel capability record must name lane, activation, exact quantizer and shape support, not merely `supports_fp4=true`.

## 5. Execution topologies and the SM120 budget

| Topology | Boundary | Mechanism and cost | Promotion condition |
| --- | --- | --- | --- |
| T0 graph baseline | Library operators | Tuned independent resources; graph removes host launch gaps | Correct incumbent and fallback |
| T1 partial fusion | Norm/quantize; GELU/quantize; combine/norm/residual | Avoid intermediate round-trips while preserving specialized GEMMs | Full-model gain, same lane |
| T2 persistent FF region | `h` through `x_l+1` | Router/shared overlap, reusable metadata, bounded worker pool | G2/G3, resource and progress review |
| T3 persistent decoder region | Two layers, then larger ranges | Attention and projection tasks within one resident grid | Better than T1/T2 with attention costs included |
| T4 whole decoder | Embedding through final norm | Fewer region boundaries but worst-phase resource/code footprint | G4 evidence; head/sampling may remain external |

Ordinary host-launched FlashInfer/CUTLASS kernels cannot be called as library operations from inside a persistent CTA. T2 leaves attention as a graph node. T3 requires device-callable implementations of every fused operation, or must end the region at attention. Keeping the faster library attention can be the winning boundary.

Query compute capability, SM count, cooperative launch, register/shared-memory limits, opt-in shared memory and watchdog environment. Then inspect compiled registers/thread, static/dynamic shared memory, spills and occupancy. The [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) lists SM120 ceilings of 48 warps, 64K registers and 128 KiB shared memory per SM, with 99 KiB per block. The CUDA [13.1.1 programming-guide table](https://docs.nvidia.com/cuda/archive/13.1.1/pdf/CUDA_C_Programming_Guide.pdf) instead lists 100 KiB/SM for compute capability 12.0. Treat this documentation discrepancy as unresolved, query the host and use compiled occupancy; neither number is a budget assigned to this kernel.

| Phase | Provisional work partition | Live resources to account for |
| --- | --- | --- |
| Norm/router | One CTA or sharded 128-row router plus deterministic reduction | 2816-element vectors, FP32 norm/softmax partials, 8 IDs/weights |
| Dense QKV/shared/head | Output-channel tiles; independently tuned GEMV or library path | BF16 weight tiles, register sums, partial reductions |
| Expert FC1/FC2 | `(selected expert, N tile[, K split])` | FP4 A/B, SFA/SFB, FP32 accumulators, epilogue temporaries |
| Attention | `(Q head, KV partition)` with deterministic merge | Q registers, bounded K/V tiles, online softmax state |
| Combine/norm | Hidden-channel tiles plus small norm reduction | Per-expert outputs and cast-aware partials |

Native builder candidates begin at documented NVFP4 tiles 128×128×128 and 128×128×256, including cooperative versus ping-pong schedules; 256×128×128 is a utilization/resource control. Smaller custom MMA tiles are separate unvalidated designs. A 128×128×128 pair of FP4 tiles contains 16 KiB of operand payload per stage before scales, alignment, epilogue and barriers; double-buffering is not free. A genuine custom M=1 path has different requirements and must produce its own resource accounting.

Eight selected experts are **eight distinct M=1 problems**. Grouping supplies parallel jobs, not M=8 weight reuse. Padding an A tile to hardware M wastes rows; count executed versus useful work. Compare native tensor cores against unpack/GEMV and a separately labeled W4A16 path. Combined FC1 N=1408 is divisible by 128, but its gate/up boundary at 704 and FC2 K=704 still require explicit layout/tail choices. K splitting increases parallelism but adds partial storage, reduction and numerical-order changes; test it rather than assume it helps.

TMA is a candidate for aligned, descriptor-compatible packed weight/KV tiles with correct byte/stride mapping. Prebuild immutable per-expert descriptors where supported; select them only after IDs exist. Tiny activations, route metadata, awkward scale gathers and tails may use ordinary vector loads or `cp.async`. No multicast or distributed shared-memory assumption. Async copies must finish before consumption or shared-slot reuse; the pinned mainloop's barriers and proxy fences are part of correctness. A grid barrier alone is not a substitute for waiting on a TMA transaction.

## 6. A progress-safe persistent schedule: I04

The first prototype is bulk-synchronous, using `cudaLaunchCooperativeKernel` and `this_grid().sync()`. Bound grid CTAs by `SM_count × active_blocks_per_SM` returned for the **actual full kernel**, block size and dynamic shared memory. Check launch success. CUTLASS's within-CTA “cooperative” schedule is distinct from CUDA cooperative-grid launch. Follow the [CUDA grid synchronization contract](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#grid-synchronization-cg).

Every `run_*` task returns only after its async operand copies, output stores and required proxy fences complete; shared storage cannot be released earlier. Here `phase_barrier()` additionally drains CTA-local async work, executes a grid sync, lets CTA0 latch the atomic first-error status into a phase control word, and executes a second grid sync before any CTA reads that word. Every subsequent payload phase either runs or becomes a uniform no-op; every barrier still executes. Use an atomic first-failure record for multi-CTA error reporting. This conservative error-broadcast cost belongs in measurements. Removing a sync requires a new progress/memory-ordering proof.

Static assignment avoids a queue in version one. Each phase has a finite logical task range and a deterministic strided mapping onto the resident worker pool; small reduction phases can leave CTAs idle. CTA0 is coordinator only between phases and also computes. All CTAs execute all barrier sites in identical order. No branch, return or cancellation test may strand peers.

```text
host: validate descriptors and capacity
      occupancy-check the compiled region; launch cooperatively

all CTAs:
  B0: phase_barrier()                       # initialized request generation visible
  for layer in fixed_region:
    run_strided_tasks(PROJECTIONS_WITH_CTA_LOCAL_INPUT_NORM)
    wait_local_async_copies(); B1: phase_barrier()
    run_strided_tasks(QKV_NORM_ROPE_AND_TENTATIVE_STORE)
    B2: phase_barrier()
    run_strided_tasks(ATTENTION_PARTIALS)
    B3: phase_barrier()
    run_strided_tasks(ATTENTION_MERGE)
    B4: phase_barrier()                     # complete attention vector visible
    run_strided_tasks(OUTPUT_PROJECTION)
    B5: phase_barrier()
    coordinator computes post_attention norm/residual => h
    B6: phase_barrier()
    coordinator computes router, e_in and required FC1 input quantizers
    other CTAs compute shared FC1 with CTA-local shared-input norm
    B7: phase_barrier()                     # IDs/scales/input and shared gate/up ready
    run_disjoint_tasks(expert FC1, shared down with CTA-local GELU)
    B8: phase_barrier()
    run_strided_tasks(expert GELU_AND_ACTIVATION_QUANTIZATION)
    B9: phase_barrier()                     # whole FC2 inputs/scales available
    run_strided_tasks(expert FC2)
    B10: phase_barrier()
    coordinator performs cast-aware expert reduction, both branch norms,
      common norm, residual and layer scalar; writes next x
    B11: phase_barrier()                    # prior activations now reusable
  drain_async_copies(); uniform_status(); phase_barrier(); return
```

This conservative T3 schedule defines correctness, not an optimal implementation. Projection CTAs redundantly compute their full input normalization locally; shared-down CTAs compute the complete shared GELU vector locally before their output tiles. Neither depends on another CTA inside that phase. These small duplicated operations avoid an unstated barrier and must be timed. Version one has no split-K; introducing it requires a producer/merge barrier and new numerical review. `run_*` never hides inter-CTA dependencies. T2 starts at B6 with `h` supplied by the graph. T1 can independently fuse adjacent work within a CTA.

No CTA waits for a block outside the resident cooperative grid. Every phase terminates after bounded tasks even when some CTAs have no work. Reductions use explicit owned slots, not unbounded spinning. Registers never transfer ownership across CTAs; shared memory is CTA-private; a global write becomes available to other CTAs through the documented grid synchronization.

Later selective overlap may replace a barrier only with an independently reviewed protocol. Use device-scope release publication and acquire consumption; tags are `(request_epoch, token_epoch, layer, phase, slot_generation)`. Publish payload before ready, acknowledge consumption before reuse, and forbid counter wrap during a request. `volatile` is not this protocol. Bounded queues need capacity and a runnable-producer proof; resident CTAs alone do not prevent cyclic queue deadlock. In particular, FC2 consumers must not occupy every worker while their FC1 producers remain unissued. Per-block readiness must include GELU and every required quantizer scale reduction, not just an FC1 store. The [source comparison](reference-comparison.md#scheduler-mechanisms-worth-learning) motivates warmup/slot invariants without importing SM100 spin loops.

### Legal overlap and lookahead

After B6, routing, routed-input normalization and the shared branch independently consume `h`. Test a router-reserved CTA against serial routing; spending a whole CTA on 128 scores might lose. Once route IDs and required activation-scale identity are published, a designated producer can fetch selected FC1 tiles while the shared branch runs. In the first overlap experiment, workers never wait mid-phase: no route yet means no prefetch, then B7 guarantees readiness. Prefetch is optional for progress.

Stage only a bounded number of selected tiles, with producer/consumer and slot generation recorded. FC2 weights become identifiable with the same IDs, but their activation is not ready until GELU/quantization. Fetching all eight experts early can evict useful shared/attention data. Compare first-tile-only, FC1-only, FC1+FC2 and no prefetch.

At a layer tail, the next layer's norm/QKV, shared and router weight addresses are known; its hidden activation and expert IDs are not. Prefetching a few deterministic QKV tiles is legal lookahead, while predicting next-layer experts is a different speculative algorithm. L2 hints offer no residency guarantee. Explicit shared staging must stay in the same CTA's owned slot until that CTA consumes it, including across barriers, and counts against every phase's occupancy. Copying immutable weights into global scratch is an extra transfer, not a free cache.

## 7. Complete single-token timeline across two layers

Example: input token at absolute p=8192, zero-based layer 4 (local) followed by 5 (global). Layers 0–3 precede this region and 6–29 follow it; the timeline covers the same token throughout. There is no simultaneous evaluation of layer 5 activations before layer 4 produces them.

| Order | Work / publication | Dependency and release |
| --- | --- | --- |
| 0 | Host submits token step with committed length p and epoch t | I03 cache valid; output capacity includes p |
| 1 | Layers 0–3 produce `x4`; region entry B0 | Graph stream order or previous region event |
| 2 | L4 norm/QKV; Q/K norm and local RoPE; tentative K4/V4 | B1/B2 before attention readers |
| 3 | L4 attends p-1023…p, using committed history plus tentative p | B3 partials; B4 merge; B5 Wo |
| 4 | L4 attention residual makes `h4` | B6 publishes shared/router fork input |
| 5 | Router4 + routed norm alongside shared FC1 | Optional selected first-tile staging; B7 |
| 6 | Eight L4 expert FC1 jobs alongside shared GELU/down | B8; shared output remains live |
| 7 | Eight GELU products and FC2 quantizers, then eight FC2 jobs | B9 then B10; no FC2 reads partial quantization |
| 8 | Weighted reduction, branch norms, common norm, residual, scalar | B11 publishes `x5`; L4 scratch reclaimed |
| 9 | L5 norm/Q plus shared raw K/V projection | Distinct processed K5/V5; global proportional rotary |
| 10 | L5 attends 0…p; partials/merge/Wo/residual | Same B1–B6 pattern, larger KV-read domain |
| 11 | Router5/shared fork, expert work and reductions | Same B7–B11; IDs5 calculated live |
| 12 | Region produces `x6`; layers 6–29 and final RMS norm | Region boundary event or in-kernel barriers |
| 13 | Full vocabulary projection; `30*tanh(logits/30)`; sampler | Correct EOS, temperature/top-k/top-p and RNG policy |
| 14 | Commit tentative KV, length p+1, RNG and sampled result | Commit completion precedes next decode launch |
| 15 | Copy status/next token; host streams and checks cancellation | Sampled token enters cache next step, not now |

For stochastic sampling, keep the same transform order, RNG accounting and sampler as the incumbent initially. Do not approximate vocabulary top-k or omit the soft cap to save head time. Greedy tie policy is frozen too. A separately launched head/sampler still belongs in GPU and end-to-end token timing.

## 8. Buffer lifetimes, cache interoperability and failure

| Buffer | Placement / approximate payload | Producer → last consumer; reuse rule |
| --- | --- | --- |
| Packed model/scales | Global, immutable | Load → request lifetime; tied head/embedding has one owner |
| `x`, `h`, next `x` | Global BF16; 5.5 KiB each | Layer entry/residual → B11; ping-pong only after last read |
| Q, K/V current | Global or CTA tiles | Projection → attention/cache transaction; geometry-specific |
| Tentative all-layer K/V | Global BF16, 220 KiB total | Each layer → successful commit or discard |
| Router / selected metadata | Global, 128 FP32 scores plus IDs/weights | Router → expert combine; epoch-tagged |
| Shared gate/up | Global BF16, 8.25 KiB before padding | Shared FC1 → activation/down; retain branch output through B11 |
| Routed intermediates | Global; per expert 2×704 then 704 elements | FC1 → GELU/quantize → FC2; quantizer scales separate |
| Expert outputs | Global; 8×2816, 44 KiB BF16 or 88 KiB FP32 | FC2 → cast-aware combine; lane contract selects dtype |
| Attention/K-split partials | Global, shape-dependent | Producer tiles → deterministic merge; bounded by plan |
| Weight/scale pipeline | CTA shared, bounded staged tiles | Async producer → MMA/epilogue completion; acknowledge before reuse |
| MMA accumulators | Thread registers | Current tile → epilogue; no cross-CTA/TMEM storage |
| Logits / sample scratch | Global; FP32 vocabulary vector is 1 MiB | Head → sampler/trace; retained only when requested |
| Status/generations | Global + pinned host result | Step coordinator → host completion event; no premature host read |

These sizes exclude alignment, pads and allocator overhead. Aliasing requires a liveness proof; a low average footprint does not justify overlapping live values.

I03 first adopts the chosen incumbent's proven physical cache layout, including page size, NHD/HND strides and BF16 representation. A `CacheView` adapter either consumes it directly or performs a measured, validated prefill conversion. Merely calling a layout “paged” is insufficient. Local bounded storage may need a separate ring/page implementation if the incumbent retains all history. Test a continuous reference against prefill→conversion→decode over chunk, page, window and capacity boundaries.

Snapshot sampler RNG state/counter at step entry; keep the advanced counter and sampled result tentative. Failed pre-commit replay uses the same snapshot. Publish the next-token result and RNG advancement only with successful KV/epoch commit, so a retry neither consumes extra randomness nor emits a duplicate token.

Use a per-token transaction even without speculation: write new K/V to tentative per-layer slots, let the current step attend those slots explicitly, and leave committed lengths unchanged. At success, copy into global append positions and local ring destinations, then publish the new committed epoch/length. A local overwrite must occur only after successful evaluation; otherwise it can destroy the previous window needed for retry. A library path that requires in-place cache writes needs a proven undo/copy-on-write adapter or a full replay strategy. Include its cost in A/B timing.

Descriptor/capacity failures reject before launch. A detected arithmetic/metadata error sets a status flag; CTAs still traverse uniform barriers, skip future payload work uniformly, drain outstanding copies, and return without commit. Host cancellation is honored between bounded token steps: stop new submissions and release buffers only after the completion event. Mid-kernel cooperative cancellation is optional future work, not an early-return branch.

A failed step before commit can replay through the verified graph fallback from committed state. A failed or interrupted commit has uncertain state: poison that cache and rebuild from the committed token log; do not reuse partially overwritten rings. Illegal accesses, launch failures or watchdog resets require host-side CUDA error handling and potentially a fresh context. An in-kernel timeout cannot reliably rescue a deadlocked grid. Sanitize and stress the schedule before enabling it.

## 9. Analytical budget and falsifiable experiments

Arithmetic estimates, not measurements: all routed weights contain `30×128×3×2816×704` scalars. At 4 bits plus one byte per 16-value block, payload is about 10.635 GiB weights + 1.329 GiB scales, before global scales/padding. Per token, the eight selected experts across 30 layers represent approximately 0.665 GiB + 0.083 GiB respectively, without cache reuse. BF16 shared MLP weights add roughly 0.997 GiB, attention projections 2.068 GiB, and the tied output matrix 1.375 GiB of potential full-pass traffic. Actual transactions, cache hits, omitted copies and packing determine measured bytes.

Selected experts are only about 14% of the listed weight payload bytes under these assumptions, not 14% of measured latency. This implies a design risk: even eliminating all expert cost may miss G4 if dense projections/head or growing global attention dominate. If profiling assigns fraction f of token time to a removable region, its ideal maximum speedup is `1/(1-f)`; use that bound before widening fusion. The tied head saves allocation, not its vocabulary projection. Record stage latency and DRAM/L2 traffic; use attainable bandwidth for comparable accesses, not peak advertising bandwidth. Predict overlap gain only from independent critical-path work after resource contention is counted.

For BF16 cache capacity C and bounded local windows, logical bytes are `4×[25×8×256×min(C,1024)+5×2×512×C]`. At C=8192 this is 360 MiB; retaining all local history is 1.719 GiB. Output capacity, tentative storage, pages, graphs, runtime workspace and the proposed 2 GiB reserve are additional. The checkpoint's approximately 18.8 GB repository size is not a measured VRAM allocation or fit proof.

| Experiment ID | Paired A/B and recorded observation | Reject fusion / hypothesis if |
| --- | --- | --- |
| E01 graph_floor | Tuned incumbent eager versus CUDA Graphs | Claimed launch benefit disappears under graphs |
| E02 expert_shape | Native W4A4 MMA versus same-lane unpack/GEMV; W4A16 separate | Padding/quantization outweigh native throughput |
| E03 epilogue_scope | Split versus norm+quantize, GELU+quantize, combine tail | Integrated latency loses or numerical boundary shifts |
| E04 router_fork | Serial versus router/shared overlap | CTA reservation/contention raises critical-path time |
| E05 selected_stage | None, first tile, FC1, FC1+FC2 lookahead | Added traffic/L2 eviction/shared footprint erases gain |
| E06 region_width | T1, T2, two layers, six layers, all 30 | Spills, barrier wait, instruction pressure or idle workers dominate |
| E07 resource_budget | Same work with stage count/tile/worker-grid sweep | Best persistent occupancy cannot match split kernels |
| E08 cache_boundary | Library attention versus validated device region | Cache conversion/transaction or long-context attention regresses |
| E09 known_lookahead | Next-QKV first-tile hint/stage versus none | Contention costs more than reduced next-layer stall |
| E10 complete_token | Best candidate versus tuned FlashInfer runtime | Same-lane full-model ITL/quality/tail gates fail |

Profile separately from timing. Freeze random seeds, prompts, cache dtype, sampling and routes-for-fixtures; production timing routes live. Warm every graph/context bucket, randomize paired order, record at least the master's initial 30 trials per cell and continue if confidence intervals cannot decide. Report p50/p95 ITL, TTFT, complete response time, GPU events, power/thermal state and memory peaks. An unsupported FlashInfer path blocks the headline comparison; it is not a win.

## 10. Interfaces, fixtures and promotion decisions

| Evidence fixture / artifact | Exact acceptance question | Contract mapping |
| --- | --- | --- |
| Pack coordinate/tail fixtures | Nibbles/scales round-trip; interpreted tensor matches pinned loader | I01, R04/R05/R08; WP0–2, G0–2 |
| W4A4 quantizer/FC1/GELU/FC2 traces | Operand codes/scales match; error within pre-frozen oracle envelope | I02, R05; WP1/2, G1/2 |
| Router near-tie/scale fixtures | IDs, renormalized weights and learned scales agree; ties explicitly resolved | I02, R05; WP1–3, G1–3 |
| Attention positions and masks | Unit score scale, proportional rotary, distinct K/V, exact visible positions | I02/I03, R05/R09; WP1/3, G1/3 |
| Two-layer teacher-forced trace | First divergence localized at every named DAG node | I02/I04, R05; WP3/4, G3/4 |
| Epoch/queue/async stress | No stale reads, nonresident waits, bounds errors, leaks or unsafe cancellation | I04, R01/R06; WP2–4, G2–4 |
| Cache commit fault injection | Abort before commit replays; partial commit poisons/rebuilds; repeated wraps pass | I03/I04, R06/R09; WP3/4, G3/4 |
| Memory timeline and paired raw results | Resident fit; dispatched incumbent verified; full costs and confidence intervals | I05, R01–04/R07/R08; WP0/1/4, G0/1/4 |

BF16 establishes semantic/quality context; the simple W4A4 oracle establishes this lane's quantized operands and cast/reduction boundaries. W4A16 does not inherit W4A4 tolerances. Measure max/relative error, normalized RMSE, router mismatch, logit agreement and first teacher-forced divergence. NaNs, invalid IDs, stale generations and unexplained systematic route changes fail regardless of speed. Operator tolerance values remain unresolved until the oracle exists; do not invent numbers or relax them after tuning.

Proposed G4 thresholds remain those in the master: at least 15% median 8K end-to-end ITL reduction, 20% stretch, p95 no more than 5% worse, no unexplained >5% TTFT/response regression, and proposed held-out quality margins. Owner freeze is required. G3/T1 can be the final architecture if it meets the same project gate.

### Open decision records to resolve before implementation

These are local proposal labels, not replacements for D01–D08.

| Proposal | Proposed choice / evidence needed | Reopens |
| --- | --- | --- |
| MD01 | Graph + partial/FF persistence first; widen only with E06 evidence | D06, I04, G3/4 |
| MD02 | Preserve native exported weight bits; inspect headers and producer loader before choosing fused packs | D04, I01/I02, G0/1 |
| MD03 | BF16 KV as explicit override of checkpoint FP8 metadata; verify both matched paths | D05, I03, G0/1 |
| MD04 | Quantizer scale sharing, expert reduction ordering and cast points frozen by oracle | D06, I02, G1/2 |
| MD05 | Bulk-synchronous cooperative grid; async flags only after independent progress review | I04, G3/4 |
| MD06 | Tentative KV then commit; cache-native alternative must prove retry safety and count cost | I03/I04, G1/3 |
| MD07 | Output head and sampler stay external initially; include them in all headline timing | R03/R07, I05, G1/4 |

### Future DSpark-style acceleration and draft training

Ordinary target-only decode is the first deliverable. After G1 and an accepted stable G3/G4 path, WP5 opens a separate dependency-based workstream; it does not enter the plain-decode headline. Evaluate an existing compatible Gemma assistant first to establish a draft baseline, while retaining draft-model training as explicit planned work:

1. Freeze verifier/assistant interfaces, tokenizer, vocabulary, proposal positions and target numerical lane; define latency and acceptance-length objectives
2. Approve a training proposal: data provenance/licensing and permitted use, train/evaluation split, target-trace generation method, architecture and loss, compute/storage budget
3. Train or adapt the draft under separate execution authorization; retain reproducible configuration, seeds, dataset hashes and checkpoint/evaluation artifacts
4. Measure held-out acceptance, draft cost, verifier cost, selected-expert union and rejected work; reject a draft whose added traffic erases emitted-token gains
5. Integrate exact greedy/stochastic acceptance, residual correction and rollback under I06; stress EOS, window wrap and partial acceptance, then review G5

No dates, GPU-hours estimate, dataset choice or loss function is frozen. This document authorizes no training run. DSpark-style scheduling does not make a Kimi-specific assistant compatible with Gemma. Attribute any comparison to the actual draft named by the source; there is no evidence here that Inferact trained its own draft.

No tile, occupancy, timing, memory-fit or quality result in this document is established on the RTX 5090. Freeze immutable source revisions and attach build, oracle, sanitizer and measurement evidence before promoting any proposal. Preserve the master plan's [primary references](../master-plan.md#primary-references); external designs inform mechanisms, while SM120 resource and Gemma numerical contracts determine what can actually transfer.

## 11. Full-prompt prefill is a separate optimization regime

[WP7 A7.1–A7.6](../action-plans/wp7.md) profiles and tunes full 2K/8K prompts after G0/G1 qualification and a qualified decode control; 32K requires measured peak fit. This branch preserves the active WP2–WP4 decode chain. Its primary evidence is complete prompt-processing latency/throughput, peak memory, exact I03 handoff, TTFT and fixed-decode continuation guards at G7. It does not inherit G4 thresholds or convert short verifier speed into a prefill claim.

The single-token vectors and B0–B11 worker schedule above are not a full-prompt schedule. For C rows in one chunk, dense projections have multirow GEMM opportunities, and each routed expert has a variable row group. Same-expert rows may reuse its weights; different experts use different weights. Measure expert occupancy, skew, empty groups, tile tails, quantization/gather/scatter and load balance. Bound working activations by chunk and tile; do not retain an all-prompt all-expert intermediate or assume M1's workspace bounds scale to C rows.

| Candidate boundary | Evidence and unchanged contract |
| --- | --- |
| Chunk/context selection | Compare supported full versus chunked prefill; sum all chunk/scheduler costs and memory peaks; preserve every row's causal/sliding visibility before recycling local KV |
| SM120 attention | Tune local/global geometry independently; keep normalization, unit score scale, absolute local/proportional RoPE and distinct processed K/V; no SM100 TMEM schedule |
| Dense/shared projections | Tune actual multirow shapes and supported epilogues with the same dtype/casts, projection-sharing and output policy |
| Grouped expert FC1/FC2 | Tune real per-expert M, scale segments, GELU tanh, FC2 quantization and weighted reference reduction; count padded work and dispatch/packing |
| Residual hotspot fusion | Justify from the tuned prefill ledger; retain specialized GEMMs and bounded liveness/progress; operator and full-prompt same-lane evidence required |
| Cache handoff/fallback | Compare continuous reference and chunked prefill→decode across chunk, window, page and prompt/output capacity boundaries; count conversion and reject partial/stale state |

I01/I02 still govern original weights, scale/activation-quantizer lane and oracle boundaries. A faster arrangement may not silently merge expert calibration domains, change reductions or substitute W4A16. Use separate profiler and paired timing runs against tuned compatible FlashInfer in the same lane. Freeze useful prefill gain and tail/TTFT/memory/quality/decode guards before tuning for acceptance; unknown criteria or resource bounds block promotion.

Full-prompt prefill builds an I03 committed prompt cache. WP5's short DSpark verifier operates on tentative anchor/proposal positions with I06 accept/rollback. Share qualified attention/GEMM/epilogue code where contracts match, but benchmark full prompts and short verifier route unions separately. G7 does not imply G5 or authorize draft training. GPU experiments remain queued under the sole owner and require separate approval; [resource bounds](../action-plans/wp7.md#resource-and-ownership-bounds) are not a transfer of ownership.

# Shared tests and measurement protocol

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
Apply this protocol to every gate. A0.1 must freeze any proposed numerical thresholds before measurements become acceptance evidence. Save exact future invocations with each run rather than implying that these file names already execute.

### Test catalog

| Test ID | Required cases and recorded evidence |
| --- | --- |
| T01  Packing | Each supported layout/scale block, padding/tails, zero/extreme values, quantizer boundaries; round-trip and oracle outputs |
| T02  Semantics | Router near ties and top-8 weights; gated GELU and all branch norms; attention geometry/RoPE/masks; logits/soft cap |
| T03  Cache | Short input; 1023/1024/1025 positions; repeated window wraps; page/capacity boundaries; chunked prefill; global growth; handoff |
| T04  Progress | All supported resource/context buckets; idle CTA barrier participation; bounded queues/generations; repeated requests; EOS/cancellation; sanitizers |
| T05  Quality | Teacher-forced logits and first divergence; greedy regressions; held-out perplexity; task/long-context scores with paired uncertainty |
| T06  Speculation | Zero through all accepted, EOS/bonus token, ring/page edges, committed-state restoration and target-distribution toy tests |
| T07  Full-prompt prefill | 2K/8K full versus chunked traces; 32K conditional fit; partial chunks, per-expert skew/tails, causal/sliding visibility before window recycling, absolute RoPE, all-layer I03 handoff and matched decode continuation |

### Timing and fairness

Workload: one active request. Proposed probes are 2048 and 8192 cached prompt tokens; 32768 only if fit. Reserve output capacity separately. Start with at least 256 generated tokens per fixed prompt plus a longer stability case; apply identical EOS/length rules and retain shorter natural-EOS cases separately.

Boundaries: report cold load/compile separately, warm TTFT from request arrival to first delivered token, ITL between delivered tokens excluding the first, and total response time. Also record GPU-resident decode with GPU events. Include logits, sampling, cache updates and host streaming in the headline path; document exact timestamp/flush locations.

Trials: begin with at least 30 randomized paired request trials per workload cell, with fixed prompt set and matched seeds/settings. Keep per-token timestamps and per-request summaries; report request-level distributions and token tails. Use paired request-level bootstrap confidence intervals to avoid treating correlated tokens as independent samples. Continue or mark inconclusive if intervals cannot decide the frozen gate.

Environment: identical checkpoint hashes, tokenizer/template, numerical lane, KV dtype and sampler. Warm supported shapes/graphs; record clocks, power, temperatures, throttling and device memory. Keep the thermal/power policy fixed. Profile separately because instrumentation perturbs timing.

Quality and memory: compare kernel fidelity to its own quantized oracle and quality to the chosen quantized baseline, with BF16 comparisons where practical. Report peak allocated/reserved/device memory at every lifecycle phase, not only steady-state allocations.

### Full-prompt prefill measurements

[WP7/G7](../action-plans/wp7.md) uses the same fairness, paired trials and independent oracle discipline, with its own frozen acceptance criteria. Begin at 2K/8K; 32K is conditional on a measured peak-fit review including output capacity and the frozen reserve. Tune the supported incumbent's prefill chunking, attention, dense and grouped-expert paths before comparing a candidate. Graph and eager comparisons require matched eligibility; the current M1 path has no complete-model graph admission.

Record request arrival, first prompt-model submission, completion of every prompt chunk, native prompt-cache readiness, completion of any I03 handoff/conversion, first delivered token and every subsequent delivered token. Define the same prompt-processing boundary for both paths, including all chunks, routing, quantization/packing, synchronization and scheduler gaps. Report whole-prompt wall latency and actual input tokens divided by that latency, plus GPU-event duration separately. Report handoff cost and the combined prompt-through-decode-ready span. If head/sampling overlaps another phase, name the actual events rather than adding overlapping durations.

Warm TTFT remains request arrival through the first delivered token, including tokenization/input preparation, full prompt processing, required cache/format conversion, first-token logits/sampling and streaming. Keep cold load/compile separate. A single chunk's throughput, saved-route GEMM or isolated attention duration cannot replace the whole-prompt result. Last-row-only logits and full teacher-forced logits are distinct output policies; compare matching policies and label quality captures separately.

Save p50/p95 prompt latency, prompt tokens/s, TTFT, full response time, chunk schedule/shape, actual per-expert row counts, useful/padded work, dispatch/fallback reason and all lifecycle memory peaks. Profile separately from uninstrumented timing. Include activation/scales, grouping/dispatch scratch, local/global KV, conversion coexistence, output reserve and graph pools only when admitted. Do not retain all prompt rows for all experts or expand all weights to BF16 to claim a resident fit.

Run the same accepted decode implementation after incumbent and candidate prefill. Check cache/logit agreement, absolute positions and committed lengths; report p50/p95 ITL, quality, output policy, memory and long-run stability under frozen decode regression guards. Hold decode fixed for this attribution; label joint changes separately. WP5 short verifier rows require their own accepted/emitted-token, route-union and I06 commit/rollback measurements even if kernels are shared. No draft training is authorized by G7.

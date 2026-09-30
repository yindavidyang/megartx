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

### Timing and fairness

Workload: one active request. Proposed probes are 2048 and 8192 cached prompt tokens; 32768 only if fit. Reserve output capacity separately. Start with at least 256 generated tokens per fixed prompt plus a longer stability case; apply identical EOS/length rules and retain shorter natural-EOS cases separately.

Boundaries: report cold load/compile separately, warm TTFT from request arrival to first delivered token, ITL between delivered tokens excluding the first, and total response time. Also record GPU-resident decode with GPU events. Include logits, sampling, cache updates and host streaming in the headline path; document exact timestamp/flush locations.

Trials: begin with at least 30 randomized paired request trials per workload cell, with fixed prompt set and matched seeds/settings. Keep per-token timestamps and per-request summaries; report request-level distributions and token tails. Use paired request-level bootstrap confidence intervals to avoid treating correlated tokens as independent samples. Continue or mark inconclusive if intervals cannot decide the frozen gate.

Environment: identical checkpoint hashes, tokenizer/template, numerical lane, KV dtype and sampler. Warm supported shapes/graphs; record clocks, power, temperatures, throttling and device memory. Keep the thermal/power policy fixed. Profile separately because instrumentation perturbs timing.

Quality and memory: compare kernel fidelity to its own quantized oracle and quality to the chosen quantized baseline, with BF16 comparisons where practical. Publish peak allocated/reserved/device memory at every lifecycle phase, not only steady-state allocations.

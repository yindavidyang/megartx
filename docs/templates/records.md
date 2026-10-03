# Decision and result templates

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
### Decision ledger entry

File: docs/decision_ledger.md. Preserve existing IDs and consult the ledger for current status: D04 checkpoint and D05 workload are selected for execution; D06 lanes and D07 margins remain proposed; D08–D10 record separately bounded historical scope/evidence. D11 adds WP7 planning and separately approved draft PR publication, without GPU execution or merge approval. Append new IDs without renumbering or promoting proposals to accepted results.

| Field | Entry to complete |
| --- | --- |
| Identity | Decision ID; date; WP/action/gate; R and I references; responsible role |
| Question and options | What must be decided; candidate paths; approved constraints |
| Evidence | Source/config/checkpoint hashes; test and raw result paths; uncertainty |
| Decision | Proceed / retain / rerun / change scope / stop; rationale; approver |
| Consequences | New dependencies; affected contracts; rollback; revisit trigger |

### Benchmark result record I05

Proposed file: `results/<wp>/<run_id>.json` plus raw token_events.jsonl and requests.csv. Keep large traces/weights outside source control with checksums and authorized retrieval locations.

Identity: run ID, task/gate/hypothesis, timestamps, git/build/environment/checkpoint hashes, command, numerical lane, host/backend and actual dispatched kernels. Record unsupported/fallback status explicitly.

Workload: prompt/tokenizer/template hashes, initial/final context, output/EOS policy, seed/sampler, KV dtype/cache mode, batch/concurrency, graph bucket, repetitions and paired baseline IDs.

Measurements: raw request and token timestamps, exact timing boundaries, GPU time, TTFT, request and token ITL summaries, total response time, confidence intervals, peak memory by phase, power/clocks/temperature, errors and sanitizer links. For speculation add proposed/accepted/emitted counts, verifier calls and all draft/cache costs.

For WP7 add actual prompt length, chunk schedule/partial tails, prompt-processing start/completion, whole-prompt latency and input tokens/s, native-cache and decode-ready handoff events, peak activations/scales/dispatch/scratch/KV, per-expert row histogram and fallback disposition. Record conditional 32K fit and the fixed decode implementation's continuation regression control. Keep verifier costs separate. Record the owner-frozen G7 criteria and resource admission; missing evidence stays missing, with no invented performance thresholds or dates.

Conclusion: correctness/quality gate, relative gain with uncertainty, tail tradeoffs, total costs omitted (if any), decision, reviewer and reproduction notes. Any excluded critical-path work invalidates a headline full-model claim.

### Pinned source intake

The master plan contains the full evidence register and architecture rationale. These official starting points must become immutable revision links in A0.2/A0.3; their current examples are not proof of compatibility or speed on this host.

[NVIDIA Gemma checkpoint and quantization model card](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)

[Transformers Gemma reference semantics](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py)

[CUTLASS SM120 functionality and supported paths](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)

[FlashInfer releases and version-specific changes](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.18)

[NVIDIA cooperative grid synchronization requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#grid-synchronization-cg)

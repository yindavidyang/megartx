# WP1 Build the oracle and semantic fixtures

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
### A1.1 Implement separate numerical references

WP1  R04 R05 R08  I01 I02  |  Owner: Model and correctness lead  |  Entry: I01/I02 frozen; reference host available

Inputs: Pinned authoritative model, tensor manifest and accepted numerical lanes.

1. Keep three named references: authoritative BF16 for semantics/quality, simple quantized oracle for kernel fidelity, and tuned compatible FlashInfer host for performance. Record when BF16 full-model evaluation needs a separately authorized resource.

2. Build native W4A4 oracle using the exact packed weights, scales, activation quantizer, rounding and reduction contract. Build W4A16 separately if evaluated; never treat its activation arithmetic as native FP4.

3. Capture deterministic intermediate fixtures from norms, routing, each selected expert, shared branch, attention, residuals and logits. Use teacher forcing to localize the first divergence. Freeze tolerances from the reference rounding envelope before tuning.

Deliver: reference/quant_oracle.py; reference/lane_registry.json; tests/fixtures/oracle_manifest.json; configs/tolerances.yaml.

Exit and review: Reviewer reproduces oracle outputs, approves tolerance rationale, and verifies no NaNs/Infs or hidden precision changes.

If blocked: Oracle work may continue during a FlashInfer compatibility gap; G1 and speed claims remain blocked. Never loosen a tolerance solely to pass a candidate.

### A1.2 Lock routing attention and cache tests

WP1  R05 R09  I02 I03  |  Owner: Model and correctness lead  |  Entry: A1.1 oracle available

Inputs: Frozen semantics, deterministic tensors, tokenizer and cache adapter.

1. Test router no-scale normalization, learned input scale and width factor, FP32 softmax, top-8 selection, renormalization and learned expert scales. Save IDs/weights for ordinary, near-tie, zero and extreme inputs; establish tie policy.

2. Test gated GELU with tanh approximation, separate shared/routed normalizations, branch reduction, common post-normalization and residual order. Compare intermediate values before validating final logits and soft cap.

3. Test Q/K/V normalization, per-layer geometry, rotary dimensions/positions, causal and local masks. Cover 1023/1024/1025, repeated ring wraps, cache/page capacity edges, chunked prefill and handoff. Shared global projection must not alias processed K/V incorrectly.

Deliver: tests/test_router.py; tests/test_layers.py; tests/test_kv.py; tests/fixtures/golden_manifest.json.

Exit and review: Every fixture compares the same numerical lane with saved expected outputs. Reviewer records max/relative error, normalized RMSE, cosine similarity and routing mismatches.

If blocked: Bisect to the first tensor or cache-state error. Autoregressive text that looks plausible cannot replace these tests.

## WP1 Tune the incumbent and attribute latency

### A1.3 Establish the tuned FlashInfer incumbent

WP1  R03 R05 R08 R09  |  Owner: Benchmark owner  |  Entry: A1.1/A1.2 pass; exact compatible FlashInfer path

Inputs: Frozen workload, environment lock, golden fixtures and supported backend matrix.

1. Run the selected host with pinned FlashInfer and capture actual attention/MoE dispatch, including every non-FlashInfer component. Verify native FP4 coverage and reject silent emulation or changed Gemma semantics.

2. Tune documented supported choices for concurrency one: attention/MoE backend, CUDA Graph capture/buckets, prefill and memory reservation. Include eligible default, CUTLASS and b12x controls; log unsupported combinations rather than timing them as failures.

3. Use the [shared paired protocol](../protocols/measurement.md). Hold model bytes, prompt formatting, KV dtype, sampler, output policy and non-target components fixed. Keep a quality-labeled llama.cpp control separate if useful; it does not replace the primary incumbent.

Deliver: configs/baseline.yaml; results/wp1/tuning.csv; results/wp1/dispatch.json; docs/baseline.md.

Exit and review: Reviewer can reproduce the fastest correct compatible configuration and its tuning search. Native W4A4 and W4A16 results remain separate.

If blocked: If compatibility cannot be established, mark G1 blocked and document the exact gap. Experimental microbenchmarks may inform research but cannot substantiate an end-to-end win.

### A1.4 Select a measurable recoverable bottleneck

WP1  R03 R07 R08  I05  |  Owner: Benchmark owner  |  Entry: A1.3 incumbent frozen

Inputs: Uninstrumented baseline records and separate profiler runs.

1. Build per-token time/byte ledger for routed experts, shared/dense work, attention/projections, final vocabulary head, scales, KV, sampling, launches, waits and host streaming. Measure real DRAM traffic, spills, occupancy and cache behavior.

2. Compare stage time with attainable bandwidth/compute on comparable access patterns. Include scattered selected-expert reads. Bound benefit with stage fractions and sequential dependencies; avoid double-counting overlap or treating peak GPU bandwidth as attainable.

3. Choose the first hotspot and smallest experiment that can resolve it. Record expected mechanism, measurements, effort range after this discovery, stop condition and the incumbent revision to recheck at the next gate.

Deliver: results/wp1/token_ledger.csv; results/wp1/traces/manifest.json; docs/gates/G1.md; docs/hypotheses/H001.md.

Exit and review: G1: golden semantics, oracle, tuned compatible incumbent and a recoverable full-model budget justify WP2.

If blocked: If expert time cannot support a useful result, propose a measured scope change or retain the incumbent. Do not escalate to persistence solely because it is the original idea.

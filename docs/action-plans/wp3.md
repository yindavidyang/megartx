# WP3 Earn a decoder fusion boundary

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
Package entry: G2 passed. Preserve I02 numerical semantics, I03 KV handoff and the graph fallback throughout. Each task produces evidence for the next; none commits to whole-model persistence.

### A3.1 Draw dependencies and live ranges

WP3  R05 R06 R07  I04  |  Owner: Kernel engineer  |  Entry: G2 passed

Inputs: Gemma reference, integrated traces and resource ledger.

1. Draw attention/residual/router/shared/expert dependencies and every buffer live range. Verify when router and shared branch inputs are ready before testing overlap.

2. List candidate fusion boundaries and bytes/launches removed, new scratch, barriers and occupancy cost. Stage selected expert tiles only after routing; do not assume future-layer expert IDs.

Deliver: docs/wp3_dag.md; results/wp3/liveness.csv.

Exit and review: Reviewer approves dependency legality and a measurable hypothesis.

If blocked: Park overlap that lacks independent inputs or a feasible memory/resource budget.

### A3.2 Implement the smallest partial fusion

WP3  R03 R05 R09  |  Owner: Kernel engineer  |  Entry: A3.1 reviewed

Inputs: Ranked boundaries and golden operator fixtures.

1. Test one boundary at a time: norm/quantization, projection epilogue, gated GELU preparation, weighted reduction or residual/norm tail. Keep attention/output-head library paths when faster.

2. Compare split kernels, CUDA Graphs and each fused boundary using the same I02 contract; attribute benefit before adding another change.

Deliver: kernels/sm120/fusion_registry.json; results/wp3/partial_ablation.csv.

Exit and review: Reviewer retains correct integrated gains, not launch-count reductions alone.

If blocked: Revert slower boundaries and preserve the simplest supported winner.

### A3.3 Test block scheduling and handoff

WP3  R05 R06 R09  I03 I04  |  Owner: Runtime engineer  |  Entry: A3.2 winning boundary

Inputs: I03 cache fixtures and reviewed resource map.

1. Try bounded block persistence or early-router/shared-branch overlap only where the DAG permits it. Measure resource contention and selected-weight staging effectiveness.

2. Run the [shared cache/stress catalog](../protocols/measurement.md), prefill handoff, long decode and fallback replay. Check register pressure, spills and progress with the actual compiled resources.

Deliver: results/wp3/block_stress.json; runtime/fusion_dispatch.json.

Exit and review: Reviewer verifies cache interoperability and safe bounded progress.

If blocked: Disable an unsafe or tail-heavy schedule; retain graph execution.

### A3.4 Review the full model and G3

WP3  R03 R05 R07 R08  |  Owner: Benchmark owner  |  Entry: A3.3 passes

Inputs: Paired raw results, quality evidence and updated token ledger.

1. Run full-model regressions and compare end-to-end ITL, TTFT, tails, memory and quality. Recheck the pinned incumbent and accepted guardrails.

2. Record proceed to WP4, retain partial fusion, or stop. Estimate remaining effort from measured unresolved costs.

Deliver: docs/gates/G3.md; results/wp3/full_model.jsonl.

Exit and review: G3 requires integrated gain, safe progress and no unexplained p95 regression.

If blocked: An uncertain gain remains inconclusive; do not label a block microbenchmark a project win.

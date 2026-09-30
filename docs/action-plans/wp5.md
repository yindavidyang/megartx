# WP5 Evaluate compatible speculation only if useful

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
Optional entry: WP1 complete and a stable accepted WP3 or WP4 path. One request may contain several verifier positions; that is not dynamic request batching.

### A5.1 Establish draft compatibility and budget

WP5  R07 R10  I06  |  Owner: Model and correctness lead  |  Entry: Stable target path accepted

Inputs: Exact target tokenizer, compatible draft support and free-memory measurements.

1. Pin a verified Gemma-compatible draft/assistant, license, revision and host support. Check tokenization, anchor semantics, logits and context/cache requirements.

2. Measure draft load/latency, VRAM, route union and target verifier cost. Compare supported proposal lengths 1/2/4/7; do not copy DSpark acceptance assumptions.

Deliver: configs/draft.yaml; results/wp5/cost_pilot.csv.

Exit and review: Reviewer finds plausible net wall-clock benefit with safe peak fit.

If blocked: If no compatible affordable draft exists, keep speculation disabled.

### A5.2 Implement target exact acceptance

WP5  R05 R10  I06  |  Owner: Model and correctness lead  |  Entry: A5.1 compatible draft

Inputs: Target/draft probability semantics and sampling configuration.

1. For greedy decode, require agreement with target-only greedy behavior. For sampling, implement the required acceptance and residual correction procedure after the exact temperature/truncation/logit processing.

2. Test tractable probability distributions, zero-probability edges and fixed-seed cases; matching independently sampled tokens alone is insufficient.

Deliver: reference/spec_acceptance.py; tests/test_spec_acceptance.py.

Exit and review: Reviewer approves target-distribution correctness, then model tests.

If blocked: Any distribution or greedy mismatch blocks speculation regardless of speed.

### A5.3 Validate transactional cache commit

WP5  R05 R09 R10  I03 I06  |  Owner: Runtime engineer  |  Entry: A5.2 correct acceptance

Inputs: Committed/tentative state design and window/cache fixtures.

1. Separate committed sequence length from tentative verifier positions. Use staging, copy-on-write or a proven rollback buffer so rejected tokens cannot overwrite needed committed ring entries.

2. Test every acceptance length from zero through all, bonus/anchor rules, EOS inside proposals, page boundaries, ring wraps and cancellation. Replay after rollback against target-only state.

Deliver: runtime/spec_cache_contract.json; tests/test_spec_rollback.py.

Exit and review: Reviewer verifies byte/state visibility and future logits after commit/rollback.

If blocked: Stale suffix visibility or damaged ring state blocks G5; retain target-only execution.

### A5.4 Decide on accepted token cost

WP5  R03 R07 R08 R10  |  Owner: Benchmark owner  |  Entry: A5.3 passes

Inputs: Same-draft FlashInfer baseline and complete timed pipeline.

1. Include draft, verification, sampling, cache transaction and host streaming. Report proposed/accepted/emitted tokens, acceptance-length distribution, verifier calls, expert union, p95 and memory.

2. Compare equally enabled FlashInfer with the same draft/settings, plus both non-speculative controls. Apply [shared paired protocol](../protocols/measurement.md) across useful workloads.

Deliver: results/wp5/paired.jsonl; docs/gates/G5.md.

Exit and review: G5 requires lower total wall-clock time per emitted token with exact target semantics.

If blocked: Disable by workload when acceptance, traffic, tails or memory erase the gain.

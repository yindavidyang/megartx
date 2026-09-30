# WP4 Test bounded cross layer persistence

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
Entry: G3 passed with a measured cross-layer opportunity. Use one bounded invocation per token; partial fusion remains valid.

### A4.1 Prove a resident synchronization design

WP4  R01 R06 R07  I04  |  Owner: Kernel engineer  |  Entry: G3 plus justified hypothesis

Inputs: Device limits, actual compiled kernel and dynamic shared-memory use.

1. Check cooperativeLaunch support and compute the resident grid limit using occupancy for the compiled kernel. Review every phase/barrier, including idle CTAs.

2. Prove required producers are resident; forbid spins on possibly nonresident producers. Define release/acquire queue publication and generation-tagged completion. PDL overlap cannot be required for progress.

Deliver: docs/wp4_progress_proof.md; results/wp4/occupancy.csv.

Exit and review: Independent reviewer signs the progress and memory-ordering argument.

If blocked: If residency or uniform-barrier participation cannot be proved, use split graph regions.

### A4.2 Implement a bounded decode schedule

WP4  R05 R06 R09  I04  |  Owner: Runtime engineer  |  Entry: A4.1 approved

Inputs: Winning blocks, global state layout and fallback runner.

1. Expand from two blocks to larger regions, then all 30 layers only while gains persist. Declare output-head/sampling boundaries; time excluded work too.

2. Preallocate queues, cache metadata and scratch; bound tasks per token and validate every buffer lifetime. Keep prefill separate and preserve the I03 handoff.

Deliver: runtime/persistent_schedule.json; results/wp4/region_ablation.csv.

Exit and review: Reviewer verifies numerical parity and resource feasibility at each expansion.

If blocked: If spills, code size, barriers or occupancy erase benefit, stop expansion and retain the best region.

### A4.3 Stress progress cancellation and recovery

WP4  R05 R06 R09  |  Owner: Runtime engineer  |  Entry: A4.2 candidate correct

Inputs: [Shared stress catalog](../protocols/measurement.md), sanitizer tools and graph fallback.

1. Test every supported context bucket, window wrap and output cap, repeated requests, EOS and cancellation at token boundaries. Check no stale task flags, leaks or cache contamination.

2. Run the agreed soak workload under recorded thermal conditions; verify bounded completion and fallback recovery. Save failing seeds and first divergent state. Profile separately.

Deliver: results/wp4/stress.jsonl; results/wp4/sanitizers.txt.

Exit and review: Reviewer approves reproducible stability evidence and bounded kernel lifetime.

If blocked: Any deadlock, memory-safety error or wrong recovery blocks release; select graph fallback.

### A4.4 Make the single GPU release decision

WP4  R03 R05 R07 R08  |  Owner: Project owner and reviewer  |  Entry: A4.3 or accepted G3 earlier stop

Inputs: Paired full-model results, quality, memory and maintenance assessment.

1. Apply accepted G4 margins to the best supported path versus tuned compatible FlashInfer. Show native/non-speculative results, every control and uncertainty.

2. Choose persistent, partial fusion, upstream integration or the incumbent. Archive raw evidence, exact reproduction, limitations and rebaseline triggers.

Deliver: docs/gates/G4.md; docs/release_decision.md.

Exit and review: G4 requires the accepted latency, tail, quality, memory and stability margins.

If blocked: If margins are unresolved, record inconclusive and the specific evidence needed.

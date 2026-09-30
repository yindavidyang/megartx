# WP5 Plan DSpark style speculation and draft training

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
Planned later entry: begin without DSpark or draft training; complete WP1 and establish a stable accepted WP3 or WP4 path first. One request may contain several verifier positions; that is not dynamic request batching.

### A5.1 Establish draft compatibility and budget

WP5  R07 R10  I06  |  Owner: Model and correctness lead  |  Entry: Stable target path accepted

Inputs: Exact target tokenizer, compatible draft support and free-memory measurements.

1. Evaluate existing Gemma-compatible drafts/assistants first. Pin the selected license, revision and host support; check tokenization, anchor semantics, logits and context/cache requirements. If no adequate draft is available, prepare the conditional training milestones below rather than removing draft training from the roadmap.

2. Measure draft load/latency, VRAM, route union and target verifier cost. Compare supported proposal lengths 1/2/4/7; do not copy DSpark acceptance assumptions.

Deliver: configs/draft.yaml; results/wp5/cost_pilot.csv.

Exit and review: Reviewer finds plausible net wall-clock benefit with safe peak fit.

If blocked: Keep speculation disabled on the initial path. Record the compatibility or cost gap and prepare the separately approved draft-training path; do not begin training or spend compute without approval.

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

## DSpark and draft training milestones

The initial system is non-speculative. DSpark-style acceleration and a trained
Gemma-compatible draft are planned later work, with dependency gates instead of
invented dates. Existing draft evaluation comes first; custom training is a
conditional planned branch when compatibility, acceptance or total cost warrants
it. The Kimi draft/checkpoint and reported acceptance rates are not transferable
assumptions. This roadmap does not assert that Inferact trained its own draft.

| Milestone | Entry and work | Deliverable and exit evidence |
| --- | --- | --- |
| M5.1 Existing draft feasibility | Stable target-only baseline; audit supported Gemma assistants and DSpark-style design compatibility | Revision/license/tokenizer contract, proposal-length pilot, draft/verify/VRAM budget, explicit reuse-versus-train decision |
| M5.2 Training data pipeline | Training branch proposed; approve specific datasets and any data/compute acquisition first | Data rights and provenance, preprocessing/tokenization, target-generation or distillation plan, held-out split and leakage checks, hashed manifests |
| M5.3 Draft training pipeline | Architecture/objective selected for exact target/verifier semantics; resource budget and execution separately approved | Reproducible recipe, configuration/seeds/software pins, bounded pilot followed by go/no-go review; no unapproved full training run |
| M5.4 Checkpoint and evaluation pipeline | Candidate trained or adapted draft, with tokenizer/config/weights frozen | Versioned checkpoint manifests and hashes; held-out acceptance-length/quality evaluation; draft latency, memory and total emitted-token cost versus existing drafts and target-only decode |
| M5.5 Verifier and cache integration | Compatible affordable draft accepted; A5.2/A5.3 correctness reviewed | Target-exact acceptance, rejected-suffix isolation, transactional cache tests, same-draft FlashInfer control and complete paired G5 evidence |

A5.1 owns M5.1–M5.4 intake and the choice to reuse or train; A5.2–A5.4 own M5.5
and the G5 decision. Dataset/recipe/checkpoint paths are future deliverables,
not implemented tools. Failure to produce a compatible profitable draft leaves
the correct target-only runtime enabled and the next research decision explicit.
Training cost and inference benefit are reported separately; acceptance alone
cannot demonstrate a wall-clock win.

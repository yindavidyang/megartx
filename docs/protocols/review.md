# First sprint and review discipline

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
The first sprint is an evidence-producing sequence, not a calendar commitment. Its stopping condition is G1, or a documented blocker that prevents an honest compatible baseline. No custom kernel implementation belongs ahead of the correctness and baseline contract.

| Order | Minimum backlog and handoff | Done evidence |
| --- | --- | --- |
| 1 | A0.1  Owner approves scope, workload, lanes and gates | Decision record and frozen configuration |
| 2 | A0.2  Runtime pins host/dependencies and support matrix | Immutable manifest and documented smoke invocation |
| 3 | A0.3  Model lead audits tensors and numerical semantics | Complete I01/I02 contracts and packing fixtures |
| 4 | A0.4  Runtime proves peak fit and cache handoff | Memory timeline, smoke outputs and G0 decision |
| 5 | A1.1 and A1.2  Model lead builds oracle and fixtures | Numerical references, frozen tolerances and golden tests |
| 6 | A1.3  Benchmark owner tunes compatible incumbent | Dispatch record, supported tuning search and baseline |
| 7 | A1.4  Benchmark owner profiles and selects first hotspot | Token ledger, hypothesis and G1 decision |

### Dependency and parallel work rules

A0.2 metadata collection and A0.3 checkpoint inspection may overlap after the selected revisions are known. After G0, oracle implementation and tuning setup may overlap, but A1.3 results are not accepted before A1.1/A1.2 pass. Independent reference work may continue during a FlashInfer gap; G1 and headline claims remain blocked. Later packages consume gate records; WP5/WP6 are optional.

### Estimate only after discovery

For each task, record measured setup time, build/timing cycle duration, known remaining variants, review/test effort and external blockers. Produce a range with confidence and assumptions after A1.4. Re-estimate when compiler support, memory fit or integration changes the work. Leave unknowns unestimated rather than assigning invented dates.

### Gate review packet

Every G0–G6 packet contains the contract/revision hashes, exact invocations, runnable configuration, numerical/test reports, raw measurements, complete cost boundaries, memory/resource reports, profiler manifest and an explicit decision. The reviewer checks the independent reference, baseline fairness and uncertainty before accepting a performance claim.

Decision choices: proceed; keep the current boundary; rerun with a specified missing measurement; change scope with owner approval; stop and retain the incumbent. Record why the simplest supported path wins. A decision to stop wider persistence is valid when the evidence supports it.

### Rebaseline triggers

Recheck upstream at major gates and when the host runtime, FlashInfer, CUDA/driver, checkpoint, quantizer, cache policy, compiler flags, hardware settings or sampling contract changes. Keep old evidence tied to its old manifest; do not merge unlike numerical lanes or environments into one headline.

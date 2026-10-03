# RTX 5090 Gemma 4 Action Plans

*Execution playbook for the master plan  |  Contract 1.0  |  30 September 2026*

Start with A0.1 through A1.4. Establish the exact model, runtime fit, numerical oracle and tuned FlashInfer baseline before selecting custom kernels. Expand fusion only when measured full-model latency and correctness justify it.

Decode-first remains the active milestone. [WP7 full-prompt prefill](wp7.md) appends A7.1–A7.6 and G7 without changing any existing package, gate or task ID. Its GPU branch opens only after baseline qualification and separate scope/resource approval; CPU planning can proceed independently. Full prefill and WP5 short verifier workloads share kernels only after separate qualification and benchmarks.

The [conditional first SM120 kernel plan](../first-sm120-kernel-plan.md) preserves
the historical CPU planning audit and bounded metadata-fusion candidate. Its
4.03% exploratory kernel-sum budget and original missing ABI/graph evidence are
not accepted performance evidence. Subsequent bounded M1 implementation and
lifecycle checks cover only their named sources/workloads; they establish no
speedup or broader numerical, quality or graph gate pass.

The first target is the owned RTX 5090 with SM120, one active request and no dynamic request batching. The research hypothesis is model-specific scheduling across layers and operator boundaries. A whole-model megakernel is an experiment; the deliverable may be a smaller winning set of fused regions.

Authority: the repository scaffold and bounded target-host/M1 work have separate historical scope records in the [decision ledger](../decision_ledger.md). The owner separately approved this documentation revision's isolated commit/push and public draft PR on 3 October 2026. Merge, SSH, GPU experiments, package installation and runtime changes remain excluded. PR15 benchmark remediation remains separate, and `01a10079-c8d1-752e-991a-4e76b523704e` remains the sole GPU owner. Later execution needs its own scope/access decisions. Proposed defaults await freeze, and no hardware/performance gate has passed. Task deliverable paths below are planned outputs, not a claim that each tool or result already exists.

### Execution map

| Package and tasks | Dependency and review decision | Plan |
| --- | --- | --- |
| WP0  A0.1 to A0.4 | Authorize and freeze contract; verify compatibility and peak fit at G0 | [WP0](wp0.md) |
| WP1  A1.1 to A1.4 | G0 → semantics, oracle, tuned incumbent and attributable budget at G1 | [WP1](wp1.md) |
| WP2  A2.1 to A2.4 | G1 → low M kernels and integrated MoE win at G2 | [WP2](wp2.md) |
| WP3  A3.1 to A3.4 | G2 → smallest useful decoder fusion boundary at G3 | [WP3](wp3.md) |
| WP4  A4.1 to A4.4 | G3 → bounded persistence or evidence-based earlier stop at G4 | [WP4](wp4.md) |
| WP5  A5.1 to A5.4 | Planned later DSpark/draft evaluation and conditional training after WP1 and a stable WP3 or WP4 path; G5 | [WP5](wp5.md) |
| WP6  A6.1 to A6.4 | Optional after actual future hardware and authorization; G6 | [WP6](wp6.md) |
| WP7  A7.1 to A7.6 | CPU intake now; G0/G1 plus qualified decode control and separate GPU scope → full-prompt prefill gain with cache/memory/decode guards at G7 | [WP7](wp7.md) |

The playbook contains the original 28 tasks plus six prefill tasks, 34 total. See the [shared tests and measurement protocol](../protocols/measurement.md), [first sprint and review discipline](../protocols/review.md), and [decision/result templates](../templates/records.md). R01–R13 and G0–G7 refer to the [master plan](../master-plan.md). Interfaces I01–I06 name the handoffs that each task must preserve.

Roles are responsibilities, not assigned people. One engineer may hold several roles; an independent reviewer checks gates. The project owner approves scope, workload and acceptance margins. Estimate effort after evidence-producing tasks reveal the work; no dates or hardware access are assumed.

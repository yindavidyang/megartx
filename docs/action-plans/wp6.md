# WP6 Assess the future four GPU system

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
Optional entry: a real four-GPU system, explicit authorization and an accepted single-GPU baseline. Four RTX PRO 6000 Blackwell GPUs are a future aspiration, not available capacity.

### A6.1 Inventory actual board and host topology

WP6  R01 R11 R12  |  Owner: Runtime engineer  |  Entry: Future hardware present and authorized

Inputs: Exact board SKUs, host inventory and vendor specifications.

1. Record per-board memory/power mode, PCIe generation and negotiated width, CPU root complexes, NUMA placement, peer access and software versions.

2. Document IOMMU/ACS and transport constraints without changing security-sensitive settings. Verify physical/driver connectivity rather than assuming NVLink or uniform peers.

Deliver: configs/future_host.lock.json; docs/topology.md.

Exit and review: Reviewer validates topology with actual device evidence.

If blocked: Without hardware/access, leave WP6 unopened; marketing specifications do not establish latency.

### A6.2 Measure the communication floor

WP6  R07 R08 R11  |  Owner: Benchmark owner  |  Entry: A6.1 complete

Inputs: Authorized benchmark tools and actual model message sizes.

1. Measure peer copies, small-message all-reduce/broadcast and expert-dispatch paths for each relevant pair and group. Include synchronization and host staging where P2P is unavailable.

2. Run latency/bandwidth distributions at model-relevant sizes and concurrency. Save NCCL transport, affinity, topology and initialization details alongside raw results.

Deliver: results/wp6/collectives.csv; results/wp6/p2p.csv.

Exit and review: Reviewer establishes a measured communication budget before partition selection.

If blocked: If the latency floor exceeds recoverable compute time, keep TP1 for latency.

### A6.3 Compare supported partition choices

WP6  R05 R07 R11  |  Owner: Kernel and runtime engineers  |  Entry: A6.2 measured budget

Inputs: Model dimensions, runtime support and single-GPU reference.

1. Model TP1, TP2/TP4, layer pipeline and expert placement only where supported. Account for each sequential collective and remote expert critical path, replicated weights/KV and imbalanced routes.

2. Test the cheapest plausible partitions with identical numerical semantics. Measure memory capacity separately from token latency; larger aggregate VRAM alone is not a speedup.

Deliver: configs/parallelism_matrix.yaml; results/wp6/partitions.jsonl.

Exit and review: Reviewer retains only correct layouts with a plausible end-to-end benefit.

If blocked: If the model/backend lacks a supported partition, record the gap rather than force incompatible sharding.

### A6.4 Make the multi GPU decision

WP6  R03 R08 R11  |  Owner: Project owner and reviewer  |  Entry: A6.3 candidate results

Inputs: Paired TP1 and multi-GPU measurements, memory and quality records.

1. Run full prefill/decode/streaming comparisons with communication included. Separate larger-context feasibility from speed at the same context.

2. Choose TP1, a supported multi-GPU layout, or a capacity-only use case. Record reliability, operational cost and maintenance tradeoffs.

Deliver: docs/gates/G6.md; docs/future_system_decision.md.

Exit and review: G6 requires actual net benefit versus one GPU or an explicit decision to keep TP1.

If blocked: No 4x scaling or NVLink-based performance claim without measured evidence.

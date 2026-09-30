# megartx

Experimental single-request inference research for the owned NVIDIA RTX 5090 (SM120), initially targeting Gemma 4 26B A4B with an NVFP4 checkpoint candidate.

The goal is lower end-to-end decode latency than a tuned, compatible FlashInfer-backed runtime, while preserving model behavior. There are no performance results yet. The initial WP0–WP1 scaffold provides CPU-only contracts and offline tooling. It does not contain a model runtime, a numerical oracle, or custom GPU kernels.

## Start here

- [Master engineering plan](docs/master-plan.md): requirements, interfaces, evidence gates and primary sources
- [SM120 execution design](docs/design/megakernel-design.md): numerical DAG, layouts, barriers, lifetimes and experiments
- [Mega MoE and Kimi K3 comparison](docs/design/reference-comparison.md): source-backed lessons and SM120 adaptations
- [Action plans](docs/action-plans/README.md): 28 concrete tasks across WP0–WP6
- [Decision ledger](docs/decision_ledger.md): confirmed scope and proposed defaults
- [Measurement protocol](docs/protocols/measurement.md): fairness, correctness catalog and paired timing
- [Review protocol](docs/protocols/review.md): first sprint, gate packets and rebaseline triggers
- [Decision and result templates](docs/templates/records.md)

## Scope and status

- First target: one RTX 5090, one active request, no dynamic request batching
- Initial work: WP0 compatibility and memory fit; WP1 numerical references and tuned FlashInfer baseline
- Candidate checkpoint revision, text-only 8K primary workload, numerical lane and 15% latency gate remain proposed until explicitly frozen
- GPU compatibility, actual dispatch, resident fit, model correctness and benchmark performance remain unverified
- Start without DSpark; [DSpark-style speculation and conditional draft-model training](docs/action-plans/wp5.md#dspark-and-draft-training-milestones) are planned after the stable target-only baseline
- Custom kernels and wider fusion are gated follow-on experiments
- Four RTX PRO 6000 Blackwell GPUs are future work; no topology or scaling benefit is assumed

Current authorization covers Markdown initialization, the draft WP0–WP1 scaffold, design/comparison documents and the later DSpark/draft-training roadmap. Actual training and GPU experiments are not authorized by that planning scope. Read the [decision ledger](docs/decision_ledger.md) before extending scope. No checkpoint weights, private host records or large traces belong in source control.

## Try the CPU scaffold

Requires Python 3.10 or newer. From the repository root, no model download, CUDA import or runtime dependency installation is needed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m megartx validate configs
PYTHONPATH=src python -m megartx readiness configs/experiment.json
PYTHONPATH=src python -m megartx inventory
PYTHONPATH=src python -m megartx memory --tensors configs/tensor_manifest.json --budget configs/memory_budget.json
```

The supplied manifests are proposed, unverified or incomplete. Validation checks structure and cross-field constraints; it does not pass a GPU gate. Readiness lists the missing freeze decisions. The memory command reports unknown components and leaves measured fit pending. The inventory command prints only allowlisted metadata; an optional `--probe-nvidia-smi` performs a read-only GPU metadata query and tolerates missing GPU tooling.

See the [scaffold guide](docs/scaffold.md) for schema export, result statistics, the future FlashInfer adapter boundary, test coverage and limitations. Every correctness fixture is a pending contract, not a passing Gemma model test. A real adapter must match pinned runtime/checkpoint/lane evidence and actual dispatched providers; the included adapter always refuses inference.

## Licensing

A repository license has not yet been selected. Review upstream licenses and model terms before reusing code or downloading weights.

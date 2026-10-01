# megartx

Experimental single-request inference research for the owned NVIDIA RTX 5090 (SM120), initially targeting Gemma 4 26B A4B with an NVFP4 checkpoint candidate.

The goal is lower end-to-end decode latency than a tuned, compatible FlashInfer-backed runtime, while preserving model behavior. The WP0–WP1 scaffold provides CPU contracts and offline tooling. An opt-in, version-pinned scale adapter and independent NumPy reference now address six unequal gate/up globals. Target-GPU measurements remain exploratory until the numerical and quality gates are complete; no performance improvement is claimed.

## Start here

- [Original-scale NVFP4 correction](docs/nvfp4-scale-correction.md): isolated adapter, numerical lanes, reproducible checks and remaining qualification
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
- Selected execution artifact: NVIDIA NVFP4 revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`; BF16 KV, 2K/8K probes and 256-token output reserve
- Performance/quality acceptance margins remain proposed
- Resident execution and stock native SM120 FlashInfer CUTLASS dispatch were observed; discarded up globals prevented correctness qualification
- Separate original-scale projections and six forced registered-runner fixtures have bounded independent evidence; the natural client corpus selected none of the affected experts
- One unchanged 1,025-token-prefix/eight-output score replay explains nonselection for that prefix; independent sampled router math is consistent, with native precision limits recorded
- Five fixed-route GPU requests now match independent projection replay and paired full-model captures within each path; [controlled live evidence](docs/controlled-scale-integration.md) retains the matched wrong-alpha control
- Full-versus-cached divergence begins before the scale correction at layer 0 position 32; natural coverage and full quality/cache qualification remain blocked, and timed clients fail closed
- Start without DSpark; [DSpark-style speculation and conditional draft-model training](docs/action-plans/wp5.md#dspark-and-draft-training-milestones) are planned after the stable target-only baseline
- Custom kernels and wider fusion are gated follow-on experiments
- Four RTX PRO 6000 Blackwell GPUs are future work; no topology or scaling benefit is assumed

Current authorization covers isolated target-host setup, the selected model download, compatibility/baseline tests, investigation and correction of six gate/up scale mismatches, numerical checks, and a draft PR with sanitized evidence. It does not include training, merge or deployment. Read the [decision ledger](docs/decision_ledger.md) before extending scope. No checkpoint weights, private host records or large traces belong in source control.

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

See the [scaffold guide](docs/scaffold.md) for schema export, result statistics, the future FlashInfer adapter boundary, test coverage and limitations. The original contract adapter still refuses inference. The separate experimental plugin is inactive unless explicitly enabled and requires the pinned eager, single-GPU BF16 lane. The pending fixture manifests and small reference tests do not by themselves pass a Gemma quality gate. See the [reference checks](numerical_reference/README.md) and [scale adapter](docs/nvfp4-scale-correction.md).

## Licensing

A repository license has not yet been selected. Review upstream licenses and model terms before reusing code or downloading weights.

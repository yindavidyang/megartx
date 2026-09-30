# megartx

Experimental single-request inference research for the owned NVIDIA RTX 5090 (SM120), initially targeting Gemma 4 26B A4B with an NVFP4 checkpoint candidate.

The goal is lower end-to-end decode latency than a tuned, compatible FlashInfer-backed runtime, while preserving model behavior. There are no performance results yet. This repository begins with an engineering plan; the WP0–WP1 compatibility and benchmark scaffold is being prepared in a draft PR.

## Start here

- [Master engineering plan](docs/master-plan.md): requirements, interfaces, evidence gates and primary sources
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
- Custom kernels, wider fusion and speculative decoding are gated follow-on experiments
- Four RTX PRO 6000 Blackwell GPUs are future work; no topology or scaling benefit is assumed

Only the Markdown initialization and draft WP0–WP1 scaffold are currently authorized. Read the [decision ledger](docs/decision_ledger.md) before extending scope. No checkpoint weights, private host records or large traces belong in source control.

## Licensing

A repository license has not yet been selected. Review upstream licenses and model terms before reusing code or downloading weights.

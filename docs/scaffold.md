# CPU-only WP0/WP1 scaffold

This foundation implements contracts and CPU utilities for A0.1–A1.4. It does
not implement Gemma, an NVFP4 quantizer/oracle, packing kernels, prefill/decode,
CUDA kernels, or a FlashInfer runtime. No model correctness, GPU fit, runtime
compatibility, or performance gate has passed. Unit tests use synthetic data
solely to test validation, byte arithmetic, statistics and failure boundaries.

The target remains one RTX 5090 (SM120), one active request, and a tuned,
compatible FlashInfer-backed end-to-end incumbent. The checkpoint repository,
text-only workload, 8K primary context, 256-token output reserve, BF16 KV,
W4A4 lane, 15% median ITL reduction and all other numerical margins are proposed
defaults. The exact immutable checkpoint revision and owner freeze remain pending.

## Run without installation or GPU dependencies

From the repository root, Python 3.10 or newer is sufficient:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m megartx validate configs
PYTHONPATH=src python -m megartx readiness configs/experiment.json
PYTHONPATH=src python -m megartx inventory
PYTHONPATH=src python -m megartx memory --tensors configs/tensor_manifest.json --budget configs/memory_budget.json
python -m compileall -q src tests
```

`validate` returning zero means the document satisfies structural and supported
cross-field rules. It does not freeze choices or validate evidence. `readiness`
reports unresolved freeze decisions and keeps G0/G1 pending even for a frozen
experiment. Frozen experiments require immutable revision/hash fields, an owner
approval reference, a hashed prompt corpus, and explicit sampling/EOS policies.
The initial sampler supports a greedy contract only; new sampler contracts need
explicit implementation and review before use.

An optional editable install provides the `megartx` entry point:

```sh
python -m pip install -e .
```

This packaging command can install a build dependency; it is unnecessary for the
CPU checks above. No PyTorch, CUDA, FlashInfer, model weights, or test dependencies
are installed by the scaffold. CI runs only stdlib checks on Python 3.10 and 3.12.

## Files and action boundaries

| Action | Implemented foundation | Still required before gate evidence |
|---|---|---|
| A0.1 | `configs/experiment.json`, freeze validation | Owner decisions, exact hashes and explicit freeze |
| A0.2 | Redacted inventory, environment/compatibility contracts | Exact-host revisions, documented smoke commands and support evidence |
| A0.3 | Physical tensor inventory schema, alias-aware byte accounting | Checkpoint audit, 30-layer/dimension reconciliation, packed format and numerical contracts |
| A0.4 | Explicit KV/output reserve and conservative memory budget | Measured load/repack/cache/graph/prefill/decode peaks and handoff tests |
| A1.1–A1.2 | Pending fixture requirements with immutable evidence/tolerance fields | Separate BF16, W4A4, optional W4A16 oracles and actual golden outputs |
| A1.3 | Dispatch evidence packet boundary and always-unavailable adapter | Version-selected host integration, true dispatch capture and supported tuning sweep |
| A1.4 | Paired result format, deterministic request-level bootstrap | Real token ledger/profiler attribution and independent G1 review |

The root `schemas/*.schema.json` files are exported Draft 2020-12 schemas.
`src/megartx/schema.py` is the canonical definition. Regenerate exports with
`PYTHONPATH=src python -m megartx schemas --directory schemas`; tests detect drift.
The local validator implements only the schema subset used here. Additional
cross-field checks are in `contracts.py`; external JSON Schema validation alone
does not apply them. `environment.lock.json` is deliberately unverified and is
not an installed package lock. A `pinned` status requires all core versions,
revisions, package-lock hash and dirty-source state; optional b12x may be null.

## Inventory privacy and failure behavior

Default inventory does not execute a subprocess or import GPU libraries. It
reports OS family, architecture, Python and allowlisted distribution versions,
and whether `nvcc`/`nvidia-smi` are on PATH. It does not print executable paths,
hostname, user names, GPU serials/UUIDs, environment variables, or raw tool errors.

On an authorized GPU host, an optional metadata-only query is available:

```sh
PYTHONPATH=src python -m megartx inventory --probe-nvidia-smi
```

The query requests only GPU name, driver version, total/free memory and compute
capability. Missing tools, unsupported queries, timeout and malformed output
produce an explicit unavailable status. This is neither a CUDA execution test nor
proof of the exact Gemma/NVFP4/SM120 path. Nothing changes host settings.

## Offline memory accounting

No network access or checkpoint download occurs. Supply audited physical tensor
metadata in a copy of `tensor_manifest.json`; scales and non-quantized tensors
must be separate entries. `logical_shape` documents semantics; `storage_shape`
and `storage_dtype` determine bytes. Scalars use `[]`. `packed_fp4` counts padded
physical nibbles and rounds the final byte upward. If the actual packed payload
is uint8, use its physical uint8 shape. This arithmetic is not a pack/unpack test.
Explicit `float8_e4m3fn` and `float8_e5m2` storage types count eight bits per
element and retain the declared scale dtype; selecting a dtype does not infer a
checkpoint's format. See the official [NVIDIA NVFP4 description](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
and [PyTorch tensor dtype definitions](https://docs.pytorch.org/docs/stable/tensor_attributes).

Identical whole-storage aliases share a `storage_id` and are counted once;
conflicting aliases are rejected. Partial storage views/offsets are unsupported
and must be audited into complete storage records before use. Retained original
copies and repacked copies belong in the explicit additional components.

KV accounting uses distinct K/V dimensions and dtypes per layer and reserves
cached context plus output capacity. A sliding attention mask does not imply
bounded allocation: only an explicitly verified `bounded_window` storage policy
uses the window limit. Physical page rounding, allocator fragmentation, padding,
staging and cache metadata must be accounted for in explicit buffers; none is
silently assumed to be zero. Duplicate layer IDs are rejected.

With empty proposed manifests, reported known bytes are zero and unknown coverage
remains explicit; that is not a zero-byte model estimate. Estimated capacity
comparison is available only when tensor/budget coverage, geometry, capacity and
all additional byte components are supplied. `measured_fit` always remains
pending. Sum of declared peak components is a conservative budget, not a measured
lifecycle timeline; separate GPU phase measurements and the proposed 2 GiB
reserve remain required. Never drop tensors or offload weights to manufacture fit.

## Correctness and adapter contracts

`fixture_contract.json` names packing, routing, normalization/branch ordering,
attention/RoPE/cache and teacher-forcing cases. Every case is pending. Changing
one to `captured` requires input/output hashes, immutable oracle revision, explicit
tolerances and their approval rationale. Captured does not mean passed. There
are no saved golden tensors, invented tolerance envelopes or model-pass tests.
Keep BF16 semantics/quality, W4A4 kernel fidelity and any W4A16 lane separate.

`check_dispatch_packet` checks a reviewed packet against the intended checkpoint,
environment, lane and explicit attention/MoE providers, plus a contained local
trace file's SHA-256. At least one intended component must be FlashInfer, and
non-FlashInfer components must be declared. Native W4A4 MoE must be marked by the
observed packet, regardless of its provider. Package presence or backend flags
cannot satisfy this check. Unsupported/fallback paths are not performance wins.
This function verifies packet consistency and integrity, not the factual truth
of a trace; independent review and actual dispatch collection are still required.
`FlashInferAdapter.run` always raises `BackendUnavailable`, even with a valid
packet receipt. No backend implementation is implied.

## Result records and statistical scope

Copy `configs/result_template.json` to an ignored local results directory. Keep
real raw token timestamps, config/environment/checkpoint/code hashes and exact
invocations with each run. The template is not summarized as measured data.
`provenance` distinguishes templates, synthetic data and measured records. A
measured label requires core hashes and raw-event provenance, but still does not
prove correctness or benchmark fairness. No real measurements are included.

Each pair is one workload cell with a matched prompt hash/seed, same output
length/policy, shared numerical lane and experiment manifest, and a recorded
order. Randomize paired request order in the eventual runner. Presence of both
order labels alone is not proof of randomization. Natural-EOS cases belong in
separate workload cells. `*_itl_ms` contains intervals between delivered tokens,
excluding the first token. Total response must equal TTFT plus those intervals.
Cold load/compile and instrumented profiling belong in separate records.

```sh
PYTHONPATH=src python -m megartx summarize results/wp1/paired.json --bootstrap-samples 2000 --seed 0
```

This command requires a supplied record with at least two pairs; the proposed
minimum is 30. It supports only delivered-token end-to-end records and rejects
GPU-resident/microbenchmark boundaries. The estimand is one minus the ratio of
the candidate/baseline medians of per-request median ITL. Paired percentile
bootstrap resamples whole matched requests, never correlated individual tokens.
Quantiles use linear interpolation (type 7). Reports retain provenance, timing
boundary, omitted critical-path costs and prerequisite statuses, and include
request-level medians/p95, pooled token p95, TTFT and total-response medians.

The median ITL bootstrap does not establish tail, TTFT, total-response, quality or
memory uncertainty. All gate decisions remain `not_evaluated`, including for a
synthetic 20% improvement. Any omitted critical-path costs invalidate a headline
full-model claim. Acceptance still needs frozen margins, correct oracles, tuned
compatible dispatch, complete costs and independent uncertainty/gate review.

Keep local inventories and benchmark outputs in the ignored `artifacts/` or
`results/` directories. Common weight, trace and environment-secret filenames
are also ignored; review every staged file rather than relying on filename
patterns to protect private data. No license choice, remote-writing utility,
CUDA execution or GPU result is part of this scaffold.

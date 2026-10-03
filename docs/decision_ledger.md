# Decision ledger

Updated 3 October 2026 for the prefill planning revision and separately approved draft publication. Stable IDs follow the [master plan](master-plan.md). Proposed values are not frozen acceptance criteria.

| ID | Status | Decision and remaining evidence |
| --- | --- | --- |
| D01 | Confirmed | Begin with the owned RTX 5090, SM120; four RTX PRO 6000 GPUs are future scope |
| D02 | Confirmed | One user and one active request; no dynamic request batching |
| D03 | Confirmed | Aim to outperform a tuned, compatible FlashInfer-backed runtime on the RTX 5090 |
| D04 | Selected for execution | NVIDIA Gemma-4-26B-A4B-NVFP4 revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`; immutable weights/tokenizer/config audited, expert-only W4A4 |
| D05 | Selected for execution | Text-only 2048/8192 prompt tokens, 256 output capacity, greedy nonspeculative requests, BF16 KV; cache/quality acceptance still pending |
| D06 | Proposed | Native expert W4A4 candidate; W4A16 remains a distinct numerical lane with a separate oracle |
| D07 | Proposed | 15% median 8K ITL improvement (20% stretch), p95/TTFT guards, quality margins and 2 GiB initial reserve; owner freeze and evidence required |
| D08 | Authorized | Isolated target-host setup, compatible stack/model download and compatibility/baseline tests; investigate/fix six gate/up scale mismatches, implement a minimal scale-preserving adapter, independent numerical checks, rerun baseline when qualified, and publish a sanitized draft PR. Training, wider custom megakernels, merge and deployment remain outside scope |
| D09 | Observed, separately qualified | Pinned FlashInfer uses layerwide activation calibration maxima, distinct from the checkpoint per-expert quantizer. Original-scale weight preservation does not establish full quantizer/quality equivalence; deterministic eager correction is a separate lane |
| D10 | Authorized diagnostic; bounded live evidence | Five fixed 33-input GPU passes at all 30 layers establish observed original-weight projection and within-path paired stage/logit/KV agreement, with a matched full-path wrong-alpha control. Full-versus-cached divergence starts upstream of the scale correction; expansion stops at that boundary. Controlled coverage cannot satisfy natural quality or performance gates. See [controlled integration evidence and next experiment](controlled-scale-integration.md) |
| D11 | Planning and draft publication authorized; execution gated | Append [WP7/G7 and A7.1–A7.6](action-plans/wp7.md) for full-prompt prefill: 2K/8K profiles, conditional 32K fit, chunking/SM120 attention, dense/grouped-expert GEMMs and measured hotspot fusion. Preserve decode-first, exact lane/oracle/cache contracts and separate WP5 verifier evidence. Separate owner approval relayed on 3 October 2026 covers an isolated documentation commit/push and public draft PR; merge and SSH/GPU/package/runtime changes remain excluded. PR15 remediation stays separate and `01a10079-c8d1-752e-991a-4e76b523704e` remains sole GPU owner |

## Current gate state

G0–G6 remain unaccepted. Actual resident execution and native SM120 dispatch were recorded, but the original exploratory loader lost six up globals. The original-scale correction has bounded operator/reference and forced registered-runner evidence. Twenty natural requests selected none of the six affected experts. A later single 1,025-prefix/eight-output replay captured scores and unchanged top-eight IDs: targets ranked 24–128, with no target in top eight, and independent checks found no sampled preprocessing discrepancy. This explains nonselection for that prefix; it does not establish broader rarity or permanent inactivity. Five later controlled GPU passes match 90 retained projection replays and paired within-path full-model captures; their artificial routes cannot replace natural coverage. Full/cache handoff diverges before the correction, so numerical/quality and corrected timing acceptance remain blocked. Prior unverified zero-hit integration/benchmark artifacts are explicitly invalidated. Full-model quality, checkpoint activation-quantizer fidelity, complete cache/semantics fixtures and accepted performance gates remain incomplete. See [scale correction](nvfp4-scale-correction.md) for the runtime distinction and stopping rule.

## Review record

The Markdown conversion preserves requirements R01–R12, decisions D01–D08, interfaces I01–I06, packages WP0–WP6, gates G0–G6, all 28 action tasks and source links. D08 and obsolete planning-only language were updated to reflect the authorized repository initialization/scaffold. Page references were replaced with Markdown links. No proposed performance or workload default was promoted to a confirmed decision.

The prefill revision appends R13, D11, WP7/G7 and six tasks without renumbering existing records. G7 is unaccepted. Its minimum useful gain, tail/TTFT, quality, memory and decode regression criteria require owner freeze; no prefill result or new numeric threshold is invented. The [bounded M1 normal-routing checks](m1-normal-correctness.md) and [PR14 lifecycle record](evidence/m1-pr14-gpu-lifecycle.md) retain their exact scope: no speedup, broad G1 quality or graph qualification. PR15 benchmark remediation remains outside this planning revision; its outcome is not presumed. The initial local-only boundary was followed by the owner's separate draft-publication approval; neither approval authorizes GPU execution or merge.

Use the [decision template](templates/records.md#decision-ledger-entry) for future decisions. Append new IDs without renumbering existing records. A scope decision is not a technical gate pass.

## Later roadmap update

The initial system will run without DSpark or draft training. DSpark-style
speculation and Gemma-compatible draft-model training are planned later work
under WP5, after the stable non-speculative baseline. Evaluate existing
compatible drafts first; retain conditional data/training/checkpoint/evaluation
and verifier milestones on the roadmap. This planning approval does not authorize
actual training, data acquisition or compute spending. Target-only compatibility
and scale-correction GPU experiments have separate approval under D08.
See the [WP5 milestones](action-plans/wp5.md#dspark-and-draft-training-milestones).

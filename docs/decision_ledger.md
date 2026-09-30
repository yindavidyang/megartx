# Decision ledger

Updated 30 September 2026. Stable IDs follow the [master plan](master-plan.md). Proposed values are not frozen acceptance criteria.

| ID | Status | Decision and remaining evidence |
| --- | --- | --- |
| D01 | Confirmed | Begin with the owned RTX 5090, SM120; four RTX PRO 6000 GPUs are future scope |
| D02 | Confirmed | One user and one active request; no dynamic request batching |
| D03 | Confirmed | Aim to outperform a tuned, compatible FlashInfer-backed runtime on the RTX 5090 |
| D04 | Proposed | Gemma 4 26B A4B instruction model; exact NVFP4 checkpoint revision, tokenizer, chat template and tensor coverage must be frozen |
| D05 | Proposed | Text-only 2K/8K probes, conditional 32K; BF16 KV and separately reserved output capacity; prompt corpus/output/sampling policies pending |
| D06 | Proposed | Native expert W4A4 candidate; W4A16 remains a distinct numerical lane with a separate oracle |
| D07 | Proposed | 15% median 8K ITL improvement (20% stretch), p95/TTFT guards, quality margins and 2 GiB initial reserve; owner freeze and evidence required |
| D08 | Partially authorized | Initialize this repository with Markdown plans and README; open a draft PR for WP0–WP1 compatibility and FlashInfer benchmark scaffold. GPU execution, host setup, custom kernels, later packages, merge and deployment are outside this initial scope |

## Current gate state

G0–G6 are pending. Repository files and CPU unit tests do not establish target hardware compatibility, resident fit, numerical correctness or performance. The initial implementation provides contracts and offline tooling to prepare the evidence required by A0.1–A1.4.

## Review record

The Markdown conversion preserves requirements R01–R12, decisions D01–D08, interfaces I01–I06, packages WP0–WP6, gates G0–G6, all 28 action tasks and source links. D08 and obsolete planning-only language were updated to reflect the authorized repository initialization/scaffold. Page references were replaced with Markdown links. No proposed performance or workload default was promoted to a confirmed decision.

Use the [decision template](templates/records.md#decision-ledger-entry) for future decisions. Append new IDs without renumbering existing records. A scope decision is not a technical gate pass.

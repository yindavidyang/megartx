# Makora DSpark adoption: source-only requirements

The user selected [makora-ai/gemma4-26b-a4b-dspark](https://huggingface.co/makora-ai/gemma4-26b-a4b-dspark)
as the starting drafter on 2026-10-04, replacing custom training for now. Reuse
the installed native DSpark path. This note extends the integration requirements
at client commit `c5d7e04e379c692df507d8da62e70023d081af3d`; it changes no client,
launcher, plugin, protocol, runner pin, or resource authorization. No training,
weight download, package installation, runtime import, device query, or model
run was performed. The target-only zero-forward receipt remains necessary.
The immutable `c5d7e04` receipt plans remain tied to that commit; any later
execution checkout requires a new exact-HEAD freeze and review.

Public metadata pins the candidate revision to
`e6f739231e892fe4512c470136490d5441ffe62f`. The [revision config](https://huggingface.co/makora-ai/gemma4-26b-a4b-dspark/blob/e6f739231e892fe4512c470136490d5441ffe62f/config.json)
has SHA-256 `a6d5250b5c1978642154993547cdc2ecbf069e1cbe1f99b2d79cb982e45f8ee2`.
Its declared verifier is `google/gemma-4-26B-A4B-it`, with a five-layer Qwen3-style
`DSparkDraftModel`, width 2816, block size seven, six greedy proposals, mask token
four, Markov rank 256, and a confidence head. The repository declares Apache-2.0.

| Required interface | Pinned vLLM 0.30 source observation | Integration requirement |
| --- | --- | --- |
| Config routing | `speculators/algos.py:update_dspark` and `config/speculative.py` normalize generic `DSparkDraftModel` + Qwen3 to `Qwen3DSparkModel`; registry selects `Qwen3DSparkForCausalLM` | Use this native loader; do not select the separate `Gemma4DSparkModel` merely because the target is Gemma |
| Five target taps | Config auxiliary IDs `[3,10,18,25,28]` remain Eagle feature IDs; conversion produces target layer indices `[2,9,17,24,27]`; Gemma4 exposes `SupportsEagle3` | Preserve ordering, layer-index convention, residual/norm semantics and token positions; five BF16 features have width 14080, matching the reported `fc[2816,14080]`. Do not append a sixth feature from model-card prose |
| Vocabulary | Native loader requires reduced-vocabulary `lm_head` and `d2t`; `map_draft_to_target` computes draft ID + stored offset; probabilistic rejection uses a target-space scatter buffer | Validate all 32000 mapped output IDs against the pinned 262144-token target vocabulary and tokenizer/special IDs; `t2d` is training-only, not an inference substitute |
| Draft ownership | Native `load_dspark_model` keeps checkpoint-owned embeddings/head and resolves draft quantization/attention separately | Preserve the actual BF16 tensors; do not infer sharing or inherit target NVFP4/backend settings from the card |
| Runner | `gpu/spec_decode:init_speculator` selects `DSparkSpeculator` in the V2 runner; the old receipt runner source has no DSpark reference | Later native DSpark needs an independently reviewed V2 lifecycle/owner interface. Current receipt requires V1 and rejects V2 before engine acquisition |

The installed routing fix corresponding to [vLLM PR52197](https://github.com/vllm-project/vllm/pull/52197)
is present; package version alone is insufficient. [Source evidence](../evidence/speculative-makora-drafter-source-compatibility.json)
records SHA-256 identities for 13 inspected installed files. These are new
compatibility observations, not replacements for the receipt's historical pins
or proof that the checkpoint loads or runs.

The [public weight-file metadata](https://huggingface.co/makora-ai/gemma4-26b-a4b-dspark/blob/e6f739231e892fe4512c470136490d5441ffe62f/model.safetensors)
reports 2,411,575,146 bytes (2.41 GB, about 2.25 GiB) and advertises SHA-256
`c383b25a4f134c736024b5d2c25fba59375a53e1ca66e79d5d7a4543479aaeab`;
this file hash has not been verified locally. The parent's earlier tensor-metadata
inspection reports BF16 `embed_tokens[262144,2816]`, `lm_head[32000,2816]`, and
`fc[2816,14080]`, contradicting the card's statement that it lacks its own
embedding/head. No tensor payload was read in this adoption pass. Reserve
persistent drafter weights separately from target allocations. Logical BF16
draft KV is `5 layers × 2 K/V × 8 KV heads × 256 head width × 2 bytes = 40960`
bytes per cached token before page padding, capacity and allocator overhead.
Five target features alone cost 28160 bytes per retained token; their retention,
context projection, Markov/confidence, logits/scatter, compiler and external
workspace peaks remain unmeasured. A later explicit bounded run must budget
these domains and retain the 2 GiB GPU/8 GiB host reserves. The existing 8 MiB
incremental verifier scratch cap is unchanged and cannot absorb or excuse the
drafter's persistent weights, KV, or unknown temporary allocations.

Our corrected target is pinned `nvidia/Gemma-4-26B-A4B-NVFP4` revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6`, not the card's BF16 verifier. The
native correction, cached prefill/decode and target verification therefore still
need independent correctness checks. Draft acceptance and confidence calibration
against corrected NVFP4 are unknown; published acceptance or speed numbers do
not transfer. Rejection must preserve the corrected target's distribution and
must never use confidence as acceptance authority.

Still absent in this integration: reviewed V2-owned pause/receipt/release/shutdown
binding; a source-bound native DSpark config/checkpoint/tokenizer admission;
five-tap feature/KV ownership and rollback evidence; vocabulary-map integrity;
and bounded persistent/temporary allocation plus corrected-target acceptance
validation. The prefill owner retains the shared plugin/launcher edits. Exact
integrated-source review and a parent-assigned sole GPU slot precede any later
checkpoint acquisition or run. The current first receipt admits no requests,
resume, draft forward, or later verifier experiment.

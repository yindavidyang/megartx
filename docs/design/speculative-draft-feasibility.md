# Draft feasibility and accepted-token break-even plan

3 October 2026; CPU-only proposal from base
`0581f09259d18044ffb5f9e1363d0760f37ddd4f`. No weights, tokenizer payloads,
datasets, packages or training resources acquired. Public config bytes were
bounded to 1 MB each; new independent [source evidence](../evidence/supplied-candidate-source-review.json)
records exact URLs, revisions, SHA-256 and scalar observations. Existing PR18
source pins and historical evidence are unchanged.

## Exact target and candidate compatibility

The [selected NVIDIA config](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/config.json)
has 30 text layers, width 2816, 128 experts with top-8 routing, routed width 704,
shared width 2112, vocabulary 262144, GELU-tanh and logit soft cap 30. Its 25 local
layers use window 1024 and 8 KV heads of width 256; five global layers use 2 KV
heads of width 512. Global `attention_k_eq_v` does not imply cached processed K
and V share payload/storage. Quantizer metadata names ModelOpt producer
`0.43.0rc2.dev91+gc79ebc014`. This is config evidence, not loaded-weight proof.

Runtime selection remains vLLM 0.30.0 / FlashInfer 0.6.18.post1 / Torch 2.13.0,
original expert NVFP4 weights, selected scale-corrected W4A4 lane and BF16 KV.
The config advertises an 8-bit floating KV scheme; prove the explicit BF16-KV
override in the actual runtime rather than adopting metadata defaults. Top-level
EOS is `[1,106]` while nested text EOS is 1: freeze the actual target generation
policy and template before use. Tokenizer identity and native arithmetic are
unverified here; existing narrow correction evidence is not full-model G1.

| Candidate | Public config evidence | Decision / remaining checks |
| --- | --- | --- |
| [Google 26B-A4B assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant/blob/6e5aaaf4c42b98394530b8fda2e95cadd65c151c/config.json) | Backbone width 2816; four 1024-wide layers, three local and one global; four shared-KV layers; vocabulary 262144 | First feasibility candidate. Check token-ID/template hashes, exact target feature/normalization and shared-KV mapping, BF16-KV override, installed support, license/weight identity and peak fit before acquisition |
| [Reviewed RedHat Gemma 31B DSpark](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark/blob/0026c7d1899651ca3c45ede471712f04849723ac/config.json) | Names dense `google/gemma-4-31B-it`; auxiliary layers `[1,17,29,47,58]`; draft width 5376, five layers; reduced vocabulary 32000, Markov rank 256 | Incompatible checkpoint: requested target layers exceed this 30-layer architecture and width/features differ. Do not remap pins or attach the head. Re-training/adaptation is a new model decision |
| Deterministic supplied candidates | No model; exact one-hot q for stochastic verification | Immediate correctness/control path; acceptance is fixture-specific and says nothing about a draft's latency or held-out quality |

[Google's MTP description](https://ai.google.dev/gemma/docs/mtp/overview) identifies
shared embeddings and final target activations, and warns that batch-one MoE
verification can load extra experts and lose speedup. The pinned [vLLM MTP source](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/gemma4_mtp.py)
reviewed in PR18 is a feature/KV-interface lead, not evidence that this installed
runtime has supported the chosen assistant. Source revision, installed bytes and
model revision must be reconciled without blindly changing historical pins.

## Small length sweep and accounting

Propose fixed k=0/1/2/4/7, hence 1/2/3/5/8 target input rows, one request at a
time. Start at cached P=2048; admit P=8192 only after fit. No 32K cell. First use
supplied candidates to force acceptance 0..k, then real candidate quality only
after a compatible assistant is qualified. Keep long-prompt prefill as a separate
setup boundary; short verifier results do not accept G7. Profile and time
separately; do not infer latency from the CPU harness.

For each actual cycle record k, generated-versus-scheduled draft length, accepted
prefix, emitted/committed-token yield, terminal cause, target verify time,
draft time, processing/sampling/transaction/streaming costs, peak memory and
selected-expert union per layer for ALL k+1 input rows, including rejected work.
Also record accepted-prefix-only union as a diagnostic, never the traffic
denominator. Do not count the old anchor as a new output; do count fallback
or bonus. Actual terminal yield supersedes the uncensored expectation.

Let t0 be full target-only time per emitted token, V(k) the actual target block
verification cost, D(g) the entire draft cost for generated length g (even if
scheduled k<g), O(k) the remaining nonoverlapping overhead, and Y(k) actual mean
committed-token yield. Then:

```
cycle cost = D(g) + V(k) + O(k)
time per emitted token = cycle cost / Y(k)
speedup = t0 * Y(k) / cycle cost
maximum profitable draft cost = t0 * Y(k) - V(k) - O(k)
uncensored Y(k) = 1 + sum(i=1..k) P(first i proposals all accepted)
```

[speculative_cost.py](../../src/megartx/speculative_cost.py) implements this
accounting and per-layer union/repeated assignments, with finite scalar checks.
Prefix survival is not marginal token agreement. With equal independent survival
alpha, the last formula reduces to a geometric sum; independence is an optional
toy assumption, never an empirical premise. EOS/output caps use measured yield
or budget-censored survival. The break-even cost can be negative: retain k=0.
No RTX timings or predicted speedup are invented here.

For top-8 across m input rows, unique expert count lies between 8 and
`min(128,8*m)`, reaching at most 64 for m=8. Repeated route assignments indicate
potential reuse; they do not prove weight residency. Measure actual useful/padded
expert rows, dispatched kernels, weight bytes/cache traffic and local/global
attention cost. Router unions and FP4 quantization/repacking can make V(k) grow
more than a dense-model approximation suggests.

Correctness diagnostics use the bounded proposal; a later performance decision
uses the shared protocol's initially 30 randomized paired requests per admitted
cell with matched target-only and same-draft incumbent/candidate controls. Report
total emitted-token cost, request-level uncertainty, p95 delivered-token latency,
warm TTFT/full response time and lifecycle memory. A mean gain with bad tails,
failed cache correctness or poor reserve does not admit speculation.

The existing extra GPU scratch cap stays **8 MiB**, including staging/COW,
logits and all other incremental buffers. Host/private trace storage is separate.
`staging_budget()` derives raw BF16-KV and page/logit allocation bounds; the
[native step](speculative-native-step.md) details lifetimes and why boundary
shapes may fail this cap. Its rowwise head is for a diagnostic only. A batched
head or a larger scratch allowance needs an explicit resource decision; neither
is implied by this proposal.

## Conditional compatible training design

A generic verifier is only one part of speculation. [DSpark v1](https://arxiv.org/html/2607.05147v1)
adds a parallel backbone, lightweight sequential Markov/RNN conditioning and a
confidence scheduler. Its published acceptance and production speedups concern
other targets and serving systems; they do not predict Gemma on a 5090.
The Google assistant is Gemma MTP, with a different feature/KV architecture.

Prefer the existing matching assistant if its measured total cost is profitable.
If it fails, propose a new target-specific research draft: projected features
from explicitly selected valid layers of the 30-layer backbone; a small parallel
masked-token transformer; shared or mapped target embeddings/head; and optional
low-rank Markov/RNN head. Freeze layer IDs, feature normalization/casts and anchor
prediction alignment after inspecting the actual target hooks. Choose depth,
width, head rank, vocabulary map and block length from measured memory/latency
budgets, not dense31B weights. Full target-vocabulary logits avoid unproven
shortlist target-law truncation; a reduced draft vocabulary needs an exact map
and actual q law. Start fixed-length greedy; confidence scheduling is a later
calibrated extension. Stochastic adaptive scheduling needs its own selection-law
proof and full target-law support.

Data proposal: approved rights-cleared prompts across code, reasoning and chat;
document-level train/calibration/held-out splits before target generation; exact
tokenizer/template and target lane frozen; private target rollouts or approved
teacher-forced traces with accepted-prefix features at absolute positions. Record
data provenance, sample manifests, feature/logit conventions and leakage checks.
Train token prediction/distillation and sequential head jointly; evaluate prefix
survival and confidence calibration on untouched splits. Keep raw prompts,
features, logits, KV/tensors, checkpoints and logs private; publish only code,
Markdown and reviewed aggregate scalars. No data generation or training occurred.

Quality evaluation must pair actual accepted lengths/yield by domain, EOS and
length behavior, reference-correct output laws, greedy first divergence and
native cache continuation with draft latency/load/peak memory and full-cycle
cost. Training wall time and amortization are separate from serving gains.
Approve a bounded pilot only after architecture/data/feature/storage plans and
resource choices are concrete. Hardware, GPU hours, precision, optimizer,
checkpoint storage, target-generation volume, dollar budget and training service
remain explicit unresolved decisions. The 5090 belongs to the decode owner;
this milestone neither reserves it for training nor starts a job.

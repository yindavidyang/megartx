# Prefill CPU preparation and future profiling contract

This packet prepares one future serialized full-prompt profiling slot. It adds
CPU intake/accounting and independent index/cache fixtures, with no GPU runner,
kernel, plugin registration or decode-runtime change. Public draft publication
is approved; GPU experiments and merge require separate authorization. The
decode task `01a10079-c8d1-752e-991a-4e76b523704e` retains sole GPU ownership.

The isolated base is `09341f2ad3f6e4b03e8f264b9e3c59fe5ddb52ed`. The separately
prepared WP7/G7 master-plan revision is proposed context and has no source
dependency here. CPU source compatibility includes the explicitly reviewed PR15
controller/plugin pair below; it does not presume GPU, timing or quality results.
The [existing master contract](../master-plan.md),
[measurement protocol](../protocols/measurement.md),
[scale correction](../nvfp4-scale-correction.md) and
[M1 evidence boundary](../m1-normal-correctness.md) remain authoritative.

## Delivered CPU tools

Run from the repository root, using an existing Python installation:

```sh
PYTHONPATH=src python3 -m megartx.prefill_plan docs/prefill/profile-plan.json \
  --source-manifest docs/prefill/source-binding.json --root .
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_prefill_plan.py' -v
python3 -m unittest discover -s numerical_reference -p 'test_prefill_cache_reference.py' -v
```

The CLI reads bounded local JSON and source bytes. It reports unresolved freeze
references and the 2K/8K chunk matrix, including actual final M and total capacity;
32K stays deferred. `gpu_enabled` must be exactly false. Filling every intake
reference still provides no execution path. GPU packages are never imported.
The plan lives outside `configs/`, preserving the existing eight-manifest CLI.

The [plan validator](../../src/megartx/prefill_plan.py) rejects unknown fields,
duplicate JSON keys, nonfinite constants, bool-as-integer controls, changed
numerical lanes, unsupported scope, unreserved output capacity, and inconsistent
chunk/timing records. Source checks detect local drift relative to the reviewed
manifest. They are not cryptographic attestation of a host, approval references,
or GPU support. Independently verify the Git head/tree and future target receipts.

The [cache reference](../../numerical_reference/prefill_cache_reference.py) uses
small ideal vectors and processed-K/V position tags. It tests causal/sliding
visibility, full versus chunked masks, retention until all chunk queries finish,
page boundaries, absolute RoPE indexing, all-layer handoff, partial-state failure
and output reserve. It intentionally models no BF16/native arithmetic, quantizer,
kernel scheduling, numerical tolerance, model quality or training behavior.
The row ledger consumes supplied IDs/weights; it never generates or changes routes.
Its tile padding and GEMM FLOPs are logical accounting, not measured execution.

## Immutable inputs and exact semantics

[Source binding](source-binding.json) hashes twelve current repository sources and
evidence files. Its historical target stack is vLLM 0.30.0, FlashInfer
0.6.18.post1, Torch 2.13.0+cu130, Transformers 5.18.0, Triton 3.7.1, Python
3.12.3 and NumPy 2.3.5, on SM120. Installed attention/Gemma source and binary
hashes come from retained repository evidence; the associated upstream vLLM
commit is `ced6857afa0ea7b2e3f0846a62e1394e90f15607`. That association does not
attest binary/source equivalence. This Mac CPU run does not reproduce that stack.

The original twelve-file source vector remains immutable. A bounded, atomic
`reviewed_source_overlays` entries additionally accept the exact controller/plugin
pair at PR15 head `dfd77d8c80d333fc9531955795476f7e9c0cbab2`, as documented in
[the CPU source review](pr15-source-reconciliation.json). Mixed original/new
pairs, unknown controller bytes and changes to any other pinned source fail
closed. This is CPU source compatibility only; source/ownership/environment
receipts and all pending GPU gates remain required for future execution.

Whole-file controller pins were inspected baseline context for M1 fallback,
forward ownership and observer boundaries, not CPU numerical dependencies.
The reviewed delta adds bounded eager benchmark admission, request/lane/counter
handling and a stricter marker check. Its benchmark-only multirow fallback
accepts exactly 256 rows and rejects capture; it cannot serve the planned
255/512/1024/etc prefill sweep. Quantizer and model arithmetic stay fixed;
twelve original AST boundaries agree, and routed arithmetic agrees after removing
exactly the reviewed counter-only block. AST equality is source evidence, not
native arithmetic or GPU qualification. The future runner needs its own accepted
prefill workload/admission contract and cannot silently reuse PR15's benchmark.

The second atomic pair binds the separate default-off decode attribution source
in [its scoped reconciliation](decode-attribution-source-reconciliation.json).
Only complete original, PR15, or attribution pairs are accepted; a third overlay,
mixed pairs, and quantizer/source drift fail closed. The original vector and PR15
receipt remain unchanged. Exact reviewed AST nodes normalize the profiling wrappers
and fail-stop head cleanup when comparing arithmetic and stream bodies; altered
work, scope names or enable conditions are rejected. Actual ledger/head failure
tests separately verify profiling stop/reset and primary-error propagation.
This compatibility repair keeps the prefill runner disabled and all GPU gates open.

The pinned NVIDIA checkpoint is
`nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6`.
This preparation reread only its two public metadata JSON files:
[config](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/config.json)
hash `4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98`, and
[quantizer metadata](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/hf_quant_config.json)
hash `fca2ea21cded31e6cff2c56ee83a162dcd5d3ff292cbf2f6083702e9fc324454`.
Selected fields and producer identity are retained in the binding; no weights,
tokenizer, prompts or private tensor payloads were downloaded or published.

| Boundary | Required contract and profile consequence |
| --- | --- |
| Model | H=2816, 30 layers, 128 experts, top 8, routed F=704, shared F=2112; one request creates many prefill rows |
| Local attention | 25 layers, Q=16, KV=8, D=256, window=1024; visible keys `max(0,p-1023)..p` inclusive |
| Global attention | Layers 5/11/17/23/29, Q=16, KV=2, D=512; all keys `0..p`; no cross-layer cache sharing |
| RoPE and norms | Absolute positions; NeoX local theta=10000; global proportional theta=1000000, fraction=.25, exponent denominator 512, active pairs `(i,i+256)` for i<64; epsilon=1e-6, learned Q/K norms, unweighted V norm, unit attention score scale |
| Global K/V | Raw projection weights can be shared; normalized/rotated K and unweighted-normalized V are distinct processed data and cache storage |
| Shared/routed branches | Router starts from post-attention residual with no-weight RMS, BF16 root and learned dimension scale, F32 logits; top-eight normalization precedes expert factors. Preserve separate branch norms, residual order and layer scalar |
| Head | Tied unscaled embedding/head after final norm, vocabulary=262144, soft cap=30; last-prompt-row logits for timing, all-row teacher forcing in separate correctness runs |
| Export | ModelOpt `0.43.0rc2.dev91+gc79ebc014`, NVFP4 group=16, routed expert W/A only; exported FP8 KV metadata does not select this plan's BF16 KV |
| Weight layout | Original E2M1 packed bytes, even K in low nibble, E4M3 block scales and F32 projection globals; gate/up scale maps are separately swizzled 128x4 because 704 crosses a tile |
| Runtime quantizer | `nvfp4_runtime.capture_experts` uses layer-max FC1 and FC2 input calibration; `quantize` uses its reciprocal with CUDA `nvfp4_quantize`, layout_128x4, no shuffle; retain rounding/clipping/zero/fast-math contract and no environment overrides |
| Adapter | Six unequal gate/up weight globals remain separate; native correction, defective gate-only control and original-checkpoint oracle are separate lanes. FC1 BF16, reviewed F32 GELU tanh/up then BF16, frozen FC2 quantizer; finalize=false, BF16 down before F32 weighting/reduction |

Fixing weight globals does not make runtime layer-max activation calibration
equal to checkpoint per-expert calibration. The current adapter still executes
suppressed stock contributions and adds corrected expert work; include both in
the ledger. Its reduction grouping is not a bit-exact reference replacement.
Natural positive coverage of affected experts and broader model/quality gates
remain prerequisites; CPU fixtures and forced routes cannot supply them.

The pinned [vLLM Gemma source](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/gemma4.py)
also identifies router, Q/K/V norms, attention and branch call boundaries. Future
observers must bind the actual installed source and forward registry before
using those names. The [bounded layer-0 attention evidence](../layer0-norm-attention-contract.md)
found FA2 for full prefill and XQA for cached decode; record actual provider and
launch geometry for every new chunk rather than extending that result to 2K/8K.

## Workload matrix and measurements

[profile-plan.json](profile-plan.json) fixes text only, one active request,
TP1/EP1, unchanged natural routing, BF16 compute/KV, no prefix reuse, dynamic
batching, LoRA or speculation, eager execution, FlashInfer attention and
FlashInfer CUTLASS MoE, and finalize=false. Graph admission is a separate phase.
The recorded matmul controls keep TF32 off and BF16 reduced partial reduction
and split-K enabled, as in the retained router diagnostic; F32 outputs alone do
not establish F32 internal accumulation. Bind actual flags/dispatch on the future
target and compare any alternative arithmetic preference in a separate lane.
Any unsupported control fails intake; the future runner must refuse explicit
unsupported opt-in while ordinary requests retain the verified library prefill
fallback. No new dispatch is connected by this packet.

| Prompt work | Proposed chunk sweep / M | Admission |
| --- | --- | --- |
| Full 2048 tokens | 255, 256, 512, 1024, 2048 | Source prepared; exact-host dispatch and resident fit pending |
| Full 8192 tokens | 255, 256, 512, 1024, 2048, 8192 | Same; 255 deliberately gives partial tails |
| Full 32768 tokens | 256, 1024, 8192, 32768 | Deferred until peak fit including 256 output positions and frozen reserve |
| Cache fixtures | 257, 1023/1024/1025, repeated wraps, page/capacity edges | CPU index semantics only; native/cache fixtures still required |
| Short speculative verifier | Existing cache plus anchor/proposals, accepted and rejected rows | Separate future WP5 phase, route-union/rollback/acceptance costs; excluded here |

Chunk length is an upper bound on rows in one forward; it is neither prompt
length nor expert M. Record actual scheduler forwards and sum exactly P prompt
rows, with no output rows mixed in. A full-prompt cell fails if row coverage,
token IDs, offsets or model calls differ from its bound schedule. Unsupported or
OOM cells are deferred, never counted as wins. Freeze exact token IDs and hashes,
tokenizer/chat-template hashes, sampler/RNG/EOS/output policy before any timing.

Use a monotonic host clock and synchronized GPU events on the actual stream(s).
Never subtract unrelated host and device clocks. In a warm request let:

| Timestamp | Definition |
| --- | --- |
| request_accept | Accepted complete request, before tokenization/host input |
| input_ready | Exact tokenized/templated IDs and required host metadata ready |
| prompt_begin | Before first chunk preparation/H2D/scheduler work; preallocation and bucket warmup already complete |
| prompt_complete | All P rows through final model hidden/norm boundary, no first-token head or sampling; last prompt GPU work complete |
| kv_ready | Exact committed all-layer I03 state available to the fixed decode control, including any conversion/handoff |
| first_token | First output token available at the declared client streaming boundary, after head/sampling/serialization |

Report whole-prompt wall latency `prompt_complete-prompt_begin`, prompt tokens/s
`P/latency`, GPU critical-path duration, sum of device stage durations (which may
overlap), handoff `kv_ready-prompt_complete`, and warm TTFT
`first_token-request_accept`. Also retain input preparation, per-chunk scheduling,
H2D, first-token head/sampling and stream serialization components. If the runtime
cannot separate a boundary, label the measured combined scope; do not silently
call a server-forward or single-chunk duration pure prefill. Whole-prompt wall
latency includes gaps between chunks. Allocate first-head/handoff overlapping
work once in a dependency timeline; do not sum overlapping spans into TTFT.

Cold load/import/compile/autotune/allocation is a separate process-lifecycle
record, with cold TTFT only when the request boundary really includes it.
Preallocate complete KV/output capacity and bounded scratch before warm timing;
warm every admitted shape/backend bucket, finish synchronization, reset sequence
state and bind initial cache/allocator state. Preallocation is excluded from warm
prefill but reported in cold time and lifecycle memory. Never precompute routes,
quantized activations or reusable prompt K/V for the timed request. No prefill
graphs or M1 scratch assumptions are inherited from decode evidence.

## Future observer and profiler design

Implement a new explicitly selected `prefill_profile` adapter only in a later
reviewed change. Keep proposed event/receipt schema below local to that adapter;
coordinate any shared I02/I03 or lifecycle interface edits with the parent first.
It must be default-off, bound to exact source/head/environment, and fail before
device work on unsupported lane/model/shape/observer mode or missing admission.
Startup and cleanup must reuse the decode owner's then-qualified ownership
protocol via a frozen interface, rather than copying PR15's mutable work.

Use three serialized modes with separate run IDs: correctness/capture, profile,
and timing. Correctness can save private intermediates/cache bits and compare
oracles; profile can annotate/copy counters and use Nsight; timing disables every
observer, `.cpu()`/`.item()` route logging, trace, forced route and profiler hook.
Keep CUDA synchronization at declared request boundaries, not every operator.
Correlate GPU launches with CPU runtime-launch correlation IDs, stream IDs and
source-bound annotations; asynchronous GPU time need not overlap a CPU span.

| Profile region | Concrete observations / source boundary |
| --- | --- |
| Dense attention projections | Gemma QKV and O projection calls; actual M/N/K, BF16 storage/accumulator policy, split-K/GEMV, global projection sharing and duplicate work |
| Norm/RoPE/cache/attention | Input, Q/K/V and branch norms; rotary reads; cache writes/metadata; local/global layer ID, Q/KV geometry, query M, visible context, FA2/XQA/other identity, grid/block and memory reads |
| Shared dense MLP | Shared gate/up/down and GELU tanh product, F=2112 boundaries; separate norm and cast costs |
| Router/dispatch | Router input/root/scale, F32 projection, pinned top-8 tie order and weights; gather/pack/quantize/dispatch/scatter/combine, not just expert GEMM |
| Grouped experts | Each expert's FC1 gate/up and FC2 M/N/K, selected M, positive-weight M, actual scheduled/padded M, tiles/tails; correction adapter work separately, including suppressed stock work |
| Exit | Final norm, selected-last-row extraction, tied head GEMM/soft cap, sampling, synchronization and stream serialization; all-row quality runs separate |

For each `(request, layer, chunk, forward, expert, projection)` record selected
and positive-weight row counts, source route hash, correction hit count,
logical N/K, padded N/K/M, actual kernel/scale-layout identity, packing bytes,
quantizer global/segment identity, scratch allocation/last-use and device span.
Summed selected M must equal `chunk_M*8`; empty groups have M=0. The correction
path's additional launches do not alter that route count. Save histograms and
min/median/max/skew, repeated hits, empty/tiny groups, useful/executed work,
gate/up segmentation and scale/tile tails. The CPU ledger projects all-selected
padding only; actual kernel zero-weight skipping, tile padding and correction
overlap must come from the profile. Live routing remains in full-model timing.
Captured routes may explain shape microbenchmarks but cannot replace live routes.

Produce a stage time/byte ledger for dense, attention, routed/shared, output head,
launch gaps and conversions. Profiler kernel sums can double-count overlap;
rank recoverable critical-path costs and their Amdahl ceiling before proposing
chunk/library/fusion changes. Many same-expert rows can reuse weights; a route
union or one user does not turn distinct expert weights into a shared-M GEMM.
Keep per-expert/projection calibration and rounding domains intact.

In separate memory observer runs record allocated/reserved/device-used peaks at
load, import/repack, compile/warmup, every chunk, conversion/commit, first head,
and fixed decode continuation. Account for model/scales, retained originals,
BF16 K/V capacity, local/global cache layout/page padding, prompt activations,
route/packing buffers, scratch and graph pools if later admitted. BF16 K/V payload
per stored position is 8192 bytes per local layer and 4096 per global layer; these
are logical payloads, not allocation or peak-fit measurements. A local multirow
chunk can need up to `1024+M-1` positions before recycling; staged/current keys,
allocator slack and all other live memory still count. Freeze measured reserve
and bounded build RSS/time, scratch, requests, wall time and retained trace bytes.
The existing proposed 2 GiB reserve and M1 build/scratch bounds do not admit WP7.

## Handoff, admission and stopping conditions

Compare continuous and chunked teacher-forced traces at window/page/tail/capacity
edges against the frozen lane oracle before timing. Inspect all-layer stored K/V
bits, processed identities, absolute positions, masks, committed length and
capacity; arithmetic tolerance cannot excuse stale ownership. Ring/page addresses
are never RoPE positions. Every chunk query must finish using its required keys
before window recycling. Conversion/handoff costs remain in the matched ledger.

Use the fixed accepted decode implementation after both incumbent and any future
prefill candidate. Compare the same prompt state, first token and separately
reserved continuation, teacher-forced/logit/quality evidence, p50/p95 ITL, long-run
stability, dispatch and lifecycle memory. A faster prefill result cannot relax
decode, quality or memory guards. W4A16 and checkpoint-calibration controls need
their own oracle and scorecard; an unmatched or unsupported incumbent is no win.

The future slot requires explicit scope approval and sole-owner handoff after
initial decode results; exact clean heads/trees, source/binary/environment/package
locks and compiler/driver/device receipts; accepted G0/G1 and decode control;
positive natural correction coverage; frozen prompt/oracle/sampling/EOS/thresholds;
finite memory/resource limits; and owned-process startup/cleanup receipts. Intake
nulls are pending decisions, never inherited authorization. Stop before launch
when any prerequisite is missing. 32K remains queued until a separate peak-fit
review passes with output reserve and the frozen safety margin.

After admission, stop on first source/dispatch mismatch, coverage gap, nonfinite
intermediate, oracle failure, cache/capacity/position error, memory/reserve breach,
process-ownership ambiguity, bound/timeout/trace overflow, or failed cleanup.
Drain only owned work; poison any uncertain partially committed cache and rebuild
using the verified library fallback from the committed token log. Do not run a
second GPU job or kill/reconfigure another owner's work. Keep failure receipts.

The first future slot should run the smallest admitted incumbent correctness and
observer case, then one 2K/8K full-prompt baseline and bounded chunk sweeps.
Tune supported library chunk/backend choices before custom device work. Paired
timing later uses frozen trial count/order/uncertainty rules from the measurement
protocol; no acceptance threshold is invented here. An insufficient stage budget
or unresolved guard yields defer/stop/library-only, not wider tuning by default.

Proposed private outputs: `input_manifest.json`, `incumbent.jsonl`,
`prefill_ledger.csv`, `expert_shapes.json`, `memory_timeline.csv`, trace hash
manifest, correctness/handoff errors and decode-regression records. Publish only
sanitized scalar summaries and exact provenance. No raw prompts/tensors/traces
belong in this preparation PR. Short verifier profiling and draft training remain
separate future phases requiring their own authorization and cache transactions.

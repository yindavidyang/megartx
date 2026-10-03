# Prefill collector and serialized launcher, CPU milestone

This implementation starts at verified green main
`0581f09259d18044ffb5f9e1363d0760f37ddd4f`, including PR20's repair of PR19
bindings. The owner authorized implementation, CPU validation and a public draft
PR. GPU execution is closed: decode task `01a1030a-782f-76ab-a9d0-e6939166d075`
owns the single5090 until the parent explicitly transfers the slot. No SSH,
CUDA import, native compilation, training, package installation or model download
was performed for this milestone. Old plans, source pins and historical evidence
remain byte-identical; this packet adds a separate source extension.

## What is implemented

- [prefill_collect.py](../../src/megartx/prefill_collect.py) consumes actual callback
  observations: exact tokens/absolute positions, all 30 layers, natural selected/
  positive/actual scheduled expert M, prompt rows, distinct K/V region ownership,
  full chunk read lifetimes, complete I03 handoff and first-logit bootstrap.
  Decode continuation has its own one-row admission and 255 input steps; the last
  prompt head already emits output #1 of 256. Interrupted frames poison state.
- [prefill_observe.py](../../src/megartx/prefill_observe.py) implements real
  per-instance callable wrapping and restoration, with module/qualname/file
  checks before invocation. `GemmaCallableObserver` discovers literal Gemma layer
  owners and wraps QKV/O projections, Q/K/V norms, attention, shared MLP, router,
  combined MoE and the last-row head in separate profile jobs. It does not wrap
  `RoutedExperts.forward_modular`: the original scale plugin checks that exact
  `__func__` internally. Norm hooks wrap the instance's `_forward_method`, which
  the pinned CustomOp dispatch actually uses. Quantizer/scale/route operands and
  returned tensor objects pass through unchanged.
- [prefill_kv.py](../../src/megartx/prefill_kv.py) supplies the dormant native
  BF16 BHNC observer. It reuses the original installed-source/registry/writer
  checks, without the controlled 33-token workload. Before dispatch it checks
  actual writer slots against retained absolute tags, capacity P+256 and alias
  bounds. After all query streams finish, it hashes processed K and V separately
  in batches of at most two rows; no whole-cache copy or weight expansion occurs.
  Raw bits stay transient/private. Initial physical policy is **full_context**;
  a bounded ring is unsupported until its installed writer/layout is reviewed.
  Its prefill-local `read_frame` requires actual tensor instances and compares
  dtype objects to the supplied module's `torch.int64`. The historical controlled
  helper still string-compares dtype to `int64`, which rejects canonical
  `torch.int64`; that inherited bug remains outside this isolated repair. An AST
  regression checks equivalence of every other frame/source/metadata/ownership
  check, with additional non-tensor writer metadata rejection.
- [prefill_launch.py](../../src/megartx/prefill_launch.py) implements serialized
  startup/request/drain/owned-cleanup orchestration through `LifecycleProvider`.
  It verifies the provider handshake and resource floors before startup, cleans
  partially failed startup, poisons failed requests and stops at the first gap.
  Timing calls do not create a collector, route capture, hooks or trace. Observed
  jobs compare their callback ledger against the PR19 normalized records, verify
  the 255 actual decode inputs against delivered tokens, then run PR19's strict
  record validator. Source/CUPTI/dispatch/fit/comparison evidence is never invented.

The attached native writer shares the collector's failure lifetime. Any callback,
hook-context exit, event synchronization/resolution, final-handoff hashing,
launcher interruption, cleanup failure or record/publication failure burns that
request. Retries cannot resolve events or publish a handoff. Owned hooks still
restore their replacements; provider cleanup still runs and retains the primary
exception when cleanup also fails.

`CudaEventClock` operates on an already imported module supplied by an admitted
provider. It records the actual current stream, rejects a stream change inside
an observed range, synchronizes each observed stream before resolving durations,
and keeps host and device clocks separate. Inclusive ranges can overlap. These
are event ranges, **not** kernel launch geometry, measured DRAM bytes or a GPU
critical path. Actual CUPTI correlation, exact stream translation and native
suboperation attribution must come from the independently pinned provider.

The CPU tests use explicit synthetic providers, cache slots and fake device
objects. They exercise callable replacement/restoration, state machines and the
complete orchestration, not installed vLLM or CUDA. No installed native method, real GPU event or target K/V bit extraction
has been run. `records_consistent` means
parser/accounting consistency; all numerical/performance/GPU gates remain false.

## Source and site boundaries

[live-binding.json](live-binding.json) extends immutable PR19/20 inputs with
only new prefill source/tests/docs. [runtime-sites.json](runtime-sites.json)
binds the exact native callable and enclosing semantic sites. Source was read
from immutable upstream vLLM commit
[`ced6857afa0ea7b2e3f0846a62e1394e90f15607`](https://github.com/vllm-project/vllm/tree/ced6857afa0ea7b2e3f0846a62e1394e90f15607).
Gemma, Attention, layernorm and FlashInfer bytes match the historical installed
hashes retained in this repository. Additional linear, routed-expert and
multimodal source hashes are upstream source observations, not fresh host or
binary receipts. Source/binary equivalence and actual callable registry remain
unattested until the future host review. The public packet includes hashes and
scalar metadata, not upstream source copies or private prompt/tensor/log payloads.

QKV/O event M/N/K comes from the actual tensor and module dimensions. Attention
ranges report query M, required K/V union N and head dimension K; a sliding mask
means each row has a different visible prefix. Norm ranges report actual input
shape. Shared MLP, router, MoE and head ranges are explicitly combined scopes;
these ranges do not isolate every fused kernel/substage. Native stock MoE and
six scale-correction paths must be separately correlated, including suppressed
stock work, original gate/up globals, both activation quantizers and FC1/FC2.
Selected M is never substituted for measured scheduled/padded M. Existing
`nvfp4_runtime.py`, controller/plugin and shared launcher are not edited.

The PR15 benchmark's M256 fallback is not used. Declarative code support covers
2K chunks 255/256/512/1024/2048 and 8K chunks
255/256/512/1024/2048/8192, including actual final M=8/32 for chunk 255.
This says nothing about installed scheduler support or fit for those cells.
Unknown shapes and 32K remain rejected. The source extension does not permit
unknown controller/plugin drift or rewrite historical review receipts.

## I03 cache and verifier bootstrap

For a local chunk `[start,end)`, retain `[max(0,start-1023),end)` until **every**
query and layer completes. For `[1024,1280)` this is `[1,1280)`, 1279 positions.
A naive 1024-slot ring overwriting positions 1..255 before that completion fails
writer admission. Only after completion may local tags shrink to
`[max(0,end-1024),end)`. Global layers 5/11/17/23/29 retain `[0,end)`, with distinct
processed K/V even when raw projection weights are shared. Cache addresses are
never absolute RoPE positions. K/V region IDs denote disjoint content regions;
they may live in a shared allocation. Cross-layer cache alias is rejected.

Final handoff retains PR19's fields: `committed_length=next_position=P`,
`capacity=P+256`, BF16, complete/unpoisoned, all-layer local/global position
coverage, independent processed K/V digests, layout digest and native comparison
receipt. A separate bootstrap observation binds request identity, prompt token
hash at P-1, one complete vocabulary logit row at P-1 predicting P, logit bits,
source site/digest, oracle comparison receipt and fixed sampler. Sampling emits
anchor z at P without writing its K/V. `cached_length=P`, emitted count=1 and
remaining capacity=255. The verifier initializes history=prompt+(z,), pending
uncached anchor z at P; its first verification row predicts P+1. EOS/budget exit
needs no verification. This prefill implementation itself uses ignore-EOS 256.
Hash/reference checks preserve identity; independent review authenticates private
logits, sampler and native comparison evidence. No I06 speculation transaction,
rollback or tentative-state publication is implemented here.

## Frozen proposed GPU protocol

[live-plan.json](live-plan.json) is disabled. Compiler limits are 2 GiB RSS and
300 seconds, host free floor 8 GiB, device free reserve 2 GiB, extra scratch
8 MiB, one GPU job. Proposed additional bounds are one selected cell per packet,
8 client requests, 1800 seconds and 8 MiB of retained traces/normalized evidence.
If a trace or event budget cannot cover a cell, narrow the profiling capture
under a new reviewed scope or defer it; do not silently truncate it. Two-row K/V
hashing is deliberately perturbing, bounded in transient storage and excluded
from timing. Full-context local allocation is accounted using P+256 rather than
assuming the logical final window is the physical allocation.

After explicit handoff, freeze final Git head/tree/patch, target packages/binaries/
source/driver/device, unchanged checkpoint, tokenizer/template and exact private
prompt IDs, source/site registry, provider/AOT/CUPTI identities, physical layout,
resource lease/prior cleanup and comparison tolerances. No current or moving
sibling checkout substitutes for those pins. The checkpoint and runtime remain
Gemma4 NVFP4 revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, vLLM 0.30.0,
FlashInfer 0.6.18.post1, Torch 2.13.0, original layer-max W4A4 quantizer and BF16 KV.
Freeze sampling `greedy_seed_1234`, output `ignore_eos_256`, eager synchronous
one-request TP1/EP1, no prefix cache/dynamic batching/LoRA/speculation and
finalize=false. Retain original scale domains and matmul flags from PR19.

1. **Fit characterization:** separately review a minimal source/owner/resource
   scope, beginning 2K chunk256 and then 8K chunk256, P+256 physical capacity.
   Record load/repack/compile/warmup, each chunk, handoff, head and continuation
   peaks with device free reserve at every phase. No fit receipt is assumed in
   order to measure fit. This reduced characterization scope requires its own
   reviewed inputs/provider; the baseline execution gate is not a bootstrap
   shortcut. Stop on OOM, missing reserve, resource/lifecycle or dispatch gaps.
2. **Cache correctness:** under a separately frozen scope, test window/page/tail
   edges, `[1024,1280)` union, full/chunked continuous-reference processed bits,
   masks, absolute positions, original quantizer scales, all-layer handoff and
   final-logit/uncached-anchor bridge. Run the fixed accepted decode control for
   256 outputs. Missing coverage/finite/native comparison evidence stops timing.
3. **One baseline:** only after accepted fit/correctness/G0/G1 and decode control,
   select one supported 2K/8K cell at a time. PR19 supplies initialization,
   correctness, separate attention/dense/expert/head profiles, memory, cold and
   warm timing: 9 runs/8 client requests. Whole-prompt wall boundaries include
   all chunks and scheduler/H2D gaps; TTFT ends at the actual first client token.
   Separate cold startup and handoff/head/sampling/streaming, report combined
   scopes where boundaries cannot be separated, and do not add overlapping
   device ranges to TTFT. Sweep the other planned chunks only after their fit
   and installed dispatch are reviewed. 32K stays deferred.

This stage produces a trustworthy incumbent baseline and an explicit hotspot
ledger. It does not select an optimization, claim speedup, accept G7, enable
graphs or begin training. Paired performance trials and acceptance margins need
separate frozen inputs after the baseline is understood.

## CPU commands and remaining integration

```sh
PYTHONPATH=src python3 -m megartx.prefill_launch --root . --workload p2048 --chunk 256
PYTHONPATH=src python3 -m megartx.prefill_launch --root . --workload p8192 --chunk 255
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_prefill_collect.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_prefill_launch.py' -v
```

`--output` creates a new file exclusively. `--execute-gpu` always rejects before
calling a factory or importing a native stack, even if all declared receipts
are filled. `SerializedLauncher.execute_serialized` is the closed public adapter;
`_collect_admitted` contains dormant orchestration and is tested with synthetic
providers. No environment flag, module path or CLI can connect a provider here.
The existing shared launcher has no prefill provider/client branch. The future
owner must implement/qualify that narrow provider and attach callbacks inside the
actual EngineCore ForwardContext, with complete startup/drain/cleanup, client
clock and stream correlation. Any shared launcher/controller/plugin changes
must be coordinated before integration; this branch makes none. `g0`, `g1`,
fit, numerical, native correction coverage, installed dispatch, ownership and
provider reviews remain unresolved. Publication of this draft accepts none of
them and does not transfer the GPU slot.

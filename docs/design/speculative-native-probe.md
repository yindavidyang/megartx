# Owned native k0/k1 diagnostic: CPU implementation and launch boundary

This patch implements actual installed-model calls in
[`speculative_native_probe.py`](../../src/megartx/speculative_native_probe.py).
It is based on exact CPU harness head `464c728a8758a558fd0bfe326f3ed7a2f10fe0c4`;
no sibling working changes or installed files are used. It does not import or
execute Torch/vLLM during CPU inspection. It has **not run natively** and grants
no GPU, quality, speedup, G1 or G5 admission. The
[minimal protocol](speculative-native-protocol.json) freezes the scope.

## Installed source binding and actual access

Read-only SSH inspected 22 installed Python files under the pinned Python 3.12
environment. Every file byte-matches the documented vLLM upstream commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`; the
[source receipt](../evidence/speculative-native-source-binding.json) records each
hash, byte bound and exact upstream URL. This is source equality, not loaded
weight, binary, CUDA, page-size or native compatibility evidence. The cached
target config was read and matches `4e379cc8…`; no weights were downloaded or
hashed by this task. Runtime binding additionally requires the existing verified
checkpoint shard/tokenizer/template manifest and actual loaded config identity.

`OwnedNativeProbe.from_runner` binds an actual `GPUModelRunner`, its actual
`get_model()` Gemma model, `static_forward_context` attention/cache owners,
`kv_cache_config` allocator placements, kernel block sizes and metadata builders.
Exact native owner classes and source bytes are checked. There are no injected
model/head/cache providers. A read-only prefill observation ticket cannot grant
this mutation capability.

The worker cannot reserve free blocks: the allocator is in EngineCore.
`run_engine_core_probe` calls the actual
[`BlockPool.get_new_blocks`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/block_pool.py#L664),
retains the real block objects at reference count one, and passes their IDs in a
PID/start-time/nonce-bound single-use RPC ticket. Each cache group receives
`ceil((2048+4)/actual_B)` serial-capacity blocks plus one COW block. IDs across
groups are distinct because groups can overlay backing storage. Actual storage
offsets, shapes, strides, padded spans and allocation bounds are checked again
in the worker; overlapping page/layer spans are rejected. Splitting manager
blocks into different kernel page sizes and layouts whose conservative padding
charge cannot fit are rejected pending a separate concrete adapter.

## Actual forward, bootstrap and transaction

The diagnostic sequentially replays eight 256-input prompt chunks using only
allocator-owned pages and the same loaded target model. It retains all P2048
prefix pages; the installed per-layer causal/window attention supplies local
visibility. It calls the head only on the final prompt hidden row to obtain the
first generated token. That token is emitted once and stays pending at P.

For the verifier, actual `CommonAttentionMetadata` uses query indptr `[0,n]`,
sequence length `c+n`, absolute positions `c..c+n-1`, and identical private block
tables for attention and physical writer slots. The actual builder's
`build(0, common)` result enters `set_forward_context(..., skip_compiled=True)`.
The driver calls the actual model on `[anchor,candidate]`, retains both returned
hidden rows, and calls actual `compute_logits(hidden[j:j+1])` for each row.
This preserves Gemma's head soft cap and multimodal token suppression; it avoids
the ordinary runner's final-row-only selection. Greedy CPU argmax uses the first
vocabulary index on a tie. Generation penalties, temperature sampling and native
stochastic RNG are outside this frozen policy. Natural routing and the original
scale-corrected target lane remain active; M1 preparation and observational
capture modes must be off.

Instance-owned writer wrappers check the actual slots against the private table,
copy distinct processed K/V to CPU, invoke the installed writer, drain its work,
and compare actual stored bytes. They compare old committed pages before and
after each layer's writer. Snapshots are taken one layer at a time, with host
copies of individual strided page views rather than a GPU whole-cache gather.
The pinned source performs the cache update before attention and reads it during
attention; the wrappers do not alter math. All 30 writer completions and actual
head rows are required before publication.

Unaligned committed fragments are copied to a private page before any write;
committed pages/tables remain intact. Publication changes one immutable
diagnostic state after completion, selection, resource checks and suffix disposal.
Only `len(new_emissions)` input rows are consumed; the final output remains
pending. Rejected suffix bytes in private pages are zeroed and checked before
continuation. The diagnostic owns this state; it does not publish scheduler
requests, output delivery acknowledgements or RNG state.

Cancellation before dispatch, after submitted work drains and before publication
returns the original state and discards private rows. Retry and the next
single-input forward are compared. Cancellation does not preempt a CUDA kernel.
Errors poison the worker for owned teardown. Device synchronization precedes
writer restoration and pool release. RPC/drain uncertainty leaves allocator
references held and the EngineCore poisoned; it must not resume serving.

## Frozen comparison and allocation accounting

The k0, forced-rejection and full-accept cases compare **every** real hidden,
FP32 logit and processed K/V row bit-for-bit against a separate serial replay.
For rejection, row 1 is compared with an additional serial hypothetical branch
that consumes the supplied wrong candidate. Its fallback continuation is compared
with the greedy serial branch. The next single-input forward after both rejection
and full acceptance checks the published cache/history frontier. Zero tolerance
is intentional; any shape-dependent numerical difference stops the diagnostic
and needs review before thresholds change. Raw rows stay private; only scalar
receipts leave the worker. The full packet is at most 85 target calls, bounded
by 96; no broad k/context matrix or benchmark is included.

Distinct BF16 KV costs 225280 bytes per input across 30 layers. The actual
private-page charge uses cache strides/allocator padding, never just this logical
payload. Native softcap reassigns division/tanh/multiplication outputs: two FP32
262144-element buffers can coexist, costing **2097152 bytes**, even with a
row-by-row head. Both BF16 hidden rows add 11264 bytes. Device metadata, MoE,
attention, external-library workspace, COW/promotion coexistence and allocator
rounding belong to the same 8388608-byte cap.

| Uniform page example; not an actual installed page-size claim | Private KV page bytes | Plus two FP32 head buffers and hidden rows |
| --- | ---: | ---: |
| B16, unpadded logical geometry | 3604480 | 5712896 |
| B32, unpadded logical geometry | 7208960 | 9317376; reject before scratch |
| B64, unpadded logical geometry | 14417920 | 16526336; reject before scratch |

Mixed group page sizes/padding must be derived from the actual runner; B remains
unset. No B16 fit is assumed. The worker rejects an infeasible padded lower bound
before query allocations. A native PyTorch allocator fraction is set before
diagnostic allocations, after drainage/cache release; it cannot bound direct
CUDA library allocations. Therefore a source-bound upper-bound receipt for
external workspace is mandatory and charged separately. Unresolved live allocator
padding is also rejected. Allocated/reserved peaks then include the explicitly
charged private pages and external bound. This conservative guard may reject a
potentially feasible installed layout; it never raises the cap or admits it based
on a raw payload estimate. Actual native scratch, live padding, library allocation
coverage and fit remain unmeasured blockers.

The same device must keep 2 GiB free, the host 8 GiB free, and private host trace
storage stays below 64 MiB. Compiler memory is 2 GiB shared across the host with
a 300-second limit; the reviewed launcher must enforce it, not an environment
label supplied to the worker. Startup, whole-run deadline and cleanup remain
owned lifecycle responsibilities, with one GPU job and no package/system changes.

## Smallest remaining launcher exposure

No shared files are edited or hooks registered by this patch. The existing
`run_scale_validation.py` owns process/resource startup and cleanup; bypassing it
with a standalone model loader would weaken that contract. The only public
command here is CPU-only source/protocol freeze:

```sh
python3 -S scripts/run_speculative_native_probe.py
python3 -S scripts/run_speculative_native_probe.py --installed-root /path/to/site-packages --output /private/new-freeze.json
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_speculative_native_probe.py' -v
```

The parent must review two explicit exposure changes before launch: configure
`--worker-extension-cls megartx.speculative_native_probe.NativeProbeWorkerExtension`
in the owned startup branch; and register one default-off EngineCore utility
`megartx_owned_native_probe` delegating to `run_engine_core_probe(..., enabled=True)`.
The latter runs through the existing EngineCoreProc synchronous utility handler
after `initialize_from_config` and an EMPTY fresh engine's
`pause_scheduler(mode="keep", clear_cache=False)`. No request/batch/transfer/DBO
work may coexist; the utility loop remains blocked during Worker RPC and the
engine stays paused until shutdown. This registration belongs in the shared
plugin/lifecycle owner, with the utility called through the existing utility
client. The model, allocator and runner interfaces are already available; this
is the concrete missing exposure, not a replacement mock provider.

Independent source/CPU review, exact-head green CI, parent source/protocol
acceptance, source-bound allocation coverage and an assigned sole-GPU slot remain
necessary. No CUDA import, model load, GPU job, training, download or remote
mutation occurred in this task.

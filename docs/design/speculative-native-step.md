# Smallest native verifier step and source-bound hook feasibility

This delivery is a CPU contract/harness, **not an installed native verifier**.
The proposal remains unapproved; decode thread
`01a1030a-782f-76ab-a9d0-e6939166d075` owns the GPU. No device launch, SSH,
checkpoint or package action occurred. The source sites below were freshly read
at vLLM commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; three existing
Gemma/Attention/FlashInfer hashes match historical repository bindings, and the
runner hash is a new public-source observation. This is not an installed source
or binary equivalence receipt. [Source hashes](../evidence/supplied-candidate-source-review.json)
remain separate from historical pins.

## Concrete next patch

Implement a separate, default-off **owned eager diagnostic driver** for one
supplied candidate (k=1, two target inputs), initially at cached P=2048. Use the
selected stock target numerical lane without the M1 one-row custom path. k=0 is
the serial control. Freeze first-output bootstrap from prefill logits, then run
`[anchor,candidate]` at `[P,P+1]`, compute both target rows, select on CPU, promote
only the consumed prefix, and compare the next single-input target forward
with separately replayed serial state. Force one rejection and one full-accept
case. No assistant checkpoint or scheduler integration is needed for this step.

The proposed new files are `src/megartx/speculative_native_probe.py`, its CPU
adapter tests and `scripts/run_speculative_native_probe.py`; they do not exist in
this PR. They would consume source-bound runner/cache owners in a fresh owned
process. First implement/review those bindings and host metadata construction
without execution; only then request a bounded slot. Keep ordinary serving and
the scale plugin inactive with respect to speculation. Reuse the qualified
scale-corrected target lane, with a clear statement that native short-row
qualification is new.

| Pinned source site | Observed mechanism and concrete adapter need |
| --- | --- |
| [Gemma4Attention.forward, 535–567](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/gemma4.py#L535) | K normalization/RoPE and distinct V normalization precede attention. Capture writer inputs here or at the cache writer, without changing math or routes |
| [Gemma causal LM head, 1588/1625–1642](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/gemma4.py#L1625) | `forward` supplies hidden states; `compute_logits` applies the configured soft cap. Select each actual row explicitly; generation penalties/constraints remain a separate processor step |
| [Attention context/update, 682–736](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/attention/attention.py#L682) | Cache owner plus layer slot mappings come from forward context; writer is called before attention. Staging needs compatible temporary page tables and slot mapping, not just an observer after forward |
| [FlashInfer metadata build, 1292–1321](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backends/flashinfer.py#L1292) | Actual sequence lengths, query indptr, block tables and page size determine dispatch. Build one causal two-row request with correct absolute positions and per-layer window; record the dispatched lane |
| [FlashInferImpl.do_kv_cache_update, 2577–2614](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backends/flashinfer.py#L2577) | Splits stored K/V and writes mapped physical slots. Feasible interception point for per-instance, owned-lifetime staged pages and post-writer byte/address receipts; no transaction hook is supplied by this function |
| [Runner row selection, 2243–2275 / 4497–4498](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py#L2243) | Ordinary path chooses only the final query row; speculative metadata chooses multiple rows. A driver must directly retain both hidden rows and call the head per row, or explicitly implement reviewed speculative metadata. Merely passing two inputs to ordinary serving loses row 0 |

Staging approach: committed page tables remain immutable; a request-local table
references old prefix pages and bounded new pages. At an unaligned last page,
copy its committed fragment to a COW page before placing new KV. Use that table
for both writer slot mapping and causal attention. After all 30 layers complete,
promote only consumed rows/tables and publish the pending token/frontier. On
failure, drain owned work, discard staged pages and restore the original table.
Rejected suffix stays inaccessible to the next forward. A sliding-ring overwrite
with length reset is not acceptable. Row/address equivalence and unchanged old
bytes must be verified natively; hook presence alone does not prove feasibility.

## Resource derivation and unresolved fit

Processed BF16 K/V bytes per input row (distinct K and V) are:

```
local:  25 layers * 2(K,V) * 8 heads * 256 width * 2 bytes = 204800
global:  5 layers * 2(K,V) * 2 heads * 512 width * 2 bytes =  20480
total: 225280 bytes = 220 KiB
k=1: 2 rows = 450560 bytes; k=7: 8 rows = 1802240 bytes
```

Staged payload lives through forward completion, selection and publication or
abort; old committed pages remain live throughout. New/COW page storage includes
page padding and any copied prefix fragment: for page size B and start c, page
count per layer is `ceil(((c mod B)+k+1)/B)`, and storage across layers is that
count times B times 225280. The exact B/layout/allocator must come from the
installed cache. Raw payload size is not an allocation or lifetime guarantee.

| Illustrative B=16 case (not an installed page-size claim) | New/COW K/V storage | With one FP32 logits row | Remaining below 8 MiB before all other scratch |
| --- | ---: | ---: | ---: |
| k=1, c=2048 aligned | 3604480 B | 4653056 B | 3735552 B |
| k=7, c=1023, two touched pages | 7208960 B | 8257536 B | 131072 B |

The full-vocabulary FP32 head costs 1048576 bytes per retained row; eight rows
alone cost 8 MiB. The diagnostic can compute/process/copy one row at a time,
retaining only scalar argmax IDs on device or one host row, then release head
outputs before publication. Record this policy and every temporary head buffer;
it cannot support batched-head performance claims. Additional QKV/hidden tensors,
attention/MoE scratch, conversion/promotion coexistence and allocator overhead
must be measured against the same **8 MiB total extra GPU cap**, with at least
2 GiB free reserve. The table is a lower bound, not an admitted fit. Larger B or
the 1023 boundary may already make the cap infeasible. Defer that shape or ask
the parent for a specific justified resource change; never silently use 64 MiB.

Host/private trace storage has a separate proposed 64 MiB ceiling and contains
no GPU allowance. Release/device-copy buffers still count in GPU peak accounting.
The source probe itself reads no device state. [staging_budget()](../../src/megartx/speculative_cost.py)
and tests reproduce these calculations; proposal page size/fit receipt remain
unset. Expand k or context only after correctness and a measured lifetime/fit
receipt; no clearance is implied.

## Shared files and parent decisions

The next driver can start with new files. Source-bound per-instance cache owner
wrappers, temporary metadata and explicit forward invocation avoid installed
source patches. Feasibility still requires the runner owner to expose its loaded
model, metadata builder and private page allocation/lifetime without ordinary
scheduler publication. If that access is unavailable, stop and agree the runner
extension with the parent; there is no supported native adapter in this PR.

`src/megartx/vllm_scale_plugin.py`, `m1_live.py`, `controlled_kv_capture.py` and
the prefill collector remain shared-owner files and untouched. Their existing
one-row/33-input collector plans cannot be expanded silently. If a common
observer/launcher entry point is needed, send the exact hook and ownership
change to the parent before integration. Reuse source/layout checks by a new
adapter; do not reinterpret observational capture as transaction support.
Parent decisions remain native implementation ownership, exact scope/gate
admission, page-layout and 8 MiB fit, and a serialized sole-owner slot. Native
stochastic sampler/RNG hooks, assistant features and wider G1/G5 are later work.

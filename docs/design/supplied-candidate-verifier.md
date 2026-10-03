# Supplied-candidate target verifier: CPU integration boundary

3 October 2026. Isolated base `0581f09259d18044ffb5f9e1363d0760f37ddd4f`.
This extends [PR18's contract](speculative-verifier-contract.md) with an opt-in
[CPU transaction harness](../../src/megartx/speculative_verifier.py). It does
not register with vLLM, change prefill/decode dispatch, execute a device, load a
model or train a draft. `enabled=False` is the default; a backend advertising
anything except `cpu_contract` is rejected. No existing source pins changed.
The base and workspace ancestors contain no applicable AGENTS.md/.agents files.
Repository WP5, review and measurement contracts were read; wider G0/G1/G5
remain unaccepted. The narrow M1 correction proof does not establish full-model
G1 or correctness of a multirow target forward.

## API and position contract

Construct `TargetVerifier(session, backend, enabled=True)` and call
`verify(candidates, remaining_budget=..., eos_token_ids=..., mode="greedy")`.
Candidates are immutable target token IDs; no draft checkpoint is required.
The returned `CycleResult` describes published emissions, acceptance, stop
state and next-forward frontier. Cancellation returns `None`; validation or
backend faults raise without publishing. This is single-thread Python only.

`Session` binds request, epoch, generation, frozen `policy_id`, full logical
history, per-layer cache snapshots, two RNG states and terminal state. The
caller must bind `policy_id` to actual model/lane/tokenizer/template/processor
hashes; its string is an identity comparison, not proof that these were checked.
Backend supported row counts, vocabulary and maximum staged K/V bytes per row
are required. Reserve the whole `k+1` staging block and maximum possible
consumed prefix before forwarding; a discarded staged suffix need not fit in
authoritative global KV. Actual returned bytes are checked against the declared
reservation. CPU returned-byte validation cannot constrain a malicious/native
backend's allocations; a native adapter must reserve actual storage first.

| Quantity | Required meaning |
| --- | --- |
| `c = len(history)-1` | Cached input length; final history token is pending at c |
| `inputs` | Already emitted anchor followed by k candidates |
| `input_positions` | Absolute c through c+k; no padding positions |
| `prediction_positions` | Absolute c+1 through c+k+1 |
| Processed row j | Target law/argmax after consuming input j; row 0 checks candidate 0 |
| Published KV count | `len(emitted)`: anchor plus newly emitted tokens except the last |
| Next anchor | Last newly emitted token, still uncached; next forward starts at new c |

`ForwardRequest.prefix(j)` supplies the exact hypothetical history for row j
processors, including the consumed input. Backend logits/laws must already
include the frozen temperature/truncation/penalty/constraint policy. The harness
does not silently invent processors or convert logits to probabilities.
`TargetBatch` echoes request/epoch/generation/policy and both position arrays,
and returns all-layer staged K/V and all k+1 processed target rows. Missing,
misaligned, stale, nonfinite or wrong-vocabulary results are rejected.

### Prefill transition agreed through the parent

After prefill completes P prompt inputs, its committed length and next position
are P; its final logits predict position P. Sample and emit first generated
token z from those logits, then initialize verifier history as `prompt + (z,)`
with cache containing only P prompt inputs. Pending anchor z is at P. Verifier
row 0 consumes z at P and predicts P+1. Do not repeat the final prefill input or
emit z again. The first generated token z already counts toward the request's
output budget; pass the remaining budget excluding z. If z is EOS or the total
budget is spent, skip verification. Bootstrap RNG and policy ownership belong
to the target sampler; the harness starts at the established pending anchor.

The prefill collector's completed cache remains separate from tentative rows:
all 30 layer completions and distinct processed-K/processed-V storage and
comparison receipts precede handoff. A local `[1024,1280)` prompt block retains
the union `[1,1280)` until its queries complete, then final window `[256,1280)`;
global history is `[0,end)`. Those I03 observations do not implement I06 rollback.
This packet changes no sibling prefill or collector file.

## Selection, stopping and RNG

Greedy mode accepts the longest exact target-argmax prefix, then emits the first
fallback or all-accepted bonus. Exact ties use the lowest vocabulary ID, which
must be checked against the eventual runtime sampler. A candidate EOS stops
only if accepted; fallback/bonus EOS also stops. Budget stops after exactly the
remaining allowed emissions, with no extra bonus. The terminal token stays
pending and is not materialized into KV. A terminal session skips forwarding.

Stochastic mode accepts exact normalized int/Fraction target and actual draft
laws. It uses `min(1,p(y)/q(y))` and normalized positive `p-q` correction, or the
final target law for a bonus, as in [Leviathan et al.](https://arxiv.org/html/2211.17192v2).
Without `q_rows`, supplied candidates have deterministic one-hot draft laws.
General supplied q must be the actual proposal law after every map/truncation
and conditional sequential head. Positive q mass is required for each proposal.
Supplying fixed arbitrary tokens with unrelated network softmax q is invalid.

The harness clones separate, domain-seeded acceptance and correction/bonus
Python RNG states. Integer `randrange` sampling avoids floating uniform-grid
rounding for rational laws. State publication joins RNG with KV/history/stop;
failure/cancellation discards the cloned draws so retry replays. Callers own a
separate proposal RNG and must ensure conditional independence. Seed separation
is an engineering measure, not a statistical proof of native independence.
Distribution correctness is mathematical under exact laws and ideal independent
random draws; Python replay and exact toy enumeration do not qualify GPU
floating-point sampling. Same-seed target-only token equality is not expected.
Choose stochastic k before proposal sampling; token-dependent confidence
scheduling and approximate laws remain unsupported pending a separate proof.

## Cache staging and publication

`LayerCache` uses immutable processed K and V byte witnesses with absolute token
and RoPE tags. Addressing uses page/offset and optional ring modulus. Its CPU
`visible()` view reads the old committed snapshot plus only the current/past
staged rows, using absolute causal/window bounds. The whole verifier suffix stays
outside committed slots even across repeated wraps. Rejection is implemented
by preparing new slots with only the consumed prefix and dropping suffix rows;
no overwritten committed bytes need an undo operation.

Every layer's replacement and the new sequence/RNG/frontier are validated before
one session assignment publishes them. The harness checks cancellation before
forward, after validation and immediately before publication. Reentrant calls
are rejected; this assignment is not thread-safe/native atomicity. In-flight
device cancellation, async writer drainage, page COW, alias lifetime, scheduler
features, emission acknowledgement and draft-context rollback are absent.

## CPU evidence and required native hooks

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_speculative*.py' -v
python3 -m unittest discover -s numerical_reference -p 'test_speculative_verifier_reference.py' -v
```

New tests compare PR18 decisions and independently rebuild serial byte witnesses,
every returned synthetic logit row, committed physical slots and next-forward
output. They exercise k=0/1/2/4/7, every acceptance count, page/window boundaries
at 1023/1024/1025 and repeated wraps, bootstrap, EOS/budget, invalid suffix rows,
capacity/reservation guards, stale identities, backend faults, three cancellation
boundaries and retry RNG. An exact two-token joint-law enumeration uses separate
proposal/acceptance/target grids and checks the serial target product law.
Generated logits/bytes are explicitly CPU witnesses; no native result is claimed.

Before a device implementation, the runtime owner must provide:

1. A causal k+1-row target forward in the selected numerical lane, returning
   every processed target row at the documented prefix; all-layer short-shape
   parity and real sampler tie/near-tie controls.
2. Separate per-layer BF16 K/V staging or qualified page COW/undo, with absolute
   tags, layer completion, bounded scratch and no committed ring destruction.
3. One request/generation-bound publication hook for sequence, cache/page tables,
   RNG, auxiliary target features, draft state and output delivery. Async work
   completion/drainage and cancellation linearization need actual native evidence.
4. A rejected-suffix rollback receipt and a next-forward continuation comparing
   real logits/K/V/addresses with serial target-only state; do not use digests
   alone without independent comparison and completion receipts.

The [frozen proposal](speculative-gpu-proposal.json) is blocked on those hooks,
runtime identity and parent admission. It is not an executable launcher or a GPU
authorization. The [feasibility plan](speculative-draft-feasibility.md) separates
generic verification, Gemma MTP and DSpark research.
The [smallest native step](speculative-native-step.md) identifies pinned source
sites, missing hooks and an explicit 8 MiB scratch/fit decision; this delivered
milestone is a CPU contract/harness, not an installed native verifier.

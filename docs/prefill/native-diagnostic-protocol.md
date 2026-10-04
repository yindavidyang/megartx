# Native prefill fit and handoff diagnostic

This implementation connects the loaded native provider, one-request client and
owned launcher. It is default off. CPU tests and source inspection do not admit
GPU execution, numerical correctness, G0/G1/G7, TTFT or a performance baseline.
The historical baseline entry remains closed.

The integration base is reviewed PR22 head
`cf656d6632b9f1b08019a527a9269a9ed0fb0a26`. The original proposal's SHA256 is
`1661b04383c03f436550d63e514bef3a5ab8500316cbf117fc621e85ef8b4207`.
The runtime semantic boundary is committed separately from the additive CPU
catalog reconciliation. Every historical catalog and receipt is retained.

## First run scope and admission

Exactly one owned server root and one request: 2048 private frozen prompt IDs,
chunk 256, BF16 full-context physical cache capacity 2304, ordinary native
original-scale correction, TP1/EP1, eager, prefix caching off, explicit
`--no-async-scheduling`, chunked prefill on and hybrid manager disabled.
M1 custom preparation, controlled routing and verifier mutation are off.
The existing six bounded startup native/reference fixtures remain unchanged.

The private plan freezes actual repository hashes, source HEAD, checkpoint
config/index hashes and shard size/mtime identities. Full checkpoint shard byte
rehashing is not implied. Existing original packed/scales/global tensor checks
remain in the unchanged startup fixture. A separate owner-authored clearance
receipt must bind this exact plan and HEAD, completed CPU review and the parent's
GPU slot clearance. There is no automatic creation of a clearance receipt.

After review, the owner can freeze the private plan on the target using:

```sh
PYTHONPATH=src python scripts/prefill_diagnostic_client.py \
  --freeze-tokens /private/p2048-ids.json \
  --checkpoint "$MEGARTX_BASE/models/gemma4-nvfp4" \
  --output /private/prefill-plan.json
```

A clean committed checkout is required. Only after exact review and slot handoff:

```sh
python scripts/run_scale_validation.py --label prefill-native-p2048 \
  --mode native --client prefill-native --trials 1 --prefill-chunk 256 \
  --prefill-native-plan /private/prefill-plan.json \
  --prefill-native-clearance /private/owner-clearance.json
```

## Actual observation binding

`LoadedModelIdentity` identifies the loaded runner's actual owner PID/start ticks,
checkpoint file identity and installed source bytes. `LoadedEngineAccess` and
single-use `OwnedFrame` grant observation only. They do not grant allocation,
scheduler pause/resume, rollback, staging, commit, release or a mutable lease.
DSpark's EngineCore utility and worker extension remain a separate capability;
no sibling uncommitted module is imported.

The provider captures actual `_prepare_inputs` logits indices, actual FlashInfer
builder `CommonAttentionMetadata`, group block tables, kernel block sizes,
query-start locations, sequence lengths and ForwardContext writer slot maps.
All30 original layer/backend/writer owners, BF16 shapes/strides, physical
allocation placements and compute windows are checked. Installed source shows
that hybrid-manager disable promotes allocation while preserving local compute.
Runtime window and allocation checks remain necessary.

Cache groups can overlay one backing allocation. Only actual request-owned page
byte intervals may be disjoint; unknown within-page layouts or overlapping owned
pages fail. Historical 131072-byte local/global layer-page shapes are CPU test
fixtures, never a current geometry or fit assumption. Full P+256 view capacity is
checked against each actual layer.

Before dispatch, absolute token/position identity and block-to-slot correspondence
are bound. After all CUDA query work synchronizes, within the same ForwardContext,
cache owners, tables and slots are rechecked. Two-row copies hash processed new
K/V rows and check retained boundary bytes. Full-context writer-slot continuity
is tracked through all eight prompt chunks and 255 one-row decode inputs. This
coverage is explicitly `new_written_rows_and_retained_boundaries`; it is not an
independent comparison of every retained tensor element.

The actual selected hidden-row bits bind `compute_logits`. Actual `_sample`
outputs and discard masks distinguish seven intermediate discarded prompt samples
from the final prompt sample. Output #1 is an uncached anchor at position2048;
255 actual decode inputs follow. Streamed token IDs and usage must match the
native sample ledger. The final committed length is2303, with256 emitted outputs
and physical capacity2304. No comparison receipt or oracle acceptance is fabricated.

## Resources, evidence and completion

Bounds: observer/candidate scratch8MiB, GPU free reserve2GiB, host available8GiB,
aggregate compiler RSS2GiB and shared compiler wall300s, total run wall1800s.
The deadline begins before startup and gates startup, client, drain and final
publication; owned cleanup has its separately bounded existing TERM/KILL budget.
Telemetry failures, a foreign GPU job, owner/source drift or any callback error
poison the run and stop only its owned identities.

New diagnostic records share a cross-process8MiB evidence budget, checked before
every append and final publication. Raw prompts/tokens/tensors remain private.
The incumbent model/cache/workspace and historical startup/reference artifacts
are separately ledgered; the FP64 startup projection floor15,859,712bytes is not
an8MiB candidate scratch violation. The two-row and single-logit-row observation
operations enforce allocation-size bounds before copies. This is a source-derived
scratch upper bound, not a fabricated measured scratch peak. Actual allocated,
reserved, device-used/free and host available memory are recorded by phase.

`observer.json` and `client.json` describe request observations. `fit.json` is an
output, published only after matching actual request/token ledgers, source
reverification, retained resource checks and complete owned-process cleanup.
It never qualifies numerical correctness or performance. A failure cannot publish
complete fit evidence. Runtime geometry, loaded-model binding and measured fit
remain pending until an admitted target run actually completes.

## Independent native correctness control, separately pending

After each cell has independently measured fit, plan two fresh owned contexts:
continuous native P2048 and native chunk256, both with exactly the same private
prompt IDs, checkpoint/tokenizer/scales/quantizer, BF16 allocation policy and
continuation. Compare all30 processed local/global K/V intervals with actual
RoPE/window/page boundaries, last-prompt logits, emitted uncached anchor direction
and active decode continuation. Freeze an independent oracle, tolerances and
failure rules before either comparison is accepted. No default bit equality or
numerical tolerance is assumed. The observational diagnostic cannot supply its
own independent correctness proof. 8K,32K, profiling, optimization and timing are
deferred behind these gates.

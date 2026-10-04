# Decode SF layout proof: counts and observed equality

The bounded operation diagnostic at exact native source
`c20677b43eb75418f6beed31b0c13a1dd27da891` passed source-bound admission.
Independent source and protocol review cleared the run beforehand. Independent
outcome review accepted the resulting evidence for operation counts and observed
equality only, with no evidence blocker. The [sanitized scalars](evidence/m1-sf-layout-gpu-scalars.json)
bind its source, plan, binary, traces, immutable private result packet and later
review evidence. Publication after that GPU head adds only this report and scalar
evidence; the measured source and private packet keep their original bytes.

The independent CPU replay verified 92 packet files, 71 transferred-file hashes,
278 source files and 27 build sources, and reproduced the local result verifier.
It independently recomputed both operation inventories and unique correlations,
checked the exact plan/input/backend/native-phase identities, matched all eight
256-token sequences to the immutable original and validated the six forced
fixture traces. Capture/observer route controls remain absent and the timing
reader rejects the profile packet. Review used retained evidence without a new
remote filesystem attestation or device run. The original private packet keeps
its pre-review status; this report records the subsequent scoped acceptance.

Each active FC1/FC2 native layout still undergoes full semantic shape/stride
comparison. The two immutable layouts are constructed and exhaustively proved
once from the pinned installed CUTLASS configuration. Descriptor readbacks,
route/owner/extent checks, quantization, arithmetic, dispatch, stream dependencies,
safe fallback and error propagation remain. The [frozen contract and inventory](m1-sf-layout-contract.md)
describe the source change and graph blockers; historical manifests and reviewed
prefill overlays retain their bytes.

The CPU source/control proof establishes removal of 225,280 repeated physical
coordinate evaluations and sixteen temporary seen vectors per qualified
preparation, after one 28,160-coordinate initialization proof. That is
27,033,600 repeated coordinate evaluations per 120-call profiled lane. These
are source-derived host work counts, not observed GPU operations or latency.
The existing host phase name `descriptor_validation_enumeration` remains for
trace compatibility; its 240 phase entries per lane do not count coordinates.

| Observed activity over four decode frames | Stock | Fused |
| --- | ---: | ---: |
| Ordered preparation calls | 120 | 120 |
| Preparation D2H copies | 1,800 | 1,800 |
| Explicit preparation stream fences | 360 | 360 |
| Preparation pointer queries | 1,200 | 1,200 |
| GPU operations with unique CUDA API correlation | 6,440 | 6,320 |

These counts and complete operation inventories match the accepted
[`cf656d6` lean path](m1-decode-lean-results.md). Each lane contains four full
model/head/sampler frames, with thirty ordered preparations per frame. Each
preparation retains fourteen descriptor copies, one 32-byte route copy, three
explicit fences and ten pointer queries. Exact operation names, geometry,
shared memory, transfer bytes, API correlations, execution identities,
containment, lane ordering and stream checks pass the current trace validator.
The transferred traces reproduce that validation on CPU.

The pilot uses retained seed-9471 inputs, one warmup and one measurement per
stock/fused 2K/8K context, eight 256-token requests and the existing separate
one-token drain. All eight output sequences exactly match the immutable accepted
original `2c65ebd` run and agree across stock/fused warmup/measurement within each
context. Actual dispatch totals are 30,600 stock, 30,600 fused and 4,800 prefill
fallback calls. All six existing forced startup BF16 fixtures report exact
equality. Natural requests selected zero correction rows, leaving positive
natural correction coverage unestablished. The 24 correction selections per
profiled lane lack enclosing routed annotations; per-layer attribution remains
withheld. Stream wait operands and device durations remain unmeasured.

The independently reviewed protocol used zero extra captured/observer calls,
artificial routes or diagnostic GPU requests. Detailed diagnostic preservation
is supported by the counted compiled helper, byte-identical bridge normalization,
real installed-header layout controls and independent lifecycle/admission
controls on CPU. The GPU pilot covers actual production layout admission and
observed numerical/dispatch equivalence. It supplies no new diagnostic
envelope/mask/payload GPU coverage. The original proposal and separate protocol
correction remain immutable private records.

Fresh native build and compiled lease/binding controls pass. Build peak aggregate
RSS is 746,909,696 bytes over a 6.214-second compiler budget span; the runtime
sampled compiler peak is 37,748,736 bytes over a 9.375-second shared span, with
zero unknown/work compiler identities. These resource scalars remain within
the 2 GiB/300-second caps and do not establish all transient peaks. Incremental
candidate device allocation is zero; existing model and startup reference
workspaces remain incumbent allocations.

Final source/AOT/binary/metadata guards, all 278 source files, copied adapter
sources, packages, checkpoint config/index hashes and shard stat, and frozen
receipts pass. Full checkpoint shards were not freshly rehashed. Run and client
exits are zero; cleanup leaves zero owned identities and GPU processes. Actual
postrun GPU state is 41 MiB used/32,101 MiB free with no compute process. The GPU
slot was explicitly released before CPU evidence transfer and comparison.

This evidence establishes operation counts and observed equality. It provides
no clean timing, speedup, general quality, G1 or graph qualification. Ordinary
benchmark compiler quiescence and existing trace/diagnostic admission stay
unchanged. New host/client elapsed fields, raw prompts, tokens, tensors and logs
remain private. Default-off native verifier code remains outside this patch's
validation.

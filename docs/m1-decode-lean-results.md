# Validated one-check decode operation path

The fresh diagnostic at exact GPU source
`cf656d6632b9f1b08019a527a9269a9ed0fb0a26` passed the source-bound operation
admission gate. Independent source/protocol review cleared that head before
the run; independent review of the resulting evidence remains pending. The
[sanitized scalars](evidence/m1-decode-lean-gpu-scalars.json) bind the source,
plan, binary, traces and immutable private count/equality packet. Publication
after that GPU head adds only this report and scalar evidence.

The minimal helper performs one complete fresh eligibility check immediately
before preparation dispatch on the capture-free path with observers disabled.
It retains dynamic descriptor/owner/route checks, original quantizer and kernel
math, safe unsupported fallback, error propagation, buffer ownership and stream
dependencies. It stores no eligibility cache or caller-supplied checked flag.
The [check inventory](m1-decode-attribution.md) separates static initialization,
necessary dynamic production checks and diagnostics. Capture/observer paths
retain their prior checks.

| Measured activity over four decode frames | Original stock | Original fused | Lean stock | Lean fused |
| --- | ---: | ---: | ---: | ---: |
| Preparation D2H copies | 1,800 | 1,920 | 1,800 | 1,800 |
| Explicit preparation stream fences | 360 | 480 | 360 | 360 |
| Preparation pointer queries | 1,200 | 2,400 | 1,200 | 1,200 |
| GPU operations with unique CUDA API correlation | 6,440 | 6,440 | 6,440 | 6,320 |

Original counts come from the independently reviewed source-specific
[`2c65ebd` attribution](m1-decode-attribution-results.md), not historical eager
timings or the rejected first lean attempt. Each new lane contains four full
model/head/argmax frames and 120 ordered layer preparations. Each preparation
has fourteen descriptor copies, one 32-byte route copy, three explicit fences
and ten pointer queries. Stock uses seven preparation kernels; fused uses six.
Full operation names, launch geometry, shared memory and transfer-byte inventories
match the frozen expectations. Every device operation has a unique launch API;
model/head/sampler execution identity and complete ordering, API process identity,
exact routed lane labels, preparation containment and stream consistency pass.

The measured fused reduction is one route D2H copy, one explicit fence and ten
pointer queries per eligible layer: 120 copies, 120 fences and 1,200 queries over
four frames. Common descriptor checks remain. Driver allocation-query reduction
is still a source count because those driver events are absent from the trace.
This confirms removal of redundant work. It establishes no speedup, CPU-gap
cause, clean timing, quality, G1 or graph qualification. New host nanoseconds,
client elapsed fields and raw traces remain private without elapsed comparison.

The same bounded seed-9471 pilot used the original private 2K/8K inputs, one
warmup and one measurement per stock/fused context, eight 256-token requests
and a separate one-token drain. All eight output sequences equal the accepted
original run exactly and match across stock/fused warmup/measurement within each
context. Actual dispatch totals are 30,600 stock, 30,600 fused and 4,800 prefill
fallback calls. All six forced startup BF16 fixtures pass exactly. Natural
requests selected zero correction rows, so positive natural correction coverage
remains unestablished. The 24 correction selections per lane lack an enclosing
routed annotation; per-layer correction attribution stays withheld. Stream wait
operands and device wait durations remain unmeasured.

The new [prospective diagnostic rule](m1-profile-only-admission.md) was approved
before this run and permanently rejects timing-summary replay, including removal
of optional trace/launch markers. Ordinary benchmark compiler quiescence remains
unchanged. The fresh run recorded zero unknown/work compiler identities; sampled
compiler RSS was 35,782,656 bytes and the shared compiler budget span was 29.203
seconds, within 2 GiB/300 seconds. These are resource accounting scalars. They
are not model timings or a proof of exact transient peaks. The failed `3f06df1`
packet remains rejected, and historical metadata-admission holds remain intact.

Fresh build and compiled lease/binding controls pass; installed pins are
unchanged. Source/AOT/binary/metadata guards, final 262-file source and adapter
checks, package identities, checkpoint config/index hashes and shard stat, and
frozen receipts pass. Full checkpoint shards were not freshly rehashed. Run and
client exits are zero, cleanup leaves no owned identity or GPU PID, and actual
postrun GPU state is 41 MiB used/32,101 MiB free with no compute PID. The GPU slot
was explicitly released. No extra GPU run, sweep or graph restoration follows
from this evidence.

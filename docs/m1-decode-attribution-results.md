# Decode attribution and one-check preparation proposal

The diagnostic run used exact source `2c65ebd3ef450b37dd88a9d89c630fd10165642d`,
with the original eligibility and preparation path. Its private source/command,
build, AOT, activation, input/dispatch, metadata, trace and cleanup receipts are
frozen together. The [sanitized scalars](evidence/m1-decode-attribution-gpu-scalars.json)
identify that source and the private artifact hashes. No historical result is
reclassified by this run, and no speedup, quality, G1 or graph gate is passed.

Both four-step 2K windows contain four model forwards, logits heads and greedy
argmax calls, and 120 layer preparations. Every one of the 6,440 GPU kernel/copy
events per lane has exactly one CUDA API correlation. The six forced startup
fixtures matched BF16 values exactly; these artificial fixtures remain separate
from the natural requests. All eight scheduled requests matched input transcripts,
usage and requested native backends. Source, adapter copies, packages, checkpoint
config/index/shard stat, and frozen receipts matched after the run. Owned cleanup
completed and the GPU slot was released. Full checkpoint shards were not rehashed.

| Measured preparation activity, four steps | Stock | Fused |
| --- | ---: | ---: |
| D2H copies | 1,800 | 1,920 |
| 32-byte route-ID copies | 120 | 240 |
| Descriptor copies | 1,680 | 1,680 |
| `cudaStreamSynchronize` calls | 360 | 480 |
| Summed GPU D2H duration | 0.694 ms | 0.733 ms |
| Summed host `cudaMemcpyAsync` duration | 11.324 ms | 11.888 ms |
| Host descriptor readback/fence phase | 10.327 ms | 10.214 ms |
| Host descriptor validation/enumeration phase | 13.998 ms | 13.355 ms |
| Inclusive native runner phase | 65.999 ms | 65.820 ms |

Each stock preparation has 15 copies and three fences; each fused preparation
has 16 and four. The source hypothesis of fourteen descriptor copies plus two
descriptor fences common to both, and one extra fused route copy/fence, is now
supported by measured per-layer counts. The readback phase includes submission,
owner checks and preceding stream work; it is not pure device-copy time. Native
runner includes all the other native phases. These totals must not be added.
Pageable D2H APIs can block internally; the explicit fence's short API duration
does not mean the readback was free.

The union of GPU kernel/copy intervals is 27.595/27.384 ms in a
201.370/199.193 ms first-to-last GPU event span (stock/fused). The complementary
173.775/171.809 ms is measured absence of those GPU events in this instrumented
window. It does not isolate a CPU cause or establish clean utilization. CPU
model-forward ranges total 190.644/188.107 ms; preparation ranges are nested
inside them. The three head-end to next model-start gaps are 2.237–2.581 ms for
stock and 2.570–2.817 ms for fused; those intervals include head execution,
sampling, synchronization and scheduler work, rather than pure CPU idle time.
Profiler/CUPTI initialization and event recording perturb these observations.

Major summed GPU kernel durations over four steps include dense GEMV projections
(10.279/10.285 ms), grouped MoE GEMMs (4.513/4.511 ms), kernels correlated with
the logits-head ranges (3.744/3.746 ms), and attention (1.896/1.892 ms, comprising
100 TRT decode and twenty one-row FlashInfer kernels). Stock maps plus expansion
take 0.537 ms across 240 kernels; fused maps/expansion takes 0.279 ms across 120.
Original input quantization, descriptor setup, activation/requantization and
finalization kernels remain present. These diagnostic totals are not matched
clean performance estimates.

There are 244 measured `cudaStreamWaitEvent` enqueue calls per lane. Preparation
GPU work appears on stream 33; surrounding model/head work appears on stream 25,
with eight copies on stream 13. The source retains the producer-to-owned and
owned-to-caller waits. This trace does not expose enough wait operands to measure
device wait duration or prove each dependency edge from the trace alone.

The 24 correction-selection ranges per lane have no enclosing routed range.
This differs from the original protocol's expected enclosing relationship.
They occur after routed preparation; associating them with a preceding layer
from order is an inference, not independent layer/request identity. Their
aggregate count and duration can be reported; layer-specific claims are withheld.
Labels have not been repaired retroactively.

The narrow lean change introduces `prepare_capture_free`: one complete fresh
dynamic eligibility check followed immediately by dispatch, returning the
completed qualification/backend decision. The bridge uses it only with capture
and observer callbacks disabled. It accepts no checked flag, stores no reusable
eligibility capability and performs no callback before a supported fused launch.
The stock lane still checks eligibility so actual descriptor validation remains
qualified. Captures and observers retain the prior path. Unsupported successful
queries call the incumbent once; CUDA/driver errors propagate before mutation,
and candidate submission errors never retry stock. Quantizer, kernel math,
descriptor checks, retained owners, allocator lifetime and stream waits stay
unchanged. Unneeded observer status-string construction is also skipped.

For eligible fused production calls the expected reduction is exactly one
32-byte D2H copy and one stream fence per layer: thirty per token, or 120 across
four steps. Ten allocation/pointer queries and repeated stream/context checks
are also removed per call. Common descriptor checks are retained. CPU controls
compile the actual header, cover every query failure occurrence, unsupported
inputs, stock qualification, submission failure, and a changed route after a
previous qualified result. The new source has no GPU validation yet. A future
profile of the lean branch records its fresh check within `map_dispatch`, with
no separate `map_eligibility` entry; original diagnostic phase semantics remain.

The next GPU step requires a separately reviewed bounded correctness/dispatch
protocol and exact source freeze. It is not an automatic benchmark sweep or
graph restoration. A clean performance claim requires separate uninstrumented
matched evidence after the lean path is validated.

# Request-bound M1 live preparation

This milestone connects the PR9 preparation adapter to the installed model runner behind an explicit, default-off opt-in. The bounded stock/fused pair passed with **30 actual request-bound fused launches**, plus two separately labeled artificial route controls. Both paths retain the installed TMA setup, grouped GEMMs, activation and routing finalization. This establishes compatibility for the captured controlled cached request; natural routing, model quality, CUDA graphs and performance remain unqualified.

A subsequent [bounded normal-routing milestone](m1-normal-correctness.md) adds two fixed prompts with constrained continuation, repeated decode and scratch reuse. Its results and limitations are separate from the controlled evidence below.

The base is PR9 main `955832939062ac6b9e7bb2698b4181c1472225c3`. Its post-merge [CPU CI](https://github.com/yindavidyang/megartx/actions/runs/36926171758) and [numerical CI](https://github.com/yindavidyang/megartx/actions/runs/36926171754) succeeded before this isolated work began. [PR9 installed compatibility](m1-installed-compatibility.md), its source ledger, four synthetic sanitizer results and prior baseline evidence are preserved. [The source overlay](evidence/m1-live-source-pins.json) explicitly replaces only the changed plugin hash in the historical source test; it does not rewrite the earlier evidence.

Python 3.12.3, vLLM 0.30.0, FlashInfer 0.6.18.post1, Torch 2.13.0+cu130 and `nvidia/Gemma-4-26B-A4B-NVFP4` revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6` were retained. No installed package, checkpoint or system configuration was changed. Compilation used the documented task-local CUDA PATH/header configuration. The shared Desktop checkout was untouched.

The controller is enabled only by `--m1-preparation stock` or `--m1-preparation fused` in the bounded native, controlled, cached runner. An absent opt-in does not construct the controller, load its library or allocate its workspace. Startup fixtures are outside the marked request and cannot count as live hits. A marked model forward records actual token/position rows; a verified registered corrected routed adapter retains the exact layer, kernel, callable and all tensor owners before entering the native lease.

The live contract is ABI v2 with 15 owner views of 32 bytes each. The versioned `megartx_m1_begin_v2` checks version, explicit array count and struct size before dereferencing the array. It exports no historical uncounted begin symbol. A compiled contract embeds native, controller and plugin source SHA256 values. Admission checks the current controller/plugin files, binary/build receipt, installed identities, compiled contract and all four actual installed vtable/PLT relocations after ordinary Torch initialization. `LD_PRELOAD` is rejected. Historical mixed-version receipts fail before library loading; compiled poison-pointer tests establish rejection before an invalid array access. Nested rejection preserves the outer lease.

The exact installed FP4-to-FP4 expansion lane uses `ActivationType::Geglu` (4), swizzled SF, H=2816, F=704, E=128 and top-k=8. Original FC1 and FC2 activation globals are FP32 `[128]`; the per-expert flags are both true. The FP4 expansion branch copies packed values and valid SF bytes, so its FC1 per-expert flag does not introduce activation arithmetic. The new trusted lane identity permits that exact typed branch; an unverified per-expert lane still falls back. The original two weight tensors and all six typed quantization tensors are retained, passed unchanged and checked against the actual runner pointers. Neither global scales nor weight scales are rewritten.

Both tactics are installed SM120 TMA warp specialized `128x128x128`, cluster `1x1x1`, schedules 0/0, no epilogue fusion and `swap_ab=false`. The actual FP4 kernel tile is `128x128x256`. PDL and fused finalization are false in both matched lanes. The corrected runner's ordinary grouped call masks the route weights of its six separately corrected experts, then adds their original separate native projection outputs. The comparison checks this existing mask against the captured controlled routes and checks the corrected intermediates separately.

The installed helper yields a dedicated **3,185,408-byte** M1 owner. The bridge obtains reserved subviews from the installed workspace ledger, checks their capacities and binds the source map to the exact trailing FFI partition. The preparation/output operands are disjoint before preparation; immutable weight/quantization views may alias one another but cannot overlap mutable output/scratch.

| Actual subview | Workspace byte offset | Reserved bytes | Lifetime |
| --- | ---: | ---: | --- |
| Sorted-to-slot map | 0 | 128 | Preparation through routing finalize |
| Expert offsets | 1,792 | 1,152 | Preparation through both GEMMs |
| Shared packed activation owner | 2,944 | 45,056 | FC1 input; reused by installed activation for FC2 input |
| Shared GEMM output owner | 48,000 | 45,056 | FC1 output; later reused for FC2 output |
| Shared activation SF owner | 95,104 | 2,883,584 | FC1 SF; later overwritten by installed activation for FC2 SF |
| FC1 TMA metadata | 2,978,688 | 27,136 | Installed descriptor setup through FC1 |
| FC2 TMA metadata | 3,005,824 | 27,136 | Installed descriptor setup through FC2 |
| GEMM workspace | 3,032,960 | 152,320 | Original grouped GEMMs |
| Slot-to-sorted source map | 3,185,280 | 128 | Trailing FFI partition through finalize |

An additional retained 32-byte owner receives the candidate's unused permuted route weights. Finalization is separate, so this owner has no GEMM/finalize consumer. The live workspace plus this owner is 3,185,440 bytes. Producer work precedes the owned nondefault stream; every operand records its stream lifetime. The caller stream waits for owned work on success and on late submission/capture errors. Native invocation state and leases are cleared, the primary error survives cleanup failures, and a failed controller requires an owned process restart. There is no retry through stock after a CUDA query/submission error.

Successful unsupported CUDA queries delegate to the unchanged installed map/expansion path; query errors propagate before preparation capture, callbacks or candidate mutation. Unsupported model geometry/capture/modes delegate the full original call. In the matched request, each lane recorded **30 M=32 prefill stock fallbacks**. Admission identity/ABI errors fail closed. Runtime fallback does not relax an identity mismatch.

The observer reads the actual installed shapes, SF layouts, pointer and stride tables after unchanged TMA setup. It enumerates the actual CuTe SF domain, including padding: each active FC1 expert has a dense 128-row, 22,528-byte SF carrier; FC2 has 5,632 bytes. Captured AQ read/output write ranges use the actual pointer tables and strides. The masks qualify payloads in the preparation/shared workspace; unchanged immutable weight consumers are bound through the original typed quantization owners and installed source identities. Source-bound group scheduling assigns no payload tile to M=0 groups. **Inactive layout metadata can still be read**, including group 0 during `load_init`; the evidence does not claim all inactive metadata reads disappear.

Two opt-in shadow calls use the same actual input AQ/SF, original model weights/quantization owners and ordinary route-weight bits. They change only the explicitly artificial IDs/output/scratch: fresh workspace with experts 11–18, then reused workspace with disjoint experts 65–72. Neither set contains expert 0. Their captures are separate from the headline controlled request. Both workspaces, retained shadow, probe output and two ID arrays total **6,376,544 bytes**, within the existing 8 MiB aggregate scratch scope. The natural model routing is not qualified by this controlled fixture.

The compiler remained below 2 GiB aggregate RSS / 300 seconds, with 8 GiB host availability and 2 GiB GPU headroom required. The successful live binary compiled in 5.995 seconds with peak aggregate RSS 730,025,984 bytes. This is a compiler resource observation, not an execution speed measurement. Capture copies/synchronizations deliberately affect execution and cannot support a speed claim.

| Matched validation | Result |
| --- | --- |
| Request-bound M1 leases/launches | 30 stock vs 30 fused, all 30 layers at actual position 32 |
| Separately artificial first-use/reuse leases | 2 stock vs 2 fused; omitted-zero disjoint sets |
| GPU launch correlation | Each of 32 CPU preparation spans binds its launch on the owned nondefault stream |
| Candidate substitution | Fused: one candidate per span, zero installed map/expansion launches in those spans; stock: original map and expansion, zero candidate launches |
| Installed grouped GEMMs | Two per span, 64 per lane; all remaining incumbent kernel identities/counts match |
| Independent byte oracle | All 32 maps, 129 offsets, 11,264 expanded AQ bytes and full 2,883,584-byte SF owner pass, preserving untouched padding |
| Matched preparations/physical masks/routed outputs | Bit exact across all 32 calls |
| Matched model captures | 99 NPZ files, 471 arrays, 1,306,330 elements bit exact: routes, six corrected expert captures, raw logits and 30 KV snapshots |
| Full stock prefill fallback | 30 calls per lane |
| Lease/dependency cleanup | All 64 leases released; producer and caller waits recorded; both owned server lifecycles completed |

The successful binary SHA256 is `10f1cb4fdd9ff5c1cf503213d520d8e46b746e76abb159f780fa694cfb611598`. [The sanitized scalar report](evidence/m1-live-matched.json) contains the exact compiled source contract and counts. Private SHA-manifest packets retain original build receipts, binary, generated contract/header, installed sources, all raw captures/traces and intermediate failed attempts. Earlier preload failures and the rejected zero-row carrier assertion are retained, not counted as successful execution. A later build-wrapper TypeError produced no admission receipt; the corrected argv helper has a CPU regression and the successful build passed actual binding controls.

Reproduce only in the preserved pinned environment with an owned adapter site copied from the same source as the build:

```sh
python scripts/build_m1_live_bridge.py --flashinfer-root "$FLASHINFER_ROOT" --cache "$FUSED_MOE_CACHE" --output "$BRIDGE_BUILD" --base-head "$BASE_HEAD"
python scripts/run_scale_validation.py --mode native --client controlled --controlled-plan "$CONTROLLED_PLAN" --controlled-path cached --m1-preparation stock --m1-bridge "$BRIDGE_BUILD/m1_live_bridge.so" --m1-build-receipt "$BRIDGE_BUILD/build.json" --m1-route-controls --label m1-stock
python scripts/run_scale_validation.py --mode native --client controlled --controlled-plan "$CONTROLLED_PLAN" --controlled-path cached --m1-preparation fused --m1-bridge "$BRIDGE_BUILD/m1_live_bridge.so" --m1-build-receipt "$BRIDGE_BUILD/build.json" --m1-route-controls --label m1-fused
PYTHONPATH=numerical_reference python numerical_reference/compare_m1_live.py --build "$BRIDGE_BUILD" --stock "$STOCK_RUN" --fused "$FUSED_RUN" --output "$NEW_REPORT"
```

The comparison reads bounded numeric captures without pickle, requires positive per-request launch correlation, rejects incomplete/failed/mixed-mode leases and compares against the retained compiled contract. Its CPU negative controls reject zero-hit traces, uncorrelated global kernel names, duplicate launch correlations, missing parents, extra stock preparation, wrong streams and graph-mode receipts. The scaffold and numerical suites pass 52 and 426 tests respectively. PR9 preparation-level runtime query/submission fault controls remain covered; this milestone does not claim full-model sanitizer coverage.

Python 3.10 CI exposed a hashing API added in Python 3.11. Admission now uses bounded 1 MiB SHA256 streaming, with an API-absence/multiple-block regression. The bridge was rebuilt and the complete bounded stock/fused pair repeated with the new controller source contract; all results above describe that replacement pair. The original successful pair and failed CI logs remain retained.

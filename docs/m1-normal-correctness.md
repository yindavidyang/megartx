# Bounded normal-routing M1 correctness

The default-off live M1 preparation passed a matched stock/fused pair on two fixed prompts with unchanged router selection and a constrained continuation. Each lane completed **210 request-bound M1 calls**, **150 full-stock prefill fallbacks**, and two owned server lifecycles with cleanup. This extends [PR10's controlled request](m1-live-preparation.md) from main `cc82e59c8053a72af37d9e06a76405d1ef31181e`; earlier evidence remains unchanged.

The coverage is deliberately narrow. Continuation was restricted to the declared non-EOS token 506 (` the`), with four output tokens per request. No router IDs or original route weights were injected. **None of the 210 M1 rows selected a separately corrected expert**, so the pair provides no new positive natural-routing coverage of the six rare correction targets. PR10's controlled expert captures remain separate evidence. Model quality, unconstrained generation, broad prompt coverage, CUDA graphs, asynchronous scheduling and performance remain unqualified.

## Exact admitted plan

The plan digest is `34218e21ec9bae675fbd0cddec2901496de14aad098225777d479ea86271f3fe`. The collector admits exactly two marked requests, four outputs each, a 256-token prefill chunk and these model frames. The actual prompt token IDs and expected continuation are retained in the private plan packet. Rehashing an expanded length or output count does not bypass the admission contract.

| Request | Prefill row counts | Cached decode input positions | M1 calls | Full-stock fallbacks |
| --- | --- | --- | ---: | ---: |
| `split257` | 256, 1 | 257, 258, 259 | 120 | 30 |
| `window1023` | 256, 256, 256, 255 | 1023, 1024, 1025 | 90 | 120 |

The single input row at position 256 is a **prefill tail**, contributing 30 M1 calls. It is distinct from the 180 cached-decode calls. The second request crosses the 1024-position context boundary. Across both requests there are 12 actual model forwards and 360 routed layer calls per lane. The higher 210-call cap is admitted only with this validated plan; the existing controlled cap remains 64. Artificial route controls are forbidden in this mode, and missing, extra or reordered frames fail closed.

The normal runner explicitly adds `--no-async-scheduling`, keeps eager execution and serves only localhost with one sequence. In retained asynchronous attempts, after the first 256-row prefill the callback exposed position 0 and the first prompt token where position 256 was expected. The collector rejected that frame before any M1 call. A CUDA fence and independent CPU copies did not resolve it. The cause remains unresolved; these attempts do not qualify asynchronous scheduling. Synchronous scheduling completed the exact original plan without changing the prompts or counts.

## Captures and comparison

The route observer records and returns the exact original router tensor objects. The preparation comparator checks the corrected runner's existing ordinary-weight mask against those original routes. Both lanes retain the same installed weights, typed quantization owners, TMA descriptors, grouped GEMMs, activation and finalization. The live ABI remains v2 with 15 counted owner views. The compiled source contract now includes the normal plan, normal collector and shared physical KV collector along with the controller and plugin. [The source overlay](evidence/m1-normal-source-pins.json) extends the historical ledger without rewriting it.

This document and its source overlay are frozen evidence for the PR12 head. The
combined reconciliation branch preserves these historical packets unchanged
and binds its reconciled controller/native sources in a separate
[m1 reconciliation ledger](evidence/m1-reconciliation-source-pins.json). No
result in this packet alone qualifies the combined head for GPU, quality,
graph or performance use.

The shared KV helper checks the actual registered attention owners, installed writer source hashes, current metadata writer slots, physical layout, capacity and BF16 bits. It snapshots the last actual writer row of every layer immediately after each forward, before later sliding-window recycling. These 360 snapshots per lane compare actual installed writer outputs; they do not independently qualify attention arithmetic or scheduler behavior, and do not claim that historical slots remain resident. Raw full-vocabulary logits are captured before the allowed-token sampler at the last prompt row and three cached input rows per request. Earlier partial-prefill head rows are excluded explicitly.

| Matched validation | Result |
| --- | --- |
| Preparation oracle | All 210 maps, 129 offsets, expanded AQ and whole SF owners pass bit exact, including untouched padding |
| Preparation payloads, physical masks and routed output | Bit exact across 210 paired calls |
| Model arrays | 938 paired NPZ files, 3,324 arrays, 5,326,728 elements bit exact: 360 routes, 210 routed outputs, 360 immediate KV snapshots, 8 raw-logit rows |
| Candidate launch correlation | Fused: exactly 210 request-bound launches, one per preparation span on the owned nondefault stream; stock: zero candidate launches |
| Installed preparation substitution | Fused preparation spans contain zero installed map/expansion launches; stock spans retain their original map and expansion |
| Installed grouped GEMMs | Two per M1 span, 420 per lane; remaining incumbent kernel identities/counts match |
| Scratch reuse | Same workspace storage owner across all 210 calls and both requests; first-use followed by 209 reuse receipts |
| Additional M1 scratch | 3,185,408-byte workspace plus 32-byte retained shadow, **3,185,440 bytes** total, below 8 MiB |
| Corrected expert stages on M1 rows | Zero observed; no positive natural correction-target claim |

The [sanitized paired report](evidence/m1-normal-matched.json) records the exact compiled contract and binary SHA256 `5c83c07d96b30eee6299d53a53a137b1e9fa46c4ad3343c5a6b436cacd453550`. The independent comparison requires successful client/server exit, complete owned cleanup, exact source and plan identities, bounded numeric archives without pickle, per-request CPU-to-GPU launch correlation, and the full admitted record counts. CPU negative controls reject expanded plans, altered identities, zero-hit/missing/extra launches, changed counts, malformed numeric captures and partial-prefill logits misclassification.

## Disposable CUDA errors

After model cleanup, a fixed synthetic child matrix used the unchanged installed preparation header and pinned CUDA/FlashInfer dependencies. Each child owned its context and allocated at most 2,919,976 device bytes, loaded no model, and terminated within its 30-second bound. The [sanitized error report](evidence/m1-normal-cuda-errors.json) binds the probe, runner, native sources and binary.

| Child control | Observed result | Stock callbacks |
| --- | --- | ---: |
| Supported preparation | Candidate completed and wrote the expected map | 0 |
| Real illegal address | One-thread null write poisoned only the child context, reporting CUDA status 700; actual `prepare` propagated `cudaStreamIsCapturing: an illegal memory access was encountered` | 0 |
| Cleared current context | Actual `cuCtxGetDevice` returned driver status 201; the production checked driver helper propagated the original invalid-context error; no preparation invocation or device allocation | 0 |

After an error, these children perform no CUDA buffer copy, synchronization, free, retry or destructor cleanup; process exit releases the child's resources. This is synthetic preparation/query and driver-helper coverage, not full-model fault injection or sanitizer coverage. Python controller regressions also verify that producer-wait failure invalidates the controller and clears layer state, that the original failure survives cleanup errors, and that later routed/direct calls cannot fall through to stock or reuse its buffers.

An exploratory destroyed-stream child crashed inside the driver's stream query with signal 11. The runner detected the failed child, stopped the matrix and verified GPU cleanup. Its source, stderr and failed report are retained, but this is **not** passing error-propagation evidence. Destroyed handles are excluded from the delivered matrix. An earlier link failure occurred before any child ran and is also retained.

## Environment, resources and reproduction

Python 3.12.3, vLLM 0.30.0, FlashInfer 0.6.18.post1, Torch 2.13.0+cu130 and `nvidia/Gemma-4-26B-A4B-NVFP4` revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6` were preserved. All 17 installed source/binary pins remained unchanged. No package, checkpoint or system configuration was changed; the shared Desktop checkout was untouched.

Compiler limits remained 2 GiB aggregate RSS / 300 seconds, host availability 8 GiB, GPU headroom 2 GiB, and additional owned M1 scratch 8 MiB. The successful model bridge compiled in 6.038 seconds with peak aggregate RSS 725,291,008 bytes; the safe synthetic control binary compiled in 5.393 seconds with peak 672,387,072 bytes. These are compiler resource observations. Capture copies, fences and profiler instrumentation cannot support an execution speed claim. All owned model/compiler/control processes stopped; fresh NVML checks showed no compute PIDs and 41 MiB GPU usage after cleanup.

Only reproduce in the preserved pinned environment, using an owned adapter site from the same source as the build and the retained exact plan:

```sh
python scripts/build_m1_live_bridge.py --flashinfer-root "$FLASHINFER_ROOT" --cache "$FUSED_MOE_CACHE" --output "$BRIDGE_BUILD" --base-head cc82e59c8053a72af37d9e06a76405d1ef31181e
python scripts/run_scale_validation.py --mode native --client normal --m1-normal-plan "$NORMAL_PLAN" --m1-preparation stock --m1-bridge "$BRIDGE_BUILD/m1_live_bridge.so" --m1-build-receipt "$BRIDGE_BUILD/build.json" --label normal-stock
python scripts/run_scale_validation.py --mode native --client normal --m1-normal-plan "$NORMAL_PLAN" --m1-preparation fused --m1-bridge "$BRIDGE_BUILD/m1_live_bridge.so" --m1-build-receipt "$BRIDGE_BUILD/build.json" --label normal-fused
PYTHONPATH=numerical_reference python numerical_reference/compare_m1_normal.py --build "$BRIDGE_BUILD" --stock "$STOCK_RUN" --fused "$FUSED_RUN" --plan "$NORMAL_PLAN" --output "$NEW_REPORT"
# Only after both owned model lifecycles and GPU cleanup:
python scripts/run_m1_cuda_error_controls.py --build "$BRIDGE_BUILD" --output "$NEW_DISPOSABLE_CONTROL_DIR"
```

Private SHA-manifest packets retain build receipts and binaries, all installed pin snapshots, raw buffers and numeric captures, per-request traces, exact prompt/continuation plan, lifecycle/resource records, all intermediate failed attempts, and source snapshots. Public evidence contains sanitized scalar summaries. The scaffold and numerical suites pass 59 and 433 tests respectively; no timing, graph, quality or asynchronous result is promoted by those tests.

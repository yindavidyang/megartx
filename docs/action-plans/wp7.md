# WP7 Optimize full-prompt prefill

[Master plan](../master-plan.md#wp7--dedicated-full-prompt-prefill-optimization) · [Action plan index](README.md) · [Prefill design](../design/megakernel-design.md#11-full-prompt-prefill-is-a-separate-optimization-regime)

Status: planning and public draft PR publication authorized; implementation and GPU experiments await separate scope approval. WP7/G7 and A7.1–A7.6 are appended stable IDs. Existing WP0–WP6, G0–G6 and their 28 tasks are unchanged. Decode-first remains the active milestone; WP7 branches from a qualified WP1 baseline without requiring wider persistence or drafting.

The owner separately approved an isolated documentation commit/push and public draft PR on 3 October 2026, as relayed by the parent. Merge, SSH, GPU experiments, package installation, model download, native builds and runtime changes remain outside this planning task. PR15 benchmark remediation is separate; `01a10079-c8d1-752e-991a-4e76b523704e` remains the sole GPU owner.

## Evidence and workload boundaries

The [M1 normal-routing evidence](../m1-normal-correctness.md) includes single-row prefill tails and full-stock multirow fallbacks; it does not qualify a full-prompt prefill optimizer. The [PR14 lifecycle evidence](../evidence/m1-pr14-gpu-lifecycle.md) validates its bounded invocation and owner lifecycle. Neither establishes a speedup, broad G1 quality acceptance, CUDA graphs or asynchronous scheduling. Existing G0–G6 and new G7 are unaccepted. PR15 remediation cannot be treated as an accepted baseline before its independent review.

Start with full 2048- and 8192-token prompts, one active request, no dynamic request batching, selected checkpoint/tokenizer, explicit BF16 KV and separately reserved 256-token output capacity. Admit 32768-token prompts only after measured resident peak fit, including output reserve. Chunking creates multiple forwards within the same request; prompt rows are not concurrent requests. Include partial final chunks and supported context buckets.

| Regime | Work and reuse to measure | Timing and state boundary |
| --- | --- | --- |
| WP2–WP4 target-only decode | One new row, eight distinct M=1 expert problems; growing KV reads | ITL and full emitted-token cost; I03 committed cache |
| WP7 full-prompt prefill | Many rows; dense GEMMs and variable per-expert row groups; attention/KV growth and chunk working set | Whole-prompt processing latency and prompt tokens/s, handoff and warm TTFT; I03 exact prompt cache |
| WP5 short DSpark verifier | Anchor/proposal rows over an existing cache; route union and rejected work | Draft/verify/accept/cache cost per emitted token; I06 tentative commit/rollback |

Many rows can improve same-expert weight reuse and compute utilization, but route skew, empty/tiny groups, padding, packing and load imbalance can erase that benefit. Measure the per-layer/per-chunk expert occupancy histogram, useful versus padded work and traffic. Do not infer these costs from M1 or a short verifier. Reuse an attention/GEMM/epilogue kernel only when its shape, lane, quantizer and cache contracts match; qualify each regime separately.

## Resource and ownership bounds

| Resource | Bound and admission rule |
| --- | --- |
| Current planning scope | Local Markdown edits and CPU docs/config checks only; no device probe or GPU import needed |
| CPU parallel preparation | Two proposed preparation lanes: workload/oracle/cache-fixture review (A7.1) and source/shape/candidate review (A7.3/A7.4). They may overlap using frozen inputs and separate output paths. Admit aggregate host memory, per-job timeout and artifact volume from available resources first; serialize heavier checks when those bounds are unknown. No dependency installation or shared environment mutation |
| GPU ownership | Exactly one GPU experiment owner: `01a10079-c8d1-752e-991a-4e76b523704e`. At most one admitted model/profiler/benchmark job at a time, one active request; all other tasks queue and do not SSH, launch, kill or reconfigure its work |
| GPU priority and lifecycle | Decode-first/PR15 owner's approved work has priority. A future WP7 slot requires that owner's explicit handoff, separate experiment approval, pinned head/environment, bounded workload/run count/wall time, owned-process receipts and cleanup before the next job |
| Memory | Record allocated, reserved and device peaks across load, compile/warmup, prefill chunks, conversion, first token and continuation; include KV/output capacity, quantization scales, dispatch, scratch, fallback and graph pools if admitted. D07's 2 GiB reserve is proposed pending freeze. No CPU weight offload, whole-model BF16 expert expansion or all-prompt all-expert materialization |
| Build and scratch | Existing M1's 2 GiB aggregate compiler RSS / 300-second build limit and 8 MiB extra M1 scratch cap remain scoped to M1. They neither authorize a WP7 build nor prove prefill fit. Freeze WP7 compiler/time/scratch bounds from its measured shapes before any future build/run; an unknown bound blocks admission |
| Timing isolation | No concurrent GPU job, profiler, compilation or unrelated host load during paired timing. Profile in separate serialized runs. Bound retained trace bytes and keep private tensors/prompts outside source control |
| Abort/recovery | Stop on the first semantic/safety/admission violation. Drain and release only owned work; uncertain or partly committed cache is poisoned and rebuilt through the verified fallback. Retain failure receipts and check cleanup before another slot |

The resource table defines planning requirements, not new hardware authorization. CPU work may continue while GPU work is blocked. No dates, speedup results or new numeric acceptance thresholds are asserted.

### A7.1 Freeze prefill experiment inputs and admission

WP7 R01 R02 R04 R05 R07 R08 R09 R12 R13 I01 I02 I03 I05 | Owner: benchmark owner with model/correctness lead | Entry: selected source/contract available; CPU planning may start now

Inputs: D04–D07, I01–I03, WP0/WP1 gate state, retained M1 evidence, current decode control and the sole owner's resource queue.

1. Prepare a workload matrix for 2K/8K and conditional 32K, exact prompt/tokenizer/template hashes, output/EOS policy, numerical lane, BF16 KV, cache layout/capacity and candidate chunk/context buckets. A chunk sweep stays within the same prompt and model semantics; declare any unsupported cell explicitly.
2. Register the authoritative BF16 semantic reference, the selected quantizer-lane oracle and tuned compatible FlashInfer-backed incumbent. Original weight preservation alone does not make the runtime activation quantizer equal to the checkpoint quantizer. W4A16 remains separately labeled with its own oracle; no silent lane change or altered scale domain is permitted.
3. Define timestamp locations and pending freeze decisions for useful prefill gain, latency/p95/TTFT, peak memory, quality and decode regression guards. Keep D07's margins proposed. Numerical tolerances come from the oracle before tuning; no tolerance is relaxed to pass a candidate.
4. Prepare experiment/run/build/trace bounds and CPU ownership lanes. List missing G0/G1 evidence, runtime graph eligibility and future scope approval. Preserve PR15 remediation as a separate input dependency, with no presumed result or execution slot.

Deliver (proposed future paths): `configs/prefill_workload.json`, `configs/prefill_sweep.json`, `docs/gates/G7-intake.md`, `results/wp7/input_manifest.json`. These files are specifications until actually delivered; this local revision does not create runtime configuration.

Exit and review: independent reviewer checks inputs, contract identity, resource bounds and oracle/tolerance rationale; owner freezes the proposed acceptance criteria and separately approves any GPU scope. G0/G1 must be accepted and the non-speculative decode control qualified before A7.2 execution.

If blocked: retain a CPU-only intake packet with explicit missing inputs. Neither unchanged M1 captures nor an available GPU slot passes G1 or authorizes tuning.

### A7.2 Profile the tuned full-prompt incumbent

WP7 R03 R07 R08 R09 R13 I03 I05 | Owner: benchmark owner | Entry: A7.1 frozen; G0/G1 accepted; qualified decode control; sole-owner GPU slot and experiment approval

Inputs: same-lane fastest correct supported incumbent, input manifest, admitted workload/run bounds and [paired measurement protocol](../protocols/measurement.md#full-prompt-prefill-measurements).

1. Establish whole-prompt 2K/8K latency, prompt tokens/s and memory timelines. Separate tokenization/host input, prompt GPU execution, per-chunk overhead, KV conversion/handoff, first-token head/sampling/streaming, warm TTFT and cold load/compile. Sum all chunks; do not report a favorable chunk as whole-prompt throughput.
2. Profile separately: SM120 attention by local/global layer and chunk, QKV/output/dense/shared projections, router, activation quantization/packing, dispatch/gather/scatter, grouped FC1/GELU/FC2/combine, norms, logits and synchronization. Record kernel identities, useful work, attainable traffic/compute, occupancy/spills and critical-path overlap.
3. Save actual per-expert M distributions, route skew, empty groups, weight reuse, tails and scratch lifetimes at each chunk. Routes saved for microbench fixtures do not replace live routing in full-model timing.
4. Admit 32K only after a bounded fit review covering peak prompt activations, local/global KV, scratch, output capacity and frozen reserve. Mark it deferred if fit is unknown or fails; do not treat an unsupported/OOM incumbent as a candidate win.
5. Rank recoverable costs and bound benefit with measured stage fractions and dependencies. Select the smallest chunking/library/kernel experiment with enough headroom; record stop conditions before implementation.

Deliver (proposed): `results/wp7/incumbent.jsonl`, `results/wp7/prefill_ledger.csv`, `results/wp7/expert_shapes.json`, `results/wp7/memory_timeline.csv`, `results/wp7/traces/manifest.json`, `docs/hypotheses/H-prefill.md`.

Exit and review: reproducible 2K/8K baseline, actual dispatch and memory records, separate 32K disposition and attributable hotspot justify A7.3/A7.4. A correct incumbent must be tuned for prefill as well as decode.

If blocked: stop GPU tuning at the first compatibility, numerical or fit gap. Keep independent CPU analysis possible and retain the verified incumbent.

### A7.3 Tune chunking and SM120 attention

WP7 R01 R05 R06 R07 R09 R13 I02 I03 I05 | Owner: runtime/kernel engineers | Entry: A7.2 attributes chunk/attention cost; bounded candidate matrix approved

Inputs: local/global attention shape traces, frozen mask/RoPE/normalization contract, chunk/capacity limits, supported pinned SM120 backends and cache fixtures.

1. Tune supported incumbent chunk sizes/context buckets before a custom kernel. Compare unchunked prefill where it fits with bounded chunks and partial tails; include scheduler/launch cost and peak activation/KV/scratch memory. Hold prompt, absolute positions and numerical lane fixed.
2. Sweep supported SM120 attention layouts/tiles/schedules separately for 25 sliding and 5 global layers, with their head geometry and explicit BF16 KV. Preserve causal and sliding visibility, unit score scale, Q/K/V normalization and proportional/global versus local RoPE. Do not import SM100 TMEM/tcgen05 or TPU storage assumptions.
3. Test chunks crossing 1023/1024/1025, page/ring wraps and prompt/output capacity edges. For a multirow chunk, prove every row can see its required prior keys before window recycling; retaining only the final window before all rows finish is invalid. Ring addresses never substitute for absolute RoPE positions.
4. Compare continuous versus chunked teacher-forced intermediates/logits and final all-layer cache state under the frozen numerical envelope. Measure any conversion, padding, mask metadata or layout cost; retain the incumbent attention on unsupported shapes.

Deliver (proposed): `results/wp7/attention_chunk_sweep.csv`, `results/wp7/attention_errors.json`, `tests/fixtures/prefill_attention_manifest.json`, `docs/wp7_attention_decision.md`.

Exit and review: a correct supported chunk/attention choice improves the measured full-prompt boundary with safe peak fit and exact I03 handoff. Passing a short attention microbenchmark alone does not promote it.

If blocked: reject the candidate or narrow dispatch to qualified shapes. Cache/mask/position errors block all affected integration and are not traded for throughput.

### A7.4 Tune dense projections and grouped expert GEMMs

WP7 R01 R04 R05 R06 R07 R13 I01 I02 I05 | Owner: kernel engineer with model/correctness lead | Entry: A7.2 real shapes and numerical contracts available; separate bounded GPU slot

Inputs: dense QKV/output/shared/head shapes, per-expert row histograms, packed weights/scales, exact GELU/quantizer/cast/reduction contract and supported SM120 libraries.

1. Tune dense projection and shared-branch GEMMs for actual chunk M, including layouts, tile/stage choices, projection sharing, tails and epilogues. Preserve precision and tied-head semantics; record whether logits are needed only at the last prompt row or for a separate teacher-forced quality run. Compare identical output policy.
2. Tune grouped expert FC1/FC2 for the measured variable-M histogram rather than padding every expert to full prompt length or treating all rows as one shared expert. Include routing, gather/scatter, quantize/pack, gate/up scale segmentation, GELU tanh, expert weighting/reduction, empty/tiny groups and load balance.
3. Check useful versus executed work and 704/2112-channel boundaries, scale-layout tails and scratch peaks. Same-expert rows may reuse weights; different expert groups may not. Do not merge distinct calibration/quantizer domains to improve reuse.
4. Compare against the matched tuned library path with preprocessing/epilogue/synchronization included. Use pinned compiled resource/dispatch records, lane-specific operator fixtures and full-prompt live-route trials. Microbenchmarks explain a mechanism; integrated latency decides retention.

Deliver (proposed): `results/wp7/projection_sweep.csv`, `results/wp7/grouped_expert_sweep.csv`, `results/wp7/operator_errors.json`, `results/wp7/build_resources.csv`, `docs/wp7_operator_decision.md`.

Exit and review: oracle-conformant, supported, memory-safe candidates survive real route skew/tails and full-prompt integration. W4A16 results cannot inherit the W4A4 claim. CPU/source preparation may overlap A7.3; their GPU runs remain serialized and each uses a frozen control.

If blocked: keep the incumbent operator, identify the first error or integration overhead, and update the shape/cost ledger before proposing broader work.

### A7.5 Evaluate measured hotspot fusion and exact handoff

WP7 R05 R06 R07 R09 R13 I02 I03 I04 I05 | Owner: runtime/kernel engineers; independent safety reviewer | Entry: A7.3/A7.4 decisions and residual ledger identify a recoverable cost

Inputs: smallest measured fusion hypothesis, buffer liveness map, accepted operator/chunk contracts, committed-cache control and verified fallback.

1. Propose only a residual hotspot fusion justified by the new profile, such as norm/quantization, GEMM/activation/quantization or combine/norm/residual. Preserve the reference's quantizer domains, casts, scale ownership and reduction order. If tuned library paths already win, deliver a decision to stop fusion.
2. Define every supported M/shape, scratch owner/lifetime, register/shared-memory use, bound and synchronization/progress argument. The M1 single-row map/expansion fusion is not a multirow prefill design; extending its admission checks alone proves neither semantics nor efficiency. Do not generalize its scratch/ABI receipts or graph eligibility.
3. Validate operators before timing; retain sanitizer and bounded stress evidence for new device work. Test same-lane full versus chunked prefill→decode across chunk/window/page/capacity boundaries, distinct K/V, absolute positions, commit lengths and output reserve. Include conversion/transaction costs in A/B timing.
4. Exercise unsupported-shape fallback, EOS/cancellation and failures before/during handoff. Commit only complete valid prompt state; partial writes cannot be reused. Restart uncertain state through the verified incumbent using the committed token log. Keep current eager fallback until separately reviewed graph admission exists.

Deliver (proposed): `docs/wp7_fusion_decision.md`, `results/wp7/fusion_ablation.csv`, `results/wp7/handoff_checks.json`, `results/wp7/safety_reports/manifest.json`, `docs/wp7_fallback_contract.md`.

Exit and review: retain only an integrated correct/safe gain after all costs, or record a valid library-only outcome. Any cache/safety/numerical failure blocks affected promotion. An inconclusive gain retains the control and states the missing evidence.

If blocked: restore the smallest verified boundary; a wider persistent prefill engine requires a new scope and safety review, not escalation by default.

### A7.6 Integrate and review G7

WP7 R01–R09 R13 I01 I02 I03 I05 | Owner: benchmark owner; independent reviewer | Entry: A7.3–A7.5 dispositions complete, A7.1 criteria frozen and sole-owner integration slot separately approved

Inputs: pinned candidate/incumbent/decode heads, all operator/cache/safety records, admitted workload/resource matrix and raw paired trials.

1. Run the [shared paired protocol](../protocols/measurement.md) with matched checkpoint, quantizer lane, prompt, tokenizer, BF16 KV, sampler/output policy and environment. Warm supported buckets; randomize order; use the existing initial trial protocol and continue or mark inconclusive when uncertainty cannot resolve the frozen criteria. Keep profiler/observer captures out of timing.
2. Report whole-prompt p50/p95 processing latency, prompt tokens/s, GPU time, handoff, warm TTFT and full response time; cold load/compile separately. Include all lifecycle peaks and a 32K pass/defer reason. No omitted conversion, packing, first-token or host cost may become a claimed end-to-end gain.
3. Run matched continuation controls: accepted decode path after incumbent versus candidate prefill, same committed positions, output reserve and sampling. Report teacher-forced/quality results, p50/p95 ITL, long-run stability, decode dispatch and memory. Hold decode implementation fixed first; attribute combined prefill/decode changes only in separately labeled ablations.
4. Review frozen useful-gain, uncertainty, latency/tail/TTFT, quality, memory and decode regression guards together with exact semantics and fallback. A microkernel gain with no full-prompt benefit fails G7. Unsupported cells retain verified dispatch; no compatibility failure counts as a speed win.
5. Keep WP5's short verifier and I06 cache transactions outside this prefill headline. A shared winning kernel requires a separate WP5 route-union/acceptance/rollback benchmark; G7 does not authorize DSpark, data acquisition or draft training.

Deliver (proposed): `results/wp7/paired.jsonl`, `results/wp7/decode_regression.jsonl`, `results/wp7/ablation.csv`, `docs/gates/G7.md`, reproducible configuration and raw-artifact hash manifest. Gate packet paths are future outputs, not observed results.

G7 decision: proceed with the qualified per-workload prefill dispatch; retain the incumbent/library-only choice; rerun a specified missing measurement; request a concrete scope change; or stop. The independent reviewer checks reproducibility, safety, lane fidelity, handoff, uncertainty and all guards before acceptance; the project owner approves material contract/scope changes. Record the decision under D11 without rewriting prior evidence. Passing G7 neither passes decode G4 nor speculation G5, and does not retire the active decode-first milestone.

# WP2 Design low M expert experiments

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
### A2.1 Capture real selected expert shapes

WP2  R01 R04 R05 R07  I01 I02  |  Owner: Kernel engineer  |  Entry: G1 passed; expert work justified

Inputs: Actual routed FC1/FC2 shapes, pack layouts, live route traces and byte ledger.

1. Enumerate FC1, gate/up organization, FC2, strides, scale tiles, alignment and tails from the manifest. Batch one with top 8 is eight independent M=1 problems with different expert weights, not one shared-weight M=8 GEMM.

2. Create representative shape/route fixtures including skew, repeated selections across verification rows, quantizer boundaries and extreme inputs. Use recorded routes for microbenchmarks only; full-model timing must route live.

3. Estimate weight/scale traffic, launch/reduction overhead, padding waste and register/shared-memory demand. Add M=2/4/8 verifier probes only as conditional later-work shapes; measure route-union growth.

Deliver: bench/expert_shapes.json; tests/fixtures/expert_manifest.json; docs/wp2_candidate_matrix.md.

Exit and review: Reviewer approves real-shape coverage and separate numerical lanes before selecting implementations.

If blocked: If real shapes/layouts differ from the assumed kernel, revise the matrix; do not transpose/pad away semantics without accounting for its cost.

### A2.2 Choose supported SM120 candidate paths

WP2  R01 R05 R07  |  Owner: Kernel engineer  |  Entry: A2.1 candidate matrix complete

Inputs: SM120 support evidence, compiler resource reports and supported upstream examples.

1. Compare a native SM120 block-scaled MMA path with unpack/GEMV and same-weight W4A16 controls. Select CUDA/CUTLASS/CuTe or another language only when its pinned path exposes the needed instructions and format.

2. Use SM120-supported instructions and layouts. Do not carry over SM100 tcgen05/TMEM or a TPU memory schedule. Pin any reused example source, license and adaptation notes; test build and format support before optimization.

3. Create bounded candidate sweeps over tile/layout/schedule choices. For each, include activation quantization, scaling, packing amortization, epilogue and tail cost; record compile resource use and estimated occupancy.

Deliver: kernels/sm120/candidate_registry.json; bench/expert_sweep.yaml; results/wp2/build_resources.csv.

Exit and review: Reviewer retains only supported candidates with a testable performance mechanism and no hidden lane substitution.

If blocked: If native FP4 support or useful utilization is absent, record the limitation and compare separate alternatives. A different lane cannot inherit the native-FP4 headline.

## WP2 Validate and integrate the MoE winner

### A2.3 Validate low M operators and epilogues

WP2  R05 R06 R07  I02  |  Owner: Kernel engineer  |  Entry: A2.2 supported candidates build

Inputs: Frozen tolerances, pack fixtures, oracle and actual-shape sweep.

1. Validate pack/unpack, scale addressing, FC1 and FC2 independently. Then add the Gemma gated GELU and any expert weighting/reduction epilogues. Check padded regions, alignment, large values, zero blocks and scale-boundary inputs.

2. Run correctness before timing; capture sanitizer reports, error metrics, spills and occupancy for each variant. Measure warmed microbenchmarks with the same live input quantization costs and scratch allocation policy.

3. Repeat favorable candidates under different route patterns and report where they lose. Keep exact source/config hashes and raw repetitions; discard wins explained by omitted preprocessing, cached-only weights or a numerical-lane change.

Deliver: tests/test_expert_ops.py; results/wp2/operator_errors.json; results/wp2/operator_times.csv; results/wp2/sanitizers.txt.

Exit and review: Reviewer accepts only oracle-conformant, memory-safe operators with reproducible benefits on relevant shapes.

If blocked: Investigate the first failing input or scale block. A fast isolated GEMM does not advance G2.

### A2.4 Integrate live routing and review G2

WP2  R02 R03 R05 R07 R08  |  Owner: Runtime engineer  |  Entry: A2.3 correct operators; graph fallback retained

Inputs: Candidate operators, reference router/shared branch and I03 cache runner.

1. Integrate router, dispatch, activation quantization, expert FC1/activation/FC2, weighted combine and scratch lifecycle. Keep one request and all steady-state buffers resident/preallocated.

2. Benchmark the complete MoE block against tuned compatible FlashInfer in the same numerical lane. Include live routing, chosen weights/scales, synchronization, shared branch and reductions. Then measure the full model to expose displaced bottlenecks.

3. Run golden layer/logit checks and paired workload measurements. Recompute the token ledger and choose whether to retain this path, investigate a narrower change, or stop custom expert work.

Deliver: runtime/moe_backend.py; results/wp2/integrated.jsonl; results/wp2/ablation.csv; docs/gates/G2.md.

Exit and review: G2: frozen operator/router tolerances pass and integrated MoE wins with all costs included. Reviewer records full-model impact before WP3.

If blocked: If integration erases the microbenchmark win, keep the baseline. Diagnose dispatch, reduction, quantization or bandwidth; widen fusion only if evidence identifies recoverable cost.

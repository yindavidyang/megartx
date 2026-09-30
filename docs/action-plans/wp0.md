# WP0 Freeze the contract and host evidence

[Master plan](../master-plan.md) · [Action plan index](../action-plans/README.md)
### A0.1 Authorize and freeze the experiment

WP0  R01 R02 R04 R08 R12  |  Owner: Project owner  |  Entry: Implementation authorization and chosen host/repository available

Inputs: Master contract 1.0; intended prompts, output lengths and integration surface.

1. Choose exact checkpoint, immutable revision, tokenizer and chat template; define one-request workload, prompt/output limits, sampling and EOS policy. Record all accepted choices and unresolved items before performance comparisons.

2. Accept or revise the proposed text-only 2K/8K probes, conditional 32K probe, BF16 KV, native W4A4 candidate and separate W4A16 lane. Reserve generated-token capacity beyond the prompt.

3. Freeze numerical/quality margins and latency gates. Proposed values are 15% median 8K ITL improvement, 20% stretch, 5% tail/TTFT guardrails, 1% relative perplexity and 1 percentage point task loss, and 2 GiB initial memory reserve.

Deliver: docs/decision_ledger.md (D04–D08); configs/workload.yaml; configs/acceptance.yaml.

Exit and review: Owner records explicit decisions, authorized scope and selected roles. Reviewer verifies no proposed default is represented as accepted.

If blocked: Keep dependent choices pending. Do not infer acceptance from silence or begin implementation on an unavailable host.

### A0.2 Pin the host and dependency matrix

WP0  R01 R03 R04 R08  |  Owner: Runtime engineer  |  Entry: A0.1 scope authorized; host available

Inputs: Host inventory; selected model; current official runtime and library support documentation.

1. Record OS, GPU SKU/UUID, free VRAM, driver, CUDA toolkit/runtime, compiler, PyTorch, host runtime, FlashInfer, CUTLASS and optional b12x revisions. Record build flags, SM target, package locks and dirty source state.

2. Build a support matrix for exact Gemma architecture, gated GELU, native NVFP4 layout/activation quantizer, attention/cache modes, graph capture and SM120. For each path record supported, unsupported or unverified with a source or minimal test.

3. Identify runnable documented smoke invocations only after version selection. Save exact command lines, environment variables, logs and dispatched backend names; do not invent flags from a different version.

Deliver: configs/environment.lock.json; docs/compatibility.csv; scripts/smoke_reference.sh.

Exit and review: Reviewer can identify every binary/source revision and the precise FlashInfer host and fallback classification.

If blocked: Resolve unsupported build/model paths before baseline claims. Installing or changing host settings needs the applicable authorization.

## WP0 Audit tensor semantics and peak memory

### A0.3 Audit the checkpoint and packed format

WP0  R04 R05 R07  I01 I02  |  Owner: Model and correctness lead  |  Entry: A0.2 selected revisions pinned

Inputs: Checkpoint config/index, tensor metadata, quantization config and reference model implementation.

1. Inventory every tensor: name, shape, stored dtype, bytes, quantized versus unquantized coverage, aliases/tied storage and multimodal ownership. Reconcile all 30 layers and model dimensions with the selected revision.

2. Write the full expert format contract: nibble order, block shape, padding, scale dtype/layout, per-tensor factors, activation quantizer, saturation/rounding, accumulation and output dtype. Include runtime repacking and whether original weights remain resident.

3. Extract routing, branch normalization, gated GELU, attention and logit-softcap semantics into I02. Verify checkpoint quantization coverage rather than assuming all weights are FP4. Validate representative pack/unpack/scale round trips.

Deliver: model/tensor_manifest.json; model/quant_contract.json; docs/model_semantics.md; tests/test_packing.py.

Exit and review: Independent reviewer reconciles tensor totals, sharing and quantization semantics against pinned sources; round-trip fixtures pass.

If blocked: Unknown scales/layouts block custom kernels. Do not substitute another 4-bit format or silently remove tensors to manufacture fit.

### A0.4 Prove load and cache memory fit

WP0  R03 R07 R09  I03  |  Owner: Runtime engineer  |  Entry: A0.3 manifest complete; accepted workload

Inputs: I01 manifest, host free memory, loader, reference prefill/decode path and cache contract.

1. Measure device-used, allocated and reserved bytes before load, after load/repacking, cache allocation, graph capture, peak prefill and warmed decode. Account for retained copies, scales, graph pools, scratch and display/runtime use.

2. Run text-only smoke prefill/decode at accepted 2K/8K probes with output capacity reserved and no CPU weight offload. Record outputs, load failures and actual kernel dispatch. Test 32K only after the measured budget permits it.

3. Check local-cache bounded storage separately from sliding masks; check global cache growth and distinct stored K/V. Exercise prompt-to-decode handoff and 1023/1024/1025 positions. Compare predicted versus measured peaks.

Deliver: results/wp0/memory.csv; results/wp0/smoke.json; runtime/kv_contract.json; docs/gates/G0.md.

Exit and review: G0: smoke correctness, safe measured peak fit and exact backend classification. A compatible FlashInfer path is required before G1 can pass.

If blocked: Classify OOM, semantic or backend failure. Propose a context/layout change for approval and re-freeze it; an unsupported FlashInfer path is never a speed win.

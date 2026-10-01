# Layer-0 normalization and attention contract

The frozen full/cached pair is compatible with the declared normalization,
RoPE and attention arithmetic, while preserving the earlier exact cache and
writer checks. All retained RMS and RoPE outputs reproduce exactly in an
operator-only GPU replay. Attention reproduces exactly too, but full prefill
uses **FA2** and cached decode uses **XQA**. Holding the inputs fixed still
produces different attention bits. Shape-dependent QKV arithmetic is one
contributor to the downstream difference; the attention provider is another.

This is a bounded diagnosis, not whole-layer or whole-model numerical acceptance.
The intervals are conditional on declared native arithmetic. They were derived
from operation counts, source casts and format precision, without fitting them
to observed errors. Their conservative width is not a healthy model tolerance.
Natural routing quality, the historical v3 anomaly, and timing remain open.

The follow-on starts from combined main
`52b69ea10566e78ef514a7959f3f986ed117f0c3`: 360 reference tests and 38 scaffold
tests passed locally, and exact-main
[reference CI](https://github.com/yindavidyang/megartx/actions/runs/36868854471)
and [scaffold CI](https://github.com/yindavidyang/megartx/actions/runs/36868854537)
passed. The original checkpoint, pinned runtime, forced route, 33-input sequence
and previous evidence stay unchanged. Source records and original norm hashes
are in [source pins](evidence/layer0-norm-attention-source-pins.json); new scalar
results are in [live evidence](evidence/layer0-norm-attention-live.json).
The final publication tree passes 376 independent reference tests (16 new),
38 scaffold tests, contract validation, the exhaustive format self-check and
Python compilation. The earlier [QKV diagnosis](layer0-arithmetic-diagnosis.md)
retains its separate conditional accumulation assumptions.

Gemma4 uses ordinary RMSNorm with direct learned weights, epsilon `1e-6`,
256-wide Q/K/V heads and 2816-wide decoder norms. V has no learned weight.
The five loaded learned norm tensors were compared directly with their original
checkpoint bytes. The installed vLLM version identifies upstream commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`; its
[RMS source](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/csrc/libtorch_stable/layernorm_kernels.cu)
reduces promoted BF16 squares in F32, applies `rsqrtf`, multiplies in F32 and
casts once to BF16. The source/version association and pinned installed binary
hash are evidence bindings, not a binary-build or instruction-rounding attestation.

The RMS interval allows any F32 reduction tree via gamma-D, the division and
`rsqrtf` allowances in the
[CUDA 13.0 mathematical-function table](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix),
F32 scaling and final BF16 rounding. The selected values stay in the guarded
normal F32 lane. Every retained RMS output also happens to equal the rounded
high-precision result; that observation does not define the interval.

The retained RoPE cache is BF16, NeoX style, with 256 rotary dimensions. In the
guarded lane both BF16 products fit F32 exactly, so the
[source pair arithmetic](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/csrc/libtorch_stable/pos_encoding_kernels.cu)
has an independent exact CPU replay. All 24,576 selected Q/K outputs match.
Changed QKV inputs explain the changed normalization and RoPE outputs for this
fixture; replaying each path's own inputs needs no extra shape tolerance there.

Attention has scale **1.0**, contiguous grouped-query mapping (two Q heads per
KV head), causal masking and a 1024-token sliding window. Both installed
implementations cast unnormalized exponential weights to BF16 before their
row sum and value accumulation. FA2 updates online over 16-key tiles. XQA uses
one nonempty 256-key tile and stores a normalized partial output in BF16 before
its multi-block merge. The conditional interval includes those casts, faithful
F32 QK/PV accumulation, reduction/rescaling uncertainty, documented exponential
allowances and final BF16 rounding. Native MMA rounding remains an assumption.
Decimal exponential evaluation keeps the CPU reference independent of Torch.

The model's XQA grid is `[21,8,1]`, block `[128,1,2]`. A first replay used a
compact three-page table and selected grid `[1,8,1]`; it reproduced the retained
bits, but the independent launch check rejected that branch mismatch. The final
replay preserves the pinned 8448-token table capacity and matches the original
21-block launch. Only the three actual prefix pages carry K/V data. Physical
page IDs are compacted while within-page offsets, logical order and K/V strides
are preserved. No runtime environment override forces XQA scheduling.

Three fixed repeats reproduce all 66,560 selected RMS outputs, 24,576 RoPE
outputs and 16,384 original attention outputs exactly. Two zero-query,
constant-value attention controls return exact BF16 ones. Twenty-one attention
launches are independently bound by CPU launch correlation, kernel-name hash,
grid and block; asynchronous GPU time need not overlap the CPU annotation.

| Position-32 comparison | Different BF16 values / 4096 | Maximum absolute difference |
| --- | ---: | ---: |
| Original full FA2 vs cached XQA | 406 | 0.0078125 |
| Same full inputs: FA2 vs XQA | 247 | 0.0078125 |
| Same cached inputs: FA2 vs XQA | 246 | 0.00390625 |
| FA2: full vs cached inputs | 191 | 0.0078125 |
| XQA: full vs cached inputs | 246 | 0.0078125 |

The ideal attention change caused by different inputs is at most
`0.00471020212273876`. Original and crossed outputs fit their independently
declared intervals. Wrong `1/sqrt(D)` scaling, shifted GQA grouping and a
V-only row permutation each reject over 4000 outputs against the correct
fixture interval. Joint K/V permutation is commutative for the final query;
it cannot certify cache ownership. Exact address, prefix, writer and stored-bit
checks remain mandatory regardless of arithmetic compatibility.

The final GPU probe peaks at 143,753,728 allocated bytes, loads no model, starts
no server and exits with code 0. Cleanup confirmed no compute process or listener
on port 18000 after that run. Profiling and copied operands preclude timing claims.
The earlier failed missing-PATH invocation and compact-table replay remain
private evidence. This publication turn performs no further GPU experiments.

To reproduce the CPU analysis with retained private artifacts:

```sh
PYTHONPATH=numerical_reference python numerical_reference/layer0_norm_attention_reference.py \
  --full <native-full-v2> --cached <native-cached-v2-a> \
  --plan <private-controlled-plan-v1> --replay <operator-replay-v3> \
  --output <fresh-private-report.json>
```

The operator script `scripts/replay_layer0_norm_attention.py` has the same
arguments except `--replay`. Run it only in an explicitly owned bounded GPU
slot, using the existing virtual environment's executable PATH. It fixes three
repeats, refuses a reused output directory and checks source/runtime pins before
CUDA work. No checkpoint or existing capture is written.

The next bounded numerical step is the original-weight output projection:
replay the selected attention rows at M33/M1 and check the split-K/GEMV result
against an independent accumulator contract. That closes the remaining
unchecked arithmetic boundary in the selected layer-0 chain. Whole-model
full/cached head differences and natural-routing quality need separate gates;
these bounds must not be propagated into a fitted end-to-end tolerance.

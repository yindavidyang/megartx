# Standalone M1 maps and packed-input expansion kernel

The [CUDA kernel](../kernels/m1_maps_expand.cuh) compiled and ran on the RTX 5090
on 2026-10-01. It fuses expert maps, packed AQ replication, route-weight bit
permutation and activation-SF expansion in one 256-thread CTA. Actual GPU outputs
matched the independent CPU oracle, and four Compute Sanitizer tools passed.
The [measured receipt](evidence/m1-maps-expand-standalone.json) records source,
binary and capture hashes, resource limits, checks and every timing sample.
This experiment starts from main `54557e2e537f56f32ad0b604aaba9361a655b634`.

The scope ends at ordinary byte buffers. It prepares no opaque TMA/CuTe
descriptors, calls no incumbent entrypoint, and executes no quantization, expert
GEMM, GELU, reduction or model operation. The existing
[`select_preparation`/`run_candidate`](../src/megartx/m1_preparation.py) remains
unchanged: selection returns the incumbent and candidate execution raises.
The manual executable is outside `src`; no runtime registration, opt-in switch,
graph capture or graph path was added.

## Explicit buffer contract

Only M=1, H=2816, E=128, top8 with eight distinct IDs in `[0,128)` is supported.
Inputs are a common **already quantized** packed FP4 row and its existing 128x4
swizzled scale storage. Inputs and outputs must be caller-owned, disjoint CUDA
allocations of at least these extents, on the device/context of the supplied
stream. The raw launcher checks geometry, nonnull pointers and alignment; it
cannot discover allocation extents or aliases. The benchmark owns and verifies
every allocation. This pointer interface is not a runtime capability receipt.

| Buffer | Extent in bytes | Required pointer alignment |
| --- | ---: | ---: |
| Input `ids`, `weight_bits` | 32 each | 4 |
| Input `aq` | 1408 | 16 |
| Input `sf` | 22528 | 4 |
| Output `slot_to_sorted`, `sorted_to_slot` | 32 each | 4 |
| Output `offsets` | 1032, all 129 int64 entries | 8 |
| Output `expanded_aq` | 11264 | 16 |
| Output `permuted_weight_bits` | 32 | 4 |
| Output `expanded_sf` | 2883584 | 4 |

Ranks count smaller IDs; offsets count IDs below each expert boundary. Packed
AQ and FP32 route-weight encodings move as integers, preserving signed zero
and subnormals without casts. For sorted rank `r` and expert `e`, the SF base is
`round_up(r + 127*e, 128) * 176`. Exactly four bytes at offsets `g*512`, for
`g=0..43`, are copied per expert. Only 1408 destination SF bytes are written;
every other SF byte retains its prior value. These formulas match the existing
[independent semantic oracle](../numerical_reference/m1_preparation_reference.py).

Device checks reject out-of-range/duplicate IDs by leaving all outputs untouched.
They return no host-visible validity receipt. The executable validates valid
fixture IDs on CPU before launch and separately exercises these device no-op
cases. Unsupported M/H/E/topK, an unaligned AQ input and a null SF output are
rejected before launch. Integration must still validate its full quantization
lane, storage ownership, native ABI, correction runner and fallback semantics.

## CPU verification and actual GPU verification

The [fixture generator/verifier](../numerical_reference/m1_kernel_fixture.py)
imports only the standard-library CPU oracle. It never imports the kernel,
Torch, vLLM or FlashInfer. Its synthetic scalar contexts only satisfy the CPU
oracle's semantic constructor; no native descriptor/global is captured or
verified by this experiment.

Three fixtures cover unsorted low/high expert IDs, repeated routes with changed
payloads, and disjoint routes reusing poisoned SF storage. Payloads include all
256 FP4 byte patterns, signed zero, finite negative/large/subnormal FP32 route
encodings, valid SF codes including negative zero, and invalid codes in unused
input padding. Full SF padding and 32-byte outer guards are checked. All input
buffers remain byte-identical. Both the fused implementation and a separately
launched map/expansion development control pass against the CPU expectations.

Actual output captures were independently recomputed on the target and again
locally, including captures from memcheck, initcheck, racecheck and synccheck:
zero errors, zero race hazards/warnings. Negative IDs, ID 128 and a duplicate ID
leave storage unchanged. M2/M4/M8, H2815, E127, top7, misalignment and a null
output are rejected. The five new CPU tests check corrupt outputs, poisoned
padding/guards, altered expectations, invalid fixture values and truncated or
symlinked captures. CPU capture verification alone does not attest its producer.

All 387 numerical-reference and 38 scaffold tests pass; eight configs validate,
Python compilation passes, and all 4064 finite FP4-times-SF products remain exact
in the existing BF16 representation self-check. These are CPU checks; the
separate captured runs above establish the narrower GPU byte result.

## Measured microbenchmark and resource limits

CUDA events cover 512 consecutive host-submitted invocations per batch, with
32 warmups and nine alternating-order rounds. Setup, H2D, file access and CPU
verification are outside timing; host submission gaps can be inside the event
interval. Buffers are hot and CUDA graphs are disabled. Oracle equality is
checked before and after timing.

| Implementation | Median microseconds/invocation | Range |
| --- | ---: | ---: |
| One-CTA fused kernel | 2.30719 | 2.30694–3.58656 |
| Our two-launch development control | 3.41519 | 3.41494–3.41912 |

The control uses one 32-thread map CTA followed by eight 256-thread expansion
CTAs for the same output contract. It is our standalone control, not installed
FlashInfer. The fused outlier is retained in the receipt. These timings establish
neither an incumbent improvement nor an end-to-end/project performance gate.

One lean NVCC build took 2.060 seconds and 213733376 bytes of sampled aggregate
compiler-group plus wrapper RSS under a 2 GiB / 300-second cap. The fused kernel
uses 36 registers/thread, 33 shared bytes and zero local bytes/spills. The
executable's guarded device allocations total 5816976 bytes, below 8 MiB.
Every phase retained at least 8 GiB host available memory and 2 GiB GPU headroom;
observed minima and each bounded phase are in the receipt. Existing packages,
caches and runtime were not changed, and no checkpoint was read. The owned
remote temporary directory was removed; no compute processes remained, with
41 MiB GPU memory used afterward.

## Reproduce with an existing CUDA toolchain

Use a fresh private fixture/output directory and an idle GPU. These commands
create only synthetic fixtures, a local standalone binary and capture files;
they install nothing. The recorded toolchain is NVCC 13.3.33, g++ 13.3.0,
Compute Sanitizer 2026.2.0.0 and Python 3.12.3 on SM120f.

```sh
python3 -B numerical_reference/m1_kernel_fixture.py create /tmp/m1-fixtures-new
/usr/local/cuda/bin/nvcc -std=c++17 -O3 --threads=1 -lineinfo -Xptxas=-v \
  -gencode=arch=compute_120f,code=sm_120f \
  benchmarks/m1_maps_expand.cu -o /tmp/m1-maps-expand
/tmp/m1-maps-expand correctness /tmp/m1-fixtures-new /tmp/m1-captures-new
python3 -B numerical_reference/m1_kernel_fixture.py verify \
  /tmp/m1-fixtures-new /tmp/m1-captures-new
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 3 \
  /tmp/m1-maps-expand correctness /tmp/m1-fixtures-new /tmp/m1-memcheck-new
# Repeat with initcheck, racecheck and synccheck, using new output directories.
/tmp/m1-maps-expand timing /tmp/m1-fixtures-new /tmp/m1-timing-new
```

The one-off compile/sanitizer coordinator enforced aggregate RSS, available
host/GPU memory, bounded wall times and owned-process cleanup. That private
coordinator is not part of this kernel API. The executable checks 8 GiB host
and 2 GiB GPU headroom before allocation and each timing round.

## Dependencies before integration

Four installed source hashes match FlashInfer commit
`8bc3b578027791336c6ae87db5c9d76f82cef8bc` and CUTLASS commit
`b46b16d003484063bca4ed365e44095c4c6ed633`; cached incumbent module/build hashes
were unchanged. The standalone build includes only CUDA runtime and standard
C++ headers. Source identity does not verify opaque native objects.

Typed native descriptor sizes/alignments/fields, consumer-read masks, actual
workspace owners/aliases, both tactics/swap/epilogue flags and a narrow C++
insertion point remain unresolved. Actual installed preparation captures and a
matched incumbent timing comparison remain pending. Corrected numerical-lane,
full/cache, graph and full-model qualification remain separate dependencies.
None of those gates is relaxed by this isolated byte kernel.

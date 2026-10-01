# Installed M1 preparation probe result

The installed source profile matches the CPU oracle's pins. The typed target
probe did not compile within the prescribed 512 MiB host allowance. No native
descriptor ABI or GPU preparation byte equality is claimed. The exact result
and sanitized identities are in the [receipt](evidence/m1-installed-preparation-probe.json).

## Evidence and stopping point

This isolated task started from merged PR #5 at
`54557e2e537f56f32ad0b604aaba9361a655b634`. Its post-merge
[scaffold](https://github.com/yindavidyang/megartx/actions/runs/36889033625) and
[reference](https://github.com/yindavidyang/megartx/actions/runs/36889033585)
workflows passed. Strict noninteractive SSH reached the owned target. Before
and after the attempt, the RTX 5090 had 41 MiB used, 32,101 MiB free, 0%
utilization and no compute PIDs.

Read-only inventory verified all four installed header hashes against
[the preparation source profile](m1-preparation-probe-handoff.md): the fused-MoE
implementation, grouped-GEMM descriptor header, quantization utilities and
CUTLASS scale layout. The installed JIT source also matches its pinned hash.
This establishes source identity for those files; it does not prove the cached
binary was built from every current transitive or generated input.

The cached `fused_moe_120.so` is 61,836,776 bytes, SHA256
`dc26a85431c946b0f636ddd2e08aa676f6340fd085361b21e8b0fe00617d28f9`.
Its Ninja manifest is SHA256
`8e01f5b25d3cd211874278a7756f6694e2fa5bb38c6e8a148610cf16aaa3de6f`.
Its CUDA profile uses NVCC 13.3.33, C++17, the SM120f target, one compiler
thread, FP4/BF16 enables and C++11 ABI=1. Symbol inspection found the stock
map and NVFP4-to-NVFP4 expansion launchers. The module was not loaded or called.

One host-only typed-probe build used the captured CUDA flags and installed
headers in a fresh target directory. The resource guard stopped its process
group after **1.496 seconds**, at a sampled group-plus-wrapper RSS of
**554,164,224 bytes (528.5 MiB)**, exceeding **536,870,912 bytes (512 MiB)**.
The 300-second time limit was not reached. No generic flag substitution, package
change, wider memory allowance or retry followed. No successful build, typed
field report, transitive include receipt or native exporter was obtained. Struct widths,
offsets, runtime ownership and loaded-module equivalence remain unverified.

The owned compiler processes stopped, and only the task-owned target directory
was removed. Post-cleanup module and Ninja hashes were unchanged. No checkpoint
payload was read, no installed runtime was modified, and no GPU preparation,
GEMM, model forward, candidate activation, graph or benchmark ran. GPU capture
count is zero; periodic GPU peak telemetry was unnecessary and was not collected.

## Smallest independent kernel implementation

The next concrete implementation is one standalone M1 kernel that fuses expert
maps with **packed activation and scale-factor expansion**. Keep opaque grouped
GEMM descriptor preparation with the incumbent until its typed ABI is verified.
This reduces the first kernel's interface to owned integer/byte buffers and raw
route-weight bits; it requires neither resident expert weights nor a model
forward to test its defined copy/permutation behavior.

For H=2816, E=128 and eight distinct experts, use the existing independent
[byte oracle](../numerical_reference/m1_preparation_reference.py) to check both
maps, all 129 prefix offsets, 11,264 expanded AQ bytes, permuted weight bits and
the complete 2,883,648-byte guarded SF snapshot. Check unchanged inputs and every
padding/guard byte. Use the prescribed unsorted route, repeat, and disjoint route
in the same scratch; retain M=1 and reject unsupported shapes. A focused native
exporter must be implemented, reviewed and successfully built before any such
GPU fixture runs. Partial results must retain their partial scope rather than
fabricating the full descriptor capture packet.

Source-matched stock comparison can use the observed map and expansion
launchers once their caller ABI and required parameters are bound. Expansion's
host wrapper takes `QuantParams`; an exported symbol alone does not authorize
constructing that object from guessed offsets. The third descriptor kernel
passes opaque descriptor and quantization objects by value and therefore retains
the stopped typed-probe prerequisite.

An independently oracle-verified standalone kernel can then receive isolated
kernel timing and a stock comparison where the baseline is bound. Whole-model
quality/cache qualification governs end-to-end claims; it does not prevent this
bounded kernel implementation and its independent correctness tests. Neither a
speedup nor native byte correctness has been measured by the present attempt.

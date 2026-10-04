# Current-main and native-prefill composition

This is a CPU/source composition of exact current main
`036ec1c63fd41a740a8a9b67e76ee294b1c74990` (tree
`2b9165ad156503850f0ee81a38288ae180dc0ab2`, merged PR30 and PR33) and the
reviewed PR27/34/35 stack at `6493affbcf0827f8702e736636bb4f2549d07305`
(tree `257ef9b3ab5a5a88d606eb22c461540de4bf4dc9`). The common ancestor is
`ca6c385562aff30e91e8bb6a7f4228c9d766c30d`. A local merge preserves both
parent histories. PR36 map borrowing is not included.

## Why shared runtime composition is required

The branches change three common files. The launcher and numerical lineage
test conflict; the launcher resource test merges textually. All native writer,
manager, cache-reader, head, sampler, client, and arithmetic-control code from
PR35 remains byte-identical. Main's descriptor and warmed-timing implementation
also remains byte-identical outside the shared launcher.

The launcher cannot be resolved by selecting one parent's text. Main requires
locked process observations, exact monotonic timestamps, a closing GPU resource
sample, resource-row validation, and warmed-boundary admission. Native prefill
requires compact capped evidence, its shared wall deadline, foreign-GPU checks,
bounded child output, and preservation of primary failures during cleanup.
These purpose-specific paths are composed explicitly and tested from the
actual launcher AST with external effects substituted.

`scripts/m1_owned_processes.py` stays byte-identical to main. The new
`scripts/prefill_owned_processes.py` is an exact byte copy of the ownership
helper executed by PR35. Both native prefill purposes select that helper.
Its path and bytes enter the frozen plan's source vector; the composed catalog
binds both helper paths and hashes. This duplication is a bounded integration
tradeoff: main's always-recorded interval history would otherwise alter native
prefill's evidence and could exhaust the fixed metadata cap when copied into
multiple cleanup receipts. It is not a general exemption from source review
or evidence retention. No cap is raised and no required record is trimmed.

The default-fit check still exercises current code. Historical PR34 tests and
snapshots remain intact, while the current launcher proof specializes only
explicit purpose branches against exact parent snapshots. Main's decode path
is checked against current main, and native prefill against validated PR35.
The current plugin, native provider and plan validation retain their separately
checked historical semantics. Added source inventory entries are explicit.

## Additive source admission

`current-main-source-reconciliation.json` records both exact parent heads and
trees, the common base, immutable parent-ledger hashes, each differing source's
two parent hashes and composed hash, and a complete current runtime vector.
It also binds the compatibility tests and exact parent launcher snapshots.
Unknown or mixed vectors, wrong helper selection, changed parent ledgers,
missing source entries, and missing equivalence evidence are rejected.

Every historical ledger, receipt, catalog, and PR35 source freeze remains
unchanged. Numerical lineage validates each branch's terminal hashes before
applying the two-parent reconciliation. Prefill's earlier correction chain
receives an additional terminal delta rather than being relabeled as current
source. The runtime merge is frozen separately from the new catalog metadata.

## Scope of evidence

The experiment owner reported one successful control on exact PR35 `6493aff`:
all 30 layers across nine frames (270 layer frames) matched processed K/V to
stored native bytes; actual manager/kernel pages were 16/16, ratio 1, group 0;
256 output tokens agreed and resource/cleanup gates passed. The owner verified
the private raw bundle. Cloud review inspected a compact sanitized readback;
it did not independently replay the raw native tensors. This composition does
not rewrite that experiment's source or transfer its execution result to a new
tree.

The unchanged proposed operation is one P2048/chunk256/BF16/capacity2304,
256-output context with zero retries. It remains limited to exhaustive storage
through the first consumed anchor and metadata/frontier continuation. The
4 GiB transfer, 8 MiB evidence, 2 MiB metadata, 8 MiB observer GPU, 1800-second
shared wall, compiler and headroom limits remain unchanged. Independent
arithmetic, numerical correctness, quality, performance and repeatability
remain unqualified; natural positive correction coverage remains unknown.

Independent review of this composed tree is required before publication or any
new experiment. CPU tests do not authorize GPU execution or a public merge.

# Prefill and decode attribution CPU composition proposal

This separate catalog composes PR23 at main
`7d1a33d1761a1719fc26c84f5b4f2a3410c31a16` with the decode branch at
`3f06df192f1792ac4ddaaa58ee27628ac5665e57`. Parent source review is pending.
It performs no GPU work or native imports and grants no execution, dispatch,
timing, quality or performance qualification.

The original [live plan](live-plan.json) and [live binding](live-binding.json)
remain byte-identical. Their four immutable input hashes refer to the earlier
prefill source catalog, so the combined tree correctly rejects that old binding.
The [reconciliation](live-attribution-reconciliation.json) records all four
old/current hashes and the exact changed JSON paths. Those paths concern the
existing attribution controller/plugin review and its dependent source digests.
Workload controls and historical source vectors are preserved.

The current [live plan](live-attribution-plan.json) differs from the historical
plan only in its binding digest. The current [binding](live-attribution-binding.json)
checks all eight launcher files and all four input files as a complete catalog.
Only the launcher and its tests have changed among the eight launcher files.
The loader selects this named plan/binding pair together; a partial pair fails.
Explicit paths must also select both files. Unknown inventory entries and every
old/current mixture of the four input hashes fail the existing strict checks.
There is no fieldwise overlay or automatic source rehash at runtime.

All eleven chunk cells, resource caps, null receipts and disabled public GPU
execution remain unchanged. The tests cover catalog identity, all fifteen
noncurrent input mixtures, partial catalogs, source drift and symlinks, native
import exclusion, and refusal to execute even with self-declared receipts.
These are CPU composition checks; they do not qualify a future lifecycle provider.

The failed lean decode run at `3f06df1` remains rejected and immutable. This
composition proposal does not change compiler or diagnostic admission, relax
ordinary timing checks, or reinterpret any prior GPU evidence. Source digests
describe local bytes; they do not attest the target environment or approval.

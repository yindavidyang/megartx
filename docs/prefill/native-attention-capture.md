# Default-off sampled native prefill attention observation

This purpose retains the original nine all-30-layer storage frames and the full
263-head/256-output client frontier. It adds 480 selected output coordinates in
layers 0 and 5, using the six unchanged CPU contract/reference/reader files.
It reports error against the independent high-precision reference; native
arithmetic acceptance is null. It makes no numerical, quality or performance
qualification.

## Opt-in and closure

`--client prefill-attention` selects the compact **prefill** owned lifecycle,
never the decode lifecycle. The launcher clears inherited selectors and sets
only `MEGARTX_PREFILL_ATTENTION_CAPTURE=1`. The plugin rejects a simultaneous
storage selector. Without the new selector, the existing default/storage runtime
policies are unchanged. No model input, dispatch or output is replaced.

The enclosing selected FlashInfer implementation and actual low-level wrapper
arguments bind layer/frame/query/cache/out/manager identity. Global D512 uses
compatible native paged decode; local XQA is observed only when naturally
selected. The cached vLLM lazy XQA alias is verified without cache clearing or
forced warmup. FlashInfer's actual decorator/trace closure is part of source
admission. The following source values are upstream contracts, **not installed
host attestations**:

- vllm.utils.flashinfer: 7144bbb7a905ad3d40994a2d3e69d69144e4e71020e7c92f487dea65b390790d
- flashinfer.api_logging: 0302b4ff890c9d0de9c91f340794d8805b68b467690fe3fbec3dfb29f05720ad
- flashinfer.trace.template: 8188ab3ff4be944bc15817f10da342ea9e0c7e25a4a29f8868140e5e685c6825
- flashinfer.utils: 6cb8ebcc25eb65521808bf40a8aaf0c5a868827283955867ae382ad7ec773815

Plans default these installed-source entries to null. Unknown entries cannot
receive native clearance. The later experiment-preparation step must establish
actual installed bytes. Separate exact-head CPU review and a fresh owned GPU
slot clearance remain mandatory. This implementation is CPU/source-only.

## Data and budgets

The storage provider's explicit no-op-default callback reuses already-charged
processed writer K/V bytes. New-purpose raw retention replaces the old 180+2
payloads with exactly eight fixed files totaling 5,395,952 bytes. The two native
full-head copies and finite/hash checks still occur; their payloads are not
retained. The exact storage transfer ledger remains charged for them.

All evidence uses one cross-process lock and one 8 MiB total/2 MiB metadata
budget including the two-byte exit signal, temporary state publication and a
first-failure reserve. Raw files are preallocated once. Per-contribution durable
completion bits and pending-before-write transactions reject missing, duplicate,
short or interrupted writes; zero-filled holes never imply completion.

The actual wrapper consumes independently read cache K/V rows, checked against
owned manager mapping and retained writer bytes. New transfers are exactly
73,388,032 cache-entry bytes, 102,400 full Q/O-head bytes, and a 65,536-byte
metadata allowance. Writer reuse adds no transfer. The existing aggregate
4 GiB D2H ledger and 8 MiB incremental observer GPU cap remain authoritative.
Native producer completion uses a source-bound device synchronization before
selected O is copied and before the enclosing forward reaches o_proj. This is
an explicitly timing-unqualified observer.

## Publication and CPU analysis

All original storage, frame, head, scalar, client, source, memory, telemetry and
owned-server cleanup gates must pass first. Capture records are tied to those
current validated frame inputs, including input 2048. The independent reader
then validates exactly eight raw manifests and twelve actual operator records.
A separate source-pinned owned stdlib process runs the frozen reference under a
hard 512 MiB RLIMIT_AS ceiling, conservatively bounding process RSS, plus CPU
and wall allowances no greater than 300 seconds within the original 1800-second
lifecycle. Linux RLIMIT_RSS is not used. Its independently calculated input
manifest must equal the validated raw manifest. Failure or unresolved reference
arithmetic cannot publish a successful attention receipt.

The preserved helper hashes are prefill
`df01a29a8089db09530eef3d36efc04d2871033b4fa6b357eb73a0aca99dae4b`
and decode
`b193ac3b991723d82f3d92b4908d5dfd5e04fce32c0b9d62435cc66f95797253`.
Actual imported origins, not adjacent files, are checked.

Runtime source is frozen before the separate additive terminal-source
reconciliation. Historical catalogs, PR36 overlay and original proofs are never
refreshed into accepting an arbitrary per-file union.

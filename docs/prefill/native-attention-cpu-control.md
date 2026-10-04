# Sparse attention CPU observation contract

This additive CPU packet starts from exact PR35 head
`6493affbcf0827f8702e736636bb4f2549d07305`, tree
`257ef9b3ab5a5a88d606eb22c461540de4bf4dc9`. No existing runtime, provider,
plugin, launcher, ownership helper, storage protocol or historical catalog is
changed. These modules cannot launch a model, grant GPU clearance, inspect
weights, publish a native result or replace existing validation.

The independently reviewed minimum samples 480 attention-output coordinates:

- Absolute queries 0,15,16,255,256,1023,1024,1792,2047,2048
- Layer 0: D256, 16 Q / 8 KV heads; Q heads 0,1 and KV head 0
- Layer 5: D512, 16 Q / 2 KV heads; Q heads 0,7,8,15 and KV heads 0,1
- Full Q/K vectors; eight V/O coordinates
  `[0,1,D/4-1,D/4,D/2-1,D/2,D-2,D-1]`
- K/V positions 0 through 2048, with original finite little-endian BF16 bits

Local visibility for query p is max(0,p-1023)..p; global is 0..p. GQA maps
head h to floor(h/(16/Hkv)). The end-aligned native causal expression is
`j + qo_len <= kv_len + q_idx`; local adds
`j + qo_len + 1023 >= kv_len + q_idx`. The first final-chunk query 1792 needs
local rows 769..1792, not the final query's 1024..2047 interval.

Require actual unit scale, BF16 Q/K/V/O, causal mask, no sink, ALiBi, custom
mask, DCP, cascade or sharing, and no attention softcap. The model already
applied Q/K normalization/RoPE and V normalization, so the FlashInfer position
encoding is NONE. Correctness of those earlier operations is outside this
control. The final logits softcap 30 is not an attention softcap.

## CPU modules and APIs

`prefill_attention_plan` freezes the exact scope, eight raw shapes, original
all-layer storage/frontier obligations, source vector and immutable caps. A
plan is explicitly CPU-only and not native-integration-ready. Its native-plan
and prompt hashes are bindings, not permission or execution attestation.

`select_writer_row(layer, position, K_bytes, V_bytes)` takes full original
writer rows already copied and charged by the native storage observer. It
returns `(filename, byte_offset, selected_bytes)` writes. It performs no file
I/O or D2H charging. `select_head_row` takes one original full Q/O head and
selects O coordinates only after the GPU-to-CPU copy. Pure helpers cannot
certify that callers supplied the named actual tensor or position.

`prefill_attention_validation.read_capture` checks all eight exact raw files,
finite BF16 representation, limits including metadata/temporary files, source
hashes, externally supplied request/current-storage binding hashes, twelve
selected call records, independent raw root reconstruction, frame alignment,
semantic parameters and resolved wrapper-source choices. It rejects old raw
sample files, missing/truncated files, links, special files, stale or reordered
calls, root mismatches, unknown/auto dispatch and numerical-acceptance claims.
Directory enumeration is incremental and limited to 256 entries, including
empty/temporary entries. Source reads are limited to 1 MiB per file. Evidence
and source opens use NOFOLLOW and NONBLOCK before regular/single-link fstat
checks, so a FIFO substitution cannot hang before type validation. Source and
evidence paths reject symlink ancestors.
The largest raw file is the global K file at 4,196,352 bytes, above 4 MiB.

The reader structurally binds caller-supplied native observations; hashes do
not establish their truth. `expected_storage_binding_sha256` must come from
the current already validated native storage/frontier/client/cleanup domain,
not from this same packet or a saved earlier run. The result always retains
`external_native_storage_frontier_validation_required=true` and
`native_execution_attested=false`. A native integration must perform those
existing gates and bind their complete current records. This reader does not
substitute a boolean or a cached receipt for them.

`prefill_attention_reference` is independent of all capture/plan helpers,
Torch, NumPy and native arithmetic. It repeats exact geometry checks, uses
integer BF16 products/sums in units of 2^-266, then computes stable softmax
and sparse PV sums in Decimal at 96 and 128 digits. Every context explicitly
fixes HALF_EVEN rounding, Emin=-999999, Emax=999999, clamp=0, and all traps
except Inexact/Rounded. Caller decimal settings cannot alter the result.
Exact dyadic construction uses 600 digits. Exponents below -10000 fail the
reference domain rather than being silently clipped.

The two reference answers must agree within 2^-180 times maximum absolute
selected V (exact zero for zero V). This is a CPU-reference sanity check,
never native atol/rtol. Disagreement or timeout makes the reference unresolved.
It is not a certified transcendental enclosure. Accordingly rounded-ideal
BF16 word/ULP diagnostics remain unavailable; they are not invented by
double-rounding through FP32. The first mode reports per-case and aggregate
max absolute error, RMSE, L2/relative L2, signed extrema, ideal range and
negative-zero count. Metrics use deterministic decimal scientific strings with
18 significant digits so tiny nonzero errors and huge relative errors cannot
silently underflow/overflow binary64 JSON numbers. Relative L2 is null for an
all-zero ideal.
The numerical report independently hashes the exact eight byte objects it
decodes; its input manifest must match the structurally validated capture
manifest before the integration binds a report to that run.

Every numerical report keeps native arithmetic acceptance null and numerical,
quality, performance and sampled-repeatability qualifications false. Natural
positive correction coverage remains null. Capturing a shape-dependent native
result is not a continuous-2048 bitmatch test or an end-to-end tolerance.

## Source arithmetic distinctions

Pinned installed-source hashes are inherited from the existing source catalog:

- vLLM FlashInfer backend:
  `8ee541fde43ed92a417b3f01ea1e6fc64af8ab2eb54cbfc28308a49aaca4ed7f`
- FlashInfer post1 prefill.py:
  `2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de`
- FlashInfer decode.py:
  `d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a`
- prefill.cuh:
  `f93cbea96e3a723254c15f5a52d6dd4c13e3892e88eb9b9fed1c0c0048c7d710`

Hash-matching official upstream bytes show that ordinary FA2 casts exponential
weights to BF16 before its denominator rowsum, while the VO-split lane sums
FP32 weights in the denominator and stores BF16 weights for PV. XQA long
contexts can use multiple nonempty CTA partials/merges. Head dimension alone
does not establish the actual dispatched lane. The old local <=33-key
conditional interval therefore cannot simply be enlarged. Future acceptance
needs actual dispatch, split/merge/cast behavior and a separately reviewed
native rounding/FTZ contract fixed before examining numerical outputs.

The packet names actual entrypoints independently of backend labels:
`paged_prefill` is the BF16 FA2 BatchPrefill wrapper in prefill.py;
`paged_decode` is the FA2 BatchDecode wrapper in decode.py;
`xqa_decode` is the XQA decode.py entry. An M1 FA2 backend therefore does not
silently inherit the prefill.py source identity. Unknown or phase-incompatible
entrypoints fail structurally; native integration must verify actual callable
owners and sources before emitting these values.
Pinned vLLM source permits XQA only for dimensions 16 through 256, divisible
by 16, and explicitly routes Gemma4 global D512 groups to native FIDecode.
Thus layer 5's M1 record requires paged_decode/FA2/decode.py; a D512 XQA record
is rejected even if every packet hash has been regenerated. This is a source
compatibility constraint, not evidence that a compatible kernel executed.

## One shared resource ledger

Raw files total 5,395,952 bytes: K 5,245,440; V 98,352; Q 51,200; O 960.
The existing 2 MiB metadata ceiling makes retained bound 7,493,104 bytes,
below 8 MiB by 895,504. It includes failure reserve, temporary publication
files and the inherited two-byte external lifecycle signal. Raw files are
headerless, K/V ordered by position then selected KV head then dimension or
coordinate; Q/O by frozen query-position list then Q head then dimension or
coordinate. Reserve each exact final size once through the single native
cross-process evidence budget; write bounded host chunks at returned offsets.
Final manifest requires all exact sizes/hashes. No independent second budget,
raw duplicate, second archive, prefill-sized GPU clone or score matrix is added.

Only the new purpose replaces the old 180 K/V sample files and two raw head
files. All prior exact all-layer checks and scalar/head/frontier obligations
remain. Keeping both raw sets would exceed 8 MiB. Historical purposes remain
unchanged and are not silently retargeted to this new raw whitelist.

At each of six selected frames in each selected layer, the actual low-level
operator's cache union is independently checked against retained writer data.
The existing full-row reader costs 5,374 local rows ×8192 plus 7,169 global
rows ×4096 =73,388,032 additional D2H bytes. Selected Q/full-O heads add
102,400 bytes. Reserve at most 65,536 additional metadata-transfer bytes.
Writer K/V bytes are reused with **zero additional D2H charge**. All domains
share the inherited 4 GiB cap; original domains must be derived from exact
source and actual manager reads, not copied from a prior observed total.

Preserve one context/request, zero retries, 8 MiB incremental GPU scratch,
2 GiB GPU-free and 8 GiB host-free floors, compiler RSS2 GiB/shared300 s,
and shared1800 s lifecycle deadline. CPU reference allowance is at most300 s
within that deadline, with a separate512 MiB RSS cap enforced by the future
owned integration. The CPU function cannot itself impose a process RSS limit.

## Minimal pending native interfaces

1. Forward already charged actual writer row bytes to pure selection helpers
2. At actual low-level attention entry, bind enclosing layer/frame, Q/cache/out
   object/view identities, actual tables/parameters/dispatch and source bytes;
   collect cache-input roots from actual reads, never regenerate them from the
   retained writer files as though those were the consumed cache
3. Copy selected actual Q heads; after original native producer completion,
   copy selected output heads and feed the same untouched output to o_proj
4. Emit twelve records, independently reconstruct their roots offline, run the
   current inherited storage/frontier/client/cleanup gates, then calculate CPU
   errors. Preserve primary error, poison and safe own-hook restoration

No callback or file-budget integration is included here. Shared-wrapper layer
association, original argument/result identity, positional/keyword binding,
producer streams, offsets/strides, errors/interruption/restoration and legacy
purpose equivalence still require exact source-extracted integration tests
and independent review. This CPU plan cannot satisfy a GPU clearance check.

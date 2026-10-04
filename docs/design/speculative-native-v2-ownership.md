# Default V2 owner and zero-forward receipt

This CPU/source step starts exactly at PR28
`c50ba1410a5c35cd470139f2cf5e90984d7c88f2`, preserving the reviewed client and
PR25 lifecycle/collector ancestry. It adds a deliberately separate V2 owner
for the user-selected [Makora drafter](speculative-makora-drafter-adoption.md).
It does not change historical source pins, the V1 collector, or its rejection
of V2. It does not force a runner selection. No training, weights, installation,
runtime imports, device queries, GPU/model execution, or remote mutation was
performed in the implementation/CPU pass. Test runtime objects are explicit
CPU fixtures, never evidence of actual engine or model execution.

The [protocol](speculative-native-v2-zero-forward-protocol.json) remains
target-only. `NativeV2DiagnosticWorkerExtension` authenticates the actual
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`, its loaded module origin,
installed bytes and `get_model` code/defaults/globals. It authenticates the
resolved config property's code and requires an unset
`VLLM_USE_V2_MODEL_RUNNER` with `use_v2_model_runner is True`. Hashing a file
alone does not accept a shadowed getter, class, instance method or origin.
The actual Worker must own the same runner, config and device references.
The receipt entry retains the original config and device before constructing
the V2 owner. Both bindings pin that original device object and its explicit
CUDA ordinal; replacing both current Worker/runner fields cannot substitute a
new drain target. An implicit device without an ordinal is rejected without a
device query. Cache tensors must match the pinned device. Collection and
release synchronize only that original device and recheck its identity before
and after synchronization. Drift leaves drain unconfirmed and retains the
original BlockPool objects for owned teardown.

V2 uses `RequestState` and persistent `BlockTables`; its `InputBatch` is
constructed per forward. The owner therefore requires both request maps empty
and `execute_model_state is None`, rather than assuming V1's persistent
`input_batch`. It retains the actual model, default model state, cache config,
registry, groups, allocation placements, attention builders and cache tensors.
It strongly retains the cache group, specification and allocation-placement
children of mutable config lists, then compares their identity with `is`.
Integer object-address snapshots alone are not ownership: Python can recycle
an address after a mutable list releases its last reference. The CPU negative
uses the exact pinned public `KVCacheTensor` dataclass extraction and verifies
that replacement cannot collect the original or pass its owner check.
It rechecks these references, topology and sealed storage geometry through
release. The first adapter rejects manager/kernel page splitting, host/draft
groups, a loaded speculator, rope state, PCP, ubatch, batch sharding and fast
prefill. Supporting those cases needs a separate concrete adapter and review.

EngineCore must be freshly born under source-verified registration, with no
request ever observed, no scheduler/input/batch/pending work, and completed
`pause_scheduler(mode="keep", clear_cache=False)`. Common lifecycle helpers
now accept an internal profile; public V1 and V2 entries fix their respective
admission checks, utility names, worker extensions and ticket purposes. The
V1 defaults and its tests remain intact. The synchronous utility reserves
real BlockPool objects and retains them immediately. It seals ingress before
the RPC and never resumes, even after a successful release. Worker drain and
revocation must succeed before freeing the objects. Unknown drain, owner drift
or cleanup failure retains the references, poisons the lease and requires
owned teardown; neither a release retry nor a request is admitted.

The collector reads existing cache placement/page byte ownership, startup
correction proof, native head policy and existing workspace owners. It does
not build attention metadata, prepare inputs, modify cache writers, call a
model, rerun startup preparation, initialize a speculator or use lazy workspace
getters. Private identities/pointers stay in the raw worker receipt; EngineCore
exports scalar decisions, default runner policy and false capability flags.
The decision stays false even if a known allocation lower bound fits. No V2
mutation lease is attached to the runner, and both verifier entry points reject.

The inherited budgets are unchanged: 8 MiB incremental candidate scratch,
8 MiB canonical serialized evidence, 8 GiB host free, 2 GiB GPU free and a
shared-host compiler budget of 2 GiB/300 seconds. One selected-device native
Torch memoryStats query remains prospectively supported. Its native
pre-acquisition host peak is unknown; the private-pool count limit applies
after acquisition and host reserve readings provide a monitored boundary.
Neither the evidence cap nor those readings claim an 8 MiB Python/native heap
peak. Torch distribution `2.13.0`, runtime `2.13.0+cu130`, CUDA and Git/source
identities retain their exact existing checks. External CUDA, FFI allocator
coverage and M1/M2/M256 temporary bounds remain unresolved.

The [installed-source binding](../evidence/speculative-native-v2-source-binding.json)
adds 14 hashes to 22 unchanged historical files, read afresh through source-only
SSH. Six installed methods are extracted with file hash, line range and exact
method-byte hash for portable CPU adversarial controls. The local installed
extraction check also compares each method against the acquired full file.
These tests exercise actual selector/getter and BlockPool method logic using
CPU dependencies; they do not import installed vLLM/Torch or query a device.
The owner/release correction adds actual V2Owner, Worker and EngineCore CPU
controls for joint device replacement, drift during receipt/drain, unchanged
primary cancellation errors, uncertain cleanup retention and single-use
confirmed cleanup. These fixtures remain CPU source controls, without native
device, fit or performance qualification. The original CPU evidence record is
preserved; the [owner/release validation delta](../evidence/speculative-native-v2-owner-release-validation.json)
records the new checks separately.

Shared composition is still blocked. The prefill/shared owner retains
`vllm_scale_plugin.py` and `scripts/run_scale_validation.py`. Its exact
`7aafc35` observer and `2cc69b6` composition commits were read only. Its readonly
`LoadedEngineAccess`/`OwnedFrame` cannot confer this mutable engine reservation.
The V2 installer is default-off and is not wired into the shared plugin here.
The owner must compose V2 registration, worker extension, client admission and
private evidence/scalar schema deliberately; V1 receipt plans and comparison
schemas cannot be repurposed. A new exact integrated-source review and parent
sole-GPU slot are required before any engine startup. `freeze(project)` records
the clean committed HEAD/tree, all dependencies and exact protocol, with all
execution gates false. It produces a CPU review plan, not a runnable supervisor
or an authorization file.

The next drafter interface must use native generic `DSparkDraftModel`/Qwen3
routing to `DSparkSpeculator`, keeping candidate revision
`e6f739231e892fe4512c470136490d5441ffe62f` and config SHA
`a6d5250b5c1978642154993547cdc2ecbf069e1cbe1f99b2d79cb982e45f8ee2`.
It needs ordered five target taps `[2,9,17,24,27]` with explicit token,
residual/norm, retention, cache-owner and rollback semantics; checked target
vocabulary mapping, tokenizer and special IDs; and independent corrected
NVFP4 target verification/rejection. Confidence cannot authorize acceptance.
The advertised 2,411,575,146-byte persistent weights and draft KV are separate
budget domains. The 8 MiB scratch cap cannot absorb them or their unknown
feature, temporary, compiler or external workspace peaks. Those interfaces,
budgets and acceptance checks remain future reviewed work.

The initial zero-forward reservation retains the original receipt capacity:
P2048 plus four suffix tokens and one private page per group. Makora's six
proposals/block size seven do not inherit that capacity or the V1 verifier's
call budget. A future V2 verification adapter must freeze its own scheduling,
bonus-token, EOS, rollback and page-capacity contract before using the drafter.

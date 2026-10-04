# First zero-forward receipt client and operational packet

This milestone adds a real synchronous/asynchronous EngineCore utility client,
source-bound plan, source/checkpoint preflight, bounded immutable private writer,
receipt comparison, and an owned supervisor. Its base is the exact reviewed
lifecycle commit `95f31095f94d254b79c6620a046ccdffce712941`; historical protocol
pins remain unchanged. No model, Torch/vLLM runtime import, device query, or CUDA
job was used to develop or verify this packet. The parent-authorized exact
collector correction `b9f8e95e9483daa7be263e3a02a1b6a63ab09844` is composed
through a merge preserving both committed ancestries. Its independent CPU review
is clear, and the parent accepts the prospective monitored native-counter query
with its acquisition limitation explicit; this grants no live GPU execution.
The CPU fixtures exercise faults
and metadata; they are never admitted native owners. GPU execution and merge
remain closed.

The [plan schema](../../schemas/speculative-native-receipt-client-plan.schema.json)
and [authorization schema](../../schemas/speculative-native-receipt-authorization.schema.json)
are enforced more strictly by Python: clean committed source, exact HEAD, exact
per-file hashes, the reviewed lifecycle ancestry, fixed environment/configuration,
and canonical plan digest must all match. A freeze is a proposal. An authorization
record must refer to real independent review, exact-head CI, parent acceptance,
owned lifecycle controls, and the parent-assigned sole GPU slot. Producing the
record does not grant those approvals.

## Minimal shared-owner integration

The prefill owner retains sole edits to `vllm_scale_plugin.py` and
`run_scale_validation.py`. This change edits neither file. The general plugin
must reject simultaneous native/prefill diagnostic modes, then invoke these
callbacks before its installed-idempotence return and before EngineCore birth:

```python
from .speculative_native_lifecycle import install_native_diagnostic
from .speculative_native_evidence import install_native_receipt_evidence
install_native_diagnostic()
install_native_receipt_evidence()
```

Both are default-off. The writer requires `MEGARTX_NATIVE_RECEIPT_EVIDENCE=1`
and `MEGARTX_NATIVE_DIAGNOSTIC=1`. It requires the exact existing lifecycle
registration, wraps only `EngineCoreProc.megartx_owned_native_receipt`, and leaves
that utility's public scalar result unchanged. It creates no new capability or
utility name, patches no worker/Torch method, and grants no probe admission.
The worker extension remains:

```text
megartx.speculative_native_lifecycle.NativeDiagnosticWorkerExtension
```

The launcher uses the dedicated client script with its owned supervisor; it does
not start an HTTP server or a prefill client. Shared launcher integration should
delegate the native branch to this script, avoiding a second model lifecycle.

## Actual installed API and lifecycle

Read-only SSH inspection of the pinned environment established these interfaces:

```python
EngineArgs(**plan["engine_kwargs"]).create_engine_config()
Executor.get_class(config)
EngineCoreClient.make_client(
    multiprocess_mode=True, asyncio_mode=(mode == "async"),
    vllm_config=config, executor_class=executor_class,
    log_stats=False, renderer=None,
)
```

The installed `SyncMPClient.call_utility(self, method: str, *args)` returns
`future.result()` internally. `AsyncMPClient.call_utility_async(self, method: str,
*args)` must be awaited. Both clients have synchronous `shutdown(self,
timeout: float | None = None)`. The three installed source SHA-256 observations
are frozen in `speculative_native_plan.CLIENT_SOURCES`; no runtime imports were
used to obtain them. The asynchronous frontend transport still uses
`async_scheduling=False` in the actual EngineCore configuration.

The only admitted utility sequence is:

```python
# sync; this first call already waits for the completed keep-pause
client.call_utility("pause_scheduler", "keep", False)
scalar = client.call_utility("megartx_owned_native_receipt", receipt_admission)
client.call_utility("megartx_owned_native_release")
client.shutdown(timeout=bounded_timeout)

# async: await each completion in the same order, then synchronous shutdown
await client.call_utility_async("pause_scheduler", "keep", False)
scalar = await client.call_utility_async("megartx_owned_native_receipt", receipt_admission)
await client.call_utility_async("megartx_owned_native_release")
client.shutdown(timeout=bounded_timeout)
```

The client is single-use. Failed pause sends no receipt. Failed/partial receipt
RPC does not enqueue another release: the engine may have released internally,
retained references, or still be executing. Uncertain drain gets one release
attempt and then owned teardown. The engine stays paused; no requests, resume,
generation, prefill, or target-probe call are exposed by this client. Cancellation
and malformed replies preserve this rule. The original lifecycle alone frees
BlockPool references after confirmed worker drain/revocation. A known local
writer failure after a completed receipt uses that existing internal release;
uncertainty poisons and retains references for teardown. Shutdown errors attach
their error type without replacing the primary failure.

## Admission and immutable private evidence

The admission preserves every field required by `check_receipt_admission` and
adds `client_purpose`, `client_plan_sha256`, `source_head`,
`client_evidence_source_sha256`, and `private_receipt_directory`. Purpose is
`first_zero_forward_layout_workspace_measurement`; the existing engine lease
purpose remains `exclusive_native_verifier`. Neither purpose enables a later
probe. The monotonic deadline is generated on the execution host, never copied
from a different host's clock.

The private writer checks the already registered actual core, live lease,
PID/start-time, nonce, purpose, checkpoint, and writer source before saving:

| File | Private content |
| --- | --- |
| `native-receipt.private.json` | Full actual cache placements, page intervals, workspace owners, allocator/callback observations, and explicit unknowns |
| `native-receipt-identity.private.json` | Engine/worker PID and start-time, lease nonce, actual reservation group IDs and block sizes, source HEAD, plan and receipt digests |
| `native-receipt.scalars.json` | Exact original scalar utility allowlist |
| `native-client-lifecycle.private.json` | Completed client states, one-shot release/shutdown evidence |
| `native-owned-cleanup.private.json` | Owned PID/start-time cleanup and compiler resource evidence |
| `runtime.private.log` | Bounded private runtime log, never published |

Directories are owned mode 0700. Evidence writes use exclusive temporary files,
fsync, immutable mode 0400, and no-replace links. A duplicate cannot overwrite a
previous receipt. JSON topology, node count, depth, strings, integer magnitude,
nonfinite values, and a conservative escaped-byte upper bound are checked before
output acquisition. Canonical JSON is streamed with a byte check before each
write; its SHA-256 must equal the actual EngineCore receipt digest. The streaming
directory inventory rejects more than 64 files and reserves 24 MiB for the three
baseline startup outputs plus 8 MiB for logs before each evidence acquisition,
under the 64 MiB run evidence disk cap. Raw receipts,
pointers, prompts, tensors, callbacks' private paths, snapshots, and logs stay
private. Only explicit scalar allowlists and source/digest references can be
published under the standing approval.

Comparison checks the private/public digest and scalar agreement, actual source
and checkpoint, engine/worker identity and nonce, core reservation versus worker
page IDs, all 30 BF16 GPU layer owners, manager/kernel sizes, dense physical page
intervals and non-aliasing, and the measured page/head/metadata lower bound.
Allocator allocated/reserved/slack arithmetic must agree. Snapshot coverage is
reported as partial/unavailable or legacy observation; missing coverage is never
interpreted as zero allocations. Unknown FFI route, native temporary workspace,
and residual external CUDA bounds remain explicit rejection reasons. Fit is
measurement output and is not a preexisting admission certificate.

## Exact proposed operational commands

The following paths are a deployment proposal, not an observation that a remote
checkout or packet has been created. Parent review must assign their destination
and freeze the final integrated commit. Create a new owned mode-0700 packet
directory first. CPU freeze and source-only inspection use the pinned interpreter:

```text
/home/yyang/projects/megartx-baseline-20260930/.venv/bin/python -S /home/yyang/projects/megartx-dspark-receipt-client/scripts/speculative_native_receipt_preflight.py --freeze --client-mode async --private-directory /home/yyang/projects/megartx-dspark-private/receipt-packet

/home/yyang/projects/megartx-baseline-20260930/.venv/bin/python -S /home/yyang/projects/megartx-dspark-receipt-client/scripts/speculative_native_receipt_preflight.py --plan /home/yyang/projects/megartx-dspark-private/receipt-packet/native-receipt-plan.private.json --installed-root /home/yyang/projects/megartx-baseline-20260930/.venv/lib/python3.12/site-packages --checkpoint-manifest /home/yyang/projects/megartx-baseline-20260930/results/checkpoint-manifest.json --private-directory /home/yyang/projects/megartx-dspark-private/receipt-packet
```

The checkpoint preflight streams a full rehash of the existing manifest's files,
including both shards and tokenizer/template. It loads no model and downloads
nothing. No full shard rehash was run during this milestone; only the existing
manifest structure/revision and installed client source were inspected by SSH.
For a synchronous frontend choose `--client-mode sync` at freeze; this changes the
plan digest and requires its own exact-source acceptance.

After all source blockers are resolved by reviewed committed code, exact-head CI,
parent acceptance and an assigned sole slot, the proposed future invocation is:

```text
/home/yyang/projects/megartx-baseline-20260930/.venv/bin/python /home/yyang/projects/megartx-dspark-receipt-client/scripts/speculative_native_receipt_client.py --execute --plan /home/yyang/projects/megartx-dspark-private/receipt-packet/native-receipt-plan.private.json --authorization /home/yyang/projects/megartx-dspark-private/receipt-packet/authorization.private.json --private-directory /home/yyang/projects/megartx-dspark-private/receipt-run-001 --checkpoint-manifest /home/yyang/projects/megartx-baseline-20260930/results/checkpoint-manifest.json
```

The supervisor creates the fresh run directory and internal child argv with an
inherited control descriptor. The child independently revalidates the plan and
authorization, installed sources, and exact environment before the resource gate
and runtime imports. The frozen `engine_kwargs`/`engine_argv` select the local
pinned checkpoint, BF16, eager TP1/PP1/DP1, one sequence, 256 batched tokens,
2 GiB existing cache allocation, FlashInfer attention/CUTLASS MoE, no prefix
caching, native worker extension, and `--no-async-scheduling`. The environment is
generated by `environment()` with offline model access, exclusive plugin allowlist,
native corrected scales, spawn, bounded compiler parallelism, and private startup
proof paths. It contains no M1 preparation or prefill diagnostic capability.

## Resource domains and remaining source blockers

The first receipt executes **zero additional diagnostic target forwards**.
Existing startup frames must remain within their separate 11-frame allowance.
The six forced expert fixture pairs and their actual correction rows are counted
separately. Existing model/cache/startup workspaces stay a reference baseline.
Measured private COW pages, native head/hidden, metadata, temporary work and
cached slack are charged against the existing 8 MiB incremental verifier cap.
This client never runs the later 85-call verifier experiment.

The proposed owned supervisor reuses the committed subreaper/PID-start-time/pidfd
controls. It samples aggregate compiler RSS under 2 GiB and a shared 300-second
compiler clock without the M256 benchmark/timing gate. GPU free memory must
retain 2 GiB and host available RAM 8 GiB before acquisition and during the run;
unknown telemetry stops owned work. Startup/receipt has a 900-second deadline,
the full owned lifecycle including cleanup has an 1800-second limit, and 60
seconds are reserved for teardown. Runtime logs are capped at 8 MiB before write.
The child applies an 8 MiB kernel per-file bound to startup evidence before
imports. A 64 MiB run evidence disk reserve is separate from these RAM/GPU limits
and is not an allocation-coverage claim.

The shared registration/evidence-hook integration is the remaining frozen source
blocker in this branch. The composed exact
correction removes global snapshot materialization, bounds targeted Python
records before copies, streams canonical hashing, and reports segment/block
snapshot coverage as unavailable. Its selected-device native Torch counter query
still materializes a private-pool dictionary before Python can guard its count.
That query's preallocation/peak host bound remains explicitly unavailable.
The writer's 8 MiB serialized byte cap cannot bound earlier native collector host
acquisition; the 8 GiB free-host reserve cannot substitute for that acquisition
bound. No absent snapshot/private-pool observation is converted into zero
allocations. Independent correction review passed 71 focused, 354 scaffold
(one skip), 20 prior and 10 new controls, with all six correction CI checks green,
as relayed by the parent from `task-5/review-bounded-summary.json`. The parent
accepts the prospective single selected-device native query under fresh exclusive
startup, host reserve checks before/after, bounded serialized evidence, and the
existing cleanup/resource guards. This is acceptance of a measured/monitored
limitation, not a hard heap bound or fit promotion. Only the exact committed
correction authorized by the parent was composed; no sibling uncommitted content
was read or incorporated.

`validate_authorization` rejects the remaining blocker before device queries, imports,
output-directory acquisition, or model startup. No supplied boolean or edited
authorization file can override them. The collector guard binds the corrected
receipt source SHA-256 and both committed ancestries; it records partial Python
acquisition protection and the exact independent review/parent scope acceptance
references, without claiming a native host peak bound. After shared integration
and the exact client/plan are reviewed, the plan generator/schema must be deliberately updated
and re-frozen at the accepted integrated HEAD. This packet makes no correctness,
quality, timing, speed, workspace coverage, or live-fit claim.

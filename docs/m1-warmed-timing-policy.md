# Prospective warmed-request decode timing

This additive policy implements the owner's 4 October 2026 selection of warmed-request timing. Its source base is `ca6c385562aff30e91e8bb6a7f4228c9d766c30d`. It does not revise historical evidence, qualify a prior failed run, authorize a GPU experiment, or establish accepted latency, quality, graphs or a performance improvement.

## Three separate authorities

- The original `megartx-m1-eager-benchmark-v1` remains the default. Its lifetime compiler gate is unchanged, including the separately explicit pinned metadata-help distinction
- `megartx-m1-decode-profile-plan-v1` remains an untimed diagnostic. Its accounted-compiler rule cannot admit timing
- The new `megartx-m1-warmed-eager-benchmark-v1` binds `sampled_compiler_quiescence_after_all_warmups_v1` into its plan digest. Preparation, launcher and client must explicitly agree through `--m1-warmed-timing`. A missing flag, mismatched schema or diagnostic combination fails before inference

Warmed plans use the complete existing profile-driver source inventory because the exact build/package/AOT file verifier is shared. Only that verifier is reused; neither profile admission nor profiling is enabled. Controller and driver hashes, build receipt, source HEAD and private AOT must all be frozen anew for this source. Old receipts are not upgraded.

## Boundary and process evidence

All startup and every scheduled warmup must complete before the first measurement. The client retains validated response-close evidence for every warmup and writes a digest-bound boundary record only after all requests complete. Request timestamps use the same host's `monotonic_ns` clock immediately before the actual POST and after the response context closes. JSON encoding and marker writes remain outside each request measurement. The measured window is exactly the first measured POST through the final measured response close, including every inter-request gap. It is not narrowed to selected tokens, callbacks, markers or GPU events.

The launcher separately records server readiness, client launch and client return. Admission verifies the full ordered request schedule, warmup completion, input/dispatch transcripts, outputs, response timing identities, boundary digest and launcher brackets. The final one-token drain remains outside the window.

The existing nominal 50 ms ownership sampler runs across startup, warmups, measurements and cleanup. Its retained snapshot intervals must bracket the entire measured window. A snapshot or neighboring pair spanning more than 250 ms rejects as uncertain coverage. This conservative coverage ceiling is source-bound. Sampling cannot prove absence of a short-lived process between observations.

A sampled compiler or compiler descendant may be excluded from this window only when its identity has a verified terminal observation or disappearance strictly before the first measured POST. A startup zombie with unavailable argv/executable remains unknown; it is never relabeled as a metadata probe. An identity still active across the opening boundary, any compiler observed in a request or gap, any first-seen identity at or after the opening boundary (including a late cleanup observation), a reappearance after disappearance, or incomplete/contradictory evidence rejects. The policy intentionally does not infer a late process's end from age or recovered argv. It may therefore reject work first seen after the measured window too.

All previous compiler classification and unknown/work history remains sticky. Successful warmed receipts disclose completed premeasurement identities and the unknown/work subset. They do not turn into successful lifetime-policy receipts.

## Whole-run invariants and replay

Compiler aggregate RSS, shared compiler time, GPU/host headroom, PID/start-time ownership and PID-safe cleanup retain their whole-run scope. Source, installed pins, bridge, build receipt, AOT loader/shims/modules and receipts receive initial byte verification, sampled file-version checks and final byte verification. Restoring changed bytes does not clear a recorded failure. Repository HEAD and installed package/build identities are checked before launch and at final verification. The optional metadata-help files retain their existing initial, sampled and final checks.

Cleanup is performed independently before timing admission. Resource or final-file failure rejects even when process cleanup succeeds. Cleanup errors do not replace an existing client/run error. A final-verification exception is recorded without skipping remaining metadata verification or cleanup reporting.

The new `megartx-m1-warmed-timing-admission-v1` receipt binds the plan, request/dispatch records, boundary files, launch manifest, ownership/cleanup records, telemetry and exit statuses. Summary replay requires this receipt, recomputes boundary and overlap checks, and compares persisted ownership evidence. Warmed output uses `megartx-m1-warmed-eager-summary-v1` and explicitly names the policy and sampling limits. Strict replay rejects warmed launch/ownership evidence even if a boundary file is removed. Diagnostic artifacts and failed exit statuses remain rejecting.

These checks detect inconsistent or modified packets relative to the admission receipt; they are not a signed attestation against an adversary able to rewrite every local evidence file and receipt.

## CPU validation and experiment boundary

`tests/test_m1_warmed_timing.py` exercises the actual client entrypoint and HTTP completion loop, launcher CLI and dispatch/final-admission blocks, process overlap evaluator, real temporary-file version/hash guard and summary replay. The substitutes are local HTTP/SSE, `/proc`, GPU queries and target build/package files. No model or CUDA imports, target-machine access or GPU experiments are needed.

Coverage includes missing/forged/reordered boundaries, incomplete warmups, schema/CLI confusion, cross-boundary and gap compiler work, uncertain terminal and late observations, sticky unknown history, NaN/malformed resource evidence, sampled-coverage gaps, source/AOT drift and restored bytes, final hash mismatch, cleanup failure and primary-error preservation. Existing lifetime and diagnostic tests remain separate regressions.

Before any future experiment, freeze and independently review the exact composed commit, regenerate source-bound build/AOT/plan evidence, and follow the separately authorized sole-owner experiment procedure. CPU success here is not permission to run or evidence of a GPU result.

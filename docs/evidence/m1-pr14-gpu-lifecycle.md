# PR14 native and GPU lifecycle evidence

## Scope

This record covers the source-bound native build and one fused, controlled, cached, capture-free request with the validation-only external observer at tested runtime commit [`80b412e`](https://github.com/yindavidyang/megartx/commit/80b412e029b74c4cef7ee048cc1b362afc97d2e9), tree `30647957fec2c831cb0e5d8ed07cc90d29561066`. The run used synchronous scheduling and no route controls. PR14 is based on PR13 head `3ed6ee85`; the verified PR11 and PR12 heads are ancestors of the integrated PR13 branch through combined merge `d002a11`.

The [scalar record](m1-pr14-gpu-lifecycle.json) contains source, build, process, trace, comparator, and private-packet identities. The [prior paired GPU evidence](m1-gpu-validation-bd09109.json) retains the stock/fused observer and captured comparison results. The [capture-free execution plan](../m1-capture-free-execution.md) documents the mode and its retained validation costs.

## Native build

The exact tested runtime source built successfully in 6.134 seconds with peak aggregate RSS of 731,369,472 bytes, below the 2 GiB and 300-second limits. Required exports, lease controls, binding controls, installed pins, and source binding passed. The native bridge SHA-256 is `d897e1f3a3b296a445003f7d297190a180ba29cdb6aa4d9b5336322a43291ced`; the source-bound build receipt SHA-256 is `12a4a74a17dd9e095856e985ef1673af803bd110bbe369fd7a212d0c7f3c5022`.

## Fused observer request and comparison

The single request completed with 30 native calls and 30 correlated fused candidate launches. Fused spans had zero incumbent map/expand calls and two incumbent GEMMs per M1 call (60 total). CUPTI stream-ID mapping and request/thread/stream binding passed. Native payloads, physical masks/descriptors, and all 99 model arrays matched the preserved fused captured control bit-for-bit: 60 route arrays, 6 stage arrays, 3 logit arrays, and 30 KV arrays.

The independent process probe observed `spawn`; EngineCore PID 247455 was a child of API-server PID 247273. The pre-dispatch owner gate passed, EngineCore owned callback registration, callback unregistration passed, and the owned process group was fully cleaned up. A fresh observer-off initialization also passed: no observer module import, sidecar, CUDA initialization, or dispatched request.

The first startup attempt failed closed before dispatch because the installed adapter did not match the tested controller/native sources. The exact-source task-local adapter passed binding checks and the retry completed. Both receipts are retained in the private evidence packet.

## Relationship to the earlier paired proof

The earlier combined-source stock/fused observer comparison remains available as scalar evidence at [`m1-gpu-validation-bd09109.json`](m1-gpu-validation-bd09109.json), report SHA-256 `65198778abc58c3826e402782a5d0edb2a1b726436e37fd8c19972b8ecd2281a`. That pair reported zero stock and 30 fused candidate launches per request, with bit-exact model arrays. PR14 did not repeat a stock observer request; the reviewed PR14 lifecycle scope was one fused request because the numerical/native path was unchanged and the prior paired proof was retained. The PR14 request establishes lifecycle ownership for its recorded run; it does not retroactively add process receipts to the earlier runs.

## Limits and private evidence

The observer deliberately perturbs execution through copies, synchronization, serialization, and profiling. These observations do not qualify timing, performance, model quality, natural routing, CUDA graphs, or asynchronous scheduling. The 8.0 MiB scratch bound and 2 GiB / 300-second compiler limits are resource gates, not performance claims.

The raw packet remains private and outside the repository. Its archive SHA-256 is `ff9f22e4f77a46907126d3cf91dafd577cf0612e7a2768af66902dcf677acea8`; the packet manifest SHA-256 is `08460a6a496472e72f7e4dd220888552082d31c582b1af8cd1c52fd74960d439`. Only this sanitized scalar summary is committed; prompts, tensors, payloads, traces, and logs are not.

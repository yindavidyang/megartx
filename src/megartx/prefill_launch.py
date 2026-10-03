"""Default-off serialized prefill launcher and frozen CPU handoff packet.

The concrete collection pipeline accepts a narrow existing lifecycle provider;
no shell, SSH, dynamic import, model download or CUDA import occurs here. The
checked-in CLI unconditionally refuses live execution before touching a provider.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Protocol

from . import prefill_runner as runner
from .prefill_collect import CHUNKS, PrefillCollector, _bounded

BASE = "0581f09259d18044ffb5f9e1363d0760f37ddd4f"
CAPS = {"max_build_rss_bytes": 2 * 2**30, "max_build_seconds": 300,
        "host_free_floor_bytes": 8 * 2**30, "gpu_free_floor_bytes": 2 * 2**30,
        "max_extra_scratch_bytes": 8 * 2**20, "memory_reserve_bytes": 2 * 2**30,
        "max_trace_bytes": 8 * 2**20,
        "max_requests": 8, "max_wall_seconds": 1800, "max_gpu_jobs": 1}
REQUIRED = {"source_review", "launcher_review", "gpu_scope", "ownership", "cleanup",
            "resource", "environment", "dispatch", "numerical", "decode_control", "fit"}
FILES = {"src/megartx/prefill_collect.py", "src/megartx/prefill_kv.py", "src/megartx/prefill_observe.py",
         "src/megartx/prefill_launch.py", "tests/test_prefill_collect.py", "tests/test_prefill_launch.py",
         "docs/prefill/live-protocol.md", "docs/prefill/runtime-sites.json"}


class LifecycleProvider(Protocol):
    """Implemented/qualified by the shared launcher owner, not a module path.

    No timing observer, token copies or profiler can be active for timing jobs.
    observe_request must call collector begin/layer/end/handoff/bootstrap at the
    actual EngineCore boundaries, provide native cache observations, and return
    the PR19 record. The provider runs the real client SSE boundary on the same
    identified monotonic host clock, with fresh process state for cold timing.
    """
    def describe_contract(self) -> dict: ...
    def admit_resources(self, bounds: dict) -> dict: ...
    def startup(self, job: dict) -> dict: ...
    def initialize(self, job: dict) -> dict: ...
    def observe_request(self, job: dict, payload: dict, collector: PrefillCollector) -> dict: ...
    def timing_request(self, job: dict, payload: dict) -> dict: ...
    def drain(self) -> None: ...
    def cleanup(self) -> dict: ...
    def poison(self) -> None: ...
    def records_provenance(self) -> dict: ...
    def host_clock_id(self) -> str: ...


def load_packet(root, plan_path=None, binding_path=None):
    root = Path(root).resolve()
    docs = root / "docs/prefill"
    manifest, plan = runner.load_inputs(docs / "runner-plan.json", docs / "profile-plan.json",
        docs / "source-binding.json", docs / "runner-binding.json", root)
    packet_path = plan_path or docs / "live-plan.json"
    binding_path = binding_path or docs / "live-binding.json"
    packet = runner.plan_contract.read_json(packet_path)
    runner.keys(packet, {"schema", "gpu_enabled", "repository_base_commit", "binding_sha256", "caps",
                         "supported_chunk_tokens", "decode_rows", "phase_order", "receipts"}, "live plan")
    runner.equal(packet["schema"], "megartx-prefill-live-plan-v1", "live schema")
    runner.equal(packet["gpu_enabled"], False, "CPU milestone keeps live admission disabled")
    runner.equal(packet["repository_base_commit"], BASE, "verified merged base")
    runner.keys(packet["caps"], CAPS, "fixed live resource caps")
    for name, value in CAPS.items():
        runner.equal(packet["caps"][name], value, "fixed live resource cap " + name)
    runner.equal(packet["supported_chunk_tokens"], [255, 256, 512, 1024, 2048, 8192], "planned sweep")
    runner.equal(packet["decode_rows"], 1, "separate decode geometry")
    runner.equal(packet["phase_order"], ["fit", "cache_correctness", "baseline"], "live phase order")
    runner.keys(packet["receipts"], REQUIRED, "live receipts")
    for key, value in packet["receipts"].items():
        if value is not None:
            runner.sha(value, key)
    runner.sha(packet["binding_sha256"], "live binding")
    binding = runner.plan_contract.read_json(binding_path)
    runner.equal(hashlib.sha256(Path(binding_path).read_bytes()).hexdigest(), packet["binding_sha256"], "live binding bytes")
    runner.keys(binding, {"schema", "repository_base_commit", "repo_files", "immutable_inputs"}, "live binding")
    runner.equal(binding["schema"], "megartx-prefill-live-source-v1", "live binding schema")
    runner.equal(binding["repository_base_commit"], BASE, "live binding base")
    runner.keys(binding["repo_files"], FILES, "live source inventory")
    runner.keys(binding["immutable_inputs"], {"docs/prefill/runner-plan.json", "docs/prefill/profile-plan.json",
        "docs/prefill/source-binding.json", "docs/prefill/runner-binding.json"}, "immutable PR19/20 input inventory")
    for name, expected in {**binding["repo_files"], **binding["immutable_inputs"]}.items():
        runner.sha(expected, name)
        relative = Path(name)
        if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError("Live source symlink: " + name)
        file = root / relative
        if not file.is_file() or file.stat().st_size > 2**20 or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError("Live source drift: " + name)
    return packet, manifest, plan


def compile_packet(packet, manifest, plan, workload, chunk):
    if workload not in {"p2048", "p8192"} or chunk not in CHUNKS[int(workload[1:])]:
        raise ValueError("Select one reviewed 2K/8K sweep cell; 32K deferred")
    # A declarative implementation overlay; preserve the immutable PR19 manifest.
    overlay = copy.deepcopy(manifest)
    overlay["launcher"]["supported_chunk_tokens"] = packet["supported_chunk_tokens"]
    protocol = runner.compile_protocol(overlay, plan, workload, chunk)
    return {"schema": "megartx-prefill-live-packet-v1", "gpu_enabled": False, "gpu_qualified": False,
            "implementation_base": BASE, "plan_sha256": runner.digest(packet),
            "code_support_is_not_dispatch_qualification": True,
            "caps": packet["caps"], "phase_order": packet["phase_order"],
            "blockers": ["GPU handoff and frozen live review pending", *protocol["blockers"],
                         *["live.receipt." + k for k, v in packet["receipts"].items() if v is None]],
            "protocol": protocol}


class SerializedLauncher:
    """Concrete dormant pipeline plus fail-closed public adapter handshake.

    The public execute method cannot enable GPU work in this CPU milestone, even
    with all self-declared receipts. _collect_admitted is exercised with fakes;
    connecting the final source-bound provider is a separate reviewed change.
    """
    def __init__(self, packet, provider_factory=None):
        self.packet, self.factory = copy.deepcopy(packet), provider_factory

    def describe_contract(self):
        return {"contract": runner.LAUNCHER_CONTRACT, "gpu_execution_available": False,
                "supported_chunk_tokens": [255, 256, 512, 1024, 2048, 8192],
                "decode_rows": 1, "provider_connected": False,
                "remaining": "exact shared lifecycle/CUPTI provider + native cache review + owner handoff"}

    def execute_serialized(self, protocol, private_prompts):
        # No factory callback, imports, resources, network or process before this gate.
        raise ValueError("GPU execution disabled: CPU milestone requires frozen source/provider review and parent slot handoff")

    def _collect_admitted(self, protocol, private_prompts, provider, clock):
        """Dormant orchestration implementation; use only after admission integration.

        It returns records only after complete owned cleanup and PR19 validation.
        A caller-provided provider is not an authorization or source attestation.
        """
        tokens = runner.validate_prompts(private_prompts, protocol)
        bounds = protocol["resource_bounds"]
        runner.keys(bounds, runner.plan_contract.RESOURCE_KEYS, "admitted resource bounds")
        for key, value in bounds.items():
            runner.equal(value, self.packet["caps"][key], "admitted resource bound " + key)
        handshake = provider.describe_contract()
        expected = {"contract": runner.LAUNCHER_CONTRACT, **protocol["launcher"]}
        runner.equal(handshake, expected, "exact lifecycle provider handshake")
        resource = provider.admit_resources(self.packet["caps"])
        runner.keys(resource, {"host_free_bytes", "gpu_free_bytes", "exclusive_slot"}, "resource admission")
        for name in ("host", "gpu"):
            runner.integer(resource[name + "_free_bytes"], self.packet["caps"][name + "_free_floor_bytes"], 2**63 - 1, name + " free floor")
        runner.equal(resource["exclusive_slot"], True, "exclusive GPU slot")
        runs, collector_evidence = [], []
        for job in protocol["jobs"]:
            startup, cleanup, error, record = None, None, None, None
            attempted = False
            try:
                attempted = True  # A partially failed startup still needs owned cleanup.
                startup = provider.startup(job)
                runner.keys(startup, {"sha256", "owned_only"}, "owned startup")
                runner.sha(startup["sha256"], "startup receipt")
                runner.equal(startup["owned_only"], True, "owned startup")
                if job["mode"] == "initialization":
                    record = provider.initialize(job)
                elif job["mode"] == "timing":
                    record = provider.timing_request(job, runner.request_payload(job, tokens[job["workload"]]))
                    if record["observer_enabled"] or record["trace_bytes"]:
                        raise ValueError("Timing job activated observation/trace")
                else:
                    collector = PrefillCollector(job, tokens[job["workload"]], protocol["cache_layout"]["layout_sha256"], clock)
                    record = provider.observe_request(job, runner.request_payload(job, tokens[job["workload"]]), collector)
                    if collector.failed or collector.active is not None or collector.index != len(job["spans"]):
                        raise ValueError("Provider returned incomplete/poisoned observed prompt")
                    runner.equal(record["forwards"], collector.forwards, "actual observed prompt ledger")
                    runner.equal(record["handoff"], collector.handoff_record, "actual observed cache handoff")
                    runner.parse_client_stream(record["client_events"], job, record["server_response_id"])
                    emitted = []
                    for event in record["client_events"]:
                        if event["data"] == "[DONE]":
                            continue
                        for choice in runner._json(event["data"])["choices"]:
                            emitted.extend(choice.get("token_ids", []))
                    runner.equal(collector.decode_input_hashes, [runner.digest([t]) for t in emitted[:255]],
                                 "actual decode inputs versus delivered output tokens")
                    if collector.decode_rows != 255:
                        raise ValueError("Observed fixed decode continuation missing 255 input rows")
                    if job["mode"] == "profile" and job["region"] == "expert":
                        runner.equal(record["profile"]["expert_rows"], collector.routes, "actual routed shape ledger")
                    if collector.bootstrap_record is None:
                        raise ValueError("Authenticated final-prompt-logit bootstrap missing")
                    for span in collector.forward_times + collector.events:
                        if not record["run_begin_ns"] <= span["host_begin_ns"] <= span["host_end_ns"] <= record["run_end_ns"]:
                            raise ValueError("Observed producer clocks outside provider run")
                    collector_evidence.append({"run_id": job["run_id"], "bootstrap": collector.bootstrap_record,
                        "event_ranges": collector.finish_events(provider.event_clock) if job["mode"] == "profile" else [],
                        "cache_frames": collector.cache.frames, "forward_times": collector.forward_times, "native_qualified": False})
                provider.drain()
            except BaseException as caught:
                error = caught
                try:
                    provider.poison()
                except BaseException as poison_error:
                    if hasattr(caught, "add_note"):
                        caught.add_note("Provider poison failed: " + str(poison_error))
                raise
            finally:
                if attempted:
                    try:
                        cleanup = provider.cleanup()
                        runner.keys(cleanup, {"sha256", "owned_only", "cleanup_complete"}, "owned cleanup")
                        runner.sha(cleanup["sha256"], "cleanup receipt")
                        runner.equal(cleanup["owned_only"], True, "owned cleanup")
                        runner.equal(cleanup["cleanup_complete"], True, "complete cleanup")
                    except BaseException as cleanup_error:
                        if error is not None:
                            if hasattr(error, "add_note"):
                                error.add_note("Owned cleanup failed: " + str(cleanup_error))
                        else:
                            raise
            record["lifecycle_receipts"] = {"startup_sha256": startup["sha256"], "cleanup_sha256": cleanup["sha256"],
                "launcher_source_manifest_sha256": protocol["launcher"]["source_manifest_sha256"],
                "ownership_scope_sha256": protocol["scope_sha256"], "owned_only": True, "cleanup_complete": True}
            # Validate each completed/cleaned job before starting the next job.
            single = {**protocol, "jobs": [job], "request_count": int(job["request_id"] is not None)}
            runner.validate_records(single, {"schema": "megartx-prefill-records-v1",
                "scope_sha256": protocol["scope_sha256"], "provenance": handshake_provenance(provider),
                "host_clock_id": provider.host_clock_id(), "runs": [record]}, private_prompts)
            runs.append(record)
        records = {"schema": "megartx-prefill-records-v1", "scope_sha256": protocol["scope_sha256"],
                   "provenance": handshake_provenance(provider), "host_clock_id": provider.host_clock_id(),
                   "runs": runs}
        # Do not manufacture kernel correlation, comparison/fit receipts or acceptance.
        summary = runner.validate_records(protocol, records, private_prompts)
        result = {**summary, "collector_evidence": collector_evidence}
        _bounded(result)
        return result


def handshake_provenance(provider):
    value = provider.records_provenance()
    runner.keys(value, {"kind", "origin_reference", "artifact_sha256", "sanitized"}, "provider provenance")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--workload", choices=("p2048", "p8192"), default="p2048")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        packet, manifest, plan = load_packet(args.root)
        result = compile_packet(packet, manifest, plan, args.workload, args.chunk)
        if args.execute_gpu:
            SerializedLauncher(packet).execute_serialized(result["protocol"], {})
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(serialized)
        else:
            print(serialized, end="")
        return 0
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, "prefill-launch: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

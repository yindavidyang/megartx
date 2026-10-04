"""Source-bound, one-request native fit diagnostic. Importing this is CPU-only."""
import hashlib
import json
import os
from pathlib import Path
import time

BASE = "cf656d6632b9f1b08019a527a9269a9ed0fb0a26"
PROPOSAL = "1661b04383c03f436550d63e514bef3a5ab8500316cbf117fc621e85ef8b4207"
REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
BOUNDS = {"prompt_tokens": 2048, "chunk_tokens": 256, "output_tokens": 256,
          "capacity_tokens": 2304, "max_requests": 1, "max_wall_seconds": 1800,
          "max_evidence_bytes": 8 << 20, "max_observer_scratch_bytes": 8 << 20,
          "gpu_free_floor_bytes": 2 << 30, "host_free_floor_bytes": 8 << 30,
          "max_build_rss_bytes": 2 << 30, "max_build_seconds": 300}
SOURCES = (
    "src/megartx/prefill_diagnostic_plan.py", "src/megartx/prefill_native.py",
    "src/megartx/loaded_engine_access.py", "src/megartx/vllm_scale_plugin.py",
    "scripts/prefill_diagnostic_client.py", "scripts/run_scale_validation.py",
    "scripts/m1_owned_processes.py", "src/megartx/nvfp4_integration.py",
    "src/megartx/nvfp4_runtime.py", "src/megartx/m1_live.py",
    "src/megartx/prefill_kv.py", "src/megartx/controlled_kv_capture.py")
INSTALLED = {
    "vllm.v1.worker.gpu_model_runner": "87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51",
    "vllm.v1.worker.block_table": "a09b8819e1417b7a186cf3472224e2138fc5d41a291c6692bb7df711b0c44259",
    "vllm.v1.kv_cache_interface": "1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb",
    "vllm.v1.core.kv_cache_utils": "2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan(path, root=None):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1 << 20:
        raise ValueError("Diagnostic plan must be a bounded regular file")
    value = json.loads(path.read_text())
    expected = {"schema", "base", "proposal_sha256", "source_head", "source_hashes",
                "checkpoint_revision", "checkpoint_identity", "tokens", "prompt_sha256", "bounds", "plan_sha256"}
    if set(value) != expected or value["schema"] != "megartx-prefill-native-plan-v1":
        raise ValueError("Unknown native diagnostic plan")
    if (value["base"] != BASE or value["proposal_sha256"] != PROPOSAL
            or value["checkpoint_revision"] != REVISION or value["bounds"] != BOUNDS
            or type(value["bounds"]) is not dict
            or any(type(v) is not int for v in value["bounds"].values())):
        raise ValueError("Native diagnostic scope/bounds changed")
    identity = value['checkpoint_identity']
    from .controlled_kv_capture import CONFIG_SHA256
    if (type(identity) is not dict or set(identity) != {'config_sha256','index_sha256','shard_stats'}
            or identity['config_sha256'] != CONFIG_SHA256 or len(identity['index_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in identity['index_sha256'])
            or type(identity['shard_stats']) is not dict or not 1 <= len(identity['shard_stats']) <= 128):
        raise ValueError('Frozen checkpoint file identity required')
    for name, stat in identity['shard_stats'].items():
        if (Path(name).name != name or set(stat) != {'size','mtime_ns'}
                or any(type(v) is not int or v <= 0 for v in stat.values())):
            raise ValueError('Bounded checkpoint shard identity required')
    tokens = value["tokens"]
    if (type(tokens) is not list or len(tokens) != 2048
            or any(type(t) is not int or not 0 <= t < 262144 for t in tokens)
            or digest(tokens) != value["prompt_sha256"]):
        raise ValueError("Expected exactly 2048 frozen private token IDs")
    unsigned = {k: v for k, v in value.items() if k != "plan_sha256"}
    if digest(unsigned) != value["plan_sha256"]:
        raise ValueError("Diagnostic plan digest changed")
    if (type(value["source_head"]) is not str or len(value["source_head"]) != 40
            or any(c not in "0123456789abcdef" for c in value["source_head"])):
        raise ValueError("Frozen source head required")
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    if set(value["source_hashes"]) != set(SOURCES) or any(
            any((root / Path(*Path(p).parts[:i])).is_symlink() for i in range(1, len(Path(p).parts)+1))
            or file_sha(root / p) != h for p, h in value["source_hashes"].items()):
        raise ValueError("Diagnostic repository source drift")
    return value


def freeze_plan(tokens, source_head, root, checkpoint):
    value = {"schema": "megartx-prefill-native-plan-v1", "base": BASE,
             "proposal_sha256": PROPOSAL, "source_head": source_head,
             "source_hashes": {p: file_sha(Path(root) / p) for p in SOURCES},
             "checkpoint_revision": REVISION, "checkpoint_identity": checkpoint, "tokens": tokens,
             "prompt_sha256": digest(tokens), "bounds": BOUNDS.copy()}
    value["plan_sha256"] = digest(value)
    return value


def require_clearance(path, plan):
    """Owner-created review/slot receipt is required; a fit receipt is output."""
    value = json.loads(Path(path).read_text())
    if (set(value) != {"schema", "plan_sha256", "source_head", "cpu_review_passed", "parent_gpu_slot_clearance"}
            or value["schema"] != "megartx-prefill-native-clearance-v1"
            or value["plan_sha256"] != plan["plan_sha256"]
            or value["source_head"] != plan["source_head"]
            or value["cpu_review_passed"] is not True
            or value["parent_gpu_slot_clearance"] is not True):
        raise ValueError("Exact CPU review and parent GPU slot clearance required")
    return value


def server_args():
    return ["--no-async-scheduling", "--enable-chunked-prefill", "--disable-hybrid-kv-cache-manager"]


class Evidence:
    """One cross-process 8 MiB budget, checked before every append/publication.

    This directory contains new diagnostic evidence. Historical startup fixture
    traces and original model workspace remain separately ledgered resources.
    """
    def __init__(self, directory, limit=8 << 20):
        self.directory, self.limit = Path(directory), limit
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def write(self, name, value, append=False):
        import fcntl
        if Path(name).name != name or name.startswith("."):
            raise ValueError("Evidence filename must be a literal basename")
        data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with (self.directory / ".budget-lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            paths = list(self.directory.iterdir())
            if any(p.is_symlink() for p in paths):
                raise ValueError("Evidence symlink refused")
            size = sum(p.stat().st_size for p in paths if p.is_file())
            if size + len(data) > self.limit:
                raise ValueError("Native evidence overflow rejected before write")
            path = self.directory / name
            with path.open("ab" if append else "xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise RuntimeError("Native diagnostic 1800 second wall budget expired")
    return value


def publish_fit(evidence, plan, ownership):
    """Only successful observed request plus owned cleanup can publish fit."""
    directory = evidence.directory
    read = lambda name: json.loads((directory/name).read_text())
    observer, client, loaded, geometry = [read(n) for n in ('observer.json', 'client.json', 'loaded.json', 'geometry.json')]
    if (ownership.get('cleanup_complete') is not True or ownership.get('failure') is not None
            or ownership.get('owned_identities_remaining') or ownership.get('owned_gpu_pids_remaining')
            or ownership.get('cleanup_errors') or observer.get('plan_sha256') != plan['plan_sha256']
            or client.get('plan_sha256') != plan['plan_sha256'] or loaded.get('plan_sha256') != plan['plan_sha256']
            or observer.get('output_ids_sha256') != client.get('output_ids_sha256')
            or observer.get('engine_request_id') != client.get('response_id', '')+'-0'
            or observer.get('status') != 'request_observed' or client.get('status') != 'complete'
            or observer.get('prompt_frames') != 8 or observer.get('decode_input_rows') != 255
            or observer.get('emitted_outputs') != 256 or observer.get('committed_length') != 2303
            or loaded.get('mutable_lease_granted') is not False
            or geometry.get('physical_policy') != 'full_context' or geometry.get('capacity_tokens') != 2304
            or geometry.get('actual_owned_page_ranges_disjoint') is not True
            or observer.get('numerical_qualified') is not False or client.get('numerical_qualified') is not False
            or observer.get('performance_qualified') is not False or client.get('performance_qualified') is not False):
        raise ValueError('Native fit requires exact request observations and complete owned cleanup')
    evidence.write('fit.json', {'schema': 'megartx-prefill-native-fit-v1', 'status': 'observed_fit',
        'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
        'prompt_tokens': 2048, 'chunk_tokens': 256, 'capacity_tokens': 2304,
        'emitted_outputs': 256, 'decode_input_rows': 255, 'cleanup_complete': True,
        'source_and_owner_bound': True, 'cache_and_sample_handoff_observed': True,
        'numerical_qualified': False, 'performance_qualified': False,
        'independent_native_correctness_pending': True,
        'startup_reference_workspace_separate_from_observer_scratch': True,
        'legacy_fp64_projection_floor_bytes': 15859712,
        'evidence_files_sha256': {p.name: file_sha(p) for p in directory.iterdir()
                                  if p.is_file() and not p.name.startswith('.') and p.name != 'request.json'}})


def checkpoint_identity(checkpoint):
    """CPU file identity only; full-shard byte rehash is not implied."""
    from .controlled_kv_capture import CONFIG_SHA256
    checkpoint = Path(checkpoint)
    if file_sha(checkpoint/'config.json') != CONFIG_SHA256:
        raise ValueError('Checkpoint config differs from selected immutable model')
    index = checkpoint/'model.safetensors.index.json'
    if index.stat().st_size > 4 << 20:
        raise ValueError('Checkpoint index exceeds bounded identity scope')
    names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    if not 1 <= len(names) <= 128 or any(Path(n).name != n for n in names):
        raise ValueError('Checkpoint shard index scope changed')
    return {'config_sha256': CONFIG_SHA256, 'index_sha256': file_sha(index),
            'shard_stats': {n: {'size': (checkpoint/n).stat().st_size,
                               'mtime_ns': (checkpoint/n).stat().st_mtime_ns} for n in names}}

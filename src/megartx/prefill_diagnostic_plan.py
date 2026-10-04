"""Source-bound, one-request native fit diagnostic. Importing this is CPU-only."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

BASE = "cf656d6632b9f1b08019a527a9269a9ed0fb0a26"
PROPOSAL = "1661b04383c03f436550d63e514bef3a5ab8500316cbf117fc621e85ef8b4207"
REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
CHECKPOINT_INDEX_MAX_BYTES = 8 << 20  # CPU input bytes; decoded heap uses the separate host reserve.
BOUNDS = {"prompt_tokens": 2048, "chunk_tokens": 256, "output_tokens": 256,
          "capacity_tokens": 2304, "max_requests": 1, "max_wall_seconds": 1800,
          "max_evidence_bytes": 8 << 20, "max_observer_gpu_scratch_bytes": 8 << 20,
          "gpu_free_floor_bytes": 2 << 30, "host_free_floor_bytes": 8 << 30,
          "max_build_rss_bytes": 2 << 30, "max_build_seconds": 300}
SOURCES = (
    "src/megartx/prefill_diagnostic_plan.py", "src/megartx/prefill_native.py",
    "src/megartx/loaded_engine_access.py", "src/megartx/vllm_scale_plugin.py",
    "scripts/prefill_diagnostic_client.py", "scripts/run_scale_validation.py",
    "scripts/m1_owned_processes.py", "src/megartx/nvfp4_integration.py",
    "src/megartx/nvfp4_runtime.py", "src/megartx/m1_live.py",
    "src/megartx/prefill_kv.py", "src/megartx/controlled_kv_capture.py",
    "src/megartx/m1_execution.py", "src/megartx/prefill_plan.py",
    "src/megartx/prefill_runner.py", "src/megartx/prefill_collect.py",
    "src/megartx/prefill_runner_binding.py")
INSTALLED = {
    "vllm.config.vllm": "956b812e5a719bcbfa3a3958801361b38b9c928c96b8c073311bbb96376dfb7a",
    "vllm.envs": "fbd370b2f56ff798d373e85705c9e044ccae893ef974f2800eec4ef6f2b4fb7f",
    "vllm.v1.worker.gpu_worker": "6994436e4547c0996ab555b3e814e8ccea26b650dfc1b378b97d640745309347",
    "vllm.v1.worker.gpu_model_runner": "87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51",
    "vllm.v1.worker.gpu.model_runner": "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
    "vllm.v1.worker.gpu.block_table": "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a",
    "vllm.v1.worker.gpu.input_batch": "d5dd956eb319bd69dd9e762047833d81ad40b21fce619fcb2133759e73620694",
    "vllm.v1.worker.gpu.attn_utils": "b9b81e59dda2720b1f9c434471381f879228832443e52756e469b4ad4905888f",
    "vllm.v1.worker.gpu.states": "99418f5df43ca612ded72609fee011620b065b2cab2f249ccb387096bf4ae71a",
    "vllm.v1.worker.gpu.sample.sampler": "832c9945d201a1e730ff9b4b52076aa83bd4a5064e1e4cfea952cf84d4a8d1f0",
    "vllm.v1.worker.gpu.sample.output": "d6e298c0f197d487a8faaf7200f500fa83df1d7a4bbf4339cadd6a18008f7f9c",
    "vllm.v1.worker.gpu.model_states": "07815d0b788fc88185772f581bbf892a675259f9288375571f29887622f8d93c",
    "vllm.v1.worker.gpu.model_states.default": "bb614f814780f052e869f9acf31dfbfa68718ee7dc6252eefa901b5730ee8a2a",
    "vllm.v1.worker.gpu.model_states.interface": "5675b6fedc7403a9ab410b1e5565974e78fac23c42ab4253ae1da52d6e09b594",
    "vllm.v1.worker.utils": "0ca3ec6bf4d20076145b7fe64e564962adebedc8f75a856b517ba0b2f2d77acf",
    "vllm.v1.kv_cache_interface": "1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb",
    "vllm.v1.core.kv_cache_utils": "2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2"}
RUNNER_POLICY = {"runner_module": "vllm.v1.worker.gpu.model_runner",
                 "runner_class": "GPUModelRunner", "resolved_v2": True,
                 "selector_env": None, "selection_override_injected": False}


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
                "checkpoint_revision", "checkpoint_identity", "tokens", "prompt_sha256", "bounds", "plan_sha256",
                "adapter_site", "runner_policy"}
    if (set(value) != expected or value["schema"] != "megartx-prefill-native-plan-v3"
            or value['runner_policy'] != RUNNER_POLICY
            or digest(value['runner_policy']) != digest(RUNNER_POLICY)):
        raise ValueError("Unknown native diagnostic plan")
    if (value["base"] != BASE or value["proposal_sha256"] != PROPOSAL
            or value["checkpoint_revision"] != REVISION or value["bounds"] != BOUNDS
            or type(value["bounds"]) is not dict
            or any(type(v) is not int for v in value["bounds"].values())):
        raise ValueError("Native diagnostic scope/bounds changed")
    if (type(value['adapter_site']) is not str or not Path(value['adapter_site']).is_absolute()
            or str(Path(value['adapter_site']).resolve()) != value['adapter_site']):
        raise ValueError('Frozen canonical adapter module origin required')
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


def freeze_plan(tokens, source_head, root, checkpoint, adapter_site=None):
    value = {"schema": "megartx-prefill-native-plan-v3", "base": BASE,
             "proposal_sha256": PROPOSAL, "source_head": source_head,
             "source_hashes": {p: file_sha(Path(root) / p) for p in SOURCES},
             "checkpoint_revision": REVISION, "checkpoint_identity": checkpoint, "tokens": tokens,
             "prompt_sha256": digest(tokens), "bounds": BOUNDS.copy(),
             "adapter_site": str(Path(adapter_site or Path(root)/'src').resolve()),
             "runner_policy": RUNNER_POLICY.copy()}
    value["plan_sha256"] = digest(value)
    return value


def verify_adapter_sources(plan):
    """Bind executed package modules to one frozen adapter site and byte vector.

    CPU-only imports are limited to the explicit project helpers. A repository
    snapshot alone cannot certify a separately installed adapter's code.
    """
    origins = {}
    for relative in SOURCES:
        if not relative.startswith('src/megartx/'):
            continue
        name = 'megartx.' + Path(relative).stem
        module = importlib.import_module(name)
        expected = Path(plan['adapter_site'])/'megartx'/Path(relative).name
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        actual = getattr(module, '__file__', None)
        if (not origin or not actual or Path(origin).resolve() != expected
                or Path(actual).resolve() != expected or expected.is_symlink()
                or file_sha(expected) != plan['source_hashes'][relative]):
            raise ValueError('Executing adapter module source/origin drift: ' + name)
        origins[name] = {'origin': str(expected), 'sha256': plan['source_hashes'][relative]}
    return origins


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
    from .prefill_runner_binding import validate_binding
    binding = validate_binding(plan, directory, require_live=False)  # Owner is already cleaned up.
    scratch = observer.get('observer_gpu_scratch', {})
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
            or loaded.get('identity', {}).get('owner_pid') != binding['owner_pid']
            or loaded.get('identity', {}).get('owner_start_ticks') != binding['owner_start_ticks']
            or geometry.get('physical_policy') != 'full_context' or geometry.get('capacity_tokens') != 2304
            or geometry.get('actual_owned_page_ranges_disjoint') is not True
            or scratch.get('domain') != 'incremental_gpu_allocator_bytes'
            or scratch.get('cap_bytes') != plan['bounds']['max_observer_gpu_scratch_bytes']
            or type(scratch.get('measured_phases')) is not int or scratch['measured_phases'] < 789
            or any(type(scratch.get(k)) is not int or not 0 <= scratch[k] <= scratch['cap_bytes']
                   for k in ('managed_tensor_simultaneous_peak_bytes', 'measured_phase_allocator_increment_peak_bytes',
                             'measured_phase_reserved_increment_peak_bytes'))
            or scratch.get('counter_policy') != 'process_global_resets_with_explicit_runwide_peak_preservation'
            or any(type(scratch.get(k)) is not int or scratch[k] < 0
                   for k in ('runwide_allocator_allocated_peak_bytes', 'runwide_allocator_reserved_peak_bytes'))
            or scratch.get('host_heap_excluded') is not True
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
        'observer_gpu_scratch': scratch,
        'host_heap_budget': 'separate_host_available_reserve',
        'evidence_files_sha256': {p.name: file_sha(p) for p in directory.iterdir()
                                  if p.is_file() and not p.name.startswith('.') and p.name != 'request.json'}})


def checkpoint_identity(checkpoint):
    """CPU file identity only; full-shard byte rehash is not implied."""
    from .controlled_kv_capture import CONFIG_SHA256
    checkpoint = Path(checkpoint)
    if file_sha(checkpoint/'config.json') != CONFIG_SHA256:
        raise ValueError('Checkpoint config differs from selected immutable model')
    index = checkpoint/'model.safetensors.index.json'
    if index.stat().st_size > CHECKPOINT_INDEX_MAX_BYTES:
        raise ValueError('Checkpoint index exceeds bounded identity scope')
    names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    if not 1 <= len(names) <= 128 or any(Path(n).name != n for n in names):
        raise ValueError('Checkpoint shard index scope changed')
    return {'config_sha256': CONFIG_SHA256, 'index_sha256': file_sha(index),
            'shard_stats': {n: {'size': (checkpoint/n).stat().st_size,
                               'mtime_ns': (checkpoint/n).stat().st_mtime_ns} for n in names}}

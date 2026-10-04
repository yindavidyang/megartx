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
# Bind the default API/internal-ID and sampler-to-SSE transport, not merely
# the model runner; verify_installed_files checks all files without importing.
TRANSPORT_FILES = {
    "vllm.entrypoints.openai.completion.serving": "75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527",
    "vllm.entrypoints.openai.completion.protocol": "5f7930635fd3924826015211032fb298206b27c7f04cd6cd2224bb128e4a3a50",
    "vllm.entrypoints.serve.engine.serving": "70143311acc2230b925e7f3603b8b9314f029ff5e98d5af990b3bdca691331f1",
    "vllm.entrypoints.generate.base.serving": "35ec53f63e86a8c1725728911db0df44eb3101d8cd65e9b2f39d048956977bad",
    "vllm.v1.engine.input_processor": "c8abd8e4e14d99929d45a7ff3b2a74898269c5e2a7fa30b826623ee85e4ce813",
    "vllm.v1.engine.async_llm": "52cbf404f1dbc99c6bebf4a50a7eb18ff842a1f53106a45f365523192d816be1",
    "vllm.v1.engine.output_processor": "e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329",
    "vllm.v1.engine.parallel_sampling": "bf3f0a3c6640aaf706d1b6f7ec3a8e09a970f3dd9e7e96fa6a901779ea84f61d",
    "vllm.v1.worker.gpu.async_utils": "77e17a4570ead2be30ae9b00888cf077a3b873018c12ca0549b853bfda02c1ba",
    "vllm.v1.core.sched.scheduler": "c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884",
    "vllm.utils": "c58b3f45deeb98bccca22c945526fd15d483a2f5eb8a3c62cf7df218b081885f",
    "vllm.outputs": "346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425",
}
RUNNER_POLICY = {"runner_module": "vllm.v1.worker.gpu.model_runner",
                 "runner_class": "GPUModelRunner", "resolved_v2": True,
                 "selector_env": None, "selection_override_injected": False}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def api_request_id(plan):
    return 'megartx-prefill-' + plan['plan_sha256'][:20]


def native_request_identity(plan, engine_request_id):
    """Exact default vLLM n=1 identity; never strip or ignore an opaque suffix.

    Completion serving appends prompt index 0 to its cmpl- response ID, then
    InputProcessor.assign_request_id appends random_uuid()[:8]. The observer
    binds the full internal ID to every actual InputBatch and continuation.
    """
    response_id = 'cmpl-' + api_request_id(plan)
    external_id = response_id + '-0'
    prefix = external_id + '-'
    if (type(engine_request_id) is not str or len(engine_request_id) != len(prefix)+8
            or not engine_request_id.startswith(prefix)
            or any(c not in '0123456789abcdef' for c in engine_request_id[len(prefix):])):
        raise ValueError('Actual internal request ID violates source-bound single-prompt UUID8 mapping')
    return {'schema': 'megartx-prefill-request-identity-v1',
            'policy': 'completion_prompt0_default_random_uuid8_n1',
            'api_request_id': api_request_id(plan), 'response_id': response_id,
            'external_request_id': external_id, 'engine_request_id': engine_request_id}


def native_ledger_comparison(plan, client, observer, stream_facts):
    """Bounded field-level diagnostics without prompt, token IDs, or raw IDs."""
    checks = []
    def describe(value, private=False):
        result = {'type': type(value).__name__}
        if private or type(value) not in (bool, int, type(None)):
            result['sha256'] = digest(value)
            if type(value) in (str, list, dict): result['length'] = len(value)
        else:
            result['value'] = value
        return result
    def equal(field, actual, expected, private=False):
        checks.append({'field': field, 'ok': type(actual) is type(expected) and actual == expected,
                       'observed': describe(actual, private), 'expected': describe(expected, private)})
    response_id = 'cmpl-' + api_request_id(plan)
    for key, expected in {'schema': 'megartx-prefill-native-client-v1', 'status': 'complete',
            'plan_sha256': plan['plan_sha256'], 'response_id': response_id,
            'stream_done': True, 'finish_reason': 'length', 'emitted_outputs': 256,
            'numerical_qualified': False, 'performance_qualified': False}.items():
        equal('client.'+key, client.get(key), expected)
    events = client.get('sse_events')
    equal('client.sse_events.bounded', type(events) is int and 3 <= events <= 300, True)
    checks[-1]['observed']['event_count'] = describe(events)
    usage = client.get('usage')
    equal('client.usage.keys', sorted(usage) if type(usage) is dict else None,
          ['completion_tokens', 'prompt_tokens', 'total_tokens'])
    for key, expected in {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304}.items():
        equal('client.usage.'+key, usage.get(key) if type(usage) is dict else None, expected)
    for key, expected in {'schema': 'megartx-prefill-native-observation-v1',
            'status': 'request_observed', 'plan_sha256': plan['plan_sha256'],
            'prompt_frames': 8, 'decode_input_rows': 255, 'emitted_outputs': 256,
            'committed_length': 2303, 'numerical_qualified': False,
            'performance_qualified': False}.items():
        equal('observer.'+key, observer.get(key), expected)
    actual_hash = observer.get('output_ids_sha256')
    expected_hash = client.get('output_ids_sha256')
    equal('output_ids_sha256', actual_hash, expected_hash)
    for side, value in (('observed', actual_hash), ('expected', expected_hash)):
        if type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value):
            checks[-1][side]['value'] = value
    equal('client.output_ids_sha256.format', type(expected_hash) is str and len(expected_hash) == 64
          and all(c in '0123456789abcdef' for c in expected_hash), True)
    engine_id = observer.get('engine_request_id')
    try:
        identity = native_request_identity(plan, engine_id)
    except ValueError:
        identity = None
    equal('observer.engine_request_id.mapping', identity is not None, True)
    equal('observer.request_identity', observer.get('request_identity'), identity)
    equal('client.request_identity', client.get('request_identity'), identity)
    equal('client.engine_request_id', client.get('engine_request_id'), engine_id, private=True)
    # Report the exact compared identity semantics without disclosing the ID.
    prefix = response_id + '-0-'
    checks[-4]['observed'].update({'engine_id': describe(engine_id, True),
        'external_prefix_matches': type(engine_id) is str and engine_id.startswith(prefix),
        'suffix_length': len(engine_id)-len(prefix) if type(engine_id) is str else None,
        'suffix_is_lower_hex': type(engine_id) is str and len(engine_id) == len(prefix)+8
            and all(c in '0123456789abcdef' for c in engine_id[len(prefix):])})
    # The separately persisted pre-validation transport record is mandatory,
    # including at fit publication. A client-complete receipt cannot stand in
    # for missing/stale/failed transport observations.
    stream = stream_facts if type(stream_facts) is dict else {}
    stream_expected = {'schema': 'megartx-prefill-client-stream-v1',
        'plan_sha256': plan['plan_sha256'], 'response_id_sha256': digest(response_id),
        'emitted_outputs': 256, 'stream_done': True, 'finish_reason': 'length',
        'numerical_qualified': False, 'performance_qualified': False}
    equal('stream.record.type', type(stream_facts).__name__, 'dict')
    equal('stream.record.keys', sorted(stream),
          sorted([*stream_expected, 'sse_events', 'output_ids_sha256', 'usage']))
    for key, expected in stream_expected.items():
        equal('stream.'+key, stream.get(key), expected)
    equal('stream.output_ids_sha256.client', stream.get('output_ids_sha256'), client.get('output_ids_sha256'))
    equal('stream.output_ids_sha256.observer', stream.get('output_ids_sha256'), observer.get('output_ids_sha256'))
    equal('stream.sse_events.client', stream.get('sse_events'), client.get('sse_events'))
    stream_events = stream.get('sse_events')
    equal('stream.sse_events.bounded', type(stream_events) is int and 3 <= stream_events <= 300, True)
    stream_usage = stream.get('usage')
    equal('stream.usage.keys', sorted(stream_usage) if type(stream_usage) is dict else None,
          ['completion_tokens', 'prompt_tokens', 'total_tokens'])
    for key, expected in {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304}.items():
        actual = stream_usage.get(key) if type(stream_usage) is dict else None
        equal('stream.usage.'+key, actual, expected)
        equal('stream.usage.'+key+'.client', actual, usage.get(key) if type(usage) is dict else None)
    failed = [check['field'] for check in checks if not check['ok']]
    return {'schema': 'megartx-prefill-native-ledger-comparison-v1',
            'plan_sha256': plan['plan_sha256'], 'status': 'match' if not failed else 'mismatch',
            'failed_fields': failed, 'checks': checks,
            'numerical_qualified': False, 'performance_qualified': False}

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
    stream_path = directory/'client-stream.json'
    if (stream_path.is_symlink() or not stream_path.is_file() or stream_path.stat().st_size > 65536):
        raise ValueError('Native fit requires a bounded persisted client-stream observation')
    stream_facts = read('client-stream.json')
    comparison = native_ledger_comparison(plan, client, observer, stream_facts)
    if comparison['failed_fields']:
        raise ValueError('Native fit ledger mismatch: ' + json.dumps(comparison, sort_keys=True))
    scratch = observer.get('observer_gpu_scratch', {})
    if (ownership.get('cleanup_complete') is not True or ownership.get('failure') is not None
            or ownership.get('owned_identities_remaining') or ownership.get('owned_gpu_pids_remaining')
            or ownership.get('cleanup_errors') or observer.get('plan_sha256') != plan['plan_sha256']
            or client.get('plan_sha256') != plan['plan_sha256'] or loaded.get('plan_sha256') != plan['plan_sha256']
            or observer.get('output_ids_sha256') != client.get('output_ids_sha256')
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

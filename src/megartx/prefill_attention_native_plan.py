"""Separate default-off native observation plan; CPU freeze is never clearance.

The frozen six-file oracle contract remains unchanged. This plan binds current
runtime source, exact historical storage obligations, and the additional actual
operator-input domain. Unknown installed forwarding-shim bytes block admission.
"""
import copy
import importlib
import os
from pathlib import Path

from . import prefill_diagnostic_plan as legacy
from . import prefill_storage_plan as storage
from . import prefill_attention_plan as cpu

SCHEMA = 'megartx-prefill-attention-native-plan-v1'
PURPOSE = 'sampled-attention-error-observation'
SELECTOR = 'MEGARTX_PREFILL_ATTENTION_CAPTURE'
SHIM_SOURCE = 'vllm.utils.flashinfer'
SHIM_UPSTREAM_SHA256 = '7144bbb7a905ad3d40994a2d3e69d69144e4e71020e7c92f487dea65b390790d'
INSTALLED_CONTRACT = {
    SHIM_SOURCE: SHIM_UPSTREAM_SHA256,
    'flashinfer.api_logging': '0302b4ff890c9d0de9c91f340794d8805b68b467690fe3fbec3dfb29f05720ad',
    'flashinfer.trace.template': '8188ab3ff4be944bc15817f10da342ea9e0c7e25a4a29f8868140e5e685c6825',
    'flashinfer.utils': '6cb8ebcc25eb65521808bf40a8aaf0c5a868827283955867ae382ad7ec773815',
}
CATALOG_SOURCE = 'docs/prefill/native-attention-source-reconciliation.json'
LINEAGE_SOURCE = 'src/megartx/prefill_attention_lineage.py'
NEW_SOURCES = (
    'src/megartx/prefill_attention_native_plan.py',
    'src/megartx/prefill_attention_native.py',
    'src/megartx/prefill_attention_hooks.py',
    'src/megartx/prefill_attention_evidence.py',
    'src/megartx/prefill_attention_analysis.py',
    'src/megartx/prefill_attention_publication.py',
    'scripts/prefill_attention_client.py',
    'scripts/prefill_attention_analyze.py',
    'docs/prefill/native-attention-capture.md',
    *cpu.SOURCE_FILES,
)
SOURCES = tuple(dict.fromkeys((*storage.SOURCES, *NEW_SOURCES, LINEAGE_SOURCE, CATALOG_SOURCE)))
BOUNDS = legacy.BOUNDS
RUNNER_POLICY = legacy.RUNNER_POLICY
digest, file_sha, remaining = legacy.digest, legacy.file_sha, legacy.remaining
checkpoint_identity = legacy.checkpoint_identity
native_ledger_comparison = legacy.native_ledger_comparison
native_request_identity = legacy.native_request_identity
CONTROL_SPEC = {
    'specification': cpu.specification(),
    'inherited_storage_obligations': {k: copy.deepcopy(v) for k,v in storage.CONTROL_SPEC.items()
        if k not in {'sample_positions','sample_combined_kv_rows','sample_kv_bytes','raw_head_rows','raw_head_bytes'}},
    'retained_combined_storage_rows': 0,
    'raw_retention': 'eight_attention_files_only',
    'head_observation_rows': 2,
    'head_payload_retention_rows': 0,
    'completion_policy': 'source_bound_global_device_synchronize_before_output_copy',
    'cpu_resource_policy': 'hard_RLIMIT_AS_512MiB_and_CPU_300s_with_owned_wall_deadline',
    'native_arithmetic_acceptance': None,
}


def _common(value):
    common = {key: copy.deepcopy(value[key]) for key in storage.LEGACY_FIELDS}
    common['schema'] = 'megartx-prefill-native-plan-v3'
    common['source_hashes'] = {p: value['source_hashes'][p] for p in legacy.SOURCES}
    common['plan_sha256'] = digest({k:v for k,v in common.items() if k != 'plan_sha256'})
    return common


def validate_plan(value, root=None):
    fields = storage.LEGACY_FIELDS | {'purpose', 'control_spec', 'installed_sources'}
    if (type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA
            or value['purpose'] != PURPOSE or not storage._exact(value['control_spec'], CONTROL_SPEC)
            or digest({k:v for k,v in value.items() if k != 'plan_sha256'}) != value['plan_sha256']):
        raise ValueError('Exact native sampled-attention purpose and scope required')
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    hashes = value['source_hashes']
    if type(hashes) is not dict or set(hashes) != set(SOURCES):
        raise ValueError('Complete current attention runtime source vector required')
    for relative, expected in hashes.items():
        storage._source_file(root, relative, expected)
    legacy.validate_plan(_common(value), root)
    installed = value['installed_sources']
    if (type(installed) is not dict or set(installed) != set(INSTALLED_CONTRACT)
            or any(installed[k] not in (None,v) for k,v in INSTALLED_CONTRACT.items())):
        raise ValueError('Unknown forwarding/helper source contract')
    from .prefill_attention_lineage import validate_source_catalog
    validate_source_catalog(root, hashes)
    return value


def load_plan(path, root=None):
    return validate_plan(storage._read_json(path), root)


def freeze_plan(tokens, source_head, root, checkpoint, adapter_site=None, *, installed_sources=None):
    value = legacy.freeze_plan(tokens, source_head, root, checkpoint, adapter_site)
    value.update(schema=SCHEMA, purpose=PURPOSE, control_spec=copy.deepcopy(CONTROL_SPEC),
                 source_hashes={p: file_sha(Path(root)/p) for p in SOURCES},
                 installed_sources={k: None for k in INSTALLED_CONTRACT} if installed_sources is None else copy.deepcopy(installed_sources))
    value['plan_sha256'] = digest({k:v for k,v in value.items() if k != 'plan_sha256'})
    return validate_plan(value, root)


def cpu_capture_plan(plan, root=None):
    return cpu.make_plan(plan['plan_sha256'], plan['prompt_sha256'], plan['source_head'],
                         root or Path(__file__).resolve().parents[2])


def require_clearance(path, plan):
    if plan['installed_sources'] != INSTALLED_CONTRACT:
        raise ValueError('Installed attention forwarding/helper closure is unknown; native capture blocked')
    value = storage._read_json(path, 65536)
    expected = {'schema': 'megartx-prefill-attention-native-clearance-v1',
        'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
        'cpu_review_passed': True, 'parent_gpu_slot_clearance': True,
        'installed_source_vector_sha256': digest(plan['installed_sources']),
        'max_contexts': 1, 'max_requests': 1, 'max_retries': 0}
    if not storage._exact(value, expected):
        raise ValueError('Exact new-purpose CPU review and separate owned GPU-slot clearance required')
    return value


def verify_adapter_sources(plan):
    origins = legacy.verify_adapter_sources(plan)
    for relative in SOURCES:
        if not relative.startswith('src/megartx/'):
            continue
        name = 'megartx.' + Path(relative).stem
        module = importlib.import_module(name)
        expected = Path(plan['adapter_site'])/'megartx'/Path(relative).name
        actual, origin = getattr(module, '__file__', None), getattr(getattr(module, '__spec__', None), 'origin', None)
        if (not actual or not origin or expected.is_symlink()
                or Path(actual).resolve() != expected or Path(origin).resolve() != expected
                or file_sha(expected) != plan['source_hashes'][relative]):
            raise ValueError('Executing attention adapter origin/source changed: '+name)
        origins[name] = {'origin': str(expected), 'sha256': plan['source_hashes'][relative]}
    root = Path(os.environ.get('MEGARTX_PREFILL_NATIVE_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
    checker = storage._source_file(root, storage.CHECKER_SOURCE, plan['source_hashes'][storage.CHECKER_SOURCE])
    origins['prefill_native_control'] = {'origin': str(checker.resolve()), 'sha256': file_sha(checker)}
    return origins


def validate_binding(plan, directory, owned_identities=None, require_live=True):
    from .prefill_runner_binding import validate_binding as base_binding
    binding = base_binding(plan, directory, owned_identities, require_live)
    value = storage._read_json(Path(directory)/'storage-binding.json', 65536)
    expected = {'schema': 'megartx-prefill-storage-binding-v1',
        'plan_sha256': plan['plan_sha256'], 'source_head': plan['source_head'],
        'owner_pid': binding['owner_pid'], 'owner_start_ticks': binding['owner_start_ticks'],
        'purpose': PURPOSE, 'writer_hook_bound': True,
        'checker_sha256': plan['source_hashes'][storage.CHECKER_SOURCE]}
    if not storage._exact(value, expected):
        raise ValueError('Current attention-purpose storage owner/checker binding required')
    attention = storage._read_json(Path(directory)/'attention-binding.json', 65536)
    expected_attention = {'schema': 'megartx-prefill-attention-binding-v1',
        'plan_sha256': plan['plan_sha256'], 'capture_plan_sha256': cpu_capture_plan(plan)['plan_sha256'],
        'owner_pid': binding['owner_pid'], 'owner_start_ticks': binding['owner_start_ticks'],
        'hooks_bound': True, 'installed_sources': plan['installed_sources'],
        'completion_policy': CONTROL_SPEC['completion_policy']}
    if not storage._exact(attention, expected_attention):
        raise ValueError('Exact selected enclosing/wrapper/cache hook binding required')
    return binding

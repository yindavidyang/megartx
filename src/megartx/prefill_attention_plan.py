"""CPU-only sparse-attention capture specification; never launch clearance.

The native integration owns original D2H copies, the single evidence budget,
all-layer storage checks, actual callback/stream identity and cleanup. Helpers
here select bytes already on CPU and never charge an existing transfer again.
"""
import hashlib
import json
from pathlib import Path
import re
import struct

BASE_HEAD = '6493affbcf0827f8702e736636bb4f2549d07305'
BASE_TREE = '257ef9b3ab5a5a88d606eb22c461540de4bf4dc9'
SCHEMA = 'megartx-prefill-attention-cpu-plan-v1'
PURPOSE = 'sparse-attention-error-observation'
POSITIONS = (0, 15, 16, 255, 256, 1023, 1024, 1792, 2047, 2048)
FRAMES = ((0, 256), (256, 512), (768, 1024), (1024, 1280), (1792, 2048), (2048, 2049))
LAYERS = {0: {'dim': 256, 'kv_heads': 8, 'selected_kv_heads': (0,), 'q_heads': (0, 1)},
          5: {'dim': 512, 'kv_heads': 2, 'selected_kv_heads': (0, 1), 'q_heads': (0, 7, 8, 15)}}
SOURCE_FILES = ('src/megartx/prefill_attention_plan.py',
                'src/megartx/prefill_attention_validation.py',
                'numerical_reference/prefill_attention_reference.py',
                'docs/prefill/native-attention-cpu-control.md')
BOUNDS = {'max_contexts': 1, 'max_requests': 1, 'max_retries': 0,
          'max_evidence_bytes': 8 << 20, 'max_metadata_bytes': 2 << 20,
          'max_gpu_scratch_bytes': 8 << 20, 'max_d2h_bytes': 4 << 30,
          'gpu_free_floor_bytes': 2 << 30, 'host_free_floor_bytes': 8 << 30,
          'max_reference_rss_bytes': 512 << 20, 'max_reference_seconds': 300,
          'max_shared_wall_seconds': 1800, 'max_compiler_rss_bytes': 2 << 30,
          'max_shared_build_seconds': 300}
STORAGE_OBLIGATIONS = {'capture_frames': 9, 'capture_end': 2049,
    'checked_layer_frames': 270, 'pre_rows': 212355, 'processed_rows': 61470,
    'post_rows': 273825, 'storage_d2h_bytes': 4024934400,
    'checked_heads': 263, 'emitted_tokens': 256, 'decode_inputs': 255,
    'committed_length': 2303, 'capacity_tokens': 2304}


def sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Exact lowercase SHA256 required')
    return value


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Exact bounded integer required')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def coordinates(layer):
    integer(layer, 0, 5)
    if layer not in LAYERS:
        raise ValueError('Only selected layers 0 and 5 are supported')
    d = LAYERS[layer]['dim']
    return (0, 1, d//4-1, d//4, d//2-1, d//2, d-2, d-1)


def raw_layouts():
    layouts = {}
    for layer, spec in LAYERS.items():
        d, h, q = spec['dim'], len(spec['selected_kv_heads']), len(spec['q_heads'])
        for role, shape in (('k', (2049, h, d)), ('v', (2049, h, 8)),
                            ('q', (10, q, d)), ('o', (10, q, 8))):
            size = 2
            for n in shape:
                size *= n
            layouts[f'attention-layer-{layer:02d}-{role}.bf16'] = {
                'layer': layer, 'role': role, 'shape': list(shape), 'bytes': size}
    return layouts


def budget():
    raw = sum(v['bytes'] for v in raw_layouts().values())
    local = sum(e-max(0,s-1023) for s,e in FRAMES)
    glob = sum(e for s,e in FRAMES)
    return {'raw_bytes': raw, 'metadata_limit_bytes': 2 << 20,
            'retained_bound_bytes': raw+(2 << 20), 'raw_files': 8,
            'query_head_cases': 60, 'output_coordinates': 480,
            'cache_entry_local_rows': local, 'cache_entry_global_rows': glob,
            'additional_cache_entry_d2h_bytes': local*8192+glob*4096,
            'additional_q_and_output_head_d2h_bytes': 102400,
            'additional_metadata_d2h_allowance_bytes': 65536,
            'reused_writer_d2h_additional_charge_bytes': 0}


def specification():
    return {'prompt_tokens': 2048, 'chunk_tokens': 256, 'output_tokens': 256,
            'capacity_tokens': 2304, 'compute_dtype': 'bfloat16', 'kv_dtype': 'bfloat16',
            'query_positions': list(POSITIONS), 'frames': [list(x) for x in FRAMES],
            'layers': {str(k): {**v, 'selected_kv_heads': list(v['selected_kv_heads']),
                               'q_heads': list(v['q_heads']), 'coordinates': list(coordinates(k))}
                       for k,v in LAYERS.items()},
            'raw_layouts': raw_layouts(), 'budget': budget(), 'bounds': BOUNDS.copy(),
            'storage_obligations': STORAGE_OBLIGATIONS.copy(),
            'raw_policy': 'replace_previous_raw_samples_in_new_purpose_only',
            'native_arithmetic_acceptance': None, 'numerical_qualified': False,
            'reference_precision_digits': [96,128], 'reference_crosscheck_vscale_power': -180,
            'local_cross_group_coverage': False, 'global_cross_group_coverage': True}


def make_plan(native_plan_sha256, prompt_sha256, source_head, root):
    for value in (native_plan_sha256, prompt_sha256):
        sha(value)
    if type(source_head) is not str or not re.fullmatch('[0-9a-f]{40}', source_head):
        raise ValueError('Exact CPU source head required')
    root = Path(root)
    sources = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular CPU source file required')
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = {'schema': SCHEMA, 'purpose': PURPOSE, 'base_head': BASE_HEAD, 'base_tree': BASE_TREE,
             'source_head': source_head, 'source_hashes': sources,
             'native_plan_sha256': native_plan_sha256, 'prompt_sha256': prompt_sha256,
             'specification': specification(), 'cpu_only': True, 'native_integration_ready': False}
    value['plan_sha256'] = digest(value)
    return value


def validate_plan(value, root=None):
    fields = {'schema','purpose','base_head','base_tree','source_head','source_hashes',
              'native_plan_sha256','prompt_sha256','specification','cpu_only',
              'native_integration_ready','plan_sha256'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('Exact CPU-only attention plan schema required')
    if (value['schema'] != SCHEMA or value['purpose'] != PURPOSE or value['base_head'] != BASE_HEAD
            or value['base_tree'] != BASE_TREE or value['cpu_only'] is not True
            or value['native_integration_ready'] is not False
            or digest(value['specification']) != digest(specification())):
        raise ValueError('CPU attention scope changed; this is not launch clearance')
    if type(value['source_head']) is not str or not re.fullmatch('[0-9a-f]{40}', value['source_head']):
        raise ValueError('Exact CPU source head required')
    for field in ('native_plan_sha256','prompt_sha256','plan_sha256'):
        sha(value[field])
    if digest({k:v for k,v in value.items() if k != 'plan_sha256'}) != value['plan_sha256']:
        raise ValueError('CPU attention plan digest mismatch')
    if type(value['source_hashes']) is not dict or set(value['source_hashes']) != set(SOURCE_FILES):
        raise ValueError('Complete CPU source vector required')
    for relative, expected in value['source_hashes'].items():
        sha(expected)
        if root is not None:
            path = Path(root) / relative
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Executing CPU source file changed: '+relative)
    return value


def finite_words(raw, count):
    if type(raw) is not bytes or len(raw) != count*2:
        raise ValueError('Exact original little-endian BF16 byte extent required')
    if any(word & 0x7f80 == 0x7f80 for (word,) in struct.iter_unpack('<H', raw)):
        raise ValueError('Nonfinite BF16 bytes')


def select_writer_row(layer, position, key, value):
    """Pure bounded CPU slicing; caller has already charged both D2H rows."""
    integer(layer, 0, 29); integer(position, 0, 2048)
    if layer not in LAYERS:
        return ()
    spec = LAYERS[layer]; d = spec['dim']; heads = spec['selected_kv_heads']
    finite_words(key, spec['kv_heads']*d); finite_words(value, spec['kv_heads']*d)
    k = b''.join(key[2*h*d:2*(h+1)*d] for h in heads)
    v = b''.join(value[2*(h*d+c):2*(h*d+c+1)] for h in heads for c in coordinates(layer))
    return ((f'attention-layer-{layer:02d}-k.bf16', position*len(k), k),
            (f'attention-layer-{layer:02d}-v.bf16', position*len(v), v))


def select_head_row(layer, position, head, role, raw):
    """Full actual Q/O head arrives; O coordinate selection is CPU-only."""
    coordinates(layer); integer(position, 0, 2048); integer(head, 0, 15)
    spec = LAYERS[layer]
    if position not in POSITIONS or head not in spec['q_heads'] or role not in ('q','o'):
        raise ValueError('Unselected actual query/head/role')
    d = spec['dim']; finite_words(raw,d)
    data = raw if role == 'q' else b''.join(raw[2*c:2*(c+1)] for c in coordinates(layer))
    row = POSITIONS.index(position)*len(spec['q_heads'])+spec['q_heads'].index(head)
    return f'attention-layer-{layer:02d}-{role}.bf16', row*len(data), data

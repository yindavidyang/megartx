"""Reject inactive or invalidated integration artifacts before comparison/timing."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def activation(directory):
    directory = Path(directory)
    if (directory / 'QUALIFICATION-INVALIDATED.json').exists():
        raise RuntimeError('Integration evidence was explicitly invalidated')
    proof = json.loads((directory / 'activation-proof.json').read_text())
    if not proof.get('forced_all_six_executed') or not proof.get('natural_model_forward_verified'):
        raise RuntimeError('Missing forced correction or guarded model dispatch proof')
    layers, captures, fixtures = (proof[k] for k in ('registered_layers', 'original_tensor_captures', 'forced_fixtures'))
    if len(layers) != 30 or len(captures) != 54 or len(fixtures) != 6:
        raise RuntimeError('Incomplete frozen integration scope')
    if len({r['layer_name'] for r in layers}) != 30 or len({r['tensor'] for r in captures}) != 54:
        raise RuntimeError('Duplicate integration identities/tensor captures')
    fixture_keys = {(f['layer_name'], f['expert']) for f in fixtures}
    if len(fixture_keys) != 6 or any(name not in {r['layer_name'] for r in layers} for name, _ in fixture_keys):
        raise RuntimeError('Duplicate or unregistered forced expert fixture')
    if not all(c['original_bytes_equal'] for c in captures):
        raise RuntimeError('Original tensor capture mismatch')
    if not all(f['forced_correction_rows'] >= 2 and f['observed_bf16_value_equal'] and f['max_absolute_difference'] == 0 and f['nonzero_output_elements'] > 0 for f in fixtures):
        raise RuntimeError('Forced native/reference fixture was bypassed or differs')
    trace = directory / 'activation-forced.json.gz'
    if hashlib.sha256(trace.read_bytes()).hexdigest() != proof['forced_trace_sha256']:
        raise RuntimeError('Forced integration trace hash differs')
    with gzip.open(trace, 'rt') as source:
        events = json.load(source)['traceEvents']
    # Profiler can duplicate CPU spans into Trace view annotations; count only
    # the actual user_annotation category rather than assuming every X is unique.
    spans = Counter(e['name'] for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation')
    for label in ('megartx::corrected_expert_native', 'megartx::corrected_expert_reference'):
        if spans[label] < 6:
            raise RuntimeError('Missing adapter-specific forced correction trace spans')
    kernels = Counter(e['name'] for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel')
    dense_sm120 = sum(count for name, count in kernels.items() if 'MainloopSm120TmaWarpSpecializedBlockScaled' in name)
    gelu = kernels['_gelu_product']
    if dense_sm120 < 18 or gelu < 6 or any('marlin' in name.lower() for name in kernels):
        raise RuntimeError('Missing native SM120 dense correction/adapter GELU kernels or unexpected Marlin dispatch')
    return {'forced_all_six_executed': True, 'native_dense_sm120_launches': dense_sm120, 'adapter_gelu_launches': gelu, 'guarded_model_dispatch': True, 'forced_adapter_spans': {k: spans[k] for k in ('megartx::corrected_expert_native', 'megartx::corrected_expert_reference')}, 'forced_trace_sha256': proof['forced_trace_sha256']}


def _require_natural_scope(record):
    # Older immutable natural records omit these fields. Explicit scope on a
    # newer record must agree; controlled events never become natural evidence
    # merely because their six counters are positive.
    if (('route_origin' in record and record['route_origin'] != 'natural')
            or ('scope' in record and record['scope'] != 'natural')
            or record.get('routing_intervention', False) is not False
            or ('routing_unchanged' in record and record['routing_unchanged'] is not True)):
        raise RuntimeError('Controlled or unknown routing scope cannot qualify natural correction coverage')


def natural_coverage(directory, mode):
    directory = Path(directory)
    if (directory / 'QUALIFICATION-INVALIDATED.json').exists():
        raise RuntimeError('Integration evidence was explicitly invalidated')
    if (directory / 'CONTROLLED-ROUTING.json').exists():
        raise RuntimeError('Controlled routing artifacts cannot qualify natural correction coverage')
    if mode not in ('native', 'reference'):
        raise RuntimeError('Natural correction coverage requires an active correction mode')
    requests = json.loads((directory / 'quality-requests.json').read_text())
    for request in requests:
        _require_natural_scope(request)
    prefixes = {r['id']: r['prompt_sha256'] for r in requests}
    manifests = [json.loads(line) for line in (directory / 'adapter-manifest.jsonl').read_text().splitlines()]
    for manifest in manifests:
        _require_natural_scope(manifest)
    if not manifests or any(m['mode'] != mode or not m['correction_active'] for m in manifests):
        raise RuntimeError('Natural manifest has inactive or unmatched correction mode')
    expected = {(m['loader_ordinal'], e['expert']): m['layer'] for m in manifests for e in m['affected_experts']}
    if len(expected) != 6 or not prefixes:
        raise RuntimeError('Natural coverage requires the six frozen experts and a client corpus')
    totals = {key: {'routed_rows': 0, 'nonzero_route_weights': 0, 'cases': set()} for key in expected}
    for line in (directory / 'route-hits.jsonl').read_text().splitlines():
        hit = json.loads(line)
        _require_natural_scope(hit)
        key = hit['loader_ordinal'], hit['expert']
        if key not in totals or hit['layer_name'] != expected[key] or hit['mode'] != mode or prefixes.get(hit['case_id']) != hit['prompt_sha256']:
            raise RuntimeError('Natural route audit differs from the actual loaded experts/client prefix')
        if not 0 <= hit['nonzero_route_weights'] <= hit['routed_rows']:
            raise RuntimeError('Invalid natural route counters')
        totals[key]['routed_rows'] += hit['routed_rows']
        totals[key]['nonzero_route_weights'] += hit['nonzero_route_weights']
        totals[key]['cases'].add(hit['case_id'])
    rows = [{'loader_ordinal': key[0], 'expert': key[1], 'layer_name': expected[key], 'routed_rows': values['routed_rows'], 'nonzero_route_weights': values['nonzero_route_weights'], 'cases': sorted(values['cases'])} for key, values in sorted(totals.items())]
    report = {'mode': mode, 'client_requests': len(requests), 'experts': rows, 'all_six_naturally_exercised': all(row['nonzero_route_weights'] > 0 for row in rows), 'qualification': 'Request-bound natural routing; forced startup fixtures are excluded'}
    (directory / 'natural-coverage.json').write_text(json.dumps(report, indent=2))
    if not report['all_six_naturally_exercised']:
        missing = [(row['loader_ordinal'], row['expert']) for row in rows if row['nonzero_route_weights'] == 0]
        raise RuntimeError('Natural client corpus did not exercise corrected experts: ' + str(missing))
    return report

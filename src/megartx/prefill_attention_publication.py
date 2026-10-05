"""Current storage/frontier/client gates followed by independent attention error.

No numerical tolerance or native arithmetic pass is inferred. Old raw retention
is replaced only here; original scalar/token/head observations remain required.
The inherited gate implementations below are source-equivalence tested against
the immutable prior functions, with only this purpose's explicit policy deltas.
"""
from pathlib import Path

from .prefill_storage_plan import (CONTROL_COUNTS, QUALIFICATIONS, _sha, _exact,
    _read_json, _records, file_sha, digest, native_ledger_comparison, validate_telemetry)
from .prefill_attention_native_plan import (PURPOSE, validate_binding, cpu_capture_plan,
    native_request_identity, verify_adapter_sources)
from .prefill_attention_validation import CAPTURE_SCHEMA, read_capture

def validate_control(plan, value, output_ids_sha256=None):
    fixed = {'schema': 'megartx-prefill-attention-storage-control-v1',
             'plan_sha256': plan['plan_sha256'], 'status': 'storage_frontier_observed',
             'storage_exact': True, 'frontier_verified': True, 'storage_callbacks_restored': True,
             **CONTROL_COUNTS, **QUALIFICATIONS, 'sample_combined_kv_rows': 0,
             'raw_head_payload_rows': 0, 'attention_raw_files': 8, 'attention_raw_bytes': 5395952,
             'attention_callbacks_restored': True}
    if type(value) is not dict or any(not _exact(value.get(key), expected) for key, expected in fixed.items()):
        raise ValueError('Complete exact storage/frontier control proof required')
    for key in ('raw_samples_sha256', 'frame_records_sha256', 'head_records_sha256',
                'sample_records_sha256', 'sample_hashes_sha256'):
        if not _sha(value.get(key)):
            raise ValueError('Storage all-row/head/sample root is absent: ' + key)
    observed_heads = value.get('observed_head_hashes')
    if (type(observed_heads) is not dict or set(observed_heads) != {'2047', '2048'}
            or any(not _sha(v) for v in observed_heads.values())):
        raise ValueError('Both actual native head hashes are required without retaining old payloads')
    frontier = value.get('frontier')
    frontier_fixed = {'checked_heads': 263, 'emitted_tokens': 256, 'decode_inputs': 255,
                      'committed_length': 2303, 'uncached_output_position': 2303,
                      'token_ids_encoding': 'canonical-compact-json-integer-array-utf8'}
    if (type(frontier) is not dict or set(frontier) != set(frontier_fixed) | {'token_ids_sha256'}
            or any(not _exact(frontier.get(key), expected) for key, expected in frontier_fixed.items())
            or not _sha(frontier.get('token_ids_sha256'))
            or (output_ids_sha256 is not None and frontier['token_ids_sha256'] != output_ids_sha256)):
        raise ValueError('Independent frontier/token transport proof differs')
    roots = value.get('storage_roots')
    if (type(roots) is not dict or set(roots) != {'pre', 'processed', 'post'}
            or any(not _sha(v) for v in roots.values())
            or not _sha(value.get('actual_sampling_params_sha256'))
            or type(value.get('processed_digest_host_bytes')) is not int
            or value['processed_digest_host_bytes'] <= 0
            or value.get('natural_positive_correction_coverage', 'missing') is not None):
        raise ValueError('Bounded all-row roots and actual sampler identity required')
    transfer = value.get('transfer')
    domains = transfer.get('domains') if type(transfer) is dict else None
    expected_domains = {'pre': 1550868480, 'processed': 461598720, 'post': 2012467200,
                        'raw_heads': 1048576, 'head_index': 2104,
                        'selected_hidden': 2962432, 'head_finite_scalar': 261,
                        'writer_slot_metadata': 552720, 'inherited_metadata_upper_bound': 41811968,
                        'attention_cache_entry': 73388032, 'attention_q_output': 102400,
                        'attention_metadata': 65536}
    if (type(transfer) is not dict
            or set(transfer) != {'limit_bytes', 'transferred_bytes', 'copy_calls', 'domains'}
            or not _exact(transfer.get('limit_bytes'), 4 << 30)
            or type(transfer.get('transferred_bytes')) is not int
            or not 4024934400 <= transfer['transferred_bytes'] <= 4 << 30
            or type(transfer.get('copy_calls')) is not int or transfer['copy_calls'] <= 0
            or type(domains) is not dict
            or set(domains) != set(expected_domains) | {'manager_table_metadata'}
            or any(not _exact(domains.get(k), v) for k, v in expected_domains.items())
            or any(type(v) is not int or v <= 0 for v in domains.values())
            or sum(domains.values()) != transfer['transferred_bytes']):
        raise ValueError('Complete exact D2H domains and shared transfer cap required')
    manager = value.get('manager')
    expected_manager_keys = {'manager_block_tokens', 'kernel_block_tokens', 'manager_to_kernel_ratios',
                             'append_calls', 'manager_table_sha256', 'expected_address_source'}
    if (type(manager) is not dict or set(manager) != expected_manager_keys
            or manager['expected_address_source'] != 'actual_append_block_ids_before_native_subdivision'
            or not _sha(manager['manager_table_sha256'])
            or type(manager['append_calls']) is not int or manager['append_calls'] <= 0):
        raise ValueError('Actual pre-subdivision manager provenance required')
    sizes, kernels, ratios = (manager[k] for k in (
        'manager_block_tokens', 'kernel_block_tokens', 'manager_to_kernel_ratios'))
    if (any(type(v) is not list for v in (sizes, kernels, ratios))
            or not 1 <= len(sizes) <= 30 or len(sizes) != len(kernels) or len(sizes) != len(ratios)
            or any(type(b) is not int or type(k) is not int or type(r) is not int
                   or k != 16 or r != 1 or b != 16 for b, k, r in zip(sizes, kernels, ratios))):
        raise ValueError('Actual manager/kernel subdivision geometry differs')
    # Each group is reconciled before and after all263frames; both persistent
    # and gathered tables are copied. Counts cannot undercut required pages or
    # exceed capacity144. Exact source-derived lower/upper bounds per group.
    if not 566016 * len(sizes) <= domains['manager_table_metadata'] <= 585984 * len(sizes):
        raise ValueError('Actual manager table transfer allowance is outside source-derived bounds')
    return value


def validate_transcripts(plan, directory, control):
    frames = _records(directory, 'control-frames.jsonl', 263)
    heads = _records(directory, 'heads.jsonl', 263)
    if (control['frame_records_sha256'] != file_sha(Path(directory) / 'control-frames.jsonl')
            or control['head_records_sha256'] != file_sha(Path(directory) / 'heads.jsonl')):
        raise ValueError('Compact frame/head transcript root changed')
    for sequence, (frame, head) in enumerate(zip(frames, heads)):
        start = sequence * 256 if sequence < 8 else 2048 + sequence - 8
        end = start + (256 if sequence < 8 else 1)
        expected_frame = {'sequence': sequence, 'start': start, 'end': end,
                          'queries_complete': True}
        if any(not _exact(frame.get(k), v) for k, v in expected_frame.items()):
            raise ValueError('Compact storage frame identity/order differs')
        if not _sha(frame.get('input_ids_sha256')):
            raise ValueError('Compact storage frame input identity absent')
        if sequence < 8 and frame['input_ids_sha256'] != digest(plan['tokens'][start:end]):
            raise ValueError('Prompt frame input digest differs from frozen private prompt')
        expected_head = {'sequence': sequence, 'logit_position': end - 1,
                         'predicts_position': end, 'expected_sampler_discard': sequence < 7}
        if (any(not _exact(head.get(k), v) for k, v in expected_head.items())
                or not _sha(head.get('hidden_bits_sha256'))
                or (head.get('logit_bits_sha256') is not None and not _sha(head['logit_bits_sha256']))):
            raise ValueError('Compact native head identity/order differs')
        expected_frame.update(plan_sha256=plan['plan_sha256'], storage_checked=sequence < 9)
        if any(not _exact(frame.get(k), v) for k, v in expected_frame.items()):
            raise ValueError('Storage checked/metadata-only frame coverage differs')
        if (head.get('plan_sha256') != plan['plan_sha256']
                or head.get('before_sampler_transforms') is not True
                or not _exact(head.get('selected_row_index'), 255 if sequence < 8 else 0)
                or head.get('native_logits_dtype') != 'torch.bfloat16'
                or not _exact(head.get('suppressed_token_count'), 0)):
            raise ValueError('Actual pre-sampler native head selection/dtype differs')
        if end - 1 in (2047, 2048):
            if head.get('logit_bits_sha256') != control['observed_head_hashes'][str(end-1)]:
                raise ValueError('Actual full-head hash differs from independent native head event')
        elif head.get('logit_bits_sha256') is not None:
            raise ValueError('Only the two admitted raw native head rows may be retained')
    expected_counts = {'pre': 0, 'processed': 0, 'post': 0}
    for frame in frames:
        if frame['storage_checked']:
            for layer in range(30):
                low = 0 if layer % 6 == 5 else max(0, frame['start'] - 1023)
                expected_counts['pre'] += frame['start'] - low
                expected_counts['processed'] += frame['end'] - frame['start']
                expected_counts['post'] += frame['end'] - low
            roots = frame.get('storage_roots')
            if (not _exact(frame.get('storage_counts'), expected_counts)
                    or type(roots) is not dict or set(roots) != set(expected_counts)
                    or any(not _sha(v) for v in roots.values())):
                raise ValueError('Complete all-layer per-frame counts/roots required')
        elif frame.get('storage_counts') is not None or frame.get('storage_roots') is not None:
            raise ValueError('Later continuation must remain metadata-only')
    if frames[8]['storage_roots'] != control['storage_roots']:
        raise ValueError('Storage final checked roots differ from completed capture')
    emissions = _records(directory, 'samples.jsonl', 256)
    if control['sample_records_sha256'] != file_sha(Path(directory) / 'samples.jsonl'):
        raise ValueError('Scalar sample transcript root changed')
    sample_hashes = []
    for index, event in enumerate(emissions):
        expected = {'output_index': index, 'head_sequence': index + 7,
                    'head_phase': 'final_prompt' if index == 0 else 'decode',
                    'cached_length': 2048 + index, 'pending_anchor_position': 2048 + index,
                    'anchor_kv_written': False}
        if (set(event) != set(expected) | {'sample_sha256'}
                or any(not _exact(event.get(k), v) for k, v in expected.items())
                or not _sha(event.get('sample_sha256'))):
            raise ValueError('Exact ordered native output/head/uncached-anchor sample bindings required')
        sample_hashes.append(event['sample_sha256'])
        if index < 255 and event['sample_sha256'] != frames[8 + index]['input_ids_sha256']:
            raise ValueError('Consumed decode input differs from preceding native sample')
    if digest(sample_hashes) != control['sample_hashes_sha256']:
        raise ValueError('Ordered native sample scalar digest root changed')
    return frames, heads


def validate_storage_domain(evidence, plan, ownership):
    """Return a digest of this current fully validated domain, never a saved root."""
    directory = evidence.directory
    if any((directory/name).exists() for name in ('storage-failure.json','attention-failure.json','INVALIDATED.json')):
        raise ValueError('Failed attention context cannot publish')
    evidence.sizes()
    names = ('observer.json','client.json','loaded.json','geometry.json','client-stream.json',
             'control.json','attention-client.json')
    observer,client,loaded,geometry,stream,control,mode = [
        _read_json(directory/name, 1<<20 if name=='loaded.json' else 65536) for name in names]
    binding = validate_binding(plan,directory,require_live=False)
    comparison = native_ledger_comparison(plan,client,observer,stream)
    if comparison['failed_fields']:
        raise ValueError('Attention native/client transport ledger mismatch')
    validate_control(plan,control,client['output_ids_sha256'])
    expected_mode = {'schema':'megartx-prefill-attention-client-v1','purpose':PURPOSE,
        'plan_sha256':plan['plan_sha256'],'status':'complete','control_sha256':digest(control),
        'output_ids_sha256':client['output_ids_sha256'],
        'sample_hashes_sha256':control['sample_hashes_sha256'],
        'storage_capture_end':2049,'metadata_only_decode_inputs':254,
        'numerical_qualified':False,'performance_qualified':False}
    if not _exact(mode,expected_mode):
        raise ValueError('Attention-purpose exact client receipt differs')
    if (type(ownership) is not dict
            or not {'cleanup_complete','failure','owned_identities_remaining',
                    'owned_gpu_pids_remaining','cleanup_errors'} <= set(ownership)
            or ownership.get('cleanup_complete') is not True or ownership.get('failure') is not None
            or any(ownership.get(k) != [] for k in ('owned_identities_remaining','owned_gpu_pids_remaining','cleanup_errors'))
            or loaded.get('plan_sha256') != plan['plan_sha256']
            or loaded.get('mutable_lease_granted') is not False
            or not _exact(loaded.get('identity',{}).get('owner_pid'),binding['owner_pid'])
            or not _exact(loaded.get('identity',{}).get('owner_start_ticks'),binding['owner_start_ticks'])
            or geometry.get('physical_policy') != 'full_context'
            or not _exact(geometry.get('capacity_tokens'),2304)
            or geometry.get('actual_owned_page_ranges_disjoint') is not True):
        raise ValueError('Bound loaded owner and complete owned server cleanup required')
    cleanup = _read_json(directory/'cleanup.json',1<<20)
    if any(cleanup.get(k) != v for k,v in ownership.items()):
        raise ValueError('Persisted cleanup is not the supplied current owned-server report')
    scratch = observer.get('observer_gpu_scratch',{})
    cap=plan['bounds']['max_observer_gpu_scratch_bytes']
    if (scratch.get('domain') != 'incremental_gpu_allocator_bytes'
            or not _exact(scratch.get('cap_bytes'),cap)
            or type(scratch.get('measured_phases')) is not int or scratch['measured_phases']<789
            or any(type(scratch.get(k)) is not int or not 0<=scratch[k]<=cap for k in (
                'managed_tensor_simultaneous_peak_bytes','measured_phase_allocator_increment_peak_bytes',
                'measured_phase_reserved_increment_peak_bytes'))
            or scratch.get('counter_policy') != 'process_global_resets_with_explicit_runwide_peak_preservation'
            or any(type(scratch.get(k)) is not int or scratch[k]<0 for k in (
                'runwide_allocator_allocated_peak_bytes','runwide_allocator_reserved_peak_bytes'))
            or scratch.get('host_heap_excluded') is not True):
        raise ValueError('Complete bounded observer GPU-scratch telemetry required')
    raw_manifest=evidence.seal_raw()
    if control['raw_samples_sha256'] != digest(raw_manifest):
        raise ValueError('Completed attention raw sample root changed')
    frames,heads=validate_transcripts(plan,directory,control)
    telemetry=validate_telemetry(directory)
    request_hash=digest(native_request_identity(plan,observer['engine_request_id']))
    source_root=Path(__file__).resolve().parents[2]
    from .prefill_attention_native_plan import validate_plan
    validate_plan(plan,source_root)
    # These are validated *current* evidence and actual owned cleanup identities.
    bound={'schema':'megartx-prefill-attention-current-storage-binding-v1',
        'plan_sha256':plan['plan_sha256'],'request_identity_sha256':request_hash,
        'files':{name:file_sha(directory/name) for name in (*names,'runner-binding.json','storage-binding.json',
            'attention-binding.json','attention-records.json','attention-stream-state.json',
            'cleanup.json','control-frames.jsonl','heads.jsonl','samples.jsonl')},
        'owned_cleanup_sha256':digest(ownership),'telemetry':telemetry,
        'raw_manifest_sha256':digest(raw_manifest)}
    return {'binding':bound,'storage_binding_sha256':digest(bound),'request_identity_sha256':request_hash,
            'raw_manifest':raw_manifest,'control':control,'observer':observer,'frames':frames,'heads':heads,
            'telemetry':telemetry,'scratch':scratch}


def _publish_attention(evidence, plan, ownership_report, *, deadline, ownership):
    """Only after all 30-layer/263-frontier/client/cleanup gates, run CPU oracle."""
    from .prefill_attention_native_plan import remaining
    from .prefill_attention_analysis import run_analysis
    remaining(deadline)
    validated=validate_storage_domain(evidence,plan,ownership_report)
    specification=cpu_capture_plan(plan)
    records=_read_json(evidence.directory/'attention-records.json',1<<20)
    expected={'schema':'megartx-prefill-attention-native-records-v1',
        'native_plan_sha256':plan['plan_sha256'],'capture_plan_sha256':specification['plan_sha256'],
        'request_identity_sha256':validated['request_identity_sha256'],
        'raw_manifest':validated['raw_manifest'],'callbacks_restored':True}
    if (type(records) is not dict or set(records) != set(expected)|{'records'}
            or any(not _exact(records.get(k),v) for k,v in expected.items())):
        raise ValueError('Complete current runtime-attention record binding required')
    # Bind every attention record to independently validated current frame input.
    for record in records['records']:
        if type(record) is not dict:
            raise ValueError('Native attention record required')
        frame=next((f for f in validated['frames'] if (f['start'],f['end'])==
                    (record.get('start'),record.get('end'))),None)
        if frame is None or record.get('frame_input_ids_sha256')!=frame['input_ids_sha256']:
            raise ValueError('Attention record uses stale prompt/decode-anchor inputs')
    capture={'schema':CAPTURE_SCHEMA,'plan_sha256':specification['plan_sha256'],
        'native_plan_sha256':plan['plan_sha256'],'prompt_sha256':plan['prompt_sha256'],
        'request_identity_sha256':validated['request_identity_sha256'],
        'inherited_storage_binding_sha256':validated['storage_binding_sha256'],
        'raw_manifest':validated['raw_manifest'],'records':records['records']}
    evidence.finalize_capture(capture)
    checked=read_capture(evidence.directory,specification,
        expected_request_sha256=validated['request_identity_sha256'],
        expected_storage_binding_sha256=validated['storage_binding_sha256'])
    result=run_analysis(evidence,specification,
        expected_request_sha256=validated['request_identity_sha256'],
        expected_storage_binding_sha256=validated['storage_binding_sha256'],
        deadline=deadline,ownership=ownership)
    remaining(deadline)
    # The separately loaded oracle owns its input traversal; no collector roots.
    if (result['result']['input_manifest']!=checked['capture']['raw_manifest']
            or result['input_manifest_matches_validated_raw_manifest'] is not True):
        raise ValueError('Independent oracle inputs differ from validated raw manifest')
    current=validate_storage_domain(evidence,plan,ownership_report)
    if current['storage_binding_sha256']!=validated['storage_binding_sha256']:
        raise ValueError('Current storage domain changed during CPU analysis')
    evidence.write('attention-errors.json',result)
    receipt={'schema':'megartx-prefill-attention-native-receipt-v1',
        'purpose':PURPOSE,'status':'independent_attention_error_observed',
        'plan_sha256':plan['plan_sha256'],'source_head':plan['source_head'],
        'request_identity_sha256':validated['request_identity_sha256'],
        'storage_binding_sha256':validated['storage_binding_sha256'],
        'capture_sha256':file_sha(evidence.directory/'attention-capture.json'),
        'errors_sha256':file_sha(evidence.directory/'attention-errors.json'),
        'storage_exact':True,'frontier_verified':True,'cleanup_complete':True,
        'contexts':1,'prompt_tokens':2048,'chunk_tokens':256,'capacity_tokens':2304,'emitted_outputs':256,
        'sampled_coordinates':480,'native_arithmetic_acceptance':None,
        **QUALIFICATIONS,'observer_gpu_scratch':validated['scratch'],
        'telemetry':validated['telemetry'],'raw_manifest':validated['raw_manifest'],
        'evidence_bytes_before_publication':evidence.sizes()}
    evidence.write('attention.json',receipt)
    return receipt


def publish_attention(evidence, plan, ownership_report, *, deadline, ownership):
    """One attempt; persist a bounded first failure without replacing its cause."""
    try:
        return _publish_attention(evidence, plan, ownership_report, deadline=deadline, ownership=ownership)
    except BaseException as primary:
        from .prefill_storage import add_failure_note
        try:
            if not (evidence.directory/'storage-failure.json').exists():
                evidence.write('storage-failure.json', {
                    'schema':'megartx-prefill-attention-publication-failure-v1',
                    'plan_sha256':plan.get('plan_sha256'), 'status':'invalidated',
                    'phase':'current_storage_validation_capture_or_cpu_analysis',
                    'exception_type':type(primary).__name__,
                    'native_arithmetic_acceptance':None})
        except BaseException as secondary:
            add_failure_note(primary,'Attention publication failure evidence unavailable: '+type(secondary).__name__)
        raise

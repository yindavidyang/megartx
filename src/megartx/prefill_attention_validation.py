"""Bounded CPU validation of the new sparse-attention artifact domain only.

The caller must independently validate the native storage/frontier/client and
owned-cleanup evidence. Hash-consistent files are not execution attestation.
This module cannot publish a successful native receipt or issue clearance.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from . import prefill_attention_plan as plan

CAPTURE_FILE = 'attention-capture.json'
CAPTURE_SCHEMA = 'megartx-prefill-attention-capture-v1'
BACKEND_SHA = '8ee541fde43ed92a417b3f01ea1e6fc64af8ab2eb54cbfc28308a49aaca4ed7f'
WRAPPER_SHA = {'fa2':'2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de',
               'xqa':'d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a'}
ENTRYPOINTS = {
    'paged_prefill':('fa2',WRAPPER_SHA['fa2']),
    'paged_decode':('fa2',WRAPPER_SHA['xqa']),
    'xqa_decode':('xqa',WRAPPER_SHA['xqa']),
}
MAX_DIRECTORY_ENTRIES = 256


def _names(directory_fd):
    names=set()
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            if len(names)>=MAX_DIRECTORY_ENTRIES:
                raise ValueError('Evidence directory entry limit exceeded')
            names.add(entry.name)
    return names


def semantic_contract(layer):
    d = 256 if layer == 0 else 512
    return {'q_dtype':'bfloat16','kv_dtype':'bfloat16','output_dtype':'bfloat16',
            'q_heads':16,'kv_heads':8 if layer==0 else 2,'head_dim':d,
            'window_left':1023 if layer==0 else -1,'causal':True,
            'sm_scale':1.0,'q_scale':1.0,'k_scale':1.0,'v_scale':1.0,
            'logits_soft_cap':0.0,'pos_encoding_mode':'NONE','custom_mask':False,
            'alibi':False,'sinks':False,'dcp':False,'cascade':False,'kv_sharing':False,
            'kernel_page_tokens':16,'manager_page_tokens':16,'manager_to_kernel_ratio':1}


def frame_identity(native_plan_sha256, request_identity_sha256, start, end, inputs_sha256):
    for h in (native_plan_sha256,request_identity_sha256,inputs_sha256):
        plan.sha(h)
    return plan.digest({'native_plan_sha256':native_plan_sha256,
        'request_identity_sha256':request_identity_sha256,'start':start,'end':end,
        'input_ids_sha256':inputs_sha256})


def record_roots(arrays, layer, start, end):
    """Independent raw-layout traversal, never a collector selection helper."""
    if layer == 0:
        dim,kh,qh,low = 256,1,2,max(0,start-1023)
    elif layer == 5:
        dim,kh,qh,low = 512,2,4,0
    else:
        raise ValueError('Unselected layer')
    prefix=f'attention-layer-{layer:02d}-'
    k,v=arrays[prefix+'k.bf16'],arrays[prefix+'v.bf16']
    tag=struct.pack('<III',layer,start,end)
    cache=hashlib.sha256(b'attention-cache-inputs\0'+tag)
    for p in range(low,end):
        cache.update(struct.pack('<I',p))
        cache.update(k[p*kh*dim*2:(p+1)*kh*dim*2])
        cache.update(v[p*kh*8*2:(p+1)*kh*8*2])
    hashes={'cache_inputs_sha256':cache.hexdigest()}
    # This explicit independent list must agree with the capture-plan tests.
    positions=(0,15,16,255,256,1023,1024,1792,2047,2048)
    for role,width,key in (('q',dim,'query_sha256'),('o',8,'output_sha256')):
        result=hashlib.sha256(('attention-'+role+'\0').encode()+tag)
        raw=arrays[prefix+role+'.bf16']
        for i,p in enumerate(positions):
            if start <= p < end:
                result.update(struct.pack('<I',p))
                result.update(raw[i*qh*width*2:(i+1)*qh*width*2])
        hashes[key]=result.hexdigest()
    return hashes


def _pairs(pairs):
    out={}
    for key,value in pairs:
        if key in out:
            raise ValueError('Duplicate metadata key')
        out[key]=value
    return out


def _read_regular(directory_fd,name,limit,exact=None):
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
    try:
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>limit
                or (exact is not None and before.st_size!=exact)):
            raise ValueError('Bounded single-link regular evidence file required: '+name)
        chunks=[];remaining=limit+1
        while remaining:
            chunk=os.read(fd,min(65536,remaining))
            if not chunk:break
            chunks.append(chunk);remaining-=len(chunk)
        raw=b''.join(chunks);after=os.fstat(fd)
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        if len(raw)>limit or len(raw)!=before.st_size or fields(before)!=fields(after):
            raise ValueError('Evidence file changed or overflowed while reading')
        return raw,fields(after)
    finally:
        os.close(fd)


def read_capture(directory, specification, *, expected_request_sha256,
                 expected_storage_binding_sha256, source_root=None):
    """Require independently supplied identities, not values copied from packet.

    expected_storage_binding_sha256 identifies the *current* already validated
    inherited storage/frontier domain; callers still own executing those gates.
    This function structurally binds the hash, not the external gate's truth.
    """
    plan.validate_plan(specification,source_root or Path(__file__).resolve().parents[2])
    plan.sha(expected_request_sha256);plan.sha(expected_storage_binding_sha256)
    directory=Path(directory)
    if directory.is_symlink() or any(p.is_symlink() for p in directory.absolute().parents):
        raise ValueError('Symlink evidence path refused')
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        layouts=plan.raw_layouts();names=_names(fd)
        if {n for n in names if n.endswith('.bf16')} != set(layouts):
            raise ValueError('Exactly eight raw files required; previous raw samples forbidden')
        if CAPTURE_FILE not in names or names & {'storage-failure.json','attention-failure.json','INVALIDATED.json'}:
            raise ValueError('Missing capture metadata or invalidated context')
        total=metadata=2  # reserve inherited external run.exit signal inside both limits
        snapshots={}
        for name in names:
            info=os.stat(name,dir_fd=fd,follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
                    or (name not in layouts and name!='.budget-lock' and not name.endswith(('.json','.jsonl')))):
                raise ValueError('Unknown, linked or nonregular evidence entry')
            if name=='.budget-lock' and info.st_size:
                raise ValueError('Budget lock must not carry hidden payload')
            total+=info.st_size;metadata+=0 if name in layouts else info.st_size
            snapshots[name]=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink)
        if total>8<<20 or metadata>2<<20:
            raise ValueError('One aggregate evidence/metadata budget exceeded')
        raw,_=_read_regular(fd,CAPTURE_FILE,1<<20)
        capture=json.loads(raw,object_pairs_hook=_pairs,
                           parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        fields={'schema','plan_sha256','native_plan_sha256','prompt_sha256','request_identity_sha256',
                'inherited_storage_binding_sha256','raw_manifest','records'}
        if type(capture) is not dict or set(capture)!=fields or capture['schema']!=CAPTURE_SCHEMA:
            raise ValueError('Exact attention capture metadata schema required')
        expected={'plan_sha256':specification['plan_sha256'],
                  'native_plan_sha256':specification['native_plan_sha256'],
                  'prompt_sha256':specification['prompt_sha256'],
                  'request_identity_sha256':expected_request_sha256,
                  'inherited_storage_binding_sha256':expected_storage_binding_sha256}
        if any(capture[k]!=v for k,v in expected.items()):
            raise ValueError('Capture request/plan/current storage binding mismatch')
        manifest=capture['raw_manifest']
        if type(manifest) is not dict or set(manifest)!=set(layouts):
            raise ValueError('Complete exact raw manifest required')
        arrays={}
        for name,layout in layouts.items():
            item=manifest[name]
            if (type(item) is not dict or set(item)!={'bytes','sha256'}
                    or type(item['bytes']) is not int or item['bytes']!=layout['bytes']):
                raise ValueError('Raw dtype/shape/extent manifest changed')
            plan.sha(item['sha256'])
            data,_=_read_regular(fd,name,layout['bytes'],layout['bytes'])
            if hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('Raw payload digest mismatch')
            plan.finite_words(data,layout['bytes']//2)
            arrays[name]=data
        records=capture['records']
        if type(records) is not list or len(records)!=12:
            raise ValueError('Exactly twelve selected native-call records required')
        frame_inputs={}
        record_fields={'layer','start','end','frame_input_ids_sha256','frame_identity_sha256',
                       'operator_binding_sha256','cache_mapping_sha256','semantics','dispatch',
                       'cache_inputs_sha256','query_sha256','output_sha256'}
        for record,(start,end,layer) in zip(records,((s,e,l) for s,e in plan.FRAMES for l in (0,5))):
            if (type(record) is not dict or set(record)!=record_fields
                    or any(type(record[k]) is not int or record[k]!=v for k,v in
                           (('layer',layer),('start',start),('end',end)))
                    or plan.digest(record['semantics'])!=plan.digest(semantic_contract(layer))):
                raise ValueError('Native call ordering, geometry, mask or scale changed')
            for key in record_fields-{'layer','start','end','semantics','dispatch'}:
                plan.sha(record[key])
            frame_hash=frame_identity(specification['native_plan_sha256'],expected_request_sha256,
                                      start,end,record['frame_input_ids_sha256'])
            if record['frame_identity_sha256']!=frame_hash:
                raise ValueError('End-aligned frame identity mismatch')
            if (start,end) in frame_inputs and frame_inputs[start,end]!=record['frame_input_ids_sha256']:
                raise ValueError('Layers disagree on actual frame inputs')
            frame_inputs[start,end]=record['frame_input_ids_sha256']
            if any(record[k]!=v for k,v in record_roots(arrays,layer,start,end).items()):
                raise ValueError('Actual query/output/cache root differs from retained bytes')
            dispatch=record['dispatch']
            if (type(dispatch) is not dict or set(dispatch)!={'provider','entrypoint','backend_source_sha256',
                    'wrapper_source_sha256','split_plan_sha256','kernel_binary_sha256','kernel_profile_sha256',
                    'native_rounding_contract'} or dispatch['entrypoint'] not in ENTRYPOINTS
                    or (dispatch['provider'],dispatch['wrapper_source_sha256']) != ENTRYPOINTS[dispatch['entrypoint']]
                    or dispatch['backend_source_sha256']!=BACKEND_SHA
                    or dispatch['native_rounding_contract'] is not None
                    or (end-start==256 and dispatch['entrypoint']!='paged_prefill')
                    or (end-start==1 and dispatch['entrypoint']=='paged_prefill')):
                raise ValueError('Unreviewed resolved dispatch/source/rounding claim')
            plan.sha(dispatch['split_plan_sha256'])
            for key in ('kernel_binary_sha256','kernel_profile_sha256'):
                if dispatch[key] is not None:plan.sha(dispatch[key])
        if _names(fd)!=names:
            raise ValueError('Evidence directory changed while reading')
        for name,before in snapshots.items():
            info=os.stat(name,dir_fd=fd,follow_symlinks=False)
            after=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink)
            if before!=after:raise ValueError('Evidence entry changed while reading')
        return {'arrays':arrays,'capture':capture,'evidence_bytes':total,'metadata_bytes':metadata,
                'new_domain_structurally_validated':True,'native_execution_attested':False,
                'external_native_storage_frontier_validation_required':True}
    finally:
        os.close(fd)

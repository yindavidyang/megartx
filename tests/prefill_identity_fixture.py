"""CPU-only execution of bounded verbatim pinned-vLLM request-ID extracts.

Only selected source methods / statements execute; no vLLM or torch imports.
Stubs stand in for data carriers and unrelated sampling/output behavior.
"""
from __future__ import annotations

import asyncio
from copy import copy
import hashlib
import json
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace
from typing import cast
import uuid

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'fixtures/prefill-native-request-identity.json').read_text())

def excerpt(name):
    item = DATA['extracts'][name]
    source = item['source']
    assert hashlib.sha256(source.encode()).hexdigest() == item['excerpt_sha256']
    return textwrap.dedent(source)

class RequestOutputKind:
    DELTA = 'delta'
    FINAL_ONLY = 'final'

class SamplingParams:
    def __init__(self, n):
        self.n = n
        self.output_kind = RequestOutputKind.DELTA
        self.seed = None

class CompletionOutput(SimpleNamespace):
    pass

class PoolingOutput(SimpleNamespace):
    pass

NS = dict(uuid=uuid, copy=copy, cast=cast, SamplingParams=SamplingParams,
    CompletionOutput=CompletionOutput, PoolingOutput=PoolingOutput,
    RequestOutput=SimpleNamespace, PoolingRequestOutput=SimpleNamespace,
    RequestOutputKind=RequestOutputKind,
    envs=SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),
    logger=SimpleNamespace(warning_once=lambda *a: None),
    RequestOutputCollector=lambda output_kind, request_id: SimpleNamespace(
        output_kind=output_kind, request_id=request_id))

def execute(name):
    exec('from __future__ import annotations\n' + excerpt(name), NS)

for name in ('mask_64_bits', 'random_uuid', 'base_request_id', 'assign_request_id'):
    execute(name)

parent_src = 'class ParentRequest:\n' + '\n'.join(
    textwrap.indent(excerpt(n), '    ')
    for n in ('parent_init', 'child_sampling_params', 'child_info'))
exec('from __future__ import annotations\n' + parent_src, NS)
fanout_src = 'async def upstream_fanout(self, request, params, is_pooling=False, prompt_text=None):\n' + textwrap.indent(excerpt('fanout_tail'), '    ')
exec('from __future__ import annotations\n' + fanout_src, NS)
for name in ('make_request_output', 'new_request_output'):
    execute(name)

async def run_case(n=1, disabled=False, prompt_index=0, base_id='fixed-request'):
    NS['envs'].VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = disabled
    # Controlled entropy to demonstrate exact formatting rather than chance.
    actual_random = NS['random_uuid']
    NS['random_uuid'] = lambda: '89abcdef01234567'
    try:
        base = NS['_base_request_id'](None, base_id)
        ids = dict(self=SimpleNamespace(_base_request_id=NS['_base_request_id']),
            raw_request=None, request=SimpleNamespace(request_id=base), i=prompt_index)
        exec(excerpt('completion_response_id'), ids)
        exec(excerpt('completion_prompt_id'), ids)
        external = ids['request_id_item']
        sse = dict(ids, CompletionStreamResponse=SimpleNamespace,
            CompletionResponseStreamChoice=SimpleNamespace, created_time=123,
            model_name='model', delta_text='', logprobs=None, finish_reason='length',
            stop_reason=None, prompt_token_ids_to_return=None, as_list=list,
            output=SimpleNamespace(token_ids=[3]))
        sse['request'] = SimpleNamespace(return_token_ids=True)
        exec(excerpt('completion_sse'), sse)
        assert sse['chunk'].id == ids['request_id']
        sse.update(final_usage_info={'prompt_tokens': 2, 'completion_tokens': 1},
            stream_per_request_metrics=None, self=SimpleNamespace(system_fingerprint=None))
        exec(excerpt('completion_usage_sse'), sse)
        assert sse['final_usage_chunk'].id == ids['request_id']
        params = SamplingParams(n)
        req = SimpleNamespace(request_id=external, external_req_id=None,
            params=params, sampling_params=params)
        observed = []
        async def add(request, prompt_text, parent, idx, queue):
            observed.append((copy(request), parent, idx))
        engine = SimpleNamespace(input_processor=SimpleNamespace(assign_request_id=NS['assign_request_id']),
            _run_output_handler=lambda: None, _add_request=add)
        queue = await NS['upstream_fanout'](engine, req, params)
        expected_parent = external if disabled else external + '-89abcdef'
        assert queue.request_id == expected_parent
        expected = [expected_parent] if n == 1 else [f'{j}_{expected_parent}' for j in range(n)]
        assert [r.request_id for r, _, _ in observed] == expected
        assert all(r.external_req_id == external for r, _, _ in observed)
        restored = []
        for r, parent, idx in observed:
            completion = CompletionOutput(index=idx)
            logprobs = SimpleNamespace(pop_prompt_logprobs=lambda: None)
            state = SimpleNamespace(external_req_id=r.external_req_id,
                request_id=r.request_id, parent_req=parent,
                output_kind=RequestOutputKind.DELTA, stream_interval=1,
                prompt_token_ids=[1, 2], prompt_embeds=None,
                logprobs_processor=logprobs, num_cached_tokens=0,
                num_cache_creation_tokens=0, lora_request=None,
                prompt=None, stats=None,
                _new_completion_output=lambda *a: completion)
            state._new_request_output = lambda *a, state=state: NS['_new_request_output'](state, *a)
            if parent is not None:
                parent.get_outputs = lambda *a: ([completion], True)
            output = NS['make_request_output'](state, [3], None, 'length', None)
            assert output.request_id == external
            restored.append(output.request_id)
        return dict(n=n, randomization_disabled=disabled, prompt_index=prompt_index,
            response_id=ids['request_id'], external_id=external,
            queue_id=queue.request_id, engine_ids=expected, output_request_ids=restored)
    finally:
        NS['random_uuid'] = actual_random

async def main():
    for supplied, expected in ((None, False), ('0', False), ('1', True)):
        ns = {'os': SimpleNamespace(getenv=lambda name, default: default if supplied is None else supplied)}
        exec('env_def = {\n' + excerpt('env_randomization_definition') + '\n}', ns)
        assert ns['env_def']['VLLM_DISABLE_REQUEST_ID_RANDOMIZATION']() is expected
    real_uuid = NS['uuid']
    NS['uuid'] = SimpleNamespace(uuid4=lambda: SimpleNamespace(int=0x123456789abcdef00000000000000001))
    assert NS['random_uuid']() == '0000000000000001'
    NS['uuid'] = real_uuid
    for _ in range(32):
        assert re.fullmatch('[0-9a-f]{16}', NS['random_uuid']())
    assert NS['_base_request_id'](SimpleNamespace(headers={'X-Request-Id': 'header'}), 'body') == 'header'
    assert NS['_base_request_id'](None, 'body') == 'body'
    bad = SimpleNamespace(request_id='r', external_req_id='already-set')
    try:
        NS['assign_request_id'](bad)
    except ValueError:
        pass
    else:
        raise AssertionError('preassigned external identity must fail')
    results = [await run_case(n, disabled, i)
        for n in (1, 2) for disabled in (False, True) for i in (0, 1)]
    assert not any(k == 'vllm' or k.startswith('vllm.') or k == 'torch' or k.startswith('torch.')
        for k in __import__('sys').modules)
    print(json.dumps({'status': 'passed', 'commit': DATA['commit'], 'cases': results}, indent=2))

if __name__ == '__main__':
    asyncio.run(main())

"""Source-executed API/internal identity and ordered token transport on CPU.

External tensors/data carriers are substitutes. No vLLM/torch imports, native
runtime admission, device work, fit publication, or numerical claims.
"""
import asyncio
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

from megartx.prefill_diagnostic_plan import (INSTALLED, TRANSPORT_FILES, Evidence, api_request_id, digest,
    native_request_identity, native_ledger_comparison, publish_fit)
from megartx.prefill_native import RequestLedger, NativeProvider
from prefill_identity_fixture import DATA, NS as ID_NS, run_case, excerpt
from test_prefill_native_head import Tensor, TORCH, sampled_counts
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prefill_diagnostic_client import StreamLedger, LedgerMismatch, validate_observation, payload, run, stream_observation

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT = json.loads((ROOT/'tests/fixtures/prefill-native-token-transport.json').read_text())
PLAN = {'plan_sha256': 'a'*64, 'tokens': list(range(2048))}


def source_method(name, **scope):
    value = TRANSPORT['extracts'][name]
    assert hashlib.sha256(value['source'].encode()).hexdigest() == value['excerpt_sha256']
    exec('from __future__ import annotations\n'+textwrap.dedent(value['source']), scope)
    return next(scope[k] for k in scope if k.startswith(('get_output', '_update_request',
               '_new_completion', 'completion_stream')))


class Carrier(NS):
    def model_dump_json(self, **kwargs):
        return json.dumps(self, default=lambda obj: vars(obj))


def source_engine_id(plan=PLAN):
    return asyncio.run(run_case(base_id=api_request_id(plan)))['engine_ids'][0]


def source_transport():
    """263 actual count transitions -> trimming -> scheduler -> output -> SSE."""
    engine_id = source_engine_id()
    request = NS(request_id=engine_id, output_token_ids=[])
    request.append_output_token_ids = request.output_token_ids.append
    trim = source_method('async_output')
    schedule = source_method('scheduler_update', check_stop=lambda req, cap: len(req.output_token_ids) == 256)
    complete = source_method('completion_output', RequestOutputKind=ID_NS['RequestOutputKind'],
                            CompletionOutput=ID_NS['CompletionOutput'])
    stream_source = source_method('completion_stream',
        should_include_usage=lambda options, force: (options.include_usage, False),
        as_list=list, CompletionStreamResponse=Carrier, CompletionResponseStreamChoice=Carrier,
        UsageInfo=Carrier, build_spec_decoding_metrics=lambda res: None, GenerationError=type('GenerationError',(Exception,),{}),
        logger=NS(exception=lambda *args: None))
    provider = NativeProvider.__new__(NativeProvider)
    provider.plan, provider.ledger, provider.torch = PLAN, RequestLedger(PLAN['tokens']), TORCH
    provider.started, provider.completed, provider.failed = True, False, False
    provider.access = NS(_runner=NS(device=TORCH.zeros(1,dtype='int32').device),
                         head_binding=NS(receipt=lambda:{}))
    provider.scratch = NS(preserve_peaks=lambda:None, receipt=lambda:{})
    provider.head_counts = dict.fromkeys(('intermediate_prompt_chunk','final_prompt','decode'),0)
    records=[]
    provider.evidence=NS(write=lambda name,value,**kw:records.append((name,value)))
    output_type=type('SamplerOutput',(),{'__module__':'vllm.v1.worker.gpu.sample.output'})
    identities={i:('cpu-identity',i) for i in range(30)}
    request_outputs=[]
    for sequence,start in enumerate([*range(0,2048,256),*range(2048,2303)]):
        rows=256 if start<2048 else 1
        tokens=PLAN['tokens'][start:start+rows] if start<2048 else provider.ledger.outputs[-1:]
        provider.ledger.begin(tokens,list(range(start,start+rows)),
            {i:list(range(start,start+rows)) for i in range(30)},identities,engine_id)
        provider.ledger.complete()
        count,rejected=sampled_counts(start+rows)
        token=4000+sequence  # Distinct synthetic IDs make order/drop/duplicate errors visible.
        sampled=output_type();sampled.sampled_token_ids=Tensor([[token]],'int64')
        sampled.num_sampled,sampled.num_rejected=Tensor([count],'int32'),Tensor([rejected],'int32')
        phase='intermediate_prompt_chunk' if start+rows<2048 else 'final_prompt' if start+rows==2048 else 'decode'
        provider.head_counts[phase]+=1;provider.logit_seen=True
        provider.sampled(sampled)
        copied=NS(copy_event=NS(synchronize=lambda:None),sampled_token_ids=sampled.sampled_token_ids,
            num_sampled_tokens_np=sampled.num_sampled,model_runner_output=NS(),sampling_mask_tensors=None,
            num_nans=None,logprobs_tensors=None,prompt_logprobs_dict=None,routed_experts_cpu=None,_has_fault=None)
        generated=trim(copied).sampled_token_ids[0]
        emitted,stopped=schedule(NS(max_model_len=2304),request,generated)
        detokenizer=NS(get_next_output_text=lambda *args:'',output_token_ids=request.output_token_ids)
        state=NS(output_kind=ID_NS['RequestOutputKind'].DELTA,detokenizer=detokenizer,
            logprobs_processor=NS(logprobs=None,cumulative_logprob=None,pop_prompt_logprobs=lambda:None),
            sampling_mask_chunks=[],routed_experts_chunks=[],request_index=0,spec_decode_metrics=None,
            external_req_id=engine_id[:-9],request_id=engine_id,parent_req=None,stream_interval=1,
            prompt_token_ids=PLAN['tokens'],prompt_embeds=None,num_cached_tokens=0,
            num_cache_creation_tokens=0,lora_request=None,prompt='',stats=None)
        state._new_completion_output=lambda *args,state=state:complete(state,*args)
        state._new_request_output=lambda *args,state=state:ID_NS['_new_request_output'](state,*args)
        result=ID_NS['make_request_output'](state,emitted,None,'length' if stopped else None,None)
        request_outputs.append(result)
    async def records_source():
        for result in request_outputs: yield 0,result
    def unexpected_error(error):raise error
    serving=NS(enable_force_include_usage=False,enable_prompt_tokens_details=False,
        enable_per_request_metrics=False,system_fingerprint=None,_raise_if_error=lambda *args:None,
        create_streaming_error_response=unexpected_error)
    metadata=NS()
    req=NS(n=1,max_tokens=256,echo=False,return_token_ids=True,logprobs=None,
           stream_options=NS(include_usage=True))
    async def consume():
        ledger=StreamLedger(PLAN)
        async for record in stream_source(serving,req,[{}],records_source(),
                'cmpl-'+api_request_id(PLAN),0,'cpu-model',1,None,metadata):
            ledger.consume(record.strip())
        return ledger
    ledger=asyncio.run(consume())
    observer=next(value for name,value in records if name=='observer.json')
    return ledger,observer,metadata,request_outputs,records


class IdentityContractTests(unittest.TestCase):
    def test_source_pins_for_every_executed_identity_and_transport_file(self):
        for data in (DATA,TRANSPORT):
            for value in data['extracts'].values():
                module=value['path'].removesuffix('/__init__.py').removesuffix('.py').replace('/','.')
                # Core transport extracts remain research-only; every executed
                # identity, API and token-transport source is admitted by hash.
                pins={**INSTALLED,**TRANSPORT_FILES}
                if module not in pins:
                    self.assertIn(module,{'vllm.v1.engine','vllm.v1.engine.core','vllm.v1.engine.core_client',
                        'vllm.v1.core.sched.output','vllm.v1.request'})
                else:self.assertEqual(pins[module],value['file_sha256'])
                self.assertEqual(hashlib.sha256(value['source'].encode()).hexdigest(),value['excerpt_sha256'])

    def test_api_files_are_pinned_without_requiring_frontend_modules_in_gpu_worker(self):
        from megartx.prefill_runner_binding import verify_installed_files
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'__init__.py').write_text('# substituted vllm source root\n')
            api=root/'entrypoints/openai/completion/serving.py';api.parent.mkdir(parents=True)
            api.write_text('# substituted API source file, deliberately not imported\n')
            package=ModuleType('vllm');package.__file__=str(root/'__init__.py')
            pins={'vllm.entrypoints.openai.completion.serving':hashlib.sha256(api.read_bytes()).hexdigest()}
            with patch.dict(sys.modules,{'vllm':package}), \
                 patch('megartx.prefill_runner_binding.INSTALLED',{}), \
                 patch('megartx.prefill_runner_binding.TRANSPORT_FILES',pins):
                verify_installed_files()
                self.assertNotIn('vllm.entrypoints.openai.completion.serving',sys.modules)
                api.write_text('# drift\n')
                with self.assertRaisesRegex(RuntimeError,'source drift'):verify_installed_files()

    def test_exact_upstream_identity_path_and_rejected_alternate_contracts(self):
        for n in (1,2):
            for disabled in (False,True):
                for prompt_index in (0,1):
                    result=asyncio.run(run_case(n,disabled,prompt_index,api_request_id(PLAN)))
                    self.assertEqual(result['output_request_ids'],[result['external_id']]*n)
                    for actual in result['engine_ids']:
                        if (n,disabled,prompt_index)==(1,False,0):
                            self.assertEqual(native_request_identity(PLAN,actual)['external_request_id'],result['external_id'])
                        else:
                            with self.assertRaises(ValueError):native_request_identity(PLAN,actual)

    def test_no_missing_extra_or_arbitrary_suffix_normalization(self):
        good=source_engine_id()
        for bad in (None,0,good[:-9],good+'-0',good+'x',good[:-1],good[:-8]+'ZZZZZZZZ',
                    good[:-8]+'0123456-',good[:-8]+'0123456\n','wrong-'+good,'0_'+good,
                    good.replace('-0-','-1-'),good.replace('cmpl-','chatcmpl-')):
            with self.subTest(value_type=type(bad).__name__),self.assertRaises(ValueError):
                native_request_identity(PLAN,bad)
        ledger=RequestLedger(PLAN['tokens']);identities={i:i for i in range(30)}
        ledger.begin(PLAN['tokens'][:256],list(range(256)),{i:list(range(256)) for i in range(30)},identities,good)
        ledger.complete();ledger.sampled(1,True)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            ledger.begin(PLAN['tokens'][256:512],list(range(256,512)),
                {i:list(range(256,512)) for i in range(30)},identities,good[:-8]+'76543210')


class TokenAndComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.stream,cls.observer,cls.metadata,cls.outputs,cls.records=source_transport()

    def test_source_transport_preserves_all_256_including_terminal_token(self):
        self.assertEqual(self.stream.tokens,list(range(4007,4263)))
        self.assertEqual(self.outputs[0].outputs[0].token_ids,[])
        self.assertEqual(self.outputs[-1].outputs[0].token_ids,[4262])
        self.assertEqual(self.stream.events,258)
        self.assertEqual(vars(self.metadata.final_usage_info),{'prompt_tokens':2048,'completion_tokens':256,'total_tokens':2304})
        client=validate_observation(PLAN,self.stream,self.observer)
        self.assertEqual(client['output_ids_sha256'],digest(list(range(4007,4263))))
        self.assertEqual(self.observer['head_counts'],{'intermediate_prompt_chunk':7,'final_prompt':1,'decode':255})
        self.assertEqual(len([r for r in self.records if r[0]=='samples.jsonl']),256)
        self.assertFalse(client['numerical_qualified']);self.assertFalse(client['performance_qualified'])

    def test_each_comparison_field_diagnosed_before_throw_without_raw_ids(self):
        mutations={'schema':'bad','status':'bad','plan_sha256':'0'*64,'engine_request_id':'wrong',
            'output_ids_sha256':'0'*64,'prompt_frames':9,'decode_input_rows':254,
            'emitted_outputs':255,'committed_length':2304,'numerical_qualified':True,
            'performance_qualified':True,'request_identity':{}}
        for key,bad in mutations.items():
            with self.subTest(field=key),tempfile.TemporaryDirectory() as d:
                with self.assertRaises(LedgerMismatch) as caught:
                    validate_observation(PLAN,self.stream,{**self.observer,key:bad},Evidence(d))
                diagnostic=json.loads((Path(d)/'client-ledger-check.json').read_text())
                self.assertEqual(diagnostic,caught.exception.diagnostic)
                self.assertEqual(diagnostic['status'],'mismatch')
                self.assertTrue(diagnostic['failed_fields'])
                self.assertFalse((Path(d)/'client.json').exists());self.assertFalse((Path(d)/'fit.json').exists())
                self.assertNotIn(self.observer['engine_request_id'],str(caught.exception))
                self.assertNotIn(str(self.stream.tokens),str(caught.exception))

    def test_same_count_reorder_duplicate_and_terminal_substitution_still_fail_hash(self):
        for mutate in (lambda t:t.reverse(),lambda t:t.__setitem__(100,t[99]),lambda t:t.__setitem__(-1,999)):
            stream=copy.deepcopy(self.stream);mutate(stream.tokens)
            with self.assertRaises(LedgerMismatch) as caught:validate_observation(PLAN,stream,self.observer)
            self.assertIn('output_ids_sha256',caught.exception.diagnostic['failed_fields'])
        for count in (255,257):
            stream=copy.deepcopy(self.stream);stream.tokens=(stream.tokens+[888])[:count]
            with self.assertRaises(LedgerMismatch) as caught:validate_observation(PLAN,stream,self.observer)
            self.assertIn('client.emitted_outputs',caught.exception.diagnostic['failed_fields'])

    def test_no_truthy_or_numeric_type_substitution_for_counts_and_flags(self):
        for key,value in [('prompt_frames',8.0),('decode_input_rows',255.0),('emitted_outputs',256.0),
                          ('committed_length',2303.0),('numerical_qualified',0),('performance_qualified',0)]:
            with self.subTest(key=key),self.assertRaises(LedgerMismatch):
                validate_observation(PLAN,self.stream,{**self.observer,key:value})
        for key,value in [('done',1),('finish',None),('usage',{'prompt_tokens':2048.0,'completion_tokens':256,'total_tokens':2304})]:
            stream=copy.deepcopy(self.stream);setattr(stream,key,value)
            with self.assertRaises(LedgerMismatch):validate_observation(PLAN,stream,self.observer)

    def test_sse_missing_terminal_usage_done_and_tokens_remain_rejected(self):
        for ids,finish in (([1]*255,'length'),([1]*257,'length'),([True],None),(None,None)):
            stream=StreamLedger(PLAN)
            with self.assertRaises(ValueError):stream.consume('data: '+json.dumps({'id':stream.response_id,
                'choices':[{'index':0,'token_ids':ids,'finish_reason':finish}]}))
        for usage in ({'prompt_tokens':2048.0,'completion_tokens':256,'total_tokens':2304},
                      {'prompt_tokens':2048,'completion_tokens':256}):
            stream=StreamLedger(PLAN)
            with self.assertRaises(ValueError):stream.consume('data: '+json.dumps({'id':stream.response_id,'choices':[],'usage':usage}))
        stream=StreamLedger(PLAN)
        with self.assertRaises(ValueError):stream.consume('data: [DONE]')

    def test_client_stream_facts_saved_before_identity_failure_or_missing_observer(self):
        lines=[{'id':self.stream.response_id,'choices':[{'index':0,'token_ids':self.stream.tokens,'finish_reason':'length'}]},
               {'id':self.stream.response_id,'choices':[],'usage':self.stream.usage}]
        raw=('\n\n'.join('data: '+json.dumps(line) for line in lines)+'\n\ndata: [DONE]\n\n').encode()
        class Response:
            def __init__(self):self.raw=io.BytesIO(raw)
            def raise_for_status(self):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
        for observer in (None,{**self.observer,'engine_request_id':self.stream.response_id+'-0'}):
            with tempfile.TemporaryDirectory() as d:
                if observer is not None:Path(d,'observer.json').write_text(json.dumps(observer))
                session=NS(post=lambda *args,**kw:Response(),close=lambda:None)
                requests=ModuleType('requests');requests.Session=lambda:session
                with patch.dict(sys.modules,{'requests':requests}), \
                     patch('megartx.prefill_runner_binding.validate_binding',return_value={}), \
                     patch('prefill_diagnostic_client.remaining',return_value=30), \
                     self.assertRaises(FileNotFoundError if observer is None else LedgerMismatch):
                    run(PLAN,d,0)
                receipt=json.loads(Path(d,'client-stream.json').read_text())
                self.assertEqual(receipt['output_ids_sha256'],self.observer['output_ids_sha256'])
                self.assertEqual(receipt['emitted_outputs'],256)
                self.assertEqual(receipt['sse_events'],3)
                self.assertIs(receipt['stream_done'],True)
                self.assertEqual(receipt['finish_reason'],'length')
                self.assertEqual(receipt['usage'],self.stream.usage)
                self.assertFalse(Path(d,'client.json').exists())
                self.assertFalse(Path(d,'fit.json').exists())
                self.assertFalse(Path(d,'request.json').exists())

    def test_fit_revalidates_mapping_usage_and_tokens_before_publication(self):
        client=validate_observation(PLAN,self.stream,self.observer)
        cases=[({**self.observer,'engine_request_id':self.stream.response_id+'-0'},client),
            (self.observer,{**client,'usage':{'prompt_tokens':2048,'completion_tokens':255,'total_tokens':2303}}),
            (self.observer,{**client,'output_ids_sha256':'0'*64}),
            (self.observer,{**client,'engine_request_id':source_engine_id()[:-8]+'76543210'})]
        for observer,bad_client in cases:
            with tempfile.TemporaryDirectory() as d,patch('megartx.prefill_runner_binding.validate_binding',return_value={}):
                for name,value in [('observer.json',observer),('client.json',bad_client),('client-stream.json',stream_observation(PLAN,self.stream)),('loaded.json',{}),('geometry.json',{})]:
                    Path(d,name).write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'Native fit ledger mismatch'):publish_fit(Evidence(d),PLAN,{})
                self.assertFalse(Path(d,'fit.json').exists())


if __name__=='__main__':unittest.main()

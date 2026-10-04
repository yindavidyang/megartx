"""Pinned source control flow and raw-head serialization on CPU substitutes.

No torch/vLLM/model import, device work, native numerical admission, or fit.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import MethodType, SimpleNamespace as NS
import unittest

import numpy as np

from megartx.controlled_kv_capture import SOURCE_HASHES
from megartx.loaded_engine_access import HEAD_SOURCES, NativeHeadBinding
from megartx.prefill_diagnostic_plan import INSTALLED, digest
from megartx.prefill_native import NativeProvider, RequestLedger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
FIXTURE = json.loads((ROOT/'tests/fixtures/prefill-native-head-selection.json').read_text())
DEVICE = 'cpu-substituted-cuda-boundary'


class Tensor:
    """Logical dtype substitute; BF16 byte conversion is only a fixture codec."""
    def __init__(self, values, dtype='bf16', device=DEVICE):
        self.values = np.asarray(values)
        self.dtype, self.device = dtype, device
    @property
    def shape(self): return self.values.shape
    @property
    def ndim(self): return self.values.ndim
    @property
    def is_cuda(self): return True
    def __len__(self): return len(self.values)
    def __getitem__(self, key):
        if isinstance(key, Tensor): key = key.values.astype(np.int64)
        return Tensor(self.values[key], self.dtype, self.device)
    def __truediv__(self, rhs): return Tensor(self.values/rhs, self.dtype, self.device)
    def __mul__(self, rhs): return Tensor(self.values*rhs, self.dtype, self.device)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def tolist(self): return self.values.tolist()
    def item(self): return self.values.item()
    def reshape(self, *shape): return Tensor(self.values.reshape(*shape), self.dtype, self.device)
    def view(self, dtype, *shape):
        if isinstance(dtype,int): return self.reshape(dtype,*shape)
        assert dtype == 'uint8'
        if self.dtype == 'bf16':
            value = (self.values.astype('<f4').view('<u4') >> 16).astype('<u2')
        elif self.dtype == 'fp32': value = self.values.astype('<f4')
        else: raise AssertionError('unexpected fixture raw conversion')
        return NS(numpy=lambda: value.view('u1'))
    def index_fill_(self, axis, indices, value): self.values[:, indices.values.astype(np.int64)] = value
    def new_ones(self, n): return Tensor(np.ones(n, dtype=np.int32), 'int32')


TORCH = NS(dtype=type('DType',(),{}), Tensor=Tensor, bfloat16='bf16', float32='fp32', int64='int64', int32='int32',
           long='int64', uint8='uint8', tanh=lambda t: Tensor(np.tanh(t.values), t.dtype),
           mm=lambda flat, weight, out_dtype: Tensor(np.zeros((len(flat),262144)), out_dtype),
           zeros=lambda n, **kw: Tensor(np.zeros(n), kw['dtype']),
           arange=lambda n, **kw: Tensor(np.arange(n), kw['dtype']))


def excerpt(key, globals_=None):
    scope = {'torch': TORCH, 'np': np, **(globals_ or {})}
    text = FIXTURE['excerpts'][key]['source']
    exec('from __future__ import annotations\n'+text, scope)
    return scope


class Pointer:
    def __init__(self, values, index=0): self.values, self.index = np.asarray(values), index
    def __add__(self, index): return Pointer(self.values, self.index+index)


class TritonCPU:
    constexpr = object
    program_id = staticmethod(lambda axis: 0)
    arange = staticmethod(np.arange)
    where = staticmethod(np.where)
    @staticmethod
    def load(p, mask=None): return p.values[p.index] if mask is None else p.values[p.index][mask]
    @staticmethod
    def store(p, value, mask=None):
        index = p.index if mask is None else np.asarray(p.index)[mask]
        values = value if mask is None else np.asarray(value)[mask]
        p.values[index] = values


SELECT = excerpt('select_kernel', {'tl': TritonCPU})['_combine_sampled_and_draft_tokens_kernel']
COUNTS = excerpt('counts_kernel', {'tl': TritonCPU})['_get_num_sampled_and_rejected_kernel']
SAMPLE = excerpt('sample')['sample']
CONDITIONAL = excerpt('conditional_head', {'async_tensor_h2d': lambda x, **kw: Tensor(x, kw['dtype'])})['compute_logits']
LANGUAGE = excerpt('language_head')['compute_logits']


def selected_index(start, rows, num_logits=1):
    ids, indices = Pointer(np.zeros(rows, dtype=np.int32)), Pointer(np.full(num_logits, -1, dtype=np.int64))
    SELECT(ids, Pointer([0]), Pointer([17]), Pointer([0, rows]), Pointer([start+rows]),
           Pointer([2048]), Pointer(np.zeros(max(1, num_logits))), max(1, num_logits),
           Pointer([0, num_logits]), indices, max(1, num_logits), 1)
    return Tensor(indices.values, 'int64'), ids.values


def sampled_counts(end, num_logits=1):
    sampled, rejected = Pointer([1]), Pointer([-1])
    COUNTS(sampled, rejected, Pointer([end]), Pointer([0, num_logits]), Pointer([0]), Pointer([2048]))
    return int(sampled.values[0]), int(rejected.values[0])


def sampler_fixture(runner, batch, start, rows, tokens):
    output_type = type('SamplerOutput', (), {'__module__':'vllm.v1.worker.gpu.sample.output'})
    output_type.__init__ = lambda self, **kw: self.__dict__.update(kw)
    def counts(sampled, lengths, cumulative, mapping, prefill):
        out, rejected = Pointer(sampled.values.copy()), Pointer([-1])
        COUNTS(out, rejected, Pointer(lengths.values), Pointer(cumulative.values),
               Pointer(mapping.values), Pointer(prefill.values))
        return Tensor(out.values,'int32'), Tensor(rejected.values,'int32')
    actual = excerpt('sampler_forward', {'SamplerOutput':output_type,
                     'get_num_sampled_and_rejected':counts})['__call__']
    class Sampler:
        __call__ = actual
        def sample(self, logits, expanded, mapping, mapping_np, positions, ids, local, **kwargs):
            self.selection = (positions.tolist(), ids.tolist())
            return Tensor([17],'int64'), logits
    sampler = Sampler()
    sampler.compute_nans, sampler.return_sampling_mask, sampler.trace_replay_state = False,False,None
    sampler.get_logprobs_dims = lambda mapping: None
    sampler.req_states = NS(prefill_len=NS(gpu=Tensor([2048],'int32')))
    batch.positions, batch.input_ids = Tensor(range(start,start+rows),'int64'),Tensor(tokens,'int32')
    batch.expanded_idx_mapping = batch.idx_mapping = Tensor([0],'int64')
    batch.idx_mapping_np = np.array([0])
    batch.cu_num_logits_np = np.array([0,1])
    batch.cu_num_logits = Tensor([0,1],'int32')
    batch.expanded_local_pos = Tensor([0],'int32')
    batch.seq_lens = Tensor([start+rows],'int32')
    runner.sampler, runner.model = sampler, runner.get_model()
    return sampler


def head_fixture(end=256, rows=256, head_dtype='bf16', suppressed=(5, 17)):
    class QuantMethod:
        def apply(self, layer, hidden, bias=None): return Tensor(np.zeros((len(hidden), 262144)), hidden.dtype)
    QuantMethod.__module__, QuantMethod.__name__ = 'vllm.model_executor.layers.vocab_parallel_embedding', 'UnquantizedEmbeddingMethod'
    class Head: pass
    Head.__module__, Head.__name__ = 'vllm.model_executor.layers.vocab_parallel_embedding', 'ParallelLMHead'
    class Processor:
        def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    Processor.__module__, Processor.__name__ = 'vllm.model_executor.layers.logits_processor', 'LogitsProcessor'
    env = {'UnquantizedEmbeddingMethod': QuantMethod, 'UnquantizedLinearMethod': type('Unused',(),{}),
           'current_platform': NS(is_cuda=lambda:True, is_rocm=lambda:False)}
    for name in ('forward', '_get_logits', '_apply_head'):
        setattr(Processor, name, excerpt('processor'+name, env)[name])
    class Language: compute_logits = LANGUAGE
    Language.__module__, Language.__name__ = 'vllm.model_executor.models.gemma4', 'Gemma4ForCausalLM'
    class Model: compute_logits = CONDITIONAL
    Model.__module__, Model.__name__ = 'vllm.model_executor.models.gemma4_mm', 'Gemma4ForConditionalGeneration'
    model, language, processor, head = Model(), Language(), Processor(), Head()
    model.language_model, model._suppress_token_ids = language, list(suppressed)
    language.logits_processor, language.lm_head = processor, head
    processor.head_dtype, processor.soft_cap, processor.scale = head_dtype, 30.0, 1.0
    processor.logits_as_input, processor.vocab_size, processor.org_vocab_size = False, 262144, 262144
    head.quant_method, head.tp_size = QuantMethod(), 1
    head.weight = NS(dtype='bf16', device=DEVICE, shape=(262144, 2816), t=lambda:object())
    runner = NS(device=DEVICE, get_model=lambda: model, batch_sharder=None, rejection_sampler=None,
                vllm_config=NS(model_config=NS(dtype='bf16', head_dtype=head_dtype)))
    binding = NativeHeadBinding(runner, model, TORCH)
    provider = NativeProvider.__new__(NativeProvider)
    provider.started, provider.completed, provider.failed = True, False, False
    provider.frame, provider.logit_seen, provider.hidden = None, False, Tensor(np.zeros((rows, 2816)))
    provider.logits_indices, _ = selected_index(end-rows, rows)
    batch = NS(logits_indices=provider.logits_indices, num_reqs=1, num_draft_tokens=0)
    provider.access = NS(_runner=runner, model=model, input_batch=batch, head_binding=binding, common={})
    provider.torch = TORCH
    provider.head_counts = dict.fromkeys(('intermediate_prompt_chunk', 'final_prompt', 'decode'), 0)
    provider.ledger = NS(frames=1, end=end)
    records = []
    provider.evidence = NS(write=lambda name, value, **kw: records.append((name, value)))
    return provider, model, runner, batch, records


class SourceSelectionTests(unittest.TestCase):
    def test_exact_source_pins_and_no_draft_single_logit_branch(self):
        pins = {**INSTALLED, **SOURCE_HASHES, **HEAD_SOURCES}
        self.assertEqual(FIXTURE['files'], {key: pins[key] for key in FIXTURE['files']})
        scope = {'np': np, 'torch': TORCH, 'num_reqs': 1, 'idx_mapping': object(), 'self': NS(device=DEVICE)}
        exec(FIXTURE['excerpts']['ordinary_logits_count']['source'], scope)
        self.assertEqual(scope['total_num_logits'], 1)
        self.assertEqual(scope['cu_num_logits_np'].tolist(), [0, 1])

    def test_actual_model_config_resolves_default_bf16_and_explicit_fp32(self):
        env = {'current_platform':NS(supported_dtypes=['bf16','fp32']),
               '_STR_DTYPE_TO_TORCH_DTYPE':{'float32':'fp32','bfloat16':'bf16'},
               'logger':NS(debug_once=lambda *a:None, warning_once=lambda *a:None)}
        env['_get_head_dtype']=excerpt('get_head_dtype',env)['_get_head_dtype']
        resolve=excerpt('head_dtype',env)['head_dtype']
        for config,expected in ((NS(),'bf16'),(NS(head_dtype='model'),'bf16'),(NS(head_dtype='float32'),'fp32')):
            self.assertEqual(resolve(NS(hf_config=config,dtype='bf16',runner_type='generate')),expected)

    def test_actual_kernel_selects_one_chunk_tail_and_only_rewrites_decode(self):
        for start, rows in ((0, 256), (256, 256), (1792, 256), (2048, 1), (2302, 1)):
            with self.subTest(start=start):
                indices, ids = selected_index(start, rows)
                self.assertEqual(indices.tolist(), [rows-1])
                self.assertEqual(ids.tolist(), [0]*rows if start < 2048 else [17])
                self.assertEqual(sampled_counts(start+rows), (0, 0) if start+rows < 2048 else (1, 0))

    def test_actual_runner_zero_and_multiple_selection_cannot_be_qualified(self):
        for count in (0,2):
            p,model,runner,batch,records=head_fixture()
            p.logits_indices,_=selected_index(0,256,count)
            batch.logits_indices=p.logits_indices
            runner.model=model
            runner.sampler=lambda *args: self.fail('Malformed head reached sampling')
            def observed(model,hidden):
                logits=CONDITIONAL(model,hidden)
                p.head(model,hidden,logits)
                return logits
            model.compute_logits=MethodType(observed,model)
            with self.subTest(count=count),self.assertRaisesRegex(RuntimeError,'shape/dtype/device'):
                SAMPLE(runner,p.hidden,batch,None)
            self.assertEqual(records[0][1]['hidden']['rows'],count)
            self.assertEqual(records[0][1]['logits']['rows'],count)
            self.assertEqual(len(records),1)
            self.assertFalse(p.logit_seen)

    def test_actual_runner_calls_interim_head_and_defers_discard_to_sampler(self):
        for end in (256, 1792, 2048, 2049):
            p, model, runner, batch, records = head_fixture(end, 256 if end <= 2048 else 1)
            rows = len(p.hidden)
            sampler = sampler_fixture(runner,batch,end-rows,rows,list(range(rows)))
            output, count, rejected = SAMPLE(runner, p.hidden, batch, None)
            self.assertEqual(output.sampled_token_ids.shape,(1,1))
            self.assertEqual(output.sampled_token_ids.dtype,'int64')
            self.assertEqual((count.dtype,rejected.dtype),('int32','int32'))
            self.assertEqual((count.item(), rejected.item()), (0, 0) if end < 2048 else (1, 0))
            self.assertEqual(sampler.selection,([end-1],[rows-1]))
            self.assertEqual(model._suppress_token_ids_cache[DEVICE].tolist(), [5, 17])


class HeadBindingTests(unittest.TestCase):
    def test_binds_native_dtype_without_cast_and_exact_suppression(self):
        for setting, dtype in ((None, 'bf16'), ('bf16', 'bf16'), ('fp32', 'fp32')):
            p, model, runner, batch, records = head_fixture(head_dtype=setting)
            self.assertEqual(p.access.head_binding.dtype, dtype)
            self.assertEqual(p.access.head_binding.suppressed, frozenset((5, 17)))
            actual = CONDITIONAL(model, p.hidden[p.logits_indices])
            self.assertEqual((actual.shape,actual.dtype),((1,262144),dtype))
            p.head(model,p.hidden[p.logits_indices],actual)
            p.access.head_binding.require_current()

    def test_rejects_dtype_math_owner_callback_and_mask_drift(self):
        for kind in ('setting', 'model-dtype', 'weight', 'weight-dtype', 'processor', 'callback',
                     'suppression', 'suppression-bool', 'softcap', 'vocab'):
            p, model, runner, batch, records = head_fixture()
            head = p.access.head_binding
            if kind == 'setting': head.processor.head_dtype = 'fp32'
            if kind == 'model-dtype': runner.vllm_config.model_config.dtype = 'fp32'
            if kind == 'weight': head.head.weight = NS(dtype='bf16', device=DEVICE, shape=(262144,2816))
            if kind == 'weight-dtype': head.weight.dtype = 'fp32'
            if kind == 'processor': model.language_model.logits_processor = object()
            if kind == 'callback': head.processor.forward = lambda *args: None
            if kind == 'suppression': model._suppress_token_ids = [5, 18]
            if kind == 'suppression-bool': model._suppress_token_ids = [True]
            if kind == 'softcap': head.processor.soft_cap = 29.0
            if kind == 'vocab': head.processor.org_vocab_size -= 1
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): head.require_current()

    def test_invalid_or_all_suppressed_mask_rejected_before_head(self):
        for mask in ((True,),(-1,),(262144,),(1,1),tuple(range(262144))):
            with self.subTest(size=len(mask)),self.assertRaisesRegex(RuntimeError,'suppression mask'):
                head_fixture(suppressed=mask)



class HeadObservationTests(unittest.TestCase):
    def invoke(self, p, model):
        selected = p.hidden[p.logits_indices]
        logits = CONDITIONAL(model, selected)
        p.head(model, selected, logits)
        return logits

    def test_interim_final_decode_receipts_keep_causal_position_and_raw_dtype(self):
        for end, phase in ((256, 'intermediate_prompt_chunk'), (2048, 'final_prompt'), (2049, 'decode')):
            p, model, runner, batch, records = head_fixture(end, 256 if end <= 2048 else 1)
            logits = self.invoke(p, model)
            self.assertTrue(p.logit_seen)
            self.assertIsNone(p.hidden)
            self.assertEqual([name for name, _ in records], ['head-invocations.jsonl', 'heads.jsonl'])
            raw, receipt = records[0][1], records[1][1]
            self.assertFalse(raw['validated'])
            self.assertEqual(raw['logits'], {'tensor':True,'shape':[1,262144],'rows':1,'dtype':'bf16','device':DEVICE})
            self.assertEqual((receipt['phase'], receipt['logit_position'], receipt['predicts_position']), (phase,end-1,end))
            self.assertEqual(receipt['final_prompt_logit_receipt'], end == 2048)
            self.assertEqual(receipt['expected_sampler_discard'], end < 2048)
            self.assertFalse(receipt['normalized_probabilities'])
            bits = logits.view('uint8').numpy().tobytes()
            self.assertEqual(receipt['logit_bits_sha256'], hashlib.sha256(bits).hexdigest())

    def test_zero_multiple_wrong_width_dtype_device_and_none_reject_with_actual_scalars(self):
        cases = ((0,262144,'bf16',DEVICE), (2,262144,'bf16',DEVICE), (1,262143,'bf16',DEVICE),
                 (1,262144,'fp32',DEVICE), (1,262144,'bf16','wrong-device'))
        for rows, cols, dtype, device in cases:
            p, model, runner, batch, records = head_fixture()
            selected = p.hidden[p.logits_indices]
            logits = Tensor(np.zeros((rows,cols)), dtype, device)
            with self.subTest(rows=rows,cols=cols,dtype=dtype,device=device), self.assertRaisesRegex(RuntimeError,'shape/dtype/device'):
                p.head(model, selected, logits)
            self.assertEqual(records[0][1]['logits']['shape'], [rows,cols])
            self.assertEqual(records[0][1]['logits']['dtype'], dtype)
            self.assertEqual(len(records), 1)
            self.assertFalse(p.logit_seen)
        p, model, _, _, records = head_fixture()
        with self.assertRaisesRegex(RuntimeError,'shape/dtype/device'): p.head(model,p.hidden[p.logits_indices],None)
        self.assertEqual(records[0][1]['logits'], {'tensor':False, 'type':'NoneType'})

    def test_hidden_and_index_type_device_selection_and_bit_mismatch_fail_closed(self):
        for kind in ('hidden-rows', 'hidden-dtype', 'hidden-device', 'index-copy', 'index-dtype',
                     'index-device', 'index-empty', 'index-multiple', 'index-position', 'hidden-bits'):
            p, model, runner, batch, records = head_fixture()
            selected = p.hidden[p.logits_indices]
            logits = CONDITIONAL(model, selected)
            if kind == 'hidden-rows': selected = p.hidden[:0]
            if kind == 'hidden-dtype': selected.dtype = 'fp32'
            if kind == 'hidden-device': selected.device = 'other'
            if kind == 'index-copy': batch.logits_indices = Tensor([255],'int64')
            if kind == 'index-dtype': p.logits_indices.dtype = 'int32'
            if kind == 'index-device': p.logits_indices.device = 'other'
            if kind == 'index-empty': p.logits_indices.values = np.array([])
            if kind == 'index-multiple': p.logits_indices.values = np.array([254,255])
            if kind == 'index-position': p.logits_indices.values = np.array([254])
            if kind == 'hidden-bits': selected.values[0,0] = 3.0
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): p.head(model,selected,logits)
            self.assertEqual(len(records),1)
            self.assertFalse(p.logit_seen)

    def test_exact_mask_accepts_only_negative_infinity_and_no_other_nonfinite(self):
        for setting, dtype in ((None,'bf16'), ('fp32','fp32')):
            for index, value in ((5,0), (5,np.inf), (5,np.nan), (6,-np.inf), (6,np.inf), (6,np.nan)):
                p, model, runner, batch, records = head_fixture(head_dtype=setting)
                selected = p.hidden[p.logits_indices]
                logits = Tensor(np.zeros((1,262144)),dtype)
                logits.values[0,[5,17]] = -np.inf
                logits.values[0,index] = value
                with self.subTest(dtype=dtype,index=index,value=value), self.assertRaises(RuntimeError): p.head(model,selected,logits)
                self.assertEqual(len(records),1)
            p, model, runner, batch, records = head_fixture(head_dtype=setting)
            logits = Tensor(np.zeros((1,262144)),dtype);logits.values[0,[5,17]]=-np.inf
            p.head(model,p.hidden[p.logits_indices],logits)
            self.assertTrue(p.logit_seen)

    def test_signed_zero_native_bytes_are_preserved_and_hashed(self):
        for setting in ('bf16','fp32'):
            p,model,_,_,records=head_fixture(head_dtype=setting,suppressed=())
            logits=Tensor(np.zeros((1,262144)),setting)
            logits.values[0,0]=-0.0
            p.head(model,p.hidden[p.logits_indices],logits)
            bits=logits.view('uint8').numpy().tobytes()
            self.assertEqual(bits[:2] if setting=='bf16' else bits[:4],
                             b'\x00\x80' if setting=='bf16' else b'\x00\x00\x00\x80')
            self.assertEqual(records[-1][1]['logit_bits_sha256'],hashlib.sha256(bits).hexdigest())

    def test_installed_head_wrapper_poisons_and_preserves_primary_on_observer_error(self):
        from test_prefill_native_safety import HookReview
        def check(runner,provider):
            primary=RuntimeError('shape diagnostic primary')
            def rejected(*args): raise primary
            provider.head=rejected
            with self.assertRaises(RuntimeError) as caught: runner.model.compute_logits(object())
            self.assertIs(caught.exception,primary)
            self.assertTrue(provider.failed)
            self.assertIn('abort',provider.log)
            with self.assertRaisesRegex(RuntimeError,'Poisoned'):runner.model.compute_logits(object())
        HookReview().exercise(check)

    def test_duplicate_head_cannot_publish_or_increment_again(self):
        p, model, runner, batch, records = head_fixture()
        self.invoke(p,model)
        with self.assertRaisesRegex(RuntimeError,'unique completed'):p.head(model,None,None)
        self.assertEqual(p.head_counts['intermediate_prompt_chunk'],1)
        self.assertEqual(len(records),2)

    def test_all_eight_chunks_and_255_decode_inputs_preserve_sample_sse_chain(self):
        from prefill_diagnostic_client import StreamLedger, validate_observation
        p, model, runner, batch, records = head_fixture()
        p.ledger = RequestLedger(list(range(2048)))
        p.plan = {'plan_sha256':'a'*64}
        p.scratch = NS(preserve_peaks=lambda: None, receipt=lambda: {})
        ids = {i:('identity',i) for i in range(30)}
        output_type = type('SamplerOutput',(),{'__module__':'vllm.v1.worker.gpu.sample.output'})
        stream = StreamLedger(p.plan)
        request_id = stream.response_id+'-0'
        for start in [*range(0,2048,256), *range(2048,2303)]:
            rows = 256 if start < 2048 else 1
            tokens = list(range(start,start+rows)) if start < 2048 else p.ledger.outputs[-1:]
            p.ledger.begin(tokens,list(range(start,start+rows)), {i:list(range(start+32,start+rows+32)) for i in range(30)},ids,request_id)
            p.ledger.complete()
            p.hidden, p.logit_seen = Tensor(np.zeros((rows,2816))), False
            p.logits_indices, _ = selected_index(start,rows)
            batch.logits_indices = p.logits_indices
            sampler = sampler_fixture(runner,batch,start,rows,tokens)
            # Source runner selection -> actual conditional/language/processor
            # excerpt -> observer -> source sampler/count kernel -> ledger.
            def observed(model, hidden):
                logits = CONDITIONAL(model,hidden)
                p.head(model,hidden,logits)
                return logits
            model.compute_logits = MethodType(observed,model)
            output,count_tensor,rejected_tensor = SAMPLE(runner,p.hidden,batch,None)
            self.assertIs(count_tensor,output.num_sampled)
            self.assertIs(rejected_tensor,output.num_rejected)
            p.sampled(output)
            count = count_tensor.item()
            self.assertEqual(sampler.selection,([start+rows-1],[tokens[-1]]))
            if count:
                index=len(p.ledger.outputs)-1
                stream.consume('data: '+json.dumps({'id':stream.response_id,'choices':[{'index':0,'token_ids':[17], 'finish_reason':'length' if index==255 else None}]}))
        stream.consume('data: '+json.dumps({'id':stream.response_id,'choices':[], 'usage':{'prompt_tokens':2048,'completion_tokens':256,'total_tokens':2304}}))
        stream.consume('data: [DONE]')
        observer = next(value for name,value in records if name=='observer.json')
        validate_observation(p.plan,stream,observer)
        self.assertTrue(p.completed)
        self.assertEqual(observer['head_counts'], {'intermediate_prompt_chunk':7,'final_prompt':1,'decode':255})
        samples=[value for name,value in records if name=='samples.jsonl']
        self.assertEqual((samples[0]['head_sequence'], samples[0]['head_phase'], samples[0]['pending_anchor_position']), (7,'final_prompt',2048))
        self.assertEqual((samples[-1]['head_sequence'],samples[-1]['pending_anchor_position']), (262,2303))
        self.assertFalse(observer['numerical_qualified']);self.assertFalse(observer['performance_qualified'])


if __name__ == '__main__': unittest.main()

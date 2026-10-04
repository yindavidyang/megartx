"""Persisted transport evidence gates fit; writes are intercepted in CPU tests."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from megartx.prefill_diagnostic_plan import (BOUNDS, INSTALLED, TRANSPORT_FILES, RUNNER_POLICY,
                                          publish_fit)
from megartx.prefill_runner_binding import HOOKS
from test_prefill_ledger_matching import PLAN, source_transport, validate_observation, stream_observation


class PersistedStreamAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stream,observer,*_=source_transport()
        cls.plan={**PLAN,'source_head':'9'*40,'runner_policy':RUNNER_POLICY,'bounds':BOUNDS}
        cls.client=validate_observation(cls.plan,stream,observer)
        cls.stream=stream_observation(cls.plan,stream)
        cls.observer=copy.deepcopy(observer)
        cls.observer['observer_gpu_scratch']={
            'domain':'incremental_gpu_allocator_bytes','cap_bytes':BOUNDS['max_observer_gpu_scratch_bytes'],
            'measured_phases':789,'managed_tensor_simultaneous_peak_bytes':0,
            'measured_phase_allocator_increment_peak_bytes':0,'measured_phase_reserved_increment_peak_bytes':0,
            'counter_policy':'process_global_resets_with_explicit_runwide_peak_preservation',
            'runwide_allocator_allocated_peak_bytes':0,'runwide_allocator_reserved_peak_bytes':0,
            'host_heap_excluded':True}
        cls.binding={'schema':'megartx-prefill-runner-binding-v1','plan_sha256':cls.plan['plan_sha256'],
            'source_head':cls.plan['source_head'],'owner_pid':1,'owner_start_ticks':1,
            'runner_policy':RUNNER_POLICY,'installed_sources':{**INSTALLED,**TRANSPORT_FILES},
            'hook_bindings':dict.fromkeys(HOOKS,True),'mutable_lease_granted':False}
        cls.loaded={'plan_sha256':cls.plan['plan_sha256'],'mutable_lease_granted':False,
                    'identity':{'owner_pid':1,'owner_start_ticks':1}}
        cls.geometry={'physical_policy':'full_context','capacity_tokens':2304,'actual_owned_page_ranges_disjoint':True}
        cls.ownership={'cleanup_complete':True,'failure':None,'owned_identities_remaining':[],
                       'owned_gpu_pids_remaining':[],'cleanup_errors':[]}

    def invoke(self, stream, client=None, mode=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,value in [('runner-binding.json',self.binding),('observer.json',self.observer),
                ('client.json',self.client if client is None else client),('loaded.json',self.loaded),('geometry.json',self.geometry)]:
                (root/name).write_text(json.dumps(value))
            path=root/'client-stream.json'
            if mode=='symlink':
                (root/'other.json').write_text(json.dumps(stream));path.symlink_to(root/'other.json')
            elif mode=='oversized':path.write_bytes(b' ' * 65537)
            elif mode=='invalid-json':path.write_text('{')
            elif mode!='missing':path.write_text(json.dumps(stream))
            writes=[]
            evidence=NS(directory=root,write=lambda name,value:writes.append((name,value)))
            try:publish_fit(evidence,self.plan,self.ownership)
            except (ValueError,TypeError) as error:
                self.assertEqual(writes,[])
                self.assertFalse((root/'fit.json').exists())
                return error
            self.assertFalse((root/'fit.json').exists())
            return writes

    def test_matching_source_driven_fixture_reaches_only_intercepted_publication(self):
        writes=self.invoke(self.stream)
        self.assertIsInstance(writes,list)
        self.assertEqual(len(writes),1)
        self.assertEqual(writes[0][0],'fit.json')
        self.assertIn('client-stream.json',writes[0][1]['evidence_files_sha256'])
        self.assertIs(writes[0][1]['numerical_qualified'],False)
        self.assertIs(writes[0][1]['performance_qualified'],False)

    def test_missing_symlink_oversized_invalid_and_nonrecord_streams_rejected(self):
        for mode in ('missing','symlink','oversized','invalid-json'):
            with self.subTest(mode=mode):self.assertIsInstance(self.invoke(self.stream,mode=mode),ValueError)
        for record in (None,False,0,[],{},'complete'):
            with self.subTest(record_type=type(record).__name__):self.assertIsInstance(self.invoke(record),ValueError)

    def test_every_persisted_field_is_mandatory(self):
        for key in self.stream:
            record={k:v for k,v in self.stream.items() if k!=key}
            with self.subTest(field=key):self.assertIsInstance(self.invoke(record),ValueError)
        for key in self.stream['usage']:
            record=copy.deepcopy(self.stream);del record['usage'][key]
            with self.subTest(usage_field=key):self.assertIsInstance(self.invoke(record),ValueError)

    def test_stale_or_failed_stream_is_diagnostics_only_despite_complete_client(self):
        changes=[('schema','unknown'),('plan_sha256','0'*64),('response_id_sha256','0'*64),
            ('output_ids_sha256','0'*64),('emitted_outputs',0),('stream_done',False),
            ('finish_reason',None),('finish_reason','stop'),('usage',None),('sse_events',0),
            ('sse_events',self.stream['sse_events']-1),('sse_events',301),
            ('numerical_qualified',True),('performance_qualified',True),('status','complete')]
        for key,value in changes:
            with self.subTest(field=key,value=value):
                self.assertIsInstance(self.invoke({**self.stream,key:value}),ValueError)
        # Exact contradiction replay from independent review; no trusted complete
        # client or native observation may replace this missing transport proof.
        record={'schema':'megartx-prefill-client-stream-v1','plan_sha256':'0'*64,
            'stream_done':False,'emitted_outputs':0,'usage':None,'output_ids_sha256':'0'*64}
        self.assertIsInstance(self.invoke(record),ValueError)

    def test_exact_types_reject_truthy_and_float_counts_at_every_level(self):
        for key,value in [('stream_done',1),('emitted_outputs',256.0),('sse_events',float(self.stream['sse_events'])),
                          ('numerical_qualified',0),('performance_qualified',0)]:
            with self.subTest(field=key):self.assertIsInstance(self.invoke({**self.stream,key:value}),ValueError)
        for key,count in self.stream['usage'].items():
            for value in (count-1,float(count),True):
                record=copy.deepcopy(self.stream);record['usage'][key]=value
                with self.subTest(usage_field=key,value=value):self.assertIsInstance(self.invoke(record),ValueError)
        record=copy.deepcopy(self.stream);record['usage']['unexpected']=0
        self.assertIsInstance(self.invoke(record),ValueError)

    def test_client_stream_agreement_cannot_replace_native_or_fixed_contract(self):
        for key,value in [('output_ids_sha256','0'*64),('stream_done',False),('emitted_outputs',0),
                          ('usage',None),('sse_events',301),('plan_sha256','0'*64)]:
            client={**self.client,key:value};record={**self.stream,key:value}
            with self.subTest(field=key):self.assertIsInstance(self.invoke(record,client),ValueError)


if __name__=='__main__':unittest.main()

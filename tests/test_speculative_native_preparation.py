"""CPU fixture preparation controls; fake bytes never count as target rehash."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from megartx import speculative_native_plan as plan
from megartx import speculative_native_preparation as prep
from megartx import speculative_native_v2_plan as v2
from megartx.speculative_native_probe import ProbeError, REVISION

ROOT = Path(__file__).resolve().parents[1]


class CheckpointPreparationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'; self.model.mkdir()
        public = plan.checkpoint_contract()
        self.contract = copy.deepcopy(public)
        self.index = {'metadata': {'total_size': 10}, 'weight_map': {'a': 'model-00001-of-00002.safetensors',
                     'b': 'model-00002-of-00002.safetensors'}}
        self.contract['index'].update(weight_count=2, metadata=self.index['metadata'],
            shard_weight_counts={value: 1 for value in self.index['weight_map'].values()})
        for row in self.contract['files']:
            content = json.dumps(self.index).encode() if row['name'] == self.contract['index']['name'] else ('fixture '+row['name']).encode()
            (self.model / row['name']).write_bytes(content)
            row['bytes'], row['sha256'] = len(content), hashlib.sha256(content).hexdigest()
        self.manifest = {'repo_id': public['repo_id'], 'revision': REVISION, 'gated': False,
            'total_bytes': sum(row['bytes'] for row in self.contract['files']),
            'files': [{k: row[k] for k in ('name', 'bytes', 'sha256')} | {'verified_upstream_lfs_sha256': row['name'].endswith('.safetensors')}
                      for row in self.contract['files']]}
        self.manifest_path = self.root / 'manifest.json'
        self.save()
        self.patcher = patch.object(plan, 'checkpoint_contract', return_value=self.contract)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def save(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def run_preflight(self):
        return plan.checkpoint_preflight(self.manifest_path, self.model)

    def test_all_selected_files_and_exact_index_shards_rehashed(self):
        with patch.object(plan, 'hash_file', wraps=plan.hash_file) as hashed:
            row = self.run_preflight()
        hashed_paths = [Path(args.args[0]) for args in hashed.call_args_list]
        self.assertTrue(all(self.model / item['name'] in hashed_paths for item in self.manifest['files']))
        self.assertEqual(len(row['files']), 11)
        self.assertEqual(len(row['shard_names']), 2)
        self.assertIs(row['full_files_rehashed'], True)
        self.assertIs(row['full_shards_rehashed'], True)
        self.assertEqual(row['verification'], 'fresh_full_streaming_hash')

    def test_unrelated_non_file_children_do_not_add_a_host_layout_requirement(self):
        (self.model / '.cache').mkdir()
        (self.model / 'consolidated-directory.safetensors').mkdir()
        (self.model / '.cache' / 'download-metadata.json').write_text('{}')
        (self.model / 'unused.safetensors').write_bytes(b'index-filtered unrelated file')
        self.assertTrue(self.run_preflight()['full_files_rehashed'])

    def test_alternate_auto_loader_input_rejected_including_nested_and_symlink(self):
        for name in ('consolidated.safetensors','.cache/consolidated-00001.safetensors'):
            path=self.model/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'unbound alternate weights')
            with self.assertRaisesRegex(ProbeError,'changes auto loader'): self.run_preflight()
            path.unlink()
        path=self.model/'consolidated.safetensors';path.symlink_to(self.model/'model-00001-of-00002.safetensors')
        with self.assertRaisesRegex(ProbeError,'changes auto loader'): self.run_preflight()

    def test_alternate_loader_input_added_during_hash_is_rejected(self):
        original=plan.hash_file
        def add_file(path,*args):
            result=original(path,*args)
            if str(path).endswith('model-00002-of-00002.safetensors'):
                (self.model/'consolidated.safetensors').write_bytes(b'unbound')
            return result
        with patch.object(plan,'hash_file',side_effect=add_file):
            with self.assertRaisesRegex(ProbeError,'changes auto loader'): self.run_preflight()

    def test_missing_extra_duplicate_manifest_rows_rejected_before_payload_hash(self):
        originals = copy.deepcopy(self.manifest['files'])
        for rows in (originals[:-1], originals+[dict(originals[0], name='extra.json')], originals+[originals[0]]):
            self.manifest['files'] = rows; self.save()
            with patch.object(plan, 'hash_file', wraps=plan.hash_file) as hashed:
                with self.assertRaises(ProbeError): self.run_preflight()
            self.assertTrue(all(Path(call.args[0]) == self.manifest_path for call in hashed.call_args_list))

    def test_path_escape_and_non_object_manifest_row_rejected(self):
        for value in ('../config.json', '/tmp/config.json', '.', '..', None):
            self.manifest['files'][0]['name'] = value; self.save()
            with self.assertRaises(ProbeError): self.run_preflight()
        self.manifest['files'][0] = None; self.save()
        with self.assertRaises(ProbeError): self.run_preflight()

    def test_stale_digest_size_total_and_unverified_shard_rejected(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (('sha256', '0'*64), ('bytes', True), ('bytes', 9999)):
            self.manifest = copy.deepcopy(original); self.manifest['files'][0][key] = value; self.save()
            with self.assertRaises(ProbeError): self.run_preflight()
        self.manifest = copy.deepcopy(original); self.manifest['total_bytes'] += 1; self.save()
        with self.assertRaises(ProbeError): self.run_preflight()
        self.manifest = copy.deepcopy(original)
        next(row for row in self.manifest['files'] if row['name'].endswith('.safetensors'))['verified_upstream_lfs_sha256'] = 1
        self.save()
        with self.assertRaises(ProbeError): self.run_preflight()

    def test_missing_file_symlink_and_same_length_wrong_bytes_rejected(self):
        name = self.manifest['files'][0]['name']; path = self.model/name; original = path.read_bytes()
        path.unlink()
        with self.assertRaises((ProbeError, FileNotFoundError)): self.run_preflight()
        outside = self.root/'outside'; outside.write_bytes(original); path.symlink_to(outside)
        with self.assertRaises(ProbeError): self.run_preflight()
        path.unlink(); path.write_bytes(b'x'*len(original))
        with self.assertRaises(ProbeError): self.run_preflight()

    def test_retained_stats_do_not_bypass_streaming_hash(self):
        original = plan.hash_file
        def fail_on_weight(path, *args):
            if str(path).endswith('.safetensors'): raise OSError('fixture read failure')
            return original(path, *args)
        with patch.object(plan, 'hash_file', side_effect=fail_on_weight):
            with self.assertRaises(OSError): self.run_preflight()

    def test_index_missing_extra_or_escaped_shard_rejected_even_in_synthetic_pin(self):
        for values in ({'a':'model-00001-of-00002.safetensors'},
                       {'a':'model-00001-of-00002.safetensors', 'b':'../outside.safetensors'},
                       {'a':'model-00001-of-00002.safetensors', 'b':'tokenizer.json'}):
            changed = dict(self.index, weight_map=values)
            path = self.model/self.contract['index']['name']; path.write_text(json.dumps(changed))
            for rows in (self.contract['files'], self.manifest['files']):
                row = next(row for row in rows if row['name']==path.name)
                row['bytes'], row['sha256'] = path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()
            self.manifest['total_bytes'] = sum(row['bytes'] for row in self.manifest['files']); self.save()
            with self.assertRaises(ProbeError): self.run_preflight()

    def test_duplicate_index_key_rejected(self):
        path = self.model/self.contract['index']['name']
        path.write_text('{"metadata":{},"weight_map":{"a":"first","a":"second"}}')
        for rows in (self.contract['files'], self.manifest['files']):
            row=next(row for row in rows if row['name']==path.name)
            row['bytes'],row['sha256']=path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest['total_bytes']=sum(row['bytes'] for row in self.manifest['files']);self.save()
        with self.assertRaises(ProbeError): self.run_preflight()

    def test_file_changed_after_hash_rejected(self):
        original = plan.hash_file; first = self.model/self.manifest['files'][0]['name']
        def mutate(path,*args):
            result = original(path,*args)
            if Path(path) == first: first.write_bytes(first.read_bytes())
            return result
        with patch.object(plan,'hash_file',side_effect=mutate):
            with self.assertRaisesRegex(ProbeError,'changed'): self.run_preflight()

    def test_real_public_contract_has_upstream_provenance_and_two_shards(self):
        contract=json.loads((ROOT/plan.CHECKPOINT_CONTRACT).read_text())
        self.assertEqual({row['name'] for row in contract['omitted_repository_files']},{'.gitattributes'})
        self.assertEqual(sum(row['bytes'] for row in contract['files']),18825678190)
        self.assertEqual(contract['index']['weight_count'],47033)
        self.assertEqual(contract['index']['sha256'],'ac5e677ed9f8d9b689170bbae0c88ce163e284d02b09100e14afddaa9ec4a15c')
        self.assertEqual(set(contract['index']['shard_weight_counts']),{row['name'] for row in contract['files'] if row['name'].endswith('.safetensors')})
        self.assertTrue(all(REVISION in row['source_url'] for row in contract['files']))


class RuntimePreparationControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ds-'); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.project=self.base/'p'; self.site=self.base/'s'; self.runtime=self.base/'r'
        (self.project/'src/megartx').mkdir(parents=True); (self.project/'src/megartx/__init__.py').write_text('# fixture\n')
        (self.project/'pyproject.toml').write_bytes((ROOT/'pyproject.toml').read_bytes())
        self.site.mkdir(); shutil.copytree(self.project/'src/megartx',self.site/'megartx')
        self.dist=self.site/'megartx-0.0.1.dist-info';self.dist.mkdir()
        for name,text in {'METADATA':'Metadata-Version: 2.1\nName: megartx\nVersion: 0.0.1\n',
                          'WHEEL':'Wheel-Version: 1.0\n','RECORD':'fixture metadata only\n',
                          'entry_points.txt':'[vllm.general_plugins]\nmegartx_scale_adapter = megartx.vllm_scale_plugin:install\n'}.items():
            (self.dist/name).write_text(text)
        self.binding=prep.runtime_binding(self.project,self.runtime,self.site)
        self.private=self.base/'evidence'

    def test_plan_bound_paths_no_shared_cache_fallback(self):
        env=v2.environment(self.project,self.private,runtime=self.binding)
        for key in ('HOME','XDG_CACHE_HOME','TMPDIR','TMP','TEMP','VLLM_RPC_BASE_PATH','HF_HOME',
                    'VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT','FLASHINFER_WORKSPACE_BASE','TRITON_CACHE_DIR',
                    'CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR'):
            self.assertTrue(Path(env[key]).is_relative_to(self.runtime),key)
        self.assertIn(str(self.site),env['PYTHONPATH'].split(':'))
        self.assertFalse(self.runtime.exists())
        self.assertFalse(self.binding['shared_cache_fallback'])

    def test_runtime_created_exclusively_owned_private_and_no_reuse(self):
        prep.create_runtime(self.binding,self.private)
        for path in self.binding['paths'].values(): self.assertEqual(Path(path).stat().st_mode & 0o777,0o700)
        prep.runtime_preflight(self.binding,self.private,created=True)
        with self.assertRaisesRegex(ProbeError,'reuse forbidden'): prep.create_runtime(self.binding,self.private)

    def test_shared_cache_root_and_path_escape_rejected(self):
        for root in (plan.BASE/'cache/private',plan.BASE,Path('/tmp/../tmp/test'),self.project/'runtime',self.site/'runtime'):
            with self.assertRaises(ProbeError): prep.runtime_binding(self.project,root,self.site)
        changed=copy.deepcopy(self.binding); changed['paths']['cache']=str(plan.BASE/'cache')
        with self.assertRaises(ProbeError): prep.runtime_preflight(changed,self.private)
        with self.assertRaises(ProbeError): prep.runtime_preflight(self.binding,self.runtime/'evidence')

    def test_evidence_cannot_write_baseline_or_source_tree(self):
        for private in (plan.BASE/'results/new-receipt',self.project/'results',self.project,plan.BASE):
            with self.assertRaisesRegex(ProbeError,'overlap'):
                prep.runtime_preflight(self.binding,private)

    def test_source_only_V2_freeze_blocks_baseline_evidence_before_writer(self):
        spec=importlib.util.spec_from_file_location('preparation_freeze_fixture',ROOT/'scripts/speculative_native_receipt_preflight.py')
        client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
        with patch.object(client,'freeze',return_value={'runtime_binding':None}),patch.object(client,'PrivateEvidence') as writer:
            with self.assertRaisesRegex(ProbeError,'overlap'):
                client.main(['--freeze','--runner-lane','v2','--private-directory',str(plan.BASE/'results')])
            writer.assert_not_called()

    def test_stale_wheel_metadata_and_source_rejected(self):
        (self.dist/'WHEEL').write_text('changed\n')
        with self.assertRaisesRegex(ProbeError,'metadata changed'): prep.runtime_preflight(self.binding,self.private)
        (self.site/'megartx/__init__.py').write_text('# stale\n')
        with self.assertRaisesRegex(ProbeError,'source differs'): prep.runtime_binding(self.project,self.runtime,self.site)

    def test_wrong_duplicate_missing_entrypoint_metadata_rejected(self):
        path=self.dist/'entry_points.txt'
        for text in ('[vllm.general_plugins]\nother = wrong:install\n',
                     '[vllm.general_plugins]\nmegartx_scale_adapter = wrong:install\n',
                     '[vllm.general_plugins]\nmegartx_scale_adapter = a:b\nmegartx_scale_adapter = a:b\n'):
            path.write_text(text)
            with self.assertRaises(ProbeError): prep.runtime_binding(self.project,self.runtime,self.site)
        path.unlink()
        with self.assertRaises(ProbeError): prep.runtime_binding(self.project,self.runtime,self.site)

    def test_search_path_injection_and_symlink_rejected(self):
        injected=self.site/'bad.pth';injected.write_text('/shared\n')
        with self.assertRaises(ProbeError): prep.runtime_binding(self.project,self.runtime,self.site)
        injected.unlink(); other=self.base/'alias';other.symlink_to(self.site,target_is_directory=True)
        with self.assertRaises(ProbeError): prep.runtime_binding(self.project,self.runtime,other)
        path=self.dist/'RECORD';path.unlink();path.symlink_to(self.dist/'WHEEL')
        with self.assertRaises(ProbeError): prep.runtime_binding(self.project,self.runtime,self.site)

    def test_existing_or_symlink_runtime_root_never_adopted(self):
        self.runtime.mkdir()
        with self.assertRaises(ProbeError): prep.runtime_preflight(self.binding,self.private)
        self.runtime.rmdir();self.runtime.symlink_to(self.site,target_is_directory=True)
        with self.assertRaises(ProbeError): prep.runtime_preflight(self.binding,self.private)

    def test_generated_source_egg_info_cannot_shadow_frozen_wheel_metadata(self):
        shadow=self.project/'src/megartx.egg-info';shadow.mkdir()
        (shadow/'PKG-INFO').write_bytes((self.dist/'METADATA').read_bytes())
        (shadow/'entry_points.txt').write_bytes((self.dist/'entry_points.txt').read_bytes())
        with self.assertRaisesRegex(ProbeError,'shadowed'):
            prep.runtime_binding(self.project,self.runtime,self.site)

    def test_conflicting_plugin_from_other_installed_distribution_rejected(self):
        installed=self.base/'installed'; installed.mkdir()
        other=installed/'other-0.0.1.dist-info';other.mkdir()
        (other/'METADATA').write_text('Name: other\nVersion: 0.0.1\n')
        (other/'entry_points.txt').write_bytes((self.dist/'entry_points.txt').read_bytes())
        with self.assertRaisesRegex(ProbeError,'duplicated'):
            prep.runtime_preflight(self.binding,self.private,installed_root=installed)

    def test_ipc_budget_counts_UTF8_suffix_and_meaningful_margin(self):
        maxroot=prep.IPC_PATH_LIMIT_BYTES-prep.IPC_UUID_SUFFIX_BYTES-prep.IPC_MARGIN_BYTES
        self.assertEqual(prep.ipc_preflight('/'+'x'*(maxroot-1))['root_utf8_bytes'],maxroot)
        for path in ('/'+'x'*maxroot, '/'+'x'*77, '/'+'é'*(maxroot//2)):
            with self.assertRaisesRegex(ProbeError,'UTF-8 socket budget'): prep.ipc_preflight(path)
        self.assertEqual(prep.IPC_UUID_SUFFIX_BYTES,37)
        self.assertGreaterEqual(prep.IPC_MARGIN_BYTES,16)

    def test_unconfigured_V2_and_legacy_cross_lane_fail_closed(self):
        with self.assertRaises(ProbeError): v2.environment(self.project,self.private)
        with self.assertRaises(ProbeError): plan.environment(self.project,self.private,runtime=self.binding)
        changed=copy.deepcopy(self.binding);changed['project_root']=str(ROOT)
        with self.assertRaises(ProbeError): v2.environment(self.project,self.private,runtime=changed)

    def test_supervisor_allowlist_and_child_reject_independent_cache_fallback(self):
        spec=importlib.util.spec_from_file_location('preparation_client_fixture',ROOT/'scripts/speculative_native_receipt_client.py')
        client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
        frozen={'runner_lane':'v2','runtime_binding':self.binding}
        inherited={'HOME':str(plan.BASE),'VLLM_MEDIA_CACHE':'/shared','VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR':'/shared',
                   'HF_HUB_CACHE':'/shared','PYTHONPATH':'/shared','LD_PRELOAD':'/shared/library.so'}
        with patch.object(client,'PROJECT',self.project):
            env=client.child_environment(frozen,self.private,inherited)
            client.validate_child_environment(frozen,self.private,env)
            for key in ('VLLM_MEDIA_CACHE','VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR','HF_HUB_CACHE','LD_PRELOAD'):
                self.assertNotIn(key,env)
                with self.assertRaises(ProbeError): client.validate_child_environment(frozen,self.private,{**env,key:'/shared'})
            with self.assertRaises(ProbeError): client.validate_child_environment(frozen,self.private,{**env,'TMPDIR':str(plan.BASE/'tmp')})


if __name__=='__main__': unittest.main()

"""Shared CPU selection/registration/delegation controls; no native fit claim."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from megartx import native_diagnostic_composition as composition
from megartx import speculative_native_plan as legacy
from megartx import speculative_native_v2_plan as v2
from megartx import vllm_scale_plugin as plugin

ROOT = Path(__file__).resolve().parents[1]


def env(lane):
    result = {"MEGARTX_SCALE_MODE": "native", "VLLM_PLUGINS": "megartx_scale_adapter"}
    flags = {"v2": composition.V2_FLAGS, "v1-legacy": composition.V1_FLAGS,
             "prefill": composition.PREFILL_FLAGS, "native-m1": ("MEGARTX_M1_PREPARATION",)}[lane]
    result.update({key: "stock" if lane == "native-m1" else "1" for key in flags})
    return result


def args():
    return NS(client="native-v2-receipt", label="cpu", mode="native", trials=1,
        backend="flashinfer_cutlass", kv="bfloat16", prefill_chunk=256,
        m1_execution="captured", layer0_capture_policy="synchronous",
        native_receipt_plan=Path("plan.json"), native_receipt_authorization=Path("auth.json"),
        native_receipt_directory=Path("private"), native_receipt_checkpoint_manifest=None,
        **{name: None for name in ("profile", "m1_decode_profile", "prefill_native_plan",
        "prefill_native_clearance", "m1_eager_benchmark_plan", "m1_private_aot", "m1_timing_metadata_help",
        "controlled_plan", "controlled_path", "layer0_boundaries", "activation_only", "routing_diagnostic",
        "router_score_only", "router_prefix_manifest", "m1_preparation", "m1_bridge", "m1_build_receipt",
        "m1_route_controls", "m1_normal_plan", "m1_external_observer")})


class SelectionControls(unittest.TestCase):
    def test_default_off_and_each_exact_mode(self):
        self.assertIsNone(composition.diagnostic_mode({}))
        for lane in ("v2", "v1-legacy", "prefill", "native-m1"):
            self.assertEqual(composition.diagnostic_mode(env(lane)), lane)
        with patch.dict(os.environ, {}, clear=True), patch.object(plugin.importlib.metadata, "version",
                side_effect=AssertionError("off path inspected runtime")):
            plugin.install()

    def test_all_pairs_and_partial_opt_ins_are_rejected(self):
        lanes = ("v2", "v1-legacy", "prefill", "native-m1")
        for i, left in enumerate(lanes):
            for right in lanes[i+1:]:
                with self.subTest(left=left, right=right), self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
                    composition.diagnostic_mode({**env(left), **env(right)})
        for lane in lanes[:3]:
            for key in (composition.V2_FLAGS if lane == 'v2' else composition.V1_FLAGS if lane == 'v1-legacy' else composition.PREFILL_FLAGS):
                bad = env(lane); del bad[key]
                with self.assertRaisesRegex(RuntimeError, "Partial"):
                    composition.diagnostic_mode(bad)
        for value in ("0", "true", "", 1, True):
            bad = env('v2'); bad[composition.V2_FLAGS[0]] = value
            with self.assertRaises(RuntimeError): composition.diagnostic_mode(bad)

    def test_unknown_overrides_unrelated_hooks_fail_before_runtime(self):
        for changed in ({"MEGARTX_NATIVE_NEW_HOOK": "1"}, {"MEGARTX_PREFILL_NATIVE_NEW_HOOK": "1"},
                        {"VLLM_USE_V2_MODEL_RUNNER": "1"}, {"MEGARTX_LOGITS_DIR": "/tmp/x"},
                        {"MEGARTX_M1_BRIDGE": "/tmp/x"}, {"MEGARTX_SCALE_MODE": "reference"},
                        {"VLLM_PLUGINS": "other"}):
            with patch.dict(os.environ, {**env('v2'), **changed}, clear=True), patch.object(
                    plugin.importlib.metadata, 'version', side_effect=AssertionError('runtime acquired')):
                with self.assertRaises(RuntimeError): plugin.install()

    def test_pair_registration_order_and_failure_short_circuit(self):
        for lane, module, install, evidence in (
                ('v2', 'megartx.speculative_native_v2_lifecycle', 'install_native_v2_diagnostic', 'install_native_v2_receipt_evidence'),
                ('v1-legacy', 'megartx.speculative_native_lifecycle', 'install_native_diagnostic', 'install_native_receipt_evidence')):
            calls=[]
            with patch(module+'.'+install, side_effect=lambda: calls.append('lifecycle') or True), patch(
                    'megartx.speculative_native_evidence.'+evidence, side_effect=lambda: calls.append('evidence') or True):
                composition.install_diagnostic_hooks(lane)
            self.assertEqual(calls, ['lifecycle','evidence'])
            with patch(module+'.'+install, return_value=False), patch('megartx.speculative_native_evidence.'+evidence) as writer:
                with self.assertRaises(RuntimeError): composition.install_diagnostic_hooks(lane)
                writer.assert_not_called()
        for lane in (None,'prefill','native-m1'):
            with patch('megartx.speculative_native_v2_lifecycle.install_native_v2_diagnostic') as install:
                composition.install_diagnostic_hooks(lane); install.assert_not_called()
        with self.assertRaises(RuntimeError): composition.install_diagnostic_hooks('unknown')

    def test_hooks_run_before_adapter_idempotence_return(self):
        fake = {}
        for name, attribute, value in (
                ('torch', None, None),
                ('vllm.model_executor.layers.quantization.modelopt','ModelOptNvFp4FusedMoE',NS(_megartx_installed=True)),
                ('vllm.model_executor.layers.fused_moe.routed_experts','RoutedExperts',object),
                ('vllm.model_executor.layers.fused_moe.experts','flashinfer_cutlass_moe',object)):
            fake[name]=ModuleType(name)
            if attribute: setattr(fake[name],attribute,value)
        calls=[]
        with patch.dict(os.environ,env('v2'),clear=True), patch.dict(sys.modules,fake), patch.object(
                plugin.importlib.metadata,'version',side_effect={'vllm':'0.30.0','torch':'2.13.0','flashinfer-python':'0.6.18.post1'}.__getitem__), patch(
                'megartx.speculative_native_v2_lifecycle.install_native_v2_diagnostic',side_effect=lambda:calls.append('lifecycle') or True), patch(
                'megartx.speculative_native_evidence.install_native_v2_receipt_evidence',side_effect=lambda:calls.append('evidence') or True):
            plugin.install()
        self.assertEqual(calls,['lifecycle','evidence'])

    def test_unknown_runner_rejected_before_engine_client_acquisition(self):
        from megartx.speculative_native_client import make_actual_client
        from test_speculative_native_v2_client import config
        for value in (False, None, 1, "True"):
            configured=config();configured.use_v2_model_runner=value
            fake={}
            for name in ('vllm.engine.arg_utils','vllm.v1.executor.abstract','vllm.v1.engine.core_client'):
                fake[name]=ModuleType(name)
            fake['vllm.engine.arg_utils'].EngineArgs=lambda **kwargs:NS(create_engine_config=lambda:configured)
            fake['vllm.v1.executor.abstract'].Executor=NS(get_class=lambda value:None)
            def acquire(**kwargs):raise AssertionError('engine acquired with unknown runner')
            fake['vllm.v1.engine.core_client'].EngineCoreClient=NS(make_client=acquire)
            fake['vllm.v1.engine.core_client'].SyncMPClient=object
            fake['vllm.v1.engine.core_client'].AsyncMPClient=object
            with patch.dict(sys.modules,fake), patch.dict(os.environ,{},clear=True):
                with self.assertRaisesRegex(RuntimeError,'default V2 configuration'):
                    make_actual_client({'runner_lane':'v2','engine_kwargs':v2.engine_kwargs(),'client_mode':'async'})

    def test_cpu_imports_have_no_model_or_runtime_imports(self):
        command="import sys; import megartx.native_diagnostic_composition, megartx.vllm_scale_plugin; assert not any(x.split('.')[0] in ('torch','vllm','flashinfer','transformers') for x in sys.modules)"
        subprocess.run([sys.executable,'-S','-c',command],env={**os.environ,'PYTHONPATH':str(ROOT/'src')},check=True,capture_output=True)


class LauncherControls(unittest.TestCase):
    def test_exact_purpose_delegates_original_owned_client_argv(self):
        candidate=args(); frozen={'schema':v2.SCHEMA}; auth={'fixture':True}
        with patch.object(legacy,'read_json',side_effect=[frozen,auth]), patch.object(legacy,'validate_plan',return_value=frozen) as validate, patch.object(legacy,'validate_authorization') as authorization:
            result=composition.receipt_client_argv(candidate,ROOT)
        validate.assert_called_once_with(frozen,ROOT);authorization.assert_called_once_with(auth,frozen)
        self.assertEqual(result,['--execute','--plan',str(candidate.native_receipt_plan.resolve()),
            '--authorization',str(candidate.native_receipt_authorization.resolve()),'--private-directory',str(candidate.native_receipt_directory.resolve())])
        self.assertNotIn('--owned-child',result)
        with patch.object(legacy,'read_json',return_value={'schema':legacy.SCHEMA}), patch.object(legacy,'validate_plan',return_value={'schema':legacy.SCHEMA}), patch.object(legacy,'validate_authorization') as auth:
            with self.assertRaisesRegex(RuntimeError,'runner lane differ'): composition.receipt_client_argv(candidate,ROOT)
            auth.assert_not_called()

    def test_mixed_cli_or_missing_paths_stop_before_plan_read(self):
        for name,value in (('prefill_native_plan',Path('/x')),('m1_preparation','fused'),('profile',True),
                ('controlled_plan',Path('/x')),('trials',2),('m1_execution','capture-free'),
                ('native_receipt_authorization',None),('mode','reference')):
            candidate=args();setattr(candidate,name,value)
            with patch.object(legacy,'read_json',side_effect=AssertionError('plan read too early')):
                with self.assertRaisesRegex(RuntimeError,'default-off'): composition.receipt_client_argv(candidate,ROOT)
        candidate=args();candidate.client='prefill-native'
        with self.assertRaisesRegex(RuntimeError,'Receipt arguments require'): composition.receipt_client_argv(candidate,ROOT)

    def test_inherited_hooks_never_silently_choose_launcher_mode(self):
        for key in (*composition.V1_FLAGS,*composition.V2_FLAGS,*composition.PREFILL_FLAGS,'MEGARTX_NATIVE_UNKNOWN'):
            with self.assertRaisesRegex(RuntimeError,'inherited diagnostic hooks'):
                composition.validate_launcher_environment({key:'1'})

    def test_receipt_launcher_rejects_inherited_m1_and_runner_override(self):
        for key in ("MEGARTX_M1_BRIDGE", "MEGARTX_M1_PREPARATION", "VLLM_USE_V2_MODEL_RUNNER", "MEGARTX_LOGITS_DIR"):
            with self.assertRaisesRegex(RuntimeError,"inherited runner overrides"):
                composition.validate_launcher_environment({key:"1"},"native-v2-receipt")

    def test_missing_v2_receipt_plan_fails_before_http_or_resources(self):
        result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_scale_validation.py'),
            '--label','cpu','--mode','native','--client','native-v2-receipt','--trials','1'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('exact default-off',result.stderr)
        self.assertNotIn('requests',result.stderr);self.assertNotIn('MEGARTX_BASE',result.stderr)

    def test_launcher_delegation_precedes_http_and_server_setup(self):
        source=(ROOT/'scripts/run_scale_validation.py').read_text()
        self.assertLess(source.index('raise SystemExit(receipt_main(receipt_argv))'),source.index('import requests'))
        self.assertLess(source.index('raise SystemExit(receipt_main(receipt_argv))'),source.index('output.mkdir'))
        self.assertIn('diagnostic_mode(os.environ)',(ROOT/'scripts/speculative_native_receipt_client.py').read_text())

    def test_protocol_schema_environment_worker_argv_agree(self):
        protocol=json.loads((ROOT/composition.PROTOCOL).read_text())
        self.assertEqual(protocol['v2'],composition.V2_COMPOSITION)
        self.assertEqual(v2.engine_kwargs()['worker_extension_cls'],composition.V2_COMPOSITION['worker_extension_cls'])
        argv=v2.engine_argv();self.assertEqual(argv[argv.index('--worker-extension-cls')+1],composition.V2_COMPOSITION['worker_extension_cls'])
        self.assertEqual(composition.diagnostic_mode(v2.environment(ROOT,'/tmp/fixture')),'v2')
        self.assertFalse(protocol['fit_admitted']);self.assertFalse(protocol['target_probe_authorized'])
        frozen={'schema':v2.SCHEMA,'purpose':v2.PURPOSE}
        self.assertIs(legacy.plan_api(frozen),v2)


if __name__=='__main__':unittest.main()

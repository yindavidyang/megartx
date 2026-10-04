"""CPU client/admission fault controls. Fixtures never establish native fit."""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from megartx import speculative_native_plan as legacy
from megartx import speculative_native_v2_plan as plan
from megartx import speculative_native_v2 as owner
from megartx import speculative_native_v2_lifecycle as lifecycle
from megartx import speculative_native_evidence as evidence
from megartx.speculative_native_client import ReceiptSession, validate_actual_config, State, EXPECTED_RELEASE
from megartx.speculative_native_compare import (validate_scalar_receipt, verify_private_receipt,
    stream_digest, V2_BLOCKER, V2_SCALAR_KEYS)
from megartx.speculative_native_probe import ProbeError
from megartx.speculative_native_receipt import TORCH_MEMORY_SOURCE_SHA256, RECEIPT_LIMITS
from test_speculative_native_receipt_client import scalar, private_fixture

ROOT = Path(__file__).resolve().parents[1]


def v2_scalar():
    row = scalar()
    row.update(schema="megartx-native-v2-zero-forward-receipt-v1", runner_policy=dict(owner.POLICY),
               mutable_verifier_lease_granted=False, drafter_loaded=False,
               metadata_built=False, input_batch_prepared=False)
    row["decision"]["blockers"].append(V2_BLOCKER)
    return row


def v2_private_fixture():
    raw, identity, row, frozen = private_fixture()
    extra = v2_scalar()
    for key in V2_SCALAR_KEYS | {"schema"}:
        row[key] = extra[key]
    row["decision"]["blockers"].append(V2_BLOCKER)
    raw["schema"] = row["schema"]
    raw["purpose"] = lifecycle.PURPOSE
    raw["source_sha256"].update(owner.source_binding()["additional_sources"])
    raw["v2_owner"] = {k: row[k] for k in V2_SCALAR_KEYS}
    raw["v2_owner"].update(request_state_identity=101, block_tables_identity=102,
        model_state_identity=103, metadata_builder_identities=[104])
    raw.update(runner_identity=105, model_identity=106,
               allocator_counter_source_sha256=TORCH_MEMORY_SOURCE_SHA256,
               metadata_acquisition_limits=dict(RECEIPT_LIMITS))
    raw["allocator"]["acquisition"] = {"query": "memory_stats_as_nested_dict", "queries": 1,
        "device": "cuda:0", "native_query_preallocation_bound_bytes": None,
        "native_query_bound_status": "not_exposed_by_pinned_Torch_no_peak_host_bound_claim",
        "host_free_before_bytes": 10 << 30, "host_free_after_bytes": 10 << 30}
    identity.update(schema="megartx-native-v2-receipt-evidence-identity-v1", purpose=plan.PURPOSE,
                    runner_lane="v2", runner_policy=dict(owner.POLICY))
    frozen.update(schema=plan.SCHEMA, runner_lane="v2", installed_v2_source_binding=owner.source_binding())
    row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
    return raw, identity, row, frozen


def config():
    return NS(use_v2_model_runner=True, is_mm_encoder_only=False, speculative_config=None,
        cache_config=NS(enable_prefix_caching=False), scheduler_config=NS(async_scheduling=False),
        model_config=NS(enforce_eager=True), parallel_config=NS(world_size=1, enable_dbo=False,
        worker_extension_cls=lifecycle.WORKER_EXTENSION), kv_transfer_config=None, ec_transfer_config=None)


def proposal():
    return {"schema": plan.SCHEMA, "purpose": plan.PURPOSE, "plan_sha256": "a" * 64,
        "source_head": plan.REVIEWED_OWNER, "v2_owner_plan_sha256": "b" * 64, "runner_lane": "v2",
        "runner_policy": dict(owner.POLICY), "gpu_authorized": False, "target_probe_authorized": False,
        "drafter_authorized": False, "preflight_blockers": list(plan.PREFLIGHT_BLOCKERS),
        "collector_materialization_guard": copy.deepcopy(legacy.COLLECTOR_GUARD), "client_mode": "async",
        "source_sha256": {"src/megartx/speculative_native_evidence.py": "c" * 64}}


def authorization(frozen):
    return {"schema": plan.AUTH_SCHEMA, "purpose": plan.PURPOSE,
        "plan_sha256": frozen["plan_sha256"], "source_head": frozen["source_head"],
        "runner_lane": "v2", "v2_owner_plan_sha256": frozen["v2_owner_plan_sha256"],
        "independent_review_clear": True, "exact_head_ci_green": True,
        "parent_source_protocol_accepted": True, "gpu_slot_assigned": True, "owned_lifecycle_verified": True,
        "target_probe_authorized": False, "drafter_authorized": False,
        **{k: "fixture_only" for k in ("parent_slot", "parent_acceptance_reference", "independent_review_reference", "ci_reference")}}


class FixtureClient:
    def __init__(self, fail=None, row=None):
        self.calls, self.shutdowns = [], []
        self.fail, self.row = fail, v2_scalar() if row is None else row
    def call_utility(self, name, *args):
        self.calls.append((name, args))
        if name == self.fail:
            raise TimeoutError("fixture uncertainty")
        if name == "pause_scheduler": return None
        if name == lifecycle.UTILITIES[0]: return self.row
        if name == lifecycle.UTILITIES[2]: return dict(EXPECTED_RELEASE)
        raise AssertionError("Unexpected V1/probe/request utility")
    async def call_utility_async(self, name, *args):
        await asyncio.sleep(0)
        return self.call_utility(name, *args)
    def shutdown(self, timeout): self.shutdowns.append(timeout)


def session(client):
    deadline = time.monotonic() + 120
    receipt = ReceiptSession(client, deadline=deadline, resources=lambda: None, runner_lane="v2")
    admission = {"phase": "zero_forward_v2_receipt", "client_purpose": plan.PURPOSE,
        "client_plan_sha256": "b" * 64, "source_head": plan.REVIEWED_OWNER,
        "deadline_monotonic": deadline, "runner_lane": "v2", "target_probe_authorized": False,
        "drafter_authorized": False}
    return receipt, admission


class V2ClientControls(unittest.TestCase):
    def test_sync_only_exact_V2_pause_receipt_release_shutdown(self):
        client = FixtureClient()
        receipt, admission = session(client)
        receipt.run_sync(admission)
        self.assertEqual(client.calls, [("pause_scheduler", ("keep", False)),
            (lifecycle.UTILITIES[0], (admission,)), (lifecycle.UTILITIES[2], ())])
        self.assertEqual(len(client.shutdowns), 1)
        self.assertEqual(receipt.state, State.CLOSED)

    def test_each_transport_failure_is_single_use_no_second_release(self):
        for utility, calls in (("pause_scheduler", 1), (lifecycle.UTILITIES[0], 2), (lifecycle.UTILITIES[2], 3)):
            with self.subTest(utility=utility):
                client = FixtureClient(fail=utility)
                receipt, admission = session(client)
                with self.assertRaises(TimeoutError): receipt.run_sync(admission)
                with self.assertRaises(ProbeError): receipt.run_sync(admission)
                self.assertEqual(len(client.calls), calls)
                self.assertEqual(len(client.shutdowns), 1)

    def test_legacy_wrong_purpose_and_later_authority_rejected_before_pause(self):
        for key, value in (("phase", "zero_forward_receipt"), ("client_purpose", legacy.PURPOSE),
                ("runner_lane", "v1-legacy"), ("target_probe_authorized", True), ("drafter_authorized", 0)):
            client = FixtureClient()
            receipt, admission = session(client)
            admission[key] = value
            with self.assertRaises(ProbeError): receipt.run_sync(admission)
            self.assertEqual(client.calls, [])
            self.assertEqual(len(client.shutdowns), 1)

    def test_V1_summary_cannot_be_relabelled_by_client(self):
        client = FixtureClient(row=scalar())
        receipt, admission = session(client)
        with self.assertRaises(ProbeError): receipt.run_sync(admission)
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(receipt.release_attempted)

    def test_unknown_lane_fails_closed(self):
        with self.assertRaises(ProbeError):
            ReceiptSession(FixtureClient(), deadline=time.monotonic()+120, resources=lambda: None, runner_lane="auto")

    def test_actual_factory_checks_resolved_V2_before_constructor(self):
        from megartx.speculative_native_client import make_actual_client
        row = config(); calls = []
        class Sync: pass
        class Async: pass
        class Args:
            def __init__(self, **kwargs): calls.append(("args", kwargs))
            def create_engine_config(self): return row
        class Executor:
            @staticmethod
            def get_class(config): return "fixture_executor"
        class CoreClient:
            @staticmethod
            def make_client(**kwargs): calls.append(("make", kwargs)); return Sync()
        names = ("vllm", "vllm.engine", "vllm.engine.arg_utils", "vllm.v1", "vllm.v1.executor",
                 "vllm.v1.executor.abstract", "vllm.v1.engine", "vllm.v1.engine.core_client")
        modules = {name: ModuleType(name) for name in names}
        modules["vllm.engine.arg_utils"].EngineArgs = Args
        modules["vllm.v1.executor.abstract"].Executor = Executor
        module = modules["vllm.v1.engine.core_client"]
        module.EngineCoreClient, module.SyncMPClient, module.AsyncMPClient = CoreClient, Sync, Async
        frozen = {"schema": plan.SCHEMA, "runner_lane": "v2", "client_mode": "sync", "engine_kwargs": plan.engine_kwargs()}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
            self.assertIs(type(make_actual_client(frozen)), Sync)
            self.assertEqual(calls[-1][1]["renderer"], None)
            self.assertEqual(calls[-1][1]["multiprocess_mode"], True)
            self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", os.environ)
            calls.clear(); row.use_v2_model_runner = False
            with self.assertRaises(ProbeError): make_actual_client(frozen)
            self.assertEqual([name for name, _ in calls], ["args"])

    def test_resolved_V2_required_no_override_or_drafter(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(validate_actual_config(config(), runner_lane="v2")["use_v2_model_runner"], True)
            for selected in (False, None, 1):
                row = config(); row.use_v2_model_runner = selected
                with self.assertRaises(ProbeError): validate_actual_config(row, runner_lane="v2")
                self.assertIs(row.use_v2_model_runner, selected)
            for key, value in (("speculative_config", object()), ("kv_transfer_config", object())):
                row = config(); setattr(row, key, value)
                with self.assertRaises(ProbeError): validate_actual_config(row, runner_lane="v2")
            row = config(); row.parallel_config.worker_extension_cls = legacy.WORKER_EXTENSION
            with self.assertRaises(ProbeError): validate_actual_config(row, runner_lane="v2")
        for value in ("1", "0", ""):
            with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": value}, clear=True):
                with self.assertRaises(ProbeError): validate_actual_config(config(), runner_lane="v2")


class V2AsyncControls(unittest.IsolatedAsyncioTestCase):
    async def test_async_exact_V2_sequence(self):
        client = FixtureClient(); receipt, admission = session(client)
        await receipt.run_async(admission)
        self.assertEqual([n for n, _ in client.calls], ["pause_scheduler", lifecycle.UTILITIES[0], lifecycle.UTILITIES[2]])
        self.assertEqual(len(client.shutdowns), 1)

    async def test_admission_is_snapshot_before_awaiting_pause(self):
        class MutateDuringPause(FixtureClient):
            async def call_utility_async(self, name, *args):
                if name == "pause_scheduler":
                    admission["client_plan_sha256"] = "0" * 64
                    admission["drafter_authorized"] = True
                return await super().call_utility_async(name, *args)
        client = MutateDuringPause(); receipt, admission = session(client)
        expected = copy.deepcopy(admission)
        await receipt.run_async(admission)
        self.assertEqual(client.calls[1][1][0], expected)

    async def test_cancelled_V2_receipt_never_releases_or_retries(self):
        entered = asyncio.Event()
        class Pending(FixtureClient):
            async def call_utility_async(self, name, *args):
                if name == lifecycle.UTILITIES[0]:
                    self.calls.append((name, args)); entered.set()
                    await asyncio.Event().wait()
                return await super().call_utility_async(name, *args)
        client = Pending(); receipt, admission = session(client)
        task = asyncio.create_task(receipt.run_async(admission))
        await entered.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.shutdowns), 1)
        with self.assertRaises(ProbeError): await receipt.run_async(admission)


class V2PlanControls(unittest.TestCase):
    def test_explicit_plan_dispatch_and_distinct_environment(self):
        self.assertIs(legacy.plan_api(runner_lane="v1-legacy"), legacy)
        self.assertIs(legacy.plan_api(runner_lane="v2"), plan)
        self.assertIs(legacy.plan_api({"schema": plan.SCHEMA}), plan)
        with self.assertRaises(ProbeError): legacy.plan_api({"schema": "native_auto"})
        env = legacy.environment(ROOT, Path('/tmp/fixture'), runner_lane="v2")
        for key in plan.FORBIDDEN_ENV: self.assertNotIn(key, env)
        self.assertEqual(env["MEGARTX_NATIVE_V2_DIAGNOSTIC"], "1")
        self.assertEqual(env["MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE"], "1")
        self.assertEqual(plan.engine_kwargs()["worker_extension_cls"], lifecycle.WORKER_EXTENSION)
        self.assertNotIn("speculative_config", plan.engine_kwargs())

    def test_V1_authorization_never_admits_V2_and_blockers_sticky(self):
        frozen = proposal(); auth = authorization(frozen)
        with self.assertRaisesRegex(ProbeError, "cannot override"): plan.validate_authorization(auth, frozen)
        frozen["preflight_blockers"] = []  # CPU fault fixture; never a source-valid plan
        self.assertEqual(plan.validate_authorization(auth, frozen), auth)
        for key, value in (("schema", "megartx-native-receipt-authorization-v1"),
                ("purpose", legacy.PURPOSE), ("v2_owner_plan_sha256", "0"*64),
                ("target_probe_authorized", True), ("drafter_authorized", 0), ("exact_head_ci_green", 1)):
            changed = copy.deepcopy(auth); changed[key] = value
            with self.assertRaises(ProbeError): plan.validate_authorization(changed, frozen)

    def test_immutable_plan_values_and_exact_boolean_types(self):
        frozen = proposal()
        with patch.object(plan, "freeze", return_value=frozen):
            plan.validate_plan(copy.deepcopy(frozen), ROOT)
            for key, value in (("gpu_authorized", 0), ("runner_lane", "v1-legacy"), ("source_head", "0"*40)):
                changed = copy.deepcopy(frozen); changed[key] = value
                with self.assertRaises(ProbeError): plan.validate_plan(changed, ROOT)

    def test_worker_rechecks_client_source_before_native_reservation(self):
        frozen = proposal(); frozen["preflight_blockers"] = []
        admission = {"client_purpose": plan.PURPOSE, "client_plan_sha256": frozen["plan_sha256"],
            "runner_lane": "v2", "source_head": frozen["source_head"], "v2_plan_sha256": frozen["v2_owner_plan_sha256"],
            "client_evidence_source_sha256": "c"*64, "target_probe_authorized": False,
            "drafter_authorized": False, "client_mode": "async"}
        with patch.object(plan, "freeze", return_value=frozen):
            plan.check_client_admission(admission, ROOT)
            for key in ("client_plan_sha256", "source_head", "v2_plan_sha256", "client_evidence_source_sha256"):
                bad = dict(admission); bad[key] = "wrong"
                with self.assertRaises(ProbeError): plan.check_client_admission(bad, ROOT)
            frozen["preflight_blockers"] = ["uncomposed"]
            with self.assertRaises(ProbeError): plan.check_client_admission(admission, ROOT)

    def test_schemas_bind_new_sources_runner_policy_and_false_authorities(self):
        schema = json.loads((ROOT/'schemas/speculative-native-v2-receipt-client-plan.schema.json').read_text())
        self.assertEqual(schema['properties']['schema']['const'], plan.SCHEMA)
        self.assertEqual(schema['properties']['runner_binding']['const'], plan.RUNNER_BINDING)
        self.assertEqual(schema['properties']['preflight_blockers']['const'], plan.PREFLIGHT_BLOCKERS)
        for path in plan.CLIENT_FILES: self.assertIn(path, schema['properties']['source_sha256']['required'])
        self.assertEqual(schema['properties']['engine_kwargs']['const'], plan.engine_kwargs())
        auth = json.loads((ROOT/'schemas/speculative-native-v2-receipt-authorization.schema.json').read_text())
        self.assertEqual(auth['properties']['schema']['const'], plan.AUTH_SCHEMA)
        self.assertIs(auth['properties']['drafter_authorized']['const'], False)

    def test_CPU_only_imports_and_explicit_preflight_help(self):
        code = "import sys; from megartx import speculative_native_v2_plan, speculative_native_client, speculative_native_evidence, speculative_native_compare; assert not any(k.split('.')[0] in ('torch','vllm','flashinfer') for k in sys.modules)"
        subprocess.run([sys.executable, '-S', '-c', code], env=dict(os.environ, PYTHONPATH=str(ROOT/'src')), check=True, capture_output=True)
        result = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/speculative_native_receipt_preflight.py'), '--freeze'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'requires explicit --runner-lane', result.stderr)


class V2ComparisonControls(unittest.TestCase):
    def test_complete_private_fixture_is_V2_only_no_fit_or_probe(self):
        result = verify_private_receipt(*v2_private_fixture())
        self.assertEqual(result['schema'], 'megartx-native-v2-receipt-comparison-v1')
        self.assertFalse(result['fit_admitted']); self.assertFalse(result['later_probe_authorized'])
        self.assertFalse(result['drafter_authorized']); self.assertIn(V2_BLOCKER, result['fit_blockers'])
        self.assertNotIn('runner_identity', result); self.assertNotIn('worker_pid', result)

    def test_cross_lane_scalar_rejection(self):
        with self.assertRaises(ProbeError): validate_scalar_receipt(v2_scalar())
        with self.assertRaises(ProbeError): validate_scalar_receipt(scalar(), runner_lane='v2')
        for key, value in (('drafter_loaded', True), ('metadata_built', 0), ('input_batch_prepared', True)):
            row = v2_scalar(); row[key] = value
            with self.assertRaises(ProbeError): validate_scalar_receipt(row, runner_lane='v2')
        row=v2_scalar(); row['runner_policy']['resolved_v2']=1
        with self.assertRaises(ProbeError): validate_scalar_receipt(row, runner_lane='v2')

    def test_missing_V2_blocker_and_raw_private_payload_fail(self):
        row=v2_scalar(); row['decision']['blockers'].remove(V2_BLOCKER)
        with self.assertRaises(ProbeError): validate_scalar_receipt(row, runner_lane='v2')
        row=v2_scalar(); row['v2_owner']={'private':42}
        with self.assertRaises(ProbeError): validate_scalar_receipt(row, runner_lane='v2')

    def test_raw_policy_boolean_aliases_fail_with_matching_receipt_digest(self):
        for key, value in (("resolved_v2", 1), ("selection_override_injected", 0)):
            raw, identity, row, frozen = v2_private_fixture()
            raw["v2_owner"]["runner_policy"] = dict(raw["v2_owner"]["runner_policy"])
            raw["v2_owner"]["runner_policy"][key] = value
            row["receipt_sha256"] = identity["receipt_sha256"] = stream_digest(raw)
            with self.assertRaises(ProbeError): lifecycle._check_result(raw)
            with self.assertRaises(ProbeError): verify_private_receipt(raw, identity, row, frozen)
            # The independent comparison also checks exact nested types even if
            # an internal owner check is replaced in this CPU fault fixture.
            with patch.object(lifecycle, "_check_result"):
                with self.assertRaisesRegex(ProbeError, "private/public owner capability"):
                    verify_private_receipt(raw, identity, row, frozen)

    def test_source_and_owner_drift_rejected_with_matching_digest(self):
        for mutation in (lambda raw:raw['source_sha256'].pop('vllm/v1/worker/gpu/model_runner.py'),
                lambda raw:raw['v2_owner'].__setitem__('drafter_loaded', True),
                lambda raw:raw['v2_owner'].__setitem__('metadata_builder_identities', [42,42]),
                lambda raw:raw.__setitem__('runner_identity', False),
                lambda raw:raw.__setitem__('purpose', 'exclusive_native_verifier')):
            raw, identity, row, frozen=v2_private_fixture(); mutation(raw)
            row['receipt_sha256']=identity['receipt_sha256']=stream_digest(raw)
            with self.assertRaises(ProbeError): verify_private_receipt(raw,identity,row,frozen)

    def test_native_query_unknown_peak_and_monitored_host_reserve_preserved(self):
        for key,value in (('queries',True),('native_query_preallocation_bound_bytes',0),
                          ('host_free_before_bytes',1<<30),('host_free_after_bytes',0),('device','cuda:1')):
            raw,identity,row,frozen=v2_private_fixture();raw['allocator']['acquisition'][key]=value
            row['receipt_sha256']=identity['receipt_sha256']=stream_digest(raw)
            with self.assertRaises(ProbeError): verify_private_receipt(raw,identity,row,frozen)

    def test_V1_evidence_identity_cannot_authorize_V2(self):
        for key,value in (('purpose',legacy.PURPOSE),('schema','megartx-native-receipt-evidence-identity-v1'),('runner_lane','v1-legacy')):
            raw,identity,row,frozen=v2_private_fixture();identity[key]=value
            with self.assertRaises(ProbeError): verify_private_receipt(raw,identity,row,frozen)


class V2EvidenceHookControls(unittest.TestCase):
    def fixture(self):
        calls=[]
        def original(core, admission): calls.append('receipt'); return v2_scalar()
        class Core: megartx_owned_native_v2_receipt=original
        modules={name:ModuleType(name) for name in ('vllm','vllm.v1','vllm.v1.engine','vllm.v1.engine.core')}
        modules['vllm.v1.engine.core'].EngineCoreProc=Core
        return original,Core,modules,calls

    def test_existing_V2_lease_writes_immutable_exact_identity_and_raw_receipt(self):
        raw, identity, row, frozen = v2_private_fixture()
        ticket = {"purpose": lifecycle.PURPOSE, "engine_pid": os.getpid(), "engine_start": "123",
            "nonce": raw["lease_nonce"], "block_sizes": identity["reserved_group_block_sizes"],
            "groups": identity["reserved_groups"]}
        lease = NS(ticket=ticket, receipt=raw, released=False, poisoned=False)
        core = NS(_megartx_native_lease=lease)
        with tempfile.TemporaryDirectory() as directory, patch.object(lifecycle, "_require_core"), \
                patch.object(evidence, "_process_start", return_value="123"):
            Path(directory).chmod(0o700)
            admission = {"client_purpose": plan.PURPOSE, "client_plan_sha256": frozen["plan_sha256"],
                "source_head": frozen["source_head"], "private_receipt_directory": directory,
                "client_evidence_source_sha256": legacy.hash_file(evidence.__file__)}
            self.assertEqual(evidence.persist_owned_v2_receipt(core, admission, row), row)
            saved = json.loads((Path(directory)/"native-receipt-identity.private.json").read_text())
            self.assertEqual(saved["schema"], "megartx-native-v2-receipt-evidence-identity-v1")
            self.assertEqual(saved["purpose"], plan.PURPOSE)
            self.assertEqual(saved["runner_policy"], owner.POLICY)
            self.assertEqual(saved["receipt_sha256"], stream_digest(raw))
            self.assertEqual((Path(directory)/"native-receipt.private.json").stat().st_mode & 0o777, 0o400)
            self.assertFalse(lease.released)
            with self.assertRaises(FileExistsError): evidence.persist_owned_v2_receipt(core, admission, row)
            lease.ticket["purpose"] = "exclusive_native_verifier"
            with self.assertRaises(ProbeError): evidence.persist_owned_v2_receipt(core, admission, row)

    def test_default_off_and_conflicting_flags_before_runtime_import(self):
        with patch.dict(os.environ,{},clear=True): self.assertFalse(evidence.install_native_v2_receipt_evidence())
        for key in plan.FORBIDDEN_ENV:
            with patch.dict(os.environ,{'MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE':'1','MEGARTX_NATIVE_V2_DIAGNOSTIC':'1',key:''},clear=True):
                with self.assertRaises(ProbeError): evidence.install_native_v2_receipt_evidence()

    def test_wrap_only_exact_V2_utility_duplicate_rejected(self):
        original,core,modules,calls=self.fixture()
        with patch.dict(sys.modules,modules),patch.object(lifecycle,'owned_receipt',original),patch.dict(os.environ,{'MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE':'1','MEGARTX_NATIVE_V2_DIAGNOSTIC':'1'},clear=True):
            self.assertTrue(evidence.install_native_v2_receipt_evidence())
            self.assertIs(core.megartx_owned_native_v2_receipt.__wrapped__,original)
            self.assertFalse(hasattr(core,'megartx_owned_native_receipt'))
            self.assertFalse(hasattr(core,'megartx_owned_native_v2_probe'))
            with self.assertRaisesRegex(ProbeError,'duplicate'): evidence.install_native_v2_receipt_evidence()

    def test_bad_client_admission_fails_before_reservation(self):
        original,core,modules,calls=self.fixture()
        with patch.dict(sys.modules,modules),patch.object(lifecycle,'owned_receipt',original),patch.object(plan,'check_client_admission',side_effect=ProbeError('unbound')),patch.dict(os.environ,{'MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE':'1','MEGARTX_NATIVE_V2_DIAGNOSTIC':'1'},clear=True):
            evidence.install_native_v2_receipt_evidence()
            with self.assertRaises(ProbeError): core().megartx_owned_native_v2_receipt({})
            self.assertEqual(calls,[])

    def test_known_writer_failure_preserves_primary_and_retained_release_route(self):
        original,core,modules,calls=self.fixture(); primary=ProbeError('writer fixture')
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules),patch.object(lifecycle,'owned_receipt',original),patch.object(plan,'check_client_admission'),patch.object(evidence,'persist_owned_v2_receipt',side_effect=primary),patch.object(lifecycle,'_require_core'),patch.object(lifecycle.common,'_release',side_effect=TimeoutError('uncertain')) as release,patch.dict(os.environ,{'MEGARTX_NATIVE_V2_RECEIPT_EVIDENCE':'1','MEGARTX_NATIVE_V2_DIAGNOSTIC':'1'},clear=True):
            Path(directory).chmod(0o700);evidence.install_native_v2_receipt_evidence()
            with self.assertRaises(ProbeError) as caught: core().megartx_owned_native_v2_receipt({'private_receipt_directory':directory})
            self.assertIs(caught.exception,primary);release.assert_called_once()
            self.assertEqual(calls,['receipt'])


if __name__ == '__main__': unittest.main()

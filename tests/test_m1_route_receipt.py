"""Pure CPU route observation protocol tests; no device or installed producer."""
import ast
from dataclasses import FrozenInstanceError, fields, replace
import gc
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import unittest
import weakref

from megartx.m1_route_receipt import (
    BindingSnapshot, GuardSnapshot, OutputSnapshot, ProducerSnapshot,
    RouteObservationError, RouteObservationLedger, RouteObservationReceipt,
    StreamSnapshot, UNKNOWN,
)


class Owner:
    pass


class EqualToEverything:
    def __eq__(self, other):
        raise AssertionError("identity objects must never invoke equality")


class Fixture:
    def __init__(self):
        refs = {f.name: Owner() for f in fields(BindingSnapshot)
                if f.name not in ("source_key", "cache_key")}
        self.bindings = BindingSnapshot(**refs, source_key="cpu-fixture-source",
                                        cache_key="cpu-fixture-cache")
        self.guards = GuardSnapshot(capture_state="none", capture_fn=None,
            replay_output=None, eplb_state=None, instance_overrides=(), pre_run_hooks=(),
            launch_hooks=(), simulation=False, diagnostics=False, controlled=False,
            forced=False, routing_variant="gemma_cuda")
        self.producer = StreamSnapshot(0, Owner(), 0)
        self.consumer = replace(self.producer, handle=17)
        self.output = OutputSnapshot(Owner(), Owner(), 4096, 4096, 32, 0, 0)
        self.frame = Owner()
        self.ledger = RouteObservationLedger(self.bindings)

    def invocation(self):
        return self.ledger.invocation(self.bindings, self.guards, self.producer, self.frame)

    def snapshot(self, **changes):
        return ProducerSnapshot(self.bindings, self.guards, self.producer, self.output, **changes)

    def produce(self, token):
        self.ledger.observe_production(token, self.snapshot(), self.snapshot())

    def dependency(self, token):
        self.ledger.observe_dependency(token, self.producer, self.consumer)

    def consume(self, token, **changes):
        kwargs = dict(bindings=self.bindings, guards=self.guards, output=self.output,
                      consumer=self.consumer, frame=self.frame)
        kwargs.update(changes)
        return self.ledger.consume(token, **kwargs)

    def complete(self, token):
        self.produce(token)
        self.dependency(token)
        return self.consume(token)


class RouteReceiptTests(unittest.TestCase):
    def assert_poisoned_and_closed(self, f):
        self.assertTrue(f.ledger.poisoned)
        self.assertFalse(f.ledger.active)
        with self.assertRaises(RouteObservationError):
            with f.invocation():
                self.fail("poisoned ledger reopened")

    def test_success_is_metadata_only_and_readback_is_unconditionally_required(self):
        f = Fixture()
        with f.invocation() as token:
            receipt = f.complete(token)
            self.assertTrue(f.ledger.active)
            self.assertTrue(receipt.route_readback_required)
            self.assertEqual((receipt.generation, receipt.device, receipt.producer_stream,
                              receipt.consumer_stream, receipt.output_pointer), (1, 0, 0, 17, 4096))
        self.assertFalse(f.ledger.active)
        self.assertFalse(f.ledger.poisoned)
        with self.assertRaises(TypeError):
            bool(receipt)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            receipt.route_readback_required = False
        with self.assertRaises(TypeError):
            RouteObservationReceipt(1, 0, 0, 17, 4096, route_readback_required=False)

    def test_generations_are_fresh_even_when_address_and_owner_are_reused(self):
        f = Fixture()
        with f.invocation() as first:
            a = f.complete(first)
        with f.invocation() as second:
            b = f.complete(second)
        self.assertIsNot(first, second)
        self.assertEqual((a.generation, b.generation), (1, 2))
        self.assertEqual(a.output_pointer, b.output_pointer)

    def test_incomplete_expected_identity_is_rejected(self):
        f = Fixture()
        for field in fields(BindingSnapshot):
            for missing in (None, UNKNOWN):
                with self.subTest(field=field.name, missing=type(missing).__name__):
                    with self.assertRaises(RouteObservationError):
                        RouteObservationLedger(replace(f.bindings, **{field.name: missing}))

    def test_unknown_guards_are_not_silently_false(self):
        f = Fixture()
        with self.assertRaises(RouteObservationError):
            with f.ledger.invocation(f.bindings, GuardSnapshot(), f.producer, f.frame):
                self.fail("unknown guard admitted")
        self.assertFalse(f.ledger.poisoned)
        with f.invocation() as token:
            f.complete(token)

    def test_each_missing_guard_is_rejected_before_open_without_poison(self):
        for field in fields(GuardSnapshot):
            f = Fixture()
            with self.subTest(field=field.name):
                with self.assertRaises(RouteObservationError):
                    with f.ledger.invocation(f.bindings, replace(f.guards, **{field.name: UNKNOWN}),
                                             f.producer, f.frame):
                        self.fail("missing guard admitted")
                self.assertFalse(f.ledger.poisoned)
                self.assertFalse(f.ledger.active)

    def test_bad_initial_binding_frame_and_stream_are_harmless_unsupported(self):
        f = Fixture()
        for bindings, stream, frame in (
                (replace(f.bindings, router=Owner()), f.producer, f.frame),
                (f.bindings, replace(f.producer, context=None), f.frame),
                (f.bindings, replace(f.producer, handle=-1), f.frame),
                (f.bindings, replace(f.producer, device=True), f.frame),
                (f.bindings, f.producer, None)):
            with self.assertRaises(RouteObservationError):
                with f.ledger.invocation(bindings, f.guards, stream, frame):
                    self.fail("bad metadata admitted")
            self.assertFalse(f.ledger.poisoned)

    def test_each_current_identity_substitution_poisons_at_production(self):
        for field in fields(BindingSnapshot):
            f = Fixture()
            value = "other-fixture-key" if field.name.endswith("key") else Owner()
            with self.subTest(field=field.name):
                with self.assertRaises(RouteObservationError):
                    with f.invocation() as token:
                        changed = replace(f.snapshot(), bindings=replace(f.bindings, **{field.name: value}))
                        f.ledger.observe_production(token, f.snapshot(), changed)
                self.assert_poisoned_and_closed(f)

    def test_both_launch_boundaries_are_validated(self):
        for boundary in ("before", "after"):
            f = Fixture()
            with self.subTest(boundary=boundary), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    observations = dict(before=f.snapshot(), after=f.snapshot())
                    observations[boundary] = replace(f.snapshot(), bindings=replace(f.bindings, cuda_function=Owner()))
                    f.ledger.observe_production(token, **observations)
            self.assert_poisoned_and_closed(f)

    def test_callback_simulator_override_prepare_and_capture_tampering(self):
        values = dict(capture_state="unknown", capture_fn=Owner(), replay_output=Owner(),
            eplb_state=Owner(), instance_overrides=("select_experts",), pre_run_hooks=(Owner(),),
            launch_hooks=(Owner(),), simulation=True, diagnostics=True, controlled=True,
            forced=True, routing_variant="torch_or_simulated")
        for name, value in values.items():
            f = Fixture()
            with self.subTest(field=name), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.ledger.observe_production(token, f.snapshot(), replace(f.snapshot(),
                        guards=replace(f.guards, **{name: value})))
            self.assert_poisoned_and_closed(f)

    def test_consumer_rechecks_all_bindings_and_guards(self):
        for field in fields(BindingSnapshot):
            f = Fixture()
            value = "other-key" if field.name.endswith("key") else Owner()
            with self.subTest(binding=field.name), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.produce(token)
                    f.dependency(token)
                    f.consume(token, bindings=replace(f.bindings, **{field.name: value}))
            self.assert_poisoned_and_closed(f)
        for field in fields(GuardSnapshot):
            f = Fixture()
            with self.subTest(guard=field.name), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.produce(token)
                    f.dependency(token)
                    f.consume(token, guards=replace(f.guards, **{field.name: UNKNOWN}))
            self.assert_poisoned_and_closed(f)

    def test_specialization_geometry_and_grid_must_match_exactly(self):
        for changes in (dict(experts=127), dict(top_k=7), dict(block_experts=256),
                        dict(num_warps=2), dict(grid=(2,)), dict(grid=(True,)), dict(num_warps=True)):
            f = Fixture()
            with self.subTest(changes=changes), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.ledger.observe_production(token, f.snapshot(), f.snapshot(**changes))
            self.assert_poisoned_and_closed(f)

    def test_output_shape_dtype_alignment_extent_and_storage_relation(self):
        changes = [dict(owner=None), dict(storage_owner=None), dict(pointer=4100),
            dict(pointer=4097), dict(storage_bytes=31), dict(offset_bytes=4),
            dict(device=1), dict(shape=(2, 8)), dict(shape=(True, 8)),
            dict(strides=(1, 8)), dict(dtype="int64"), dict(view_bytes=64),
            dict(storage_pointer=0), dict(pointer=True)]
        for change in changes:
            f = Fixture()
            with self.subTest(change=change), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    invalid = replace(f.snapshot(), output=replace(f.output, **change))
                    f.ledger.observe_production(token, invalid, invalid)
            self.assert_poisoned_and_closed(f)

    def test_same_pointer_different_owner_is_rejected_at_both_boundaries(self):
        for key in ("owner", "storage_owner"):
            for place in ("producer", "consumer"):
                f = Fixture()
                with self.subTest(key=key, place=place), self.assertRaises(RouteObservationError):
                    with f.invocation() as token:
                        other = replace(f.output, **{key: Owner()})
                        if place == "producer":
                            f.ledger.observe_production(token, f.snapshot(), replace(f.snapshot(), output=other))
                        else:
                            f.produce(token)
                            f.dependency(token)
                            f.consume(token, output=other)
                self.assert_poisoned_and_closed(f)

    def test_consumer_view_metadata_change_rejected_even_if_owner_matches(self):
        f = Fixture()
        with self.assertRaises(RouteObservationError):
            with f.invocation() as token:
                f.produce(token)
                f.dependency(token)
                f.consume(token, output=replace(f.output, storage_bytes=64))
        self.assert_poisoned_and_closed(f)

    def test_producer_stream_device_and_context_must_match(self):
        for change in (dict(handle=7), dict(device=1), dict(context=Owner()), dict(handle=True)):
            f = Fixture()
            with self.subTest(change=change), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    invalid = replace(f.snapshot(), stream=replace(f.producer, **change))
                    f.ledger.observe_production(token, invalid, invalid)
            self.assert_poisoned_and_closed(f)

    def test_dependency_must_follow_production_and_match_context(self):
        for producer_change, consumer_change in ((dict(handle=99), {}), ({}, dict(device=1)),
                ({}, dict(context=Owner())), ({}, dict(handle=0))):
            f = Fixture()
            with self.subTest(producer=producer_change, consumer=consumer_change), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.produce(token)
                    f.ledger.observe_dependency(token, replace(f.producer, **producer_change),
                                                 replace(f.consumer, **consumer_change))
            self.assert_poisoned_and_closed(f)

    def test_consumer_stream_and_frame_must_match(self):
        for key, value in (("consumer", StreamSnapshot(0, Owner(), 17)), ("frame", Owner())):
            f = Fixture()
            with self.subTest(key=key), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.produce(token)
                    f.dependency(token)
                    f.consume(token, **{key: value})
            self.assert_poisoned_and_closed(f)

    def test_order_and_single_use_matrix(self):
        sequences = ("D", "C", "PP", "PDD", "PDCC", "PDCP")
        for sequence in sequences:
            f = Fixture()
            with self.subTest(sequence=sequence), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    for op in sequence:
                        {"P": f.produce, "D": f.dependency, "C": f.consume}[op](token)
            self.assert_poisoned_and_closed(f)

    def test_incomplete_normal_exits_poison_and_release(self):
        for phase in ("", "P", "PD"):
            f = Fixture()
            with self.subTest(phase=phase), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    for op in phase:
                        {"P": f.produce, "D": f.dependency}[op](token)
            self.assert_poisoned_and_closed(f)

    def test_nested_scope_cannot_replace_outer_scope(self):
        f = Fixture()
        with self.assertRaises(RouteObservationError):
            with f.invocation() as token:
                with self.assertRaises(RouteObservationError):
                    with f.invocation():
                        self.fail("nested scope admitted")
                self.assertIs(f.ledger._active.token, token)
        self.assert_poisoned_and_closed(f)

    def test_forged_stale_and_foreign_ledger_tokens_never_consume_current(self):
        for kind in ("forged", "stale", "foreign"):
            f = Fixture()
            if kind == "stale":
                with f.invocation() as wrong:
                    f.complete(wrong)
            elif kind == "foreign":
                g = Fixture()
                with g.invocation() as wrong:
                    g.complete(wrong)
            else:
                wrong = object()
            with self.subTest(kind=kind), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    f.produce(token)
                    f.dependency(token)
                    with self.assertRaises(RouteObservationError):
                        f.consume(wrong)
                    self.assertIs(f.ledger._active.token, token)
            self.assert_poisoned_and_closed(f)

    def test_foreign_thread_use_poisons_but_cannot_clear_owners(self):
        for operation in ("produce", "dependency", "consume", "cleanup"):
            f = Fixture()
            errors = []
            with self.subTest(operation=operation), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    if operation in ("dependency", "consume"):
                        f.produce(token)
                    if operation == "consume":
                        f.dependency(token)
                    def run():
                        try:
                            if operation == "cleanup":
                                f.ledger._cleanup(token, None)
                            else:
                                getattr(f, operation)(token)
                        except BaseException as e:
                            errors.append(e)
                    thread = threading.Thread(target=run)
                    thread.start()
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], RouteObservationError)
                    self.assertIs(f.ledger._active.token, token)
                    self.assertIs(f.ledger._active.thread, threading.current_thread())
            self.assert_poisoned_and_closed(f)

    def test_primary_exception_identity_survives_each_phase(self):
        for phase in ("", "P", "PD", "PDC"):
            f = Fixture()
            primary = KeyboardInterrupt("fixture primary")
            with self.subTest(phase=phase):
                try:
                    with f.invocation() as token:
                        for op in phase:
                            {"P": f.produce, "D": f.dependency, "C": f.consume}[op](token)
                        raise primary
                except BaseException as error:
                    self.assertIs(error, primary)
                else:
                    self.fail("primary exception swallowed")
                self.assert_poisoned_and_closed(f)

    def test_cleanup_failure_preserves_primary_and_releases_owned_scope(self):
        class CleanupFailure(RouteObservationLedger):
            def _cleanup(self, token, primary):
                raise RuntimeError("fixture cleanup")
        for with_primary in (False, True):
            f = Fixture()
            f.ledger = CleanupFailure(f.bindings)
            primary = RuntimeError("fixture primary")
            try:
                with f.invocation() as token:
                    f.complete(token)
                    if with_primary:
                        raise primary
            except RuntimeError as error:
                if with_primary:
                    self.assertIs(error, primary)
                else:
                    self.assertEqual(str(error), "fixture cleanup")
            else:
                self.fail("cleanup exception swallowed")
            self.assert_poisoned_and_closed(f)

    def test_caught_internal_failure_does_not_erase_poison(self):
        f = Fixture()
        with self.assertRaises(RouteObservationError):
            with f.invocation() as token:
                with self.assertRaises(RouteObservationError):
                    f.consume(token)
        self.assert_poisoned_and_closed(f)
        f.ledger._cleanup(token, None)  # Idempotent after owned release.
        self.assertTrue(f.ledger.poisoned)

    def test_retained_frame_and_output_owners_release_after_close(self):
        f = Fixture()
        owner, storage, frame = weakref.ref(f.output.owner), weakref.ref(f.output.storage_owner), weakref.ref(f.frame)
        context = f.invocation()
        token = context.__enter__()
        receipt = f.complete(token)
        f.output = None
        f.frame = None
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertIsNotNone(storage())
        self.assertIsNotNone(frame())
        context.__exit__(None, None, None)
        gc.collect()
        self.assertIsNone(owner())
        self.assertIsNone(storage())
        self.assertIsNone(frame())
        self.assertTrue(receipt.route_readback_required)

    def test_identity_comparisons_do_not_call_overloaded_equality(self):
        f = Fixture()
        expected = EqualToEverything()
        f.bindings = replace(f.bindings, router=expected)
        f.producer = replace(f.producer, context=EqualToEverything())
        f.consumer = replace(f.producer, handle=17)
        f.output = replace(f.output, owner=EqualToEverything(), storage_owner=EqualToEverything())
        f.frame = EqualToEverything()
        f.ledger = RouteObservationLedger(f.bindings)
        with f.invocation() as token:
            f.complete(token)
        with self.assertRaises(RouteObservationError):
            with f.invocation() as token:
                f.produce(token)
                f.dependency(token)
                f.consume(token, bindings=replace(f.bindings, router=EqualToEverything()))

    def test_metadata_cannot_detect_owner_content_mutation(self):
        # Deliberate limitation: metadata success is NOT route validation.
        f = Fixture()
        f.output.owner.ids = list(range(8))
        with f.invocation() as token:
            f.produce(token)
            f.output.owner.ids = [128] * 8
            f.dependency(token)
            receipt = f.consume(token)
        self.assertTrue(receipt.route_readback_required)

    def test_shape_stride_and_grid_elements_never_call_overloaded_equality(self):
        for where in ("shape", "strides", "grid"):
            f = Fixture()
            bomb = EqualToEverything()
            with self.subTest(where=where), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    if where == "grid":
                        bad = replace(f.snapshot(), grid=(bomb,))
                    else:
                        bad = replace(f.snapshot(), output=replace(f.output, **{where: (bomb, 8)}))
                    with self.assertRaises(RouteObservationError):
                        f.ledger.observe_production(token, bad, bad)
                    self.assertTrue(f.ledger.poisoned)
                    with self.assertRaises(RouteObservationError):
                        f.complete(token)
            self.assert_poisoned_and_closed(f)

    def test_cleanup_formatting_and_overridden_notes_cannot_mask_primary(self):
        class BadCleanup(BaseException):
            def __str__(self):
                raise RuntimeError("cleanup formatting must not run")
        class Primary(KeyboardInterrupt):
            def add_note(self, note):
                raise RuntimeError("overridden note must not run")
        class CleanupFailure(RouteObservationLedger):
            def _cleanup(self, token, primary):
                raise BadCleanup()
        for bad_notes in (False, True):
            f = Fixture()
            f.ledger = CleanupFailure(f.bindings)
            primary = Primary("exact primary")
            if bad_notes:
                primary.__notes__ = "malformed fixture notes"
            try:
                with f.invocation() as token:
                    f.complete(token)
                    raise primary
            except BaseException as error:
                self.assertIs(error, primary)
            else:
                self.fail("primary exception swallowed")
            self.assert_poisoned_and_closed(f)

    def test_unexpected_validator_exceptions_cannot_be_caught_then_retried(self):
        for phase in ("production", "dependency", "consumption"):
            f = Fixture()
            with self.subTest(phase=phase), self.assertRaises(RouteObservationError):
                with f.invocation() as token:
                    if phase != "production":
                        f.produce(token)
                    if phase == "consumption":
                        f.dependency(token)
                    with self.assertRaises(AttributeError):
                        if phase == "production":
                            malformed = object.__new__(ProducerSnapshot)
                            f.ledger.observe_production(token, malformed, malformed)
                        elif phase == "dependency":
                            f.ledger.observe_dependency(token, f.producer, object.__new__(StreamSnapshot))
                        else:
                            f.consume(token, bindings=object.__new__(BindingSnapshot))
                    self.assertTrue(f.ledger.poisoned)
                    with self.assertRaises(RouteObservationError):
                        f.consume(token)
            self.assert_poisoned_and_closed(f)

    def test_no_runtime_import_hook_environment_switch_or_device_dependency(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "src/megartx/m1_route_receipt.py"
        tree = ast.parse(source.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertEqual(set(imports), {"contextlib", "dataclasses", "threading"})
        for path in (root / "src").rglob("*.py"):
            if path != source:
                self.assertNotIn("m1_route_receipt", path.read_text(), str(path))
        for path in (root / "kernels").glob("*"):
            if path.is_file():
                self.assertNotIn("route_receipt", path.read_text(), str(path))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("triton", sys.modules)


class ExistingNativeRouteCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing pure-CPU mock setup, but add invalid-ID cases only
        # to its temporary harness. The production header remains byte-identical.
        root = Path(__file__).resolve().parents[1]
        path = root / "numerical_reference/test_m1_preparation_control_flow.py"
        spec = importlib.util.spec_from_file_location("route_check_cpu_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.control = module.PreparationControlFlowTests
        cls.control.setUpClass()
        cls.addClassCleanup(cls.control.doClassCleanups)
        cls.work = cls.control.work
        source = cls.control.source.read_text()
        anchor = 'else if(test=="duplicate")fixture->ids[0]=fixture->ids[1];'
        if source.count(anchor) != 1:
            raise AssertionError("exact native fixture anchor changed")
        source = source.replace(anchor, anchor + '\n    else if(test=="negative_id")fixture->ids[0]=-1;'
                                + '\n    else if(test=="id_128")fixture->ids[0]=128;')
        fixture = cls.work / "route-negative-controls.cpp"
        fixture.write_text(source)
        cls.binary = cls.work / "route-negative-controls"
        compiler = shutil.which("c++") or shutil.which("g++")
        result = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-I", str(cls.work),
                                 str(fixture), "-o", str(cls.binary)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stderr)

    def test_production_header_is_unchanged(self):
        self.assertEqual(hashlib.sha256((self.work / "m1_installed_preparation.cuh").read_bytes()).digest(),
                         hashlib.sha256(self.control.header).digest())

    def test_negative_and_upper_bound_ids_still_use_checked_stock_fallback(self):
        for prefix in ([], ["lean"]):
            for case in ("negative_id", "id_128"):
                with self.subTest(prefix=prefix, case=case):
                    result = subprocess.run([str(self.binary), *prefix, case], capture_output=True,
                                            text=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

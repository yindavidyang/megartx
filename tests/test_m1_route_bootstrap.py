"""CPU-only natural lookup/launcher bootstrap, using the reviewed JIT body."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
import weakref

# The existing observer fixture explicitly loads its module. Load it before
# importing the adapter so focused-suite order preserves one class/code identity.
import test_m1_route_observer
import test_m1_route_cache_seam as cache_test
from m1_route_bootstrap import build_bootstrap


class Fixture(cache_test.Fixture):
    def __init__(self, test, limit=30):
        super().__init__()
        self.arguments = tuple(object() for _ in range(4)) + (128, 8, 128)
        self.grid = (1,)
        self.owners = tuple(object() for _ in range(30))
        self.kernel.module = object()
        pending = []
        def observe(**event):
            self.observe(**event)
            pending[0].cache_selection_observed(**event)
            self.invoke("after_bootstrap_lookup")
        candidate = cache_test.seam.build_cache_selection_candidate(cache_test._SOURCE,
            expected_sha256=cache_test._SHA, namespace=self.namespace, observe_selection=observe)
        # CPU modules model actual module dictionaries. The reference candidate's
        # copied dictionary alone deliberately cannot satisfy module ownership.
        module = types.ModuleType("route_bootstrap_fixture_" + str(id(self)))
        sys.modules[module.__name__] = module
        test.addCleanup(sys.modules.pop, module.__name__, None)
        module.__dict__.update(candidate.run.__globals__)
        module.__dict__["__name__"] = "route_bootstrap_fixture_" + str(id(self))
        module.__dict__["__builtins__"] = dict(candidate.run.__builtins__)
        fn = types.FunctionType(candidate.run.__code__, vars(module), "run", candidate.run.__defaults__)
        fn.__kwdefaults__ = candidate.run.__kwdefaults__
        module.run = fn
        exec("def launch_call(self, *args, **kwargs):\n    return self.original(*args, **kwargs)\n", vars(module))
        self.Launcher = type("Launcher", (), {"__call__": module.launch_call})
        self.launcher = self.Launcher()
        self.launcher.original = self.kernel.launch
        self.launcher.launch = self.kernel.launch
        self.kernel._run = self.launcher
        self.original_jit = types.MethodType(fn, self.jit)
        self.jit.run = self.original_jit
        self.candidate, self.module = candidate, module
        self.bootstrap = build_bootstrap(enabled=True, owners=self.owners, jit=self.jit,
            jit_run=self.original_jit, launcher_call=module.launch_call, kernel_type=type(self.kernel),
            launcher_type=self.Launcher, module_functions=((module, fn), (module, module.launch_call)),
            callback_failures=lambda: candidate.callback_failures, limit=limit)
        pending.append(self.bootstrap)
        self.jit.run = self.bootstrap.wrap_jit(self.original_jit)
        self.Launcher.__call__ = self.bootstrap.wrap_launcher(module.launch_call)

    def run(self, owner=0, **kwargs):
        with self.bootstrap.invocation(self.owners[owner]):
            return self.jit.run(*self.arguments, grid=self.grid, warmup=False, **kwargs)


class BootstrapTests(unittest.TestCase):
    def test_default_off_does_not_inspect_any_supplied_owner_or_callback(self):
        class Hostile:
            def __getattribute__(self, name):
                raise AssertionError("disabled path inspected an owner")
        self.assertIsNone(build_bootstrap(owners=Hostile(), jit=Hostile(), callback_failures=Hostile()))
        with self.assertRaises(ValueError):
            build_bootstrap(enabled=1)

    def test_actual_lookup_then_original_launch_return_captures_later_profile_inputs(self):
        f = Fixture(self)
        selected = f.run()
        self.assertIs(selected, f.kernel)
        captures = f.bootstrap.take_captures()
        self.assertEqual(len(captures), 1)
        captured = captures[0]
        self.assertIs(captured.owner, f.owners[0])
        self.assertIs(captured.cache, f.cache)
        self.assertIs(captured.selected, f.kernel)
        self.assertIs(captured.launcher, f.launcher)
        self.assertIs(captured.cuda_function, f.kernel.function)
        self.assertIs(captured.packed_metadata, f.kernel.packed_metadata)
        self.assertEqual(captured.inner_stream_handle, 71)
        self.assertEqual((len(f.binds), len(f.key_calls), len(f.cache.gets), len(f.launches)), (1, 1, 1, 1))
        self.assertTrue(captured.qualification_missing)
        with self.assertRaises(TypeError):
            bool(captured)
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertIsNone(f.bootstrap._active)

    def test_thirty_owners_share_one_jit_and_launcher_wrapper(self):
        f = Fixture(self)
        for i in range(30):
            self.assertIs(f.bootstrap.wrap_jit(f.original_jit), f.jit.run)
            self.assertIs(f.bootstrap.wrap_launcher(f.module.launch_call), f.Launcher.__call__)
            f.run(i)
        captures = f.bootstrap.take_captures()
        self.assertEqual(len(captures), 30)
        self.assertTrue(all(c.owner is owner for c, owner in zip(captures, f.owners)))
        self.assertTrue(all(c.selected is f.kernel for c in captures))
        f.run()
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertEqual((len(f.binds), len(f.cache.gets), len(f.launches)), (31, 31, 31))

    def test_cold_lookup_cannot_be_retroactively_promoted_after_original_compile(self):
        f = Fixture(self)
        f.cache.entries.clear()
        f.compile_result = f.kernel
        self.assertIs(f.run(), f.kernel)
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertEqual((len(f.binds), len(f.cache.gets), len(f.compiles), len(f.launches)), (1, 1, 1, 1))
        # The mock compiler does not populate its cache, unlike the original
        # runtime. Model its completed ordinary compile, then a later lookup.
        f.cache.entries[f.warm_key] = f.kernel
        f.run(1)
        self.assertEqual(len(f.bootstrap.take_captures()), 1)
        self.assertEqual((len(f.binds), len(f.cache.gets), len(f.compiles), len(f.launches)), (2, 2, 1, 2))

    def test_original_property_initializes_naturally_but_bootstrap_never_does(self):
        f = Fixture(self)
        f.kernel._run = None
        def initialize():
            f.events.append("lazy_init")
            f.kernel._run = f.launcher
        f.kernel._init_handles = initialize
        self.assertNotIn("lazy_init", f.events)
        f.run()
        self.assertEqual(f.events.count("lazy_init"), 1)
        self.assertEqual(len(f.bootstrap.take_captures()), 1)

    def test_partial_lookup_callback_failure_invalidates_capture(self):
        f = Fixture(self)
        f.callbacks["after_bootstrap_lookup"] = lambda: (_ for _ in ()).throw(cache_test.HostileError())
        self.assertIs(f.run(), f.kernel)
        self.assertEqual(f.candidate.callback_failures, 1)
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertEqual(len(f.launches), 1)

    def test_primary_launcher_exception_identity_survives_observation_failure(self):
        f = Fixture(self)
        primary = cache_test.HostileError()
        def fail():
            f.module.len = lambda value: len(value)
            raise primary
        f.callbacks["launch"] = fail
        with self.assertRaises(cache_test.HostileError) as caught:
            f.run()
        self.assertIs(caught.exception, primary)
        self.assertIsNone(primary.__context__)
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertEqual((len(f.binds), len(f.cache.gets), len(f.launches)), (1, 1, 1))
        self.assertIsNone(f.bootstrap._active)

    def test_module_global_and_builtin_resolution_drift_rejects_without_retry(self):
        for mutation in (lambda f: setattr(f.module, "len", lambda value: len(value)),
                         lambda f: sys.modules.__setitem__(f.module.__name__, types.ModuleType(f.module.__name__)),
                         lambda f: setattr(f.module, "compute_cache_key", lambda *args: f.compute_key(*args)),
                         lambda f: f.module.run.__builtins__.__setitem__("len", lambda value: len(value))):
            with self.subTest(mutation=mutation):
                f = Fixture(self)
                f.callbacks["launch"] = lambda: mutation(f)
                f.run()
                self.assertEqual(f.bootstrap.take_captures(), ())
                self.assertEqual(len(f.launches), 1)

    def test_pending_capture_is_not_available_during_original_launch(self):
        f = Fixture(self)
        def inspect_pending():
            with self.assertRaisesRegex(Exception, "have not drained"):
                f.bootstrap.take_captures()
        f.callbacks["launch"] = inspect_pending
        f.run()
        self.assertEqual(len(f.bootstrap.take_captures()), 1)

    def test_same_thread_nested_runner_remains_exact_once_and_poisoned(self):
        f = Fixture(self)
        def nested():
            del f.callbacks["launch"]
            f.run(1)
        f.callbacks["launch"] = nested
        f.run()
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertEqual((len(f.binds), len(f.cache.gets), len(f.launches)), (2, 2, 2))
        self.assertEqual(f.bootstrap._inflight, 0)

    def test_mismatched_function_or_bound_arguments_never_become_profile_inputs(self):
        for mutation in (lambda f: setattr(f.kernel, "function", object()),
                         lambda f: setattr(f.kernel, "packed_metadata", object()),
                         lambda f: setattr(f.kernel, "module", object())):
            f = Fixture(self)
            f.callbacks["launch"] = lambda: mutation(f)
            f.run()
            self.assertEqual(f.bootstrap.take_captures(), ())
            self.assertEqual(len(f.launches), 1)

    def test_overlapping_registered_owners_poison_and_drain_without_duplicate_calls(self):
        f = Fixture(self)
        errors = []
        def other():
            try:
                f.run(1)
            except BaseException as error:
                errors.append(error)
        def invoke():
            del f.callbacks["launch"]
            thread = threading.Thread(target=other)
            thread.start(); thread.join(10)
            self.assertFalse(thread.is_alive())
        f.callbacks["launch"] = invoke
        f.run()
        self.assertEqual(errors, [])
        self.assertEqual(f.bootstrap._inflight, 0)
        self.assertEqual(f.bootstrap.take_captures(), ())
        self.assertEqual(len(f.launches), 2)

    def test_unrelated_foreign_thread_jit_delegates_once_without_joining_scope(self):
        f = Fixture(self)
        errors = []
        def other():
            try:
                f.jit.run(*f.arguments, grid=f.grid, warmup=False)
            except BaseException as error:
                errors.append(error)
        def invoke():
            del f.callbacks["launch"]
            thread = threading.Thread(target=other)
            thread.start(); thread.join(10)
            self.assertFalse(thread.is_alive())
        f.callbacks["launch"] = invoke
        f.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(f.bootstrap.take_captures()), 1)
        self.assertEqual(len(f.launches), 2)

    def test_foreign_scope_read_can_retire_before_membership_check(self):
        f = Fixture(self)
        read_active, release = threading.Event(), threading.Event()
        errors = []
        method = f.bootstrap._scope.__func__
        def trace(frame, event, arg):
            if event == "line" and frame.f_code is method.__code__ and frame.f_lineno == method.__code__.co_firstlineno + 2:
                sys.settrace(None)
                read_active.set()
                if not release.wait(10):
                    raise AssertionError("foreign membership release timeout")
            return trace
        def foreign():
            sys.settrace(trace)
            try:
                f.jit.run(*f.arguments, grid=f.grid, warmup=False)
            except BaseException as error:
                errors.append(error)
            finally:
                sys.settrace(None)
        thread = threading.Thread(target=foreign)
        try:
            with f.bootstrap.invocation(f.owners[0]):
                thread.start()
                self.assertTrue(read_active.wait(10))
            self.assertIsNone(f.bootstrap._active)
        finally:
            release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual((len(f.binds), len(f.cache.gets), len(f.launches)), (1, 1, 1))
        self.assertEqual(f.bootstrap.take_captures(), ())

    def test_close_inside_original_call_does_not_prevent_it_or_reenable_observation(self):
        f = Fixture(self)
        f.callbacks["launch"] = f.bootstrap.close
        self.assertIs(f.run(), f.kernel)
        self.assertEqual(f.bootstrap.take_captures(), ())
        f.run()
        self.assertEqual(len(f.launches), 2)
        self.assertEqual(f.bootstrap.take_captures(), ())

    def test_captures_have_no_per_invocation_tensor_argument_fields(self):
        f = Fixture(self)
        class Tensor:
            pass
        owners = tuple(Tensor() for _ in range(4))
        refs = [weakref.ref(x) for x in owners]
        f.arguments = owners + (128, 8, 128)
        f.run()
        captures = f.bootstrap.take_captures()
        self.assertEqual(len(captures), 1)
        # The fixture's ordinary execution records intentionally retain args;
        # clear only those fixture records before checking the new bootstrap.
        f.arguments = ()
        f.binds.clear(); f.launches.clear(); f.bound_args.clear(); f.metadata_calls.clear()
        del owners
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertFalse(any(name in vars(captures[0]) for name in ("ids", "args", "output", "pointer")))

    def test_factory_is_unwired_and_imports_no_native_runtime(self):
        root = Path(__file__).parents[1]
        text = (root / "probes/m1_route_bootstrap.py").read_text()
        imports = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {"contextlib", "dataclasses", "sys", "threading", "types",
            "m1_route_binding_adapter", "m1_route_observer", "megartx.m1_route_receipt"})
        for path in (root / "src").rglob("*.py"):
            self.assertNotIn("m1_route_bootstrap", path.read_text())

    def test_exact_shared_coordination_baseline_stays_default_off(self):
        root = Path(__file__).parents[1]
        plan = json.loads((root / "docs/route-observation-wiring-proposal.json").read_text())
        self.assertEqual(plan["base_commit"], "0c00ddd8e81fe9c1af335edf7ce57001dd31ddd0")
        for flag in ("installed", "experiment_admitted", "runtime_qualified", "default_enabled"):
            self.assertIs(plan[flag], False)
        self.assertEqual(plan["maximum_runner_invocations"], 30)
        self.assertEqual(plan["jit_lookup_slice_sha256"], cache_test._SHA)
        for item in plan["shared_sources"]:
            self.assertEqual(hashlib.sha256((root / item["path"]).read_bytes()).hexdigest(), item["sha256"])
        self.assertEqual(plan["required_checks"], ["route D2H readback", "synchronization", "native route checks"])


if __name__ == "__main__":
    unittest.main()

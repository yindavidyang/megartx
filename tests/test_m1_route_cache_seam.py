"""Disconnected CPU controls for the one-expression actual-selection candidate.

The JIT and cache-key bodies are the merged exact-source excerpts. This fixture's
cache has no observer callback: only the candidate's actual expression delta
reports a selection. No installed runtime, accelerator, compiler or wiring runs.
"""
import ast
from dataclasses import FrozenInstanceError
import hashlib
import gc
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace as NS
import unittest
import weakref


_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "m1_route_cache_seam", _ROOT / "probes" / "m1_route_cache_seam.py",
)
seam = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = seam
_SPEC.loader.exec_module(seam)
_REF_SPEC = importlib.util.spec_from_file_location(
    "cache_seam_exact_reference", _ROOT / "tests" / "fixtures" / "route_observer_reference.py",
)
reference = importlib.util.module_from_spec(_REF_SPEC)
_REF_SPEC.loader.exec_module(reference)
_SOURCE = reference.SOURCES["JITFunction.run"]
_SHA = reference.MANIFEST["slices"]["JITFunction.run"]["sha256"]
_HELPER = "__m1_route_actual_cache_selection__"


class HostileError(BaseException):
    def __str__(self):
        raise AssertionError("never format an observation or primary exception")

    def add_note(self, note):
        raise AssertionError("never annotate an observation or primary exception")


class Cache:
    """Only the upstream get operation is permitted; no synthetic observation."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.entries = {}
        self.gets = []

    def get(self, key, default):
        self.fixture.events.append("lookup")
        self.gets.append((key, default))
        self.fixture.invoke("lookup")
        return self.entries.get(key, default)

    def __iter__(self):
        raise AssertionError("do not scan the cache")

    def __getitem__(self, key):
        raise AssertionError("do not perform another lookup")

    def items(self):
        raise AssertionError("do not scan the cache")

    def keys(self):
        raise AssertionError("do not scan the cache")


class Kernel:
    run = property(reference.load("CompiledKernel.run"))

    def __init__(self, fixture):
        self.fixture = fixture
        self.function, self.packed_metadata = object(), object()
        self._run = self.launch

    @property
    def hash(self):
        raise AssertionError("kernel.hash is not the actual selection key")

    def _init_handles(self):
        self.fixture.events.append("lazy_init")
        self.fixture.invoke("lazy_init")
        self._run = self.launch

    def launch_metadata(self, *args):
        self.fixture.events.append("launch_metadata")
        self.fixture.metadata_calls.append(args)
        self.fixture.invoke("launch_metadata")
        return self.fixture.launch_metadata

    def launch(self, *args):
        self.fixture.events.append("launch")
        self.fixture.launches.append(args)
        self.fixture.invoke("launch")
        return self.fixture.launch_result


class Fixture:
    def __init__(self):
        self.events, self.binds, self.key_calls = [], [], []
        self.compiles, self.packs, self.launches, self.metadata_calls = [], [], [], []
        self.observations, self.callbacks = [], {}
        self.target, self.backend = object(), object()
        self.launch_metadata, self.launch_result = object(), object()
        self.arguments = (object(), object(), object())
        self.specialization, self.options = [("ptr", "cpu-only")], {"num_warps": 1}
        self.compile_result = None
        self.runtime = NS(debug=False, add_stages_inspection_hook=None,
                          launch_enter_hook=object(), launch_exit_hook=object())
        self.knobs = NS(runtime=self.runtime, compilation=NS(instrumentation_mode=""))
        self.key_function = reference.load("compute_cache_key", {
            "JITCallable": type("JITCallable", (), {}),
            "is_namedtuple": lambda value: isinstance(value, tuple) and hasattr(value, "_fields"),
        })
        self.key_cache = {}
        self.warm_key = self.key_function(self.key_cache, self.specialization, self.options)
        self.kernel = Kernel(self)
        self.cache = Cache(self)
        self.cache.entries[self.warm_key] = self.kernel
        self.jit = NS(debug=False, pre_run_hooks=[], used_global_vals={},
                      device_caches={7: (self.cache, self.key_cache, self.target,
                                         self.backend, self.binder)},
                      _pack_args=self.pack, _do_compile=self.compile)
        self.namespace = {
            "driver": NS(active=NS(get_current_device=self.device,
                                    get_current_stream=self.stream)),
            "knobs": self.knobs, "compute_cache_key": self.compute_key,
        }
        self.original = reference.load("JITFunction.run", self.namespace)
        self.jit.run = self.original

    def invoke(self, name):
        if name in self.callbacks:
            self.callbacks[name]()

    def device(self):
        self.events.append("device")
        self.invoke("device")
        return 7

    def stream(self, device):
        self.events.append("stream")
        assert device == 7
        self.invoke("stream")
        return 71

    def binder(self, *args, **kwargs):
        self.events.append("binder")
        self.binds.append((args, kwargs))
        self.invoke("binder")
        self.bound_args = dict(enumerate(args))
        self.actual_specialization, self.actual_options = list(self.specialization), dict(self.options)
        return self.bound_args, self.actual_specialization, self.actual_options

    def compute_key(self, cache, specialization, options):
        self.events.append("key")
        self.key_calls.append((cache, specialization, options))
        self.invoke("key")
        self.actual_key = self.key_function(cache, specialization, options)
        return self.actual_key

    def pack(self, *args):
        self.events.append("pack")
        self.packs.append(args)
        self.invoke("pack")
        return args[-1], {"signature": "cpu-only"}, {}, {}

    def compile(self, *args):
        self.events.append("compile")
        self.compiles.append(args)
        self.invoke("compile")
        return self.compile_result

    def observe(self, **event):
        self.events.append("observe")
        self.observations.append(event)
        self.invoke("observe")
        return object()  # A callback return must never replace the chosen kernel.

    def candidate(self, callback=None):
        return seam.build_cache_selection_candidate(
            _SOURCE, expected_sha256=_SHA, namespace=self.namespace,
            observe_selection=self.observe if callback is None else callback,
        )

    def call(self, run, **overrides):
        kwargs = {"grid": (1,), "warmup": False, "num_warps": 1}
        kwargs.update(overrides)
        return run(self.jit, *self.arguments, **kwargs)


def raise_error(error):
    def raise_it():
        raise error
    return raise_it


class CacheSelectionSeamTests(unittest.TestCase):
    def assert_original_calls_once(self, fixture, *, launches=1, compiles=0):
        self.assertEqual(len(fixture.binds), 1)
        self.assertEqual(len(fixture.key_calls), 1)
        self.assertEqual(len(fixture.cache.gets), 1)
        self.assertEqual(len(fixture.observations), 1)
        self.assertEqual(len(fixture.launches), launches)
        self.assertEqual(len(fixture.compiles), compiles)
        self.assertIs(fixture.cache.gets[0][0], fixture.actual_key)
        self.assertIsNone(fixture.cache.gets[0][1])

    def test_exact_source_provenance_and_independent_ast_restoration(self):
        self.assertTrue(reference.validate_sources())
        self.assertEqual(_SHA, seam.SUPPORTED_JIT_RUN_SHA256)
        f = Fixture()
        candidate = f.candidate()
        proof = candidate.proof
        self.assertEqual(proof.source_ast_sha256, proof.restored_ast_sha256)
        self.assertNotEqual(proof.source_ast_sha256, proof.candidate_ast_sha256)
        self.assertEqual(proof.replacement_count, 1)
        self.assertEqual(proof.evidence_kind, "reference-source-only")
        self.assertFalse(proof.installed)
        self.assertFalse(proof.loaded_bindings_verified)
        tree = ast.parse(candidate.transformed_source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == _HELPER]
        self.assertEqual(len(calls), 1)
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and node.value is calls[0])
        assignment.value = calls[0].args[0]
        self.assertEqual(ast.dump(tree), ast.dump(ast.parse(textwrap.dedent(_SOURCE))))
        self.assertEqual(inspect.signature(candidate.run), inspect.signature(f.original))
        self.assertEqual(candidate.run.__code__.co_firstlineno, 708)
        with self.assertRaises(FrozenInstanceError):
            proof.installed = True

    def test_factories_do_not_install_or_execute_and_namespace_is_unchanged(self):
        f = Fixture()
        globals_before, cache_before = dict(f.namespace), dict(f.cache.entries)
        candidate = f.candidate()
        self.assertIs(f.jit.run, f.original)
        self.assertEqual(f.namespace, globals_before)
        self.assertEqual(f.cache.entries, cache_before)
        self.assertEqual(f.events, [])
        self.assertNotIn(_HELPER, f.namespace)
        self.assertEqual(candidate.callback_failures, 0)
        self.assertIsNone(candidate.last_callback_failure)

    def test_actual_cache_key_selected_and_compilation_locals_are_retained_by_identity(self):
        f = Fixture()
        result = f.call(f.candidate().run)
        self.assertIs(result, f.kernel)
        self.assert_original_calls_once(f)
        event = f.observations[0]
        for name, expected in (("cache", f.cache), ("key", f.actual_key), ("selected", f.kernel),
                               ("target", f.target), ("specialization", f.actual_specialization),
                               ("options", f.actual_options)):
            self.assertIs(event[name], expected, name)
        self.assertEqual(f.events, ["device", "stream", "binder", "key", "lookup", "observe",
                                    "launch_metadata", "launch"])
        launch = f.launches[0]
        self.assertEqual(launch[:4], (1, 1, 1, 71))
        self.assertIs(launch[4], f.kernel.function)
        self.assertIs(launch[5], f.kernel.packed_metadata)
        self.assertIs(launch[6], f.launch_metadata)
        self.assertIs(launch[7], f.runtime.launch_enter_hook)
        self.assertIs(launch[8], f.runtime.launch_exit_hook)
        for actual, expected in zip(launch[9:], f.arguments):
            self.assertIs(actual, expected)
        self.assertIsNot(result, f.launch_result)

    def test_original_driver_binder_key_lookup_launch_trace_is_preserved(self):
        for run_candidate in (False, True):
            f = Fixture()
            run = f.candidate().run if run_candidate else f.original
            self.assertIs(f.call(run, debug=True), f.kernel)
            self.assertEqual([event for event in f.events if event != "observe"],
                             ["device", "stream", "binder", "key", "lookup", "launch_metadata", "launch"])
            self.assertEqual(f.binds[0][1], {"num_warps": 1, "debug": True, "instrumentation_mode": ""})

    def test_cache_miss_async_compile_none_returns_none_without_retry(self):
        f = Fixture()
        f.cache.entries.clear()
        self.assertIsNone(f.call(f.candidate().run))
        self.assert_original_calls_once(f, launches=0, compiles=1)
        self.assertIsNone(f.observations[0]["selected"])
        self.assertEqual(f.events[-3:], ["observe", "pack", "compile"])
        self.assertIs(f.compiles[0][0], f.actual_key)

    def test_cold_compiled_result_does_not_rewrite_cold_observation(self):
        f = Fixture()
        f.cache.entries.clear()
        f.compile_result = f.kernel
        self.assertIs(f.call(f.candidate().run), f.kernel)
        self.assert_original_calls_once(f, compiles=1)
        self.assertIsNone(f.observations[0]["selected"])
        self.assertEqual(f.events.count("lookup"), 1)

    def test_unknown_specialization_uses_original_cold_path(self):
        f = Fixture()
        f.specialization.append(("new", "unqualified"))
        f.options["num_warps"] = 4
        self.assertIsNone(f.call(f.candidate().run))
        self.assert_original_calls_once(f, launches=0, compiles=1)
        self.assertIsNone(f.observations[0]["selected"])
        self.assertEqual(f.observations[0]["options"], {"num_warps": 4})

    def test_each_call_observes_current_actual_cache_target_and_kernel(self):
        f = Fixture()
        candidate = f.candidate()
        self.assertIs(f.call(candidate.run), f.kernel)
        original_cache, original_kernel, original_target = f.cache, f.kernel, f.target
        new_cache, new_kernel, new_target = Cache(f), Kernel(f), object()
        new_cache.entries[f.warm_key] = new_kernel
        f.jit.device_caches[7] = (new_cache, f.key_cache, new_target, f.backend, f.binder)
        self.assertIs(f.call(candidate.run), new_kernel)
        self.assertEqual(len(original_cache.gets), 1)
        self.assertEqual(len(new_cache.gets), 1)
        self.assertEqual(len(f.binds), 2)
        self.assertEqual(len(f.key_calls), 2)
        self.assertEqual(len(f.observations), 2)
        for event, cache, kernel, target in (
            (f.observations[0], original_cache, original_kernel, original_target),
            (f.observations[1], new_cache, new_kernel, new_target),
        ):
            self.assertIs(event["cache"], cache)
            self.assertIs(event["selected"], kernel)
            self.assertIs(event["target"], target)

    def test_unknown_selected_object_keeps_original_failure_after_observation(self):
        f = Fixture()
        unknown = object()
        f.cache.entries[f.warm_key] = unknown
        with self.assertRaises(AttributeError):
            f.call(f.candidate().run)
        self.assert_original_calls_once(f, launches=0)
        self.assertIs(f.observations[0]["selected"], unknown)
        self.assertEqual(f.compiles, [])

    def test_stage_hook_specialization_is_the_actual_post_hook_value(self):
        f = Fixture()
        def hook():
            f.events.append("stages")
            return "unused", "stage-hash"
        f.runtime.add_stages_inspection_hook = hook
        self.assertIsNone(f.call(f.candidate().run))
        self.assert_original_calls_once(f, launches=0, compiles=1)
        self.assertEqual(f.events.count("stages"), 1)
        self.assertEqual(f.observations[0]["specialization"][-1], '("custom_pipeline", stage-hash)')

    def test_warmup_and_callable_grid_keep_original_paths(self):
        f = Fixture()
        self.assertIs(f.call(f.candidate().run, warmup=True, grid=None), f.kernel)
        self.assert_original_calls_once(f, launches=0)
        f = Fixture()
        def grid(bound):
            f.events.append("grid")
            self.assertIs(bound, f.bound_args)
            return (2, 3, 4)
        self.assertIs(f.call(f.candidate().run, grid=grid), f.kernel)
        self.assert_original_calls_once(f)
        self.assertEqual(f.launches[0][:3], (2, 3, 4))
        self.assertEqual(f.events.count("grid"), 1)

    def test_uninitialized_kernel_keeps_existing_lazy_path_once(self):
        f = Fixture()
        f.kernel._run = None
        self.assertIs(f.call(f.candidate().run), f.kernel)
        self.assert_original_calls_once(f)
        self.assertEqual(f.events.count("lazy_init"), 1)

    def test_failures_before_selection_never_call_observer_or_retry(self):
        for stage in ("device", "stream", "binder", "key", "lookup"):
            with self.subTest(stage=stage):
                f, primary = Fixture(), HostileError()
                f.callbacks[stage] = raise_error(primary)
                candidate = f.candidate()
                with self.assertRaises(HostileError) as caught:
                    f.call(candidate.run)
                self.assertIs(caught.exception, primary)
                self.assertEqual(f.events.count(stage), 1)
                self.assertIsNone(primary.__context__)
                self.assertEqual(f.observations, [])
                self.assertEqual(f.launches, [])
                self.assertEqual(candidate.callback_failures, 0)

    def test_callback_failure_keeps_async_none_return_and_cold_observation(self):
        f, secondary = Fixture(), HostileError()
        f.cache.entries.clear()
        f.callbacks["observe"] = raise_error(secondary)
        candidate = f.candidate()
        self.assertIsNone(f.call(candidate.run))
        self.assert_original_calls_once(f, launches=0, compiles=1)
        self.assertIsNone(f.observations[0]["selected"])
        self.assertEqual(candidate.callback_failures, 1)
        self.assertEqual(candidate.last_callback_failure, "observation_callback_failed")

    def test_observer_failures_preserve_successful_delegation_and_return_identity(self):
        for observer_error in (RuntimeError("callback"), KeyboardInterrupt(), SystemExit(), HostileError()):
            with self.subTest(kind=type(observer_error).__name__):
                f = Fixture()
                f.callbacks["observe"] = raise_error(observer_error)
                candidate = f.candidate()
                self.assertIs(f.call(candidate.run), f.kernel)
                self.assert_original_calls_once(f)
                self.assertEqual(candidate.callback_failures, 1)
                self.assertEqual(candidate.last_callback_failure, "observation_callback_failed")
                # A failed earlier callback neither disables ordinary execution
                # nor creates a synthetic successful evidence receipt later.
                self.assertIs(f.call(candidate.run), f.kernel)
                self.assertEqual(candidate.callback_failures, 2)
                self.assertEqual(len(f.cache.gets), 2)

    def test_callback_signature_failure_is_diagnostic_not_a_changed_run_signature(self):
        f = Fixture()
        candidate = f.candidate(callback=lambda: None)
        self.assertIs(f.call(candidate.run), f.kernel)
        self.assertEqual(candidate.callback_failures, 1)
        self.assertEqual(candidate.last_callback_failure, "observation_callback_failed")
        self.assertEqual(len(f.binds), 1)
        self.assertEqual(len(f.cache.gets), 1)
        self.assertEqual(len(f.launches), 1)
        f = Fixture()
        candidate = f.candidate()
        with self.assertRaises(TypeError):
            candidate.run(f.jit, *f.arguments)  # Required grid/warmup stay required.
        self.assertEqual(f.events, [])

    def test_callback_diagnostics_release_exception_type_traceback_and_invocation_owners(self):
        f = Fixture()
        exceptions = []

        class Storage:
            pass

        class Tensor:
            def __init__(self):
                self.storage = Storage()

        def temporary_hostile_failure():
            class OpaqueError(BaseException):
                def __getattribute__(self, name):
                    raise AssertionError("callback exception attributes must never be read")

                def __str__(self):
                    raise AssertionError("callback exception must never be formatted")

            error = OpaqueError()
            exceptions.append((weakref.ref(error), weakref.ref(OpaqueError)))
            raise error

        f.callbacks["observe"] = temporary_hostile_failure
        candidate = f.candidate()

        def invoke_with_temporary_owners():
            tensor = Tensor()
            refs = (weakref.ref(tensor), weakref.ref(tensor.storage))
            f.arguments = (tensor,)
            self.assertIs(f.call(candidate.run), f.kernel)
            # Remove only the CPU fixture's deliberate call-history ownership.
            # Keep the candidate, callback, JIT, cache and fixture alive: retaining
            # the exception traceback would still keep run's args and this frame.
            f.arguments = ()
            f.binds.clear()
            f.bound_args.clear()
            f.launches.clear()
            f.metadata_calls.clear()
            return refs

        for count in (1, 2):
            tensor_ref, storage_ref = invoke_with_temporary_owners()
            gc.collect()
            self.assertIsNone(tensor_ref())
            self.assertIsNone(storage_ref())
            for error_ref, error_type_ref in exceptions:
                self.assertIsNone(error_ref())
                self.assertIsNone(error_type_ref())
            self.assertEqual(candidate.callback_failures, count)
            self.assertEqual(candidate.last_callback_failure, "observation_callback_failed")
            self.assertEqual(vars(candidate._failure), {"count": count})

    def test_observer_failure_never_masks_primary_compile_pack_or_launch_error(self):
        for stage in ("pack", "compile", "launch_metadata", "launch", "lazy_init"):
            with self.subTest(stage=stage):
                f, primary, secondary = Fixture(), HostileError(), HostileError()
                if stage in ("pack", "compile"):
                    f.cache.entries.clear()
                if stage == "lazy_init":
                    f.kernel._run = None
                f.callbacks[stage] = raise_error(primary)
                f.callbacks["observe"] = raise_error(secondary)
                candidate = f.candidate()
                with self.assertRaises(HostileError) as caught:
                    f.call(candidate.run)
                self.assertIs(caught.exception, primary)
                self.assertEqual(candidate.last_callback_failure, "observation_callback_failed")
                self.assertIsNone(primary.__context__)
                self.assertEqual(candidate.callback_failures, 1)
                self.assertEqual(len(f.binds), 1)
                self.assertEqual(len(f.key_calls), 1)
                self.assertEqual(len(f.cache.gets), 1)
                self.assertEqual(len(f.observations), 1)
                self.assertEqual(f.events.count(stage), 1)

    def test_original_pre_hook_global_validation_and_grid_failure_remain_active(self):
        f, primary = Fixture(), HostileError()
        f.jit.pre_run_hooks = [lambda *args, **kwargs: raise_error(primary)()]
        with self.assertRaises(HostileError) as caught:
            f.call(f.candidate().run)
        self.assertIs(caught.exception, primary)
        self.assertEqual(f.binds, [])
        f = Fixture()
        f.jit.used_global_vals = {("mutated", 0): (1, {"mutated": 2})}
        with self.assertRaisesRegex(RuntimeError, "Global variable mutated has changed"):
            f.call(f.candidate().run)
        self.assert_original_calls_once(f, launches=0)
        f, primary = Fixture(), HostileError()
        with self.assertRaises(HostileError) as caught:
            f.call(f.candidate().run, grid=lambda bound: raise_error(primary)())
        self.assertIs(caught.exception, primary)
        self.assert_original_calls_once(f, launches=0)

    def test_source_mutations_and_self_attested_new_digests_are_rejected(self):
        f = Fixture()
        mutations = (
            _SOURCE + "\n", _SOURCE.replace("grid, warmup", "grid, warmup=False"),
            _SOURCE.replace("kernel_cache.get(key, None)", "kernel_cache.get(key)"),
            _SOURCE.replace("return kernel", "return None"),
            _SOURCE.replace("binder(*args, **kwargs)", "binder(*args, **kwargs); binder(*args, **kwargs)"),
        )
        for source in mutations:
            for digest in (_SHA, hashlib.sha256(source.encode()).hexdigest()):
                with self.subTest(digest=digest):
                    with self.assertRaises(seam.CacheSeamSourceError):
                        seam.build_cache_selection_candidate(source, expected_sha256=digest,
                            namespace=f.namespace, observe_selection=f.observe)
        self.assertEqual(f.events, [])

    def test_restoration_rejects_every_additional_executable_delta(self):
        f = Fixture()
        source = f.candidate().transformed_source
        mutations = (
            source.replace("grid, warmup", "grid, warmup=False"),
            source.replace("return kernel", "return None"),
            source.replace("kernel_cache.get(key, None)", "kernel_cache.get(key)"),
            source.replace(", kernel_cache, key, target, specialization, options)",
                           ", kernel_cache, key, target, options, specialization)"),
            source.replace(_HELPER + "(", "another_helper("),
            source.replace("kernel = " + _HELPER, "unused = " + _HELPER),
            source + "unrelated = 1\n",
            source.replace("bound_args, specialization, options = binder(*args, **kwargs)",
                           "bound_args, specialization, options = binder(*args, **kwargs)\n    binder(*args, **kwargs)"),
            source.replace("kernel = " + _HELPER, "kernel = " + _HELPER, 1)
                  + "extra = " + _HELPER + "(kernel_cache.get(key, None), kernel_cache, key, target, specialization, options)\n",
        )
        for changed in mutations:
            with self.subTest(changed=changed[-80:]):
                self.assertNotEqual(changed, source)
                with self.assertRaises(seam.CacheSeamSourceError):
                    seam.verify_cache_selection_restoration(_SOURCE, changed, expected_sha256=_SHA)

    def test_private_helper_collision_and_missing_cpu_namespace_are_rejected(self):
        f = Fixture()
        for namespace in ({**f.namespace, _HELPER: object()}, {}, []):
            with self.subTest(namespace=type(namespace).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    seam.build_cache_selection_candidate(_SOURCE, expected_sha256=_SHA,
                        namespace=namespace, observe_selection=f.observe)
        with self.assertRaises(TypeError):
            seam.build_cache_selection_candidate(_SOURCE, expected_sha256=_SHA,
                namespace=f.namespace, observe_selection=None)
        self.assertEqual(f.events, [])

    def test_no_accelerator_imports_or_production_wiring(self):
        source = (_ROOT / "probes" / "m1_route_cache_seam.py").read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"__future__", "ast", "dataclasses", "hashlib", "textwrap", "types"})
        for path in (_ROOT / "src").rglob("*.py"):
            self.assertNotIn("m1_route_cache_seam", path.read_text(), str(path))
        for root in ("torch", "triton", "vllm"):
            # This suite never imports any upstream package. Existing test
            # processes may already contain one, so inspect only module globals.
            self.assertNotIn(root, seam.__dict__)


if __name__ == "__main__":
    unittest.main()

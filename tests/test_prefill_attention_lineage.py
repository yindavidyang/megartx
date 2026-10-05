"""Current exact source admission; CPU fixtures cannot attest an installed host."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from megartx import prefill_attention_lineage as attention
from megartx import prefill_attention_native_plan as native
from megartx import prefill_storage_plan as storage
from megartx import prefill_diagnostic_plan as legacy
from megartx import m1_map_borrow_lineage as borrow
from megartx.controlled_kv_capture import CONFIG_SHA256

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = {'config_sha256': CONFIG_SHA256, 'index_sha256': 'a' * 64,
              'shard_stats': {'synthetic.safetensors': {'size': 1, 'mtime_ns': 1}}}


class AttentionLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name) / 'source'
        shutil.copytree(ROOT, cls.root, ignore=shutil.ignore_patterns('.git', '__pycache__'))

    def validate(self):
        return attention.validate_source_catalog(self.root,
            {name: storage.file_sha(self.root / name) for name in native.SOURCES})

    def test_current_new_plan_freezes_validates_and_unknown_installation_blocks_clearance(self):
        plan = native.freeze_plan(list(range(2048)), '3' * 40, ROOT, CHECKPOINT)
        self.assertIs(native.validate_plan(plan, ROOT), plan)
        self.assertEqual(plan['installed_sources'], dict.fromkeys(native.INSTALLED_CONTRACT))
        self.assertEqual(plan['source_hashes'], {p: native.file_sha(ROOT / p) for p in native.SOURCES})
        with self.assertRaisesRegex(ValueError, 'unknown'):
            native.require_clearance('/must-not-read-a-clearance', plan)
        origins = native.verify_adapter_sources(plan)
        self.assertEqual(origins['megartx.prefill_attention_lineage']['sha256'], storage.ATTENTION_HELPER_SHA256)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('vllm', sys.modules)

    def test_current_default_and_storage_plans_keep_actual_sources_and_adapter_origins(self):
        for contract in (legacy, storage):
            plan = contract.freeze_plan(list(range(2048)), '3' * 40, ROOT, CHECKPOINT)
            self.assertIs(contract.validate_plan(plan, ROOT), plan)
            self.assertEqual(plan['source_hashes'], {p: contract.file_sha(ROOT / p) for p in contract.SOURCES})
            origins = contract.verify_adapter_sources(plan)
            self.assertIn('megartx.vllm_scale_plugin', origins)
            if contract is storage:
                self.assertIn('megartx.prefill_attention_lineage', origins)
                self.assertIn('megartx.m1_map_borrow_lineage', origins)

    def test_exact_gate_removal_restores_previous_validator_and_pr36_parent(self):
        source = (ROOT / attention.VALIDATOR_SOURCE).read_text()
        parent = attention.parent_validator_source(source)
        self.assertEqual(hashlib.sha256(parent.encode()).hexdigest(), attention.PARENT_VALIDATOR_SHA256)
        previous = borrow.parent_validator_source(parent)
        self.assertEqual(hashlib.sha256(previous.encode()).hexdigest(),
                         '42eab7949b1b2b0a64892c4efe01084da50339d8d3bbc446d227694153e7d580')
        for before, after in (("'max_contexts': 1", "'max_contexts': 2"),
                              ('    # BEGIN ATTENTION SOURCE ADMISSION\n', ''),
                              ('from . import prefill_attention_lineage as attention_lineage\n', '')):
            bad = source.replace(before, after)
            try:
                restored = attention.parent_validator_source(bad)
            except ValueError:
                continue
            self.assertNotEqual(hashlib.sha256(restored.encode()).hexdigest(), attention.PARENT_VALIDATOR_SHA256)

    def test_catalog_retains_frozen_runtime_and_exact_historical_endpoint_provenance(self):
        value = attention.load_catalog(ROOT)
        rows = value['source_records']
        self.assertEqual([r['path'] for r in rows], sorted({r['path'] for r in rows}))
        for row in rows:
            name = row['path']
            if name.startswith(('src/', 'scripts/')) and name != attention.VALIDATOR_SOURCE:
                self.assertEqual(row['current_sha256'], row['frozen_sha256'], name)
            if name.startswith('docs/') and name.endswith('.json'):
                self.assertEqual(row['current_sha256'], row['frozen_sha256'], name)
            for old in ('parent_sha256', 'frozen_sha256'):
                if row[old] is not None:
                    self.assertRegex(row[old], '^[0-9a-f]{64}$')
        for flag in ('gpu_authorized', 'gpu_executed', 'numerical_qualified', 'quality_qualified', 'performance_qualified'):
            self.assertIs(value[flag], False)
        self.assertEqual(value['installed_sources'], dict.fromkeys(native.INSTALLED_CONTRACT))
        self.assertEqual(attention.map_borrow_catalog(ROOT), json.loads((ROOT / borrow.CATALOG_SOURCE).read_text()))

    def test_catalog_unknown_missing_duplicate_reordered_and_repinned_endpoints_reject(self):
        path = self.root / attention.CATALOG_SOURCE
        original = path.read_bytes()
        try:
            for mutation in ('missing', 'unknown', 'duplicate', 'reordered', 'parent', 'frozen', 'current',
                             'qualification', 'installation', 'inventory', 'extra', 'duplicate_json'):
                value = json.loads(original)
                rows = value['source_records']
                if mutation == 'missing': rows.pop()
                elif mutation == 'unknown': rows[0]['path'] = 'unknown.py'
                elif mutation == 'duplicate': rows.append(rows[0])
                elif mutation == 'reordered': rows.reverse()
                elif mutation in ('parent', 'frozen', 'current'): rows[0][mutation + '_sha256'] = '0' * 64
                elif mutation == 'qualification': value['gpu_authorized'] = True
                elif mutation == 'installation': value['installed_sources'] = native.INSTALLED_CONTRACT
                elif mutation == 'inventory': value['plan_sources']['native_attention'].pop()
                else: value['accept_any_parent'] = True
                raw = json.dumps(value).encode()
                if mutation == 'duplicate_json': raw = b'{"schema":"foreign",' + original.lstrip()[1:]
                path.write_bytes(raw)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.validate()
        finally:
            path.write_bytes(original)

    def test_changed_source_plus_repinned_catalog_and_plan_cannot_be_admitted(self):
        catalog_path = self.root / attention.CATALOG_SOURCE
        original_catalog = catalog_path.read_bytes()
        for name in ('scripts/run_scale_validation.py', 'src/megartx/vllm_scale_plugin.py',
                     'src/megartx/prefill_storage_native.py', 'src/megartx/prefill_attention_hooks.py',
                     'scripts/prefill_owned_processes.py', 'src/megartx/m1_map_borrow_lineage.py',
                     'tests/test_prefill_attention_lineage.py'):
            path = self.root / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n')
                value = json.loads(original_catalog)
                row = next(r for r in value['source_records'] if r['path'] == name)
                row['current_sha256'] = storage.file_sha(path)
                catalog_path.write_text(json.dumps(value))
                with self.subTest(path=name), self.assertRaises(ValueError):
                    self.validate()
            finally:
                path.write_bytes(original)
                catalog_path.write_bytes(original_catalog)

    def test_actual_parent_runtime_endpoint_mixed_into_current_tree_rejects(self):
        manifest = json.loads((ROOT / 'tests/fixtures/prefill-attention-base/manifest.json').read_text())
        for name, record in manifest['files'].items():
            path = self.root / name
            original = path.read_bytes()
            try:
                path.write_bytes((ROOT / 'tests/fixtures/prefill-attention-base' / record['fixture']).read_bytes())
                with self.subTest(parent_endpoint=name), self.assertRaises(ValueError):
                    self.validate()
            finally:
                path.write_bytes(original)

    def test_omitted_unknown_and_wrong_plan_endpoints_reject(self):
        hashes = {p: storage.file_sha(ROOT / p) for p in native.SOURCES}
        for name in native.SOURCES:
            missing = dict(hashes); missing.pop(name)
            with self.subTest(omitted=name), self.assertRaises(ValueError):
                attention.validate_source_catalog(ROOT, missing)
        for changed in ({**hashes, 'unknown.py': '0' * 64},
                        {**hashes, attention.HELPER_SOURCE: '0' * 64}):
            with self.assertRaises(ValueError): attention.validate_source_catalog(ROOT, changed)

    def test_terminal_resolver_rejects_wrong_parent_unknown_and_stale_endpoints(self):
        value = attention.load_catalog(ROOT)
        name = 'scripts/run_scale_validation.py'
        row = next(r for r in value['source_records'] if r['path'] == name)
        self.assertEqual(attention.terminal_sha(value, name, row['parent_sha256']), storage.file_sha(ROOT / name))
        for bad in ('0' * 64, row['frozen_sha256'], None):
            with self.assertRaises(ValueError): attention.terminal_sha(value, name, bad)
        with self.assertRaises(ValueError): attention.terminal_sha(value, 'unknown.py', '0' * 64)

    def test_helper_actual_origin_and_foreign_same_bytes_cannot_pass(self):
        for helper in (attention, borrow):
            with patch.object(helper, '__spec__', SimpleNamespace(origin='/wrong/helper.py')):
                with self.assertRaisesRegex(ValueError, 'helper origin'): self.validate()
            foreign = Path(self.temp.name) / 'foreign.py'
            foreign.write_bytes(Path(helper.__file__).read_bytes())
            with patch.object(helper, '__file__', str(foreign)), \
                    patch.object(helper, '__spec__', SimpleNamespace(origin=str(foreign))):
                with self.assertRaisesRegex(ValueError, 'helper origin'): self.validate()

    def test_helper_pin_is_not_free_to_repin_in_root_validator(self):
        path = self.root / attention.VALIDATOR_SOURCE
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(storage.ATTENTION_HELPER_SHA256.encode(), b'0' * 64))
            with self.assertRaises(ValueError): self.validate()
        finally:
            path.write_bytes(original)

    def test_symlinked_current_endpoint_rejects(self):
        for name in (attention.CATALOG_SOURCE, attention.HELPER_SOURCE, 'scripts/run_scale_validation.py'):
            path = self.root / name
            original = path.read_bytes()
            try:
                path.unlink(); path.symlink_to(ROOT / name)
                with self.subTest(path=name), self.assertRaises(ValueError): self.validate()
            finally:
                path.unlink(); path.write_bytes(original)

    def test_cpu_plan_import_has_no_runtime_dependencies(self):
        result = subprocess.run([sys.executable, '-S', '-c',
            "import megartx.prefill_attention_native_plan, sys; "
            "assert not any(n.split('.')[0] in {'torch','vllm','numpy','requests'} for n in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

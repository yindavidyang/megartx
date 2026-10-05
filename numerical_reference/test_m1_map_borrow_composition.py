"""CPU-only exact two-parent composition and source-admission controls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('map_borrow_lineage',
    ROOT / 'src/megartx/m1_map_borrow_lineage.py')
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class MapBorrowCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name) / 'repo'
        shutil.copytree(ROOT, cls.root, ignore=shutil.ignore_patterns('.git', '__pycache__'))
        sys.path.insert(0, str(ROOT / 'src'))
        try:
            from megartx import prefill_storage_plan
            from megartx import prefill_attention_lineage
            from megartx.controlled_kv_capture import CONFIG_SHA256
        finally:
            sys.path.pop(0)
        cls.storage = prefill_storage_plan
        cls.attention = prefill_attention_lineage
        cls.config_sha256 = CONFIG_SHA256

    def validate(self, root=None):
        root = root or self.root
        hashes = {name: self.storage.file_sha(root / name) for name in self.storage.SOURCES}
        return self.storage.validate_source_catalog(root, hashes)

    def test_current_root_freezes_and_validates_exact_complete_plan(self):
        storage = self.storage
        plan = storage.freeze_plan(list(range(2048)), '3' * 40, ROOT, {
            'config_sha256': self.config_sha256, 'index_sha256': 'a' * 64,
            'shard_stats': {'synthetic.safetensors': {'size': 1, 'mtime_ns': 1}}})
        self.assertIs(storage.validate_plan(plan, ROOT), plan)
        origins = storage.verify_adapter_sources(plan)
        self.assertEqual(origins['megartx.m1_map_borrow_lineage']['sha256'],
                         plan['source_hashes'][lineage.HELPER_SOURCE])
        for name in (lineage.CATALOG_SOURCE, lineage.HELPER_SOURCE):
            self.assertEqual(plan['source_hashes'][name], sha((ROOT / name).read_bytes()))
            bad = copy.deepcopy(plan)
            del bad['source_hashes'][name]
            bad['plan_sha256'] = storage.digest({k: v for k, v in bad.items() if k != 'plan_sha256'})
            with self.subTest(missing=name), self.assertRaises(ValueError):
                storage.validate_plan(bad, ROOT)
            hashes = {**plan['source_hashes'], name: '0' * 64}
            with self.subTest(wrong_hash=name), self.assertRaises(ValueError):
                storage.validate_source_catalog(ROOT, hashes)

    def test_validator_restores_exact_main_without_touching_execution(self):
        source = self.attention.parent_validator_source((ROOT / 'src/megartx/prefill_storage_plan.py').read_text())
        expected = '42eab7949b1b2b0a64892c4efe01084da50339d8d3bbc446d227694153e7d580'
        self.assertEqual(sha(lineage.parent_validator_source(source).encode()), expected)
        for marker in ('    # BEGIN MAP-BORROW SOURCE ADMISSION\n',
                       'from . import m1_map_borrow_lineage as map_borrow\n'):
            with self.subTest(duplicate=marker), self.assertRaises(ValueError):
                lineage.parent_validator_source(source.replace(marker, marker + marker))
        for before, after in (("'max_contexts': 1", "'max_contexts': 2"),
                              ("'max_retries': 0", "'max_retries': 1"),
                              ("        _source_file(root, relative, expected)", "        pass")):
            changed = source.replace(before, after)
            self.assertNotEqual(changed, source)
            self.assertNotEqual(sha(lineage.parent_validator_source(changed).encode()), expected)

    def test_borrow_parent_ledger_and_both_terminal_paths_are_preserved(self):
        attention_catalog = self.attention.load_catalog(ROOT)
        value = self.attention.map_borrow_catalog(ROOT)
        rows = {record['path']: record for record in value['superseded_sources']}
        borrow = json.loads((ROOT / 'docs/evidence/m1-invocation-regions-source-pins.json').read_text())
        descriptor_path = ROOT / 'docs/evidence/m1-tma-descriptor-source-pins.json'
        descriptor = json.loads(descriptor_path.read_text())
        self.assertEqual(borrow['schema'], 'megartx-m1-invocation-regions-source-pins-v1')
        self.assertEqual(borrow['parent_head'], 'aa36de98af05fff7e1e3862fe35ec30e2418f8db')
        self.assertEqual(borrow['parent_tree'], '2b9165ad156503850f0ee81a38288ae180dc0ab2')
        self.assertEqual(borrow['parent_ledger_sha256'], sha(descriptor_path.read_bytes()))
        for field in ('native_build_verified', 'gpu_execution_verified', 'quality_qualified',
                      'graphs_qualified', 'performance_qualified'):
            self.assertIs(borrow[field], False)
        self.assertEqual(set(borrow['preserved_historical_ledgers']),
            set(descriptor['preserved_historical_ledgers']) | {'docs/evidence/m1-tma-descriptor-source-pins.json'})
        for name, expected in borrow['preserved_historical_ledgers'].items():
            self.assertEqual(sha((ROOT / name).read_bytes()), expected, name)
        changes = {record['path']: record for record in borrow['superseded_sources']}
        self.assertEqual(len(changes), len(borrow['superseded_sources']))
        self.assertEqual(set(changes), lineage.BORROW_SOURCES | {'numerical_reference/test_m1_preparation_reference.py'})
        common = {record['path']: record['current_sha256'] for record in descriptor['superseded_sources']}
        common.update(descriptor['added_source_hashes'])
        for name, record in changes.items():
            self.assertEqual(set(record), {'path', 'prior_sha256', 'current_sha256'})
            self.assertEqual(record['prior_sha256'], common[name], name)
            self.assertEqual(record['current_sha256'], rows[name]['borrow_parent_sha256'], name)
            self.assertEqual(self.attention.terminal_sha(attention_catalog, name,
                lineage.terminal_sha(value, name, record['current_sha256'], 'borrow_parent_sha256')),
                             sha((ROOT / name).read_bytes()), name)
        main = json.loads((ROOT / 'docs/prefill/current-main-source-reconciliation.json').read_text())
        for record in main['superseded_sources']:
            self.assertEqual(self.attention.terminal_sha(attention_catalog, record['path'],
                lineage.terminal_sha(value, record['path'], record['current_sha256'])),
                             sha((ROOT / record['path']).read_bytes()), record['path'])
        self.assertEqual(set(borrow['added_source_hashes']), {'probes/m1_invocation_regions_flow_test.cpp',
            'numerical_reference/test_m1_invocation_regions.py', 'docs/m1-invocation-regions.md',
            'docs/evidence/m1-invocation-regions-cpu-proof.json'})
        for name, expected in borrow['added_source_hashes'].items():
            self.assertEqual(sha((ROOT / name).read_bytes()), expected, name)

    def test_original_borrow_ledger_negative_controls_remain_effective(self):
        ledger = ROOT / 'docs/evidence/m1-invocation-regions-source-pins.json'
        read_text = Path.read_text
        for mutation in ('prior_sha256', 'current_sha256', 'parent_ledger_sha256', 'parent_head', 'parent_tree',
                         'unknown_source', 'removed_source', 'duplicate_source', 'historical_ledger', 'gpu_execution_verified'):
            def changed_text(path, *args, **kwargs):
                data = read_text(path, *args, **kwargs)
                if path != ledger:
                    return data
                value = json.loads(data)
                if mutation in ('prior_sha256', 'current_sha256'):
                    next(row for row in value['superseded_sources'] if row['path'] == 'probes/m1_live_bridge.cu')[mutation] = '0' * 64
                elif mutation in ('parent_ledger_sha256', 'parent_head', 'parent_tree'):
                    value[mutation] = '0' * len(value[mutation])
                elif mutation == 'unknown_source': value['added_source_hashes']['unknown.cuh'] = '0' * 64
                elif mutation == 'removed_source': value['superseded_sources'].pop()
                elif mutation == 'duplicate_source': value['superseded_sources'].append(value['superseded_sources'][0])
                elif mutation == 'historical_ledger':
                    value['preserved_historical_ledgers']['docs/evidence/m1-tma-descriptor-source-pins.json'] = '0' * 64
                else: value[mutation] = True
                return json.dumps(value)
            with self.subTest(mutation=mutation), patch.object(Path, 'read_text', changed_text), \
                    self.assertRaises((ValueError, AssertionError)):
                self.test_borrow_parent_ledger_and_both_terminal_paths_are_preserved()

    def test_terminal_catalog_inventory_parents_and_flags_are_closed(self):
        path = self.root / lineage.CATALOG_SOURCE
        original = path.read_bytes()
        try:
            for mutation in (*lineage.PARENTS, *lineage.QUALIFICATIONS, 'missing', 'unknown', 'duplicate',
                             'reordered', 'old_endpoint', 'new_endpoint', 'parent_only', 'extra_field'):
                value = json.loads(original)
                records = value['superseded_sources']
                if mutation in lineage.PARENTS: value[mutation] = '0' * 40
                elif mutation in lineage.QUALIFICATIONS: value[mutation] = True
                elif mutation == 'missing': records.pop()
                elif mutation == 'unknown': records[0]['path'] = 'unknown.py'
                elif mutation == 'duplicate': records.append(dict(records[0]))
                elif mutation == 'reordered': records.reverse()
                elif mutation == 'old_endpoint': records[0]['main_parent_sha256'] = '0' * 64
                elif mutation == 'new_endpoint': records[0]['current_sha256'] = '0' * 64
                elif mutation == 'parent_only':
                    row = next(row for row in records if row['path'] == 'probes/m1_live_bridge.cu')
                    row['current_sha256'] = row['main_parent_sha256']
                else: value['accept_any_parent'] = True
                path.write_text(json.dumps(value))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.validate()
        finally:
            path.write_bytes(original)

    def test_repinned_checker_validator_helper_and_proof_cannot_pass(self):
        catalog_path = self.root / lineage.CATALOG_SOURCE
        original_catalog = catalog_path.read_bytes()
        names = ('numerical_reference/test_m1_preparation_reference.py',
                 'numerical_reference/test_m1_tma_descriptor_contract.py',
                 'numerical_reference/test_m1_map_borrow_composition.py',
                 'src/megartx/prefill_storage_plan.py', lineage.HELPER_SOURCE,
                 'probes/m1_live_bridge.cu', 'docs/evidence/m1-invocation-regions-cpu-proof.json',
                 'docs/prefill/current-main-source-reconciliation.json',
                 'scripts/m1_owned_processes.py', 'scripts/prefill_owned_processes.py')
        for name in names:
            path = self.root / name
            original = path.read_bytes()
            try:
                changed = original + b'\n'
                if name == 'src/megartx/prefill_storage_plan.py':
                    changed = original.replace(b"'max_contexts': 1", b"'max_contexts': 2")
                path.write_bytes(changed)
                value = json.loads(original_catalog)
                for record in value['superseded_sources']:
                    if record['path'] == name: record['current_sha256'] = sha(changed)
                for field in ('parent_ledgers', 'preserved_borrow_sources'):
                    if name in value[field]: value[field][name] = sha(changed)
                catalog_path.write_text(json.dumps(value))
                with self.subTest(path=name), self.assertRaises(ValueError):
                    self.validate()
            finally:
                path.write_bytes(original)
                catalog_path.write_bytes(original_catalog)

    def test_missing_duplicate_json_and_symlink_catalogs_fail_closed(self):
        path = self.root / lineage.CATALOG_SOURCE
        original = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)): self.validate()
            path.write_bytes(b'{"schema": "duplicate",' + original.lstrip()[1:])
            with self.assertRaises(ValueError): self.validate()
            path.unlink()
            path.symlink_to(ROOT / lineage.CATALOG_SOURCE)
            with self.assertRaises(ValueError): self.validate()
        finally:
            if path.is_symlink(): path.unlink()
            path.write_bytes(original)

    def test_actual_main_bridge_or_checker_cannot_replace_composed_source(self):
        catalog_path = self.root / lineage.CATALOG_SOURCE
        original_catalog = catalog_path.read_bytes()
        for name in ('probes/m1_live_bridge.cu', 'numerical_reference/test_m1_preparation_reference.py'):
            path = self.root / name
            original = path.read_bytes()
            source = original.decode()
            if name == 'probes/m1_live_bridge.cu':
                source = source.replace(
                    '  // Borrow the fresh runMoe stack map; ClearInvocation clears before it dies.\n'
                    '  std::map<std::string,std::pair<size_t,size_t>> const& regions;\n',
                    '  std::map<std::string,std::pair<size_t,size_t>> regions;\n')
                source = source.replace('  auto const regions=getWorkspaceDeviceBufferSizes(',
                                        '  auto regions=getWorkspaceDeviceBufferSizes(')
            else:
                start = source.index('        import importlib.util\n')
                end = source.index('        terminal_changes = current_changes\n', start)
                source = source[:start] + source[end:]
                source = source.replace('                return terminal(path, record["current_sha256"])\n'
                                        '            return terminal(path, previous)\n',
                                        '                return record["current_sha256"]\n'
                                        '            return previous\n', 1)
            value = json.loads(original_catalog)
            record = next(row for row in value['superseded_sources'] if row['path'] == name)
            self.assertEqual(sha(source.encode()), record['main_parent_sha256'])
            try:
                path.write_text(source)
                record['current_sha256'] = record['main_parent_sha256']
                catalog_path.write_text(json.dumps(value))
                with self.subTest(main_source=name), self.assertRaises(ValueError):
                    self.validate()
            finally:
                path.write_bytes(original)
                catalog_path.write_bytes(original_catalog)

    def test_executing_helper_origin_and_bytes_must_match_compiled_pin(self):
        helper = self.storage.map_borrow
        with patch.object(helper, '__spec__', SimpleNamespace(origin='/wrong/helper.py')):
            with self.assertRaisesRegex(ValueError, 'helper origin'):
                self.validate()
        path = Path(self.temp.name) / 'foreign_helper.py'
        path.write_bytes((ROOT / lineage.HELPER_SOURCE).read_bytes() + b'\n')
        with patch.object(helper, '__file__', str(path)), \
                patch.object(helper, '__spec__', SimpleNamespace(origin=str(path))):
            with self.assertRaisesRegex(ValueError, 'helper origin'):
                self.validate()
        path.write_bytes((ROOT / lineage.HELPER_SOURCE).read_bytes())
        with patch.object(helper, '__file__', str(path)), \
                patch.object(helper, '__spec__', SimpleNamespace(origin=str(path))):
            with self.assertRaisesRegex(ValueError, 'helper origin'):
                self.validate()
        name = 'numerical_reference/test_m1_preparation_reference.py'
        path = self.root / name
        original = path.read_bytes()
        try:
            path.unlink()
            path.symlink_to(ROOT / name)
            with self.assertRaises(ValueError): self.validate()
        finally:
            path.unlink()
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()

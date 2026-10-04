"""Reject mixed composition identities without rewriting historical evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from megartx import prefill_storage_plan as storage
from test_prefill_storage_plan import source_fixture

ROOT = Path(__file__).resolve().parents[1]


class CurrentMainReconciliationTests(unittest.TestCase):
    def test_complete_current_vector_and_fixed_parent_ledgers(self):
        hashes = {path: storage.file_sha(ROOT/path) for path in storage.SOURCES}
        catalog = storage.validate_source_catalog(ROOT, hashes)
        self.assertEqual(catalog['parent_ledgers'], storage.PARENT_LEDGERS)
        self.assertEqual(catalog['executing_helpers'], storage.EXECUTING_HELPERS)
        self.assertEqual(catalog['runtime_source_hashes'], {
            path: value for path, value in hashes.items() if path != storage.CATALOG_SOURCE})
        self.assertEqual(set(catalog['previous_source_hashes']), set(catalog['changed_source_hashes']))
        for path, value in catalog['changed_source_hashes'].items():
            self.assertEqual(storage.file_sha(ROOT/path), value, path)
            self.assertNotEqual(value, catalog['previous_source_hashes'][path], path)
        paths = [record['path'] for record in catalog['superseded_sources']]
        self.assertEqual(paths, sorted(set(paths)))
        for record in catalog['superseded_sources']:
            self.assertEqual(set(record), {'path', 'main_parent_sha256',
                                          'prefill_parent_sha256', 'current_sha256'})
            self.assertEqual(storage.file_sha(ROOT/record['path']), record['current_sha256'])
        self.assertTrue(set(storage.EXECUTING_HELPERS) <= set(storage.SOURCES))

    def test_wrong_missing_or_mixed_parent_catalog_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = source_fixture(root)
            path = root/storage.CATALOG_SOURCE
            original = json.loads(path.read_text())
            mutations = []
            for field in storage.COMPOSITION_PARENTS:
                value = copy.deepcopy(original); value[field] = '0'*40
                mutations.append((field, value))
            for field in ('parent_ledgers', 'executing_helpers', 'unchanged_historical_catalogs'):
                for mutation in ('missing', 'unknown', 'wrong'):
                    value = copy.deepcopy(original)
                    key = next(iter(value[field]))
                    if mutation == 'missing': value[field].pop(key)
                    elif mutation == 'unknown': value[field]['unknown.py'] = '0'*64
                    else: value[field][key] = '0'*64
                    mutations.append((field+':'+mutation, value))
            for label, value in mutations:
                path.write_text(json.dumps(value))
                hashes = {**plan['source_hashes'], storage.CATALOG_SOURCE:storage.file_sha(path)}
                with self.subTest(mutation=label), self.assertRaises(ValueError):
                    storage.validate_source_catalog(root, hashes)

    def test_source_lineage_inventory_and_terminal_delta_are_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = source_fixture(root)
            path = root/storage.CATALOG_SOURCE
            original = json.loads(path.read_text())
            for mutation in ('missing', 'duplicate', 'unknown', 'prior', 'current',
                             'reordered', 'previous_delta', 'current_delta'):
                value = copy.deepcopy(original)
                records = value['superseded_sources']
                if mutation == 'missing': records.pop()
                elif mutation == 'duplicate': records.append(dict(records[0]))
                elif mutation == 'unknown': records[0]['path'] = 'unknown.py'
                elif mutation == 'prior': records[0]['main_parent_sha256'] = '0'*64
                elif mutation == 'current': records[0]['current_sha256'] = '0'*64
                elif mutation == 'reordered': records.reverse()
                elif mutation == 'previous_delta': value['previous_source_hashes'].clear()
                else: value['changed_source_hashes'].clear()
                path.write_text(json.dumps(value))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    storage.validate_source_catalog(root, plan['source_hashes'])

    def test_complete_plan_vector_required_and_parent_hash_cannot_relabel_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = source_fixture(root)
            hashes = plan['source_hashes']
            missing = dict(hashes); missing.pop('scripts/prefill_owned_processes.py')
            for changed in (missing, {**hashes, 'unknown.py':'0'*64}):
                with self.assertRaises(ValueError): storage.validate_source_catalog(root, changed)
            helpers = list(storage.EXECUTING_HELPERS)
            for target, foreign in (helpers, helpers[::-1]):
                with self.subTest(target=target):
                    original = (root/target).read_bytes()
                    (root/target).write_bytes((root/foreign).read_bytes())
                    catalog_path = root/storage.CATALOG_SOURCE
                    catalog = json.loads(catalog_path.read_text())
                    prior = catalog['runtime_source_hashes'][target]
                    wrong = storage.file_sha(root/target)
                    catalog['runtime_source_hashes'][target] = wrong
                    catalog_path.write_text(json.dumps(catalog))
                    with self.assertRaisesRegex(ValueError, 'ownership helper'):
                        storage.validate_source_catalog(root, {**hashes, target:wrong})
                    (root/target).write_bytes(original)
                    catalog['runtime_source_hashes'][target] = prior
                    catalog_path.write_text(json.dumps(catalog))

    def test_parent_ledger_bytes_cannot_be_replaced_even_with_new_plan_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = source_fixture(root)
            path = root/storage.PREFILL_CATALOG_SOURCE
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                storage.validate_source_catalog(root, plan['source_hashes'])

    def test_current_equivalence_tests_and_parent_fixtures_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = source_fixture(root)
            catalog = json.loads((root/storage.CATALOG_SOURCE).read_text())
            eq = catalog['default_fit_equivalence']
            for name in [eq['test_path'], *eq['additional_tests'], *eq['source_fixtures']]:
                path = root/name; original = path.read_bytes()
                path.write_bytes(original+b'\n# altered semantic proof\n')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'source drift'):
                    storage.validate_source_catalog(root, plan['source_hashes'])
                path.write_bytes(original)
            for field in ('additional_tests', 'source_fixtures'):
                altered = copy.deepcopy(catalog)
                altered['default_fit_equivalence'][field] = {}
                (root/storage.CATALOG_SOURCE).write_text(json.dumps(altered))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    storage.validate_source_catalog(root, plan['source_hashes'])
                (root/storage.CATALOG_SOURCE).write_text(json.dumps(catalog))


if __name__ == '__main__':
    unittest.main()

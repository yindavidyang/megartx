"""Exact source compatibility for main plus the synchronous map borrow.

This stdlib-only catalog admits source bytes, never GPU execution or numerical
qualification. Historical catalogs retain their original terminal hashes.
"""
import hashlib
import json
from pathlib import Path
import re


CATALOG_SOURCE = 'docs/evidence/m1-map-borrow-main-composition.json'
HELPER_SOURCE = 'src/megartx/m1_map_borrow_lineage.py'
PARENTS = {
    'main_parent': '2d571d3bb4e00b9c0f0045b89eb35ec6c03cf7e3',
    'main_parent_tree': '26748804fda0d24bf24a81c59c2eb5fb392c7a88',
    'borrow_parent': '4a3b2144d36a0fa82a35d71f91abb0c331cce65e',
    'borrow_parent_tree': '9b29b47d9662851c79ed4d31c7f1f768b305389a',
    'common_base': 'aa36de98af05fff7e1e3862fe35ec30e2418f8db',
}
PARENT_LEDGERS = {
    'docs/prefill/current-main-source-reconciliation.json':
        '5c3c14405e0555f0f3be04d097777c0f648194222db7d4bc3151b3f3bdf14036',
    'docs/evidence/m1-invocation-regions-source-pins.json':
        '08be9695065e22206c1ad7108251c4e4353397a9d175145635fc19a3b0484b7b',
}
# Closed, ordered parent records independently extracted from both exact trees.
PARENT_RECORDS_SHA256 = 'ad9cb33b1118e0b6eb1dbd72194730a022501f67cf8817234a46759080c67481'
BORROW_SOURCES = {'probes/m1_live_bridge.cu',
                  'numerical_reference/test_m1_tma_descriptor_contract.py'}
QUALIFICATIONS = ('gpu_executed', 'gpu_authorized', 'numerical_qualified',
                  'quality_qualified', 'performance_qualified')
# Current CPU proof/document bytes are independent fixed endpoints, not values
# that a replacement catalog can freely repin. The validator pins this helper.
PROOF_SOURCES = {
    "docs/m1-map-borrow-main-composition.md": "be7e177a85b8ad182e88a372e16f639c52dd42f531c60f4c951b77eaebebdc40",
    "numerical_reference/test_m1_map_borrow_composition.py": "7c91563d2db91f79313fc160fbe8b7254950d6875f78cbf7c6c989b3488b2831",
    "numerical_reference/test_m1_preparation_reference.py": "8995bca8848e61b189ca169cd258c9caf0e6370bf8e0bdb4b03abea79686ea12",
    "tests/test_prefill_current_main_reconciliation.py": "e3a8286fb25da2b7e3a97c88d32d4f43717258d6e626287f9ee33039ef2da5d1",
    "tests/test_prefill_head_lineage.py": "d95f8d4f391ec796c7aaa07f036de150dfc497af7806c04bffe284b7db904797",
    "tests/test_prefill_ledger_lineage.py": "18b4b16757a99cd27f3a045cafe0f06278dee814ab8268b2324113623c91d81e",
    "tests/test_prefill_storage_plan.py": "a807d4789eea57aea309cf7550cf74e24f73ee7303832b10d113ae1065fe73b0",
    "tests/test_prefill_stream_lineage.py": "7a797527d2c962c1a26be6b3ca8c8dc65e182ddbeadc24fb5e6270d37dec3d36"
}


def parent_validator_source(source):
    """Remove only the source-admission extension, retaining every old check."""
    begin = '    # BEGIN MAP-BORROW SOURCE ADMISSION\n'
    end = '    # END MAP-BORROW SOURCE ADMISSION\n'
    block = (begin +
        "    helper_sha = '[0-9a-f]{64}'\n"
        "    _source_file(root, map_borrow.HELPER_SOURCE, helper_sha)\n"
        "    helper_path = Path(map_borrow.__file__)\n"
        "    if (not helper_path.is_absolute() or map_borrow.__spec__.origin != str(helper_path)\n"
        "            or helper_path.resolve() != Path(__file__).with_name('m1_map_borrow_lineage.py').resolve()):\n"
        "        raise ValueError('Unknown map-borrow helper origin')\n"
        "    _source_file(helper_path.parent, helper_path.name, helper_sha)\n"
        "    composition = map_borrow.load_catalog(root)\n"
        "    def terminal(path, expected_hash):\n"
        "        return map_borrow.terminal_sha(composition, path, expected_hash)\n" + end)
    # Only the helper digest varies with the reviewed CPU proof vector.
    pattern = re.escape(block).replace(re.escape('[0-9a-f]{64}'), '[0-9a-f]{64}')
    source, count = re.subn(pattern, '', source)
    if count != 1 or begin in source or end in source:
        raise ValueError('Exact map-borrow validator insertion required')
    changes = (
        ('    for relative in (CATALOG_SOURCE, map_borrow.CATALOG_SOURCE, map_borrow.HELPER_SOURCE):\n'
         '        _source_file(root, relative, hashes[relative])\n', ''),
        ('    for relative in (*EXTRA_SOURCES, map_borrow.HELPER_SOURCE):\n',
         '    for relative in EXTRA_SOURCES:\n'),
        ('from . import m1_map_borrow_lineage as map_borrow\n', ''),
        ('SOURCES = (*legacy.SOURCES, *EXTRA_SOURCES, map_borrow.CATALOG_SOURCE, map_borrow.HELPER_SOURCE)\n',
         'SOURCES = (*legacy.SOURCES, *EXTRA_SOURCES)\n'),
        ('    expected = {path: hashes[path] for path in SOURCES\n'
         '                if path not in (CATALOG_SOURCE, map_borrow.CATALOG_SOURCE, map_borrow.HELPER_SOURCE)}\n',
         '    expected = {path: hashes[path] for path in SOURCES if path != CATALOG_SOURCE}\n'),
        ("            or {path: terminal(path, pin) for path, pin in catalog.get('runtime_source_hashes', {}).items()} != expected\n",
         "            or catalog.get('runtime_source_hashes') != expected\n"),
        ("        _source_file(root, row['path'], terminal(row['path'], row['current_sha256']))\n",
         "        _source_file(root, row['path'], row['current_sha256'])\n"),
    )
    for current, previous in changes:
        if source.count(current) != 1:
            raise ValueError('Exact map-borrow validator change required')
        source = source.replace(current, previous, 1)
    return source


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate map-borrow catalog field')
        result[key] = value
    return result


def _source(root, path, expected):
    relative = Path(path)
    source = root / relative
    if (relative.is_absolute() or '..' in relative.parts
            or any((root / Path(*relative.parts[:i])).is_symlink()
                   for i in range(1, len(relative.parts) + 1))
            or not source.is_file() or type(expected) is not str
            or not re.fullmatch('[0-9a-f]{64}', expected)
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
        raise ValueError('Map-borrow composition source drift: ' + path)


def load_catalog(root):
    """Validate the additive vector before resolving any historical reference."""
    root = Path(root)
    path = root / CATALOG_SOURCE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError('Bounded regular map-borrow composition catalog required')
    value = json.loads(path.read_text(), object_pairs_hook=_pairs)
    fields = {'schema', 'scope', 'parent_ledgers', 'superseded_sources',
              'preserved_borrow_sources', *PARENTS, *QUALIFICATIONS}
    if (type(value) is not dict or set(value) != fields
            or value['schema'] != 'megartx-m1-map-borrow-main-composition-v1'
            or value['scope'] != 'source_compatibility_only'
            or any(value[key] != expected for key, expected in PARENTS.items())
            or value['parent_ledgers'] != PARENT_LEDGERS
            or any(value[key] is not False for key in QUALIFICATIONS)):
        raise ValueError('Unknown map-borrow composition identity or scope')
    for name, expected in PARENT_LEDGERS.items():
        _source(root, name, expected)
    records = value['superseded_sources']
    if (type(records) is not list or any(type(record) is not dict or set(record) != {
            'path', 'main_parent_sha256', 'borrow_parent_sha256', 'current_sha256'} for record in records)
            or _digest([{key: item for key, item in record.items() if key != 'current_sha256'}
                        for record in records]) != PARENT_RECORDS_SHA256):
        raise ValueError('Unknown, mixed, missing, or duplicate map-borrow source lineage')
    for record in records:
        name = record['path']
        if name in BORROW_SOURCES and record['current_sha256'] != record['borrow_parent_sha256']:
            raise ValueError('Exact PR36 source required: ' + name)
        _source(root, name, record['current_sha256'])
    for name, expected in PROOF_SOURCES.items():
        _source(root, name, expected)
    validator = 'src/megartx/prefill_storage_plan.py'
    restored = parent_validator_source((root / validator).read_text())
    if hashlib.sha256(restored.encode()).hexdigest() != '42eab7949b1b2b0a64892c4efe01084da50339d8d3bbc446d227694153e7d580':
        raise ValueError('Unrelated storage validator source change: ' + validator)
    borrow = json.loads((root / 'docs/evidence/m1-invocation-regions-source-pins.json').read_text())
    if value['preserved_borrow_sources'] != borrow['added_source_hashes']:
        raise ValueError('Immutable PR36 proof vector required')
    for name, expected in value['preserved_borrow_sources'].items():
        _source(root, name, expected)
    return value


def terminal_sha(catalog, path, previous, parent='main_parent_sha256'):
    """Resolve one exact parent terminal, never an arbitrary prior source hash."""
    for record in catalog['superseded_sources']:
        if record['path'] == path:
            if record[parent] != previous:
                raise ValueError('Map-borrow parent terminal mismatch: ' + path)
            return record['current_sha256']
    return previous

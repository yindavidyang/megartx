"""Closed additive source admission for the frozen sampled-attention runtime.

This CPU-only gate preserves every historical catalog and the PR36 overlay.
It admits a whole exact terminal vector, not independent historical endpoints.
Source compatibility never attests installed packages or grants GPU clearance.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

CATALOG_SOURCE = 'docs/prefill/native-attention-source-reconciliation.json'
HELPER_SOURCE = 'src/megartx/prefill_attention_lineage.py'
VALIDATOR_SOURCE = 'src/megartx/prefill_storage_plan.py'
CATALOG_SHA256 = 'a46e406548c4f12aeacf396b727494e1f4fa3b80b337e60160463c18892d274c'
PARENT_VALIDATOR_SHA256 = 'e9a82a14bd3d44416cbdfffad5f2a520f165aec7d10b06215cbb8545d95f0a66'
MAP_HELPER_SHA256 = '7e4f093d9d0dc0c9a1cd9640ce3d59e3a2092144255326af3d02677f1d8233e7'
MAP_CATALOG_SOURCE = 'docs/evidence/m1-map-borrow-main-composition.json'
STORAGE_CATALOG_SOURCE = 'docs/prefill/current-main-source-reconciliation.json'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate attention source catalog field')
        value[key] = item
    return value


def _read(root, relative):
    root, path = Path(root), Path(relative)
    if (path.is_absolute() or '..' in path.parts or str(path) != relative
            or any((root / Path(*path.parts[:i])).is_symlink()
                   for i in range(1, len(path.parts) + 1))
            or not (root / path).is_file()):
        raise ValueError('Attention source drift or nonregular path: ' + relative)
    return (root / path).read_bytes()


def parent_validator_source(source):
    """Remove exactly the additive gate; restore all PR36 code byte-for-byte."""
    changes = (
        ('from . import prefill_attention_lineage as attention_lineage\n', ''),
        ('SOURCES = (*SOURCES, attention_lineage.CATALOG_SOURCE, attention_lineage.HELPER_SOURCE)\n', ''),
        ('    # BEGIN ATTENTION SOURCE ADMISSION\n'
         '    _source_file(root, attention_lineage.HELPER_SOURCE, ATTENTION_HELPER_SHA256)\n'
         '    return attention_lineage.validate_storage_catalog(root, hashes)\n'
         '    # END ATTENTION SOURCE ADMISSION\n', ''),
        ('    for relative in (*EXTRA_SOURCES, map_borrow.HELPER_SOURCE, attention_lineage.HELPER_SOURCE):\n',
         '    for relative in (*EXTRA_SOURCES, map_borrow.HELPER_SOURCE):\n'),
    )
    for current, previous in changes:
        if source.count(current) != 1:
            raise ValueError('Exact attention validator insertion required')
        source = source.replace(current, previous, 1)
    source, count = re.subn("ATTENTION_HELPER_SHA256 = '[0-9a-f]{64}'\\n", '', source)
    if count != 1 or 'ATTENTION SOURCE ADMISSION' in source:
        raise ValueError('Exact attention helper pin insertion required')
    return source


def _origin(module, expected, digest):
    actual = getattr(module, '__file__', None)
    origin = getattr(getattr(module, '__spec__', None), 'origin', None)
    expected = Path(expected)
    if (not actual or not origin or not Path(actual).is_absolute()
            or actual != origin or Path(actual).resolve() != expected.resolve()
            or expected.is_symlink() or _sha(expected.read_bytes()) != digest):
        raise ValueError('Unknown attention source helper origin or source drift')


def _executing_helpers(root):
    # The already executing storage validator is the non-self-referential pin
    # anchor. Its exact non-admission semantics are restored and hashed below.
    from megartx import prefill_storage_plan as storage
    from megartx import m1_map_borrow_lineage as map_borrow
    helper = sys.modules[__name__]
    expected = Path(storage.__file__).with_name('prefill_attention_lineage.py')
    _origin(helper, expected, storage.ATTENTION_HELPER_SHA256)
    if _sha(_read(root, HELPER_SOURCE)) != storage.ATTENTION_HELPER_SHA256:
        raise ValueError('Attention source drift: ' + HELPER_SOURCE)
    _origin(map_borrow, expected.with_name('m1_map_borrow_lineage.py'), MAP_HELPER_SHA256)
    loaded = Path(storage.__file__)
    if (getattr(storage.__spec__, 'origin', None) != str(loaded)
            or not loaded.is_absolute() or loaded.is_symlink()
            or _read(root, VALIDATOR_SOURCE) != loaded.read_bytes()
            or _sha(parent_validator_source(loaded.read_text()).encode()) != PARENT_VALIDATOR_SHA256):
        raise ValueError('Executing storage source validator origin or semantics changed')
    return storage


def load_catalog(root):
    """Require every exact current source before resolving any old endpoint."""
    root = Path(root)
    _executing_helpers(root)
    data = _read(root, CATALOG_SOURCE)
    if len(data) > 256 << 10 or _sha(data) != CATALOG_SHA256:
        raise ValueError('Attention source drift: exact closed terminal catalog required')
    value = json.loads(data, object_pairs_hook=_pairs)
    for record in value['source_records']:
        name = record['path']
        data = _read(root, name)
        if name == VALIDATOR_SOURCE:
            data = parent_validator_source(data.decode()).encode()
        if _sha(data) != record['current_sha256']:
            raise ValueError('Attention source drift: ' + name)
    return value


def terminal_sha(catalog, path, previous):
    """Resolve only the exact parent endpoint, never a historical hash union."""
    for record in catalog['source_records']:
        if record['path'] == path:
            if record['parent_sha256'] != previous:
                raise ValueError('Attention parent terminal mismatch: ' + path)
            # The validator's literal helper pin is the only normalized field.
            # Its real bytes remain bound independently by every new plan.
            if path == VALIDATOR_SOURCE:
                from megartx import prefill_storage_plan as storage
                return _sha(Path(storage.__file__).read_bytes())
            return record['current_sha256']
    raise ValueError('Unknown attention terminal path: ' + path)


def map_borrow_catalog(root):
    """Return the immutable PR36 overlay only after current terminal admission."""
    load_catalog(root)
    return json.loads(_read(root, MAP_CATALOG_SOURCE), object_pairs_hook=_pairs)


def validate_source_catalog(root, hashes):
    storage = _executing_helpers(root)
    catalog = load_catalog(root)
    if (type(hashes) is not dict
            or list(catalog['plan_sources']['storage']) != list(storage.SOURCES)
            or set(hashes) not in (set(catalog['plan_sources']['storage']),
                                  set(catalog['plan_sources']['native_attention']))):
        raise ValueError('Unknown, mixed, or incomplete attention source vector')
    for name, expected in hashes.items():
        if type(expected) is not str or _sha(_read(root, name)) != expected:
            raise ValueError('Attention source drift in plan: ' + name)
    return catalog


def validate_storage_catalog(root, hashes):
    catalog = validate_source_catalog(root, hashes)
    if set(hashes) != set(catalog['plan_sources']['storage']):
        raise ValueError('Complete current storage source vector required')
    return json.loads(_read(root, STORAGE_CATALOG_SOURCE), object_pairs_hook=_pairs)

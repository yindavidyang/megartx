"""Run immutable historical packet tests against their explicit PR34 sources.

New storage source is tested separately. Never weaken a historical exact-byte
catalog to accept different source, and never use these fixtures for a live run.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

_TEMP = None
_ROOT = None


def historical_root():
    global _TEMP, _ROOT
    if _ROOT is not None:
        return _ROOT
    root = Path(__file__).resolve().parents[1]
    fixture = root / 'tests/fixtures/prefill-storage-base'
    manifest = json.loads((fixture/'manifest.json').read_text())
    if manifest['source_head'] != '3e9f50f509c5f79da8505dd04b251cf950502283':
        raise ValueError('Historical source fixture identity changed')
    _TEMP = tempfile.TemporaryDirectory(prefix='megartx-pr34-cpu-')
    _ROOT = Path(_TEMP.name)/'repo'
    shutil.copytree(root, _ROOT, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    for relative, record in manifest['files'].items():
        data = (fixture/record['fixture']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Historical source fixture bytes changed: '+relative)
        (_ROOT/relative).write_bytes(data)
    return _ROOT

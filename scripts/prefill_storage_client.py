"""Exactly one default-off native storage/frontier control; private bounded data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from megartx.prefill_storage_plan import (
    CompactEvidence, PURPOSE, digest, freeze_plan, load_plan, remaining,
    checkpoint_identity, native_ledger_comparison, validate_binding, validate_control,
)
# These CPU-only helpers preserve the exact source-bound API/SSE/token contract.
from prefill_diagnostic_client import (
    StreamLedger, LedgerMismatch, payload, request_id, stream_observation, client_receipt,
)


def validate_observation(plan, stream, observer, evidence=None):
    client = client_receipt(plan, stream, observer)
    comparison = native_ledger_comparison(plan, client, observer, stream_observation(plan, stream))
    compact = {'schema': 'megartx-prefill-storage-client-ledger-v1',
               'plan_sha256': plan['plan_sha256'], 'status': comparison['status'],
               'failed_fields': comparison['failed_fields'],
               'checked_fields': len(comparison['checks']),
               'comparison_sha256': digest(comparison),
               'checks': [check for check in comparison['checks'] if not check['ok']],
               'numerical_qualified': False, 'performance_qualified': False}
    if evidence is not None:
        evidence.write('client-ledger-check.json', compact)
    if comparison['failed_fields']:
        raise LedgerMismatch(compact)
    return client


def storage_client_receipt(plan, control, client):
    validate_control(plan, control, client['output_ids_sha256'])
    return {'schema': 'megartx-prefill-storage-client-v1',
            'purpose': PURPOSE, 'plan_sha256': plan['plan_sha256'], 'status': 'complete',
            'control_sha256': digest(control), 'output_ids_sha256': client['output_ids_sha256'],
            'storage_capture_end': 2049, 'metadata_only_decode_inputs': 254,
            'numerical_qualified': False, 'performance_qualified': False}


def run(plan, directory, deadline):
    validate_binding(plan, directory)
    import requests
    evidence = CompactEvidence(directory)
    session = requests.Session()
    session.trust_env = False
    stream = StreamLedger(plan)
    try:
        # The shared native ownership hook requires this precise bounded marker.
        evidence.write('request.json', {'schema': 'megartx-prefill-native-request-v1',
                                         'plan_sha256': plan['plan_sha256']})
        try:
            with session.post('http://127.0.0.1:18000/v1/completions', json=payload(plan), stream=True,
                              timeout=(5, remaining(deadline))) as response:
                response.raise_for_status()
                while True:
                    remaining(deadline)
                    raw = response.raw.readline(65537)
                    if not raw:
                        break
                    if len(raw) > 65536:
                        raise ValueError('SSE overflow rejected before parse')
                    line = raw.decode('utf-8').strip()
                    if line:
                        stream.consume(line)
        except BaseException as primary:
            try:
                evidence.write('client-stream.json', stream_observation(plan, stream))
            except BaseException as error:
                if hasattr(primary, 'add_note'):
                    primary.add_note('Storage client stream diagnostics could not be saved: ' + type(error).__name__)
            raise
        evidence.write('client-stream.json', stream_observation(plan, stream))
        from megartx.prefill_storage_plan import _read_json
        observer = _read_json(Path(directory) / 'observer.json', 65536)
        client = validate_observation(plan, stream, observer, evidence)
        control = _read_json(Path(directory) / 'control.json', 65536)
        mode = storage_client_receipt(plan, control, client)
        evidence.write('client.json', client)
        evidence.write('storage-client.json', mode)
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = None
        for cleanup in (session.close, lambda: (Path(directory) / 'request.json').unlink(missing_ok=True)):
            try:
                cleanup()
            except BaseException as error:
                if primary is not None and hasattr(primary, 'add_note'):
                    primary.add_note('Storage client cleanup failed: ' + type(error).__name__)
                elif cleanup_error is None:
                    cleanup_error = error
        if primary is None and cleanup_error is not None:
            raise cleanup_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-tokens', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--adapter-site', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.freeze_tokens:
        if args.plan or args.checkpoint is None or args.adapter_site is None:
            parser.error('Choose freeze or storage client execution')
        from megartx.prefill_storage_plan import _read_json
        tokens = _read_json(args.freeze_tokens)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
            parser.error('Commit reviewed storage source before freezing private plan')
        value = freeze_plan(tokens, head, root, checkpoint_identity(args.checkpoint), args.adapter_site)
        # Validate before creating the private plan, and create it mode 0600.
        from megartx.prefill_storage_plan import validate_plan
        validate_plan(value, root)
        fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, allow_nan=False)
        return 0
    if not args.plan or not os.environ.get('MEGARTX_PREFILL_NATIVE_DEADLINE'):
        parser.error('Storage client requires the owned launcher and exact storage plan')
    run(load_plan(args.plan, root), args.output / 'prefill-storage',
        float(os.environ['MEGARTX_PREFILL_NATIVE_DEADLINE']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Exactly one native 2K/chunk256 request; all output stays private and bounded."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from megartx.prefill_diagnostic_plan import (Evidence, digest, freeze_plan, load_plan, remaining,
    checkpoint_identity, api_request_id, native_request_identity, native_ledger_comparison)


def request_id(plan):
    return api_request_id(plan)


def payload(plan):
    return {'model': 'gemma4-nvfp4', 'prompt': plan['tokens'], 'max_tokens': 256,
            'temperature': 0, 'seed': 1234, 'ignore_eos': True, 'n': 1,
            'stream': True, 'stream_options': {'include_usage': True},
            'return_token_ids': True, 'request_id': request_id(plan)}


class StreamLedger:
    def __init__(self, plan):
        self.plan, self.tokens, self.usage, self.finish, self.done = plan, [], None, None, False
        self.events = 0
        self.response_id = 'cmpl-' + request_id(plan)

    def consume(self, line):
        if self.done or len(line) > 65536 or not line.startswith('data: '):
            raise ValueError('Invalid/excessive completion SSE record')
        self.events += 1
        if self.events > 300:
            raise ValueError('Native client event budget exceeded')
        if line == 'data: [DONE]':
            if self.finish != 'length' or self.usage != {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304} or len(self.tokens) != 256:
                raise ValueError('Incomplete native completion/token/usage ledger')
            self.done = True
            return
        value = json.loads(line[6:])
        if value.get('id') != self.response_id or 'error' in value:
            raise ValueError('Completion response identity/error changed')
        for choice in value.get('choices', []):
            if type(choice.get('index')) is not int or choice['index'] != 0 or self.finish is not None:
                raise ValueError('Repeated/unknown completion choice')
            ids = choice.get('token_ids')
            if type(ids) is not list or any(type(t) is not int or not 0 <= t < 262144 for t in ids):
                raise ValueError('Actual completion delta token IDs required')
            if len(self.tokens)+len(ids) > 256:
                raise ValueError('Native emitted token budget exceeded')
            prompt = choice.get('prompt_token_ids')
            if prompt is not None and prompt != self.plan['tokens']:
                raise ValueError('Returned prompt token identity changed')
            self.tokens.extend(ids)
            finish = choice.get('finish_reason')
            if finish is not None:
                if finish != 'length' or len(self.tokens) != 256:
                    raise ValueError('Native completion stopped before fixed output budget')
                self.finish = finish
        if value.get('usage') is not None:
            if self.usage is not None:
                raise ValueError('Repeated native usage record')
            usage = value['usage']
            if type(usage) is not dict or any(type(usage.get(k)) is not int for k in
                    ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                raise ValueError('Native usage requires exact integer counts')
            self.usage = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


class LedgerMismatch(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__('Client/native actual request or emitted-token ledger mismatch: ' +
                         json.dumps(diagnostic, sort_keys=True, allow_nan=False))


def stream_observation(plan, stream):
    """Save final transport facts before any observer read or comparison."""
    return {'schema': 'megartx-prefill-client-stream-v1', 'plan_sha256': plan['plan_sha256'],
            'response_id_sha256': digest(stream.response_id), 'sse_events': stream.events,
            'output_ids_sha256': digest(stream.tokens), 'emitted_outputs': len(stream.tokens),
            'stream_done': stream.done, 'finish_reason': stream.finish, 'usage': stream.usage,
            'numerical_qualified': False, 'performance_qualified': False}


def client_receipt(plan, stream, observer):
    engine_id = observer.get('engine_request_id')
    try:
        identity = native_request_identity(plan, engine_id)
    except ValueError:
        identity = None
    return {'schema': 'megartx-prefill-native-client-v1',
            'plan_sha256': plan['plan_sha256'], 'response_id': stream.response_id,
            'engine_request_id': engine_id, 'request_identity': identity,
            'output_ids_sha256': digest(stream.tokens), 'emitted_outputs': len(stream.tokens),
            'usage': stream.usage, 'stream_done': stream.done, 'finish_reason': stream.finish,
            'sse_events': stream.events,
            'status': 'complete' if stream.done else 'incomplete',
            'numerical_qualified': False, 'performance_qualified': False}


def validate_observation(plan, stream, observer, evidence=None):
    client = client_receipt(plan, stream, observer)
    diagnostic = native_ledger_comparison(plan, client, observer, stream_observation(plan, stream))
    # Persist every comparison, including mismatch, before the exception. This
    # is a diagnostic record, never a successful client or observed-fit receipt.
    if evidence is not None:
        evidence.write('client-ledger-check.json', diagnostic)
    if diagnostic['failed_fields']:
        raise LedgerMismatch(diagnostic)
    return client


def run(plan, directory, deadline):
    from megartx.prefill_runner_binding import validate_binding
    validate_binding(plan, directory)
    import requests
    evidence = Evidence(directory)
    evidence.write('request.json', {'schema': 'megartx-prefill-native-request-v1', 'plan_sha256': plan['plan_sha256']})
    session = requests.Session()
    session.trust_env = False
    stream = StreamLedger(plan)
    try:
        try:
            with session.post('http://127.0.0.1:18000/v1/completions', json=payload(plan), stream=True,
                              timeout=(5, remaining(deadline))) as response:
                response.raise_for_status()
                # readline(size) bounds a line before JSON/SSE parsing or evidence writes.
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
                    primary.add_note('Client stream diagnostics could not be saved: ' + type(error).__name__)
            raise
        evidence.write('client-stream.json', stream_observation(plan, stream))
        observer = json.loads((Path(directory) / 'observer.json').read_text())
        client = validate_observation(plan, stream, observer, evidence)
        evidence.write('client.json', client)
    finally:
        session.close()
        (Path(directory) / 'request.json').unlink(missing_ok=True)


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
            parser.error('Choose freeze or client execution')
        tokens = json.loads(args.freeze_tokens.read_text())
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
            parser.error('Commit reviewed source before freezing private plan')
        value = freeze_plan(tokens, head, root, checkpoint_identity(args.checkpoint), args.adapter_site)
        with args.output.open('x') as out:
            json.dump(value, out, indent=2, allow_nan=False)
        load_plan(args.output, root)
        os.chmod(args.output, 0o600)
        return 0
    if not args.plan or not os.environ.get('MEGARTX_PREFILL_NATIVE_DEADLINE'):
        parser.error('Native client requires the owned launcher and exact plan')
    run(load_plan(args.plan, root), args.output / 'prefill-native',
        float(os.environ['MEGARTX_PREFILL_NATIVE_DEADLINE']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

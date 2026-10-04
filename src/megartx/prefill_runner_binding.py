"""Default V2 diagnostic admission and live observer receipt. CPU-only imports."""
import inspect
import json
import os
from pathlib import Path
import sys

from .prefill_diagnostic_plan import INSTALLED, TRANSPORT_FILES, RUNNER_POLICY, digest, file_sha

HOOKS = ('runner_constructor', 'initialize_kv_cache', 'execute_model', 'prepare_inputs', 'prepare_attn', 'sample',
         'metadata_build', 'model_forward', 'compute_logits')


def require_default_selection(config):
    if (os.environ.get('VLLM_USE_V2_MODEL_RUNNER') is not None
            or config.use_v2_model_runner is not True
            or config.is_mm_encoder_only):
        raise RuntimeError('Native prefill requires the resolved default V2 runner; no override is admitted')
    return RUNNER_POLICY.copy()


def verify_installed_files():
    """Check selection/adapter source before constructors, without runtime imports.

    At plugin registration some pinned modules have not been imported yet. Their
    files are checked here. Runner module origins are also checked by the access
    interface after cache initialization. API/transport files are not required
    to be imported in the separate GPU worker process.
    """
    root = Path(inspect.getsourcefile(sys.modules['vllm'])).resolve().parent
    for name, expected in {**INSTALLED, **TRANSPORT_FILES}.items():
        relative = Path(*name.split('.')[1:])
        path = root / relative.with_suffix('.py')
        if not path.is_file():
            path = root / relative / '__init__.py'
        if path.is_symlink() or file_sha(path) != expected:
            raise RuntimeError('Installed runner binding source drift: ' + name)


def start_ticks(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    fields = stat[stat.rfind(')')+2:].split()
    if fields[0] in ('Z', 'X'):
        raise RuntimeError('Native observer owner is no longer live')
    return int(fields[19])


def validate_binding(plan, directory, owned_identities=None, require_live=True):
    """Fail before POST if the source-bound provider was never installed/live."""
    path = Path(directory)/'runner-binding.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise RuntimeError('Native observer runner binding is absent before request dispatch')
    value = json.loads(path.read_text())
    expected = {'schema', 'plan_sha256', 'source_head', 'owner_pid', 'owner_start_ticks',
                'runner_policy', 'installed_sources', 'hook_bindings', 'mutable_lease_granted'}
    if (set(value) != expected or value['schema'] != 'megartx-prefill-runner-binding-v1'
            or value['plan_sha256'] != plan['plan_sha256'] or value['source_head'] != plan['source_head']
            or value['runner_policy'] != RUNNER_POLICY or plan['runner_policy'] != RUNNER_POLICY
            or digest(value['runner_policy']) != digest(RUNNER_POLICY)
            or value['installed_sources'] != {**INSTALLED, **TRANSPORT_FILES}
            or set(value['hook_bindings']) != set(HOOKS)
            or any(v is not True for v in value['hook_bindings'].values())
            or value['mutable_lease_granted'] is not False
            or type(value['owner_pid']) is not int or value['owner_pid'] <= 0
            or type(value['owner_start_ticks']) is not int or value['owner_start_ticks'] <= 0):
        raise RuntimeError('Native observer runner binding differs from frozen admission')
    identity = (value['owner_pid'], value['owner_start_ticks'])
    if ((require_live and start_ticks(identity[0]) != identity[1])
            or (owned_identities is not None and identity not in owned_identities)):
        raise RuntimeError('Native observer runner binding owner is not the live owned process')
    return value

"""Default-off actual selected native attention observation, never replay.

Retains the proven all-layer storage/frontier engine. Only its explicit policy
seams differ: no old selected raw payloads, original head hashing still occurs,
and the independently read actual operator inputs/outputs add eight raw files.
"""
import hashlib
import os
from pathlib import Path

from .prefill_storage_native import StorageProvider
from .prefill_storage import add_failure_note
from .prefill_attention_evidence import StreamingEvidence
from .prefill_attention_hooks import AttentionHooks
from .prefill_attention_native_plan import (PURPOSE, SELECTOR, CONTROL_SPEC, SHIM_SOURCE,
    SHIM_UPSTREAM_SHA256, INSTALLED_CONTRACT, cpu_capture_plan, digest, load_plan, verify_adapter_sources)
from .prefill_diagnostic_plan import native_request_identity


class AttentionProvider(StorageProvider):
    evidence_factory = StreamingEvidence
    source_verifier = staticmethod(verify_adapter_sources)
    storage_binding_purpose = PURPOSE

    def __init__(self, runner, plan, directory, torch, hook_checks):
        self.attention_hooks = None
        self.attention_head_hashes = {}
        self.attention_raw_manifest = None
        root = Path(os.environ['MEGARTX_PREFILL_NATIVE_SOURCE_ROOT'])
        self.attention_plan = cpu_capture_plan(plan, root)
        if plan['installed_sources'] != INSTALLED_CONTRACT:
            raise RuntimeError('Unknown installed attention forwarding/helper closure')
        super().__init__(runner, plan, directory, torch, hook_checks)
        try:
            self.evidence.create_capture(self.attention_plan['plan_sha256'])
            self.transfer.reserve(65536, 'attention_metadata')
            self.attention_hooks = AttentionHooks(self, installed_sources=plan['installed_sources'])
            self.attention_hooks.require_current()
            self.evidence.write('attention-binding.json', {
                'schema': 'megartx-prefill-attention-binding-v1',
                'plan_sha256': plan['plan_sha256'], 'capture_plan_sha256': self.attention_plan['plan_sha256'],
                'owner_pid': self.access.identity.owner_pid,
                'owner_start_ticks': self.access.identity.owner_start_ticks,
                'hooks_bound': True, 'installed_sources': plan['installed_sources'],
                'completion_policy': CONTROL_SPEC['completion_policy']})
        except BaseException as error:
            self.poison_storage(error)
            raise

    def on_processed_row(self, layer, position, slot, key, value):
        self.evidence.writer_row(layer, position, key, value)
        self.attention_hooks.processed_row(layer, position, slot, key, value)

    def retain_processed_row(self, layer, position, key, value):
        # All old processed/storage checks remain; their raw policy is replaced.
        pass

    def retain_head_row(self, position, raw):
        if position in self.attention_head_hashes:
            raise RuntimeError('Duplicate original native head observation')
        self.attention_head_hashes[position] = hashlib.sha256(raw).hexdigest()

    def require_hooks(self):
        super().require_hooks()
        hooks = getattr(self, 'attention_hooks', None)
        if hooks is not None and not self.completed:
            hooks.require_current()

    def finish(self, ticket, result):
        self.attention_hooks.require_frame_complete(ticket)
        super().finish(ticket, result)

    def sample_retention_receipt(self):
        if self.sample_count != 0 or self.raw_heads != 2 or set(self.attention_head_hashes) != {2047, 2048}:
            raise RuntimeError('Attention purpose retained old raw samples or lost head observations')
        self.attention_hooks.require_complete()
        self.attention_raw_manifest = self.evidence.seal_raw()
        return {'sample_combined_kv_rows': 0, 'raw_head_rows': 2,
            'raw_head_payload_rows': 0, 'attention_raw_files': 8,
            'attention_raw_bytes': 5395952,
            'observed_head_hashes': {str(k):v for k,v in self.attention_head_hashes.items()},
            'raw_samples_sha256': digest(self.attention_raw_manifest)}

    def restore_storage(self, primary=None):
        try:
            super().restore_storage(primary)
        finally:
            hooks = getattr(self, 'attention_hooks', None)
            if hooks is not None:
                try:
                    hooks.restore(primary)
                except BaseException as error:
                    self.failed = True
                    if primary is None:
                        raise
                    add_failure_note(primary, 'Attention hook restoration failed: '+repr(error))

    def write_control_receipt(self, receipt):
        if self.failed or self.attention_raw_manifest is None:
            raise RuntimeError('Poisoned/incomplete attention capture cannot seal')
        receipt = {**receipt, 'schema': 'megartx-prefill-attention-storage-control-v1',
                   'attention_callbacks_restored': True}
        self.evidence.write('control.json', receipt)
        self.evidence.write('attention-records.json', {
            'schema': 'megartx-prefill-attention-native-records-v1',
            'native_plan_sha256': self.plan['plan_sha256'],
            'capture_plan_sha256': self.attention_plan['plan_sha256'],
            'request_identity_sha256': digest(native_request_identity(self.plan, self.ledger.request_id)),
            'raw_manifest': self.attention_raw_manifest,
            'records': self.attention_hooks.records,
            'callbacks_restored': True})


def install_attention_observer(torch, model_cls):
    if (os.environ.get(SELECTOR) != '1'
            or os.environ.get('MEGARTX_PREFILL_STORAGE_CONTROL') is not None):
        raise RuntimeError('Attention capture requires its explicit exclusive native purpose')
    from .prefill_native import install_native_observer
    return install_native_observer(torch, model_cls, provider_factory=AttentionProvider,
                                   plan_loader=load_plan, source_verifier=verify_adapter_sources)

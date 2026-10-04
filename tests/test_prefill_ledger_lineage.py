"""Additive request-ledger correction; historical evidence is never rewritten."""
import hashlib
import json
from pathlib import Path
import unittest
from megartx import m1_map_borrow_lineage as map_borrow
from megartx.prefill_diagnostic_plan import HISTORICAL_SOURCES as SOURCES, TRANSPORT_FILES

ROOT=Path(__file__).resolve().parents[1]
CATALOG='docs/prefill/native-ledger-matching-correction.json'


def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


class LedgerLineageTests(unittest.TestCase):
    def test_current_exact_vectors_and_all_historical_catalogs(self):
        value=json.loads((ROOT/CATALOG).read_text())
        self.assertEqual(set(value['runtime_source_hashes']),set(SOURCES))
        follow=json.loads((ROOT/'docs/prefill/native-stream-fit-admission-correction.json').read_text())
        storage=json.loads((ROOT/'docs/prefill/native-storage-source-reconciliation.json').read_text())
        composition=json.loads((ROOT/'docs/prefill/current-main-source-reconciliation.json').read_text())
        terminal=map_borrow.load_catalog(ROOT)
        for field in ('runtime_source_hashes','changed_source_hashes','new_control_source_hashes',
                      'unchanged_historical_catalogs','unchanged_shared_sources'):
            for path,expected in value[field].items():
                if path in follow['changed_source_hashes']:
                    self.assertEqual(follow['previous_source_hashes'][path],expected,path)
                    expected=follow['changed_source_hashes'][path]
                if path in storage['changed_source_hashes']:
                    self.assertEqual(storage['previous_source_hashes'][path],expected,path)
                    expected=storage['changed_source_hashes'][path]
                if path in composition['changed_source_hashes']:
                    self.assertEqual(composition['previous_source_hashes'][path],expected,path)
                    expected=composition['changed_source_hashes'][path]
                self.assertEqual(sha(path),map_borrow.terminal_sha(terminal,path,expected),path)
        self.assertEqual(value['installed_transport_file_hashes'],TRANSPORT_FILES)
        changed=sorted(path for path in SOURCES if value['runtime_source_hashes'][path]!=value['previous_runtime_source_hashes'][path])
        self.assertEqual(changed,['scripts/prefill_diagnostic_client.py','src/megartx/prefill_diagnostic_plan.py',
                                'src/megartx/prefill_native.py','src/megartx/prefill_runner_binding.py'])
        self.assertEqual(value['runtime_changes'],changed)

    def test_confirmed_identity_failure_does_not_invent_missing_client_evidence(self):
        value=json.loads((ROOT/CATALOG).read_text());failed=value['preserved_failed_attempt']
        self.assertEqual(failed['source_head'],'783dcd11e766d2606c2061034ae2358f07ee627e')
        self.assertEqual(failed['completed_native_frames'],263)
        self.assertEqual(failed['accepted_head_receipts'],263)
        self.assertEqual(failed['native_emitted_outputs'],256)
        self.assertIs(failed['engine_id_comparison_equal'],False)
        self.assertEqual(failed['actual_internal_suffix_semantics'],'hyphen_plus_8_lowercase_hex')
        for key in ('client_output_ids_sha256','client_emitted_outputs','sse_events','done','finish_reason','usage',
                    'client_native_token_agreement'):
            self.assertIsNone(failed[key])
        self.assertIs(failed['clearance_consumed'],True);self.assertIs(failed['cleanup_verified'],True)
        self.assertIs(failed['gpu_slot_released'],True);self.assertIs(failed['final_fit_published'],False)
        for key in ('fit_qualified','numerical_qualified','performance_qualified','gpu_executed','gpu_retry_authorized'):
            self.assertIs(value[key],False)


if __name__=='__main__':unittest.main()

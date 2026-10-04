"""Additive head correction vector; every prior catalog remains immutable."""
import hashlib
import json
from pathlib import Path
import unittest

from megartx.loaded_engine_access import HEAD_SOURCES
from megartx.prefill_diagnostic_plan import HISTORICAL_SOURCES as SOURCES

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


class HeadLineageTests(unittest.TestCase):
    def test_exact_new_runtime_and_control_vector_preserves_historical_catalogs(self):
        value=json.loads((ROOT/'docs/prefill/native-head-observation-correction.json').read_text())
        self.assertEqual(set(value['runtime_source_hashes']),set(SOURCES))
        delta=json.loads((ROOT/'docs/prefill/native-ledger-matching-correction.json').read_text())
        self.assertEqual(delta['previous_catalog_sha256'],sha('docs/prefill/native-head-observation-correction.json'))
        follow=json.loads((ROOT/'docs/prefill/native-stream-fit-admission-correction.json').read_text())
        storage=json.loads((ROOT/'docs/prefill/native-storage-source-reconciliation.json').read_text())
        composition=json.loads((ROOT/'docs/prefill/current-main-source-reconciliation.json').read_text())
        for key in ('runtime_source_hashes','controls_source_sha256','unchanged_historical_catalogs'):
            for path,digest in value[key].items():
                if path in delta['changed_source_hashes']:
                    self.assertEqual(delta['previous_source_hashes'][path],digest,path)
                    digest=delta['changed_source_hashes'][path]
                if path in follow['changed_source_hashes']:
                    self.assertEqual(follow['previous_source_hashes'][path],digest,path)
                    digest=follow['changed_source_hashes'][path]
                if path in storage['changed_source_hashes']:
                    self.assertEqual(storage['previous_source_hashes'][path],digest,path)
                    digest=storage['changed_source_hashes'][path]
                if path in composition['changed_source_hashes']:
                    self.assertEqual(composition['previous_source_hashes'][path],digest,path)
                    digest=composition['changed_source_hashes'][path]
                self.assertEqual(sha(path),digest,path)
        self.assertEqual(value['installed_head_source_hashes'],HEAD_SOURCES)
        previous=value['previous_runtime_source_hashes']
        changed=sorted(path for path in SOURCES if previous[path]!=value['runtime_source_hashes'][path])
        self.assertEqual(changed,['src/megartx/loaded_engine_access.py','src/megartx/prefill_native.py'])
        self.assertEqual(changed,value['runtime_changes'])

    def test_unknown_failure_fields_and_cpu_only_correction_cannot_qualify(self):
        value=json.loads((ROOT/'docs/prefill/native-head-observation-correction.json').read_text())
        failed=value['preserved_failed_attempt']
        for key in ('actual_head_hidden_shape','actual_logits_shape','actual_logits_dtype','actual_loaded_suppression_count'):
            self.assertIsNone(failed[key])
        self.assertEqual(failed['completed_prompt_frames'],1)
        self.assertEqual(failed['completed_decode_frames'],0)
        for key in ('clearance_consumed','cleanup_verified','gpu_slot_released'):
            self.assertIs(failed[key],True)
        for key in ('fit_qualified','numerical_qualified','performance_qualified','gpu_executed','gpu_retry_authorized'):
            self.assertIs(value[key],False)
        self.assertEqual(value['expected_complete_head_counts'],{'intermediate_prompt_chunk':7,'final_prompt':1,'decode':255})


if __name__=='__main__':unittest.main()

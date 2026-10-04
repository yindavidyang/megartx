"""Independent-review transport-admission fix is additive to the prior freeze."""
import hashlib
import json
from pathlib import Path
import unittest
from megartx.prefill_diagnostic_plan import SOURCES

ROOT=Path(__file__).resolve().parents[1]


class StreamAdmissionLineageTests(unittest.TestCase):
    def test_narrow_followon_preserves_previous_freeze_and_all_historical_catalogs(self):
        value=json.loads((ROOT/'docs/prefill/native-stream-fit-admission-correction.json').read_text())
        self.assertEqual(value['previous_local_head'],'960418e222de82db8ffa6d38e4e2483d2d40edc2')
        self.assertEqual(value['previous_local_tree'],'4e76d6d863d9ae4c3649734b3db594a95369d480')
        self.assertEqual(set(value['runtime_source_hashes']),set(SOURCES))
        storage=json.loads((ROOT/'docs/prefill/native-storage-source-reconciliation.json').read_text())
        for field in ('runtime_source_hashes','changed_source_hashes','new_control_source_hashes','unchanged_historical_catalogs'):
            for path,expected in value[field].items():
                if path in storage['changed_source_hashes']:
                    self.assertEqual(storage['previous_source_hashes'][path],expected,path)
                    expected=storage['changed_source_hashes'][path]
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected,path)
        changed=sorted(path for path in SOURCES if value['runtime_source_hashes'][path]!=value['previous_runtime_source_hashes'][path])
        self.assertEqual(changed,['scripts/prefill_diagnostic_client.py','src/megartx/prefill_diagnostic_plan.py'])
        for key in ('fit_qualified','numerical_qualified','performance_qualified','gpu_executed','gpu_retry_authorized'):
            self.assertIs(value[key],False)
        self.assertEqual(value['review_finding']['status'],'reproduced_and_cpu_corrected')
        self.assertIs(value['review_finding']['reproduction_writes_intercepted'],True)
        self.assertIs(value['review_finding']['native_fit_published'],False)


if __name__=='__main__':unittest.main()

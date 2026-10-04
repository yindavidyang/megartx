"""Exact source vector extension; historical catalogs are immutable inputs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from megartx import prefill_plan

ROOT=Path(__file__).resolve().parents[1]


class NativeReconciliationTests(unittest.TestCase):
    def test_mixed_plugin_launcher_unknown_vector_and_math_reject(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'repo'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('.git','__pycache__'))
            manifest=root/'docs/prefill/source-binding-native.json'
            plan=prefill_plan.read_json(root/'docs/prefill/profile-plan-native.json')
            self.assertEqual(prefill_plan.verify_source_binding(plan,manifest,root),12)
            review=prefill_plan.read_json(root/'docs/prefill/native-source-reconciliation.json')
            for path in ('src/megartx/vllm_scale_plugin.py','scripts/run_scale_validation.py','src/megartx/m1_live.py','src/megartx/nvfp4_runtime.py','src/megartx/native_diagnostic_composition.py'):
                file=root/path; original=file.read_bytes()
                variants=[original+b'\n# unknown vector\n']
                if path=='src/megartx/vllm_scale_plugin.py':
                    changed=original.replace(b'y.float() * topk_weights',b'y.float() + topk_weights')
                    self.assertNotEqual(changed,original)
                    variants.append(changed)
                if path in review['prior_repo_files'] and review['prior_repo_files'][path]!=review['repo_files'][path]:
                    fixture=ROOT/'tests/fixtures/prefill-native-base'/(Path(path).stem+'.source')
                    self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),review['prior_repo_files'][path])
                    variants.append(fixture.read_bytes())
                for changed in variants:
                    file.write_bytes(changed)
                    with self.subTest(path=path),self.assertRaisesRegex(ValueError,'Source drift'):
                        prefill_plan.verify_source_binding(plan,manifest,root)
                file.write_bytes(original)

    def test_composition_cannot_select_parent_or_mix_rehashed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns('.git', '__pycache__'))
            manifest_path = root / 'docs/prefill/source-binding-native.json'
            original = prefill_plan.read_json(manifest_path)
            plan = prefill_plan.read_json(root / 'docs/prefill/profile-plan-native.json')
            evidence_path = root / 'docs/prefill/native-composition-lineage.json'
            evidence_bytes = evidence_path.read_bytes()
            for mutation in ('absent', 'partial', 'unknown', 'parent', 'prior_vector', 'evidence_hash', 'qualification'):
                manifest = copy.deepcopy(original)
                overlay = manifest['native_composition_overlay']
                review = json.loads(evidence_bytes)
                if mutation == 'absent':
                    del manifest['native_composition_overlay']
                elif mutation == 'partial':
                    overlay['repo_files'].pop('src/megartx/native_diagnostic_composition.py')
                elif mutation == 'unknown':
                    overlay['repo_files']['unreviewed.py'] = '0' * 64
                elif mutation == 'parent':
                    overlay['prefill_parent'] = '0' * 40
                elif mutation == 'prior_vector':
                    review['previous_shared_repo_files']['scripts/run_scale_validation.py'] = '0' * 64
                elif mutation == 'qualification':
                    review['gpu_executed'] = True
                else:
                    overlay['evidence_sha256'] = '0' * 64
                if mutation in ('prior_vector', 'qualification'):
                    evidence_path.write_text(json.dumps(review))
                    overlay['evidence_sha256'] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                plan['binding']['source_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    prefill_plan.verify_source_binding(plan, manifest_path, root)
                evidence_path.write_bytes(evidence_bytes)

    def test_historical_catalog_hashes_and_plugin_arithmetic_ast_preserved(self):
        receipt=json.loads((ROOT/'docs/prefill/native-diagnostic-binding.json').read_text())
        for path,expected in receipt['historical_catalogs'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected,path)
        for path,expected in receipt['historical_source_fixtures'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected,path)
        old=ast.parse((ROOT/'tests/fixtures/prefill-native-base/vllm_scale_plugin.source').read_text())
        new=ast.parse((ROOT/'src/megartx/vllm_scale_plugin.py').read_text())
        def functions(tree):
            return {node.name:ast.dump(node) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in {'load','routed_adapter','deterministic_fused','incumbent_routed','model_forward','logits_forward'}}
        self.assertEqual(functions(old),functions(new))
        self.assertFalse(receipt['gpu_executed'])
        self.assertFalse(receipt['numerical_qualified'])
        self.assertFalse(receipt['performance_qualified'])


if __name__=='__main__':unittest.main()

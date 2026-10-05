"""Current-source default-fit equivalence, separate from historical packet tests."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from megartx import prefill_diagnostic_plan as legacy
from megartx import prefill_storage_plan as storage
from megartx.controlled_kv_capture import CONFIG_SHA256

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/prefill-storage-base'
MANIFEST = json.loads((FIXTURES/'manifest.json').read_text())


def previous(path):
    record = MANIFEST['files'][path]
    data = (FIXTURES/record['fixture']).read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Historical fixture changed')
    return ast.parse(data)


def dump(node):
    return ast.dump(node, include_attributes=False)


class DefaultPurpose(ast.NodeTransformer):
    """Evaluate explicit storage/attention gates with both selectors absent."""
    def known(self, node):
        if isinstance(node, ast.Name) and node.id in ('prefill_storage', 'prefill_attention'):
            return False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.known(node.operand)
            return not value if value is not None else None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            values = [self.known(x) for x in node.values]
            return False if False in values else None
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id in ('storage', 'attention_capture')
                and len(node.ops) == len(node.comparators) == 1):
            if isinstance(node.ops[0], ast.IsNot) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                return False
            if isinstance(node.ops[0], ast.Eq) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == '1':
                return False
        return None

    def visit_FunctionDef(self, node):
        if node.name == "storage_write":
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        # Normalize nested purpose branches before selecting one so consecutive
        # attention/storage gates still produce a flat, executable statement list.
        node = self.generic_visit(node)
        known = self.known(node.test)
        if known is not None:
            return node.body if known else node.orelse
        return node

    def visit_IfExp(self, node):
        known = self.known(node.test)
        return self.visit(node.body if known else node.orelse) if known is not None else self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in
               ('storage', 'prefill_storage', 'attention_capture', 'prefill_attention')
               for t in node.targets):
            return None
        if any(isinstance(t, ast.Name) and t.id == 'prefill_native' for t in node.targets):
            node = copy.deepcopy(node)
            node.value = ast.parse('args.client == "prefill-native"', mode='eval').body
            return node
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        node.elts = [x for x in node.elts if not (isinstance(x, ast.Constant) and x.value in
                    ('prefill-storage', 'MEGARTX_PREFILL_STORAGE_CONTROL',
                     'prefill-attention', 'MEGARTX_PREFILL_ATTENTION_CAPTURE'))]
        return node

    def visit_Set(self, node):
        node = self.generic_visit(node)
        node.elts = [x for x in node.elts if not (isinstance(x, ast.Constant) and x.value in
                    ('prefill-storage', 'prefill-attention'))]
        return node


class NativeDefault(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        if node.name == 'NativeProvider':
            node.body = [x for x in node.body if not (isinstance(x, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id in ('source_verifier', 'evidence_factory', 'observe_cache_hashes') for t in x.targets))]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == 'install_native_observer':
            if [x.arg for x in node.args.kwonlyargs] != ['provider_factory','plan_loader','source_verifier']:
                raise AssertionError('Unknown native extension signature')
            if any(not isinstance(x, ast.Constant) or x.value is not None for x in node.args.kw_defaults):
                raise AssertionError('Non-null default provider override')
            node.args.kwonlyargs, node.args.kw_defaults = [], []
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.Or) and len(node.values) == 2:
            a, b = node.values
            if isinstance(b, ast.Name) and b.id in ('verify_adapter_sources','NativeProvider','load_plan'):
                return b
        if isinstance(node.op, ast.And) and len(node.values) == 2 and dump(node.values[0]) == dump(ast.parse('self.observe_cache_hashes',mode='eval').body):
            return self.visit(node.values[1])
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if dump(node) == dump(ast.parse('self.evidence_factory', mode='eval').body):
            return ast.Name(id='Evidence', ctx=ast.Load())
        return self.generic_visit(node)


class CurrentDefaultCompatibilityTests(unittest.TestCase):
    def test_current_plugin_default_branch_ast_equal_pr34(self):
        path = 'src/megartx/vllm_scale_plugin.py'
        current = DefaultPurpose().visit(ast.parse((ROOT/path).read_text()))
        self.assertEqual(dump(current), dump(previous(path)))

    def test_historical_prefill_launcher_default_branch_ast_equal_pr34(self):
        # This remains a whole-file historical equivalence check. The current
        # shared launcher also carries newer decode safeguards from main.
        from test_shared_launcher_composition import parent_source
        historical = DefaultPurpose().visit(ast.parse(parent_source('prefill')))
        self.assertEqual(dump(historical), dump(previous('scripts/run_scale_validation.py')))

    def test_current_launcher_purpose_lifecycles_equal_exact_pinned_parents(self):
        from test_shared_launcher_composition import SelectPurpose, parent_source, run_node
        self.assertEqual(dump(SelectPurpose(False).visit(run_node())),
                         dump(run_node(parent_source('main'))))
        for storage_purpose in (False, True):
            current = SelectPurpose(True, storage_purpose).visit(run_node())
            prior = SelectPurpose(True, storage_purpose).visit(run_node(parent_source('prefill')))
            self.assertEqual(dump(current), dump(prior))

    def test_current_native_default_provider_ast_equal_pr34(self):
        path='src/megartx/prefill_native.py'
        current=NativeDefault().visit(ast.parse((ROOT/path).read_text()))
        self.assertEqual(dump(current), dump(previous(path)))

    def test_legacy_plan_refactor_preserves_complete_validation_ast(self):
        path='src/megartx/prefill_diagnostic_plan.py'
        old=previous(path)
        current=ast.parse((ROOT/path).read_text())
        old_functions={x.name:x for x in old.body if isinstance(x,ast.FunctionDef)}
        new_functions={x.name:x for x in current.body if isinstance(x,ast.FunctionDef)}
        old_body=old_functions['load_plan'].body
        new_body=new_functions['load_plan'].body
        self.assertEqual([dump(x) for x in old_body[:2]], [dump(x) for x in new_body[:2]])
        self.assertEqual([dump(x) for x in old_body[3:]], [dump(x) for x in new_functions['validate_plan'].body[1:]])
        for name,node in old_functions.items():
            if name != 'load_plan': self.assertEqual(dump(node),dump(new_functions[name]),name)

    def test_current_default_plan_and_client_still_bind_exact_current_sources(self):
        plan=legacy.freeze_plan(list(range(2048)), '3'*40, ROOT, {
            'config_sha256': CONFIG_SHA256,'index_sha256':'a'*64,
            'shard_stats': {'synthetic.safetensors': {'size':1,'mtime_ns':1}}})
        self.assertIs(legacy.validate_plan(plan,ROOT),plan)
        self.assertEqual(len(legacy.verify_adapter_sources(plan)), 14)
        with self.assertRaises(ValueError): storage.validate_plan(plan,ROOT)
        alternate=copy.deepcopy(plan);alternate['schema']=storage.SCHEMA
        with self.assertRaises(ValueError): legacy.validate_plan(alternate,ROOT)

    def test_no_mode_or_incomplete_storage_selection_imports_runtime(self):
        code='import megartx.vllm_scale_plugin as p,sys; p.install(); assert "torch" not in sys.modules and "vllm" not in sys.modules'
        import os
        env={k:v for k,v in os.environ.items() if not k.startswith('MEGARTX_')}
        result=subprocess.run([sys.executable,'-c',code],env=env,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for client in ('prefill-native','prefill-storage'):
            result=subprocess.run([sys.executable,'-S','scripts/run_scale_validation.py','--mode','native',
                '--label','cpu-rejected','--client',client],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertNotIn('ModuleNotFoundError',result.stderr)


if __name__=='__main__': unittest.main()

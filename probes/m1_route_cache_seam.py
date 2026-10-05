"""Uninstalled, exact-source CPU candidate for Triton's actual cache lookup.

This module imports no accelerator runtime and changes no class, instance,
module, cache, hook, or file. The factory compiles the already reviewed source
excerpt against an explicitly supplied CPU namespace and returns a plain
function. It is NOT a wrapper for an arbitrary loaded JIT function, evidence of
installed bytes, or permission to install the candidate in a worker.

The only executable AST delta wraps the result of the original expression
``kernel_cache.get(key, None)``. That original expression executes once, before
the observer sees its actual cache, key, selected object and local compilation
inputs. The binder and key calculation remain in the original body, once each.
Cold/unknown selections still follow that body's original compile/launch path.
An observer failure is diagnostic only and never supplies a replacement result,
retries an operation, formats an exception, or changes exception propagation.

The observer is a trusted, read-only callback: receiving live references does
not sandbox arbitrary Python that deliberately mutates those objects. A future
adapter must reject evidence after any callback failure and independently prove
loaded callable/global/source ownership. No proof here qualifies GPU execution,
loaded native metadata, route-check elision, or an installation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import textwrap
from types import FunctionType


# Exact UTF-8 JITFunction.run slice, including its original indentation.
# Triton f797708c0626e5f9840ca5b0a98790e2c7cb09ad,
# python/triton/runtime/jit.py:708-763. This is a reference pin, NOT a live pin.
SUPPORTED_JIT_RUN_SHA256 = "8686893714b9aef17074011b857f5a40eab9a545a35976d7eea895b7d352221d"
_OBSERVER_NAME = "__m1_route_actual_cache_selection__"
_LOOKUP_TEXT = "kernel_cache.get(key, None)"
_LOCAL_NAMES = ("kernel_cache", "key", "target", "specialization", "options")
_REFERENCE_FILENAME = (
    "reference:f797708c0626e5f9840ca5b0a98790e2c7cb09ad:python/triton/runtime/jit.py"
)


class CacheSeamSourceError(ValueError):
    """The supplied source/delta is not the one explicitly reviewed candidate."""


def _dump(tree):
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checked_source(source, expected_sha256):
    # An arbitrary source accompanied by its own recomputed digest is not an
    # admitted version. A different upstream version requires a new review.
    if type(source) is not str or type(expected_sha256) is not str:
        raise CacheSeamSourceError("source and expected digest must be exact strings")
    if expected_sha256 != SUPPORTED_JIT_RUN_SHA256:
        raise CacheSeamSourceError("unreviewed JIT source digest")
    if _digest(source) != expected_sha256:
        raise CacheSeamSourceError("JIT source bytes differ from the reviewed slice")
    tree = ast.parse(textwrap.dedent(source))
    if (len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef
            or tree.body[0].name != "run"):
        raise CacheSeamSourceError("expected exactly JITFunction.run")
    return tree


class _AddSelection(ast.NodeTransformer):
    def __init__(self):
        self.count = 0
        self.lookup = _dump(ast.parse(_LOOKUP_TEXT, mode="eval").body)

    def visit_Call(self, node):
        if _dump(node) != self.lookup:
            return self.generic_visit(node)
        self.count += 1
        # The first argument IS the complete original lookup expression. Python
        # evaluates it before the local references and before invoking the helper.
        # A primary get failure therefore bypasses the observer entirely.
        replacement = ast.Call(
            func=ast.Name(id=_OBSERVER_NAME, ctx=ast.Load()),
            args=[node] + [ast.Name(id=name, ctx=ast.Load()) for name in _LOCAL_NAMES],
            keywords=[],
        )
        return ast.copy_location(replacement, node)


class _RemoveSelection(ast.NodeTransformer):
    def __init__(self):
        self.count = 0
        self.expected_arguments = [
            _dump(ast.parse(_LOOKUP_TEXT, mode="eval").body),
            *[_dump(ast.Name(id=name, ctx=ast.Load())) for name in _LOCAL_NAMES],
        ]

    def visit_Call(self, node):
        if not (type(node.func) is ast.Name and node.func.id == _OBSERVER_NAME):
            return self.generic_visit(node)
        if node.keywords or [_dump(arg) for arg in node.args] != self.expected_arguments:
            raise CacheSeamSourceError("cache observation delta has changed")
        self.count += 1
        return node.args[0]


@dataclass(frozen=True)
class CacheSelectionProof:
    """Structural reference-source evidence only; never live admission evidence."""

    source_sha256: str
    source_ast_sha256: str
    candidate_ast_sha256: str
    restored_ast_sha256: str
    replacement_count: int
    evidence_kind: str = "reference-source-only"
    installed: bool = False
    loaded_bindings_verified: bool = False


def verify_cache_selection_restoration(source, transformed_source, *, expected_sha256):
    """Remove exactly the allowed observer delta and compare the entire AST.

    This independent check rejects changed arguments, moved/duplicated lookups,
    altered signatures, defaults, returns, exception paths and any other body
    edits. AST equality deliberately ignores formatting/line locations; exact
    original source bytes are separately pinned before this check.
    """
    original = _checked_source(source, expected_sha256)
    if type(transformed_source) is not str:
        raise CacheSeamSourceError("candidate source must be an exact string")
    try:
        candidate = ast.parse(transformed_source)
        restored = ast.parse(transformed_source)
    except SyntaxError as error:
        raise CacheSeamSourceError("candidate source is not valid Python") from error
    remover = _RemoveSelection()
    restored = remover.visit(restored)
    if remover.count != 1:
        raise CacheSeamSourceError("expected exactly one cache observation delta")
    if _dump(restored) != _dump(original):
        raise CacheSeamSourceError("removing the delta does not restore the complete source AST")
    return CacheSelectionProof(
        source_sha256=expected_sha256,
        source_ast_sha256=_digest(_dump(original)),
        candidate_ast_sha256=_digest(_dump(candidate)),
        restored_ast_sha256=_digest(_dump(restored)),
        replacement_count=remover.count,
    )


@dataclass
class _CallbackFailure:
    count: int = 0


@dataclass(frozen=True)
class CacheSelectionCandidate:
    """An uninstalled plain run function and its reference-only delta proof.

    Callback diagnostics are not receipts. A nonzero count invalidates any
    observation relying on this candidate, even if the main operation succeeds.
    Diagnostics retain only a scalar count and expose a constant failure marker,
    never an exception, exception type, traceback, or invocation owner.
    Both the supplied namespace and returned Python function remain ordinary
    Python objects; this factory is not a security boundary against mutation.
    """

    run: FunctionType
    proof: CacheSelectionProof
    transformed_source: str
    _failure: _CallbackFailure = field(repr=False, compare=False)

    @property
    def callback_failures(self):
        return self._failure.count

    @property
    def last_callback_failure(self):
        """Constant marker, or None; deliberately excludes exception details."""
        return "observation_callback_failed" if self._failure.count else None


def build_cache_selection_candidate(source, *, expected_sha256, namespace, observe_selection):
    """Return an uninstalled exact-source CPU candidate and restoration proof.

    ``namespace`` must explicitly provide CPU implementations of ``driver``,
    ``knobs`` and ``compute_cache_key``. It is shallow-copied, never mutated.
    ``observe_selection`` receives keyword arguments ``cache``, ``key``,
    ``selected``, ``target``, ``specialization`` and ``options`` by identity at the
    actual lookup, including selected=None for cold paths. Its return is ignored.
    Every BaseException from the callback increments a scalar failure count and
    is ignored by the original execution; neither the exception nor its type or
    traceback is retained. Errors from original operations still propagate.

    This does not inspect or import installed Triton, bind a loaded JIT object,
    access a lazy property, query a device, warm a kernel, or install anything.
    """
    tree = _checked_source(source, expected_sha256)
    if type(namespace) is not dict or any(type(key) is not str for key in namespace):
        raise TypeError("namespace must be an exact dict with string keys")
    if _OBSERVER_NAME in namespace:
        raise CacheSeamSourceError("namespace already contains the private observation helper")
    if not {"driver", "knobs", "compute_cache_key"}.issubset(namespace):
        raise ValueError("supply explicit CPU driver, knobs and compute_cache_key globals")
    if not callable(observe_selection):
        raise TypeError("observe_selection must be callable")

    transformer = _AddSelection()
    tree = transformer.visit(tree)
    if transformer.count != 1:
        raise CacheSeamSourceError("expected exactly one original kernel-cache lookup")
    ast.fix_missing_locations(tree)
    transformed_source = ast.unparse(tree) + "\n"
    proof = verify_cache_selection_restoration(
        source, transformed_source, expected_sha256=expected_sha256,
    )
    failure = _CallbackFailure()

    def selected_at_actual_lookup(selected, cache, key, target, specialization, options):
        try:
            observe_selection(cache=cache, key=key, selected=selected, target=target,
                              specialization=specialization, options=options)
        except BaseException:
            # Never retain, inspect, format, annotate or interact with the
            # exception. Its traceback would retain helper/run frames and bound
            # tensor/storage owners after the invocation has finished.
            failure.count += 1
        return selected

    globals_ = dict(namespace)
    globals_.setdefault("__name__", "m1_route_cache_source_candidate")
    globals_[_OBSERVER_NAME] = selected_at_actual_lookup
    ast.increment_lineno(tree, 707)
    exec(compile(tree, _REFERENCE_FILENAME, "exec"), globals_)
    return CacheSelectionCandidate(globals_["run"], proof, transformed_source, failure)

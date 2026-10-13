"""Sandboxed evaluation of user-supplied curve expressions.

Users may supply any profile (bow, fret heights, custom action curves...) as a small
math expression in ``x`` (mm from nut).  We evaluate it with a locked-down namespace:
no builtins, only ``math``/``numpy`` and a few helpers.  This is *not* a security
boundary against a hostile config file — configs come from the shop's own users —
but it prevents accidents and keeps expressions readable.
"""
from __future__ import annotations

import ast
import math
from typing import Callable

import numpy as np

# Allowed AST node types for expressions.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Call, ast.Name, ast.Load, ast.Constant, ast.Attribute,
    ast.IfExp, ast.Tuple, ast.List,
)

_SAFE_ATTRS = {"math", "np", "numpy", "min", "max", "abs", "where", "clip"}


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SyntaxError(f"disallowed syntax in expression: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            # allow math.sin etc but nothing dunder-y
            if node.attr.startswith("_"):
                raise SyntaxError("private attribute access not allowed")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.value is not None:
                base = f.value
                if isinstance(base, ast.Name) and base.id not in _SAFE_ATTRS:
                    raise SyntaxError(f"cannot call methods on '{base.id}'")
            if isinstance(f, ast.Name) and f.id not in _SAFE_ATTRS:
                raise SyntaxError(f"unknown function '{f.id}'")


def make_namespace() -> dict:
    ns: dict = {
        "math": math,
        "np": np,
        "numpy": np,
        "min": np.minimum,   # elementwise-friendly
        "max": np.maximum,
        "abs": np.abs,
        "where": np.where,
        "clip": np.clip,
        "pi": math.pi,
        "e": math.e,
    }
    for name in ("sin cos tan asin acos atan atan2 sinh cosh tanh exp log log2 log10 "
                 "sqrt cbrt floor ceil fabs pow hypot").split():
        ns[name] = getattr(math, name)
    return ns


def compile_expr(expr: str) -> Callable[[np.ndarray | float], np.ndarray]:
    """Compile an expression string into a vectorized callable f(x)."""
    tree = ast.parse(expr.strip(), mode="eval")
    _check_ast(tree)
    code = compile(tree, "<expr>", "eval")
    ns = make_namespace()

    def f(x):
        xs = np.asarray(x, dtype=float)
        env = dict(ns)
        env["x"] = xs
        result = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - controlled namespace
        return np.asarray(result, dtype=float) * np.ones_like(xs) + 0.0 \
            if np.isscalar(result) or np.ndim(result) == 0 else np.asarray(result, dtype=float)

    return f

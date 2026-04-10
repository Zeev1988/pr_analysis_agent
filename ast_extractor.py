"""
Part 2 – AST Context Slicer
Given a file's full source and the line numbers that changed in the PR diff,
use the AST to locate the enclosing function(s) and return their complete
raw source text — avoiding both the "Context Cliff" (too little context) and
the "Noise Flood" (too much context from unrelated code).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Known dangerous call methods — used to flag slices before sending to LLM
# ---------------------------------------------------------------------------

SINK_METHODS = {
    "execute", "executemany",          # DB execution
    "system", "popen", "run", "call",  # subprocess / shell
    "eval", "exec",                    # code injection
    "loads",                           # pickle / yaml deserialization
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class FunctionSlice:
    """
    A complete function extracted from a risky file.

    - raw_source: the entire function as written, ready to paste into an LLM prompt.
    - changed_lines: which of the PR's diff lines fall inside this function,
      so the LLM knows exactly what the developer touched.
    - has_sink: quick pre-screen flag — True if any known dangerous call exists
      anywhere in the function, not just on the changed lines.
    """
    filename: str
    function_name: str
    params: list[str]
    start_line: int
    end_line: int
    raw_source: str
    changed_lines: list[int]
    has_sink: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_call_chain(node: ast.expr) -> str:
    """Resolve a (possibly chained) call target to a dotted string."""
    if isinstance(node, ast.Attribute):
        return f"{_resolve_call_chain(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return "<expr>"


def _function_has_sink(func_node: ast.FunctionDef) -> bool:
    """Return True if any call inside the function matches a known sink method."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            chain = _resolve_call_chain(node.func)
            if chain.split(".")[-1] in SINK_METHODS:
                return True
    return False


def _collect_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Return all FunctionDef and AsyncFunctionDef nodes in the tree."""
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def slice_context(
    filename: str,
    source_code: str,
    changed_lines: list[int],
) -> list[FunctionSlice]:
    """
    Find every function in *source_code* that contains at least one of the
    *changed_lines* from the PR diff.

    Returns a list of FunctionSlice objects — one per matching function —
    each carrying the complete raw function text and metadata for the LLM.

    If the file has a syntax error or no changed lines fall inside any function
    (e.g. a change at module level), returns an empty list.
    """
    if not changed_lines:
        return []

    try:
        tree = ast.parse(source_code, filename=filename)
    except SyntaxError:
        return []

    source_lines = source_code.splitlines()
    changed_set = set(changed_lines)
    slices: list[FunctionSlice] = []

    for func in _collect_functions(tree):
        start, end = func.lineno, func.end_lineno
        hits = sorted(changed_set & set(range(start, end + 1)))
        if not hits:
            continue

        raw = "\n".join(source_lines[start - 1 : end])
        params = [arg.arg for arg in func.args.args]

        slices.append(FunctionSlice(
            filename=filename,
            function_name=func.name,
            params=params,
            start_line=start,
            end_line=end,
            raw_source=raw,
            changed_lines=hits,
            has_sink=_function_has_sink(func),
        ))

    return slices


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate the full file after the PR is applied.
    # The developer added line 4 (the request.args.get call).
    # The sink (cursor.execute) is on line 6 — NOT in the diff.
    database_py = """\
import os

def get_user(user_id):
    user_id = request.args.get('id')
    query = f'SELECT * FROM users WHERE id = {user_id}'
    cursor.execute(query)
    return cursor.fetchone()

def unrelated_helper():
    return 42
"""

    # Only line 4 was in the git diff
    changed = [4]

    results = slice_context("database.py", database_py, changed)

    for s in results:
        print(f"Function : {s.function_name}({', '.join(s.params)})")
        print(f"Lines    : {s.start_line}–{s.end_line}")
        print(f"Changed  : {s.changed_lines}")
        print(f"Has sink : {s.has_sink}")
        print(f"\n--- raw source ---\n{s.raw_source}")
        print("-" * 40)

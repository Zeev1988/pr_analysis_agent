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

    - raw_source:     the entire function after the PR (new-file coordinates).
    - old_raw_source: the function before the PR; None when there were no deletions.
    - changed_lines:  new-file line numbers added in the diff.
    - deleted_lines:  old-file line numbers removed in the diff (different coordinate
                      space from changed_lines — they index into old_raw_source).
    - has_sink:       True if any known dangerous call exists in the NEW source.
    """
    filename: str
    function_name: str
    params: list[str]
    start_line: int
    end_line: int
    raw_source: str
    changed_lines: list[int]
    has_sink: bool
    old_raw_source: str | None = None
    deleted_lines: list[int] = field(default_factory=list)


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


def _collect_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return (qualified_name, node) pairs for every function in the tree.

    Qualified names include the enclosing class, e.g. ``MyClass.validate``,
    so same-named methods in different classes don't collide in the dedup index.
    """
    results: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._class_stack.append(node.name)
            self.generic_visit(node)
            self._class_stack.pop()

        def _visit_func(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            prefix = ".".join(self._class_stack)
            qualified = f"{prefix}.{node.name}" if prefix else node.name
            results.append((qualified, node))
            self.generic_visit(node)

        visit_FunctionDef = _visit_func
        visit_AsyncFunctionDef = _visit_func

    _Visitor().visit(tree)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def slice_context(
    filename: str,
    source_code: str,
    changed_lines: list[int],
    *,
    old_source_code: str | None = None,
    deleted_lines: list[int] | None = None,
) -> list[FunctionSlice]:
    """
    Find every function that contains at least one changed or deleted line.

    changed_lines / source_code   — additions, resolved against the new file.
    deleted_lines / old_source_code — removals, resolved against the old file.
      Both old_source_code and deleted_lines must be provided together; if
      only one is given the deletions are silently ignored.

    Returns a list of FunctionSlice objects — one per matching function —
    each carrying the complete raw function text and metadata for the LLM.
    If a function has both additions and deletions, the new-file version is
    used as the canonical slice and the deleted hit lines are merged into its
    changed_lines so the LLM sees the full picture.
    """
    slices: list[FunctionSlice] = []
    # Maps function name → index in slices for O(1) merge lookups.
    index: dict[str, int] = {}

    def _process(code: str, lines: list[int], *, is_old: bool = False) -> None:
        if not lines:
            return
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError:
            return

        source_lines = code.splitlines()
        line_set = set(lines)

        for qualified_name, func in _collect_functions(tree):
            start, end = func.lineno, func.end_lineno
            hits = sorted(line_set & set(range(start, end + 1)))
            if not hits:
                continue

            raw = "\n".join(source_lines[start - 1 : end])

            if qualified_name in index:
                existing = slices[index[qualified_name]]
                if is_old:
                    # Store old-file hits in deleted_lines (separate coordinate space).
                    slices[index[qualified_name]] = FunctionSlice(
                        filename=existing.filename,
                        function_name=existing.function_name,
                        params=existing.params,
                        start_line=existing.start_line,
                        end_line=existing.end_line,
                        raw_source=existing.raw_source,
                        changed_lines=existing.changed_lines,
                        has_sink=existing.has_sink,
                        old_raw_source=raw,
                        deleted_lines=hits,
                    )
                else:
                    # Two additions in the same new-file function — merge.
                    merged = sorted(set(existing.changed_lines) | set(hits))
                    slices[index[qualified_name]] = FunctionSlice(
                        filename=existing.filename,
                        function_name=existing.function_name,
                        params=existing.params,
                        start_line=existing.start_line,
                        end_line=existing.end_line,
                        raw_source=existing.raw_source,
                        changed_lines=merged,
                        has_sink=existing.has_sink,
                        old_raw_source=existing.old_raw_source,
                        deleted_lines=existing.deleted_lines,
                    )
            else:
                params = [arg.arg for arg in func.args.args]
                index[qualified_name] = len(slices)
                slices.append(FunctionSlice(
                    filename=filename,
                    function_name=qualified_name,
                    params=params,
                    start_line=start,
                    end_line=end,
                    raw_source=raw,
                    # has_sink is only meaningful for new-file source.
                    has_sink=_function_has_sink(func) if not is_old else False,
                    changed_lines=[] if is_old else hits,
                    old_raw_source=raw if is_old else None,
                    deleted_lines=hits if is_old else [],
                ))

    _process(source_code, changed_lines)

    if old_source_code is not None and deleted_lines:
        _process(old_source_code, deleted_lines, is_old=True)

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

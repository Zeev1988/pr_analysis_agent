"""
Part 1b – Diff Extractor
Parse a unified git diff and return the added/modified line numbers per file.
Addition lines (+) are tracked against the new file; deletion lines (-) are
tracked against the old file so callers can analyse removed code as well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FileDiff:
    """Changed-line metadata for a single file in the PR diff."""
    filename: str
    changed_lines: list[int] = field(default_factory=list)
    deleted_lines: list[int] = field(default_factory=list)


def parse_diff(diff_text: str) -> dict[str, FileDiff]:
    """
    Parse unified diff text (e.g. from `git diff`) into a mapping of
    filename → FileDiff.

    changed_lines: new-file line numbers that were added (+).
    deleted_lines: old-file line numbers that were removed (-), so callers
                   can locate removed code in the pre-merge AST.
    """
    results: dict[str, FileDiff] = {}
    current: FileDiff | None = None
    new_line = 0
    old_line = 0

    for raw in diff_text.splitlines():
        if raw.startswith("+++ b/"):
            filename = raw[6:]
            current = FileDiff(filename=filename)
            results[filename] = current
            new_line = 0
            old_line = 0

        elif raw.startswith("@@"):
            # Format: @@ -old_start[,old_count] +new_start[,new_count] @@
            m_old = re.search(r"-(\d+)(?:,\d+)?", raw)
            m_new = re.search(r"\+(\d+)(?:,\d+)?", raw)
            if m_old:
                old_line = int(m_old.group(1)) - 1
            if m_new:
                new_line = int(m_new.group(1)) - 1

        elif raw.startswith("+") and not raw.startswith("+++") and current is not None:
            new_line += 1
            current.changed_lines.append(new_line)

        elif raw.startswith("-") and not raw.startswith("---") and current is not None:
            old_line += 1
            current.deleted_lines.append(old_line)

        elif raw.startswith(" ") and current is not None:
            # Unchanged context line — advances both counters.
            new_line += 1
            old_line += 1

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_diff = """\
diff --git a/database.py b/database.py
index abc1234..def5678 100644
--- a/database.py
+++ b/database.py
@@ -8,6 +8,7 @@ def setup():
     pass
 
 def get_user(user_id):
+    user_id = request.args.get('id')
     query = f'SELECT * FROM users WHERE id = {user_id}'
     cursor.execute(query)
     return cursor.fetchone()
"""

    diffs = parse_diff(sample_diff)
    for fname, fd in diffs.items():
        print(f"{fname}: changed lines → {fd.changed_lines}")

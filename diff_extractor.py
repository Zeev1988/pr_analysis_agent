"""
Part 1b – Diff Extractor
Parse a unified git diff and return the added/modified line numbers per file.
Only addition lines (+) are tracked — these are the lines the developer wrote
and the ones we need to locate inside the AST.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FileDiff:
    """Changed-line metadata for a single file in the PR diff."""
    filename: str
    changed_lines: list[int] = field(default_factory=list)


def parse_diff(diff_text: str) -> dict[str, FileDiff]:
    """
    Parse unified diff text (e.g. from `git diff`) into a mapping of
    filename → FileDiff containing the new-file line numbers that were added.

    Removed lines (-) are ignored: they no longer exist in the new file and
    cannot be located in the AST of the post-merge code.
    """
    results: dict[str, FileDiff] = {}
    current: FileDiff | None = None
    new_line = 0

    for raw in diff_text.splitlines():
        if raw.startswith("+++ b/"):
            # Start of a new file block
            filename = raw[6:]
            current = FileDiff(filename=filename)
            results[filename] = current
            new_line = 0

        elif raw.startswith("@@"):
            # Hunk header — tells us which line in the new file this hunk starts at.
            # Format: @@ -old_start[,old_count] +new_start[,new_count] @@
            m = re.search(r"\+(\d+)(?:,\d+)?", raw)
            if m:
                # Subtract 1 because we increment before recording each line.
                new_line = int(m.group(1)) - 1

        elif raw.startswith("+") and not raw.startswith("+++") and current is not None:
            # Added line — increment first so line numbers are 1-based.
            new_line += 1
            current.changed_lines.append(new_line)

        elif raw.startswith(" ") and current is not None:
            # Unchanged context line — still advances the new-file counter.
            new_line += 1

        # Lines starting with "-" are removals; they don't exist in the new
        # file so we skip them without incrementing new_line.

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

"""
End-to-end pipeline demo:
  git diff → diff_extractor → fast_filter → ast_extractor (context slicer) → llm_evaluator

Shows how a change that only touches the *source* (user input line) causes the
slicer to return the full function — including the *sink* that was never in the diff.
Also demonstrates deletion tracking: functions where lines were removed are also
sent to the LLM using the pre-merge file content.
"""

from diff_extractor import parse_diff
from fast_filter import classify_file
from ast_extractor import slice_context
from llm_evaluator import evaluate_slice

# ---------------------------------------------------------------------------
# Simulate a PR: one added line in database.py, one added line in readme.md
# ---------------------------------------------------------------------------

FAKE_DIFF = """\
diff --git a/database.py b/database.py
index abc1234..def5678 100644
--- a/database.py
+++ b/database.py
@@ -2,7 +2,7 @@

 def get_user(user_id):
-    user_id = sanitize(request.args.get('id'))
+    user_id = request.args.get('id')
     query = f'SELECT * FROM users WHERE id = {user_id}'
     cursor.execute(query)
     return cursor.fetchone()
diff --git a/readme.md b/readme.md
index 1111111..2222222 100644
--- a/readme.md
+++ b/readme.md
@@ -1,3 +1,4 @@
 # My App
+Added new user lookup endpoint.
"""

# Simulate the full post-merge file content for each changed file.
FILE_CONTENTS = {
    "database.py": """\
import os

def get_user(user_id):
    user_id = request.args.get('id')
    query = f'SELECT * FROM users WHERE id = {user_id}'
    cursor.execute(query)
    return cursor.fetchone()

def unrelated_helper():
    return 42
""",
    "readme.md": """\
# My App
Added new user lookup endpoint.
""",
}

# Simulate the pre-merge file content (needed to resolve deleted lines).
OLD_FILE_CONTENTS = {
    "database.py": """\
import os

def get_user(user_id):
    user_id = sanitize(request.args.get('id'))
    query = f'SELECT * FROM users WHERE id = {user_id}'
    cursor.execute(query)
    return cursor.fetchone()

def unrelated_helper():
    return 42
""",
}

# ---------------------------------------------------------------------------
# Step 1: Parse the diff
# ---------------------------------------------------------------------------

print("=" * 50)
print("STEP 1 — Diff Extractor")
print("=" * 50)

file_diffs = parse_diff(FAKE_DIFF)
for fname, fd in file_diffs.items():
    print(f"  {fname}: added lines → {fd.changed_lines}  deleted lines → {fd.deleted_lines}")

# ---------------------------------------------------------------------------
# Step 2: Fast filter — discard safe files before touching the AST
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("STEP 2 — Fast Filter")
print("=" * 50)

risky_diffs = {}
for fname, fd in file_diffs.items():
    content = FILE_CONTENTS.get(fname, "")
    verdict = classify_file(fname, content)
    print(f"  {fname}: {verdict}")
    if verdict == "RISKY":
        risky_diffs[fname] = fd

# ---------------------------------------------------------------------------
# Step 3: AST Context Slicer — extract the enclosing function for each change
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("STEP 3 — AST Context Slicer")
print("=" * 50)

for fname, fd in risky_diffs.items():
    content = FILE_CONTENTS[fname]
    old_content = OLD_FILE_CONTENTS.get(fname)
    slices = slice_context(
        fname, content, fd.changed_lines,
        old_source_code=old_content,
        deleted_lines=fd.deleted_lines,
    )

    for s in slices:
        print(f"\n  Function : {s.function_name}({', '.join(s.params)})")
        print(f"  Lines    : {s.start_line}–{s.end_line}")
        print(f"  Changed  : {s.changed_lines}")
        print(f"  Has sink : {s.has_sink}")
        print(f"\n  ┌── full function sent to LLM ──────────────────────")
        for line in s.raw_source.splitlines():
            print(f"  │  {line}")
        print(f"  └───────────────────────────────────────────────────")

print("\n✓ LLM receives the complete function — additions AND deletions covered.")

# ---------------------------------------------------------------------------
# Step 4: LLM Evaluator — send each slice to the LLM for a security verdict
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("STEP 4 — LLM Evaluator")
print("=" * 50)

for fname, fd in risky_diffs.items():
    content = FILE_CONTENTS[fname]
    old_content = OLD_FILE_CONTENTS.get(fname)
    slices = slice_context(
        fname, content, fd.changed_lines,
        old_source_code=old_content,
        deleted_lines=fd.deleted_lines,
    )
    for s in slices:
        print(f"\n  Evaluating {s.function_name}() …")
        verdict = evaluate_slice(s)
        print(f"  Verdict  : {verdict.verdict}")
        print(f"  Type     : {verdict.vulnerability_type}")
        print(f"  Confidence: {verdict.confidence:.2f}")
        print(f"  Reasoning: {verdict.reasoning}")

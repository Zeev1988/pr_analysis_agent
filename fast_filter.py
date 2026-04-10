"""
Part 1 – Fast Filter
Cheaply route PR files into RISKY or SAFE buckets before any expensive analysis.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

RISKY_EXTENSIONS = {".py", ".js", ".ts", ".rb", ".php", ".go", ".java", ".cs"}

SAFE_EXTENSIONS = {".md", ".txt", ".rst", ".css", ".scss", ".less",
                   ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
                   ".json", ".yaml", ".yml", ".toml", ".lock"}

# Substrings that hint at security-sensitive logic regardless of extension.
RISKY_CONTENT_PATTERNS = [
    "execute(",       # raw SQL execution
    "cursor.execute", # DB cursors
    "f'SELECT",       # f-string SQL
    'f"SELECT',
    "subprocess",     # shell injection
    "os.system",
    "eval(",          # code injection
    "exec(",
    "pickle.loads",   # deserialisation
    "__import__",
    "open(",          # file I/O (path traversal)
    "request.args",   # unsanitised HTTP input
    "request.form",
]


def classify_file(filename: str, content: str) -> str:
    """Return 'RISKY' or 'SAFE' for a single file."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in SAFE_EXTENSIONS:
        return "SAFE"

    if ext in RISKY_EXTENSIONS:
        return "RISKY"

    # Unknown extension – fall back to content scanning.
    for pattern in RISKY_CONTENT_PATTERNS:
        if pattern in content:
            return "RISKY"

    return "SAFE"


def run_fast_filter(pr_files: dict[str, str]) -> dict[str, list[str]]:
    """
    Classify every file in the PR payload.

    Returns:
        {"risky": [...filenames...], "safe": [...filenames...]}
    """
    risky: list[str] = []
    safe: list[str] = []

    for filename, content in pr_files.items():
        bucket = classify_file(filename, content)
        if bucket == "RISKY":
            risky.append(filename)
        else:
            safe.append(filename)

    return {"risky": risky, "safe": safe}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pull_request_files = {
        "readme.md": "Updated the installation instructions.",
        "styles.css": "body { background-color: #f0f0f0; }",
        "database.py": (
            "def get_user(user_id):\n"
            "    query = f'SELECT * FROM users WHERE id = {user_id}'\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()"
        ),
        "utils.py": "def add(a, b):\n    return a + b",
    }

    result = run_fast_filter(pull_request_files)

    print("=== Fast Filter Results ===")
    print(f"  RISKY ({len(result['risky'])}): {result['risky']}")
    print(f"  SAFE  ({len(result['safe'])}):  {result['safe']}")
    print()
    print("Only risky files will be forwarded to the AST Extractor (Part 2).")

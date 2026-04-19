from pathlib import Path


HOOK_CONTENT = """#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
script = repo_root / "scripts" / "normalize_notebooks.py"

result = subprocess.run([sys.executable, str(script), "Exercise1", "Exercise2"], cwd=repo_root)
raise SystemExit(result.returncode)
"""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    hook_path = repo_root / ".git" / "hooks" / "pre-commit"
    hook_path.write_text(HOOK_CONTENT, encoding="utf-8")
    print(f"installed {hook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

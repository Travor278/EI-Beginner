import argparse
import json
from pathlib import Path


NOTEBOOK_SUFFIX = ".ipynb"
DEFAULT_KERNELSPEC = {
    "display_name": "python3",
    "language": "python",
    "name": "python3",
}
ALLOWED_LANGUAGE_INFO_KEYS = (
    "codemirror_mode",
    "file_extension",
    "mimetype",
    "name",
    "nbconvert_exporter",
    "pygments_lexer",
)


def normalize_notebook_data(notebook: dict) -> bool:
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

        if cell.get("metadata") != {}:
            cell["metadata"] = {}
            changed = True

    metadata = notebook.setdefault("metadata", {})

    if metadata.get("kernelspec") != DEFAULT_KERNELSPEC:
        metadata["kernelspec"] = dict(DEFAULT_KERNELSPEC)
        changed = True

    language_info = metadata.get("language_info", {})
    normalized_language_info = {
        key: language_info[key]
        for key in ALLOWED_LANGUAGE_INFO_KEYS
        if key in language_info
    }
    if metadata.get("language_info") != normalized_language_info:
        metadata["language_info"] = normalized_language_info
        changed = True

    return changed


def normalize_notebook_file(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = normalize_notebook_data(notebook)
    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def iter_notebooks(paths: list[str]) -> list[Path]:
    notebook_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            notebook_paths.extend(sorted(path.rglob(f"*{NOTEBOOK_SUFFIX}")))
        elif path.suffix == NOTEBOOK_SUFFIX and path.exists():
            notebook_paths.append(path)
    return notebook_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Jupyter notebooks to reduce machine-specific noise."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Notebook files or directories to normalize.",
    )
    args = parser.parse_args()

    notebook_paths = iter_notebooks(args.paths)
    changed_count = 0
    for path in notebook_paths:
        if normalize_notebook_file(path):
            changed_count += 1
            print(f"normalized {path}")

    if not notebook_paths:
        print("no notebooks found")
    elif changed_count == 0:
        print("notebooks already normalized")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

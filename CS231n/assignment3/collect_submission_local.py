from __future__ import annotations

import zipfile
from pathlib import Path

from makepdf import main as makepdf_main


NOTEBOOKS = (
    "Transformer_Captioning.ipynb",
    "Self_Supervised_Learning.ipynb",
    "CLIP_DINO.ipynb",
    "DDPM.ipynb",
)

ZIP_FILENAME = "a3_code_submission.zip"
PDF_FILENAME = "a3_inline_submission.pdf"


def verify_files(root: Path) -> None:
    required = list(NOTEBOOKS) + ["cs231n"]
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def build_zip(root: Path) -> None:
    zip_path = root / ZIP_FILENAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in NOTEBOOKS:
            zf.write(root / rel_path, arcname=rel_path)

        for src in root.rglob("*"):
            if not src.is_file():
                continue
            rel_path = src.relative_to(root).as_posix()
            if rel_path == "makepdf.py":
                continue
            if rel_path.startswith("html_exports/"):
                continue
            if src.suffix in {".py", ".pyx", ".ipynb"}:
                zf.write(src, arcname=rel_path)

    print(f"Created {ZIP_FILENAME}.")


def main() -> int:
    root = Path(__file__).resolve().parent
    verify_files(root)
    build_zip(root)

    try:
        makepdf_main(list(NOTEBOOKS), PDF_FILENAME)
    except Exception as exc:
        print(
            "\nPDF generation did not complete locally.\n"
            "The code zip is ready, but the PDF step usually needs Pandoc and a TeX engine.\n"
            f"Details: {exc}"
        )
        return 1

    print(f"### Done! Please submit {ZIP_FILENAME} and {PDF_FILENAME} to Gradescope. ###")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

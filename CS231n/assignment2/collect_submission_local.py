from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from makepdf import main as makepdf_main


CODE = (
    "cs231n/layers.py",
    "cs231n/classifiers/fc_net.py",
    "cs231n/optim.py",
    "cs231n/solver.py",
    "cs231n/classifiers/cnn.py",
    "cs231n/im2col_cython.pyx",
    "cs231n/classifiers/rnn_pytorch.py",
)

NOTEBOOKS = (
    "BatchNormalization.ipynb",
    "Dropout.ipynb",
    "ConvolutionalNetworks.ipynb",
    "PyTorch.ipynb",
    "RNN_Captioning_pytorch.ipynb",
)

ZIP_FILENAME = "a2_code_submission.zip"
PDF_FILENAME = "a2_inline_submission.pdf"


def verify_files(root: Path) -> None:
    required = list(CODE) + list(NOTEBOOKS)
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
            rel_path = src.relative_to(root)
            rel_posix = rel_path.as_posix()
            if rel_posix == "makepdf.py":
                continue
            if rel_posix.startswith("html_exports/"):
                continue
            if src.suffix in {".py", ".pyx"}:
                zf.write(src, arcname=rel_posix)

        saved_dir = root / "cs231n" / "saved"
        if saved_dir.exists():
            for saved_file in saved_dir.rglob("*"):
                if saved_file.is_file():
                    rel_path = saved_file.relative_to(root)
                    zf.write(saved_file, arcname=rel_path.as_posix())

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

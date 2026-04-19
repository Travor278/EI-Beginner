import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from pypdf import PdfMerger

    MERGE = True
except ImportError:
    try:
        from PyPDF2 import PdfMerger

        MERGE = True
    except ImportError:
        print("Could not find pypdf/PyPDF2. Leaving pdf files unmerged.")
        MERGE = False


def render_pdf(notebook):
    env = os.environ.copy()
    fallback_tool_dirs = [
        Path("D:/Tools/Pandoc"),
        Path("D:/Tools/MiKTeX/miktex/bin/x64"),
        Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64",
    ]
    existing_dirs = [str(path) for path in fallback_tool_dirs if path.exists()]
    if existing_dirs:
        env["PATH"] = os.pathsep.join(existing_dirs + [env.get("PATH", "")])

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--log-level",
        "CRITICAL",
        "--to",
        "pdf",
        notebook,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Failed to convert {} to PDF. Install Pandoc and a TeX engine "
            "(for example MiKTeX or TinyTeX) and retry.\n{}".format(
                notebook, details or "No additional error details were reported."
            )
        )


def main(files, pdf_name):
    for f in files:
        render_pdf(f)
        print("Created PDF {}.".format(f))
    if MERGE:
        pdfs = [f.split(".")[0] + ".pdf" for f in files]
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(pdf_name)
        merger.close()
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We pass in a explicit notebook arg so that we can provide an ordered list
    # and produce an ordered PDF.
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()
    main(args.notebooks, args.pdf_filename)

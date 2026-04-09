# CS231n Assignment 1 Local Setup

This folder was adapted for local Jupyter use on Windows/Anaconda.

Recommended environment:
- `py313`

Why `py313`:
- It already has `numpy`, `scipy`, `matplotlib`, `jupyter`, and `ipykernel`.
- `py311` is missing notebook support.
- `py312` is missing several scientific packages.
- `base` also works, but `py313` is a cleaner project environment.

Start from PowerShell:

```powershell
conda activate py313
cd D:\Code\Learning\EI\EI-learning-notes\CS231n\assignment1
jupyter notebook
```

Notes:
- The notebooks now auto-download datasets with a Python script instead of `bash get_datasets.sh`.
- Legacy `future/past.xrange` imports were removed so the code runs in plain Python 3.
- `collect_submission.ipynb` now calls a local Python submission helper instead of Colab commands.

Optional tools for local PDF export:
- `pypdf` or `PyPDF2` for merging notebook PDFs
- Pandoc
- A TeX engine such as MiKTeX or TinyTeX

Official references used for this local adaptation:
- 2025 A1 Colab package: `assignments/2025/assignment1_colab.zip`
- 2020 A1 Jupyter package: `assignments/2020/assignment1_jupyter.zip`

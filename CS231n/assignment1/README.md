# CS231n Assignment 1

This folder contains a local Jupyter-friendly version of CS231n Assignment 1.
The notebooks were adapted to run outside Colab while keeping the official
workflow and file layout as intact as possible.

## Environment

Any recent Python 3 environment with the usual scientific notebook stack should
work. A setup with the following packages is recommended:

- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`
- `ipykernel`

If you use Conda, activating your environment and starting Jupyter from this
folder is enough:

```powershell
conda activate <your-env>
cd D:\Code\Learning\EI\EI-learning-notes\CS231n\assignment1
jupyter notebook
```

## Local adaptations

- Dataset download is handled by a local Python helper instead of shell-only
  Colab setup.
- Legacy compatibility imports such as `future` / `past.xrange` were removed
  where they were no longer needed for Python 3.
- `collect_submission.ipynb` now calls a local Python submission helper instead
  of Colab commands.

## Optional tools

These are only needed if you want local PDF export:

- `pypdf` or `PyPDF2` for merging notebook PDFs
- `pandoc`
- A TeX engine such as MiKTeX or TinyTeX

## References

This local adaptation was based on the official CS231n assignment materials,
primarily the 2025 Colab package and the older 2020 Jupyter package.

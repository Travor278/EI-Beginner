# CS231n Assignment 2

This folder contains a local Jupyter-friendly version of CS231n Assignment 2.
The notebooks were adapted to run outside Colab while staying close to the
official notebook flow.

## Environment

Any recent Python 3 environment with the standard scientific notebook stack
should work. A setup with the following packages is recommended:

- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`
- `ipykernel`

If you use Conda, a typical workflow looks like:

```powershell
conda activate <your-env>
cd D:\Code\Learning\EI\EI-learning-notes\CS231n\assignment2
jupyter notebook
```

## Local adaptations

- Notebook setup cells use local paths instead of Colab / Google Drive mount.
- Dataset download is handled by `cs231n/datasets/get_datasets.py`.
- `collect_submission.ipynb` calls a local Python submission helper instead of
  Colab shell commands.
- `makepdf.py` is aligned with the local Assignment 1 workflow.

## Optional tools

These are only needed if you want local PDF export:

- `pypdf` or `PyPDF2` for merging notebook PDFs
- `pandoc`
- A TeX engine such as MiKTeX or TinyTeX

## References

This local adaptation was based on the official CS231n assignment materials and
kept intentionally close to the original structure used in the course.

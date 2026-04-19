# CS231n Assignment 3

This folder contains a local Jupyter-friendly version of CS231n Assignment 3.
The notebooks were adapted to run outside Colab while staying close to the
official notebook flow.

## Environment

Any recent Python 3 environment with the standard scientific notebook stack
should work. In this workspace, `py313` is the recommended environment.

Typical workflow:

```powershell
conda activate py313
cd D:\Code\Learning\EI\EI-learning-notes\CS231n\assignment3
jupyter notebook
```

## Local adaptations

- Notebook setup cells use local paths instead of Colab / Google Drive mount.
- Dataset download is handled by `cs231n/datasets/get_datasets.py`.
- `collect_submission.ipynb` calls a local Python submission helper instead of
  Colab shell commands.
- `makepdf.py` is aligned with the local Assignment 1 / 2 workflow.
- `cs231n/image_utils.py` uses in-memory image loading for better Windows
  compatibility.

## Additional notebook dependencies

Some notebooks install extra packages on demand. The main ones are:

- `ftfy`
- `regex`
- `tqdm`
- `decord`
- OpenAI `CLIP`

These are installed from inside the notebooks with `%pip`, so they land in the
currently active kernel environment.

## Optional tools

These are only needed if you want local PDF export:

- `pypdf` or `PyPDF2` for merging notebook PDFs
- `pandoc`
- A TeX engine such as MiKTeX or TinyTeX

## References

This local adaptation was based on the official CS231n Assignment 3 starter
materials and kept intentionally close to the original structure used in the
course.

import copy
import json
import unittest
from pathlib import Path

from scripts.normalize_notebooks import normalize_notebook_data
from scripts.normalize_notebooks import normalize_notebook_file


class NormalizeNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {"keep": True},
                    "source": ["# Title\n"],
                },
                {
                    "cell_type": "code",
                    "execution_count": 7,
                    "metadata": {"collapsed": False},
                    "outputs": [{"output_type": "stream", "text": ["x\n"]}],
                    "source": ["print('x')\n"],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "py313",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.13.12",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 2,
        }

    def test_normalize_notebook_data_keeps_content_and_strips_noise(self) -> None:
        notebook = copy.deepcopy(self.notebook)

        changed = normalize_notebook_data(notebook)

        self.assertTrue(changed)
        self.assertEqual(notebook["cells"][0]["source"], ["# Title\n"])
        self.assertEqual(notebook["cells"][1]["source"], ["print('x')\n"])
        self.assertIsNone(notebook["cells"][1]["execution_count"])
        self.assertEqual(notebook["cells"][1]["outputs"], [])
        self.assertEqual(notebook["cells"][1]["metadata"], {})
        self.assertEqual(
            notebook["metadata"]["kernelspec"],
            {
                "display_name": "python3",
                "language": "python",
                "name": "python3",
            },
        )
        self.assertNotIn("version", notebook["metadata"]["language_info"])

    def test_normalize_notebook_file_is_idempotent(self) -> None:
        notebook_path = Path("tests") / "_tmp_sample.ipynb"
        try:
            notebook_path.write_text(
                json.dumps(self.notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )

            first_changed = normalize_notebook_file(notebook_path)
            second_changed = normalize_notebook_file(notebook_path)

            self.assertTrue(first_changed)
            self.assertFalse(second_changed)
        finally:
            notebook_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

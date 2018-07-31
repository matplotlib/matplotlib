from pathlib import Path
import subprocess
import tempfile

import pytest

nbformat = pytest.importorskip('nbformat')

# From https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/


def _notebook_run(nb_file):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
          "--ExecutePreprocessor.timeout=500",
          "--output", fout.name, nb_file]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
                     for output in cell["outputs"]
                     if output.output_type == "error"]
    return nb, errors


def test_ipynb():
    nb, errors = _notebook_run(
        str(Path(__file__).parent / 'test_nbagg_01.ipynb'))
    assert errors == []

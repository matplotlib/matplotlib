import pytest

from matplotlib.testing.decorators import check_figures_equal


def test_plot_directive_fails_on_exception(tmp_path):
    """
    Ensure plot directive does not silently swallow exceptions.
    """

    # Create minimal Sphinx project structure
    srcdir = tmp_path / "src"
    outdir = tmp_path / "out"
    srcdir.mkdir()
    outdir.mkdir()

    # conf.py
    (srcdir / "conf.py").write_text(
        """
extensions = ['matplotlib.sphinxext.plot_directive']
master_doc = 'index'
"""
    )

    # index.rst with failing plot
    (srcdir / "index.rst").write_text(
        """
Test plot failure
=================

.. plot::

   raise RuntimeError("boom")
"""
    )

    # Run sphinx-build and assert failure
    import sphinx.cmd.build

    with pytest.raises(SystemExit):
        sphinx.cmd.build.main([
            "-b", "html",
            "-W",            # treat warnings as errors
            "-T",            # show traceback
            str(srcdir),
            str(outdir),
        ])


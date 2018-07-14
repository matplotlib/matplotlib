import os
from fnmatch import fnmatch
from sphinx_gallery import binder

# Blacklist contains patterns for which no binder/pyodide links
# should be created
blacklist = ["*_sgskip.py",
             "*pipong.py",
             "*pgf*.py",
             "*svg_filter*.py",
             "*font_indexing.py",
             "*ftface_props.py",
             "*multipage_pdf.py",
             "*pick_event_demo2.py"]


# Rewrite "download" section
# {0} is .py link, {1} is ipynb link,
# {2} is additional rst text  (used here for additional links)
# {3} is ref_fname
dlcode = """
.. _sphx_glr_download_{3}:
\n.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example

  .. container:: sphx-glr-download
  
     :download:`Download Python source code: {0} <{0}>`\n
  .. container:: sphx-glr-download
  
     :download:`Download Jupyter notebook: {1} <{1}>`\n
     {2}
     """

def gen_pyodide_url(fname):
    base = "https://iodide.io/pyodide-demo/matplotlib-sideload.html?sideload="
    head, tail = os.path.split(fname)
    return base + "https://matplotlib.org/_downloads/" + tail

def gen_links(fname, binder_conf, gallery_conf):
    """Generate the RST + links for the additional links.
    This is a replaced version of the original gen_binder_rst.

    Parameters
    ----------
    fname: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict | None
        If a dictionary it must have the following keys:
        'url': The URL of the BinderHub instance that's running a Binder
            service.
        'org': The GitHub organization to which the documentation will be
            pushed.
        'repo': The GitHub repository to which the documentation will be
            pushed.
        'branch': The Git branch on which the documentation exists (e.g.,
            gh-pages).
        'dependencies': A list of paths to dependency files that match the
            Binderspec.
    Returns
    -------
    rst : str
        The reStructuredText for the Binder badge and pyodide link.
    """
    binder_conf = binder.check_binder_conf(binder_conf)
    binder_url = binder.gen_binder_url(fname, binder_conf, gallery_conf)
    pyodide_url = gen_pyodide_url(fname)

    for pattern in blacklist:
        if fnmatch(fname, pattern):
            return "\n"

    rst_binder = (
        "\n"
        "  .. container:: sphx-glr-download\n\n"
        "     `Open with binder (experimental) <{}>`_\n"
        ).format(binder_url)
    rst_pyodide = (
        "\n"
        "  .. container:: sphx-glr-download\n\n"
        "     `Open with pyodide (experimental) <{}>`_\n"
        ).format(pyodide_url)

    # leave binder out for now , because configuration is unclear
    return rst_pyodide #+ rst_binder


# Matplotlib documentation build configuration file, created by
# sphinx-quickstart on Fri May  2 12:33:25 2008.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import os
import shutil
import subprocess
import sys

import matplotlib
import sphinx

from datetime import datetime

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('.'))
sys.path.append('.')

# General configuration
# ---------------------

# Strip backslahes in function's signature
# To be removed when numpydoc > 0.9.x
strip_signature_backslash = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.plot_directive',
    'sphinxext.custom_roles',
    'sphinxext.github',
    'sphinxext.math_symbol_table',
    'sphinxext.missing_references',
    'sphinxext.mock_gui_toolkits',
    'sphinxext.skip_deprecated',
    'sphinx_copybutton',
]

exclude_patterns = ['api/api_changes/*', 'users/whats_new/*']


def _check_dependencies():
    names = {
        "colorspacious": 'colorspacious',
        "IPython.sphinxext.ipython_console_highlighting": 'ipython',
        "matplotlib": 'matplotlib',
        "numpydoc": 'numpydoc',
        "PIL.Image": 'pillow',
        "sphinx_copybutton": 'sphinx_copybutton',
        "sphinx_gallery": 'sphinx_gallery',
    }
    missing = []
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(names[name])
    if missing:
        raise ImportError(
            "The following dependencies are missing to build the "
            "documentation: {}".format(", ".join(missing)))
    if shutil.which('dot') is None:
        raise OSError(
            "No binary named dot - graphviz must be installed to build the "
            "documentation")

_check_dependencies()


# Import only after checking for dependencies.
# gallery_order.py from the sphinxext folder provides the classes that
# allow custom ordering of sections and subsections of the gallery
import sphinxext.gallery_order as gallery_order
# The following import is only necessary to monkey patch the signature later on
from sphinx_gallery import gen_rst

# On Linux, prevent plt.show() from emitting a non-GUI backend warning.
os.environ.pop("DISPLAY", None)

autosummary_generate = True

autodoc_docstring_signature = True
if sphinx.version_info < (1, 8):
    autodoc_default_flags = ['members', 'undoc-members']
else:
    autodoc_default_options = {'members': None, 'undoc-members': None}

# missing-references names matches sphinx>=3 behavior, so we can't be nitpicky
# for older sphinxes.
nitpicky = sphinx.version_info >= (3,)
# change this to True to update the allowed failures
missing_references_write_json = False
missing_references_warn_unused_ignores = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'cycler': ('https://matplotlib.org/cycler', None),
    'dateutil': ('https://dateutil.readthedocs.io/en/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'Pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'pytest': ('https://pytest.org/en/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}


# Sphinx gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../tutorials'],
    'filename_pattern': '^((?!sgskip).)*$',
    'gallery_dirs': ['gallery', 'tutorials'],
    'doc_module': ('matplotlib', 'mpl_toolkits'),
    'reference_url': {
        'matplotlib': None,
        'numpy': 'https://docs.scipy.org/doc/numpy',
        'scipy': 'https://docs.scipy.org/doc/scipy/reference',
    },
    'backreferences_dir': 'api/_as_gen',
    'subsection_order': gallery_order.sectionorder,
    'within_subsection_order': gallery_order.subsectionorder,
    'remove_config_comments': True,
    'min_reported_time': 1,
}

plot_gallery = 'True'

# Monkey-patching gallery signature to include search keywords
gen_rst.SPHX_GLR_SIG = """\n
.. only:: html

 .. rst-class:: sphx-glr-signature

    Keywords: matplotlib code example, codex, python plot, pyplot
    `Gallery generated by Sphinx-Gallery
    <https://sphinx-gallery.readthedocs.io>`_\n"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# This is the default encoding, but it doesn't hurt to be explicit
source_encoding = "utf-8"

# The master toctree document.
master_doc = 'contents'

# General substitutions.
try:
    SHA = subprocess.check_output(
        ['git', 'describe', '--dirty']).decode('utf-8').strip()
# Catch the case where git is not installed locally, and use the versioneer
# version number instead
except (subprocess.CalledProcessError, FileNotFoundError):
    SHA = matplotlib.__version__

html_context = {'sha': SHA}

project = 'Matplotlib'
copyright = ('2002 - 2012 John Hunter, Darren Dale, Eric Firing, '
             'Michael Droettboom and the Matplotlib development '
             f'team; 2012 - {datetime.now().year} The Matplotlib development team')


# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.

version = matplotlib.__version__
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

default_role = 'obj'

# Plot directive configuration
# ----------------------------

plot_formats = [('png', 100), ('pdf', 100)]

# GitHub extension

github_project_url = "https://github.com/matplotlib/matplotlib/"

# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
#html_style = 'matplotlib.css'
html_style = f'mpl.css?{SHA}'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
#html_logo = 'logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If nonempty, this is the file name suffix for generated HTML files.  The
# default is ``".html"``.
html_file_suffix = '.html'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# Content template for the index page.
html_index = 'index.html'

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Custom sidebar templates, maps page names to templates.
html_sidebars = {
    'index': [
        # 'sidebar_announcement.html',
        'sidebar_versions.html',
        'donate_sidebar.html'],
    '**': ['localtoc.html', 'relations.html', 'pagesource.html']
}

# If false, no module index is generated.
#html_use_modindex = True
html_domain_indices = ["py-modindex"]

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.
html_use_opensearch = 'False'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Matplotlibdoc'

# Use typographic quote characters.
smartquotes = False

# Path to favicon
html_favicon = '_static/favicon.ico'

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = 'letter'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).

latex_documents = [
    ('contents', 'Matplotlib.tex', 'Matplotlib',
     'John Hunter\\and Darren Dale\\and Eric Firing\\and Michael Droettboom'
     '\\and and the matplotlib development team', 'manual'),
]


# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

latex_elements = {}
# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r"""
   % One line per author on title page
   \DeclareRobustCommand{\and}%
     {\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}}%
   % In the parameters section, place a newline after the Parameters
   % header.  (This is stolen directly from Numpy's conf.py, since it
   % affects Numpy-style docstrings).
   \usepackage{expdlist}
   \let\latexdescription=\description
   \def\description{\latexdescription{}{} \breaklabel}

   \usepackage{amsmath}
   \usepackage{amsfonts}
   \usepackage{amssymb}
   \usepackage{txfonts}

   % The enumitem package provides unlimited nesting of lists and
   % enums.  Sphinx may use this in the future, in which case this can
   % be removed.  See
   % https://bitbucket.org/birkenfeld/sphinx/issue/777/latex-output-too-deeply-nested
   \usepackage{enumitem}
   \setlistdepth{2048}
"""
latex_elements['pointsize'] = '11pt'

# Documents to append as an appendix to all manuals.
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

if hasattr(sphinx, 'version_info') and sphinx.version_info[:2] >= (1, 4):
    latex_toplevel_sectioning = 'part'
else:
    latex_use_parts = True

# Show both class-level docstring and __init__ docstring in class
# documentation
autoclass_content = 'both'

texinfo_documents = [
    ("contents", 'matplotlib', 'Matplotlib Documentation',
     'John Hunter@*Darren Dale@*Eric Firing@*Michael Droettboom@*'
     'The matplotlib development team',
     'Matplotlib', "Python plotting package", 'Programming',
     1),
]

# numpydoc config

numpydoc_show_class_members = False

latex_engine = 'xelatex'  # or 'lualatex'

latex_elements = {
    'babel': r'\usepackage{babel}',
    'fontpkg': r'\setmainfont{DejaVu Serif}',
}

html4_writer = True

inheritance_node_attrs = dict(fontsize=16)

graphviz_dot = shutil.which('dot')
graphviz_output_format = 'svg'


def setup(app):
    if any(st in version for st in ('post', 'alpha', 'beta')):
        bld_type = 'dev'
    else:
        bld_type = 'rel'
    app.add_config_value('releaselevel', bld_type, 'env')

# -*- coding: utf-8 -*-
#
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

import matplotlib
import os
import sys
import sphinx
import six
from glob import glob

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('.'))

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.only_directives',
    'matplotlib.sphinxext.plot_directive',
    'sphinxext.custom_roles',
    'sphinxext.github',
    'sphinxext.math_symbol_table',
    'sphinxext.mock_gui_toolkits',
    'sphinxext.skip_deprecated',
]

exclude_patterns = ['api/api_changes/*', 'users/whats_new/*']


def _check_deps():
    names = {"colorspacious": 'colorspacious',
             "IPython.sphinxext.ipython_console_highlighting": 'ipython',
             "matplotlib": 'matplotlib',
             "numpydoc": 'numpydoc',
             "PIL.Image": 'pillow',
             "sphinx_gallery": 'sphinx_gallery'}
    if sys.version_info < (3, 3):
        names["mock"] = 'mock'
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

_check_deps()

# Import only after checking for dependencies.
from sphinx_gallery.sorting import ExplicitOrder

if six.PY2:
    from distutils.spawn import find_executable
    has_dot = find_executable('dot') is not None
else:
    from shutil import which  # Python >= 3.3
    has_dot = which('dot') is not None
if not has_dot:
    raise OSError(
        "No binary named dot - you need to install the Graph Visualization "
        "software (usually packaged as 'graphviz') to build the documentation")

autosummary_generate = True

autodoc_docstring_signature = True
autodoc_default_flags = ['members', 'undoc-members']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None)
}

explicit_order_folders = [
                          '../examples/api',
                          '../examples/pyplots',
                          '../examples/subplots_axes_and_figures',
                          '../examples/color',
                          '../examples/statistics',
                          '../examples/lines_bars_and_markers',
                          '../examples/images_contours_and_fields',
                          '../examples/shapes_and_collections',
                          '../examples/text_labels_and_annotations',
                          '../examples/pie_and_polar_charts',
                          '../examples/style_sheets',
                          '../examples/axes_grid',
                          '../examples/showcase',
                          '../tutorials/introductory',
                          '../tutorials/intermediate',
                          '../tutorials/advanced']
for folder in sorted(glob('../examples/*') + glob('../tutorials/*')):
    if not os.path.isdir(folder) or folder in explicit_order_folders:
        continue
    explicit_order_folders.append(folder)

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
    'subsection_order': ExplicitOrder(explicit_order_folders),
    'min_reported_time': 1,
}

plot_gallery = 'True'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# This is the default encoding, but it doesn't hurt to be explicit
source_encoding = "utf-8"

# The master toctree document.
master_doc = 'contents'

# General substitutions.
project = 'Matplotlib'
copyright = ('2002 - 2012 John Hunter, Darren Dale, Eric Firing, '
             'Michael Droettboom and the Matplotlib development '
             'team; 2012 - 2017 The Matplotlib development team')

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

# Github extension

github_project_url = "https://github.com/matplotlib/matplotlib/"

# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
#html_style = 'matplotlib.css'
html_style = 'mpl.css'

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

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Content template for the index page.
html_index = 'index.html'

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Custom sidebar templates, maps page names to templates.
html_sidebars = {
    'index': ['donate_sidebar.html', 'searchbox.html'],
    '**': ['localtoc.html', 'relations.html',
           'sourcelink.html', 'searchbox.html']
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
     'John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the '
     'matplotlib development team', 'manual'),
]


# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

latex_elements = {}
# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r"""
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

rst_epilog = """
.. |minimum_numpy_version| replace:: %s
""" % matplotlib.__version__numpy__

texinfo_documents = [
    ("contents", 'matplotlib', 'Matplotlib Documentation',
     'John Hunter@*Darren Dale@*Eric Firing@*Michael Droettboom@*'
     'The matplotlib development team',
     'Matplotlib', "Python plotting package", 'Programming',
     1),
]

# numpydoc config

numpydoc_show_class_members = False

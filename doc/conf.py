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

import os
import sys
import sphinx

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('.'))

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['matplotlib.sphinxext.mathmpl', 'sphinxext.math_symbol_table',
              'sphinx.ext.autodoc', 'matplotlib.sphinxext.only_directives',
              'sphinx.ext.doctest', 'sphinx.ext.autosummary',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.inheritance_diagram',
              'sphinxext.gen_gallery', 'sphinxext.gen_rst',
              'sphinxext.github',
              'numpydoc']

exclude_patterns = ['api/api_changes/*', 'users/whats_new/*']

# Use IPython's console highlighting by default
try:
    from IPython.sphinxext import ipython_console_highlighting
except ImportError:
    raise ImportError(
        "IPython must be installed to build the matplotlib docs")
else:
    extensions.append('IPython.sphinxext.ipython_console_highlighting')
    extensions.append('IPython.sphinxext.ipython_directive')

try:
    import numpydoc
except ImportError:
    raise ImportError("No module named numpydoc - you need to install "
                      "numpydoc to build the documentation.")


autosummary_generate = True

autodoc_docstring_signature = True

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
copyright = '2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the matplotlib development team; 2012 - 2014 The matplotlib development team'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
try:
    import matplotlib
except ImportError:
    msg = "Error: matplotlib must be installed before building the documentation"
    sys.exit(msg)

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

plot_formats = [('svg', 72), ('png', 80)]

# Subdirectories in 'examples/' directory of package and titles for gallery
mpl_example_sections = (
    ('lines_bars_and_markers', 'Lines, bars, and markers'),
    ('shapes_and_collections', 'Shapes and collections'),
    ('statistics', 'Statistical plots'),
    ('images_contours_and_fields', 'Images, contours, and fields'),
    ('pie_and_polar_charts', 'Pie and polar charts'),
    ('color', 'Color'),
    ('text_labels_and_annotations', 'Text, labels, and annotations'),
    ('ticks_and_spines', 'Ticks and spines'),
    ('scales', 'Axis scales'),
    ('subplots_axes_and_figures', 'Subplots, axes, and figures'),
    ('style_sheets', 'Style sheets'),
    ('specialty_plots', 'Specialty plots'),
    ('showcase', 'Showcase'),
    ('api', 'API'),
    ('pylab_examples', 'pylab examples'),
    ('mplot3d', 'mplot3d toolkit'),
    ('axes_grid1', 'axes_grid1 toolkit'),
    ('axisartist', 'axisartist toolkit'),
    ('units', 'units'),
    ('widgets', 'widgets'),
    )


# Github extension

github_project_url = "http://github.com/matplotlib/matplotlib/"

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
    'index': ['badgesidebar.html','donate_sidebar.html',
              'indexsidebar.html', 'searchbox.html'],
    '**': ['badgesidebar.html', 'localtoc.html',
           'relations.html', 'sourcelink.html', 'searchbox.html']
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {'index': 'index.html',
                         'gallery':'gallery.html',
                         'citing': 'citing.html'}

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


# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
latex_font_size = '11pt'

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

# Additional stuff for the LaTeX preamble.
latex_preamble = r"""
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

# Documents to append as an appendix to all manuals.
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

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

try:
    from unittest.mock import MagicMock
except:
    from mock import MagicMock


class MyWX(MagicMock):
    class Panel(object):
        pass

    class ToolBar(object):
        pass

    class Frame(object):
        pass

    VERSION_STRING = '2.8.12'


class MyPyQt4(MagicMock):
    class QtGui(object):
        class QToolBar(object):
            pass

        class QDialog(object):
            pass

        class QWidget(object):
            pass

        class QMainWindow(object):
            pass


class MySip(MagicMock):
    def getapi(*args):
        return 1


mockwxversion = MagicMock()
mockwx = MyWX()
mocksip = MySip()
mockpyqt4 = MyPyQt4()
sys.modules['wxversion'] = mockwxversion
sys.modules['wx'] = mockwx
sys.modules['sip'] = mocksip
sys.modules['PyQt4'] = mockpyqt4

################# numpydoc config ####################
numpydoc_show_class_members = False

# Matplotlib documentation build configuration file, created by
# sphinx-quickstart on Fri May  2 12:33:25 2008.
#
# This file is execfile()d with the current directory set to its containing
# dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed
# automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import matplotlib
import sphinx

from datetime import datetime
import time

# Release mode enables optimizations and other related options.
is_release_build = tags.has('release')  # noqa

# are we running circle CI?
CIRCLECI = 'CIRCLECI' in os.environ

# Parse year using SOURCE_DATE_EPOCH, falling back to current time.
# https://reproducible-builds.org/specs/source-date-epoch/
sourceyear = datetime.utcfromtimestamp(
    int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))).year

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('.'))
sys.path.append('.')

# General configuration
# ---------------------

# Unless we catch the warning explicitly somewhere, a warning should cause the
# docs build to fail. This is especially useful for getting rid of deprecated
# usage in the gallery.
warnings.filterwarnings('error', append=True)

# Strip backslashes in function's signature
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
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.inkscapeconverter',
    'sphinxext.custom_roles',
    'sphinxext.github',
    'sphinxext.math_symbol_table',
    'sphinxext.missing_references',
    'sphinxext.mock_gui_toolkits',
    'sphinxext.skip_deprecated',
    'sphinxext.redirect_from',
    'sphinx_copybutton',
    'sphinx_panels',
]

exclude_patterns = [
    'api/prev_api_changes/api_changes_*/*',
]

panels_add_bootstrap_css = False


def _check_dependencies():
    names = {
        "colorspacious": 'colorspacious',
        "IPython.sphinxext.ipython_console_highlighting": 'ipython',
        "matplotlib": 'matplotlib',
        "numpydoc": 'numpydoc',
        "PIL.Image": 'pillow',
        "pydata_sphinx_theme": 'pydata_sphinx_theme',
        "sphinx_copybutton": 'sphinx_copybutton',
        "sphinx_gallery": 'sphinx_gallery',
        "sphinxcontrib.inkscapeconverter": 'sphinxcontrib-svg2pdfconverter',
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

# we should ignore warnings coming from importing deprecated modules for
# autodoc purposes, as this will disappear automatically when they are removed
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='importlib',  # used by sphinx.autodoc.importer
                        message=r'(\n|.)*module was deprecated.*')

autodoc_docstring_signature = True
autodoc_default_options = {'members': None, 'undoc-members': None}

# make sure to ignore warnings that stem from simply inspecting deprecated
# class-level attributes
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='sphinx.util.inspect')

# missing-references names matches sphinx>=3 behavior, so we can't be nitpicky
# for older sphinxes.
nitpicky = sphinx.version_info >= (3,)
# change this to True to update the allowed failures
missing_references_write_json = False
missing_references_warn_unused_ignores = False

intersphinx_mapping = {
    'Pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'cycler': ('https://matplotlib.org/cycler/', None),
    'dateutil': ('https://dateutil.readthedocs.io/en/stable/', None),
    'ipykernel': ('https://ipykernel.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pytest': ('https://pytest.org/en/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'tornado': ('https://www.tornadoweb.org/en/stable/', None),
}


# Sphinx gallery configuration

sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../tutorials', '../plot_types'],
    'filename_pattern': '^((?!sgskip).)*$',
    'gallery_dirs': ['gallery', 'tutorials', 'plot_types'],
    'doc_module': ('matplotlib', 'mpl_toolkits'),
    'reference_url': {
        'matplotlib': None,
        'numpy': 'https://numpy.org/doc/stable/',
        'scipy': 'https://docs.scipy.org/doc/scipy/reference/',
    },
    'backreferences_dir': Path('api') / Path('_as_gen'),
    'subsection_order': gallery_order.sectionorder,
    'within_subsection_order': gallery_order.subsectionorder,
    'remove_config_comments': True,
    'min_reported_time': 1,
    'thumbnail_size': (320, 224),
    # Compression is a significant effort that we skip for local and CI builds.
    'compress_images': ('thumbnails', 'images') if is_release_build else (),
    'matplotlib_animations': True,
    'image_srcset': ["2x"],
    'junit': '../test-results/sphinx-gallery/junit.xml' if CIRCLECI else '',
}

mathmpl_fontsize = 11.0
mathmpl_srcset = ['2x']

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
master_doc = 'users/index'

# General substitutions.
try:
    SHA = subprocess.check_output(
        ['git', 'describe', '--dirty']).decode('utf-8').strip()
# Catch the case where git is not installed locally, and use the setuptools_scm
# version number instead
except (subprocess.CalledProcessError, FileNotFoundError):
    SHA = matplotlib.__version__

html_context = {
    "sha": SHA,
}

project = 'Matplotlib'
copyright = (
    '2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom '
    'and the Matplotlib development team; '
    f'2012 - {sourceyear} The Matplotlib development team'
)


# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.

version = matplotlib.__version__
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

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
# html_style = 'matplotlib.css'
# html_style = f"mpl.css?{SHA}"
html_css_files = [
    f"mpl.css?{SHA}",
]

html_theme = "mpl_sphinx_theme"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = "_static/logo2.svg"
html_theme_options = {
    "native_site": True,
    "logo_link": "index",
    # collapse_navigation in pydata-sphinx-theme is slow, so skipped for local
    # and CI builds https://github.com/pydata/pydata-sphinx-theme/pull/386
    "collapse_navigation": not is_release_build,
    "show_prev_next": False,
}
include_analytics = is_release_build
if include_analytics:
    html_theme_options["google_analytics_id"] = "UA-55954603-1"

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
# html_sidebars = {}

# Custom sidebar templates, maps page names to templates.
html_sidebars = {
    "index": [
        'search-field.html',
        # 'sidebar_announcement.html',
        "sidebar_versions.html",
        "cheatsheet_sidebar.html",
        "donate_sidebar.html",
    ],
    # '**': ['localtoc.html', 'pagesource.html']
}

# If true, add an index to the HTML documents.
html_use_index = False

# If true, generate domain-specific indices in addition to the general index.
# For e.g. the Python domain, this is the global module index.
html_domain_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
# html_copy_source = True

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

# Grouping the document tree into LaTeX files.
# List of tuples:
#   (source start file, target name, title, author,
#    document class [howto/manual])

latex_documents = [
    ('contents', 'Matplotlib.tex', 'Matplotlib',
     'John Hunter\\and Darren Dale\\and Eric Firing\\and Michael Droettboom'
     '\\and and the matplotlib development team', 'manual'),
]


# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

# Use Unicode aware LaTeX engine
latex_engine = 'xelatex'  # or 'lualatex'

latex_elements = {}

# Keep babel usage also with xelatex (Sphinx default is polyglossia)
# If this key is removed or changed, latex build directory must be cleaned
latex_elements['babel'] = r'\usepackage{babel}'

# Font configuration
# Fix fontspec converting " into right curly quotes in PDF
# cf https://github.com/sphinx-doc/sphinx/pull/6888/
latex_elements['fontenc'] = r'''
\usepackage{fontspec}
\defaultfontfeatures[\rmfamily,\sffamily,\ttfamily]{}
'''

# Sphinx 2.0 adopts GNU FreeFont by default, but it does not have all
# the Unicode codepoints needed for the section about Mathtext
# "Writing mathematical expressions"
fontpkg = r"""
\IfFontExistsTF{XITS}{
 \setmainfont{XITS}
}{
 \setmainfont{XITS}[
  Extension      = .otf,
  UprightFont    = *-Regular,
  ItalicFont     = *-Italic,
  BoldFont       = *-Bold,
  BoldItalicFont = *-BoldItalic,
]}
\IfFontExistsTF{FreeSans}{
 \setsansfont{FreeSans}
}{
 \setsansfont{FreeSans}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]}
\IfFontExistsTF{FreeMono}{
 \setmonofont{FreeMono}
}{
 \setmonofont{FreeMono}[
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]}
% needed for \mathbb (blackboard alphabet) to actually work
\usepackage{unicode-math}
\IfFontExistsTF{XITS Math}{
 \setmathfont{XITS Math}
}{
 \setmathfont{XITSMath-Regular}[
  Extension      = .otf,
]}
"""
latex_elements['fontpkg'] = fontpkg

# Sphinx <1.8.0 or >=2.0.0 does this by default, but the 1.8.x series
# did not for latex_engine = 'xelatex' (as it used Latin Modern font).
# We need this for code-blocks as FreeMono has wide glyphs.
latex_elements['fvset'] = r'\fvset{fontsize=\small}'
# Fix fancyhdr complaining about \headheight being too small
latex_elements['passoptionstopackages'] = r"""
    \PassOptionsToPackage{headheight=14pt}{geometry}
"""

# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r"""
   % One line per author on title page
   \DeclareRobustCommand{\and}%
     {\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}}%
   \usepackage{etoolbox}
   \AtBeginEnvironment{sphinxthebibliography}{\appendix\part{Appendices}}
   \usepackage{expdlist}
   \let\latexdescription=\description
   \def\description{\latexdescription{}{} \breaklabel}
   % But expdlist old LaTeX package requires fixes:
   % 1) remove extra space
   \makeatletter
   \patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
   \makeatother
   % 2) fix bug in expdlist's way of breaking the line after long item label
   \makeatletter
   \def\breaklabel{%
       \def\@breaklabel{%
           \leavevmode\par
           % now a hack because Sphinx inserts \leavevmode after term node
           \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
      }%
   }
   \makeatother
"""
# Sphinx 1.5 provides this to avoid "too deeply nested" LaTeX error
# and usage of "enumitem" LaTeX package is unneeded.
# Value can be increased but do not set it to something such as 2048
# which needlessly would trigger creation of thousands of TeX macros
latex_elements['maxlistdepth'] = '10'
latex_elements['pointsize'] = '11pt'

# Better looking general index in PDF
latex_elements['printindex'] = r'\footnotesize\raggedright\printindex'

# Documents to append as an appendix to all manuals.
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

latex_toplevel_sectioning = 'part'

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

html4_writer = True

inheritance_node_attrs = dict(fontsize=16)

graphviz_dot = shutil.which('dot')
# Still use PNG until SVG linking is fixed
# https://github.com/sphinx-doc/sphinx/issues/3176
# graphviz_output_format = 'svg'


def reduce_plot_formats(app):
    # Fox CI and local builds, we don't need all the default plot formats, so
    # only generate the directly useful one for the current builder.
    if app.builder.name == 'html':
        keep = 'png'
    elif app.builder.name == 'latex':
        keep = 'pdf'
    else:
        return
    app.config.plot_formats = [entry
                               for entry in app.config.plot_formats
                               if entry[0] == keep]


def setup(app):
    if any(st in version for st in ('post', 'alpha', 'beta')):
        bld_type = 'dev'
    else:
        bld_type = 'rel'
    app.add_config_value('releaselevel', bld_type, 'env')

    if not is_release_build:
        app.connect('builder-inited', reduce_plot_formats)

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------
link_github = True
# You can add build old with link_github = False

if link_github:
    import inspect
    from packaging.version import parse

    extensions.append('sphinx.ext.linkcode')

    def linkcode_resolve(domain, info):
        """
        Determine the URL corresponding to Python object
        """
        if domain != 'py':
            return None

        modname = info['module']
        fullname = info['fullname']

        submod = sys.modules.get(modname)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split('.'):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        try:
            fn = inspect.getsourcefile(obj)
        except TypeError:
            fn = None
        if not fn or fn.endswith('__init__.py'):
            try:
                fn = inspect.getsourcefile(sys.modules[obj.__module__])
            except (TypeError, AttributeError, KeyError):
                fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        linespec = (f"#L{lineno:d}-L{lineno + len(source) - 1:d}"
                    if lineno else "")

        startdir = Path(matplotlib.__file__).parent.parent
        fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, '/')

        if not fn.startswith(('matplotlib/', 'mpl_toolkits/')):
            return None

        version = parse(matplotlib.__version__)
        tag = 'master' if version.is_devrelease else f'v{version.public}'
        return ("https://github.com/matplotlib/matplotlib/blob"
                f"/{tag}/lib/{fn}{linespec}")
else:
    extensions.append('sphinx.ext.viewcode')

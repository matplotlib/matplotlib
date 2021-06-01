import sys
from os.path import join as pjoin, abspath
import sphinx
from distutils.version import LooseVersion

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, abspath(pjoin('..', '..')))

# -- General configuration ------------------------------------------------

extensions = ['matplotlib.sphinxext.plot_directive']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'tinypages'
copyright = '2014, Matplotlib developers'
version = '0.1'
release = '0.1'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

# -- Options for HTML output ----------------------------------------------

if LooseVersion(sphinx.__version__) >= LooseVersion('1.3'):
    html_theme = 'classic'
else:
    html_theme = 'default'

html_static_path = ['_static']

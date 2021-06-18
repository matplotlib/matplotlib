import sphinx
from distutils.version import LooseVersion

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

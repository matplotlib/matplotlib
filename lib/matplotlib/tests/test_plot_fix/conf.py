import sys
import os
from pathlib import Path

# Absolute path to your modified matplotlib
matplotlib_path = Path.home() / "matplotlib" / "lib"
sys.path.insert(0, str(matplotlib_path))

# Sphinx extensions
extensions = [
    "matplotlib.sphinxext.plot_directive",
]

# HTML output
html_theme = 'alabaster'
html_static_path = ['_static']

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# TODOs
todo_include_todos = True


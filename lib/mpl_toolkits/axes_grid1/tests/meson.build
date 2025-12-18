python_sources = [
  '__init__.py',
  'conftest.py',
  'test_axes_grid1.py',
]

py3.install_sources(python_sources, subdir: 'mpl_toolkits/axes_grid1/tests')

install_subdir(
  'baseline_images',
  install_tag: 'tests',
  install_dir: py3.get_install_dir(subdir: 'mpl_toolkits/axes_grid1/tests'))

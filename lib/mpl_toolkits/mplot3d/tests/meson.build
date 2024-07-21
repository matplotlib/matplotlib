python_sources = [
  '__init__.py',
  'conftest.py',
  'test_art3d.py',
  'test_axes3d.py',
  'test_legend3d.py',
]

py3.install_sources(python_sources, subdir: 'mpl_toolkits/mplot3d/tests')

install_subdir(
  'baseline_images',
  install_tag: 'tests',
  install_dir: py3.get_install_dir(subdir: 'mpl_toolkits/mplot3d/tests'))

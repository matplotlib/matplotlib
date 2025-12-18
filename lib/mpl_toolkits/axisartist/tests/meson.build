python_sources = [
  '__init__.py',
  'conftest.py',
  'test_angle_helper.py',
  'test_axis_artist.py',
  'test_axislines.py',
  'test_floating_axes.py',
  'test_grid_finder.py',
  'test_grid_helper_curvelinear.py',
]

py3.install_sources(python_sources, subdir: 'mpl_toolkits/axisartist/tests')

install_subdir(
  'baseline_images',
  install_tag: 'tests',
  install_dir: py3.get_install_dir(subdir: 'mpl_toolkits/axisartist/tests'))

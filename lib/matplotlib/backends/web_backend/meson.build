install_dir = py3.get_install_dir(subdir: 'matplotlib/backends/web_backend')

install_data(
  'all_figures.html',
  'ipython_inline_figure.html',
  'single_figure.html',
  install_tag: 'data',
  install_dir: install_dir)
install_subdir('css', install_tag: 'data', install_dir: install_dir)
install_subdir('js', install_tag: 'data', install_dir: install_dir)
install_data(
  'nbagg_uat.ipynb',
  install_tag: 'tests',
  install_dir: install_dir)

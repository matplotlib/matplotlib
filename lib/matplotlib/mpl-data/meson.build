custom_target('matplotlibrc',
  command: [
    find_program(meson.project_source_root() / 'tools/generate_matplotlibrc.py'),
    '@INPUT@',
    '@OUTPUT@',
    get_option('rcParams-backend')
  ],
  input: 'matplotlibrc',
  output: 'matplotlibrc',
  install: true,
  install_tag: 'data',
  install_dir: py3.get_install_dir(subdir: 'matplotlib/mpl-data'))

install_data(
  'kpsewhich.lua',
  install_tag: 'data',
  install_dir: py3.get_install_dir(subdir: 'matplotlib/mpl-data'))

foreach dir : ['fonts', 'images', 'plot_directive', 'sample_data', 'stylelib']
  install_subdir(
    dir,
    install_tag: 'data',
    install_dir: py3.get_install_dir(subdir: 'matplotlib/mpl-data'))
endforeach

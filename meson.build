project(
  'matplotlib',
  'c', 'cpp',
  version: '3.9.0.dev0',
  # qt_editor backend is MIT
  # ResizeObserver at end of lib/matplotlib/backends/web_backend/js/mpl.js is CC0
  # Carlogo, STIX and Computer Modern is OFL
  # DejaVu is Bitstream Vera and Public Domain
  license: 'PSF-2.0 AND MIT AND CC0-1.0 AND OFL-1.1 AND Bitstream-Vera AND Public-Domain',
  meson_version: '>=1.1.0',
  default_options: [
    'b_lto=true',
    'cpp_std=c++11',
    'auto_features=disabled',  # Force FreeType to avoid extra dependencies.
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()

pybind11_dep = dependency('pybind11', version: '>=2.6')

subdir('extern')
subdir('src')
subdir('lib')

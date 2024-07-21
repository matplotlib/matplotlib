ttconv_lib = static_library('ttconv',
  'pprdrv_tt2.cpp',
  'pprdrv_tt.cpp',
  'ttutil.cpp',
  'pprdrv.h',
  'truetype.h',
  dependencies: [py3_dep],
  gnu_symbol_visibility: 'inlineshidden',
)

ttconv_dep = declare_dependency(
  include_directories: include_directories('.'),
  link_with: ttconv_lib,
)

# Bundled code.
subdir('agg24-svn')

# External code.

# FreeType 2.3 has libtool version 9.11.3 as can be checked from the tarball.
# For FreeType>=2.4, there is a conversion table in docs/VERSIONS.txt in the
# FreeType source tree.
if get_option('system-freetype')
  freetype_dep = dependency('freetype2', version: '>=9.11.3')
else
  freetype_proj = subproject(
    'freetype2',
    default_options: [
      'default_library=static',
      'brotli=disabled',
      'bzip2=disabled',
      'harfbuzz=disabled',
      'mmap=auto',
      'png=disabled',
      'tests=disabled',
      'zlib=internal',
  ])
  freetype_dep = freetype_proj.get_variable('freetype_dep')
endif

if get_option('system-libraqm')
  libraqm_dep = dependency('raqm', version: '>=0.10.3')
else
  subproject('harfbuzz',
    default_options: [
      'default_library=static',
      'cairo=disabled',
      'coretext=disabled',
      'directwrite=disabled',
      'fontations=disabled',
      'freetype=enabled',
      'gdi=disabled',
      'glib=disabled',
      'gobject=disabled',
      'harfruzz=disabled',
      'icu=disabled',
      'tests=disabled',
    ]
  )
  subproject('sheenbidi', default_options: ['default_library=static'])
  libraqm_proj = subproject('libraqm-0.10.3',
    default_options: [
      'default_library=static',
      'sheenbidi=true',
    ]
  )
  libraqm_dep = libraqm_proj.get_variable('libraqm_dep')
endif

if get_option('system-qhull')
  qhull_dep = dependency('qhull_r', version: '>=8.0.2', required: false)
  if not qhull_dep.found()
    cc.check_header('libqhull_r/qhull_ra.h', required: true)
    qhull_dep = cc.find_library('qhull_r')
  endif
else
  qhull_proj = subproject('qhull')
  qhull_dep = qhull_proj.get_variable('qhull_dep')
endif

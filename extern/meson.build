# Bundled code.
subdir('agg24-svn')
subdir('ttconv')

# External code.

# FreeType 2.3 has libtool version 9.11.3 as can be checked from the tarball.
# For FreeType>=2.4, there is a conversion table in docs/VERSIONS.txt in the
# FreeType source tree.
if get_option('system-freetype')
  freetype_dep = dependency('freetype2', version: '>=9.11.3')
else
  # This is the version of FreeType to use when building a local version. It
  # must match the value in `lib/matplotlib.__init__.py`. Also update the docs
  # in `docs/devel/dependencies.rst`. Bump the cache key in
  # `.circleci/config.yml` when changing requirements.
  TESTING_VERSION_OF_FREETYPE = '2.6.1'
  if host_machine.system() == 'windows' and host_machine.cpu_family() == 'aarch64'
    # Older versions of freetype are not supported for win/arm64
    # Matplotlib tests will not pass
    LOCAL_FREETYPE_VERSION = '2.11.1'
  else
    LOCAL_FREETYPE_VERSION = TESTING_VERSION_OF_FREETYPE
  endif

  freetype_proj = subproject(
    f'freetype-@LOCAL_FREETYPE_VERSION@',
    default_options: ['default_library=static'])
  freetype_dep = freetype_proj.get_variable('freetype_dep')
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

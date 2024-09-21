# For cross-compilation it is often not possible to run the Python interpreter in order
# to retrieve the platform-specific /dev/null. It can be specified in the cross file
# instead:
#
#   [properties]
#   devnull = /dev/null
#
# This uses the value as is, and avoids running the interpreter.
devnull = meson.get_external_property('devnull', 'not-given')
if devnull == 'not-given'
  devnull = run_command(py3, '-c', 'import os; print(os.devnull)',
                        capture: true, check: true).stdout().strip()
endif

# Will only exist on Linux with older glibc.
dl = dependency('dl', required: false)

# With Meson >= 1.2.0, use cpp_winlibs instead of manually searching.
if ['cygwin', 'windows'].contains(host_machine.system())
  comctl32 = cc.find_library('comctl32')
  ole32 = cc.find_library('ole32')
  psapi = cc.find_library('psapi')
  shell32 = cc.find_library('shell32')
  user32 = cc.find_library('user32')
else
  comctl32 = []
  ole32 = []
  psapi = []
  shell32 = []
  user32 = []
endif

extension_data = {
  '_backend_agg': {
    'subdir': 'matplotlib/backends',
    'sources': files(
      '_backend_agg.cpp',
      '_backend_agg_wrapper.cpp',
    ),
    'dependencies': [agg_dep, freetype_dep, pybind11_dep],
  },
  '_c_internal_utils': {
    'subdir': 'matplotlib',
    'sources': files(
      '_c_internal_utils.cpp',
    ),
    'dependencies': [pybind11_dep, dl, ole32, shell32, user32],
  },
  'ft2font': {
    'subdir': 'matplotlib',
    'sources': files(
      'ft2font.cpp',
      'ft2font_wrapper.cpp',
    ),
    'dependencies': [
      freetype_dep, pybind11_dep, agg_dep.partial_dependency(includes: true),
    ],
    'cpp_args': [
      '-DFREETYPE_BUILD_TYPE="@0@"'.format(
        freetype_dep.type_name() == 'internal' ? 'local' : 'system',
      ),
    ],
  },
  '_image': {
    'subdir': 'matplotlib',
    'sources': files(
      '_image_wrapper.cpp',
      'py_converters.cpp',
    ),
    'dependencies': [
      pybind11_dep,
      # Only need source code files agg_image_filters.cpp and agg_trans_affine.cpp
      agg_dep,
    ],
  },
  '_path': {
    'subdir': 'matplotlib',
    'sources': files(
      '_path_wrapper.cpp',
    ),
    'dependencies': [agg_dep, pybind11_dep],
  },
  '_qhull': {
    'subdir': 'matplotlib',
    'sources': files(
      '_qhull_wrapper.cpp',
    ),
    'dependencies': [pybind11_dep, qhull_dep],
    'c_args': [f'-DMPL_DEVNULL=@devnull@'],
    'cpp_args': [f'-DMPL_DEVNULL=@devnull@'],
  },
  '_tkagg': {
    'subdir': 'matplotlib/backends',
    'sources': files(
      '_tkagg.cpp',
    ),
    'include_directories': include_directories('.'),
    # The dl/psapi libraries are needed for finding Tcl/Tk at run time.
    'dependencies': [
      pybind11_dep, agg_dep.partial_dependency(includes: true), dl, comctl32, psapi,
    ],
  },
  '_tri': {
    'subdir': 'matplotlib',
    'sources': files(
      'tri/_tri.cpp',
      'tri/_tri_wrapper.cpp',
    ),
    'dependencies': [pybind11_dep],
  },
  '_ttconv': {
    'subdir': 'matplotlib',
    'sources': files(
      '_ttconv.cpp',
    ),
    'dependencies': [ttconv_dep, pybind11_dep],
  },
}

foreach ext, kwargs : extension_data
  py3.extension_module(ext, install: true, kwargs: kwargs)
endforeach

if get_option('macosx') and host_machine.system() == 'darwin'
  add_languages('objc', native: false)
  py3.extension_module(
    '_macosx',
    subdir: 'matplotlib/backends',
    sources: files(
      '_macosx.m',
    ),
    dependencies: dependency('appleframeworks', modules: 'Cocoa'),
    override_options: ['werror=true'],
    install: true,
  )
endif

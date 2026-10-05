project(
  'nanobind',
  'cpp',
  version: run_command(
    'src/version.py',
    check: true,
  ).stdout().strip(),
  meson_version: '>=1.0.0',
  default_options: [
    'cpp_std=c++17',
    'b_ndebug=if-release',
    'default_library=static',
  ],
  license: 'BSD-3-Clause',
)

py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

robin_map_dep = dependency(
  'tsl-robin-map',
  version: '>=1.4.0',
)

incdir = include_directories('include')

# Arguments are derived from nanobind_build_library() in CMake configuration.
dep_compile_args = []
dep_link_args = []

compiler = meson.get_compiler('cpp')
if compiler.get_argument_syntax() == 'msvc'
  dep_compile_args += ['/D_CRT_SECURE_NO_WARNINGS']
endif

# Discussion in WrapDB PR #1556 as to whether or not the following is necessary:
# dep_compile_args += ['-fno-strict-aliasing']

freethreaded = py.get_variable('Py_GIL_DISABLED', 0) != 0
if freethreaded
  dep_compile_args += ['-DNB_FREE_THREADED']
endif

if not get_option('debug')
  dep_compile_args += ['-DNB_COMPACT_ASSERTIONS']

  # The following, from nanobind_strip(), applies both to the nanobind shared
  # library (if it's built) and to the developer's extension:
  if host_machine.system() == 'darwin'
    dep_link_args += ['-Wl,-dead_strip', '-Wl,-x', '-Wl,-S']
  elif host_machine.system() != 'windows'
    dep_link_args += ['-Wl,-s']
  endif
endif

if host_machine.system() == 'darwin'
  py_soabi = py.get_variable('SOABI')
  if py_soabi.startswith('cp')
    resp_file = meson.project_source_root() / 'cmake/darwin-ld-cpython.sym'
  else
    resp_file = meson.project_source_root() / 'cmake/darwin-ld-pypy.sym'
  endif
  dep_link_args += ['-Wl,@' + resp_file]
endif

if get_option('default_library') == 'shared'
  dep_compile_args += ['-DNB_SHARED']
  add_project_arguments(
    [dep_compile_args, '-DNB_BUILD'],
    language: 'cpp',
  )
  add_project_link_arguments(
    dep_link_args,
    language: 'cpp',
  )
elif host_machine.system() != 'windows' and host_machine.system() != 'darwin'
  dep_compile_args += ['-ffunction-sections', '-fdata-sections']
  add_project_arguments(
    dep_compile_args,
    language: 'cpp',
  )
  dep_link_args += ['-Wl,--gc-sections']
else
  add_project_arguments(
    dep_compile_args,
    language: 'cpp',
  )
endif

nanobind_sources = files('src/nb_combined.cpp')

nanobind_dep = declare_dependency(
  sources: nanobind_sources,
  include_directories: [incdir],
  dependencies: [py_dep, robin_map_dep],
  compile_args: dep_compile_args,
  link_args: dep_link_args,
  variables: {
    'stubgen': meson.current_source_dir() / 'src/stubgen.py',
  },
)

meson.override_dependency('nanobind', nanobind_dep)

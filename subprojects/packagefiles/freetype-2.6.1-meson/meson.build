project('freetype2', 'c',
  version: '2.6.1',
  license: '(FTL OR GPL-2.0-or-later) AND BSD-3-Clause AND MIT AND MIT-Modern-Variant AND Zlib',
  license_files: [
    'docs/LICENSE.TXT',
    'docs/FTL.TXT',
    'docs/GPLv2.TXT',
  ],
  meson_version: '>=1.1.0')

# NOTE about FreeType versions
# There are 3 versions numbers associated with each releases:
# - official release number (eg. 2.6.1) - accessible via
#   FREETYPE_{MAJOR,MINOR,PATCH} macros from FT_FREETYPE_H
# - libtool-specific version number, this is what is returned by
#   freetype-config --version / pkg-config --modversion (eg. 22.1.16)
# - the platform-specific shared object version number (eg. 6.16.1)
# See https://git.savannah.gnu.org/cgit/freetype/freetype2.git/tree/docs/VERSIONS.TXT
# for more information
release_version = meson.project_version()
libtool_version = '18.1.12'
so_version = '6.12.1'
so_soversion = '6'

pkgmod = import('pkgconfig')

cc = meson.get_compiler('c')

base_sources = [
  'src/autofit/autofit.c',
  'src/base/ftbase.c',
  'src/base/ftbbox.c',
  'src/base/ftbdf.c',
  'src/base/ftbitmap.c',
  'src/base/ftcid.c',
  'src/base/ftfntfmt.c',
  'src/base/ftfstype.c',
  'src/base/ftgasp.c',
  'src/base/ftglyph.c',
  'src/base/ftgxval.c',
  'src/base/ftinit.c',
  'src/base/ftlcdfil.c',
  'src/base/ftmm.c',
  'src/base/ftotval.c',
  'src/base/ftpatent.c',
  'src/base/ftpfr.c',
  'src/base/ftstroke.c',
  'src/base/ftsynth.c',
  'src/base/ftsystem.c',
  'src/base/fttype1.c',
  'src/base/ftwinfnt.c',
  'src/bdf/bdf.c',
  'src/bzip2/ftbzip2.c',
  'src/cache/ftcache.c',
  'src/cff/cff.c',
  'src/cid/type1cid.c',
  'src/gzip/ftgzip.c',
  'src/lzw/ftlzw.c',
  'src/pcf/pcf.c',
  'src/pfr/pfr.c',
  'src/psaux/psaux.c',
  'src/pshinter/pshinter.c',
  'src/psnames/psnames.c',
  'src/raster/raster.c',
  'src/sfnt/sfnt.c',
  'src/smooth/smooth.c',
  'src/truetype/truetype.c',
  'src/type1/type1.c',
  'src/type42/type42.c',
  'src/winfonts/winfnt.c',
]

ft2build_h = [
  'include/ft2build.h',
]

ft_headers = [
  'include/freetype/freetype.h',
  'include/freetype/ftadvanc.h',
  'include/freetype/ftautoh.h',
  'include/freetype/ftbbox.h',
  'include/freetype/ftbdf.h',
  'include/freetype/ftbitmap.h',
  'include/freetype/ftbzip2.h',
  'include/freetype/ftcache.h',
  'include/freetype/ftcffdrv.h',
  'include/freetype/ftchapters.h',
  'include/freetype/ftcid.h',
  'include/freetype/fterrdef.h',
  'include/freetype/fterrors.h',
  'include/freetype/ftfntfmt.h',
  'include/freetype/ftgasp.h',
  'include/freetype/ftglyph.h',
  'include/freetype/ftgxval.h',
  'include/freetype/ftgzip.h',
  'include/freetype/ftimage.h',
  'include/freetype/ftincrem.h',
  'include/freetype/ftlcdfil.h',
  'include/freetype/ftlist.h',
  'include/freetype/ftlzw.h',
  'include/freetype/ftmac.h',
  'include/freetype/ftmm.h',
  'include/freetype/ftmodapi.h',
  'include/freetype/ftmoderr.h',
  'include/freetype/ftotval.h',
  'include/freetype/ftoutln.h',
  'include/freetype/ftpfr.h',
  'include/freetype/ftrender.h',
  'include/freetype/ftsizes.h',
  'include/freetype/ftsnames.h',
  'include/freetype/ftstroke.h',
  'include/freetype/ftsynth.h',
  'include/freetype/ftsystem.h',
  'include/freetype/fttrigon.h',
  'include/freetype/ftttdrv.h',
  'include/freetype/fttypes.h',
  'include/freetype/ftwinfnt.h',
  'include/freetype/t1tables.h',
  'include/freetype/ttnameid.h',
  'include/freetype/tttables.h',
  'include/freetype/tttags.h',
  'include/freetype/ttunpat.h',
]

ft_config_headers = [
  'include/freetype/config/ftconfig.h',
  'include/freetype/config/ftheader.h',
  'include/freetype/config/ftmodule.h',
  'include/freetype/config/ftoption.h',
  'include/freetype/config/ftstdlib.h',
]

if host_machine.system() == 'windows'
  base_sources += [
    'builds/windows/ftdebug.c',
  ]
else
  base_sources += [
    'src/base/ftdebug.c',
  ]
endif

c_args = [
  '-DFT2_BUILD_LIBRARY',
  '-DFT_CONFIG_CONFIG_H=<ftconfig.h>',
  '-DFT_CONFIG_OPTIONS_H=<ftoption.h>'
]

check_headers = []

if ['linux', 'darwin', 'cygwin'].contains(host_machine.system())
  check_headers += [
    ['unistd.h'],
    ['fcntl.h'],
    ['stdint.h'],
  ]
  ftconfig_h_in = files('builds/unix/ftconfig.h.in')
else
  ftconfig_h_in = files('include/freetype/config/ftconfig.h')
endif

conf = configuration_data()
deps = []
incbase = include_directories(['include'])

foreach check : check_headers
  name = check[0]

  if cc.has_header(name)
    conf.set('HAVE_@0@'.format(name.to_upper().underscorify()), 1)
  endif
endforeach

configure_file(input: ftconfig_h_in,
  output: 'ftconfig.h',
  configuration: conf)

ft_config_headers += [configure_file(input: 'include/freetype/config/ftoption.h.in',
  output: 'ftoption.h',
  configuration: conf)]

libfreetype = static_library('freetype', base_sources,
  include_directories: incbase,
  dependencies: deps,
  c_args: c_args,
  gnu_symbol_visibility: 'inlineshidden',
)

freetype_dep = declare_dependency(
  link_with: libfreetype,
  include_directories : incbase,
  dependencies: deps,
)

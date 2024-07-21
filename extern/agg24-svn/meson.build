# We need a patched Agg not available elsewhere, so always use the vendored
# version.

agg_incdir = include_directories('include')

agg_lib = static_library('agg',
  'src/agg_bezier_arc.cpp',
  'src/agg_curves.cpp',
  'src/agg_image_filters.cpp',
  'src/agg_trans_affine.cpp',
  'src/agg_vcgen_contour.cpp',
  'src/agg_vcgen_dash.cpp',
  'src/agg_vcgen_stroke.cpp',
  'src/agg_vpgen_segmentator.cpp',
  include_directories : agg_incdir,
  gnu_symbol_visibility: 'inlineshidden',
)

agg_dep = declare_dependency(
  include_directories: agg_incdir,
  link_with: agg_lib,
)

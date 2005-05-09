SUBDIRS = ctrl . platform

INCLUDES = -I$(top_srcdir)/include

lib_LTLIBRARIES = libagg.la

libagg_la_LDFLAGS = -version-info 2:1:0
libagg_la_SOURCES = agg_arc.cpp agg_arrowhead.cpp agg_bezier_arc.cpp agg_bspline.cpp agg_curves.cpp \
					agg_vcgen_bspline.cpp agg_vcgen_contour.cpp agg_vcgen_dash.cpp agg_vcgen_markers_term.cpp \
					agg_vcgen_smooth_poly1.cpp agg_vcgen_stroke.cpp agg_gsv_text.cpp agg_image_filters.cpp \
					agg_path_storage.cpp agg_rasterizer_scanline_aa.cpp agg_line_aa_basics.cpp agg_line_profile_aa.cpp \
					agg_rounded_rect.cpp agg_sqrt_tables.cpp agg_embedded_raster_fonts.cpp agg_trans_affine.cpp \
					agg_trans_single_path.cpp agg_trans_double_path.cpp \
					agg_trans_warp_magnifier.cpp agg_vpgen_clip_polygon.cpp agg_vpgen_clip_polyline.cpp \
					agg_vpgen_segmentator.cpp
libagg_la_LIBADD = $(top_builddir)/gpc/libagggpc.la $(top_builddir)/src/ctrl/libaggctrl.la

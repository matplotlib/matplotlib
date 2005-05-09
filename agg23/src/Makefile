include ../Makefile.in.$(shell uname)

CXXFLAGS= $(AGGCXXFLAGS) -I../include -L./

SRC_CXX=\
agg_arc.cpp \
agg_arrowhead.cpp \
agg_bezier_arc.cpp \
agg_bspline.cpp \
agg_curves.cpp \
agg_vcgen_contour.cpp \
agg_vcgen_dash.cpp \
agg_vcgen_markers_term.cpp \
agg_vcgen_smooth_poly1.cpp \
agg_vcgen_stroke.cpp \
agg_vcgen_bspline.cpp \
agg_gsv_text.cpp \
agg_image_filters.cpp \
agg_path_storage.cpp \
agg_rasterizer_scanline_aa.cpp \
agg_line_aa_basics.cpp \
agg_line_profile_aa.cpp \
agg_rounded_rect.cpp \
agg_sqrt_tables.cpp \
agg_embedded_raster_fonts.cpp \
agg_trans_affine.cpp \
agg_trans_warp_magnifier.cpp \
agg_trans_single_path.cpp \
agg_trans_double_path.cpp \
agg_vpgen_clip_polygon.cpp \
agg_vpgen_clip_polyline.cpp \
agg_vpgen_segmentator.cpp \
ctrl/agg_cbox_ctrl.cpp \
ctrl/agg_gamma_ctrl.cpp \
ctrl/agg_gamma_spline.cpp \
ctrl/agg_rbox_ctrl.cpp \
ctrl/agg_slider_ctrl.cpp \
ctrl/agg_spline_ctrl.cpp \
ctrl/agg_scale_ctrl.cpp \
ctrl/agg_polygon_ctrl.cpp \
ctrl/agg_bezier_ctrl.cpp

SRC_C=\
../gpc/gpc.c


OBJ=$(SRC_CXX:.cpp=.o) $(SRC_C:.c=.o)

all:	$(OBJ)
	$(LIB) libagg.a $(OBJ) 

clean:
	rm -f *.o *.a ctrl/*.o ../gpc/*.o
	rm -rf SunWS_cache
	rm -rf ctrl/SunWS_cache

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $*.cpp -o $@

%.o: %.c
	$(C) -c $(CXXFLAGS) $*.c -o $@

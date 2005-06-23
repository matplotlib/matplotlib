#include "agg_conv_stroke.h"
%include "agg_conv_stroke.h"

#include "agg_vcgen_stroke.h"
%include "agg_vcgen_stroke.h"

#include "agg_conv_adaptor_vcgen.h"
%include "agg_conv_adaptor_vcgen.h"


%template(conv_adaptor_vcgen_path) agg::conv_adaptor_vcgen<path_t, agg::vcgen_stroke,agg::null_markers>;


%template(conv_adaptor_vcgen_transpath) agg::conv_adaptor_vcgen<transpath_t, agg::vcgen_stroke,agg::null_markers>;

%template(conv_adaptor_vcgen_curve) agg::conv_adaptor_vcgen<curve_t, agg::vcgen_stroke,agg::null_markers>;

%template(conv_adaptor_vcgen_transcurve) agg::conv_adaptor_vcgen<transcurve_t, agg::vcgen_stroke,agg::null_markers>;


%template(conv_adaptor_vcgen_curvetrans) agg::conv_adaptor_vcgen<curvetrans_t, agg::vcgen_stroke,agg::null_markers>;


/*
%template() agg::conv_adaptor_vcgen<path_t, agg::vcgen_stroke,agg::null_markers>;


%template() agg::conv_adaptor_vcgen<transpath_t, agg::vcgen_stroke,agg::null_markers>;

%template() agg::conv_adaptor_vcgen<curve_t, agg::vcgen_stroke,agg::null_markers>;

%template() agg::conv_adaptor_vcgen<transcurve_t, agg::vcgen_stroke,agg::null_markers>;


%template() agg::conv_adaptor_vcgen<curvetrans_t, agg::vcgen_stroke,agg::null_markers>;


%template(conv_adaptor_vcgen_curvetrans) agg::conv_adaptor_vcgen<agg::conv_curve<agg::conv_transform<agg::path_storage, agg::trans_affine> >, agg::vcgen_stroke,agg::null_markers>;

*/

%template(conv_stroke_path) agg::conv_stroke<path_t>;
%template(conv_stroke_transpath) agg::conv_stroke<transpath_t>;
%template(conv_stroke_curve) agg::conv_stroke<curve_t>;
%template(conv_stroke_transcurve) agg::conv_stroke<transcurve_t>;
%template(conv_stroke_curvetrans) agg::conv_stroke<curvetrans_t>;
//%template(conv_stroke_curvetrans) agg::conv_stroke<agg::conv_curve<agg::conv_transform<agg::path_storage, agg::trans_affine> > >;


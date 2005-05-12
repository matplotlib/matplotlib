#include "agg_conv_curve.h"
%include "agg_conv_curve.h"

%template(conv_curve_path) agg::conv_curve<path_t>;
%template(conv_curve_trans) agg::conv_curve<transpath_t>;

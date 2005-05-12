#include "agg_conv_transform.h"
%include "agg_conv_transform.h"

%template(conv_transform_path) agg::conv_transform<path_t, agg::trans_affine>;
%template(conv_transform_curve) agg::conv_transform<curve_t, agg::trans_affine>;

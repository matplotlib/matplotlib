#include "agg_span_image_filter.h"
%include "agg_span_image_filter.i"
#include "agg_span_image_filter_rgba.h"
%include "agg_span_image_filter_rgba.h"

// instantiate the base class
//%template (span_image_filter) agg::span_image_filter<agg::rgba8,interpolator_linear_t, agg::span_allocator<agg::rgba8> >;



%template (span_image_filter_rgba_nn_linear) agg::span_image_filter_rgba_nn<agg::rgba8,agg::order_rgba, interpolator_linear_t, agg::span_allocator<agg::rgba8> >;


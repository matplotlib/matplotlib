// ** agg basics **
%include "agg_basics.h"

//instantiate a few templates
%template(rect) agg::rect_base<int>;
%template(rect_d) agg::rect_base<double>;
%template(unite_rectangles) agg::unite_rectangles<rect>;
%template(unite_rectangles_d) agg::unite_rectangles<rect_d>;
%template(intersect_rectangles) agg::intersect_rectangles<rect>;
%template(intersect_rectangles_d) agg::intersect_rectangles<rect_d>;

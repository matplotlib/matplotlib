#include "agg_renderer_base.h"
%include "agg_renderer_base.h"
%template(renderer_base_rgba) agg::renderer_base<pixfmt_rgba_t>;
%extend agg::renderer_base<pixfmt_rgba_t> { 
  void clear_rgba8(const agg::rgba8& color) {
    self->clear(color);
  }
  void clear_rgba(const agg::rgba& color) {
    self->clear(color);
  }
}





#include "agg_renderer_scanline.h"
%include "agg_renderer_scanline.h"

%template(renderer_scanline_aa_solid_rgba) agg::renderer_scanline_aa_solid<renderer_base_rgba_t>;
%extend agg::renderer_scanline_aa_solid<renderer_base_rgba_t> { 
  void color_rgba8(const agg::rgba8& color) {
    self->color(color);
  }
  void color_rgba(const agg::rgba& color) {
    self->color(color);
  }
}




%template(renderer_scanline_bin_solid_rgba) agg::renderer_scanline_bin_solid<renderer_base_rgba_t>;
%extend agg::renderer_scanline_bin_solid<renderer_base_rgba_t>{ 
  void color_rgba8(const agg::rgba8& color) {
    self->color(color);
  }
  void color_rgba(const agg::rgba& color) {
    self->color(color);
  }
}


//%template(renderer_scanline_aa_imfilt_nn) agg::renderer_scanline_aa<renderer_base_rgba_t, span_imfilt_rgba_nn_interplinear_t>;

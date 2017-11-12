#pragma once

#include "_util.h"

namespace matplotlib::ft2 {

FT_Library library;

struct Face {
  std::shared_ptr<FT_FaceRec_> const ptr;
  std::string const path;
  double const hinting_factor;

  Face(std::string const& path, FT_Long index, double hinting_factor);
};

struct Glyph {
  std::shared_ptr<FT_GlyphRec_> const ptr;

  // Fields copied from the glyph slot.
  FT_Glyph_Metrics const metrics;
  FT_Fixed const linearHoriAdvance;
  FT_Fixed const linearVertAdvance;
  double const hinting_factor;

  Glyph(FT_Face const& face, double hinting_factor);
};

}

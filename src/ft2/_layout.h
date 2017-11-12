#pragma once

#include "_util.h"

namespace matplotlib::ft2 {

using rect_t = std::tuple<double, double, double, double>;

FT_BBox compute_bbox(
  FT_Vector pos,
  std::vector<FT_Glyph> const& glyphs,
  std::vector<rect_t> const& rectangles);

struct Layout {
  private:
  std::vector<FT_Glyph> glyphs;  // Needs to be non-const for rendering :-(
  std::vector<rect_t> rectangles;

  public:
  FT_Int32 const flags;
  FT_BBox const bbox;  // 26.6.

  private:
  bool moved = false;

  Layout(
    std::vector<FT_Glyph> const& glyphs,
    std::vector<rect_t> const& rectangles,
    FT_Int32 flags,
    FT_BBox const& bbox);

  public:
  static Layout simple(
    std::u32string const& string,
    FT_Face const& face,
    FT_Int32 flags);
  static Layout manual(
    std::vector<std::tuple<FT_Glyph, double, double>> const& positioned_glyphs,
    std::vector<rect_t> const& rectangles,
    FT_Int32 flags);
  ~Layout();
  Layout(const Layout& other) = delete;
  Layout(Layout&& other);

  py::array_t<uint8_t> render(bool antialiased);
};

}

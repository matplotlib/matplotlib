#include "_layout.h"
// Use 26.6 throughout.

namespace matplotlib::ft2 {

FT_BBox compute_bbox(
  FT_Vector pos,
  std::vector<FT_Glyph> const& glyphs,
  std::vector<rect_t> const& rectangles)
{
  // Use the advance, because spaces are reported as xMin = xMax = 0, so those
  // at the end would be ignored (OTOH, (0, 0) is always in the bbox so we
  // don't need to special-case that).
  auto bbox = FT_BBox{0, 0, pos.x, pos.y};
  for (auto& glyph: glyphs) {
    auto glyph_bbox = FT_BBox{};
    FT_Glyph_Get_CBox(glyph, FT_GLYPH_BBOX_SUBPIXELS, &glyph_bbox);
    bbox.xMin = std::min(bbox.xMin, glyph_bbox.xMin);
    bbox.xMax = std::max(bbox.xMax, glyph_bbox.xMax);
    bbox.yMin = std::min(bbox.yMin, glyph_bbox.yMin);
    bbox.yMax = std::max(bbox.yMax, glyph_bbox.yMax);
  }
  for (auto& [x0, x1, y0, y1]: rectangles) {
    bbox.xMin = std::min(bbox.xMin, FT_Pos(std::floor(x0 * 64)));
    bbox.xMax = std::max(bbox.xMax, FT_Pos(std::ceil(x1 * 64)));
    bbox.yMin = std::min(bbox.yMin, FT_Pos(std::floor(-y1 * 64)));
    bbox.yMax = std::max(bbox.yMax, FT_Pos(std::ceil(-y0 * 64)));
  }
  return bbox;
}

Layout::Layout(
  std::vector<FT_Glyph> const& glyphs,
  std::vector<rect_t> const& rectangles,
  FT_Int32 flags,
  FT_BBox const& bbox) :
  glyphs{glyphs}, rectangles{rectangles}, flags{flags}, bbox{bbox}
{}

Layout Layout::simple(
  std::u32string const& string,
  FT_Face const& face,
  FT_Int32 flags)
{
  auto has_kerning = FT_HAS_KERNING(face);
  auto previous = char32_t{};
  auto pos = FT_Vector{};
  auto glyphs = std::vector<FT_Glyph>{};
  glyphs.reserve(string.size());
  for (auto& codepoint: string) {
    auto current = FT_Get_Char_Index(face, codepoint);
    if (has_kerning && previous && current) {
      auto delta = FT_Vector{};
      FT_CHECK(
        FT_Get_Kerning, face, previous, current, FT_KERNING_DEFAULT, &delta);
      pos.x += delta.x;
      pos.y += delta.y;
    }
    FT_CHECK(FT_Load_Glyph, face, current, flags);
    auto glyph = FT_Glyph{};
    FT_CHECK(FT_Get_Glyph, face->glyph, &glyph);
    FT_CHECK(FT_Glyph_Transform, glyph, nullptr, &pos);
    glyphs.push_back(glyph);
    // 16.16 -> 26.6.
    pos.x += glyph->advance.x >> 10;
    pos.y += glyph->advance.y >> 10;
  }
  return {glyphs, {}, flags, compute_bbox(pos, glyphs, {})};
}

Layout Layout::manual(
  std::vector<std::tuple<FT_Glyph, double, double>> const& positioned_glyphs,
  std::vector<rect_t> const& rectangles,
  FT_Int32 flags)
{
  auto glyphs = std::vector<FT_Glyph>{};
  glyphs.reserve(positioned_glyphs.size());
  for (auto& [glyph, x, y]: positioned_glyphs) {
    auto copy = FT_Glyph{};
    FT_CHECK(FT_Glyph_Copy, glyph, &copy);
    auto pos =
      FT_Vector{FT_Pos(std::round(x * 64)), FT_Pos(std::round(-y * 64))};
    FT_Glyph_Transform(copy, nullptr, &pos);
    glyphs.push_back(copy);
  }
  return {glyphs, rectangles, flags, compute_bbox({}, glyphs, rectangles)};
}

Layout::~Layout()
{
  if (!moved) {
    std::for_each(glyphs.begin(), glyphs.end(), FT_Done_Glyph);
  }
}

Layout::Layout(Layout&& other) :
  glyphs{std::move(other.glyphs)},
  rectangles{std::move(other.rectangles)},
  flags{std::move(other.flags)},
  bbox{std::move(other.bbox)},
  moved{true}
{}

py::array_t<uint8_t> Layout::render(bool antialiased)
{
  auto xmin = int(std::floor(bbox.xMin / 64.)),
       xmax = int(std::ceil(bbox.xMax / 64.)),
       ymin = int(std::floor(bbox.yMin / 64.)),
       ymax = int(std::ceil(bbox.yMax / 64.)),
       width = xmax - xmin,
       height = ymax - ymin;
  auto array =
    py::array_t<uint8_t>{size_t(height * width), nullptr}
    .attr("reshape")(height, width).cast<py::array_t<uint8_t>>();
  auto buf = array.mutable_unchecked().mutable_data(0);
  std::memset(buf, 0, width * height);
  for (auto glyph: glyphs) {
    if (glyph->format != FT_GLYPH_FORMAT_BITMAP) {
      // Do *not* destroy the previous glyph: this would invalidate a
      // Python-level Glyph holding the old value.
      FT_CHECK(
        FT_Glyph_To_Bitmap, &glyph,
        antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,
        nullptr, false);
    }
    auto b_glyph = reinterpret_cast<FT_BitmapGlyph>(glyph);
    auto bitmap = b_glyph->bitmap;
    if (bitmap.pitch < 0) {  // FIXME: Pitch can be negative.
      throw std::runtime_error("Negative pitches are not supported");
    }
    switch (bitmap.pixel_mode) {
      case FT_PIXEL_MODE_MONO:
        for (auto i = 0u; i < bitmap.rows; ++i) {
          auto src = bitmap.buffer + i * bitmap.pitch,
              dst = buf + (ymax - b_glyph->top + i) * width
                    + b_glyph->left - xmin;
          uint8_t k = 7;
          for (auto j = 0u; j < bitmap.width; ++j, --k, k %= 8) {
            if (*src & (1 << k)) {  // MSB order.
              *dst = 0xff;
            }
            if (!k) {
              ++src;
            }
            ++dst;
          }
        }
        break;
      case FT_PIXEL_MODE_GRAY:
        for (auto i = 0u; i < bitmap.rows; ++i) {
          auto src = bitmap.buffer + i * bitmap.pitch,
              dst = buf + (ymax - b_glyph->top + i) * width
                    + b_glyph->left - xmin;
          for (auto j = 0u; j < bitmap.width; ++j) {
            *dst = std::max(*dst, *src);
            ++src;
            ++dst;
          }
        }
        break;
      default:
        throw std::runtime_error(
          "Unsupported pixel mode: " + std::to_string(bitmap.pixel_mode));
    }
    FT_Done_Glyph(glyph);  // Destroy the new glyph.
  }
  for (auto& [x0, x1, y0, y1]: rectangles) {
    // FIXME: Aliased version.
    auto x0i = int(std::ceil(x0)), x1i = int(std::floor(x1)),
         y0i = int(std::ceil(y0)), y1i = int(std::floor(y1));
    for (auto y = y0i; y < y1i; ++y) {
      std::memset(buf + y * width + x0i - xmin, 0xff, x1i - x0i);
    }
    // We only bother with antialiasing of thin horizontal lines.
    if (y0i > y1i) {  // e.g. y0 = 1.2, y1 = 1.8 -> fill = 0.6.
      auto fill = uint8_t(0xff * (y1 - y0));
      std::transform(
        buf + y1i * width + x0i - xmin,
        buf + y1i * width + x1i - xmin,
        buf + y1i * width + x0i - xmin,
        [&](uint8_t value) { return std::max(fill, value); });
    } else if (y0i == y1i) {  // e.g. y0 = 1.6, y1 = 2.3.
      auto fill0 = uint8_t(0xff * (y0i - y0));
      auto fill1 = uint8_t(0xff * (y1 - y1i));
      std::transform(
        buf + (y0i - 1) * width + x0i - xmin,
        buf + (y0i - 1) * width + x1i - xmin,
        buf + (y0i - 1) * width + x0i - xmin,
        [&](uint8_t value) { return std::max(fill0, value); });
      std::transform(
        buf + y0i * width + x0i - xmin,
        buf + y0i * width + x1i - xmin,
        buf + y0i * width + x0i - xmin,
        [&](uint8_t value) { return std::max(fill1, value); });
    }
  }
  return array;
}

}

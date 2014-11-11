/* -*- mode: c++; c-basic-offset: 4 -*- */

#define NO_IMPORT_ARRAY

#include "_backend_agg.h"
#include "mplutils.h"
#include "MPL_isnan.h"

RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi)
    : width(width),
      height(height),
      dpi(dpi),
      NUMBYTES(width * height * sizeof(pixfmt::color_type)),
      pixBuffer(NULL),
      renderingBuffer(),
      alphaBuffer(NULL),
      alphaMaskRenderingBuffer(),
      alphaMask(alphaMaskRenderingBuffer),
      pixfmtAlphaMask(alphaMaskRenderingBuffer),
      rendererBaseAlphaMask(),
      rendererAlphaMask(),
      scanlineAlphaMask(),
      slineP8(),
      slineBin(),
      pixFmt(),
      rendererBase(),
      rendererAA(),
      rendererBin(),
      theRasterizer(4096),
      lastclippath(NULL),
      _fill_color(agg::rgba(1, 1, 1, 0))
{
    unsigned stride(width * sizeof(pixfmt::color_type));

    pixBuffer = new agg::int8u[NUMBYTES];
    renderingBuffer.attach(pixBuffer, width, height, stride);
    pixFmt.attach(renderingBuffer);
    rendererBase.attach(pixFmt);
    rendererBase.clear(_fill_color);
    rendererAA.attach(rendererBase);
    rendererBin.attach(rendererBase);
    hatchRenderingBuffer.attach(hatchBuffer, HATCH_SIZE, HATCH_SIZE, HATCH_SIZE * 4);
}

RendererAgg::~RendererAgg()
{
    delete[] alphaBuffer;
    delete[] pixBuffer;
}

void RendererAgg::create_alpha_buffers()
{
    if (!alphaBuffer) {
        alphaBuffer = new agg::int8u[width * height];
        alphaMaskRenderingBuffer.attach(alphaBuffer, width, height, width);
        rendererBaseAlphaMask.attach(pixfmtAlphaMask);
        rendererAlphaMask.attach(rendererBaseAlphaMask);
    }
}

bool RendererAgg::render_clippath(py::PathIterator &clippath,
                                  const agg::trans_affine &clippath_trans)
{
    typedef agg::conv_transform<py::PathIterator> transformed_path_t;
    typedef agg::conv_curve<transformed_path_t> curve_t;

    bool has_clippath = (clippath.total_vertices() != 0);

    if (has_clippath &&
        (clippath.get_id() != lastclippath || clippath_trans != lastclippath_transform)) {
        create_alpha_buffers();
        agg::trans_affine trans(clippath_trans);
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        rendererBaseAlphaMask.clear(agg::gray8(0, 0));
        transformed_path_t transformed_clippath(clippath, trans);
        curve_t curved_clippath(transformed_clippath);
        theRasterizer.add_path(curved_clippath);
        rendererAlphaMask.color(agg::gray8(255, 255));
        agg::render_scanlines(theRasterizer, scanlineAlphaMask, rendererAlphaMask);
        lastclippath = clippath.get_id();
        lastclippath_transform = clippath_trans;
    }

    return has_clippath;
}

void RendererAgg::tostring_rgb(uint8_t *buf)
{
    // "Return the rendered buffer as an RGB string"

    int row_len = width * 3;

    agg::rendering_buffer renderingBufferTmp;
    renderingBufferTmp.attach(buf, width, height, row_len);

    agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_rgb24());
}

void RendererAgg::tostring_argb(uint8_t *buf)
{
    //"Return the rendered buffer as an RGB string";

    int row_len = width * 4;

    agg::rendering_buffer renderingBufferTmp;
    renderingBufferTmp.attach(buf, width, height, row_len);
    agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_argb32());
}

void RendererAgg::tostring_bgra(uint8_t *buf)
{
    //"Return the rendered buffer as an RGB string";

    int row_len = width * 4;

    agg::rendering_buffer renderingBufferTmp;
    renderingBufferTmp.attach(buf, width, height, row_len);

    agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_bgra32());
}

agg::rect_i RendererAgg::get_content_extents()
{
    agg::rect_i r(width, height, 0, 0);

    // Looks at the alpha channel to find the minimum extents of the image
    unsigned char *pixel = pixBuffer + 3;
    for (int y = 0; y < (int)height; ++y) {
        for (int x = 0; x < (int)width; ++x) {
            if (*pixel) {
                if (x < r.x1)
                    r.x1 = x;
                if (y < r.y1)
                    r.y1 = y;
                if (x > r.x2)
                    r.x2 = x;
                if (y > r.y2)
                    r.y2 = y;
            }
            pixel += 4;
        }
    }

    r.x1 = std::max(0, r.x1 - 1);
    r.y1 = std::max(0, r.y1 - 1);
    r.x2 = std::max(r.x2 + 1, (int)width);
    r.y2 = std::max(r.y2 + 1, (int)height);

    return r;
}

void RendererAgg::clear()
{
    //"clear the rendered buffer";

    rendererBase.clear(_fill_color);
}

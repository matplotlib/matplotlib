/* -*- mode: c++; c-basic-offset: 4 -*- */

#define NO_IMPORT_ARRAY

#include "_backend_agg.h"
#include "mplutils.h"

void BufferRegion::to_string_argb(uint8_t *buf)
{
    unsigned char *pix;
    unsigned char tmp;
    size_t i, j;

    memcpy(buf, data, height * stride);

    for (i = 0; i < (size_t)height; ++i) {
        pix = buf + i * stride;
        for (j = 0; j < (size_t)width; ++j) {
            // Convert rgba to argb
            tmp = pix[2];
            pix[2] = pix[0];
            pix[0] = tmp;
            pix += 4;
        }
    }
}

RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi)
    : width(width),
      height(height),
      dpi(dpi),
      NUMBYTES(width * height * 4),
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
      theRasterizer(8192),
      lastclippath(NULL),
      _fill_color(agg::rgba(1, 1, 1, 0))
{
    unsigned stride(width * 4);

    pixBuffer = new agg::int8u[NUMBYTES];
    renderingBuffer.attach(pixBuffer, width, height, stride);
    pixFmt.attach(renderingBuffer);
    rendererBase.attach(pixFmt);
    rendererBase.clear(_fill_color);
    rendererAA.attach(rendererBase);
    rendererBin.attach(rendererBase);
    hatch_size = int(dpi);
    hatchBuffer = new agg::int8u[hatch_size * hatch_size * 4];
    hatchRenderingBuffer.attach(hatchBuffer, hatch_size, hatch_size, hatch_size * 4);
}

RendererAgg::~RendererAgg()
{
    delete[] hatchBuffer;
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

BufferRegion *RendererAgg::copy_from_bbox(agg::rect_d in_rect)
{
    agg::rect_i rect(
        (int)in_rect.x1, height - (int)in_rect.y2, (int)in_rect.x2, height - (int)in_rect.y1);

    BufferRegion *reg = NULL;
    reg = new BufferRegion(rect);

    agg::rendering_buffer rbuf;
    rbuf.attach(reg->get_data(), reg->get_width(), reg->get_height(), reg->get_stride());

    pixfmt pf(rbuf);
    renderer_base rb(pf);
    rb.copy_from(renderingBuffer, &rect, -rect.x1, -rect.y1);

    return reg;
}

void RendererAgg::restore_region(BufferRegion &region)
{
    if (region.get_data() == NULL) {
        throw std::runtime_error("Cannot restore_region from NULL data");
    }

    agg::rendering_buffer rbuf;
    rbuf.attach(region.get_data(), region.get_width(), region.get_height(), region.get_stride());

    rendererBase.copy_from(rbuf, 0, region.get_rect().x1, region.get_rect().y1);
}

// Restore the part of the saved region with offsets
void
RendererAgg::restore_region(BufferRegion &region, int xx1, int yy1, int xx2, int yy2, int x, int y )
{
    if (region.get_data() == NULL) {
        throw std::runtime_error("Cannot restore_region from NULL data");
    }

    agg::rect_i &rrect = region.get_rect();

    agg::rect_i rect(xx1 - rrect.x1, (yy1 - rrect.y1), xx2 - rrect.x1, (yy2 - rrect.y1));

    agg::rendering_buffer rbuf;
    rbuf.attach(region.get_data(), region.get_width(), region.get_height(), region.get_stride());

    rendererBase.copy_from(rbuf, &rect, x, y);
}

bool RendererAgg::render_clippath(py::PathIterator &clippath,
                                  const agg::trans_affine &clippath_trans,
                                  e_snap_mode snap_mode)
{
    typedef agg::conv_transform<py::PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removed_t;
    typedef PathClipper<nan_removed_t> clipped_t;
    typedef PathSnapper<clipped_t> snapped_t;
    typedef PathSimplifier<snapped_t> simplify_t;
    typedef agg::conv_curve<simplify_t> curve_t;

    bool has_clippath = (clippath.total_vertices() != 0);

    if (has_clippath &&
        (clippath.get_id() != lastclippath || clippath_trans != lastclippath_transform)) {
        create_alpha_buffers();
        agg::trans_affine trans(clippath_trans);
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        rendererBaseAlphaMask.clear(agg::gray8(0, 0));
        transformed_path_t transformed_clippath(clippath, trans);
        nan_removed_t nan_removed_clippath(transformed_clippath, true, clippath.has_curves());
        clipped_t clipped_clippath(nan_removed_clippath, !clippath.has_curves(), width, height);
        snapped_t snapped_clippath(clipped_clippath, snap_mode, clippath.total_vertices(), 0.0);
        simplify_t simplified_clippath(snapped_clippath,
                                       clippath.should_simplify() && !clippath.has_curves(),
                                       clippath.simplify_threshold());
        curve_t curved_clippath(simplified_clippath);
        theRasterizer.add_path(curved_clippath);
        rendererAlphaMask.color(agg::gray8(255, 255));
        agg::render_scanlines(theRasterizer, scanlineAlphaMask, rendererAlphaMask);
        lastclippath = clippath.get_id();
        lastclippath_transform = clippath_trans;
    }

    return has_clippath;
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

    if (r.x1 == (int)width && r.x2 == 0) {
      // The buffer is completely empty.
      r.x1 = r.y1 = r.x2 = r.y2 = 0;
    } else {
      r.x1 = std::max(0, r.x1);
      r.y1 = std::max(0, r.y1);
      r.x2 = std::min(r.x2 + 1, (int)width);
      r.y2 = std::min(r.y2 + 1, (int)height);
    }

    return r;
}

void RendererAgg::clear()
{
    //"clear the rendered buffer";

    rendererBase.clear(_fill_color);
}

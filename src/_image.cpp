/* -*- mode: c++; c-basic-offset: 4 -*- */

#define NO_IMPORT_ARRAY

#include <vector>

#include "agg_color_rgba.h"
#include "agg_conv_transform.h"
#include "agg_image_accessors.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_rasterizer_sl_clip.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_u.h"
#include "agg_span_allocator.h"
#include "agg_span_image_filter_rgb.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_linear.h"
#include "util/agg_color_conv_rgb8.h"

#include "_image.h"
#include "mplutils.h"
#include "agg_workaround.h"

typedef fixed_blender_rgba_plain<agg::rgba8, agg::order_rgba> fixed_blender_rgba32_plain;
typedef agg::pixfmt_alpha_blend_rgba<fixed_blender_rgba32_plain, agg::rendering_buffer> pixfmt;
typedef fixed_blender_rgba_pre<agg::rgba8, agg::order_rgba> fixed_blender_rgba32_pre;
typedef agg::pixfmt_alpha_blend_rgba<fixed_blender_rgba32_pre, agg::rendering_buffer> pixfmt_pre;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::span_interpolator_linear<> interpolator_type;
typedef agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl> rasterizer;

Image::Image()
    : bufferIn(NULL),
      rbufIn(NULL),
      colsIn(0),
      rowsIn(0),
      bufferOut(NULL),
      rbufOut(NULL),
      colsOut(0),
      rowsOut(0),
      BPP(4),
      interpolation(BILINEAR),
      aspect(ASPECT_FREE),
      bg(1, 1, 1, 0),
      resample(true)
{

}

Image::Image(unsigned numrows, unsigned numcols, bool isoutput)
    : bufferIn(NULL),
      rbufIn(NULL),
      colsIn(0),
      rowsIn(0),
      bufferOut(NULL),
      rbufOut(NULL),
      colsOut(0),
      rowsOut(0),
      BPP(4),
      interpolation(BILINEAR),
      aspect(ASPECT_FREE),
      bg(1, 1, 1, 0),
      resample(true)
{
    if (isoutput) {
        rowsOut = numrows;
        colsOut = numcols;
        unsigned NUMBYTES(numrows * numcols * BPP);
        bufferOut = new agg::int8u[NUMBYTES];
        rbufOut = new agg::rendering_buffer;
        rbufOut->attach(bufferOut, colsOut, rowsOut, colsOut * BPP);
    } else {
        rowsIn = numrows;
        colsIn = numcols;
        unsigned NUMBYTES(numrows * numcols * BPP);
        bufferIn = new agg::int8u[NUMBYTES];
        rbufIn = new agg::rendering_buffer;
        rbufIn->attach(bufferIn, colsIn, rowsIn, colsIn * BPP);
    }
}

Image::~Image()
{
    delete[] bufferIn;
    bufferIn = NULL;
    delete rbufIn;
    rbufIn = NULL;
    delete rbufOut;
    rbufOut = NULL;
    delete[] bufferOut;
    bufferOut = NULL;
}

void Image::apply_rotation(double r)
{
    agg::trans_affine M = agg::trans_affine_rotation(r * agg::pi / 180.0);
    srcMatrix *= M;
    imageMatrix *= M;
}

void Image::set_bg(double r, double g, double b, double a)
{
    bg.r = r;
    bg.g = g;
    bg.b = b;
    bg.a = a;
}

void Image::apply_scaling(double sx, double sy)
{
    agg::trans_affine M = agg::trans_affine_scaling(sx, sy);
    srcMatrix *= M;
    imageMatrix *= M;
}

void Image::apply_translation(double tx, double ty)
{
    agg::trans_affine M = agg::trans_affine_translation(tx, ty);
    srcMatrix *= M;
    imageMatrix *= M;
}

void Image::as_rgba_str(agg::int8u *outbuf)
{
    agg::rendering_buffer rb;
    rb.attach(outbuf, colsOut, rowsOut, colsOut * 4);
    rb.copy_from(*rbufOut);
}

void Image::color_conv(int format, agg::int8u *outbuf)
{
    int row_len = colsOut * 4;

    agg::rendering_buffer rtmp;
    rtmp.attach(outbuf, colsOut, rowsOut, row_len);

    switch (format) {
    case 0:
        agg::color_conv(&rtmp, rbufOut, agg::color_conv_rgba32_to_bgra32());
        break;
    case 1:
        agg::color_conv(&rtmp, rbufOut, agg::color_conv_rgba32_to_argb32());
        break;
    default:
        throw "Image::color_conv unknown format";
    }
}

void Image::reset_matrix(void)
{
    srcMatrix.reset();
    imageMatrix.reset();
}

void Image::resize(int numcols, int numrows, int norm, double radius)
{
    if (bufferIn == NULL) {
        throw "You must first load the image";
    }

    if (numcols <= 0 || numrows <= 0) {
        throw "Width and height must have positive values";
    }

    colsOut = numcols;
    rowsOut = numrows;

    size_t NUMBYTES(numrows * numcols * BPP);

    delete[] bufferOut;
    bufferOut = new agg::int8u[NUMBYTES];
    if (bufferOut == NULL) // todo: also handle allocation throw
    {
        throw "Image::resize could not allocate memory";
    }

    delete rbufOut;
    rbufOut = new agg::rendering_buffer;
    rbufOut->attach(bufferOut, numcols, numrows, numcols * BPP);

    // init the output rendering/rasterizing stuff
    pixfmt pixf(*rbufOut);
    renderer_base rb(pixf);
    rb.clear(bg);
    rasterizer ras;
    agg::scanline_u8 sl;

    ras.clip_box(0, 0, numcols, numrows);

    imageMatrix.invert();
    interpolator_type interpolator(imageMatrix);

    typedef agg::span_allocator<agg::rgba8> span_alloc_type;
    span_alloc_type sa;

    // the image path
    agg::path_storage path;
    agg::rendering_buffer rbufPad;

    double x0, y0, x1, y1;

    x0 = 0.0;
    x1 = colsIn;
    y0 = 0.0;
    y1 = rowsIn;

    path.move_to(x0, y0);
    path.line_to(x1, y0);
    path.line_to(x1, y1);
    path.line_to(x0, y1);
    path.close_polygon();
    agg::conv_transform<agg::path_storage> imageBox(path, srcMatrix);
    ras.add_path(imageBox);

    typedef agg::wrap_mode_reflect reflect_type;
    typedef agg::image_accessor_wrap<pixfmt_pre, reflect_type, reflect_type> img_accessor_type;

    pixfmt_pre pixfmtin(*rbufIn);
    img_accessor_type ia(pixfmtin);
    switch (interpolation) {

    case NEAREST: {
        typedef agg::span_image_filter_rgba_nn<img_accessor_type, interpolator_type> span_gen_type;
        typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type>
        renderer_type;
        span_gen_type sg(ia, interpolator);
        renderer_type ri(rb, sa, sg);
        agg::render_scanlines(ras, sl, ri);
    } break;

    case HANNING:
    case HAMMING:
    case HERMITE: {
        agg::image_filter_lut filter;
        switch (interpolation) {
        case HANNING:
            filter.calculate(agg::image_filter_hanning(), norm);
            break;
        case HAMMING:
            filter.calculate(agg::image_filter_hamming(), norm);
            break;
        case HERMITE:
            filter.calculate(agg::image_filter_hermite(), norm);
            break;
        }
        if (resample) {
            typedef agg::span_image_resample_rgba_affine<img_accessor_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type>
            renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        } else {
            typedef agg::span_image_filter_rgba_2x2<img_accessor_type, interpolator_type>
            span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type>
            renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        }
    } break;
    case BILINEAR:
    case BICUBIC:
    case SPLINE16:
    case SPLINE36:
    case KAISER:
    case QUADRIC:
    case CATROM:
    case GAUSSIAN:
    case BESSEL:
    case MITCHELL:
    case SINC:
    case LANCZOS:
    case BLACKMAN: {
        agg::image_filter_lut filter;
        switch (interpolation) {
        case BILINEAR:
            filter.calculate(agg::image_filter_bilinear(), norm);
            break;
        case BICUBIC:
            filter.calculate(agg::image_filter_bicubic(), norm);
            break;
        case SPLINE16:
            filter.calculate(agg::image_filter_spline16(), norm);
            break;
        case SPLINE36:
            filter.calculate(agg::image_filter_spline36(), norm);
            break;
        case KAISER:
            filter.calculate(agg::image_filter_kaiser(), norm);
            break;
        case QUADRIC:
            filter.calculate(agg::image_filter_quadric(), norm);
            break;
        case CATROM:
            filter.calculate(agg::image_filter_catrom(), norm);
            break;
        case GAUSSIAN:
            filter.calculate(agg::image_filter_gaussian(), norm);
            break;
        case BESSEL:
            filter.calculate(agg::image_filter_bessel(), norm);
            break;
        case MITCHELL:
            filter.calculate(agg::image_filter_mitchell(), norm);
            break;
        case SINC:
            filter.calculate(agg::image_filter_sinc(radius), norm);
            break;
        case LANCZOS:
            filter.calculate(agg::image_filter_lanczos(radius), norm);
            break;
        case BLACKMAN:
            filter.calculate(agg::image_filter_blackman(radius), norm);
            break;
        }
        if (resample) {
            typedef agg::span_image_resample_rgba_affine<img_accessor_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type>
            renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        } else {
            typedef agg::span_image_filter_rgba<img_accessor_type, interpolator_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type>
            renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        }
    } break;
    }
}

void Image::clear()
{
    pixfmt pixf(*rbufOut);
    renderer_base rb(pixf);
    rb.clear(bg);
}

void Image::blend_image(Image &im, unsigned ox, unsigned oy, bool apply_alpha, float alpha)
{
    unsigned thisx = 0, thisy = 0;

    pixfmt pixf(*rbufOut);
    renderer_base rb(pixf);

    bool isflip = (im.rbufOut->stride()) < 0;
    size_t ind = 0;
    for (unsigned j = 0; j < im.rowsOut; j++) {
        if (isflip) {
            thisy = im.rowsOut - j + oy;
        } else {
            thisy = j + oy;
        }

        for (unsigned i = 0; i < im.colsOut; i++) {
            thisx = i + ox;

            if (thisx >= colsOut || thisy >= rowsOut) {
                ind += 4;
                continue;
            }

            pixfmt::color_type p;
            p.r = *(im.bufferOut + ind++);
            p.g = *(im.bufferOut + ind++);
            p.b = *(im.bufferOut + ind++);
            if (apply_alpha) {
                p.a = (pixfmt::value_type) * (im.bufferOut + ind++) * alpha;
            } else {
                p.a = *(im.bufferOut + ind++);
            }
            pixf.blend_pixel(thisx, thisy, p, 255);
        }
    }
}

// utilities for irregular grids
void _bin_indices_middle(
    unsigned int *irows, int nrows, const float *ys1, unsigned long ny, float dy, float y_min)
{
    int i, j, j_last;
    unsigned int *rowstart = irows;
    const float *ys2 = ys1 + 1;
    const float *yl = ys1 + ny;
    float yo = y_min + dy / 2.0;
    float ym = 0.5f * (*ys1 + *ys2);
    // y/rows
    j = 0;
    j_last = j;
    for (i = 0; i < nrows; i++, yo += dy, rowstart++) {
        while (ys2 != yl && yo > ym) {
            ys1 = ys2;
            ys2 = ys1 + 1;
            ym = 0.5f * (*ys1 + *ys2);
            j++;
        }
        *rowstart = j - j_last;
        j_last = j;
    }
}

void _bin_indices_middle_linear(float *arows,
                                unsigned int *irows,
                                int nrows,
                                const float *y,
                                unsigned long ny,
                                float dy,
                                float y_min)
{
    int i;
    int ii = 0;
    int iilast = (int)ny - 1;
    float sc = 1 / dy;
    int iy0 = (int)floor(sc * (y[ii] - y_min));
    int iy1 = (int)floor(sc * (y[ii + 1] - y_min));
    float invgap = 1.0f / (iy1 - iy0);
    for (i = 0; i < nrows && i <= iy0; i++) {
        irows[i] = 0;
        arows[i] = 1.0;
    }
    for (; i < nrows; i++) {
        while (i > iy1 && ii < iilast) {
            ii++;
            iy0 = iy1;
            iy1 = (int)floor(sc * (y[ii + 1] - y_min));
            invgap = 1.0f / (iy1 - iy0);
        }
        if (i >= iy0 && i <= iy1) {
            irows[i] = ii;
            arows[i] = (iy1 - i) * invgap;
        } else
            break;
    }
    for (; i < nrows; i++) {
        irows[i] = iilast - 1;
        arows[i] = 0.0;
    }
}

void _bin_indices(int *irows, int nrows, const double *y, unsigned long ny, double sc, double offs)
{
    int i;
    if (sc * (y[ny - 1] - y[0]) > 0) {
        int ii = 0;
        int iilast = (int)ny - 1;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii + 1] - offs));
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii < iilast) {
                ii++;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii + 1] - offs));
            }
            if (i >= iy0 && i <= iy1)
                irows[i] = ii;
            else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    } else {
        int iilast = (int)ny - 1;
        int ii = iilast;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii - 1] - offs));
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii > 1) {
                ii--;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii - 1] - offs));
            }
            if (i >= iy0 && i <= iy1)
                irows[i] = ii - 1;
            else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    }
}

void _bin_indices_linear(
    float *arows, int *irows, int nrows, double *y, unsigned long ny, double sc, double offs)
{
    int i;
    if (sc * (y[ny - 1] - y[0]) > 0) {
        int ii = 0;
        int iilast = (int)ny - 1;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii + 1] - offs));
        float invgap = 1.0 / (iy1 - iy0);
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii < iilast) {
                ii++;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii + 1] - offs));
                invgap = 1.0 / (iy1 - iy0);
            }
            if (i >= iy0 && i <= iy1) {
                irows[i] = ii;
                arows[i] = (iy1 - i) * invgap;
            } else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    } else {
        int iilast = (int)ny - 1;
        int ii = iilast;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii - 1] - offs));
        float invgap = 1.0 / (iy1 - iy0);
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii > 1) {
                ii--;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii - 1] - offs));
                invgap = 1.0 / (iy1 - iy0);
            }
            if (i >= iy0 && i <= iy1) {
                irows[i] = ii - 1;
                arows[i] = (i - iy0) * invgap;
            } else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    }
}

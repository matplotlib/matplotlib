/* -*- mode: c++; c-basic-offset: 4 -*- */

/* A rewrite of _backend_agg using PyCXX to handle ref counting, etc..
 */

/* Python API mandates Python.h is included *first* */
#include "Python.h"

/* TODO: Remove this dependency */
#include "ft2font.h"
#include "_image.h"
#include "_backend_agg.h"
#include "mplutils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <time.h>
#include <algorithm>

#include "agg_conv_curve.h"
#include "agg_conv_transform.h"
#include "agg_image_accessors.h"
#include "agg_renderer_primitives.h"
#include "agg_scanline_storage_aa.h"
#include "agg_scanline_storage_bin.h"
#include "agg_span_allocator.h"
#include "agg_span_converter.h"
#include "agg_span_image_filter_gray.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_linear.h"
#include "agg_span_pattern_rgba.h"
#include "agg_span_gouraud_rgba.h"
#include "agg_conv_shorten_path.h"
#include "util/agg_color_conv_rgb8.h"

#include "MPL_isnan.h"

#include "numpy/arrayobject.h"
#include "agg_py_transforms.h"

#include "file_compat.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif
#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616
#endif
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif


/*
 Convert dashes from the Python representation as nested sequences to
 the C++ representation as a std::vector<std::pair<double, double> >
 (GCAgg::dash_t) */
void
convert_dashes(const Py::Tuple& dashes, double dpi,
               GCAgg::dash_t& dashes_out, double& dashOffset_out)
{
    if (dashes.length() != 2)
    {
        throw Py::ValueError(
            Printf("Dash descriptor must be a length 2 tuple; found %d",
                   dashes.length()).str()
        );
    }

    dashes_out.clear();
    dashOffset_out = 0.0;
    if (dashes[0].ptr() == Py_None)
    {
        return;
    }

    dashOffset_out = double(Py::Float(dashes[0])) * dpi / 72.0;

    Py::SeqBase<Py::Object> dashSeq = dashes[1];

    size_t Ndash = dashSeq.length();
    if (Ndash % 2 != 0)
    {
        throw Py::ValueError(
            Printf("Dash sequence must be an even length sequence; found %d", Ndash).str()
        );
    }

    dashes_out.clear();
    dashes_out.reserve(Ndash / 2);

    double val0, val1;
    for (size_t i = 0; i < Ndash; i += 2)
    {
        val0 = double(Py::Float(dashSeq[i])) * dpi / 72.0;
        val1 = double(Py::Float(dashSeq[i+1])) * dpi / 72.0;
        dashes_out.push_back(std::make_pair(val0, val1));
    }
}


Py::Object
BufferRegion::to_string(const Py::Tuple &args)
{
    // owned=true to prevent memory leak
    #if PY3K
    return Py::Bytes
    #else
    return Py::String
    #endif
        (PyBytes_FromStringAndSize((const char*)data, height*stride), true);
}


Py::Object
BufferRegion::set_x(const Py::Tuple &args)
{
    args.verify_length(1);
    size_t x = (long) Py::Int(args[0]);
    rect.x1 = x;
    return Py::Object();
}


Py::Object
BufferRegion::set_y(const Py::Tuple &args)
{
    args.verify_length(1);
    size_t y = (long)Py::Int(args[0]);
    rect.y1 = y;
    return Py::Object();
}


Py::Object
BufferRegion::get_extents(const Py::Tuple &args)
{
    args.verify_length(0);

    Py::Tuple extents(4);
    extents[0] = Py::Int(rect.x1);
    extents[1] = Py::Int(rect.y1);
    extents[2] = Py::Int(rect.x2);
    extents[3] = Py::Int(rect.y2);

    return extents;
}


Py::Object
BufferRegion::to_string_argb(const Py::Tuple &args)
{
    // owned=true to prevent memory leak
    Py_ssize_t length;
    unsigned char* pix;
    unsigned char* begin;
    unsigned char tmp;
    size_t i, j;

    PyObject* str = PyBytes_FromStringAndSize((const char*)data, height * stride);
    if (PyBytes_AsStringAndSize(str, (char**)&begin, &length))
    {
        throw Py::TypeError("Could not create memory for blit");
    }

    for (i = 0; i < (size_t)height; ++i)
    {
        pix = begin + i * stride;
        for (j = 0; j < (size_t)width; ++j)
        {
            // Convert rgba to argb
            tmp = pix[2];
            pix[2] = pix[0];
            pix[0] = tmp;
            pix += 4;
        }
    }

    #if PY3K
    return Py::Bytes
    #else
    return Py::String
    #endif
        (str, true);
}


GCAgg::GCAgg(const Py::Object &gc, double dpi) :
        dpi(dpi), isaa(true), dashOffset(0.0)
{
    _VERBOSE("GCAgg::GCAgg");
    linewidth = points_to_pixels(gc.getAttr("_linewidth")) ;
    alpha = Py::Float(gc.getAttr("_alpha"));
    forced_alpha = Py::Boolean(gc.getAttr("_forced_alpha"));
    color = get_color(gc);
    _set_antialiased(gc);
    _set_linecap(gc);
    _set_joinstyle(gc);
    _set_dashes(gc);
    _set_clip_rectangle(gc);
    _set_clip_path(gc);
    _set_snap(gc);
    _set_hatch_path(gc);
    _set_sketch_params(gc);
}


void
GCAgg::_set_antialiased(const Py::Object& gc)
{
    _VERBOSE("GCAgg::antialiased");
    isaa = Py::Boolean(gc.getAttr("_antialiased"));
}


agg::rgba
GCAgg::get_color(const Py::Object& gc)
{
    _VERBOSE("GCAgg::get_color");
    Py::Tuple rgb = Py::Tuple(gc.getAttr("_rgb"));

    double r = Py::Float(rgb[0]);
    double g = Py::Float(rgb[1]);
    double b = Py::Float(rgb[2]);
    double a = Py::Float(rgb[3]);
    return agg::rgba(r, g, b, a);
}


double
GCAgg::points_to_pixels(const Py::Object& points)
{
    _VERBOSE("GCAgg::points_to_pixels");
    double p = Py::Float(points) ;
    return p * dpi / 72.0;
}


void
GCAgg::_set_linecap(const Py::Object& gc)
{
    _VERBOSE("GCAgg::_set_linecap");

    std::string capstyle = Py::String(gc.getAttr("_capstyle")).encode("utf-8");

    if (capstyle == "butt")
    {
        cap = agg::butt_cap;
    }
    else if (capstyle == "round")
    {
        cap = agg::round_cap;
    }
    else if (capstyle == "projecting")
    {
        cap = agg::square_cap;
    }
    else
    {
        throw Py::ValueError(Printf("GC _capstyle attribute must be one of butt, round, projecting; found %s", capstyle.c_str()).str());
    }
}


void
GCAgg::_set_joinstyle(const Py::Object& gc)
{
    _VERBOSE("GCAgg::_set_joinstyle");

    std::string joinstyle = Py::String(gc.getAttr("_joinstyle")).encode("utf-8");

    if (joinstyle == "miter")
    {
        join = agg::miter_join_revert;
    }
    else if (joinstyle == "round")
    {
        join = agg::round_join;
    }
    else if (joinstyle == "bevel")
    {
        join = agg::bevel_join;
    }
    else
    {
        throw Py::ValueError(Printf("GC _joinstyle attribute must be one of butt, round, projecting; found %s", joinstyle.c_str()).str());
    }
}


void
GCAgg::_set_dashes(const Py::Object& gc)
{
    //return the dashOffset, dashes sequence tuple.
    _VERBOSE("GCAgg::_set_dashes");

    Py::Object dash_obj(gc.getAttr("_dashes"));
    if (dash_obj.ptr() == Py_None)
    {
        dashes.clear();
        return;
    }

    convert_dashes(dash_obj, dpi, dashes, dashOffset);
}


void
GCAgg::_set_clip_rectangle(const Py::Object& gc)
{
    //set the clip rectangle from the gc

    _VERBOSE("GCAgg::_set_clip_rectangle");

    Py::Object o(gc.getAttr("_cliprect"));
    cliprect = o;
}


void
GCAgg::_set_clip_path(const Py::Object& gc)
{
    //set the clip path from the gc

    _VERBOSE("GCAgg::_set_clip_path");

    Py::Object method_obj = gc.getAttr("get_clip_path");
    Py::Callable method(method_obj);
    Py::Tuple path_and_transform = method.apply(Py::Tuple());
    if (path_and_transform[0].ptr() != Py_None)
    {
        clippath = path_and_transform[0];
        clippath_trans = py_to_agg_transformation_matrix(path_and_transform[1].ptr());
    }
}


void
GCAgg::_set_snap(const Py::Object& gc)
{
    //set the snap setting

    _VERBOSE("GCAgg::_set_snap");

    Py::Object method_obj = gc.getAttr("get_snap");
    Py::Callable method(method_obj);
    Py::Object py_snap = method.apply(Py::Tuple());
    if (py_snap.isNone())
    {
        snap_mode = SNAP_AUTO;
    }
    else if (py_snap.isTrue())
    {
        snap_mode = SNAP_TRUE;
    }
    else
    {
        snap_mode = SNAP_FALSE;
    }
}


void
GCAgg::_set_hatch_path(const Py::Object& gc)
{
    _VERBOSE("GCAgg::_set_hatch_path");

    Py::Object method_obj = gc.getAttr("get_hatch_path");
    Py::Callable method(method_obj);
    hatchpath = method.apply(Py::Tuple());
    if (hatchpath.ptr() == NULL)
        throw Py::Exception();
}

void
GCAgg::_set_sketch_params(const Py::Object& gc)
{
    _VERBOSE("GCAgg::_get_sketch_params");

    Py::Object method_obj = gc.getAttr("get_sketch_params");
    Py::Callable method(method_obj);
    Py::Object result = method.apply(Py::Tuple());
    if (result.ptr() == Py_None) {
        sketch_scale = 0.0;
    } else {
        Py::Tuple sketch_params(result);
        sketch_scale = Py::Float(sketch_params[0]);
        sketch_length = Py::Float(sketch_params[1]);
        sketch_randomness = Py::Float(sketch_params[2]);
    }
}


const size_t
RendererAgg::PIXELS_PER_INCH(96);


RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi,
                         int debug) :
    width(width),
    height(height),
    dpi(dpi),
    NUMBYTES(width*height*4),
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
    theRasterizer(),
    debug(debug),
    _fill_color(agg::rgba(1, 1, 1, 0))
{
    _VERBOSE("RendererAgg::RendererAgg");
    unsigned stride(width*4);

    pixBuffer = new agg::int8u[NUMBYTES];
    renderingBuffer.attach(pixBuffer, width, height, stride);
    pixFmt.attach(renderingBuffer);
    rendererBase.attach(pixFmt);
    rendererBase.clear(_fill_color);
    rendererAA.attach(rendererBase);
    rendererBin.attach(rendererBase);
    hatchRenderingBuffer.attach(hatchBuffer, HATCH_SIZE, HATCH_SIZE,
                                HATCH_SIZE*4);
}


void
RendererAgg::create_alpha_buffers()
{
    if (!alphaBuffer)
    {
        alphaBuffer = new agg::int8u[width * height];
        alphaMaskRenderingBuffer.attach(alphaBuffer, width, height, width);
        rendererBaseAlphaMask.attach(pixfmtAlphaMask);
        rendererAlphaMask.attach(rendererBaseAlphaMask);
    }
}


template<class R>
void
RendererAgg::set_clipbox(const Py::Object& cliprect, R& rasterizer)
{
    //set the clip rectangle from the gc

    _VERBOSE("RendererAgg::set_clipbox");

    double l, b, r, t;
    if (py_convert_bbox(cliprect.ptr(), l, b, r, t))
    {
        rasterizer.clip_box(std::max(int(floor(l + 0.5)), 0),
                            std::max(int(floor(height - b + 0.5)), 0),
                            std::min(int(floor(r + 0.5)), int(width)),
                            std::min(int(floor(height - t + 0.5)), int(height)));
    }
    else
    {
        rasterizer.clip_box(0, 0, width, height);
    }

    _VERBOSE("RendererAgg::set_clipbox done");
}


std::pair<bool, agg::rgba>
RendererAgg::_get_rgba_face(const Py::Object& rgbFace, double alpha, bool forced_alpha)
{
    _VERBOSE("RendererAgg::_get_rgba_face");
    std::pair<bool, agg::rgba> face;

    if (rgbFace.ptr() == Py_None)
    {
        face.first = false;
    }
    else
    {
        face.first = true;
        Py::Tuple rgb = Py::Tuple(rgbFace);
        if (forced_alpha || rgb.length() < 4)
        {
            face.second = rgb_to_color(rgb, alpha);
        }
        else
        {
            face.second = rgb_to_color(rgb, Py::Float(rgb[3]));
        }
    }
    return face;
}


Py::Object
RendererAgg::copy_from_bbox(const Py::Tuple& args)
{
    //copy region in bbox to buffer and return swig/agg buffer object
    args.verify_length(1);

    Py::Object box_obj = args[0];
    double l, b, r, t;
    if (!py_convert_bbox(box_obj.ptr(), l, b, r, t))
    {
        throw Py::TypeError("Invalid bbox provided to copy_from_bbox");
    }

    agg::rect_i rect((int)l, height - (int)t, (int)r, height - (int)b);

    BufferRegion* reg = NULL;
    try
    {
        reg = new BufferRegion(rect, true);
    }
    catch (...)
    {
        throw Py::MemoryError(
            "RendererAgg::copy_from_bbox could not allocate memory for buffer");
    }

    if (!reg)
    {
        throw Py::MemoryError(
            "RendererAgg::copy_from_bbox could not allocate memory for buffer");
    }

    try
    {
        agg::rendering_buffer rbuf;
        rbuf.attach(reg->data, reg->width, reg->height, reg->stride);

        pixfmt pf(rbuf);
        renderer_base rb(pf);
        rb.copy_from(renderingBuffer, &rect, -rect.x1, -rect.y1);
    }
    catch (...)
    {
        delete reg;
        throw Py::RuntimeError("An unknown error occurred in copy_from_bbox");
    }
    return Py::asObject(reg);
}


Py::Object
RendererAgg::restore_region(const Py::Tuple& args)
{
    //copy BufferRegion to buffer
    args.verify_length(1);
    BufferRegion* region  = static_cast<BufferRegion*>(args[0].ptr());

    if (region->data == NULL)
    {
        throw Py::ValueError("Cannot restore_region from NULL data");
    }

    agg::rendering_buffer rbuf;
    rbuf.attach(region->data,
                region->width,
                region->height,
                region->stride);

    rendererBase.copy_from(rbuf, 0, region->rect.x1, region->rect.y1);

    return Py::Object();
}


// Restore the part of the saved region with offsets
Py::Object
RendererAgg::restore_region2(const Py::Tuple& args)
{
    //copy BufferRegion to buffer
    args.verify_length(7);

    int x(0), y(0), xx1(0), yy1(0), xx2(0), yy2(0);
    try
    {
        xx1 = Py::Int(args[1]);
        yy1 = Py::Int(args[2]);
        xx2 = Py::Int(args[3]);
        yy2 = Py::Int(args[4]);
        x = Py::Int(args[5]);
        y = Py::Int(args[6]);
    }
    catch (Py::TypeError)
    {
        throw Py::TypeError("Invalid input arguments to restore_region2");
    }


    BufferRegion* region  = static_cast<BufferRegion*>(args[0].ptr());

    if (region->data == NULL)
    {
        throw Py::ValueError("Cannot restore_region from NULL data");
    }

    agg::rect_i rect(xx1 - region->rect.x1, (yy1 - region->rect.y1),
                     xx2 - region->rect.x1, (yy2 - region->rect.y1));

    agg::rendering_buffer rbuf;
    rbuf.attach(region->data,
                region->width,
                region->height,
                region->stride);

    rendererBase.copy_from(rbuf, &rect, x, y);

    return Py::Object();
}


bool
RendererAgg::render_clippath(const Py::Object& clippath,
                             const agg::trans_affine& clippath_trans)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef agg::conv_curve<transformed_path_t> curve_t;

    bool has_clippath = (clippath.ptr() != Py_None);

    if (has_clippath &&
        (clippath.ptr() != lastclippath.ptr() ||
         clippath_trans != lastclippath_transform))
    {
        create_alpha_buffers();
        agg::trans_affine trans(clippath_trans);
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        PathIterator clippath_iter(clippath);
        rendererBaseAlphaMask.clear(agg::gray8(0, 0));
        transformed_path_t transformed_clippath(clippath_iter, trans);
        curve_t curved_clippath(transformed_clippath);
        try {
            theRasterizer.add_path(curved_clippath);
        } catch (std::overflow_error &e) {
            throw Py::OverflowError(e.what());
        }
        rendererAlphaMask.color(agg::gray8(255, 255));
        agg::render_scanlines(theRasterizer, scanlineAlphaMask, rendererAlphaMask);
        lastclippath = clippath;
        lastclippath_transform = clippath_trans;
    }

    return has_clippath;
}

#define MARKER_CACHE_SIZE 512


Py::Object
RendererAgg::draw_markers(const Py::Tuple& args)
{
    typedef agg::conv_transform<PathIterator>                  transformed_path_t;
    typedef PathNanRemover<transformed_path_t>                 nan_removed_t;
    typedef PathSnapper<nan_removed_t>                         snap_t;
    typedef agg::conv_curve<snap_t>                            curve_t;
    typedef agg::conv_stroke<curve_t>                          stroke_t;
    typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
    typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
    typedef agg::renderer_scanline_aa_solid<amask_ren_type>    amask_aa_renderer_type;
    args.verify_length(5, 6);

    Py::Object        gc_obj          = args[0];
    Py::Object        marker_path_obj = args[1];
    agg::trans_affine marker_trans    = py_to_agg_transformation_matrix(args[2].ptr());
    Py::Object        path_obj        = args[3];
    agg::trans_affine trans           = py_to_agg_transformation_matrix(args[4].ptr());
    Py::Object        face_obj;
    if (args.size() == 6)
    {
        face_obj = args[5];
    }

    GCAgg gc(gc_obj, dpi);

    // Deal with the difference in y-axis direction
    marker_trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_translation(0.5, (double)height + 0.5);

    PathIterator       marker_path(marker_path_obj);
    transformed_path_t marker_path_transformed(marker_path, marker_trans);
    nan_removed_t      marker_path_nan_removed(marker_path_transformed, true, marker_path.has_curves());
    snap_t             marker_path_snapped(marker_path_nan_removed,
                                           gc.snap_mode,
                                           marker_path.total_vertices(),
                                           gc.linewidth);
    curve_t            marker_path_curve(marker_path_snapped);

    PathIterator path(path_obj);
    transformed_path_t path_transformed(path, trans);
    nan_removed_t      path_nan_removed(path_transformed, false, false);
    snap_t             path_snapped(path_nan_removed,
                                    SNAP_FALSE,
                                    path.total_vertices(),
                                    0.0);
    curve_t            path_curve(path_snapped);
    path_curve.rewind(0);

    facepair_t face = _get_rgba_face(face_obj, gc.alpha, gc.forced_alpha);

    //maxim's suggestions for cached scanlines
    agg::scanline_storage_aa8 scanlines;
    theRasterizer.reset();
    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    agg::rect_i marker_size(0x7FFFFFFF, 0x7FFFFFFF, -0x7FFFFFFF, -0x7FFFFFFF);

    agg::int8u  staticFillCache[MARKER_CACHE_SIZE];
    agg::int8u  staticStrokeCache[MARKER_CACHE_SIZE];
    agg::int8u* fillCache = staticFillCache;
    agg::int8u* strokeCache = staticStrokeCache;

    try
    {
        unsigned fillSize = 0;
        if (face.first)
        {
            try {
                theRasterizer.add_path(marker_path_curve);
            } catch (std::overflow_error &e) {
                throw Py::OverflowError(e.what());
            }
            agg::render_scanlines(theRasterizer, slineP8, scanlines);
            fillSize = scanlines.byte_size();
            if (fillSize >= MARKER_CACHE_SIZE)
            {
                fillCache = new agg::int8u[fillSize];
            }
            scanlines.serialize(fillCache);
            marker_size = agg::rect_i(scanlines.min_x(), scanlines.min_y(),
                                      scanlines.max_x(), scanlines.max_y());
        }

        stroke_t stroke(marker_path_curve);
        stroke.width(gc.linewidth);
        stroke.line_cap(gc.cap);
        stroke.line_join(gc.join);
        theRasterizer.reset();
        try {
            theRasterizer.add_path(stroke);
        } catch (std::overflow_error &e) {
            throw Py::OverflowError(e.what());
        }
        agg::render_scanlines(theRasterizer, slineP8, scanlines);
        unsigned strokeSize = scanlines.byte_size();
        if (strokeSize >= MARKER_CACHE_SIZE)
        {
            strokeCache = new agg::int8u[strokeSize];
        }
        scanlines.serialize(strokeCache);
        marker_size = agg::rect_i(std::min(marker_size.x1, scanlines.min_x()),
                                  std::min(marker_size.y1, scanlines.min_y()),
                                  std::max(marker_size.x2, scanlines.max_x()),
                                  std::max(marker_size.y2, scanlines.max_y()));

        theRasterizer.reset_clipping();
        rendererBase.reset_clipping(true);
        set_clipbox(gc.cliprect, rendererBase);
        bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

        double x, y;

        agg::serialized_scanlines_adaptor_aa8 sa;
        agg::serialized_scanlines_adaptor_aa8::embedded_scanline sl;

        agg::rect_d clipping_rect(
            -1.0 - marker_size.x2,
            -1.0 - marker_size.y2,
            1.0 + width - marker_size.x1,
            1.0 + height - marker_size.y1);

        if (has_clippath)
        {
            while (path_curve.vertex(&x, &y) != agg::path_cmd_stop)
            {
                if (MPL_notisfinite64(x) || MPL_notisfinite64(y))
                {
                    continue;
                }

                /* These values are correctly snapped above -- so we don't want
                   to round here, we really only want to truncate */
                x = floor(x);
                y = floor(y);

                // Cull points outside the boundary of the image.
                // Values that are too large may overflow and create
                // segfaults.
                // http://sourceforge.net/tracker/?func=detail&aid=2865490&group_id=80706&atid=560720
                if (!clipping_rect.hit_test(x, y))
                {
                    continue;
                }

                pixfmt_amask_type pfa(pixFmt, alphaMask);
                amask_ren_type r(pfa);
                amask_aa_renderer_type ren(r);

                if (face.first)
                {
                    ren.color(face.second);
                    sa.init(fillCache, fillSize, x, y);
                    agg::render_scanlines(sa, sl, ren);
                }
                ren.color(gc.color);
                sa.init(strokeCache, strokeSize, x, y);
                agg::render_scanlines(sa, sl, ren);
            }
        }
        else
        {
            while (path_curve.vertex(&x, &y) != agg::path_cmd_stop)
            {
                if (MPL_notisfinite64(x) || MPL_notisfinite64(y))
                {
                    continue;
                }

                /* These values are correctly snapped above -- so we don't want
                   to round here, we really only want to truncate */
                x = floor(x);
                y = floor(y);

                // Cull points outside the boundary of the image.
                // Values that are too large may overflow and create
                // segfaults.
                // http://sourceforge.net/tracker/?func=detail&aid=2865490&group_id=80706&atid=560720
                if (!clipping_rect.hit_test(x, y))
                {
                    continue;
                }

                if (face.first)
                {
                    rendererAA.color(face.second);
                    sa.init(fillCache, fillSize, x, y);
                    agg::render_scanlines(sa, sl, rendererAA);
                }

                rendererAA.color(gc.color);
                sa.init(strokeCache, strokeSize, x, y);
                agg::render_scanlines(sa, sl, rendererAA);
            }
        }
    }
    catch (...)
    {
        if (fillCache != staticFillCache)
            delete[] fillCache;
        if (strokeCache != staticStrokeCache)
            delete[] strokeCache;
        theRasterizer.reset_clipping();
        rendererBase.reset_clipping(true);
        throw;
    }

    if (fillCache != staticFillCache)
        delete[] fillCache;
    if (strokeCache != staticStrokeCache)
        delete[] strokeCache;

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);

    return Py::Object();
}


/**
 * This is a custom span generator that converts spans in the
 * 8-bit inverted greyscale font buffer to rgba that agg can use.
 */
template<class ChildGenerator>
class font_to_rgba
{
public:
    typedef ChildGenerator child_type;
    typedef agg::rgba8 color_type;
    typedef typename child_type::color_type child_color_type;
    typedef agg::span_allocator<child_color_type> span_alloc_type;

private:
    child_type* _gen;
    color_type _color;
    span_alloc_type _allocator;

public:
    font_to_rgba(child_type* gen, color_type color) :
        _gen(gen),
        _color(color)
    {

    }

    inline void
    generate(color_type* output_span, int x, int y, unsigned len)
    {
        _allocator.allocate(len);
        child_color_type* input_span = _allocator.span();
        _gen->generate(input_span, x, y, len);

        do
        {
            *output_span = _color;
            output_span->a = ((unsigned int)_color.a *
                              (unsigned int)input_span->v) >> 8;
            ++output_span;
            ++input_span;
        }
        while (--len);
    }

    void
    prepare()
    {
        _gen->prepare();
    }
};


// MGDTODO: Support clip paths
Py::Object
RendererAgg::draw_text_image(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_text");

    typedef agg::span_allocator<agg::rgba8> color_span_alloc_type;
    typedef agg::span_interpolator_linear<> interpolator_type;
    typedef agg::image_accessor_clip<agg::pixfmt_gray8> image_accessor_type;
    typedef agg::span_image_filter_gray<image_accessor_type,
                                        interpolator_type> image_span_gen_type;
    typedef font_to_rgba<image_span_gen_type> span_gen_type;
    typedef agg::renderer_scanline_aa<renderer_base, color_span_alloc_type,
                                      span_gen_type> renderer_type;

    args.verify_length(5);

    const unsigned char* buffer = NULL;
    int width, height;
    Py::Object image_obj = args[0];

    if (PyArray_Check(image_obj.ptr()))
    {
        PyObject* image_array = PyArray_FromObject(
            image_obj.ptr(), PyArray_UBYTE, 2, 2);
        if (!image_array)
        {
            throw Py::ValueError(
                "First argument to draw_text_image must be a FT2Font.Image object or a Nx2 uint8 numpy array.");
        }
        image_obj = Py::Object(image_array, true);
        buffer = (unsigned char *)PyArray_DATA(image_array);
        width = PyArray_DIM(image_array, 1);
        height = PyArray_DIM(image_array, 0);
    }
    else
    {
        FT2Image* image = static_cast<FT2Image *>(
            Py::getPythonExtensionBase(image_obj.ptr()));
        if (!image->get_buffer())
        {
            throw Py::ValueError(
                "First argument to draw_text_image must be a FT2Font.Image object or a Nx2 uint8 numpy array.");
        }
        buffer = image->get_buffer();
        width = image->get_width();
        height = image->get_height();
    }

    int x(0), y(0);
    try
    {
        x = Py::Int(args[1]);
        y = Py::Int(args[2]);
    }
    catch (Py::TypeError)
    {
        throw Py::TypeError("Invalid input arguments to draw_text_image");
    }

    double angle = Py::Float(args[3]);

    GCAgg gc(args[4], dpi);

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);

    agg::rendering_buffer srcbuf((agg::int8u*)buffer, width, height, width);
    agg::pixfmt_gray8 pixf_img(srcbuf);

    agg::trans_affine mtx;
    mtx *= agg::trans_affine_translation(0, -height);
    mtx *= agg::trans_affine_rotation(-angle * agg::pi / 180.0);
    mtx *= agg::trans_affine_translation(x, y);

    agg::path_storage rect;
    rect.move_to(0, 0);
    rect.line_to(width, 0);
    rect.line_to(width, height);
    rect.line_to(0, height);
    rect.line_to(0, 0);
    agg::conv_transform<agg::path_storage> rect2(rect, mtx);

    agg::trans_affine inv_mtx(mtx);
    inv_mtx.invert();

    agg::image_filter_lut filter;
    filter.calculate(agg::image_filter_spline36());
    interpolator_type interpolator(inv_mtx);
    color_span_alloc_type sa;
    image_accessor_type ia(pixf_img, 0);
    image_span_gen_type image_span_generator(ia, interpolator, filter);
    span_gen_type output_span_generator(&image_span_generator, gc.color);
    renderer_type ri(rendererBase, sa, output_span_generator);

    try {
        theRasterizer.add_path(rect2);
    } catch (std::overflow_error &e) {
        throw Py::OverflowError(e.what());
    }
    agg::render_scanlines(theRasterizer, slineP8, ri);

    return Py::Object();
}

class span_conv_alpha
{
public:
    typedef agg::rgba8 color_type;

    double m_alpha;

    span_conv_alpha(double alpha) :
        m_alpha(alpha)
    {
    }

    void prepare() {}
    void generate(color_type* span, int x, int y, unsigned len) const
    {
        do
            {
                span->a = (agg::int8u)((double)span->a * m_alpha);
                ++span;
            }
        while(--len);
    }
};


Py::Object
RendererAgg::draw_image(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_image");

    args.verify_length(4, 7); // 7 if affine matrix if given

    GCAgg gc(args[0], dpi);
    Image *image = static_cast<Image*>(args[3].ptr());
    bool has_clippath = false;
    agg::trans_affine affine_trans;
    bool has_affine = false;
    double x, y, w, h;
    double alpha;

    if (args.size() == 7)
    {
        has_affine = true;
        x = Py::Float(args[1]);
        y = Py::Float(args[2]);
        w = Py::Float(args[4]);
        h = Py::Float(args[5]);
        affine_trans = py_to_agg_transformation_matrix(args[6].ptr());
    }
    else
    {
        x = mpl_round(Py::Float(args[1]));
        y = mpl_round(Py::Float(args[2]));
        w = h = 0; /* w and h not used in this case, but assign to prevent
                  warnings from the compiler */
    }

    alpha = gc.alpha;

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    Py::Tuple empty;
    image->flipud_out(empty);
    pixfmt pixf(*(image->rbufOut));

    if (has_affine | has_clippath)
    {
        agg::trans_affine mtx;
        agg::path_storage rect;

        if (has_affine)
        {
            mtx *= agg::trans_affine_scaling(1, -1);
            mtx *= agg::trans_affine_translation(0, image->rowsOut);
            mtx *= agg::trans_affine_scaling(w / (image->colsOut),
                                             h / (image->rowsOut));
            mtx *= agg::trans_affine_translation(x, y);
            mtx *= affine_trans;
            mtx *= agg::trans_affine_scaling(1.0, -1.0);
            mtx *= agg::trans_affine_translation(0.0, (double) height);
        }
        else
        {
            mtx *= agg::trans_affine_translation(
                (int)x,
                (int)(height - (y + image->rowsOut)));
        }

        rect.move_to(0, 0);
        rect.line_to(image->colsOut, 0);
        rect.line_to(image->colsOut, image->rowsOut);
        rect.line_to(0, image->rowsOut);
        rect.line_to(0, 0);

        agg::conv_transform<agg::path_storage> rect2(rect, mtx);

        agg::trans_affine inv_mtx(mtx);
        inv_mtx.invert();

        typedef agg::span_allocator<agg::rgba8> color_span_alloc_type;
        typedef agg::image_accessor_clip<agg::pixfmt_rgba32_plain>
            image_accessor_type;
        typedef agg::span_interpolator_linear<> interpolator_type;
        typedef agg::span_image_filter_rgba_nn<image_accessor_type,
                                               interpolator_type> image_span_gen_type;
        typedef agg::span_converter<image_span_gen_type, span_conv_alpha> span_conv;

        color_span_alloc_type sa;
        image_accessor_type ia(pixf, agg::rgba8(0, 0, 0, 0));
        interpolator_type interpolator(inv_mtx);
        image_span_gen_type image_span_generator(ia, interpolator);
        span_conv_alpha conv_alpha(alpha);
        span_conv spans(image_span_generator, conv_alpha);

        if (has_clippath)
        {
            typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type>
                pixfmt_amask_type;
            typedef agg::renderer_base<pixfmt_amask_type> amask_ren_type;
            typedef agg::renderer_scanline_aa<amask_ren_type,
                                              color_span_alloc_type,
                                              span_conv>
                renderer_type_alpha;

            pixfmt_amask_type pfa(pixFmt, alphaMask);
            amask_ren_type r(pfa);
            renderer_type_alpha ri(r, sa, spans);

            try {
                theRasterizer.add_path(rect2);
            } catch (std::overflow_error &e) {
                throw Py::OverflowError(e.what());
            }
            agg::render_scanlines(theRasterizer, scanlineAlphaMask, ri);
        }
        else
        {
            typedef agg::renderer_base<pixfmt> ren_type;
            typedef agg::renderer_scanline_aa<ren_type,
                                              color_span_alloc_type,
                                              span_conv>
                renderer_type;

            ren_type r(pixFmt);
            renderer_type ri(r, sa, spans);

            try {
                theRasterizer.add_path(rect2);
            } catch (std::overflow_error &e) {
                throw Py::OverflowError(e.what());
            }
            agg::render_scanlines(theRasterizer, slineP8, ri);
        }

    }
    else
    {
        set_clipbox(gc.cliprect, rendererBase);
        rendererBase.blend_from(
            pixf, 0, (int)x, (int)(height - (y + image->rowsOut)),
            (agg::int8u)(alpha * 255));
    }

    rendererBase.reset_clipping(true);
    image->flipud_out(empty);

    return Py::Object();
}


template<class path_t>
void RendererAgg::_draw_path(path_t& path, bool has_clippath,
                             const facepair_t& face, const GCAgg& gc)
{
    typedef agg::conv_stroke<path_t>                           stroke_t;
    typedef agg::conv_dash<path_t>                             dash_t;
    typedef agg::conv_stroke<dash_t>                           stroke_dash_t;
    typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
    typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
    typedef agg::renderer_scanline_aa_solid<amask_ren_type>    amask_aa_renderer_type;
    typedef agg::renderer_scanline_bin_solid<amask_ren_type>   amask_bin_renderer_type;

    // Render face
    if (face.first)
    {
        try {
            theRasterizer.add_path(path);
        } catch (std::overflow_error &e) {
            throw Py::OverflowError(e.what());
        }

        if (gc.isaa)
        {
            if (has_clippath)
            {
                pixfmt_amask_type pfa(pixFmt, alphaMask);
                amask_ren_type r(pfa);
                amask_aa_renderer_type ren(r);
                ren.color(face.second);
                agg::render_scanlines(theRasterizer, scanlineAlphaMask, ren);
            }
            else
            {
                rendererAA.color(face.second);
                agg::render_scanlines(theRasterizer, slineP8, rendererAA);
            }
        }
        else
        {
            if (has_clippath)
            {
                pixfmt_amask_type pfa(pixFmt, alphaMask);
                amask_ren_type r(pfa);
                amask_bin_renderer_type ren(r);
                ren.color(face.second);
                agg::render_scanlines(theRasterizer, scanlineAlphaMask, ren);
            }
            else
            {
                rendererBin.color(face.second);
                agg::render_scanlines(theRasterizer, slineP8, rendererBin);
            }
        }
    }

    // Render hatch
    if (!gc.hatchpath.isNone())
    {
        // Reset any clipping that may be in effect, since we'll be
        // drawing the hatch in a scratch buffer at origin (0, 0)
        theRasterizer.reset_clipping();
        rendererBase.reset_clipping(true);

        // Create and transform the path
        typedef agg::conv_transform<PathIterator> hatch_path_trans_t;
        typedef agg::conv_curve<hatch_path_trans_t> hatch_path_curve_t;
        typedef agg::conv_stroke<hatch_path_curve_t> hatch_path_stroke_t;

        PathIterator hatch_path(gc.hatchpath);
        agg::trans_affine hatch_trans;
        hatch_trans *= agg::trans_affine_scaling(1.0, -1.0);
        hatch_trans *= agg::trans_affine_translation(0.0, 1.0);
        hatch_trans *= agg::trans_affine_scaling(HATCH_SIZE, HATCH_SIZE);
        hatch_path_trans_t hatch_path_trans(hatch_path, hatch_trans);
        hatch_path_curve_t hatch_path_curve(hatch_path_trans);
        hatch_path_stroke_t hatch_path_stroke(hatch_path_curve);
        hatch_path_stroke.width(1.0);
        hatch_path_stroke.line_cap(agg::square_cap);

        // Render the path into the hatch buffer
        pixfmt hatch_img_pixf(hatchRenderingBuffer);
        renderer_base rb(hatch_img_pixf);
        renderer_aa rs(rb);
        rb.clear(_fill_color);
        rs.color(gc.color);

        try {
            theRasterizer.add_path(hatch_path_curve);
        } catch (std::overflow_error &e) {
            throw Py::OverflowError(e.what());
        }
        agg::render_scanlines(theRasterizer, slineP8, rs);
        try {
            theRasterizer.add_path(hatch_path_stroke);
        } catch (std::overflow_error &e) {
            throw Py::OverflowError(e.what());
        }
        agg::render_scanlines(theRasterizer, slineP8, rs);

        // Put clipping back on, if originally set on entry to this
        // function
        set_clipbox(gc.cliprect, theRasterizer);
        if (has_clippath)
            render_clippath(gc.clippath, gc.clippath_trans);

        // Transfer the hatch to the main image buffer
        typedef agg::image_accessor_wrap < pixfmt,
        agg::wrap_mode_repeat_auto_pow2,
        agg::wrap_mode_repeat_auto_pow2 > img_source_type;
        typedef agg::span_pattern_rgba<img_source_type> span_gen_type;
        agg::span_allocator<agg::rgba8> sa;
        img_source_type img_src(hatch_img_pixf);
        span_gen_type sg(img_src, 0, 0);
        try {
            theRasterizer.add_path(path);
        } catch (std::overflow_error &e) {
            throw Py::OverflowError(e.what());
        }

        if (has_clippath)
        {
           pixfmt_amask_type pfa(pixFmt, alphaMask);
           amask_ren_type ren(pfa);
           agg::render_scanlines_aa(theRasterizer, slineP8, ren, sa, sg);
        }
        else
        {
           agg::render_scanlines_aa(theRasterizer, slineP8, rendererBase, sa, sg);
        }
    }

    // Render stroke
    if (gc.linewidth != 0.0)
    {
        double linewidth = gc.linewidth;
        if (!gc.isaa)
        {
            linewidth = (linewidth < 0.5) ? 0.5 : mpl_round(linewidth);
        }
        if (gc.dashes.size() == 0)
        {
            stroke_t stroke(path);
            stroke.width(linewidth);
            stroke.line_cap(gc.cap);
            stroke.line_join(gc.join);
            try {
                theRasterizer.add_path(stroke);
            } catch (std::overflow_error &e) {
                throw Py::OverflowError(e.what());
            }
        }
        else
        {
            dash_t dash(path);
            for (GCAgg::dash_t::const_iterator i = gc.dashes.begin();
                    i != gc.dashes.end(); ++i)
            {
                double val0 = i->first;
                double val1 = i->second;
                if (!gc.isaa)
                {
                    val0 = (int)val0 + 0.5;
                    val1 = (int)val1 + 0.5;
                }
                dash.add_dash(val0, val1);
            }
            stroke_dash_t stroke(dash);
            stroke.line_cap(gc.cap);
            stroke.line_join(gc.join);
            stroke.width(linewidth);
            try {
                theRasterizer.add_path(stroke);
            } catch (std::overflow_error &e) {
                throw Py::OverflowError(e.what());
            }
        }

        if (gc.isaa)
        {
            if (has_clippath)
            {
                pixfmt_amask_type pfa(pixFmt, alphaMask);
                amask_ren_type r(pfa);
                amask_aa_renderer_type ren(r);
                ren.color(gc.color);
                agg::render_scanlines(theRasterizer, scanlineAlphaMask, ren);
            }
            else
            {
                rendererAA.color(gc.color);
                agg::render_scanlines(theRasterizer, slineP8, rendererAA);
            }
        }
        else
        {
            if (has_clippath)
            {
                pixfmt_amask_type pfa(pixFmt, alphaMask);
                amask_ren_type r(pfa);
                amask_bin_renderer_type ren(r);
                ren.color(gc.color);
                agg::render_scanlines(theRasterizer, scanlineAlphaMask, ren);
            }
            else
            {
                rendererBin.color(gc.color);
                agg::render_scanlines(theRasterizer, slineBin, rendererBin);
            }
        }
    }
}


Py::Object
RendererAgg::draw_path(const Py::Tuple& args)
{
    typedef agg::conv_transform<PathIterator>  transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removed_t;
    typedef PathClipper<nan_removed_t>         clipped_t;
    typedef PathSnapper<clipped_t>             snapped_t;
    typedef PathSimplifier<snapped_t>          simplify_t;
    typedef agg::conv_curve<simplify_t>        curve_t;
    typedef Sketch<curve_t>                    sketch_t;

    _VERBOSE("RendererAgg::draw_path");
    args.verify_length(3, 4);

    GCAgg gc(args[0], dpi);
    PathIterator path(args[1]);
    agg::trans_affine trans = py_to_agg_transformation_matrix(args[2].ptr());
    Py::Object face_obj;
    if (args.size() == 4)
        face_obj = args[3];

    facepair_t face = _get_rgba_face(face_obj, gc.alpha, gc.forced_alpha);

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_translation(0.0, (double)height);
    bool clip = !face.first && gc.hatchpath.isNone() && !path.has_curves();
    bool simplify = path.should_simplify() && clip;
    double snapping_linewidth = gc.linewidth;
    if (gc.color.a == 0.0) {
        snapping_linewidth = 0.0;
    }

    transformed_path_t tpath(path, trans);
    nan_removed_t      nan_removed(tpath, true, path.has_curves());
    clipped_t          clipped(nan_removed, clip, width, height);
    snapped_t          snapped(clipped, gc.snap_mode, path.total_vertices(), snapping_linewidth);
    simplify_t         simplified(snapped, simplify, path.simplify_threshold());
    curve_t            curve(simplified);
    sketch_t           sketch(curve, gc.sketch_scale, gc.sketch_length, gc.sketch_randomness);

    try
    {
        _draw_path(sketch, has_clippath, face, gc);
    }
    catch (const char* e)
    {
        throw Py::RuntimeError(e);
    }

    return Py::Object();
}


template<class PathGenerator, int check_snap, int has_curves>
Py::Object
RendererAgg::_draw_path_collection_generic
(GCAgg&                         gc,
 agg::trans_affine              master_transform,
 const Py::Object&              cliprect,
 const Py::Object&              clippath,
 const agg::trans_affine&       clippath_trans,
 const PathGenerator&           path_generator,
 const Py::Object&              transforms_obj,
 const Py::Object&              offsets_obj,
 const agg::trans_affine&       offset_trans,
 const Py::Object&              facecolors_obj,
 const Py::Object&              edgecolors_obj,
 const Py::SeqBase<Py::Float>&  linewidths,
 const Py::SeqBase<Py::Object>& linestyles_obj,
 const Py::SeqBase<Py::Int>&    antialiaseds,
 const bool                     data_offsets)
{
    typedef agg::conv_transform<typename PathGenerator::path_iterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t>                         nan_removed_t;
    typedef PathClipper<nan_removed_t>                                 clipped_t;
    typedef PathSnapper<clipped_t>                                     snapped_t;
    typedef agg::conv_curve<snapped_t>                                 snapped_curve_t;
    typedef agg::conv_curve<clipped_t>                                 curve_t;

    PyArrayObject* offsets = (PyArrayObject*)PyArray_FromObject
        (offsets_obj.ptr(), PyArray_DOUBLE, 0, 2);
    if (!offsets ||
        (PyArray_NDIM(offsets) == 2 && PyArray_DIM(offsets, 1) != 2) ||
        (PyArray_NDIM(offsets) == 1 && PyArray_DIM(offsets, 0) != 0))
    {
        Py_XDECREF(offsets);
        throw Py::ValueError("Offsets array must be Nx2");
    }
    Py::Object offsets_arr_obj((PyObject*)offsets, true);

    PyArrayObject* facecolors = (PyArrayObject*)PyArray_FromObject
        (facecolors_obj.ptr(), PyArray_DOUBLE, 1, 2);
    if (!facecolors ||
        (PyArray_NDIM(facecolors) == 1 && PyArray_DIM(facecolors, 0) != 0) ||
        (PyArray_NDIM(facecolors) == 2 && PyArray_DIM(facecolors, 1) != 4))
    {
        Py_XDECREF(facecolors);
        throw Py::ValueError("Facecolors must be a Nx4 numpy array or empty");
    }
    Py::Object facecolors_arr_obj((PyObject*)facecolors, true);

    PyArrayObject* edgecolors = (PyArrayObject*)PyArray_FromObject
        (edgecolors_obj.ptr(), PyArray_DOUBLE, 1, 2);
    if (!edgecolors ||
        (PyArray_NDIM(edgecolors) == 1 && PyArray_DIM(edgecolors, 0) != 0) ||
        (PyArray_NDIM(edgecolors) == 2 && PyArray_DIM(edgecolors, 1) != 4))
    {
        Py_XDECREF(edgecolors);
        throw Py::ValueError("Edgecolors must be a Nx4 numpy array");
    }
    Py::Object edgecolors_arr_obj((PyObject*)edgecolors, true);

    PyArrayObject* transforms_arr = (PyArrayObject*)PyArray_FromObject
        (transforms_obj.ptr(), PyArray_DOUBLE, 1, 3);
    if (!transforms_arr ||
        (PyArray_NDIM(transforms_arr) == 1 && PyArray_DIM(transforms_arr, 0) != 0) ||
        (PyArray_NDIM(transforms_arr) == 2) ||
        (PyArray_NDIM(transforms_arr) == 3 &&
         ((PyArray_DIM(transforms_arr, 1) != 3) ||
          (PyArray_DIM(transforms_arr, 2) != 3))))
    {
        Py_XDECREF(transforms_arr);
        throw Py::ValueError("Transforms must be a Nx3x3 numpy array");
    }

    size_t Npaths      = path_generator.num_paths();
    size_t Noffsets    = offsets->dimensions[0];
    size_t N           = std::max(Npaths, Noffsets);
    size_t Ntransforms = transforms_arr->dimensions[0];
    size_t Nfacecolors = facecolors->dimensions[0];
    size_t Nedgecolors = edgecolors->dimensions[0];
    size_t Nlinewidths = linewidths.length();
    size_t Nlinestyles = std::min(linestyles_obj.length(), N);
    size_t Naa         = antialiaseds.length();

    if ((Nfacecolors == 0 && Nedgecolors == 0) || Npaths == 0)
    {
        Py_XDECREF(transforms_arr);
        return Py::Object();
    }

    size_t i = 0;

    // Convert all of the transforms up front
    typedef std::vector<agg::trans_affine> transforms_t;
    transforms_t transforms;
    transforms.reserve(Ntransforms);
    for (i = 0; i < Ntransforms; ++i)
    {
        /* TODO: Use a Numpy iterator */
        agg::trans_affine trans(
            *(double *)PyArray_GETPTR3(transforms_arr, i, 0, 0),
            *(double *)PyArray_GETPTR3(transforms_arr, i, 1, 0),
            *(double *)PyArray_GETPTR3(transforms_arr, i, 0, 1),
            *(double *)PyArray_GETPTR3(transforms_arr, i, 1, 1),
            *(double *)PyArray_GETPTR3(transforms_arr, i, 0, 2),
            *(double *)PyArray_GETPTR3(transforms_arr, i, 1, 2));
        trans *= master_transform;

        transforms.push_back(trans);
    }

    // Convert all the dashes up front
    typedef std::vector<std::pair<double, GCAgg::dash_t> > dashes_t;
    dashes_t dashes;
    dashes.resize(Nlinestyles);
    i = 0;
    for (dashes_t::iterator d = dashes.begin();
         d != dashes.end(); ++d, ++i)
    {
        convert_dashes(Py::Tuple(linestyles_obj[i]), dpi, d->second,
                       d->first);
    }

    // Handle any clipping globally
    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(cliprect, theRasterizer);
    bool has_clippath = render_clippath(clippath, clippath_trans);

    // Set some defaults, assuming no face or edge
    gc.linewidth = 0.0;
    facepair_t face;
    face.first = Nfacecolors != 0;
    agg::trans_affine trans;

    for (i = 0; i < N; ++i)
    {
        typename PathGenerator::path_iterator path = path_generator(i);

        if (Ntransforms)
        {
            trans = transforms[i % Ntransforms];
        }
        else
        {
            trans = master_transform;
        }

        if (Noffsets)
        {
            double xo = *(double*)PyArray_GETPTR2(offsets, i % Noffsets, 0);
            double yo = *(double*)PyArray_GETPTR2(offsets, i % Noffsets, 1);
            offset_trans.transform(&xo, &yo);
            if (data_offsets) {
                trans = agg::trans_affine_translation(xo, yo) * trans;
            } else {
                trans *= agg::trans_affine_translation(xo, yo);
            }
        }

        // These transformations must be done post-offsets
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        if (Nfacecolors)
        {
            size_t fi = i % Nfacecolors;
            face.second = agg::rgba(
                *(double*)PyArray_GETPTR2(facecolors, fi, 0),
                *(double*)PyArray_GETPTR2(facecolors, fi, 1),
                *(double*)PyArray_GETPTR2(facecolors, fi, 2),
                *(double*)PyArray_GETPTR2(facecolors, fi, 3));
        }

        if (Nedgecolors)
        {
            size_t ei = i % Nedgecolors;
            gc.color = agg::rgba(
                *(double*)PyArray_GETPTR2(edgecolors, ei, 0),
                *(double*)PyArray_GETPTR2(edgecolors, ei, 1),
                *(double*)PyArray_GETPTR2(edgecolors, ei, 2),
                *(double*)PyArray_GETPTR2(edgecolors, ei, 3));

            if (Nlinewidths)
            {
                gc.linewidth = double(Py::Float(linewidths[i % Nlinewidths])) * dpi / 72.0;
            }
            else
            {
                gc.linewidth = 1.0;
            }
            if (Nlinestyles)
            {
                gc.dashes = dashes[i % Nlinestyles].second;
                gc.dashOffset = dashes[i % Nlinestyles].first;
            }
        }

        bool do_clip = !face.first && gc.hatchpath.isNone() && !has_curves;

        if (check_snap)
        {
            gc.isaa = Py::Boolean(antialiaseds[i % Naa]);

            transformed_path_t tpath(path, trans);
            nan_removed_t      nan_removed(tpath, true, has_curves);
            clipped_t          clipped(nan_removed, do_clip, width, height);
            snapped_t          snapped(clipped, gc.snap_mode,
                                       path.total_vertices(), gc.linewidth);
            if (has_curves)
            {
                snapped_curve_t curve(snapped);
                _draw_path(curve, has_clippath, face, gc);
            }
            else
            {
                _draw_path(snapped, has_clippath, face, gc);
            }
        }
        else
        {
            gc.isaa = Py::Boolean(antialiaseds[i % Naa]);

            transformed_path_t tpath(path, trans);
            nan_removed_t      nan_removed(tpath, true, has_curves);
            clipped_t          clipped(nan_removed, do_clip, width, height);
            if (has_curves)
            {
                curve_t curve(clipped);
                _draw_path(curve, has_clippath, face, gc);
            }
            else
            {
                _draw_path(clipped, has_clippath, face, gc);
            }
        }
    }

    Py_XDECREF(transforms_arr);

    return Py::Object();
}


class PathListGenerator
{
    const Py::SeqBase<Py::Object>& m_paths;
    size_t m_npaths;

public:
    typedef PathIterator path_iterator;

    inline
    PathListGenerator(const Py::SeqBase<Py::Object>& paths) :
        m_paths(paths), m_npaths(paths.size())
    {

    }

    inline size_t
    num_paths() const
    {
        return m_npaths;
    }

    inline path_iterator
    operator()(size_t i) const
    {
        return PathIterator(m_paths[i % m_npaths]);
    }
};


Py::Object
RendererAgg::draw_path_collection(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_path_collection");
    args.verify_length(13);

    Py::Object gc_obj = args[0];
    GCAgg gc(gc_obj, dpi);
    agg::trans_affine       master_transform = py_to_agg_transformation_matrix(args[1].ptr());
    Py::SeqBase<Py::Object> path   = args[2];
    PathListGenerator       path_generator(path);
    Py::Object              transforms_obj   = args[3];
    Py::Object              offsets_obj      = args[4];
    agg::trans_affine       offset_trans     = py_to_agg_transformation_matrix(args[5].ptr());
    Py::Object              facecolors_obj   = args[6];
    Py::Object              edgecolors_obj   = args[7];
    Py::SeqBase<Py::Float>  linewidths       = args[8];
    Py::SeqBase<Py::Object> linestyles_obj   = args[9];
    Py::SeqBase<Py::Int>    antialiaseds     = args[10];
    // We don't actually care about urls for Agg, so just ignore it.
    // Py::SeqBase<Py::Object> urls             = args[11];
    std::string             offset_position  = Py::String(args[12]).encode("utf-8");

    bool data_offsets = (offset_position == "data");

    try
    {
        _draw_path_collection_generic<PathListGenerator, 1, 1>
        (gc,
         master_transform,
         gc.cliprect,
         gc.clippath,
         gc.clippath_trans,
         path_generator,
         transforms_obj,
         offsets_obj,
         offset_trans,
         facecolors_obj,
         edgecolors_obj,
         linewidths,
         linestyles_obj,
         antialiaseds,
         data_offsets);
    }
    catch (const char *e)
    {
        throw Py::RuntimeError(e);
    }

    return Py::Object();
}


class QuadMeshGenerator
{
    size_t m_meshWidth;
    size_t m_meshHeight;
    PyArrayObject* m_coordinates;

    class QuadMeshPathIterator
    {
        size_t m_iterator;
        size_t m_m, m_n;
        PyArrayObject* m_coordinates;

    public:
        QuadMeshPathIterator(size_t m, size_t n, PyArrayObject* coordinates) :
            m_iterator(0), m_m(m), m_n(n), m_coordinates(coordinates)
        {

        }

    private:
        inline unsigned
        vertex(unsigned idx, double* x, double* y)
        {
            size_t m = m_m + ((idx     & 0x2) >> 1);
            size_t n = m_n + (((idx + 1) & 0x2) >> 1);
            double* pair = (double*)PyArray_GETPTR2(m_coordinates, n, m);
            *x = *pair++;
            *y = *pair;
            return (idx) ? agg::path_cmd_line_to : agg::path_cmd_move_to;
        }

    public:
        inline unsigned
        vertex(double* x, double* y)
        {
            if (m_iterator >= total_vertices())
            {
                return agg::path_cmd_stop;
            }
            return vertex(m_iterator++, x, y);
        }

        inline void
        rewind(unsigned path_id)
        {
            m_iterator = path_id;
        }

        inline unsigned
        total_vertices()
        {
            return 5;
        }

        inline bool
        should_simplify()
        {
            return false;
        }
    };

public:
    typedef QuadMeshPathIterator path_iterator;

    inline
    QuadMeshGenerator(size_t meshWidth, size_t meshHeight, PyObject* coordinates) :
        m_meshWidth(meshWidth), m_meshHeight(meshHeight), m_coordinates(NULL)
    {
        PyArrayObject* coordinates_array = \
            (PyArrayObject*)PyArray_ContiguousFromObject(
                coordinates, PyArray_DOUBLE, 3, 3);

        if (!coordinates_array)
        {
            throw Py::ValueError("Invalid coordinates array.");
        }

        m_coordinates = coordinates_array;
    }

    inline
    ~QuadMeshGenerator()
    {
        Py_XDECREF(m_coordinates);
    }

    inline size_t
    num_paths() const
    {
        return m_meshWidth * m_meshHeight;
    }

    inline path_iterator
    operator()(size_t i) const
    {
        return QuadMeshPathIterator(i % m_meshWidth, i / m_meshWidth, m_coordinates);
    }
};

Py::Object
RendererAgg::draw_quad_mesh(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_quad_mesh");
    args.verify_length(10);

    //segments, trans, clipbox, colors, linewidths, antialiaseds
    GCAgg gc(args[0], dpi);
    agg::trans_affine master_transform = py_to_agg_transformation_matrix(args[1].ptr());
    size_t            mesh_width       = (long)Py::Int(args[2]);
    size_t            mesh_height      = (long)Py::Int(args[3]);
    Py::Object        coordinates      = args[4];
    Py::Object        offsets_obj      = args[5];
    agg::trans_affine offset_trans     = py_to_agg_transformation_matrix(args[6].ptr());
    Py::Object        facecolors_obj   = args[7];
    bool              antialiased      = (bool)Py::Boolean(args[8]);
    Py::Object        edgecolors_obj   = args[9];

    QuadMeshGenerator path_generator(mesh_width, mesh_height, coordinates.ptr());

    Py::Object transforms_obj = Py::List(0);
    Py::Tuple linewidths(1);
    linewidths[0] = Py::Float(gc.linewidth * 72.0 / dpi);
    Py::SeqBase<Py::Object> linestyles_obj;
    Py::Tuple antialiaseds(1);
    antialiaseds[0] = Py::Int(antialiased ? 1 : 0);

    if (edgecolors_obj.isNone()) {
        if (antialiased)
        {
            edgecolors_obj = facecolors_obj;
        }
        else
        {
            npy_intp dims[] = { 0, 0 };
            edgecolors_obj = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
        }
    }

    try
    {
        _draw_path_collection_generic<QuadMeshGenerator, 0, 0>
            (gc,
             master_transform,
             gc.cliprect,
             gc.clippath,
             gc.clippath_trans,
             path_generator,
             transforms_obj,
             offsets_obj,
             offset_trans,
             facecolors_obj,
             edgecolors_obj,
             linewidths,
             linestyles_obj,
             antialiaseds,
             false);
    }
    catch (const char* e)
    {
        throw Py::RuntimeError(e);
    }

    return Py::Object();
}

void
RendererAgg::_draw_gouraud_triangle(const double* points,
                                    const double* colors,
                                    agg::trans_affine trans,
                                    bool has_clippath)
{
    typedef agg::rgba8                                         color_t;
    typedef agg::span_gouraud_rgba<color_t>                    span_gen_t;
    typedef agg::span_allocator<color_t>                       span_alloc_t;

    trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_translation(0.0, (double)height);

    double tpoints[6];

    for (int i = 0; i < 6; i += 2)
    {
        tpoints[i] = points[i];
        tpoints[i+1] = points[i+1];
        trans.transform(&tpoints[i], &tpoints[i+1]);
    }

    span_alloc_t span_alloc;
    span_gen_t span_gen;

    span_gen.colors(
        agg::rgba(colors[0], colors[1], colors[2], colors[3]),
        agg::rgba(colors[4], colors[5], colors[6], colors[7]),
        agg::rgba(colors[8], colors[9], colors[10], colors[11]));
    span_gen.triangle(
        tpoints[0], tpoints[1],
        tpoints[2], tpoints[3],
        tpoints[4], tpoints[5],
        0.5);

    try {
        theRasterizer.add_path(span_gen);
    } catch (std::overflow_error &e) {
        throw Py::OverflowError(e.what()
                                );
    }

    if (has_clippath)
    {
        typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
        typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
        typedef agg::renderer_scanline_aa<amask_ren_type, span_alloc_t, span_gen_t>
        amask_aa_renderer_type;

        pixfmt_amask_type pfa(pixFmt, alphaMask);
        amask_ren_type r(pfa);
        amask_aa_renderer_type ren(r, span_alloc, span_gen);
        agg::render_scanlines(theRasterizer, scanlineAlphaMask, ren);
    }
    else
    {
        agg::render_scanlines_aa(theRasterizer, slineP8, rendererBase, span_alloc, span_gen);
    }
}


Py::Object
RendererAgg::draw_gouraud_triangle(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_gouraud_triangle");
    args.verify_length(4);

    GCAgg             gc(args[0], dpi);
    Py::Object        points_obj = args[1];
    Py::Object        colors_obj = args[2];
    agg::trans_affine trans      = py_to_agg_transformation_matrix(args[3].ptr());

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    PyArrayObject* points = (PyArrayObject*)PyArray_ContiguousFromAny
        (points_obj.ptr(), PyArray_DOUBLE, 2, 2);
    if (!points ||
        PyArray_DIM(points, 0) != 3 || PyArray_DIM(points, 1) != 2)
    {
        Py_XDECREF(points);
        throw Py::ValueError("points must be a 3x2 numpy array");
    }
    points_obj = Py::Object((PyObject*)points, true);

    PyArrayObject* colors = (PyArrayObject*)PyArray_ContiguousFromAny
        (colors_obj.ptr(), PyArray_DOUBLE, 2, 2);
    if (!colors ||
        PyArray_DIM(colors, 0) != 3 || PyArray_DIM(colors, 1) != 4)
    {
        Py_XDECREF(colors);
        throw Py::ValueError("colors must be a 3x4 numpy array");
    }
    colors_obj = Py::Object((PyObject*)colors, true);

    _draw_gouraud_triangle(
        (double*)PyArray_DATA(points), (double*)PyArray_DATA(colors),
        trans, has_clippath);

    return Py::Object();
}


Py::Object
RendererAgg::draw_gouraud_triangles(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_gouraud_triangles");
    args.verify_length(4);

    typedef agg::rgba8                      color_t;
    typedef agg::span_gouraud_rgba<color_t> span_gen_t;

    GCAgg             gc(args[0], dpi);
    Py::Object        points_obj = args[1];
    Py::Object        colors_obj = args[2];
    agg::trans_affine trans      = py_to_agg_transformation_matrix(args[3].ptr());
    double            c_points[6];
    double            c_colors[12];

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    PyArrayObject* points = (PyArrayObject*)PyArray_FromObject
        (points_obj.ptr(), PyArray_DOUBLE, 3, 3);
    if (!points ||
        PyArray_DIM(points, 1) != 3 || PyArray_DIM(points, 2) != 2)
    {
        Py_XDECREF(points);
        throw Py::ValueError("points must be a Nx3x2 numpy array");
    }
    points_obj = Py::Object((PyObject*)points, true);

    PyArrayObject* colors = (PyArrayObject*)PyArray_FromObject
        (colors_obj.ptr(), PyArray_DOUBLE, 3, 3);
    if (!colors ||
        PyArray_DIM(colors, 1) != 3 || PyArray_DIM(colors, 2) != 4)
    {
        Py_XDECREF(colors);
        throw Py::ValueError("colors must be a Nx3x4 numpy array");
    }
    colors_obj = Py::Object((PyObject*)colors, true);

    if (PyArray_DIM(points, 0) != PyArray_DIM(colors, 0))
    {
        throw Py::ValueError("points and colors arrays must be the same length");
    }

    for (int i = 0; i < PyArray_DIM(points, 0); ++i)
    {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                c_points[j*2+k] = *(double *)PyArray_GETPTR3(points, i, j, k);
            }
        }

        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_colors[j*4+k] = *(double *)PyArray_GETPTR3(colors, i, j, k);
            }
        }

        _draw_gouraud_triangle(
                c_points, c_colors, trans, has_clippath);
    }

    return Py::Object();
}


Py::Object
RendererAgg::write_rgba(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::write_rgba");

    args.verify_length(1);

    FILE *fp = NULL;
    mpl_off_t offset;
    Py::Object py_fileobj = Py::Object(args[0]);
    PyObject* py_file = NULL;
    bool close_file = false;

    if (py_fileobj.isString())
    {
        if ((py_file = mpl_PyFile_OpenFile(py_fileobj.ptr(), (char *)"wb")) == NULL) {
            throw Py::Exception();
        }
    }
    else
    {
        py_file = py_fileobj.ptr();
    }

    if ((fp = mpl_PyFile_Dup(py_file, (char *)"wb", &offset)))
    {
        if (fwrite(pixBuffer, 1, NUMBYTES, fp) != NUMBYTES)
        {
            if (mpl_PyFile_DupClose(py_file, fp, offset)) {
              throw Py::RuntimeError("Error closing dupe file handle");
            }

            if (close_file) {
                mpl_PyFile_CloseFile(py_file);
                Py_DECREF(py_file);
            }

            throw Py::RuntimeError("Error writing to file");
        }

        if (mpl_PyFile_DupClose(py_file, fp, offset)) {
          throw Py::RuntimeError("Error closing dupe file handle");
        }

        if (close_file) {
            mpl_PyFile_CloseFile(py_file);
            Py_DECREF(py_file);
        }
    }
    else
    {
        PyErr_Clear();
        PyObject* write_method = PyObject_GetAttrString(py_fileobj.ptr(),
                                                        "write");
        if (!(write_method && PyCallable_Check(write_method)))
        {
            Py_XDECREF(write_method);
            throw Py::TypeError(
                "Object does not appear to be a 8-bit string path or a Python file-like object");
        }

        #if PY3K
        PyObject_CallFunction(write_method, (char *)"y#", pixBuffer, NUMBYTES);
        #else
        PyObject_CallFunction(write_method, (char *)"s#", pixBuffer, NUMBYTES);
        #endif

        Py_XDECREF(write_method);
    }

    return Py::Object();
}


Py::Object
RendererAgg::tostring_rgb(const Py::Tuple& args)
{
    //"Return the rendered buffer as an RGB string";

    _VERBOSE("RendererAgg::tostring_rgb");

    args.verify_length(0);
    int row_len = width * 3;
    unsigned char* buf_tmp = new unsigned char[row_len * height];
    if (buf_tmp == NULL)
    {
        //todo: also handle allocation throw
        throw Py::MemoryError(
            "RendererAgg::tostring_rgb could not allocate memory");
    }

    try
    {
        agg::rendering_buffer renderingBufferTmp;
        renderingBufferTmp.attach(buf_tmp,
                                  width,
                                  height,
                                  row_len);

        agg::color_conv(&renderingBufferTmp, &renderingBuffer,
                        agg::color_conv_rgba32_to_rgb24());
    }
    catch (...)
    {
        delete [] buf_tmp;
        throw Py::RuntimeError("Unknown exception occurred in tostring_rgb");
    }

    //todo: how to do this with native CXX
    #if PY3K
    PyObject* o = Py_BuildValue("y#", buf_tmp, row_len * height);
    #else
    PyObject* o = Py_BuildValue("s#", buf_tmp, row_len * height);
    #endif

    delete [] buf_tmp;
    return Py::asObject(o);
}


Py::Object
RendererAgg::tostring_argb(const Py::Tuple& args)
{
    //"Return the rendered buffer as an RGB string";

    _VERBOSE("RendererAgg::tostring_argb");

    args.verify_length(0);
    int row_len = width * 4;
    unsigned char* buf_tmp = new unsigned char[row_len * height];
    if (buf_tmp == NULL)
    {
        //todo: also handle allocation throw
        throw Py::MemoryError("RendererAgg::tostring_argb could not allocate memory");
    }

    try
    {
        agg::rendering_buffer renderingBufferTmp;
        renderingBufferTmp.attach(buf_tmp, width, height, row_len);
        agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_argb32());
    }
    catch (...)
    {
        delete [] buf_tmp;
        throw Py::RuntimeError("Unknown exception occurred in tostring_argb");
    }

    //todo: how to do this with native CXX

    #if PY3K
    PyObject* o = Py_BuildValue("y#", buf_tmp, row_len * height);
    #else
    PyObject* o = Py_BuildValue("s#", buf_tmp, row_len * height);
    #endif
    delete [] buf_tmp;
    return Py::asObject(o);
}


Py::Object
RendererAgg::tostring_bgra(const Py::Tuple& args)
{
    //"Return the rendered buffer as an RGB string";

    _VERBOSE("RendererAgg::tostring_bgra");

    args.verify_length(0);
    int row_len = width * 4;
    unsigned char* buf_tmp = new unsigned char[row_len * height];
    if (buf_tmp == NULL)
    {
        //todo: also handle allocation throw
        throw Py::MemoryError("RendererAgg::tostring_bgra could not allocate memory");
    }

    try
    {
        agg::rendering_buffer renderingBufferTmp;
        renderingBufferTmp.attach(buf_tmp,
                                  width,
                                  height,
                                  row_len);

        agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_bgra32());
    }
    catch (...)
    {
        delete [] buf_tmp;
        throw Py::RuntimeError("Unknown exception occurred in tostring_bgra");
    }

    //todo: how to do this with native CXX
    #if PY3K
    PyObject* o = Py_BuildValue("y#", buf_tmp, row_len * height);
    #else
    PyObject* o = Py_BuildValue("s#", buf_tmp, row_len * height);
    #endif
    delete [] buf_tmp;
    return Py::asObject(o);
}


Py::Object
RendererAgg::buffer_rgba(const Py::Tuple& args)
{
    //"expose the rendered buffer as Python buffer object, starting from postion x,y";

    _VERBOSE("RendererAgg::buffer_rgba");

    args.verify_length(0);

    #if PY3K
    return Py::asObject(PyMemoryView_FromObject(this));
    #else
    int row_len = width * 4;
    return Py::asObject(PyBuffer_FromReadWriteMemory(
                            pixBuffer, row_len*height));
    #endif
}


Py::Object
RendererAgg::tostring_rgba_minimized(const Py::Tuple& args)
{
    args.verify_length(0);

    int xmin = width;
    int ymin = height;
    int xmax = 0;
    int ymax = 0;

    // Looks at the alpha channel to find the minimum extents of the image
    unsigned char* pixel = pixBuffer + 3;
    for (int y = 0; y < (int)height; ++y)
    {
        for (int x = 0; x < (int)width; ++x)
        {
            if (*pixel)
            {
                if (x < xmin) xmin = x;
                if (y < ymin) ymin = y;
                if (x > xmax) xmax = x;
                if (y > ymax) ymax = y;
            }
            pixel += 4;
        }
    }

    int newwidth = 0;
    int newheight = 0;
    PyObject *data;

    if (xmin < xmax && ymin < ymax)
    {
        // Expand the bounds by 1 pixel on all sides
        xmin = std::max(0, xmin - 1);
        ymin = std::max(0, ymin - 1);
        xmax = std::min(xmax, (int)width);
        ymax = std::min(ymax, (int)height);

        newwidth    = xmax - xmin;
        newheight   = ymax - ymin;
        int newsize = newwidth * newheight * 4;

        // NULL pointer causes Python to allocate uninitialized memory.
        // We then grab Python's pointer to uninitialized memory using
        // the _AsString() API.
        unsigned int* dst;

        data = PyBytes_FromStringAndSize(NULL, newsize);
        if (data == NULL)
        {
            throw Py::MemoryError("RendererAgg::tostring_rgba_minimized could not allocate memory");
        }
        dst = (unsigned int *)PyBytes_AsString(data);

        unsigned int*  src = (unsigned int*)pixBuffer;
        for (int y = ymin; y < ymax; ++y)
        {
            for (int x = xmin; x < xmax; ++x, ++dst)
            {
                *dst = src[y * width + x];
            }
        }
    } else {
        data = PyBytes_FromStringAndSize(NULL, 0);
        if (data == NULL)
        {
            throw Py::MemoryError("RendererAgg::tostring_rgba_minimized could not allocate memory");
        }
    }

    Py::Tuple bounds(4);
    bounds[0] = Py::Int(xmin);
    bounds[1] = Py::Int(ymin);
    bounds[2] = Py::Int(newwidth);
    bounds[3] = Py::Int(newheight);

    Py::Tuple result(2);
    result[0] = Py::Object(data, true);
    result[1] = bounds;

    return result;
}


Py::Object
RendererAgg::clear(const Py::Tuple& args)
{
    //"clear the rendered buffer";

    _VERBOSE("RendererAgg::clear");

    args.verify_length(0);
    rendererBase.clear(_fill_color);

    return Py::Object();
}


agg::rgba
RendererAgg::rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha)
{
    _VERBOSE("RendererAgg::rgb_to_color");

    double r = Py::Float(rgb[0]);
    double g = Py::Float(rgb[1]);
    double b = Py::Float(rgb[2]);
    return agg::rgba(r, g, b, alpha);
}


double
RendererAgg::points_to_pixels(const Py::Object& points)
{
    _VERBOSE("RendererAgg::points_to_pixels");
    double p = Py::Float(points) ;
    return p * dpi / 72.0;
}

#if PY3K
int
RendererAgg::buffer_get( Py_buffer* buf, int flags )
{
    return PyBuffer_FillInfo(buf, this, pixBuffer, width * height * 4, 1,
                             PyBUF_SIMPLE);
}
#endif

RendererAgg::~RendererAgg()
{

    _VERBOSE("RendererAgg::~RendererAgg");

    delete [] alphaBuffer;
    delete [] pixBuffer;
}

/* ------------ module methods ------------- */
Py::Object _backend_agg_module::new_renderer(const Py::Tuple &args,
        const Py::Dict &kws)
{

    if (args.length() != 3)
    {
        throw Py::RuntimeError("Incorrect # of args to RendererAgg(width, height, dpi).");
    }

    int debug;
    if (kws.hasKey("debug"))
    {
        debug = Py::Int(kws["debug"]);
    }
    else
    {
        debug = 0;
    }

    unsigned int width = (int)Py::Int(args[0]);
    unsigned int height = (int)Py::Int(args[1]);
    double dpi = Py::Float(args[2]);

    if (width > 1 << 15 || height > 1 << 15)
    {
        throw Py::ValueError("width and height must each be below 32768");
    }

    if (dpi <= 0.0)
    {
        throw Py::ValueError("dpi must be positive");
    }

    RendererAgg* renderer = NULL;
    try
    {
        renderer = new RendererAgg(width, height, dpi, debug);
    }
    catch (std::bad_alloc)
    {
        throw Py::RuntimeError("Could not allocate memory for image");
    }

    return Py::asObject(renderer);
}


void BufferRegion::init_type()
{
    behaviors().name("BufferRegion");
    behaviors().doc("A wrapper to pass agg buffer objects to and from the python level");


    add_varargs_method("set_x", &BufferRegion::set_x,
                       "set_x(x)");

    add_varargs_method("set_y", &BufferRegion::set_y,
                       "set_y(y)");

    add_varargs_method("get_extents", &BufferRegion::get_extents,
                       "get_extents()");

    add_varargs_method("to_string", &BufferRegion::to_string,
                       "to_string()");
    add_varargs_method("to_string_argb", &BufferRegion::to_string_argb,
                       "to_string_argb()");
}


void RendererAgg::init_type()
{
    behaviors().name("RendererAgg");
    behaviors().doc("The agg backend extension module");

    add_varargs_method("draw_path", &RendererAgg::draw_path,
                       "draw_path(gc, path, transform, rgbFace)\n");
    add_varargs_method("draw_path_collection", &RendererAgg::draw_path_collection,
                       "draw_path_collection(gc, master_transform, paths, transforms, offsets, offsetTrans, facecolors, edgecolors, linewidths, linestyles, antialiaseds)\n");
    add_varargs_method("draw_quad_mesh", &RendererAgg::draw_quad_mesh,
                       "draw_quad_mesh(gc, master_transform, meshWidth, meshHeight, coordinates, offsets, offsetTrans, facecolors, antialiaseds, showedges)\n");
    add_varargs_method("draw_gouraud_triangle", &RendererAgg::draw_gouraud_triangle,
                       "draw_gouraud_triangle(gc, points, colors, master_transform)\n");
    add_varargs_method("draw_gouraud_triangles", &RendererAgg::draw_gouraud_triangles,
                       "draw_gouraud_triangles(gc, points, colors, master_transform)\n");
    add_varargs_method("draw_markers", &RendererAgg::draw_markers,
                       "draw_markers(gc, marker_path, marker_trans, path, rgbFace)\n");
    add_varargs_method("draw_text_image", &RendererAgg::draw_text_image,
                       "draw_text_image(font_image, x, y, r, g, b, a)\n");
    add_varargs_method("draw_image", &RendererAgg::draw_image,
                       "draw_image(gc, x, y, im)");
    add_varargs_method("write_rgba", &RendererAgg::write_rgba,
                       "write_rgba(fname)");
    add_varargs_method("tostring_rgb", &RendererAgg::tostring_rgb,
                       "s = tostring_rgb()");
    add_varargs_method("tostring_argb", &RendererAgg::tostring_argb,
                       "s = tostring_argb()");
    add_varargs_method("tostring_bgra", &RendererAgg::tostring_bgra,
                       "s = tostring_bgra()");
    add_varargs_method("tostring_rgba_minimized", &RendererAgg::tostring_rgba_minimized,
                       "s = tostring_rgba_minimized()");
    add_varargs_method("buffer_rgba", &RendererAgg::buffer_rgba,
                       "buffer = buffer_rgba()");
    add_varargs_method("clear", &RendererAgg::clear,
                       "clear()");
    add_varargs_method("copy_from_bbox", &RendererAgg::copy_from_bbox,
                       "copy_from_bbox(bbox)");
    add_varargs_method("restore_region", &RendererAgg::restore_region,
                       "restore_region(region)");
    add_varargs_method("restore_region2", &RendererAgg::restore_region2,
                       "restore_region(region, x1, y1, x2, y2, x3, y3)");

    #if PY3K
    behaviors().supportBufferType();
    #endif
}

PyMODINIT_FUNC
#if PY3K
PyInit__backend_agg(void)
#else
init_backend_agg(void)
#endif
{
    //static _backend_agg_module* _backend_agg = new _backend_agg_module;

    _VERBOSE("init_backend_agg");

    import_array();

    static _backend_agg_module* _backend_agg = NULL;
    _backend_agg = new _backend_agg_module;

    #if PY3K
    return _backend_agg->module().ptr();
    #endif
}

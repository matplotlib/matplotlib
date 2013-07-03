/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Python API mandates Python.h is included *first* */
#include "Python.h"
#include <string>

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>

#include "numpy/arrayobject.h"

#include "agg_color_rgba.h"
#include "agg_conv_transform.h"
#include "agg_image_accessors.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_u.h"
#include "agg_span_allocator.h"
#include "agg_span_image_filter_rgb.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_linear.h"
#include "agg_rasterizer_sl_clip.h"
#include "util/agg_color_conv_rgb8.h"
#include "_image.h"
#include "mplutils.h"


typedef agg::pixfmt_rgba32_plain pixfmt;
typedef agg::pixfmt_rgba32_pre pixfmt_pre;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::span_interpolator_linear<> interpolator_type;
typedef agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl> rasterizer;


Image::Image() :
        bufferIn(NULL), rbufIn(NULL), colsIn(0), rowsIn(0),
        bufferOut(NULL), rbufOut(NULL), colsOut(0), rowsOut(0),  BPP(4),
        interpolation(BILINEAR), aspect(ASPECT_FREE), bg(1, 1, 1, 0), resample(true)
{
    _VERBOSE("Image::Image");
}

Image::~Image()
{
    _VERBOSE("Image::~Image");
    delete [] bufferIn;
    bufferIn = NULL;
    delete rbufIn;
    rbufIn = NULL;
    delete rbufOut;
    rbufOut = NULL;
    delete [] bufferOut;
    bufferOut = NULL;
}

int
Image::setattr(const char * name, const Py::Object & value)
{
    _VERBOSE("Image::setattr");
    __dict__[name] = value;
    return 0;
}

Py::Object
Image::getattr(const char * name)
{
    _VERBOSE("Image::getattro");
    if (__dict__.hasKey(name)) return __dict__[name];
    else return getattr_default(name);
}

char Image::apply_rotation__doc__[] =
    "apply_rotation(angle)\n"
    "\n"
    "Apply the rotation (degrees) to image"
    ;
Py::Object
Image::apply_rotation(const Py::Tuple& args)
{
    _VERBOSE("Image::apply_rotation");

    args.verify_length(1);
    double r = Py::Float(args[0]);


    agg::trans_affine M = agg::trans_affine_rotation(r * agg::pi / 180.0);
    srcMatrix *= M;
    imageMatrix *= M;
    return Py::Object();
}

char Image::flipud_out__doc__[] =
    "flipud()\n"
    "\n"
    "Flip the output image upside down"
    ;

char Image::flipud_in__doc__[] =
    "flipud()\n"
    "\n"
    "Flip the input image upside down"
    ;
Py::Object
Image::flipud_in(const Py::Tuple& args)
{
    _VERBOSE("Image::flipud_in");

    args.verify_length(0);
    int stride = rbufIn->stride();
    rbufIn->attach(bufferIn, colsIn, rowsIn, -stride);

    return Py::Object();
}

char Image::set_bg__doc__[] =
    "set_bg(r,g,b,a)\n"
    "\n"
    "Set the background color"
    ;

Py::Object
Image::set_bg(const Py::Tuple& args)
{
    _VERBOSE("Image::set_bg");

    args.verify_length(4);
    bg.r = Py::Float(args[0]);
    bg.g = Py::Float(args[1]);
    bg.b = Py::Float(args[2]);
    bg.a = Py::Float(args[3]);
    return Py::Object();
}

char Image::apply_scaling__doc__[] =
    "apply_scaling(sx, sy)\n"
    "\n"
    "Apply the scale factors sx, sy to the transform matrix"
    ;

Py::Object
Image::apply_scaling(const Py::Tuple& args)
{
    _VERBOSE("Image::apply_scaling");

    args.verify_length(2);
    double sx = Py::Float(args[0]);
    double sy = Py::Float(args[1]);

    //printf("applying scaling %1.2f, %1.2f\n", sx, sy);
    agg::trans_affine M = agg::trans_affine_scaling(sx, sy);
    srcMatrix *= M;
    imageMatrix *= M;

    return Py::Object();
}

char Image::apply_translation__doc__[] =
    "apply_translation(tx, ty)\n"
    "\n"
    "Apply the translation tx, ty to the transform matrix"
    ;

Py::Object
Image::apply_translation(const Py::Tuple& args)
{
    _VERBOSE("Image::apply_translation");

    args.verify_length(2);
    double tx = Py::Float(args[0]);
    double ty = Py::Float(args[1]);

    //printf("applying translation %1.2f, %1.2f\n", tx, ty);
    agg::trans_affine M = agg::trans_affine_translation(tx, ty);
    srcMatrix *= M;
    imageMatrix *= M;

    return Py::Object();
}

char Image::as_rgba_str__doc__[] =
    "numrows, numcols, s = as_rgba_str()"
    "\n"
    "Call this function after resize to get the data as string\n"
    "The string is a numrows by numcols x 4 (RGBA) unsigned char buffer\n"
    ;

Py::Object
Image::as_rgba_str(const Py::Tuple& args, const Py::Dict& kwargs)
{
    _VERBOSE("Image::as_rgba_str");

    args.verify_length(0);

    std::pair<agg::int8u*, bool> bufpair = _get_output_buffer();

    #if PY3K
    Py::Object ret =  Py::asObject(Py_BuildValue("nny#", rowsOut, colsOut,
                                   bufpair.first, colsOut * rowsOut * 4));
    #else
    Py::Object ret =  Py::asObject(Py_BuildValue("nns#", rowsOut, colsOut,
                                   bufpair.first, colsOut * rowsOut * 4));
    #endif

    if (bufpair.second) delete [] bufpair.first;
    return ret;
}


char Image::color_conv__doc__[] =
    "numrows, numcols, buffer = color_conv(format)"
    "\n"
    "format 0(BGRA) or 1(ARGB)\n"
    "Convert image to format and return in a writable buffer\n"
    ;
Py::Object
Image::color_conv(const Py::Tuple& args)
{
    _VERBOSE("Image::color_conv");

    args.verify_length(1);
    int format = Py::Int(args[0]);
    PyObject* py_buffer = NULL;
    int row_len = colsOut * 4;
#if PY3K
    unsigned char* buf = (unsigned char *)malloc(row_len * rowsOut);
    if (buf == NULL)
        throw Py::MemoryError("Image::color_conv could not allocate memory");
#else
    py_buffer = PyBuffer_New(row_len * rowsOut);
    if (py_buffer == NULL)
        throw Py::MemoryError("Image::color_conv could not allocate memory");

    void* buf;
    Py_ssize_t buffer_len;
    int ret = PyObject_AsWriteBuffer(py_buffer, &buf, &buffer_len);
    if (ret != 0)
    {
        Py_XDECREF(py_buffer);
        throw Py::MemoryError("Image::color_conv could not allocate memory");
    }
#endif

    agg::rendering_buffer rtmp;
    rtmp.attach(reinterpret_cast<unsigned char*>(buf), colsOut, rowsOut,
                row_len);

    switch (format)
    {
    case 0:
        agg::color_conv(&rtmp, rbufOut, agg::color_conv_rgba32_to_bgra32());
        break;
    case 1:
        agg::color_conv(&rtmp, rbufOut, agg::color_conv_rgba32_to_argb32());
        break;
    default:
        Py_XDECREF(py_buffer);
        throw Py::ValueError("Image::color_conv unknown format");
    }

#if PY3K
    py_buffer = PyByteArray_FromStringAndSize((char *)buf, row_len * rowsOut);
    if (py_buffer == NULL) {
        free(buf);
    }
#endif

    PyObject* o = Py_BuildValue("nnN", rowsOut, colsOut, py_buffer);
    return Py::asObject(o);
}

char Image::buffer_rgba__doc__[] =
    "buffer = buffer_rgba()"
    "\n"
    "Return the image buffer as rgba32\n"
    ;
Py::Object
Image::buffer_rgba(const Py::Tuple& args)
{
    //"Return the image object as rgba";

    _VERBOSE("RendererAgg::buffer_rgba");

    args.verify_length(0);
    int row_len = colsOut * 4;
    PyObject* o = Py_BuildValue("nns#", rowsOut, colsOut,
                                rbufOut, row_len * rowsOut);
    return Py::asObject(o);
}

char Image::reset_matrix__doc__[] =
    "reset_matrix()"
    "\n"
    "Reset the transformation matrix"
    ;

Py::Object
Image::reset_matrix(const Py::Tuple& args)
{
    _VERBOSE("Image::reset_matrix");

    args.verify_length(0);
    srcMatrix.reset();
    imageMatrix.reset();

    return Py::Object();
}

char Image::get_matrix__doc__[] =
    "(m11,m21,m12,m22,m13,m23) = get_matrix()\n"
    "\n"
    "Get the affine transformation matrix\n"
    "  /m11,m12,m13\\\n"
    "  /m21,m22,m23|\n"
    "  \\ 0 , 0 , 1 /"
    ;

Py::Object
Image::get_matrix(const Py::Tuple& args)
{
    _VERBOSE("Image::get_matrix");

    args.verify_length(0);

    double m[6];
    srcMatrix.store_to(m);
    Py::Tuple ret(6);
    for (int i = 0;i < 6;i++)
    {
        ret[i] = Py::Float(m[i]);
    }
    return ret;
}

char Image::resize__doc__[] =
    "resize(width, height, norm=1, radius=4.0)\n"
    "\n"
    "Resize the image to width, height using interpolation\n"
    "norm and radius are optional args for some of the filters and must be\n"
    "passed as kwargs\n"
    ;

Py::Object
Image::resize(const Py::Tuple& args, const Py::Dict& kwargs)
{
    _VERBOSE("Image::resize");

    args.verify_length(2);

    int norm = 1;
    if (kwargs.hasKey("norm"))
    {
        norm = Py::Int(kwargs["norm"]);
    }

    double radius = 4.0;
    if (kwargs.hasKey("radius"))
    {
        radius = Py::Float(kwargs["radius"]);
    }

    if (bufferIn == NULL)
    {
        throw Py::RuntimeError("You must first load the image");
    }

    int numcols = Py::Int(args[0]);
    int numrows = Py::Int(args[1]);

    colsOut = numcols;
    rowsOut = numrows;

    size_t NUMBYTES(numrows * numcols * BPP);

    delete [] bufferOut;
    bufferOut = new agg::int8u[NUMBYTES];
    if (bufferOut == NULL) //todo: also handle allocation throw
    {
        throw Py::MemoryError("Image::resize could not allocate memory");
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

    //srcMatrix *= resizingMatrix;
    //imageMatrix *= resizingMatrix;
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
    switch (interpolation)
    {

    case NEAREST:
    {
        typedef agg::span_image_filter_rgba_nn<img_accessor_type, interpolator_type> span_gen_type;
        typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type> renderer_type;
        span_gen_type sg(ia, interpolator);
        renderer_type ri(rb, sa, sg);
        agg::render_scanlines(ras, sl, ri);
    }
    break;

    case HANNING:
    case HAMMING:
    case HERMITE:
    {
        agg::image_filter_lut filter;
        switch (interpolation)
        {
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
        if (resample)
        {
            typedef agg::span_image_resample_rgba_affine<img_accessor_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type> renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        }
        else
        {
            typedef agg::span_image_filter_rgba_2x2<img_accessor_type, interpolator_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type> renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        }
    }
    break;
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
    case BLACKMAN:
    {
        agg::image_filter_lut filter;
        switch (interpolation)
        {
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
        if (resample)
        {
            typedef agg::span_image_resample_rgba_affine<img_accessor_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type> renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        }
        else
        {
            typedef agg::span_image_filter_rgba<img_accessor_type, interpolator_type> span_gen_type;
            typedef agg::renderer_scanline_aa<renderer_base, span_alloc_type, span_gen_type> renderer_type;
            span_gen_type sg(ia, interpolator, filter);
            renderer_type ri(rb, sa, sg);
            agg::render_scanlines(ras, sl, ri);
        }
    }
    break;

    }

    return Py::Object();
}



char Image::get_interpolation__doc__[] =
    "get_interpolation()\n"
    "\n"
    "Get the interpolation scheme to one of the module constants, "
    "one of image.NEAREST, image.BILINEAR, etc..."
    ;

Py::Object
Image::get_interpolation(const Py::Tuple& args)
{
    _VERBOSE("Image::get_interpolation");

    args.verify_length(0);
    return Py::Int((int)interpolation);
}


char Image::get_aspect__doc__[] =
    "get_aspect()\n"
    "\n"
    "Get the aspect constraint constants"
    ;

Py::Object
Image::get_aspect(const Py::Tuple& args)
{
    _VERBOSE("Image::get_aspect");

    args.verify_length(0);
    return Py::Int((int)aspect);
}

char Image::get_size__doc__[] =
    "numrows, numcols = get_size()\n"
    "\n"
    "Get the number or rows and columns of the input image"
    ;

Py::Object
Image::get_size(const Py::Tuple& args)
{
    _VERBOSE("Image::get_size");

    args.verify_length(0);

    Py::Tuple ret(2);
    ret[0] = Py::Int((long)rowsIn);
    ret[1] = Py::Int((long)colsIn);
    return ret;

}

char Image::get_resample__doc__[] =
    "get_resample()\n"
    "\n"
    "Get the resample flag."
    ;

Py::Object
Image::get_resample(const Py::Tuple& args)
{
    _VERBOSE("Image::get_resample");

    args.verify_length(0);
    return Py::Int((int)resample);
}

char Image::get_size_out__doc__[] =
    "numrows, numcols = get_size()\n"
    "\n"
    "Get the number or rows and columns of the output image"
    ;

Py::Object
Image::get_size_out(const Py::Tuple& args)
{
    _VERBOSE("Image::get_size_out");

    args.verify_length(0);

    Py::Tuple ret(2);
    ret[0] = Py::Int((long)rowsOut);
    ret[1] = Py::Int((long)colsOut);
    return ret;
}

//get the output buffer, flipped if necessary.  The second element of
//the pair is a bool that indicates whether you need to free the
//memory
std::pair<agg::int8u*, bool>
Image::_get_output_buffer()
{
    _VERBOSE("Image::_get_output_buffer");
    std::pair<agg::int8u*, bool> ret;
    bool flipy = rbufOut->stride() < 0;
    if (flipy)
    {
        agg::int8u* buffer = new agg::int8u[rowsOut*colsOut*4];
        agg::rendering_buffer rb;
        rb.attach(buffer, colsOut, rowsOut, colsOut*4);
        rb.copy_from(*rbufOut);
        ret.first = buffer;
        ret.second = true;
    }
    else
    {
        ret.first = bufferOut;
        ret.second = false;
    }
    return ret;

}

char Image::set_interpolation__doc__[] =
    "set_interpolation(scheme)\n"
    "\n"
    "Set the interpolation scheme to one of the module constants, "
    "eg, image.NEAREST, image.BILINEAR, etc..."
    ;

Py::Object
Image::set_interpolation(const Py::Tuple& args)
{
    _VERBOSE("Image::set_interpolation");

    args.verify_length(1);

    size_t method = (long)Py::Int(args[0]);
    interpolation = (unsigned)method;
    return Py::Object();
}

char Image::set_resample__doc__[] =
    "set_resample(boolean)\n"
    "\n"
    "Set the resample flag."
    ;

Py::Object
Image::set_resample(const Py::Tuple& args)
{
    _VERBOSE("Image::set_resample");
    args.verify_length(1);
    int flag = Py::Int(args[0]);
    resample = (bool)flag;
    return Py::Object();
}


char Image::set_aspect__doc__[] =
    "set_aspect(scheme)\n"
    "\n"
    "Set the aspect ration to one of the image module constant."
    "eg, one of image.ASPECT_PRESERVE, image.ASPECT_FREE"
    ;
Py::Object
Image::set_aspect(const Py::Tuple& args)
{
    _VERBOSE("Image::set_aspect");

    args.verify_length(1);
    size_t method = (long)Py::Int(args[0]);
    aspect = (unsigned)method;
    return Py::Object();

}

void
Image::init_type()
{
    _VERBOSE("Image::init_type");

    behaviors().name("Image");
    behaviors().doc("Image");
    behaviors().supportGetattr();
    behaviors().supportSetattr();

    add_varargs_method("apply_rotation", &Image::apply_rotation, Image::apply_rotation__doc__);
    add_varargs_method("apply_scaling",  &Image::apply_scaling, Image::apply_scaling__doc__);
    add_varargs_method("apply_translation", &Image::apply_translation, Image::apply_translation__doc__);
    add_keyword_method("as_rgba_str", &Image::as_rgba_str, Image::as_rgba_str__doc__);
    add_varargs_method("color_conv", &Image::color_conv, Image::color_conv__doc__);
    add_varargs_method("buffer_rgba", &Image::buffer_rgba, Image::buffer_rgba__doc__);
    add_varargs_method("get_aspect", &Image::get_aspect, Image::get_aspect__doc__);
    add_varargs_method("get_interpolation", &Image::get_interpolation, Image::get_interpolation__doc__);
    add_varargs_method("get_resample", &Image::get_resample, Image::get_resample__doc__);
    add_varargs_method("get_size", &Image::get_size, Image::get_size__doc__);
    add_varargs_method("get_size_out", &Image::get_size_out, Image::get_size_out__doc__);
    add_varargs_method("reset_matrix", &Image::reset_matrix, Image::reset_matrix__doc__);
    add_varargs_method("get_matrix", &Image::get_matrix, Image::get_matrix__doc__);
    add_keyword_method("resize", &Image::resize, Image::resize__doc__);
    add_varargs_method("set_interpolation", &Image::set_interpolation, Image::set_interpolation__doc__);
    add_varargs_method("set_resample", &Image::set_resample, Image::set_resample__doc__);
    add_varargs_method("set_aspect", &Image::set_aspect, Image::set_aspect__doc__);
    add_varargs_method("set_bg", &Image::set_bg, Image::set_bg__doc__);
    add_varargs_method("flipud_out", &Image::flipud_out, Image::flipud_out__doc__);
    add_varargs_method("flipud_in", &Image::flipud_in, Image::flipud_in__doc__);
}




char _image_module_from_images__doc__[] =
    "from_images(numrows, numcols, seq)\n"
    "\n"
    "return an image instance with numrows, numcols from a seq of image\n"
    "instances using alpha blending.  seq is a list of (Image, ox, oy)"
    ;
Py::Object
_image_module::from_images(const Py::Tuple& args)
{
    _VERBOSE("_image_module::from_images");

    args.verify_length(3);

    size_t numrows = (long)Py::Int(args[0]);
    size_t numcols = (long)Py::Int(args[1]);

    if (numrows >= 32768 || numcols >= 32768)
    {
        throw Py::RuntimeError("numrows and numcols must both be less than 32768");
    }

    Py::SeqBase<Py::Object> tups = args[2];
    size_t N = tups.length();

    if (N == 0)
    {
        throw Py::RuntimeError("Empty list of images");
    }

    Py::Tuple tup;

    size_t ox(0), oy(0), thisx(0), thisy(0);
    float alpha;
    bool apply_alpha;

    //copy image 0 output buffer into return images output buffer
    Image* imo = new Image;
    imo->rowsOut  = numrows;
    imo->colsOut  = numcols;

    size_t NUMBYTES(numrows * numcols * imo->BPP);
    imo->bufferOut = new agg::int8u[NUMBYTES];
    if (imo->bufferOut == NULL) //todo: also handle allocation throw
    {
        throw Py::MemoryError("_image_module::from_images could not allocate memory");
    }

    delete imo->rbufOut;
    imo->rbufOut = new agg::rendering_buffer;
    imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);

    pixfmt pixf(*imo->rbufOut);
    renderer_base rb(pixf);

    rb.clear(agg::rgba(0, 0, 0, 0));
    for (size_t imnum = 0; imnum < N; imnum++)
    {
        tup = Py::Tuple(tups[imnum]);
        Image* thisim = static_cast<Image*>(tup[0].ptr());
        ox = (long)Py::Int(tup[1]);
        oy = (long)Py::Int(tup[2]);
        if (tup.size() <= 3 || tup[3].ptr() == Py_None)
        {
            apply_alpha = false;
        }
        else
        {
            apply_alpha = true;
            alpha = Py::Float(tup[3]);
        }

        bool isflip = (thisim->rbufOut->stride()) < 0;
        //std::cout << "from images " << isflip << "; stride=" << thisim->rbufOut->stride() << std::endl;
        size_t ind = 0;
        for (size_t j = 0; j < thisim->rowsOut; j++)
        {
            for (size_t i = 0; i < thisim->colsOut; i++)
            {
                thisx = i + ox;

                if (isflip)
                {
                    thisy = thisim->rowsOut - j + oy;
                }
                else
                {
                    thisy = j + oy;
                }

                if (thisx >= numcols || thisy >= numrows)
                {
                    ind += 4;
                    continue;
                }

                pixfmt::color_type p;
                p.r = *(thisim->bufferOut + ind++);
                p.g = *(thisim->bufferOut + ind++);
                p.b = *(thisim->bufferOut + ind++);
                if (apply_alpha)
                {
                    p.a = (pixfmt::value_type) *(thisim->bufferOut + ind++) * alpha;
                }
                else
                {
                    p.a = *(thisim->bufferOut + ind++);
                }
                pixf.blend_pixel(thisx, thisy, p, 255);
            }
        }
    }

    return Py::asObject(imo);
}


char _image_module_fromarray__doc__[] =
    "fromarray(A, isoutput)\n"
    "\n"
    "Load the image from a numpy array\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling"
    ;
Py::Object
_image_module::fromarray(const Py::Tuple& args)
{
    _VERBOSE("_image_module::fromarray");

    args.verify_length(2);

    Py::Object x = args[0];
    int isoutput = Py::Int(args[1]);
    PyArrayObject *A = (PyArrayObject *) PyArray_FromObject(x.ptr(), PyArray_DOUBLE, 2, 3);
    if (A == NULL)
    {
        throw Py::ValueError("Array must be rank 2 or 3 of doubles");
    }
    Py::Object A_obj((PyObject *)A, true);

    Image* imo = new Image;

    imo->rowsIn  = A->dimensions[0];
    imo->colsIn  = A->dimensions[1];

    size_t NUMBYTES(imo->colsIn * imo->rowsIn * imo->BPP);
    agg::int8u *buffer = new agg::int8u[NUMBYTES];
    if (buffer == NULL) //todo: also handle allocation throw
    {
        throw Py::MemoryError("_image_module::fromarray could not allocate memory");
    }

    if (isoutput)
    {
        // make the output buffer point to the input buffer
        imo->rowsOut  = imo->rowsIn;
        imo->colsOut  = imo->colsIn;

        imo->rbufOut = new agg::rendering_buffer;
        imo->bufferOut = buffer;
        imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);
    }
    else
    {
        imo->bufferIn = buffer;
        imo->rbufIn = new agg::rendering_buffer;
        imo->rbufIn->attach(buffer, imo->colsIn, imo->rowsIn, imo->colsIn*imo->BPP);
    }

    if (A->nd == 2)     //assume luminance for now;
    {
        agg::int8u gray;
        for (size_t rownum = 0; rownum < imo->rowsIn; rownum++)
        {
            for (size_t colnum = 0; colnum < imo->colsIn; colnum++)
            {
                double val = *(double *)(A->data + rownum * A->strides[0] + colnum * A->strides[1]);

                gray = int(255 * val);
                *buffer++ = gray;       // red
                *buffer++ = gray;       // green
                *buffer++ = gray;       // blue
                *buffer++   = 255;        // alpha
            }
        }
    }
    else if (A->nd == 3)     // assume RGB
    {

        if (A->dimensions[2] != 3 && A->dimensions[2] != 4)
        {
            throw Py::ValueError(Printf("3rd dimension must be length 3 (RGB) or 4 (RGBA); found %d", A->dimensions[2]).str());
        }

        int rgba = A->dimensions[2] == 4;
        double r, g, b, alpha;
        size_t offset = 0;

        for (size_t rownum = 0; rownum < imo->rowsIn; rownum++)
        {
            for (size_t colnum = 0; colnum < imo->colsIn; colnum++)
            {
                offset = rownum * A->strides[0] + colnum * A->strides[1];
                r = *(double *)(A->data + offset);
                g = *(double *)(A->data + offset + A->strides[2]);
                b = *(double *)(A->data + offset + 2 * A->strides[2]);

                if (rgba)
                {
                    alpha = *(double *)(A->data + offset + 3 * A->strides[2]);
                }
                else
                {
                    alpha = 1.0;
                }

                *buffer++ = int(255 * r);       // red
                *buffer++ = int(255 * g);       // green
                *buffer++ = int(255 * b);       // blue
                *buffer++ = int(255 * alpha);   // alpha
            }
        }
    }
    else     // error
    {
        throw Py::ValueError("Illegal array rank; must be rank; must 2 or 3");
    }
    buffer -= NUMBYTES;

    return Py::asObject(imo);
}

char _image_module_fromarray2__doc__[] =
    "fromarray2(A, isoutput)\n"
    "\n"
    "Load the image from a numpy array\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling"
    ;
Py::Object
_image_module::fromarray2(const Py::Tuple& args)
{
    _VERBOSE("_image_module::fromarray2");

    args.verify_length(2);

    Py::Object x = args[0];
    int isoutput = Py::Int(args[1]);
    PyArrayObject *A = (PyArrayObject *) PyArray_ContiguousFromObject(x.ptr(), PyArray_DOUBLE, 2, 3);
    if (A == NULL)
    {
        throw Py::ValueError("Array must be rank 2 or 3 of doubles");
    }
    Py::Object A_obj((PyObject*)A, true);

    Image* imo = new Image;

    imo->rowsIn  = A->dimensions[0];
    imo->colsIn  = A->dimensions[1];

    size_t NUMBYTES(imo->colsIn * imo->rowsIn * imo->BPP);
    agg::int8u *buffer = new agg::int8u[NUMBYTES];
    if (buffer == NULL) //todo: also handle allocation throw
    {
        throw Py::MemoryError("_image_module::fromarray could not allocate memory");
    }

    if (isoutput)
    {
        // make the output buffer point to the input buffer
        imo->rowsOut  = imo->rowsIn;
        imo->colsOut  = imo->colsIn;

        imo->rbufOut = new agg::rendering_buffer;
        imo->bufferOut = buffer;
        imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);
    }
    else
    {
        imo->bufferIn = buffer;
        imo->rbufIn = new agg::rendering_buffer;
        imo->rbufIn->attach(buffer, imo->colsIn, imo->rowsIn, imo->colsIn*imo->BPP);
    }

    if (A->nd == 2)     //assume luminance for now;
    {
        agg::int8u gray;
        const size_t N = imo->rowsIn * imo->colsIn;
        size_t i = 0;
        while (i++ < N)
        {
            double val = *(double *)(A->data++);

            gray = int(255 * val);
            *buffer++ = gray;       // red
            *buffer++ = gray;       // green
            *buffer++ = gray;       // blue
            *buffer++   = 255;        // alpha
        }

    }
    else if (A->nd == 3)     // assume RGB
    {
        if (A->dimensions[2] != 3 && A->dimensions[2] != 4)
        {
            throw Py::ValueError(Printf("3rd dimension must be length 3 (RGB) or 4 (RGBA); found %d", A->dimensions[2]).str());

        }

        int rgba = A->dimensions[2] == 4;
        double r, g, b, alpha;
        const size_t N = imo->rowsIn * imo->colsIn;
        for (size_t i = 0; i < N; ++i)
        {
            r = *(double *)(A->data++);
            g = *(double *)(A->data++);
            b = *(double *)(A->data++);

            if (rgba)
                alpha = *(double *)(A->data++);
            else
                alpha = 1.0;

            *buffer++ = int(255 * r);       // red
            *buffer++ = int(255 * g);       // green
            *buffer++ = int(255 * b);       // blue
            *buffer++ = int(255 * alpha);   // alpha

        }
    }
    else     // error
    {
        throw Py::ValueError("Illegal array rank; must be rank; must 2 or 3");
    }
    buffer -= NUMBYTES;

    return Py::asObject(imo);
}

char _image_module_frombyte__doc__[] =
    "frombyte(A, isoutput)\n"
    "\n"
    "Load the image from a byte array.\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling."
    ;
Py::Object
_image_module::frombyte(const Py::Tuple& args)
{
    _VERBOSE("_image_module::frombyte");

    args.verify_length(2);

    Py::Object x = args[0];
    int isoutput = Py::Int(args[1]);

    PyArrayObject *A = (PyArrayObject *) PyArray_FromObject(x.ptr(), PyArray_UBYTE, 3, 3);
    if (A == NULL)
    {
        throw Py::ValueError("Array must have 3 dimensions");
    }
    Py::Object A_obj((PyObject*)A, true);

    if (A->dimensions[2] < 3 || A->dimensions[2] > 4)
    {
        throw Py::ValueError("Array dimension 3 must have size 3 or 4");
    }

    Image* imo = new Image;

    imo->rowsIn = A->dimensions[0];
    imo->colsIn = A->dimensions[1];

    agg::int8u *arrbuf;
    agg::int8u *buffer;
    agg::int8u *dstbuf;

    arrbuf = reinterpret_cast<agg::int8u *>(A->data);

    size_t NUMBYTES(imo->colsIn * imo->rowsIn * imo->BPP);
    buffer = dstbuf = new agg::int8u[NUMBYTES];

    if (buffer == NULL) //todo: also handle allocation throw
    {
        throw Py::MemoryError("_image_module::frombyte could not allocate memory");
    }

    if PyArray_ISCONTIGUOUS(A)
    {
        if (A->dimensions[2] == 4)
        {
            memmove(dstbuf, arrbuf, imo->rowsIn * imo->colsIn * 4);
        }
        else
        {
            size_t i = imo->rowsIn * imo->colsIn;
            while (i--)
            {
                *dstbuf++ = *arrbuf++;
                *dstbuf++ = *arrbuf++;
                *dstbuf++ = *arrbuf++;
                *dstbuf++ = 255;
            }
        }
    }
    else if ((A->strides[1] == 4) && (A->strides[2] == 1))
    {
        const size_t N = imo->colsIn * 4;
        const size_t stride = A->strides[0];
        for (size_t rownum = 0; rownum < imo->rowsIn; rownum++)
        {
            memmove(dstbuf, arrbuf, N);
            arrbuf += stride;
            dstbuf += N;
        }
    }
    else if ((A->strides[1] == 3) && (A->strides[2] == 1))
    {
        const size_t stride = A->strides[0] - imo->colsIn * 3;
        for (size_t rownum = 0; rownum < imo->rowsIn; rownum++)
        {
            for (size_t colnum = 0; colnum < imo->colsIn; colnum++)
            {
                *dstbuf++ = *arrbuf++;
                *dstbuf++ = *arrbuf++;
                *dstbuf++ = *arrbuf++;
                *dstbuf++ = 255;
            }
            arrbuf += stride;
        }
    }
    else
    {
        PyArrayIterObject *iter;
        iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)A);
        if (A->dimensions[2] == 4)
        {
            while (iter->index < iter->size) {
                *dstbuf++ = *((unsigned char *)iter->dataptr);
                PyArray_ITER_NEXT(iter);
            }
        }
        else
        {
            while (iter->index < iter->size) {
                *dstbuf++ = *((unsigned char *)iter->dataptr);
                PyArray_ITER_NEXT(iter);
                *dstbuf++ = *((unsigned char *)iter->dataptr);
                PyArray_ITER_NEXT(iter);
                *dstbuf++ = *((unsigned char *)iter->dataptr);
                PyArray_ITER_NEXT(iter);
                *dstbuf++ = 255;
            }
        }
        Py_DECREF(iter);
    }

    if (isoutput)
    {
        // make the output buffer point to the input buffer

        imo->rowsOut  = imo->rowsIn;
        imo->colsOut  = imo->colsIn;

        imo->rbufOut = new agg::rendering_buffer;
        imo->bufferOut = buffer;
        imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);

    }
    else
    {
        imo->bufferIn = buffer;
        imo->rbufIn = new agg::rendering_buffer;
        imo->rbufIn->attach(buffer, imo->colsIn, imo->rowsIn, imo->colsIn*imo->BPP);
    }

    return Py::asObject(imo);
}

char _image_module_frombuffer__doc__[] =
    "frombuffer(buffer, width, height, isoutput)\n"
    "\n"
    "Load the image from a character buffer\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling."
    ;
Py::Object
_image_module::frombuffer(const Py::Tuple& args)
{
    _VERBOSE("_image_module::frombuffer");

    args.verify_length(4);

    PyObject *bufin = new_reference_to(args[0]);
    size_t x = (long)Py::Int(args[1]);
    size_t y = (long)Py::Int(args[2]);

    if (x >= 32768 || y >= 32768)
    {
        throw Py::ValueError("x and y must both be less than 32768");
    }

    int isoutput = Py::Int(args[3]);

    if (PyObject_CheckReadBuffer(bufin) != 1)
        throw Py::ValueError("First argument must be a buffer.");

    Image* imo = new Image;

    imo->rowsIn = y;
    imo->colsIn = x;
    Py_ssize_t NUMBYTES(imo->colsIn * imo->rowsIn * imo->BPP);

    Py_ssize_t buflen;
    const agg::int8u *rawbuf;
    if (PyObject_AsReadBuffer(bufin, reinterpret_cast<const void**>(&rawbuf), &buflen) != 0)
    {
        throw Py::ValueError("Cannot get buffer from object.");
    }

    // Check buffer is required size.
    if (buflen != NUMBYTES)
    {
        throw Py::ValueError("Buffer length must be width * height * 4.");
    }

    // Copy from input buffer to new buffer for agg.
    agg::int8u* buffer = new agg::int8u[NUMBYTES];
    if (buffer == NULL) //todo: also handle allocation throw
    {
        throw Py::MemoryError("_image_module::frombuffer could not allocate memory");
    }
    memmove(buffer, rawbuf, NUMBYTES);

    if (isoutput)
    {
        // make the output buffer point to the input buffer
        imo->rowsOut  = imo->rowsIn;
        imo->colsOut  = imo->colsIn;

        imo->rbufOut = new agg::rendering_buffer;
        imo->bufferOut = buffer;
        imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);

    }
    else
    {
        imo->bufferIn = buffer;
        imo->rbufIn = new agg::rendering_buffer;
        imo->rbufIn->attach(buffer, imo->colsIn, imo->rowsIn, imo->colsIn*imo->BPP);
    }

    return Py::asObject(imo);
}

// utilities for irregular grids
void _bin_indices_middle(unsigned int *irows, int nrows, float *ys1, int ny, float dy, float y_min)
{
    int  i, j, j_last;
    unsigned  int * rowstart = irows;
    float *ys2 = ys1 + 1;
    float *yl = ys1 + ny ;
    float yo = y_min + dy / 2.0;
    float ym = 0.5f * (*ys1 + *ys2);
    // y/rows
    j = 0;
    j_last = j;
    for (i = 0;i < nrows;i++, yo += dy, rowstart++)
    {
        while (ys2 != yl && yo > ym)
        {
            ys1 = ys2;
            ys2 = ys1 + 1;
            ym = 0.5f * (*ys1 + *ys2);
            j++;
        }
        *rowstart = j - j_last;
        j_last = j;
    }
}

void _bin_indices_middle_linear(float *arows, unsigned int *irows, int nrows, float *y, int ny, float dy, float y_min)
{
    int i;
    int ii = 0;
    int iilast = ny - 1;
    float sc = 1 / dy;
    int iy0 = (int)floor(sc * (y[ii]  - y_min));
    int iy1 = (int)floor(sc * (y[ii+1]  - y_min));
    float invgap = 1.0f / (iy1 - iy0);
    for (i = 0; i < nrows && i <= iy0; i++)
    {
        irows[i] = 0;
        arows[i] = 1.0;
        //std::cerr<<"i="<<i<<"  ii="<<0<<" a="<< arows[i]<< std::endl;
    }
    for (; i < nrows; i++)
    {
        while (i > iy1 && ii < iilast)
        {
            ii++;
            iy0 = iy1;
            iy1 = (int)floor(sc * (y[ii+1] - y_min));
            invgap = 1.0f / (iy1 - iy0);
        }
        if (i >= iy0 && i <= iy1)
        {
            irows[i] = ii;
            arows[i] = (iy1 - i) * invgap;
            //std::cerr<<"i="<<i<<"  ii="<<ii<<" a="<< arows[i]<< std::endl;
        }
        else break;
    }
    for (; i < nrows; i++)
    {
        irows[i] = iilast - 1;
        arows[i] = 0.0;
        //std::cerr<<"i="<<i<<"  ii="<<iilast-1<<" a="<< arows[i]<< std::endl;
    }
}

void _bin_indices(int *irows, int nrows, double *y, int ny,
                  double sc, double offs)
{
    int i;
    if (sc*(y[ny-1] - y[0]) > 0)
    {
        int ii = 0;
        int iilast = ny - 1;
        int iy0 = (int)floor(sc * (y[ii]  - offs));
        int iy1 = (int)floor(sc * (y[ii+1]  - offs));
        for (i = 0; i < nrows && i < iy0; i++)
        {
            irows[i] = -1;
        }
        for (; i < nrows; i++)
        {
            while (i > iy1 && ii < iilast)
            {
                ii++;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii+1] - offs));
            }
            if (i >= iy0 && i <= iy1) irows[i] = ii;
            else break;
        }
        for (; i < nrows; i++)
        {
            irows[i] = -1;
        }
    }
    else
    {
        int iilast = ny - 1;
        int ii = iilast;
        int iy0 = (int)floor(sc * (y[ii]  - offs));
        int iy1 = (int)floor(sc * (y[ii-1]  - offs));
        for (i = 0; i < nrows && i < iy0; i++)
        {
            irows[i] = -1;
        }
        for (; i < nrows; i++)
        {
            while (i > iy1 && ii > 1)
            {
                ii--;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii-1] - offs));
            }
            if (i >= iy0 && i <= iy1) irows[i] = ii - 1;
            else break;
        }
        for (; i < nrows; i++)
        {
            irows[i] = -1;
        }
    }
}

void _bin_indices_linear(float *arows, int *irows, int nrows, double *y, int ny,
                         double sc, double offs)
{
    int i;
    if (sc*(y[ny-1] - y[0]) > 0)
    {
        int ii = 0;
        int iilast = ny - 1;
        int iy0 = (int)floor(sc * (y[ii]  - offs));
        int iy1 = (int)floor(sc * (y[ii+1]  - offs));
        float invgap = 1.0 / (iy1 - iy0);
        for (i = 0; i < nrows && i < iy0; i++)
        {
            irows[i] = -1;
        }
        for (; i < nrows; i++)
        {
            while (i > iy1 && ii < iilast)
            {
                ii++;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii+1] - offs));
                invgap = 1.0 / (iy1 - iy0);
            }
            if (i >= iy0 && i <= iy1)
            {
                irows[i] = ii;
                arows[i] = (iy1 - i) * invgap;
            }
            else break;
        }
        for (; i < nrows; i++)
        {
            irows[i] = -1;
        }
    }
    else
    {
        int iilast = ny - 1;
        int ii = iilast;
        int iy0 = (int)floor(sc * (y[ii]  - offs));
        int iy1 = (int)floor(sc * (y[ii-1]  - offs));
        float invgap = 1.0 / (iy1 - iy0);
        for (i = 0; i < nrows && i < iy0; i++)
        {
            irows[i] = -1;
        }
        for (; i < nrows; i++)
        {
            while (i > iy1 && ii > 1)
            {
                ii--;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii-1] - offs));
                invgap = 1.0 / (iy1 - iy0);
            }
            if (i >= iy0 && i <= iy1)
            {
                irows[i] = ii - 1;
                arows[i] = (i - iy0) * invgap;
            }
            else break;
        }
        for (; i < nrows; i++)
        {
            irows[i] = -1;
        }
    }
}



char __image_module_pcolor__doc__[] =
    "pcolor(x, y, data, rows, cols, bounds)\n"
    "\n"
    "Generate a pseudo-color image from data on a non-uniform grid using\n"
    "nearest neighbour or linear interpolation.\n"
    "bounds = (x_min, x_max, y_min, y_max)\n"
    "interpolation = NEAREST or BILINEAR \n"
    ;

void _pcolor_cleanup(PyArrayObject* x, PyArrayObject* y,  PyArrayObject *d,
                     unsigned int * rowstarts , unsigned int*colstarts ,
                     float *acols , float *arows)
{
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(d);
    if (rowstarts)
    {
        PyMem_Free(rowstarts);
    }
    if (colstarts)
    {
        PyMem_Free(colstarts);
    }
    if (acols)
    {
        PyMem_Free(acols);
    }
    if (arows)
    {
        PyMem_Free(arows);
    }
    return;
}

Py::Object
_image_module::pcolor(const Py::Tuple& args)
{
    _VERBOSE("_image_module::pcolor");


    if (args.length() != 7)
    {
        throw Py::TypeError("Incorrect number of arguments (7 expected)");
    }

    Py::Object xp = args[0];
    Py::Object yp = args[1];
    Py::Object dp = args[2];
    unsigned int rows = (unsigned long)Py::Int(args[3]);
    unsigned int cols = (unsigned long)Py::Int(args[4]);
    Py::Tuple bounds = args[5];
    unsigned int interpolation = (unsigned long)Py::Int(args[6]);

    if (rows >= 32768 || cols >= 32768)
    {
        throw Py::ValueError("rows and cols must both be less than 32768");
    }

    if (bounds.length() != 4)
    {
        throw Py::TypeError("Incorrect number of bounds (4 expected)");
    }

    float x_min = Py::Float(bounds[0]);
    float x_max = Py::Float(bounds[1]);
    float y_min = Py::Float(bounds[2]);
    float y_max = Py::Float(bounds[3]);
    float width = x_max - x_min;
    float height = y_max - y_min;
    float dx = width / ((float) cols);
    float dy = height / ((float) rows);

    // Check we have something to output to
    if (rows == 0 || cols == 0)
    {
        throw Py::ValueError("Cannot scale to zero size");
    }

    PyArrayObject *x = NULL;
    PyArrayObject *y = NULL;
    PyArrayObject *d = NULL;
    unsigned int *rowstarts = NULL;
    unsigned int *colstarts = NULL;
    float *acols = NULL;
    float *arows = NULL;

    // Get numpy arrays
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xp.ptr(), PyArray_FLOAT, 1, 1);
    if (x == NULL)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::ValueError("x is of incorrect type (wanted 1D float)");
    }
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yp.ptr(), PyArray_FLOAT, 1, 1);
    if (y == NULL)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::ValueError("y is of incorrect type (wanted 1D float)");
    }
    d = (PyArrayObject *) PyArray_ContiguousFromObject(dp.ptr(), PyArray_UBYTE, 3, 3);
    if (d == NULL)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::ValueError("data is of incorrect type (wanted 3D UInt8)");
    }
    if (d->dimensions[2] != 4)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::ValueError("data must be in RGBA format");
    }

    // Check dimensions match
    int nx = x->dimensions[0];
    int ny = y->dimensions[0];
    if (nx != d->dimensions[1] || ny != d->dimensions[0])
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::ValueError("data and axis dimensions do not match");
    }

    // Allocate memory for pointer arrays
    rowstarts = reinterpret_cast<unsigned int*>(PyMem_Malloc(sizeof(unsigned int) * rows));
    if (rowstarts == NULL)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::MemoryError("Cannot allocate memory for lookup table");
    }
    colstarts = reinterpret_cast<unsigned int*>(PyMem_Malloc(sizeof(unsigned int) * cols));
    if (colstarts == NULL)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::MemoryError("Cannot allocate memory for lookup table");
    }

    // Create output
    Image* imo = new Image;
    imo->rowsIn = rows;
    imo->colsIn = cols;
    imo->rowsOut = rows;
    imo->colsOut = cols;
    size_t NUMBYTES(rows * cols * 4);
    agg::int8u *buffer = new agg::int8u[NUMBYTES];
    if (buffer == NULL)
    {
        _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
        throw Py::MemoryError("Could not allocate memory for image");
    }


    // Calculate the pointer arrays to map input x to output x
    unsigned int i, j;
    unsigned int * colstart = colstarts;
    unsigned int * rowstart = rowstarts;
    float *xs1 = reinterpret_cast<float*>(x->data);
    float *ys1 = reinterpret_cast<float*>(y->data);


    // Copy data to output buffer
    unsigned char *start;
    unsigned char *inposition;
    size_t inrowsize(nx*4);
    size_t rowsize(cols*4);
    rowstart = rowstarts;
    agg::int8u * position = buffer;
    agg::int8u * oldposition = NULL;
    start = reinterpret_cast<unsigned char*>(d->data);
    int s0 = d->strides[0];
    int s1 = d->strides[1];

    if (interpolation == Image::NEAREST)
    {
        _bin_indices_middle(colstart, cols, xs1,  nx, dx, x_min);
        _bin_indices_middle(rowstart, rows, ys1,  ny, dy, y_min);
        for (i = 0;i < rows;i++, rowstart++)
        {
            if (i > 0 && *rowstart == 0)
            {
                memcpy(position, oldposition, rowsize*sizeof(agg::int8u));
                oldposition = position;
                position += rowsize;
            }
            else
            {
                oldposition = position;
                start += *rowstart * inrowsize;
                inposition = start;
                for (j = 0, colstart = colstarts;j < cols;j++, position += 4, colstart++)
                {
                    inposition += *colstart * 4;
                    memcpy(position, inposition, 4*sizeof(agg::int8u));
                }
            }
        }
    }
    else if (interpolation == Image::BILINEAR)
    {
        arows = reinterpret_cast<float *>(PyMem_Malloc(sizeof(float) * rows));
        if (arows == NULL)
        {
            _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
            throw Py::MemoryError("Cannot allocate memory for lookup table");
        }
        acols = reinterpret_cast<float*>(PyMem_Malloc(sizeof(float) * cols));
        if (acols == NULL)
        {
            _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);
            throw Py::MemoryError("Cannot allocate memory for lookup table");
        }

        _bin_indices_middle_linear(acols, colstart, cols, xs1,  nx, dx, x_min);
        _bin_indices_middle_linear(arows, rowstart, rows, ys1,  ny, dy, y_min);
        double a00, a01, a10, a11, alpha, beta;


        agg::int8u * start00;
        agg::int8u * start01;
        agg::int8u * start10;
        agg::int8u * start11;
        // Copy data to output buffer
        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                alpha = arows[i];
                beta = acols[j];

                a00 = alpha * beta;
                a01 = alpha * (1.0 - beta);
                a10 = (1.0 - alpha) * beta;
                a11 = 1.0 - a00 - a01 - a10;

                start00 = (agg::int8u *)(start + s0 * rowstart[i] + s1 * colstart[j]);
                start01 = start00 + s1;
                start10 = start00 + s0;
                start11 = start10 + s1;
                position[0] = (agg::int8u)(start00[0] * a00 + start01[0] * a01 + start10[0] * a10 + start11[0] * a11);
                position[1] = (agg::int8u)(start00[1] * a00 + start01[1] * a01 + start10[1] * a10 + start11[1] * a11);
                position[2] = (agg::int8u)(start00[2] * a00 + start01[2] * a01 + start10[2] * a10 + start11[2] * a11);
                position[3] = (agg::int8u)(start00[3] * a00 + start01[3] * a01 + start10[3] * a10 + start11[3] * a11);
                position += 4;
            }
        }

    }

    // Attach output buffer to output buffer
    imo->rbufOut = new agg::rendering_buffer;
    imo->bufferOut = buffer;
    imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);

    _pcolor_cleanup(x, y, d, rowstarts, colstarts, acols, arows);

    return Py::asObject(imo);

}

void _pcolor2_cleanup(PyArrayObject* x, PyArrayObject* y, PyArrayObject *d,
                      PyArrayObject* bg, int *irows, int*jcols)
{
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(d);
    Py_XDECREF(bg);
    if (irows)
    {
        PyMem_Free(irows);
    }
    if (jcols)
    {
        PyMem_Free(jcols);
    }
}


char __image_module_pcolor2__doc__[] =
    "pcolor2(x, y, data, rows, cols, bounds, bg)\n"
    "\n"
    "Generate a pseudo-color image from data on a non-uniform grid\n"
    "specified by its cell boundaries.\n"
    "bounds = (x_left, x_right, y_bot, y_top)\n"
    "bg = ndarray of 4 uint8 representing background rgba\n"
    ;
Py::Object
_image_module::pcolor2(const Py::Tuple& args)
{
    _VERBOSE("_image_module::pcolor2");

    if (args.length() != 7)
    {
        throw Py::TypeError("Incorrect number of arguments (6 expected)");
    }

    Py::Object xp = args[0];
    Py::Object yp = args[1];
    Py::Object dp = args[2];
    int rows = Py::Int(args[3]);
    int cols = Py::Int(args[4]);
    Py::Tuple bounds = args[5];
    Py::Object bgp = args[6];

    if (rows >= 32768 || cols >= 32768)
    {
        throw Py::ValueError("rows and cols must both be less than 32768");
    }

    if (bounds.length() != 4)
    {
        throw Py::TypeError("Incorrect number of bounds (4 expected)");
    }

    double x_left = Py::Float(bounds[0]);
    double x_right = Py::Float(bounds[1]);
    double y_bot = Py::Float(bounds[2]);
    double y_top = Py::Float(bounds[3]);

    // Check we have something to output to
    if (rows == 0 || cols == 0)
    {
        throw Py::ValueError("rows or cols is zero; there are no pixels");
    }

    PyArrayObject* x = NULL;
    PyArrayObject* y = NULL;
    PyArrayObject* d = NULL;
    PyArrayObject* bg = NULL;
    int* irows = NULL;
    int* jcols = NULL;

    // Get numpy arrays
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xp.ptr(), PyArray_DOUBLE, 1, 1);
    if (x == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("x is of incorrect type (wanted 1D double)");
    }
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yp.ptr(), PyArray_DOUBLE, 1, 1);
    if (y == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("y is of incorrect type (wanted 1D double)");
    }
    d = (PyArrayObject *) PyArray_ContiguousFromObject(dp.ptr(), PyArray_UBYTE, 3, 3);
    if (d == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("data is of incorrect type (wanted 3D uint8)");
    }
    if (d->dimensions[2] != 4)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("data must be in RGBA format");
    }

    // Check dimensions match
    int nx = x->dimensions[0];
    int ny = y->dimensions[0];
    if (nx != d->dimensions[1] + 1 || ny != d->dimensions[0] + 1)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("data and axis bin boundary dimensions are incompatible");
    }

    bg = (PyArrayObject *) PyArray_ContiguousFromObject(bgp.ptr(), PyArray_UBYTE, 1, 1);
    if (bg == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("bg is of incorrect type (wanted 1D uint8)");
    }
    if (bg->dimensions[0] != 4)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::ValueError("bg must be in RGBA format");
    }

    // Allocate memory for pointer arrays
    irows = reinterpret_cast<int*>(PyMem_Malloc(sizeof(int) * rows));
    if (irows == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::MemoryError("Cannot allocate memory for lookup table");
    }
    jcols = reinterpret_cast<int*>(PyMem_Malloc(sizeof(int) * cols));
    if (jcols == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::MemoryError("Cannot allocate memory for lookup table");
    }

    // Create output
    Image* imo = new Image;
    imo->rowsIn = rows;
    imo->rowsOut = rows;
    imo->colsIn = cols;
    imo->colsOut = cols;
    size_t NUMBYTES(rows * cols * 4);
    agg::int8u *buffer = new agg::int8u[NUMBYTES];
    if (buffer == NULL)
    {
        _pcolor2_cleanup(x, y, d, bg, irows, jcols);
        throw Py::MemoryError("Could not allocate memory for image");
    }

    // Calculate the pointer arrays to map input x to output x
    int i, j;
    double *x0 = reinterpret_cast<double*>(x->data);
    double *y0 = reinterpret_cast<double*>(y->data);
    double sx = cols / (x_right - x_left);
    double sy = rows / (y_top - y_bot);
    _bin_indices(jcols, cols, x0, nx, sx, x_left);
    _bin_indices(irows, rows, y0, ny, sy, y_bot);

    // Copy data to output buffer
    agg::int8u * position = buffer;
    unsigned char *start = reinterpret_cast<unsigned char*>(d->data);
    unsigned char *bgptr = reinterpret_cast<unsigned char*>(bg->data);
    int s0 = d->strides[0];
    int s1 = d->strides[1];

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            if (irows[i] == -1 || jcols[j] == -1)
            {
                memcpy(position, bgptr, 4*sizeof(agg::int8u));
            }
            else
            {
                memcpy(position, (start + s0*irows[i] + s1*jcols[j]),
                       4*sizeof(agg::int8u));
            }
            position += 4;
        }
    }

    // Attach output buffer to output buffer
    imo->rbufOut = new agg::rendering_buffer;
    imo->bufferOut = buffer;
    imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);

    _pcolor2_cleanup(x, y, d, bg, irows, jcols);

    return Py::asObject(imo);
}

#if PY3K
PyMODINIT_FUNC
PyInit__image(void)
#else
PyMODINIT_FUNC
init_image(void)
#endif
{
    _VERBOSE("init_image");

    static _image_module* _image = new _image_module;

    import_array();
    Py::Dict d = _image->moduleDictionary();

    d["NEAREST"] = Py::Int(Image::NEAREST);
    d["BILINEAR"] = Py::Int(Image::BILINEAR);
    d["BICUBIC"] = Py::Int(Image::BICUBIC);
    d["SPLINE16"] = Py::Int(Image::SPLINE16);
    d["SPLINE36"] = Py::Int(Image::SPLINE36);
    d["HANNING"] = Py::Int(Image::HANNING);
    d["HAMMING"] = Py::Int(Image::HAMMING);
    d["HERMITE"] = Py::Int(Image::HERMITE);
    d["KAISER"]   = Py::Int(Image::KAISER);
    d["QUADRIC"]   = Py::Int(Image::QUADRIC);
    d["CATROM"]  = Py::Int(Image::CATROM);
    d["GAUSSIAN"]  = Py::Int(Image::GAUSSIAN);
    d["BESSEL"]  = Py::Int(Image::BESSEL);
    d["MITCHELL"]  = Py::Int(Image::MITCHELL);
    d["SINC"]  = Py::Int(Image::SINC);
    d["LANCZOS"]  = Py::Int(Image::LANCZOS);
    d["BLACKMAN"] = Py::Int(Image::BLACKMAN);

    d["ASPECT_FREE"] = Py::Int(Image::ASPECT_FREE);
    d["ASPECT_PRESERVE"] = Py::Int(Image::ASPECT_PRESERVE);

#if PY3K
    return _image->module().ptr();
#endif
}

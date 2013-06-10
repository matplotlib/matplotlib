/* -*- mode: c++; c-basic-offset: 4 -*- */

#include "ft2font.h"
#include "mplutils.h"
#include <sstream>

#include "numpy/arrayobject.h"

/*
 By definition, FT_FIXED as 2 16bit values stored in a single long.
 We cast to long to ensure the correct Py::Int convertor is called
 */
#define FIXED_MAJOR(val) (long) ((val & 0xffff000) >> 16)
#define FIXED_MINOR(val) (long) (val & 0xffff)

/**
 To improve the hinting of the fonts, this code uses a hack
 presented here:

 http://antigrain.com/research/font_rasterization/index.html

 The idea is to limit the effect of hinting in the x-direction, while
 preserving hinting in the y-direction.  Since freetype does not
 support this directly, the dpi in the x-direction is set higher than
 in the y-direction, which affects the hinting grid.  Then, a global
 transform is placed on the font to shrink it back to the desired
 size.  While it is a bit surprising that the dpi setting affects
 hinting, whereas the global transform does not, this is documented
 behavior of freetype, and therefore hopefully unlikely to change.
 The freetype 2 tutorial says:

      NOTE: The transformation is applied to every glyph that is
      loaded through FT_Load_Glyph and is completely independent of
      any hinting process. This means that you won't get the same
      results if you load a glyph at the size of 24 pixels, or a glyph
      at the size at 12 pixels scaled by 2 through a transform,
      because the hints will have been computed differently (except
      you have disabled hints).
 */

FT_Library _ft2Library;

FT2Image::FT2Image(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds) :
    Py::PythonClass< FT2Image >(self, args, kwds),
    _isDirty(true),
    _buffer(NULL),
    _width(0), _height(0)
{
    _VERBOSE("FT2Image::FT2Image");

    args.verify_length(2);
    int width = Py::Int(args[0]);
    int height = Py::Int(args[1]);

    resize(width, height);
}

FT2Image::~FT2Image() {
    delete [] _buffer;
    _buffer = NULL;
}

Py::PythonClassObject<FT2Image> FT2Image::factory(int width, int height)
{
    Py::Callable class_type(type());
    Py::Tuple args(2);
    args[0] = Py::Int(width);
    args[1] = Py::Int(height);
    Py::PythonClassObject<FT2Image> o = Py::PythonClassObject<FT2Image>(
        class_type.apply(args, Py::Dict()));
    return o;
}

void
FT2Image::resize(long width, long height)
{
    if (width < 0)
    {
        width = 1;
    }
    if (height < 0)
    {
        height = 1;
    }
    size_t numBytes = width * height;

    if ((unsigned long)width != _width || (unsigned long)height != _height)
    {
        if (numBytes > _width*_height)
        {
            delete [] _buffer;
            _buffer = NULL;
            _buffer = new unsigned char [numBytes];
        }

        _width = (unsigned long)width;
        _height = (unsigned long)height;
    }

    memset(_buffer, 0, numBytes);

    _isDirty = true;
}

void
FT2Image::draw_bitmap(FT_Bitmap*  bitmap,
                      FT_Int      x,
                      FT_Int      y)
{
    _VERBOSE("FT2Image::draw_bitmap");
    FT_Int image_width = (FT_Int)_width;
    FT_Int image_height = (FT_Int)_height;
    FT_Int char_width =  bitmap->width;
    FT_Int char_height = bitmap->rows;

    FT_Int x1 = CLAMP(x, 0, image_width);
    FT_Int y1 = CLAMP(y, 0, image_height);
    FT_Int x2 = CLAMP(x + char_width, 0, image_width);
    FT_Int y2 = CLAMP(y + char_height, 0, image_height);

    FT_Int x_start = MAX(0, -x);
    FT_Int y_offset = y1 - MAX(0, -y);

    if (bitmap->pixel_mode == FT_PIXEL_MODE_GRAY) {
        for (FT_Int i = y1; i < y2; ++i)
        {
            unsigned char* dst = _buffer + (i * image_width + x1);
            unsigned char* src = bitmap->buffer + (((i - y_offset) * bitmap->pitch) + x_start);
            for (FT_Int j = x1; j < x2; ++j, ++dst, ++src)
                *dst |= *src;
        }
    } else if (bitmap->pixel_mode == FT_PIXEL_MODE_MONO) {
        for (FT_Int i = y1; i < y2; ++i)
        {
            unsigned char* dst = _buffer + (i * image_width + x1);
            unsigned char* src = bitmap->buffer + ((i - y_offset) * bitmap->pitch);
            for (FT_Int j = x1; j < x2; ++j, ++dst) {
                int x = (j - x1 + x_start);
                int val = *(src + (x >> 3)) & (1 << (7 - (x & 0x7)));
                *dst = val ? 255 : *dst;
            }
        }
    } else {
        throw Py::Exception("Unknown pixel mode");
    }

    _isDirty = true;
}

void
FT2Image::write_bitmap(const char* filename) const
{
    FILE *fh = fopen(filename, "w");

    for (size_t i = 0; i < _height; i++)
    {
        for (size_t j = 0; j < _width; ++j)
        {
            if (_buffer[j + i*_width])
            {
                fputc('#', fh);
            }
            else
            {
                fputc(' ', fh);
            }
        }
        fputc('\n', fh);
    }

    fclose(fh);
}

char FT2Image::write_bitmap__doc__[] =
    "write_bitmap(fname)\n"
    "\n"
    "Write the bitmap to file fname\n"
    ;
Py::Object
FT2Image::py_write_bitmap(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::write_bitmap");

    args.verify_length(1);

    std::string filename = Py::String(args[0]);

    write_bitmap(filename.c_str());

    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_write_bitmap)

void
FT2Image::draw_rect(unsigned long x0, unsigned long y0,
                    unsigned long x1, unsigned long y1)
{
    if (x0 > _width || x1 > _width ||
        y0 > _height || y1 > _height)
    {
        throw Py::ValueError("Rect coords outside image bounds");
    }

    size_t top = y0 * _width;
    size_t bottom = y1 * _width;
    for (size_t i = x0; i < x1 + 1; ++i)
    {
        _buffer[i + top] = 255;
        _buffer[i + bottom] = 255;
    }

    for (size_t j = y0 + 1; j < y1; ++j)
    {
        _buffer[x0 + j*_width] = 255;
        _buffer[x1 + j*_width] = 255;
    }

    _isDirty = true;
}

char FT2Image::draw_rect__doc__[] =
    "draw_rect(x0, y0, x1, y1)\n"
    "\n"
    "Draw a rect to the image.\n"
    "\n"
    ;
Py::Object
FT2Image::py_draw_rect(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::draw_rect");

    args.verify_length(4);

    long x0 = Py::Int(args[0]);
    long y0 = Py::Int(args[1]);
    long x1 = Py::Int(args[2]);
    long y1 = Py::Int(args[3]);

    draw_rect(x0, y0, x1, y1);

    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_draw_rect)

void
FT2Image::draw_rect_filled(unsigned long x0, unsigned long y0,
                           unsigned long x1, unsigned long y1)
{
    x0 = std::min(x0, _width);
    y0 = std::min(y0, _height);
    x1 = std::min(x1, _width);
    y1 = std::min(y1, _height);

    for (size_t j = y0; j < y1 + 1; j++)
    {
        for (size_t i = x0; i < x1 + 1; i++)
        {
            _buffer[i + j*_width] = 255;
        }
    }

    _isDirty = true;
}

char FT2Image::draw_rect_filled__doc__[] =
    "draw_rect_filled(x0, y0, x1, y1)\n"
    "\n"
    "Draw a filled rect to the image.\n"
    "\n"
    ;
Py::Object
FT2Image::py_draw_rect_filled(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::draw_rect_filled");

    args.verify_length(4);

    long x0 = Py::Int(args[0]);
    long y0 = Py::Int(args[1]);
    long x1 = Py::Int(args[2]);
    long y1 = Py::Int(args[3]);

    draw_rect_filled(x0, y0, x1, y1);

    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_draw_rect_filled)

char FT2Image::as_str__doc__[] =
    "s = image.as_str()\n"
    "\n"
    "Return the image buffer as a string\n"
    "\n"
    ;
Py::Object
FT2Image::py_as_str(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::as_str");
    args.verify_length(0);

    return Py::asObject
      (PyBytes_FromStringAndSize((const char *)_buffer, _width*_height));
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_as_str)

char FT2Image::as_rgba_str__doc__[] =
    "s = image.as_rgba_str()\n"
    "\n"
    "Return the image buffer as a RGBA string\n"
    "\n"
    ;
Py::Object
FT2Image::py_as_rgba_str(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::as_str");
    args.verify_length(0);

    Py_ssize_t size = _width*_height*4;
    PyObject* result = PyBytes_FromStringAndSize(NULL, size);

    unsigned char *src     = _buffer;
    unsigned char *src_end = src + (_width * _height);
    unsigned char *dst     = (unsigned char *)PyBytes_AS_STRING(result);

    while (src != src_end)
    {
        *dst++ = 0;
        *dst++ = 0;
        *dst++ = 0;
        *dst++ = *src++;
    }

    return Py::asObject(result);
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_as_rgba_str)

/* TODO: This could take a color as an argument, but for
   now it defaults to black on white background */
char FT2Image::as_rgb_str__doc__[] =
    "s = image.as_rgb_str()\n"
    "\n"
    "Return the image buffer as a RGB string\n"
    "\n"
    ;
Py::Object
FT2Image::py_as_rgb_str(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::as_str");
    args.verify_length(0);

    Py_ssize_t size = _width*_height*3;
    PyObject* result = PyBytes_FromStringAndSize(NULL, size);

    unsigned char *src     = _buffer;
    unsigned char *src_end = src + (_width * _height);
    unsigned char *dst     = (unsigned char *)PyBytes_AS_STRING(result);

    while (src != src_end)
    {
        unsigned char tmp = 255 - *src++;
        *dst++ = tmp;
        *dst++ = tmp;
        *dst++ = tmp;
    }

    return Py::asObject(result);
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_as_rgb_str)

char FT2Image::as_array__doc__[] =
    "x = image.as_array()\n"
    "\n"
    "Return the image buffer as a width x height numpy array of ubyte \n"
    "\n"
    ;
Py::Object
FT2Image::py_as_array(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::as_array");
    args.verify_length(0);

    npy_intp dimensions[2];
    dimensions[0] = get_height();  //numrows
    dimensions[1] = get_width();   //numcols


    PyArrayObject *A = (PyArrayObject *) PyArray_SimpleNewFromData(2, dimensions, PyArray_UBYTE, _buffer);

    return Py::asObject((PyObject*)A);
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_as_array)

Py::Object
FT2Image::py_get_width(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::get_width");
    args.verify_length(0);

    return Py::Int((long)get_width());
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_get_width)

Py::Object
FT2Image::py_get_height(const Py::Tuple & args)
{
    _VERBOSE("FT2Image::get_height");
    args.verify_length(0);

    return Py::Int((long)get_height());
}
PYCXX_VARARGS_METHOD_DECL(FT2Image, py_get_height)

Py::PythonClassObject<Glyph> Glyph::factory(
        const FT_Face& face, const FT_Glyph& glyph, size_t ind, long hinting_factor)
{
    Py::Callable class_type(type());
    Py::PythonClassObject<Glyph> obj = Py::PythonClassObject<Glyph>(
        class_type.apply(Py::Tuple(), Py::Dict()));
    Glyph* o = obj.getCxxObject();

    o->glyphInd = ind;
    FT_BBox bbox;
    FT_Glyph_Get_CBox(glyph, ft_glyph_bbox_subpixels, &bbox);

    o->setattro("width",        Py::Int(face->glyph->metrics.width / hinting_factor));
    o->setattro("height",       Py::Int(face->glyph->metrics.height));
    o->setattro("horiBearingX", Py::Int(face->glyph->metrics.horiBearingX / hinting_factor));
    o->setattro("horiBearingY", Py::Int(face->glyph->metrics.horiBearingY));
    o->setattro("horiAdvance",  Py::Int(face->glyph->metrics.horiAdvance));
    o->setattro("linearHoriAdvance",  Py::Int(face->glyph->linearHoriAdvance / hinting_factor));
    o->setattro("vertBearingX", Py::Int(face->glyph->metrics.vertBearingX));

    o->setattro("vertBearingY", Py::Int(face->glyph->metrics.vertBearingY));
    o->setattro("vertAdvance",  Py::Int(face->glyph->metrics.vertAdvance));

    Py::Tuple abbox(4);

    abbox[0] = Py::Int(bbox.xMin);
    abbox[1] = Py::Int(bbox.yMin);
    abbox[2] = Py::Int(bbox.xMax);
    abbox[3] = Py::Int(bbox.yMax);
    o->setattro("bbox", abbox);

    return obj;
}

Glyph::~Glyph()
{
    _VERBOSE("Glyph::~Glyph");
}

int
Glyph::setattro(const Py::String &name, const Py::Object &value)
{
    _VERBOSE("Glyph::setattr");
    __dict__[name] = value;
    return 0;
}

Py::Object
Glyph::getattro(const Py::String &name)
{
    _VERBOSE("Glyph::getattr");
    if (__dict__.hasKey(name)) return __dict__[name];
    else return genericGetAttro(name);
}

inline double conv(int v)
{
    return double(v) / 64.0;
}


char FT2Font::get_path__doc__[] =
    "get_path()\n"
    "\n"
    "Get the path data from the currently loaded glyph as a tuple of vertices, codes.\n"
    ;
Py::Object
FT2Font::get_path()
{
    //get the glyph as a path, a list of (COMMAND, *args) as desribed in matplotlib.path
    // this code is from agg's decompose_ft_outline with minor modifications

    if (!face->glyph) {
        throw Py::ValueError("No glyph loaded");
    }

    enum {STOP = 0,
          MOVETO = 1,
          LINETO = 2,
          CURVE3 = 3,
          CURVE4 = 4,
          ENDPOLY = 0x4f};
    FT_Outline& outline = face->glyph->outline;
    bool flip_y = false; //todo, pass me as kwarg

    FT_Vector   v_last;
    FT_Vector   v_control;
    FT_Vector   v_start;

    FT_Vector*  point;
    FT_Vector*  limit;
    char*       tags;

    int   n;         // index of contour in outline
    int   first;     // index of first point in contour
    char  tag;       // current point's state
    int   count;

    count = 0;
    first = 0;
    for (n = 0; n < outline.n_contours; n++)
    {
        int  last;  // index of last point in contour
        bool starts_with_last;

        last  = outline.contours[n];
        limit = outline.points + last;

        v_start = outline.points[first];
        v_last  = outline.points[last];

        v_control = v_start;

        point = outline.points + first;
        tags  = outline.tags  + first;
        tag   = FT_CURVE_TAG(tags[0]);

        // A contour cannot start with a cubic control point!
        if (tag == FT_CURVE_TAG_CUBIC)
        {
            throw Py::RuntimeError("A contour cannot start with a cubic control point");
        }
        else if (tag == FT_CURVE_TAG_CONIC)
        {
            starts_with_last = true;
        } else {
            starts_with_last = false;
        }

        count++;

        while (point < limit)
        {
            if (!starts_with_last) {
                point++;
                tags++;
            }
            starts_with_last = false;

            tag = FT_CURVE_TAG(tags[0]);
            switch (tag)
            {
            case FT_CURVE_TAG_ON:  // emit a single line_to
            {
                count++;
                continue;
            }

            case FT_CURVE_TAG_CONIC:  // consume conic arcs
            {
            Count_Do_Conic:
                if (point < limit)
                {
                    point++;
                    tags++;
                    tag = FT_CURVE_TAG(tags[0]);

                    if (tag == FT_CURVE_TAG_ON)
                    {
                        count += 2;
                        continue;
                    }

                    if (tag != FT_CURVE_TAG_CONIC)
                    {
                        throw Py::RuntimeError("Invalid font");
                    }

                    count += 2;

                    goto Count_Do_Conic;
                }

                count += 2;

                goto Count_Close;
            }

            default:  // FT_CURVE_TAG_CUBIC
            {
                if (point + 1 > limit || FT_CURVE_TAG(tags[1]) != FT_CURVE_TAG_CUBIC)
                {
                    throw Py::RuntimeError("Invalid font");
                }

                point += 2;
                tags  += 2;

                if (point <= limit)
                {
                    count += 3;
                    continue;
                }

                count += 3;

                goto Count_Close;
            }
            }
        }

        count++;

    Count_Close:
        first = last + 1;
    }

    PyArrayObject* vertices = NULL;
    PyArrayObject* codes = NULL;
    Py::Tuple result(2);

    npy_intp vertices_dims[2] = {count, 2};
    vertices = (PyArrayObject*)PyArray_SimpleNew(
        2, vertices_dims, PyArray_DOUBLE);
    if (vertices == NULL) {
        throw;
    }
    npy_intp codes_dims[1] = {count};
    codes = (PyArrayObject*)PyArray_SimpleNew(
        1, codes_dims, PyArray_UINT8);
    if (codes == NULL) {
        throw;
    }

    result[0] = Py::Object((PyObject*)vertices, true);
    result[1] = Py::Object((PyObject*)codes, true);

    double* outpoints = (double *)PyArray_DATA(vertices);
    unsigned char* outcodes = (unsigned char *)PyArray_DATA(codes);

    first = 0;
    for (n = 0; n < outline.n_contours; n++)
    {
        int last;  // index of last point in contour
        bool starts_with_last;

        last  = outline.contours[n];
        limit = outline.points + last;

        v_start = outline.points[first];
        v_last  = outline.points[last];

        v_control = v_start;

        point = outline.points + first;
        tags  = outline.tags  + first;
        tag   = FT_CURVE_TAG(tags[0]);

        double x, y;
        if (tag != FT_CURVE_TAG_ON)
        {
            x = conv(v_last.x);
            y = flip_y ? -conv(v_last.y) : conv(v_last.y);
            starts_with_last = true;
        } else {
            x = conv(v_start.x);
            y = flip_y ? -conv(v_start.y) : conv(v_start.y);
            starts_with_last = false;
        }

        *(outpoints++) = x;
        *(outpoints++) = y;
        *(outcodes++) = MOVETO;

        while (point < limit)
        {
            if (!starts_with_last) {
                point++;
                tags++;
            }
            starts_with_last = false;

            tag = FT_CURVE_TAG(tags[0]);
            switch (tag)
            {
            case FT_CURVE_TAG_ON:  // emit a single line_to
            {
                double x = conv(point->x);
                double y = flip_y ? -conv(point->y) : conv(point->y);
                *(outpoints++) = x;
                *(outpoints++) = y;
                *(outcodes++) = LINETO;
                continue;
            }

            case FT_CURVE_TAG_CONIC:  // consume conic arcs
            {
                v_control.x = point->x;
                v_control.y = point->y;

            Do_Conic:
                if (point < limit)
                {
                    FT_Vector vec;
                    FT_Vector v_middle;

                    point++;
                    tags++;
                    tag = FT_CURVE_TAG(tags[0]);

                    vec.x = point->x;
                    vec.y = point->y;

                    if (tag == FT_CURVE_TAG_ON)
                    {
                        double xctl = conv(v_control.x);
                        double yctl = flip_y ? -conv(v_control.y) : conv(v_control.y);
                        double xto = conv(vec.x);
                        double yto = flip_y ? -conv(vec.y) : conv(vec.y);
                        *(outpoints++) = xctl;
                        *(outpoints++) = yctl;
                        *(outpoints++) = xto;
                        *(outpoints++) = yto;
                        *(outcodes++) = CURVE3;
                        *(outcodes++) = CURVE3;
                        continue;
                    }

                    v_middle.x = (v_control.x + vec.x) / 2;
                    v_middle.y = (v_control.y + vec.y) / 2;

                    double xctl = conv(v_control.x);
                    double yctl = flip_y ? -conv(v_control.y) : conv(v_control.y);
                    double xto = conv(v_middle.x);
                    double yto = flip_y ? -conv(v_middle.y) : conv(v_middle.y);
                    *(outpoints++) = xctl;
                    *(outpoints++) = yctl;
                    *(outpoints++) = xto;
                    *(outpoints++) = yto;
                    *(outcodes++) = CURVE3;
                    *(outcodes++) = CURVE3;

                    v_control = vec;
                    goto Do_Conic;
                }
                double xctl = conv(v_control.x);
                double yctl = flip_y ? -conv(v_control.y) : conv(v_control.y);
                double xto = conv(v_start.x);
                double yto = flip_y ? -conv(v_start.y) : conv(v_start.y);

                *(outpoints++) = xctl;
                *(outpoints++) = yctl;
                *(outpoints++) = xto;
                *(outpoints++) = yto;
                *(outcodes++) = CURVE3;
                *(outcodes++) = CURVE3;

                goto Close;
            }

            default:  // FT_CURVE_TAG_CUBIC
            {
                FT_Vector vec1, vec2;

                vec1.x = point[0].x;
                vec1.y = point[0].y;
                vec2.x = point[1].x;
                vec2.y = point[1].y;

                point += 2;
                tags  += 2;

                if (point <= limit)
                {
                    FT_Vector vec;

                    vec.x = point->x;
                    vec.y = point->y;

                    double xctl1 = conv(vec1.x);
                    double yctl1 = flip_y ? -conv(vec1.y) : conv(vec1.y);
                    double xctl2 = conv(vec2.x);
                    double yctl2 = flip_y ? -conv(vec2.y) : conv(vec2.y);
                    double xto = conv(vec.x);
                    double yto = flip_y ? -conv(vec.y) : conv(vec.y);

                    (*outpoints++) = xctl1;
                    (*outpoints++) = yctl1;
                    (*outpoints++) = xctl2;
                    (*outpoints++) = yctl2;
                    (*outpoints++) = xto;
                    (*outpoints++) = yto;
                    (*outcodes++) = CURVE4;
                    (*outcodes++) = CURVE4;
                    (*outcodes++) = CURVE4;
                    continue;
                }

                double xctl1 = conv(vec1.x);
                double yctl1 = flip_y ? -conv(vec1.y) : conv(vec1.y);
                double xctl2 = conv(vec2.x);
                double yctl2 = flip_y ? -conv(vec2.y) : conv(vec2.y);
                double xto = conv(v_start.x);
                double yto = flip_y ? -conv(v_start.y) : conv(v_start.y);
                (*outpoints++) = xctl1;
                (*outpoints++) = yctl1;
                (*outpoints++) = xctl2;
                (*outpoints++) = yctl2;
                (*outpoints++) = xto;
                (*outpoints++) = yto;
                (*outcodes++) = CURVE4;
                (*outcodes++) = CURVE4;
                (*outcodes++) = CURVE4;

                goto Close;
            }
            }
        }

        (*outpoints++) = 0.0;
        (*outpoints++) = 0.0;
        (*outcodes++) = ENDPOLY;

    Close:
        first = last + 1;
    }

    if (outcodes - (unsigned char *)PyArray_DATA(codes) != count) {
        throw Py::RuntimeError("Font path size doesn't match");
    }

    return result;
}
PYCXX_NOARGS_METHOD_DECL(FT2Font, get_path)

FT2Font::FT2Font(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds) :
    Py::PythonClass<FT2Font>(self, args, kwds),
    image()
{
    args.verify_length(1);
    std::string facefile = Py::String(args[0]);

    _VERBOSE(Printf("FT2Font::FT2Font %s", facefile.c_str()).str());

    clear(Py::Tuple(0));

    int error = FT_New_Face(_ft2Library, facefile.c_str(), 0, &face);

    if (error == FT_Err_Unknown_File_Format)
    {
        std::ostringstream s;
        s << "Could not load facefile " << facefile << "; Unknown_File_Format" << std::endl;
        throw Py::RuntimeError(s.str());
    }
    else if (error == FT_Err_Cannot_Open_Resource)
    {
        std::ostringstream s;
        s << "Could not open facefile " << facefile << "; Cannot_Open_Resource" << std::endl;
        throw Py::RuntimeError(s.str());
    }
    else if (error == FT_Err_Invalid_File_Format)
    {
        std::ostringstream s;
        s << "Could not open facefile " << facefile << "; Invalid_File_Format" << std::endl;
        throw Py::RuntimeError(s.str());
    }
    else if (error)
    {
        std::ostringstream s;
        s << "Could not open facefile " << facefile << "; freetype error code " << error << std::endl;
        throw Py::RuntimeError(s.str());
    }

    // set a default fontsize 12 pt at 72dpi
    hinting_factor = 8;
    if (kwds.hasKey("hinting_factor"))
    {
        hinting_factor = Py::Long(kwds["hinting_factor"]);
    }

    error = FT_Set_Char_Size(face, 12 * 64, 0, 72 * hinting_factor, 72);
    static FT_Matrix transform = { 65536 / hinting_factor, 0, 0, 65536 };
    FT_Set_Transform(face, &transform, 0);
    if (error)
    {
        std::ostringstream s;
        s << "Could not set the fontsize for facefile  " << facefile << std::endl;
        throw Py::RuntimeError(s.str());
    }

    // set some face props as attributes
    //small memory leak fixed after 2.1.8
    //fields can be null so we have to check this first

    const char* ps_name = FT_Get_Postscript_Name(face);
    if (ps_name == NULL)
    {
        ps_name = "UNAVAILABLE";
    }

    const char* family_name = face->family_name;
    if (family_name == NULL)
    {
        family_name = "UNAVAILABLE";
    }

    const char* style_name = face->style_name;
    if (style_name == NULL)
    {
        style_name = "UNAVAILABLE";
    }

    setattro("postscript_name", Py::String(ps_name));
    setattro("num_faces",       Py::Int(face->num_faces));
    setattro("family_name",     Py::String(family_name));
    setattro("style_name",      Py::String(style_name));
    setattro("face_flags",      Py::Int(face->face_flags));
    setattro("style_flags",     Py::Int(face->style_flags));
    setattro("num_glyphs",      Py::Int(face->num_glyphs));
    setattro("num_fixed_sizes", Py::Int(face->num_fixed_sizes));
    setattro("num_charmaps",    Py::Int(face->num_charmaps));

    int scalable = FT_IS_SCALABLE(face);

    setattro("scalable", Py::Int(scalable));

    if (scalable)
    {
        setattro("units_per_EM", Py::Int(face->units_per_EM));

        Py::Tuple bbox(4);
        bbox[0] = Py::Int(face->bbox.xMin);
        bbox[1] = Py::Int(face->bbox.yMin);
        bbox[2] = Py::Int(face->bbox.xMax);
        bbox[3] = Py::Int(face->bbox.yMax);
        setattro("bbox",  bbox);
        setattro("ascender",            Py::Int(face->ascender));
        setattro("descender",           Py::Int(face->descender));
        setattro("height",              Py::Int(face->height));
        setattro("max_advance_width",   Py::Int(face->max_advance_width));
        setattro("max_advance_height",  Py::Int(face->max_advance_height));
        setattro("underline_position",  Py::Int(face->underline_position));
        setattro("underline_thickness", Py::Int(face->underline_thickness));
    }

    setattro("fname", Py::String(facefile));

    _VERBOSE("FT2Font::FT2Font done");
}

FT2Font::~FT2Font()
{
    _VERBOSE("FT2Font::~FT2Font");

    FT_Done_Face(face);

    for (size_t i = 0; i < glyphs.size(); i++)
    {
        FT_Done_Glyph(glyphs[i]);
    }
}

int
FT2Font::setattro(const Py::String &name, const Py::Object &value)
{
    _VERBOSE("FT2Font::setattr");
    __dict__[name] = value;
    return 1;
}

Py::Object
FT2Font::getattro(const Py::String &name)
{
    _VERBOSE("FT2Font::getattr");
    if (__dict__.hasKey(name)) return __dict__[name];
    else return genericGetAttro(name);
}

char FT2Font::clear__doc__[] =
    "clear()\n"
    "\n"
    "Clear all the glyphs, reset for a new set_text"
    ;

Py::Object
FT2Font::clear(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::clear");
    args.verify_length(0);

    angle = 0.0;

    pen.x = 0;
    pen.y = 0;

    for (size_t i = 0; i < glyphs.size(); i++)
    {
        FT_Done_Glyph(glyphs[i]);
    }

    glyphs.clear();

    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, clear)

char FT2Font::set_size__doc__[] =
    "set_size(ptsize, dpi)\n"
    "\n"
    "Set the point size and dpi of the text.\n"
    ;

Py::Object
FT2Font::set_size(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::set_size");
    args.verify_length(2);

    double ptsize = Py::Float(args[0]);
    double dpi = Py::Float(args[1]);

    int error = FT_Set_Char_Size(face, (long)(ptsize * 64), 0,
                                 (unsigned int)dpi * hinting_factor,
                                 (unsigned int)dpi);
    static FT_Matrix transform = { 65536 / hinting_factor, 0, 0, 65536 };
    FT_Set_Transform(face, &transform, 0);

    if (error)
    {
        throw Py::RuntimeError("Could not set the fontsize");
    }
    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, set_size)

char FT2Font::set_charmap__doc__[] =
    "set_charmap(i)\n"
    "\n"
    "Make the i-th charmap current\n"
    ;

Py::Object
FT2Font::set_charmap(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::set_charmap");
    args.verify_length(1);

    int i = Py::Int(args[0]);
    if (i >= face->num_charmaps)
    {
        throw Py::ValueError("i exceeds the available number of char maps");
    }
    FT_CharMap charmap = face->charmaps[i];
    if (FT_Set_Charmap(face, charmap))
    {
        throw Py::ValueError("Could not set the charmap");
    }
    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, set_charmap)

char FT2Font::select_charmap__doc__[] =
    "select_charmap(i)\n"
    "\n"
    "select charmap i where i is one of the FT_Encoding number\n"
    ;

Py::Object
FT2Font::select_charmap(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::set_charmap");
    args.verify_length(1);

    unsigned long i = Py::Long(args[0]);
    //if (FT_Select_Charmap( face, FT_ENCODING_ADOBE_CUSTOM ))
    if (FT_Select_Charmap(face, (FT_Encoding) i))
    {
        throw Py::ValueError("Could not set the charmap");
    }
    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, select_charmap)

FT_BBox
FT2Font::compute_string_bbox()
{
    _VERBOSE("FT2Font::compute_string_bbox");

    FT_BBox bbox;
    /* initialize string bbox to "empty" values */
    bbox.xMin = bbox.yMin = 32000;
    bbox.xMax = bbox.yMax = -32000;

    int right_side = 0;
    for (size_t n = 0; n < glyphs.size(); n++)
    {
        FT_BBox glyph_bbox;
        FT_Glyph_Get_CBox(glyphs[n], ft_glyph_bbox_subpixels, &glyph_bbox);
        if (glyph_bbox.xMin < bbox.xMin) bbox.xMin = glyph_bbox.xMin;
        if (glyph_bbox.yMin < bbox.yMin) bbox.yMin = glyph_bbox.yMin;
        if (glyph_bbox.xMin == glyph_bbox.xMax)
        {
            right_side += glyphs[n]->advance.x >> 10;
            if (right_side > bbox.xMax) bbox.xMax = right_side;
        }
        else
        {
            if (glyph_bbox.xMax > bbox.xMax) bbox.xMax = glyph_bbox.xMax;
        }
        if (glyph_bbox.yMax > bbox.yMax) bbox.yMax = glyph_bbox.yMax;
    }
    /* check that we really grew the string bbox */
    if (bbox.xMin > bbox.xMax)
    {
        bbox.xMin = 0;
        bbox.yMin = 0;
        bbox.xMax = 0;
        bbox.yMax = 0;
    }
    return bbox;
}

char FT2Font::get_kerning__doc__[] =
    "dx = get_kerning(left, right, mode)\n"
    "\n"
    "Get the kerning between left char and right glyph indices\n"
    "mode is a kerning mode constant\n"
    "  KERNING_DEFAULT  - Return scaled and grid-fitted kerning distances\n"
    "  KERNING_UNFITTED - Return scaled but un-grid-fitted kerning distances\n"
    "  KERNING_UNSCALED - Return the kerning vector in original font units\n"
    ;
Py::Object
FT2Font::get_kerning(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_kerning");
    args.verify_length(3);
    int left = Py::Int(args[0]);
    int right = Py::Int(args[1]);
    int mode = Py::Int(args[2]);


    if (!FT_HAS_KERNING(face))
    {
        return Py::Int(0);
    }
    FT_Vector delta;

    if (!FT_Get_Kerning(face, left, right, mode, &delta))
    {
        return Py::Int(delta.x / hinting_factor);
    }
    else
    {
        return Py::Int(0);

    }
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_kerning)


char FT2Font::set_text__doc__[] =
    "set_text(s, angle)\n"
    "\n"
    "Set the text string and angle.\n"
    "You must call this before draw_glyphs_to_bitmap\n"
    "A sequence of x,y positions is returned";
Py::Object
FT2Font::set_text(const Py::Tuple & args, const Py::Dict & kwargs)
{
    _VERBOSE("FT2Font::set_text");
    args.verify_length(2);


    Py::String text(args[0]);
    std::string stdtext = "";
    Py_UNICODE* pcode = NULL;
    size_t N = 0;
    if (PyUnicode_Check(text.ptr()))
    {
        pcode = PyUnicode_AsUnicode(text.ptr());
        N = PyUnicode_GetSize(text.ptr());
    }
    else
    {
        stdtext = text.as_std_string();
        N = stdtext.size();
    }


    angle = Py::Float(args[1]);

    angle = angle / 360.0 * 2 * 3.14159;

    long flags = FT_LOAD_FORCE_AUTOHINT;
    if (kwargs.hasKey("flags"))
    {
        flags = Py::Long(kwargs["flags"]);
    }

    //this computes width and height in subpixels so we have to divide by 64
    matrix.xx = (FT_Fixed)(cos(angle) * 0x10000L);
    matrix.xy = (FT_Fixed)(-sin(angle) * 0x10000L);
    matrix.yx = (FT_Fixed)(sin(angle) * 0x10000L);
    matrix.yy = (FT_Fixed)(cos(angle) * 0x10000L);

    FT_Bool use_kerning = FT_HAS_KERNING(face);
    FT_UInt previous = 0;

    glyphs.resize(0);
    pen.x = 0;
    pen.y = 0;

    Py::Tuple xys(N);
    for (unsigned int n = 0; n < N; n++)
    {
        std::string thischar("?");
        FT_UInt glyph_index;


        if (pcode == NULL)
        {
            // plain ol string
            thischar = stdtext[n];
            glyph_index = FT_Get_Char_Index(face, stdtext[n]);
        }
        else
        {
            //unicode
            glyph_index = FT_Get_Char_Index(face, pcode[n]);
        }

        // retrieve kerning distance and move pen position
        if (use_kerning && previous && glyph_index)
        {
            FT_Vector delta;
            FT_Get_Kerning(face, previous, glyph_index,
                           FT_KERNING_DEFAULT, &delta);
            pen.x += delta.x / hinting_factor;
        }
        error = FT_Load_Glyph(face, glyph_index, flags);
        if (error)
        {
            std::cerr << "\tcould not load glyph for " << thischar << std::endl;
            continue;
        }
        // ignore errors, jump to next glyph

        // extract glyph image and store it in our table

        FT_Glyph thisGlyph;
        error = FT_Get_Glyph(face->glyph, &thisGlyph);

        if (error)
        {
            std::cerr << "\tcould not get glyph for " << thischar << std::endl;
            continue;
        }
        // ignore errors, jump to next glyph

        FT_Glyph_Transform(thisGlyph, 0, &pen);
        Py::Tuple xy(2);
        xy[0] = Py::Float(pen.x);
        xy[1] = Py::Float(pen.y);
        xys[n] = xy;
        pen.x += face->glyph->advance.x;

        previous = glyph_index;
        glyphs.push_back(thisGlyph);
    }

    // now apply the rotation
    for (unsigned int n = 0; n < glyphs.size(); n++)
    {
        FT_Glyph_Transform(glyphs[n], &matrix, 0);
    }

    _VERBOSE("FT2Font::set_text done");
    return xys;
}
PYCXX_KEYWORDS_METHOD_DECL(FT2Font, set_text)

char FT2Font::get_num_glyphs__doc__[] =
    "get_num_glyphs()\n"
    "\n"
    "Return the number of loaded glyphs\n"
    ;
Py::Object
FT2Font::get_num_glyphs(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_num_glyphs");
    args.verify_length(0);

    return Py::Int((long)glyphs.size());
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_num_glyphs)

char FT2Font::load_char__doc__[] =
    "load_char(charcode, flags=LOAD_FORCE_AUTOHINT)\n"
    "\n"
    "Load character with charcode in current fontfile and set glyph.\n"
    "The flags argument can be a bitwise-or of the LOAD_XXX constants.\n"
    "Return value is a Glyph object, with attributes\n"
    "  width          # glyph width\n"
    "  height         # glyph height\n"
    "  bbox           # the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "  horiBearingX   # left side bearing in horizontal layouts\n"
    "  horiBearingY   # top side bearing in horizontal layouts\n"
    "  horiAdvance    # advance width for horizontal layout\n"
    "  vertBearingX   # left side bearing in vertical layouts\n"
    "  vertBearingY   # top side bearing in vertical layouts\n"
    "  vertAdvance    # advance height for vertical layout\n"
    ;
Py::Object
FT2Font::load_char(const Py::Tuple & args, const Py::Dict & kwargs)
{
    _VERBOSE("FT2Font::load_char");
    //load a char using the unsigned long charcode

    args.verify_length(1);
    long charcode = Py::Long(args[0]), flags = Py::Long(FT_LOAD_FORCE_AUTOHINT);
    if (kwargs.hasKey("flags"))
    {
        flags = Py::Long(kwargs["flags"]);
    }

    int error = FT_Load_Char(face, (unsigned long)charcode, flags);

    if (error)
    {
        throw Py::RuntimeError(Printf("Could not load charcode %d", charcode).str());
    }

    FT_Glyph thisGlyph;
    error = FT_Get_Glyph(face->glyph, &thisGlyph);

    if (error)
    {
        throw Py::RuntimeError(Printf("Could not get glyph for char %d", charcode).str());
    }

    size_t num = glyphs.size();  //the index into the glyphs list
    glyphs.push_back(thisGlyph);
    return Glyph::factory(face, thisGlyph, num, hinting_factor);
}
PYCXX_KEYWORDS_METHOD_DECL(FT2Font, load_char)

char FT2Font::load_glyph__doc__[] =
    "load_glyph(glyphindex, flags=LOAD_FORCE_AUTOHINT)\n"
    "\n"
    "Load character with glyphindex in current fontfile and set glyph.\n"
    "The flags argument can be a bitwise-or of the LOAD_XXX constants.\n"
    "Return value is a Glyph object, with attributes\n"
    "  width          # glyph width\n"
    "  height         # glyph height\n"
    "  bbox           # the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "  horiBearingX   # left side bearing in horizontal layouts\n"
    "  horiBearingY   # top side bearing in horizontal layouts\n"
    "  horiAdvance    # advance width for horizontal layout\n"
    "  vertBearingX   # left side bearing in vertical layouts\n"
    "  vertBearingY   # top side bearing in vertical layouts\n"
    "  vertAdvance    # advance height for vertical layout\n"
    ;
Py::Object
FT2Font::load_glyph(const Py::Tuple & args, const Py::Dict & kwargs)
{
    _VERBOSE("FT2Font::load_glyph");
    //load a char using the unsigned long charcode

    args.verify_length(1);
    long glyph_index = Py::Long(args[0]), flags = Py::Long(FT_LOAD_FORCE_AUTOHINT);
    if (kwargs.hasKey("flags"))
    {
        flags = Py::Long(kwargs["flags"]);
    }

    int error = FT_Load_Glyph(face, glyph_index, flags);

    if (error)
    {
        throw Py::RuntimeError(Printf("Could not load glyph index %d", glyph_index).str());
    }

    FT_Glyph thisGlyph;
    error = FT_Get_Glyph(face->glyph, &thisGlyph);

    if (error)
    {
        throw Py::RuntimeError(Printf("Could not get glyph for glyph index %d", glyph_index).str());
    }

    size_t num = glyphs.size();  //the index into the glyphs list
    glyphs.push_back(thisGlyph);
    return Glyph::factory(face, thisGlyph, num, hinting_factor);
}
PYCXX_KEYWORDS_METHOD_DECL(FT2Font, load_glyph)

char FT2Font::get_width_height__doc__[] =
    "w, h = get_width_height()\n"
    "\n"
    "Get the width and height in 26.6 subpixels of the current string set by set_text\n"
    "The rotation of the string is accounted for.  To get width and height\n"
    "in pixels, divide these values by 64\n"
    ;
Py::Object
FT2Font::get_width_height(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_width_height");
    args.verify_length(0);

    FT_BBox bbox = compute_string_bbox();

    Py::Tuple ret(2);
    ret[0] = Py::Int(bbox.xMax - bbox.xMin);
    ret[1] = Py::Int(bbox.yMax - bbox.yMin);
    return ret;
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_width_height)

char FT2Font::get_descent__doc__[] =
    "d = get_descent()\n"
    "\n"
    "Get the descent of the current string set by set_text in 26.6 subpixels.\n"
    "The rotation of the string is accounted for.  To get the descent\n"
    "in pixels, divide this value by 64.\n"
    ;
Py::Object
FT2Font::get_descent(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_descent");
    args.verify_length(0);

    FT_BBox bbox = compute_string_bbox();
    return Py::Int(- bbox.yMin);;
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_descent)

char FT2Font::draw_glyphs_to_bitmap__doc__[] =
    "draw_glyphs_to_bitmap()\n"
    "\n"
    "Draw the glyphs that were loaded by set_text to the bitmap\n"
    "The bitmap size will be automatically set to include the glyphs\n"
    ;
Py::Object
FT2Font::draw_glyphs_to_bitmap(const Py::Tuple &args, const Py::Dict &kwargs)
{

    _VERBOSE("FT2Font::draw_glyphs_to_bitmap");
    args.verify_length(0);

    long antialiased = 1;
    if (kwargs.hasKey("antialiased"))
    {
        antialiased = Py::Long(kwargs["antialiased"]);
    }

    FT_BBox string_bbox = compute_string_bbox();
    size_t width = (string_bbox.xMax - string_bbox.xMin) / 64 + 2;
    size_t height = (string_bbox.yMax - string_bbox.yMin) / 64 + 2;

    image = FT2Image::factory(width, height);
    FT2Image* image_cxx = Py::PythonClassObject<FT2Image>(image).getCxxObject();

    for (size_t n = 0; n < glyphs.size(); n++)
    {
        FT_BBox bbox;
        FT_Glyph_Get_CBox(glyphs[n], ft_glyph_bbox_pixels, &bbox);

        error = FT_Glyph_To_Bitmap(
            &glyphs[n],
            antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,
            0,
            1);
        if (error)
        {
            throw Py::RuntimeError("Could not convert glyph to bitmap");
        }

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[n];
        // now, draw to our target surface (convert position)

        //bitmap left and top in pixel, string bbox in subpixel
        FT_Int x = (FT_Int)(bitmap->left - (string_bbox.xMin / 64.));
        FT_Int y = (FT_Int)((string_bbox.yMax / 64.) - bitmap->top + 1);

        image_cxx->draw_bitmap(&bitmap->bitmap, x, y);
    }

    return Py::Object();
}
PYCXX_KEYWORDS_METHOD_DECL(FT2Font, draw_glyphs_to_bitmap)

char FT2Font::get_xys__doc__[] =
    "get_xys()\n"
    "\n"
    "Get the xy locations of the current glyphs\n"
    ;
Py::Object
FT2Font::get_xys(const Py::Tuple &args, const Py::Dict &kwargs)
{
    _VERBOSE("FT2Font::get_xys");
    args.verify_length(0);

    long antialiased = 1;
    if (kwargs.hasKey("antialiased"))
    {
        antialiased = Py::Long(kwargs["antialiased"]);
    }

    FT_BBox string_bbox = compute_string_bbox();
    Py::Tuple xys(glyphs.size());

    for (size_t n = 0; n < glyphs.size(); n++)
    {

        FT_BBox bbox;
        FT_Glyph_Get_CBox(glyphs[n], ft_glyph_bbox_pixels, &bbox);

        error = FT_Glyph_To_Bitmap(
            &glyphs[n],
            antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,
            0,
            1);
        if (error)
        {
            throw Py::RuntimeError("Could not convert glyph to bitmap");
        }

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[n];


        //bitmap left and top in pixel, string bbox in subpixel
        FT_Int x = (FT_Int)(bitmap->left - string_bbox.xMin / 64.);
        FT_Int y = (FT_Int)(string_bbox.yMax / 64. - bitmap->top + 1);
        //make sure the index is non-neg
        x = x < 0 ? 0 : x;
        y = y < 0 ? 0 : y;
        Py::Tuple xy(2);
        xy[0] = Py::Float(x);
        xy[1] = Py::Float(y);
        xys[n] = xy;
    }

    return xys;
}
PYCXX_KEYWORDS_METHOD_DECL(FT2Font, get_xys)

char FT2Font::draw_glyph_to_bitmap__doc__[] =
    "draw_glyph_to_bitmap(bitmap, x, y, glyph)\n"
    "\n"
    "Draw a single glyph to the bitmap at pixel locations x,y\n"
    "Note it is your responsibility to set up the bitmap manually\n"
    "with set_bitmap_size(w,h) before this call is made.\n"
    "\n"
    "If you want automatic layout, use set_text in combinations with\n"
    "draw_glyphs_to_bitmap.  This function is intended for people who\n"
    "want to render individual glyphs at precise locations, eg, a\n"
    "a glyph returned by load_char\n";

Py::Object
FT2Font::draw_glyph_to_bitmap(const Py::Tuple &args, const Py::Dict &kwargs)
{
    _VERBOSE("FT2Font::draw_glyph_to_bitmap");
    args.verify_length(4);

    FT2Image* im = Py::PythonClassObject<FT2Image>(args[0]).getCxxObject();

    double xd = Py::Float(args[1]);
    double yd = Py::Float(args[2]);
    long x = (long)xd;
    long y = (long)yd;
    FT_Vector sub_offset;
    sub_offset.x = 0; // int((xd - (double)x) * 64.0);
    sub_offset.y = 0; // int((yd - (double)y) * 64.0);

    Glyph* glyph = Py::PythonClassObject<Glyph>(args[3]).getCxxObject();

    long antialiased = 1;
    if (kwargs.hasKey("antialiased"))
    {
        antialiased = Py::Long(kwargs["antialiased"]);
    }

    if (glyph->glyphInd >= glyphs.size())
    {
        throw Py::ValueError("glyph num is out of range");
    }

    error = FT_Glyph_To_Bitmap(
        &glyphs[glyph->glyphInd],
        antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,
        &sub_offset,  // additional translation
        1   //destroy image
        );
    if (error)
    {
        throw Py::RuntimeError("Could not convert glyph to bitmap");
    }

    FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[glyph->glyphInd];

    im->draw_bitmap(&bitmap->bitmap, x + bitmap->left, y);
    return Py::Object();
}
PYCXX_KEYWORDS_METHOD_DECL(FT2Font, draw_glyph_to_bitmap)

char FT2Font::get_glyph_name__doc__[] =
    "get_glyph_name(index)\n"
    "\n"
    "Retrieves the ASCII name of a given glyph in a face.\n"
    ;
Py::Object
FT2Font::get_glyph_name(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_glyph_name");
    args.verify_length(1);

    char buffer[128];
    FT_UInt glyph_number = (FT_UInt)(unsigned long long)Py::Int(args[0]);

    if (!FT_HAS_GLYPH_NAMES(face))
    {
        /* Note that this generated name must match the name that
           is generated by ttconv in ttfont_CharStrings_getname. */
        PyOS_snprintf(buffer, 128, "uni%08x", glyph_number);
    } else {
        if (FT_Get_Glyph_Name(face, glyph_number, buffer, 128))
        {
            throw Py::RuntimeError("Could not get glyph names.");
        }
     }

    return Py::String(buffer);
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_glyph_name)

char FT2Font::get_charmap__doc__[] =
    "get_charmap()\n"
    "\n"
    "Returns a dictionary that maps the character codes of the selected charmap\n"
    "(Unicode by default) to their corresponding glyph indices.\n"
    ;
Py::Object
FT2Font::get_charmap(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_charmap");
    args.verify_length(0);

    FT_UInt index;
    Py::Dict charmap;

    //std::cout << "asd" << face->charmaps[1]->encoding << std::endl;
    FT_ULong code = FT_Get_First_Char(face, &index);
    while (index != 0)
    {
        charmap[Py::Long((long) code)] = Py::Int((int) index);
        code = FT_Get_Next_Char(face, code, &index);
    }
    return charmap;
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_charmap)

// ID        Platform       Encoding
// 0         Unicode        Reserved (set to 0)
// 1         Macintoch      The Script Manager code
// 2         ISO            ISO encoding
// 3         Microsoft      Microsoft encoding
// 240-255   User-defined   Reserved for all nonregistered platforms

// Code      ISO encoding scheme
// 0         7-bit ASCII
// 1         ISO 10646
// 2         ISO 8859-1

// Code      Language       Code      Language       Code
// 0         English        10        Hebrew         20        Urdu
// 1         French         11        Japanese       21        Hindi
// 2         German         12        Arabic         22        Thai
// 3         Italian        13        Finnish
// 4         Dutch          14        Greek
// 5         Swedish        15        Icelandic
// 6         Spanish        16        Maltese
// 7         Danish         17        Turkish
// 8         Portuguese     18        Yugoslavian
// 9         Norwegian      19        Chinese

// Code      Meaning        Description
// 0         Copyright notice     e.g. "Copyright Apple Computer, Inc. 1992
// 1         Font family name     e.g. "New York"
// 2         Font style           e.g. "Bold"
// 3         Font identification  e.g. "Apple Computer New York Bold Ver 1"
// 4         Full font name       e.g. "New York Bold"
// 5         Version string       e.g. "August 10, 1991, 1.08d21"
// 6         Postscript name      e.g. "Times-Bold"
// 7         Trademark
// 8         Designer             e.g. "Apple Computer"

char FT2Font::get_sfnt__doc__[] =
    "get_sfnt(name)\n"
    "\n"
    "Get all values from the SFNT names table.  Result is a dictionary whose"
    "key is the platform-ID, ISO-encoding-scheme, language-code, and"
    "description.\n"
    /*
      "The font name identifier codes are:\n"
      "\n"
      "  0    Copyright notice     e.g. Copyright Apple Computer, Inc. 1992\n"
      "  1    Font family name     e.g. New York\n"
      "  2    Font style           e.g. Bold\n"
      "  3    Font identification  e.g. Apple Computer New York Bold Ver 1\n"
      "  4    Full font name       e.g. New York Bold\n"
      "  5    Version string       e.g. August 10, 1991, 1.08d21\n"
      "  6    Postscript name      e.g. Times-Bold\n"
      "  7    Trademark            \n"
      "  8    Designer             e.g. Apple Computer\n"
      "  11   URL                  e.g. http://www.apple.com\n"
      "  13   Copyright license    \n"
    */
    ;
Py::Object
FT2Font::get_sfnt(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_sfnt");
    args.verify_length(0);

    if (!(face->face_flags & FT_FACE_FLAG_SFNT))
    {
        throw Py::RuntimeError("No SFNT name table");
    }

    size_t count = FT_Get_Sfnt_Name_Count(face);

    Py::Dict names;
    for (size_t j = 0; j < count; j++)
    {
        FT_SfntName sfnt;
        FT_Error error = FT_Get_Sfnt_Name(face, j, &sfnt);

        if (error)
        {
            throw Py::RuntimeError("Could not get SFNT name");
        }

        Py::Tuple key(4);
        key[0] = Py::Int(sfnt.platform_id);
        key[1] = Py::Int(sfnt.encoding_id);
        key[2] = Py::Int(sfnt.language_id);
        key[3] = Py::Int(sfnt.name_id);

        names[key] = Py::asObject
            (PyBytes_FromStringAndSize(
                (const char *)sfnt.string, sfnt.string_len));
    }
    return names;
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_sfnt)

char FT2Font::get_name_index__doc__[] =
    "get_name_index(name)\n"
    "\n"
    "Returns the glyph index of a given glyph name.\n"
    "The glyph index 0 means `undefined character code'.\n"
    ;
Py::Object
FT2Font::get_name_index(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_name_index");
    args.verify_length(1);
    std::string glyphname = Py::String(args[0]);

    return Py::Long((long)
                    FT_Get_Name_Index(face, (FT_String *) glyphname.c_str()));
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_name_index)

char FT2Font::get_ps_font_info__doc__[] =
    "get_ps_font_info()\n"
    "\n"
    "Return the information in the PS Font Info structure.\n"
    ;
Py::Object
FT2Font::get_ps_font_info(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_ps_font_info");
    args.verify_length(0);
    PS_FontInfoRec fontinfo;

    FT_Error error = FT_Get_PS_Font_Info(face, &fontinfo);
    if (error)
    {
        Py::RuntimeError("Could not get PS font info");
        return Py::Object();
    }

    Py::Tuple info(9);
    info[0] = Py::String(fontinfo.version ? fontinfo.version : "");
    info[1] = Py::String(fontinfo.notice ? fontinfo.notice : "");
    info[2] = Py::String(fontinfo.full_name ? fontinfo.full_name : "");
    info[3] = Py::String(fontinfo.family_name ? fontinfo.family_name : "");
    info[4] = Py::String(fontinfo.weight ? fontinfo.weight : "");
    info[5] = Py::Long(fontinfo.italic_angle);
    info[6] = Py::Int(fontinfo.is_fixed_pitch);
    info[7] = Py::Int(fontinfo.underline_position);
    info[8] = Py::Int(fontinfo.underline_thickness);
    return info;
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_ps_font_info)

char FT2Font::get_sfnt_table__doc__[] =
    "get_sfnt_table(name)\n"
    "\n"
    "Return one of the following SFNT tables: head, maxp, OS/2, hhea, "
    "vhea, post, or pclt.\n"
    ;
Py::Object
FT2Font::get_sfnt_table(const Py::Tuple & args)
{
    _VERBOSE("FT2Font::get_sfnt_table");
    args.verify_length(1);
    std::string tagname = Py::String(args[0]);

    int tag;
    const char *tags[] = {"head", "maxp", "OS/2", "hhea",
                          "vhea", "post", "pclt",  NULL
                         };

    for (tag = 0; tags[tag] != NULL; tag++)
    {
        if (strcmp(tagname.c_str(), tags[tag]) == 0)
        {
            break;
        }
    }

    void *table = FT_Get_Sfnt_Table(face, (FT_Sfnt_Tag) tag);
    if (!table)
    {
        return Py::Object();
    }

    switch (tag)
    {
    case 0:
        {
            char head_dict[] = "{s:(h,h), s:(h,h), s:l, s:l, s:i, s:i,"
                "s:(l,l), s:(l,l), s:h, s:h, s:h, s:h, s:i, s:i, s:h, s:h, s:h}";
            TT_Header *t = (TT_Header *)table;
            return Py::asObject(Py_BuildValue(head_dict,
                                              "version",
                                              FIXED_MAJOR(t->Table_Version),
                                              FIXED_MINOR(t->Table_Version),
                                              "fontRevision",
                                              FIXED_MAJOR(t->Font_Revision),
                                              FIXED_MINOR(t->Font_Revision),
                                              "checkSumAdjustment", t->CheckSum_Adjust,
                                              "magicNumber" ,       t->Magic_Number,
                                              "flags", (unsigned)t->Flags,
                                              "unitsPerEm", (unsigned)t->Units_Per_EM,
                                              "created",            t->Created[0], t->Created[1],
                                              "modified",           t->Modified[0], t->Modified[1],
                                              "xMin",               t->xMin,
                                              "yMin",               t->yMin,
                                              "xMax",               t->xMax,
                                              "yMax",               t->yMax,
                                              "macStyle", (unsigned)t->Mac_Style,
                                              "lowestRecPPEM", (unsigned)t->Lowest_Rec_PPEM,
                                              "fontDirectionHint",  t->Font_Direction,
                                              "indexToLocFormat",   t->Index_To_Loc_Format,
                                              "glyphDataFormat",    t->Glyph_Data_Format));
        }
    case 1:
        {
            char maxp_dict[] = "{s:(h,h), s:i, s:i, s:i, s:i, s:i, s:i,"
                "s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i}";
            TT_MaxProfile *t = (TT_MaxProfile *)table;
            return Py::asObject(Py_BuildValue(maxp_dict,
                                              "version",
                                              FIXED_MAJOR(t->version),
                                              FIXED_MINOR(t->version),
                                              "numGlyphs", (unsigned)t->numGlyphs,
                                              "maxPoints", (unsigned)t->maxPoints,
                                              "maxContours", (unsigned)t->maxContours,
                                              "maxComponentPoints",
                                              (unsigned)t->maxCompositePoints,
                                              "maxComponentContours",
                                              (unsigned)t->maxCompositeContours,
                                              "maxZones", (unsigned)t->maxZones,
                                              "maxTwilightPoints", (unsigned)t->maxTwilightPoints,
                                              "maxStorage", (unsigned)t->maxStorage,
                                              "maxFunctionDefs", (unsigned)t->maxFunctionDefs,
                                              "maxInstructionDefs",
                                              (unsigned)t->maxInstructionDefs,
                                              "maxStackElements", (unsigned)t->maxStackElements,
                                              "maxSizeOfInstructions",
                                              (unsigned)t->maxSizeOfInstructions,
                                              "maxComponentElements",
                                              (unsigned)t->maxComponentElements,
                                              "maxComponentDepth",
                                              (unsigned)t->maxComponentDepth));
        }
    case 2:
        {
            #if PY3K
            char os_2_dict[] = "{s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:h,"
                "s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:y#, s:(llll),"
                "s:y#, s:h, s:h, s:h}";
            #else
            char os_2_dict[] = "{s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:h,"
                "s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:s#, s:(llll),"
                "s:s#, s:h, s:h, s:h}";
            #endif
            TT_OS2 *t = (TT_OS2 *)table;
            return Py::asObject(Py_BuildValue(os_2_dict,
                                              "version", (unsigned)t->version,
                                              "xAvgCharWidth",      t->xAvgCharWidth,
                                              "usWeightClass", (unsigned)t->usWeightClass,
                                              "usWidthClass", (unsigned)t->usWidthClass,
                                              "fsType",             t->fsType,
                                              "ySubscriptXSize",    t->ySubscriptXSize,
                                              "ySubscriptYSize",    t->ySubscriptYSize,
                                              "ySubscriptXOffset",  t->ySubscriptXOffset,
                                              "ySubscriptYOffset",  t->ySubscriptYOffset,
                                              "ySuperscriptXSize",  t->ySuperscriptXSize,
                                              "ySuperscriptYSize",  t->ySuperscriptYSize,
                                              "ySuperscriptXOffset", t->ySuperscriptXOffset,
                                              "ySuperscriptYOffset", t->ySuperscriptYOffset,
                                              "yStrikeoutSize",     t->yStrikeoutSize,
                                              "yStrikeoutPosition", t->yStrikeoutPosition,
                                              "sFamilyClass",       t->sFamilyClass,
                                              "panose",             t->panose, 10,
                                              "ulCharRange",
                                              (unsigned long) t->ulUnicodeRange1,
                                              (unsigned long) t->ulUnicodeRange2,
                                              (unsigned long) t->ulUnicodeRange3,
                                              (unsigned long) t->ulUnicodeRange4,
                                              "achVendID",          t->achVendID, 4,
                                              "fsSelection", (unsigned)t->fsSelection,
                                              "fsFirstCharIndex", (unsigned)t->usFirstCharIndex,
                                              "fsLastCharIndex", (unsigned)t->usLastCharIndex));
        }
    case 3:
        {
            char hhea_dict[] = "{s:(h,h), s:h, s:h, s:h, s:i, s:h, s:h, s:h,"
                "s:h, s:h, s:h, s:h, s:i}";
            TT_HoriHeader *t = (TT_HoriHeader *)table;
            return Py::asObject(Py_BuildValue(hhea_dict,
                                              "version",
                                              FIXED_MAJOR(t->Version),
                                              FIXED_MINOR(t->Version),
                                              "ascent",             t->Ascender,
                                              "descent",            t->Descender,
                                              "lineGap",            t->Line_Gap,
                                              "advanceWidthMax", (unsigned)t->advance_Width_Max,
                                              "minLeftBearing",     t->min_Left_Side_Bearing,
                                              "minRightBearing",    t->min_Right_Side_Bearing,
                                              "xMaxExtent",         t->xMax_Extent,
                                              "caretSlopeRise",     t->caret_Slope_Rise,
                                              "caretSlopeRun",      t->caret_Slope_Run,
                                              "caretOffset",        t->caret_Offset,
                                              "metricDataFormat",   t->metric_Data_Format,
                                              "numOfLongHorMetrics",
                                              (unsigned)t->number_Of_HMetrics));
        }
    case 4:
        {
            char vhea_dict[] = "{s:(h,h), s:h, s:h, s:h, s:i, s:h, s:h, s:h,"
                "s:h, s:h, s:h, s:h, s:i}";
            TT_VertHeader *t = (TT_VertHeader *)table;
            return Py::asObject(Py_BuildValue(vhea_dict,
                                              "version",
                                              FIXED_MAJOR(t->Version),
                                              FIXED_MINOR(t->Version),
                                              "vertTypoAscender",   t->Ascender,
                                              "vertTypoDescender",  t->Descender,
                                              "vertTypoLineGap",    t->Line_Gap,
                                              "advanceHeightMax", (unsigned)t->advance_Height_Max,
                                              "minTopSideBearing",  t->min_Top_Side_Bearing,
                                              "minBottomSizeBearing", t->min_Bottom_Side_Bearing,
                                              "yMaxExtent",         t->yMax_Extent,
                                              "caretSlopeRise",     t->caret_Slope_Rise,
                                              "caretSlopeRun",      t->caret_Slope_Run,
                                              "caretOffset",        t->caret_Offset,
                                              "metricDataFormat",   t->metric_Data_Format,
                                              "numOfLongVerMetrics",
                                              (unsigned)t->number_Of_VMetrics));
        }
    case 5:
        {
            TT_Postscript *t = (TT_Postscript *)table;
            Py::Dict post;
            Py::Tuple format(2), angle(2);
            format[0] = Py::Int(FIXED_MAJOR(t->FormatType));
            format[1] = Py::Int(FIXED_MINOR(t->FormatType));
            post["format"]             = format;
            angle[0]  = Py::Int(FIXED_MAJOR(t->italicAngle));
            angle[1]  = Py::Int(FIXED_MINOR(t->italicAngle));
            post["italicAngle"]        = angle;
            post["underlinePosition"]  = Py::Int(t->underlinePosition);
            post["underlineThickness"] = Py::Int(t->underlineThickness);
            post["isFixedPitch"]       = Py::Long((long) t->isFixedPitch);
            post["minMemType42"]       = Py::Long((long) t->minMemType42);
            post["maxMemType42"]       = Py::Long((long) t->maxMemType42);
            post["minMemType1"]        = Py::Long((long) t->minMemType1);
            post["maxMemType1"]        = Py::Long((long) t->maxMemType1);
            return post;
        }
    case 6:
        {
            TT_PCLT *t = (TT_PCLT *)table;
            Py::Dict pclt;
            Py::Tuple version(2);
            version[0] = Py::Int(FIXED_MAJOR(t->Version));
            version[1] = Py::Int(FIXED_MINOR(t->Version));
            pclt["version"]            = version;
            pclt["fontNumber"]         = Py::Long((long) t->FontNumber);
            pclt["pitch"]              = Py::Int((short) t->Pitch);
            pclt["xHeight"]            = Py::Int((short) t->xHeight);
            pclt["style"]              = Py::Int((short) t->Style);
            pclt["typeFamily"]         = Py::Int((short) t->TypeFamily);
            pclt["capHeight"]          = Py::Int((short) t->CapHeight);
            pclt["symbolSet"]          = Py::Int((short) t->SymbolSet);
            #if PY3K
            pclt["typeFace"]           = Py::String((char *) t->TypeFace, 16, "latin-1");
            pclt["characterComplement"] = Py::Bytes((char *) t->CharacterComplement, 8);
            #else
            pclt["typeFace"]           = Py::String((char *) t->TypeFace, 16);
            pclt["characterComplement"] = Py::String((char *) t->CharacterComplement, 8);
            #endif
            // pclt["filename"]           = Py::String((char *) t->FileName, 6);
            pclt["strokeWeight"]       = Py::Int((int) t->StrokeWeight);
            pclt["widthType"]          = Py::Int((int) t->WidthType);
            pclt["serifStyle"]         = Py::Int((int) t->SerifStyle);
            return pclt;
        }
    default:
        return Py::Object();
    }
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_sfnt_table)

char FT2Font::get_image__doc__ [] =
    "get_image()\n"
    "\n"
    "Returns the underlying image buffer for this font object.\n";
Py::Object
FT2Font::get_image(const Py::Tuple &args)
{
    args.verify_length(0);
    if (!image.isNone())
    {
        return image;
    }
    throw Py::RuntimeError("You must call .set_text() before .get_image()");
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, get_image)

char FT2Font::attach_file__doc__ [] =
    "attach_file(filename)\n"
    "\n"
    "Attach a file with extra information on the font\n"
    "(in practice, an AFM file with the metrics of a Type 1 font).\n"
    "Throws an exception if unsuccessful.\n";
Py::Object
FT2Font::attach_file(const Py::Tuple &args)
{
    args.verify_length(1);

    std::string filename = Py::String(args[0]);
    FT_Error error = FT_Attach_File(face, filename.c_str());

    if (error)
    {
        std::ostringstream s;
        s << "Could not attach file " << filename
        << " (freetype error code " << error << ")" << std::endl;
        throw Py::RuntimeError(s.str());
    }
    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(FT2Font, attach_file)

void
FT2Image::init_type(void)
{
    _VERBOSE("FT2Image::init_type");
    behaviors().name("FT2Image");
    behaviors().doc("FT2Image");

    PYCXX_ADD_VARARGS_METHOD(write_bitmap, py_write_bitmap,
                             FT2Image::write_bitmap__doc__);
    PYCXX_ADD_VARARGS_METHOD(draw_rect, py_draw_rect,
                             FT2Image::draw_rect__doc__);
    PYCXX_ADD_VARARGS_METHOD(draw_rect_filled, py_draw_rect_filled,
                             FT2Image::draw_rect_filled__doc__);
    PYCXX_ADD_VARARGS_METHOD(as_array, py_as_array,
                             FT2Image::as_array__doc__);
    PYCXX_ADD_VARARGS_METHOD(as_str, py_as_str,
                             FT2Image::as_str__doc__);
    PYCXX_ADD_VARARGS_METHOD(as_rgb_str, py_as_rgb_str,
                             FT2Image::as_rgb_str__doc__);
    PYCXX_ADD_VARARGS_METHOD(as_rgba_str, py_as_rgba_str,
                             FT2Image::as_rgba_str__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_width, py_get_width,
                             "Returns the width of the image");
    PYCXX_ADD_VARARGS_METHOD(get_height, py_get_height,
                             "Returns the height of the image");

    behaviors().readyType();
}

void
Glyph::init_type()
{
    _VERBOSE("Glyph::init_type");
    behaviors().name("Glyph");
    behaviors().doc("Glyph");
    behaviors().supportGetattro();
    behaviors().supportSetattro();
    behaviors().readyType();
}

void
FT2Font::init_type()
{
    _VERBOSE("FT2Font::init_type");
    behaviors().name("FT2Font");
    behaviors().doc("FT2Font");
    behaviors().supportGetattro();
    behaviors().supportSetattro();

    PYCXX_ADD_VARARGS_METHOD(clear, clear,
                             FT2Font::clear__doc__);
    PYCXX_ADD_KEYWORDS_METHOD(draw_glyph_to_bitmap, draw_glyph_to_bitmap,
                             FT2Font::draw_glyph_to_bitmap__doc__);
    PYCXX_ADD_KEYWORDS_METHOD(draw_glyphs_to_bitmap, draw_glyphs_to_bitmap,
                             FT2Font::draw_glyphs_to_bitmap__doc__);
    PYCXX_ADD_KEYWORDS_METHOD(get_xys, get_xys,
                             FT2Font::get_xys__doc__);

    PYCXX_ADD_VARARGS_METHOD(get_num_glyphs, get_num_glyphs,
                             FT2Font::get_num_glyphs__doc__);
    PYCXX_ADD_KEYWORDS_METHOD(load_char, load_char,
                              FT2Font::load_char__doc__);
    PYCXX_ADD_KEYWORDS_METHOD(load_glyph, load_glyph,
                              FT2Font::load_glyph__doc__);
    PYCXX_ADD_KEYWORDS_METHOD(set_text, set_text,
                              FT2Font::set_text__doc__);
    PYCXX_ADD_VARARGS_METHOD(set_size, set_size,
                             FT2Font::set_size__doc__);
    PYCXX_ADD_VARARGS_METHOD(set_charmap, set_charmap,
                             FT2Font::set_charmap__doc__);
    PYCXX_ADD_VARARGS_METHOD(select_charmap, select_charmap,
                             FT2Font::select_charmap__doc__);

    PYCXX_ADD_VARARGS_METHOD(get_width_height, get_width_height,
                             FT2Font::get_width_height__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_descent, get_descent,
                             FT2Font::get_descent__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_glyph_name, get_glyph_name,
                             FT2Font::get_glyph_name__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_charmap, get_charmap,
                             FT2Font::get_charmap__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_kerning, get_kerning,
                             FT2Font::get_kerning__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_sfnt, get_sfnt,
                             FT2Font::get_sfnt__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_name_index, get_name_index,
                             FT2Font::get_name_index__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_ps_font_info, get_ps_font_info,
                             FT2Font::get_ps_font_info__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_sfnt_table, get_sfnt_table,
                             FT2Font::get_sfnt_table__doc__);
    PYCXX_ADD_VARARGS_METHOD(get_image, get_image,
                             FT2Font::get_image__doc__);
    PYCXX_ADD_VARARGS_METHOD(attach_file, attach_file,
                             FT2Font::attach_file__doc__);
    PYCXX_ADD_NOARGS_METHOD(get_path, get_path,
                            FT2Font::get_path__doc__);

    behaviors().readyType();
}

//todo add module docs strings

char ft2font__doc__[] =
    "ft2font\n"
    "\n"
    "Methods:\n"
    "  FT2Font(ttffile)\n"
    "Face Constants\n"
    "  SCALABLE               scalable\n"
    "  FIXED_SIZES            \n"
    "  FIXED_WIDTH            \n"
    "  SFNT                   \n"
    "  HORIZONTAL             \n"
    "  VERTICAL               \n"
    "  KERNING                \n"
    "  FAST_GLYPHS            \n"
    "  MULTIPLE_MASTERS       \n"
    "  GLYPH_NAMES            \n"
    "  EXTERNAL_STREAM        \n"
    "Style Constants\n"
    "  ITALIC                 \n"
    "  BOLD                   \n"
    ;

/* Function of no arguments returning new FT2Font object */
char ft2font_new__doc__[] =
    "FT2Font(ttffile)\n"
    "\n"
    "Create a new FT2Font object\n"
    "The following global font attributes are defined:\n"
    "  num_faces              number of faces in file\n"
    "  face_flags             face flags  (int type); see the ft2font constants\n"
    "  style_flags            style flags  (int type); see the ft2font constants\n"
    "  num_glyphs             number of glyphs in the face\n"
    "  family_name            face family name\n"
    "  style_name             face syle name\n"
    "  num_fixed_sizes        number of bitmap in the face\n"
    "  scalable               face is scalable\n"
    "\n"
    "The following are available, if scalable is true:\n"
    "  bbox                   face global bounding box (xmin, ymin, xmax, ymax)\n"
    "  units_per_EM           number of font units covered by the EM\n"
    "  ascender               ascender in 26.6 units\n"
    "  descender              descender in 26.6 units\n"
    "  height                 height in 26.6 units; used to compute a default\n"
    "                         line spacing (baseline-to-baseline distance)\n"
    "  max_advance_width      maximum horizontal cursor advance for all glyphs\n"
    "  max_advance_height     same for vertical layout\n"
    "  underline_position     vertical position of the underline bar\n"
    "  underline_thickness    vertical thickness of the underline\n"
    "  postscript_name        PostScript name of the font\n"
    ;

ft2font_module::ft2font_module()
    : Py::ExtensionModule<ft2font_module>("ft2font")
{
    FT2Image::init_type();
    Glyph::init_type();
    FT2Font::init_type();

    initialize("The ft2font module");

    Py::Dict d(moduleDictionary());
    Py::Object ft2font_type(FT2Font::type());
    d["FT2Font"] = ft2font_type;
    Py::Object ft2image_type(FT2Image::type());
    d["FT2Image"] = ft2image_type;
}

ft2font_module::~ft2font_module()
{
    FT_Done_FreeType(_ft2Library);
}

PyMODINIT_FUNC
#if PY3K
PyInit_ft2font(void)
#else
initft2font(void)
#endif
{
    static ft2font_module* ft2font = new ft2font_module;
    Py::Dict d = ft2font->moduleDictionary();
    d["SCALABLE"]         = Py::Int(FT_FACE_FLAG_SCALABLE);
    d["FIXED_SIZES"]      = Py::Int(FT_FACE_FLAG_FIXED_SIZES);
    d["FIXED_WIDTH"]      = Py::Int(FT_FACE_FLAG_FIXED_WIDTH);
    d["SFNT"]             = Py::Int(FT_FACE_FLAG_SFNT);
    d["HORIZONTAL"]       = Py::Int(FT_FACE_FLAG_HORIZONTAL);
    d["VERTICAL"]         = Py::Int(FT_FACE_FLAG_SCALABLE);
    d["KERNING"]          = Py::Int(FT_FACE_FLAG_KERNING);
    d["FAST_GLYPHS"]      = Py::Int(FT_FACE_FLAG_FAST_GLYPHS);
    d["MULTIPLE_MASTERS"] = Py::Int(FT_FACE_FLAG_MULTIPLE_MASTERS);
    d["GLYPH_NAMES"]      = Py::Int(FT_FACE_FLAG_GLYPH_NAMES);
    d["EXTERNAL_STREAM"]  = Py::Int(FT_FACE_FLAG_EXTERNAL_STREAM);
    d["ITALIC"]           = Py::Int(FT_STYLE_FLAG_ITALIC);
    d["BOLD"]             = Py::Int(FT_STYLE_FLAG_BOLD);
    d["KERNING_DEFAULT"]  = Py::Int(FT_KERNING_DEFAULT);
    d["KERNING_UNFITTED"]  = Py::Int(FT_KERNING_UNFITTED);
    d["KERNING_UNSCALED"]  = Py::Int(FT_KERNING_UNSCALED);

    d["LOAD_DEFAULT"]          = Py::Long(FT_LOAD_DEFAULT);
    d["LOAD_NO_SCALE"]         = Py::Long(FT_LOAD_NO_SCALE);
    d["LOAD_NO_HINTING"]       = Py::Long(FT_LOAD_NO_HINTING);
    d["LOAD_RENDER"]           = Py::Long(FT_LOAD_RENDER);
    d["LOAD_NO_BITMAP"]        = Py::Long(FT_LOAD_NO_BITMAP);
    d["LOAD_VERTICAL_LAYOUT"]  = Py::Long(FT_LOAD_VERTICAL_LAYOUT);
    d["LOAD_FORCE_AUTOHINT"]   = Py::Long(FT_LOAD_FORCE_AUTOHINT);
    d["LOAD_CROP_BITMAP"]      = Py::Long(FT_LOAD_CROP_BITMAP);
    d["LOAD_PEDANTIC"]         = Py::Long(FT_LOAD_PEDANTIC);
    d["LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH"] =
        Py::Long(FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH);
    d["LOAD_NO_RECURSE"]       = Py::Long(FT_LOAD_NO_RECURSE);
    d["LOAD_IGNORE_TRANSFORM"] = Py::Long(FT_LOAD_IGNORE_TRANSFORM);
    d["LOAD_MONOCHROME"]       = Py::Long(FT_LOAD_MONOCHROME);
    d["LOAD_LINEAR_DESIGN"]    = Py::Long(FT_LOAD_LINEAR_DESIGN);
    // These need casting because large-valued numeric literals could
    // be either longs or unsigned longs:
    d["LOAD_NO_AUTOHINT"]      = Py::Long((unsigned long)FT_LOAD_NO_AUTOHINT);
    d["LOAD_TARGET_NORMAL"]    = Py::Long((unsigned long)FT_LOAD_TARGET_NORMAL);
    d["LOAD_TARGET_LIGHT"]     = Py::Long((unsigned long)FT_LOAD_TARGET_LIGHT);
    d["LOAD_TARGET_MONO"]      = Py::Long((unsigned long)FT_LOAD_TARGET_MONO);
    d["LOAD_TARGET_LCD"]       = Py::Long((unsigned long)FT_LOAD_TARGET_LCD);
    d["LOAD_TARGET_LCD_V"]     = Py::Long((unsigned long)FT_LOAD_TARGET_LCD_V);

    //initialize library
    int error = FT_Init_FreeType(&_ft2Library);

    if (error)
    {
        throw Py::RuntimeError("Could not find initialize the freetype2 library");
    }

    {
        FT_Int major, minor, patch;
        char version_string[64];

        FT_Library_Version(_ft2Library, &major, &minor, &patch);
        sprintf(version_string, "%d.%d.%d", major, minor, patch);

        d["__freetype_version__"] = Py::String(version_string);
    }

    import_array();

    #if PY3K
    return ft2font->module().ptr();
    #endif
}

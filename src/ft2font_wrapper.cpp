#include "mplutils.h"
#include "ft2font.h"
#include "file_compat.h"
#include "py_converters.h"
#include "py_exceptions.h"
#include "numpy_cpp.h"

// From Python
#include <structmember.h>

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

static PyObject *convert_xys_to_array(std::vector<double> &xys)
{
    npy_intp dims[] = {(npy_intp)xys.size() / 2, 2 };
    if (dims[0] > 0) {
        return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, &xys[0]);
    } else {
        return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    }
}

/**********************************************************************
 * FT2Image
 * */

typedef struct
{
    PyObject_HEAD
    FT2Image *x;
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];
    Py_ssize_t suboffsets[2];
} PyFT2Image;

static PyObject *PyFT2Image_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFT2Image *self;
    self = (PyFT2Image *)type->tp_alloc(type, 0);
    self->x = NULL;
    return (PyObject *)self;
}

static int PyFT2Image_init(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    double width;
    double height;

    if (!PyArg_ParseTuple(args, "dd:FT2Image", &width, &height)) {
        return -1;
    }

    CALL_CPP_INIT("FT2Image", (self->x = new FT2Image(width, height)));

    return 0;
}

static void PyFT2Image_dealloc(PyFT2Image *self)
{
    delete self->x;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char *PyFT2Image_draw_rect__doc__ =
    "draw_rect(x0, y0, x1, y1)\n"
    "\n"
    "Draw a rect to the image.\n"
    "\n";

static PyObject *PyFT2Image_draw_rect(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    double x0, y0, x1, y1;

    if (!PyArg_ParseTuple(args, "dddd:draw_rect", &x0, &y0, &x1, &y1)) {
        return NULL;
    }

    CALL_CPP("draw_rect", (self->x->draw_rect(x0, y0, x1, y1)));

    Py_RETURN_NONE;
}

const char *PyFT2Image_draw_rect_filled__doc__ =
    "draw_rect_filled(x0, y0, x1, y1)\n"
    "\n"
    "Draw a filled rect to the image.\n"
    "\n";

static PyObject *PyFT2Image_draw_rect_filled(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    double x0, y0, x1, y1;

    if (!PyArg_ParseTuple(args, "dddd:draw_rect_filled", &x0, &y0, &x1, &y1)) {
        return NULL;
    }

    CALL_CPP("draw_rect_filled", (self->x->draw_rect_filled(x0, y0, x1, y1)));

    Py_RETURN_NONE;
}

const char *PyFT2Image_as_str__doc__ =
    "s = image.as_str()\n"
    "\n"
    "[*Deprecated*]\n"
    "Return the image buffer as a string\n"
    "\n";

static PyObject *PyFT2Image_as_str(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    if (PyErr_WarnEx(PyExc_FutureWarning,
                     "FT2Image.as_str is deprecated since Matplotlib 3.2 and "
                     "will be removed in Matplotlib 3.4; convert the FT2Image "
                     "to a NumPy array with np.asarray instead.",
                     1)) {
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)self->x->get_buffer(),
                                     self->x->get_width() * self->x->get_height());
}

const char *PyFT2Image_as_rgba_str__doc__ =
    "s = image.as_rgba_str()\n"
    "\n"
    "[*Deprecated*]\n"
    "Return the image buffer as a RGBA string\n"
    "\n";

static PyObject *PyFT2Image_as_rgba_str(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    if (PyErr_WarnEx(PyExc_FutureWarning,
                     "FT2Image.as_rgba_str is deprecated since Matplotlib 3.2 and "
                     "will be removed in Matplotlib 3.4; convert the FT2Image "
                     "to a NumPy array with np.asarray instead.",
                     1)) {
        return NULL;
    }
    npy_intp dims[] = {(npy_intp)self->x->get_height(), (npy_intp)self->x->get_width(), 4 };
    numpy::array_view<unsigned char, 3> result(dims);

    unsigned char *src = self->x->get_buffer();
    unsigned char *end = src + (self->x->get_width() * self->x->get_height());
    unsigned char *dst = result.data();

    while (src != end) {
        *dst++ = 0;
        *dst++ = 0;
        *dst++ = 0;
        *dst++ = *src++;
    }

    return result.pyobj();
}

const char *PyFT2Image_as_array__doc__ =
    "x = image.as_array()\n"
    "\n"
    "[*Deprecated*]\n"
    "Return the image buffer as a width x height numpy array of ubyte \n"
    "\n";

static PyObject *PyFT2Image_as_array(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    if (PyErr_WarnEx(PyExc_FutureWarning,
                     "FT2Image.as_array is deprecated since Matplotlib 3.2 and "
                     "will be removed in Matplotlib 3.4; convert the FT2Image "
                     "to a NumPy array with np.asarray instead.",
                     1)) {
        return NULL;
    }
    npy_intp dims[] = {(npy_intp)self->x->get_height(), (npy_intp)self->x->get_width() };
    return PyArray_SimpleNewFromData(2, dims, NPY_UBYTE, self->x->get_buffer());
}

static PyObject *PyFT2Image_get_width(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    if (PyErr_WarnEx(PyExc_FutureWarning,
                     "FT2Image.get_width is deprecated since Matplotlib 3.2 and "
                     "will be removed in Matplotlib 3.4; convert the FT2Image "
                     "to a NumPy array with np.asarray instead.",
                     1)) {
        return NULL;
    }
    return PyLong_FromLong(self->x->get_width());
}

static PyObject *PyFT2Image_get_height(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    if (PyErr_WarnEx(PyExc_FutureWarning,
                     "FT2Image.get_height is deprecated since Matplotlib 3.2 and "
                     "will be removed in Matplotlib 3.4; convert the FT2Image "
                     "to a NumPy array with np.asarray instead.",
                     1)) {
        return NULL;
    }
    return PyLong_FromLong(self->x->get_height());
}

static int PyFT2Image_get_buffer(PyFT2Image *self, Py_buffer *buf, int flags)
{
    FT2Image *im = self->x;

    Py_INCREF(self);
    buf->obj = (PyObject *)self;
    buf->buf = im->get_buffer();
    buf->len = im->get_width() * im->get_height();
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 2;
    self->shape[0] = im->get_height();
    self->shape[1] = im->get_width();
    buf->shape = self->shape;
    self->strides[0] = im->get_width();
    self->strides[1] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static PyTypeObject PyFT2ImageType;

static PyTypeObject *PyFT2Image_init_type(PyObject *m, PyTypeObject *type)
{
    static PyMethodDef methods[] = {
        {"draw_rect", (PyCFunction)PyFT2Image_draw_rect, METH_VARARGS, PyFT2Image_draw_rect__doc__},
        {"draw_rect_filled", (PyCFunction)PyFT2Image_draw_rect_filled, METH_VARARGS, PyFT2Image_draw_rect_filled__doc__},
        {"as_str", (PyCFunction)PyFT2Image_as_str, METH_NOARGS, PyFT2Image_as_str__doc__},
        {"as_rgba_str", (PyCFunction)PyFT2Image_as_rgba_str, METH_NOARGS, PyFT2Image_as_rgba_str__doc__},
        {"as_array", (PyCFunction)PyFT2Image_as_array, METH_NOARGS, PyFT2Image_as_array__doc__},
        {"get_width", (PyCFunction)PyFT2Image_get_width, METH_NOARGS, NULL},
        {"get_height", (PyCFunction)PyFT2Image_get_height, METH_NOARGS, NULL},
        {NULL}
    };

    static PyBufferProcs buffer_procs;
    memset(&buffer_procs, 0, sizeof(PyBufferProcs));
    buffer_procs.bf_getbuffer = (getbufferproc)PyFT2Image_get_buffer;

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib.ft2font.FT2Image";
    type->tp_basicsize = sizeof(PyFT2Image);
    type->tp_dealloc = (destructor)PyFT2Image_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    type->tp_methods = methods;
    type->tp_new = PyFT2Image_new;
    type->tp_init = (initproc)PyFT2Image_init;
    type->tp_as_buffer = &buffer_procs;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "FT2Image", (PyObject *)type)) {
        return NULL;
    }

    return type;
}

/**********************************************************************
 * Glyph
 * */

typedef struct
{
    PyObject_HEAD
    size_t glyphInd;
    long width;
    long height;
    long horiBearingX;
    long horiBearingY;
    long horiAdvance;
    long linearHoriAdvance;
    long vertBearingX;
    long vertBearingY;
    long vertAdvance;
    FT_BBox bbox;
} PyGlyph;

static PyTypeObject PyGlyphType;

static PyObject *
PyGlyph_new(const FT_Face &face, const FT_Glyph &glyph, size_t ind, long hinting_factor)
{
    PyGlyph *self;
    self = (PyGlyph *)PyGlyphType.tp_alloc(&PyGlyphType, 0);

    self->glyphInd = ind;

    FT_Glyph_Get_CBox(glyph, ft_glyph_bbox_subpixels, &self->bbox);

    self->width = face->glyph->metrics.width / hinting_factor;
    self->height = face->glyph->metrics.height;
    self->horiBearingX = face->glyph->metrics.horiBearingX / hinting_factor;
    self->horiBearingY = face->glyph->metrics.horiBearingY;
    self->horiAdvance = face->glyph->metrics.horiAdvance;
    self->linearHoriAdvance = face->glyph->linearHoriAdvance / hinting_factor;
    self->vertBearingX = face->glyph->metrics.vertBearingX;
    self->vertBearingY = face->glyph->metrics.vertBearingY;
    self->vertAdvance = face->glyph->metrics.vertAdvance;

    return (PyObject *)self;
}

static void PyGlyph_dealloc(PyGlyph *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyGlyph_get_bbox(PyGlyph *self, void *closure)
{
    return Py_BuildValue(
        "llll", self->bbox.xMin, self->bbox.yMin, self->bbox.xMax, self->bbox.yMax);
}

static PyTypeObject *PyGlyph_init_type(PyObject *m, PyTypeObject *type)
{
    static PyMemberDef members[] = {
        {(char *)"width", T_LONG, offsetof(PyGlyph, width), READONLY, (char *)""},
        {(char *)"height", T_LONG, offsetof(PyGlyph, height), READONLY, (char *)""},
        {(char *)"horiBearingX", T_LONG, offsetof(PyGlyph, horiBearingX), READONLY, (char *)""},
        {(char *)"horiBearingY", T_LONG, offsetof(PyGlyph, horiBearingY), READONLY, (char *)""},
        {(char *)"horiAdvance", T_LONG, offsetof(PyGlyph, horiAdvance), READONLY, (char *)""},
        {(char *)"linearHoriAdvance", T_LONG, offsetof(PyGlyph, linearHoriAdvance), READONLY, (char *)""},
        {(char *)"vertBearingX", T_LONG, offsetof(PyGlyph, vertBearingX), READONLY, (char *)""},
        {(char *)"vertBearingY", T_LONG, offsetof(PyGlyph, vertBearingY), READONLY, (char *)""},
        {(char *)"vertAdvance", T_LONG, offsetof(PyGlyph, vertAdvance), READONLY, (char *)""},
        {NULL}
    };

    static PyGetSetDef getset[] = {
        {(char *)"bbox", (getter)PyGlyph_get_bbox, NULL, NULL, NULL},
        {NULL}
    };

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib.ft2font.Glyph";
    type->tp_basicsize = sizeof(PyGlyph);
    type->tp_dealloc = (destructor)PyGlyph_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    type->tp_members = members;
    type->tp_getset = getset;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    /* Don't need to add to module, since you can't create glyphs
       directly from Python */

    return type;
}

/**********************************************************************
 * FT2Font
 * */

typedef struct
{
    PyObject_HEAD
    FT2Font *x;
    PyObject *fname;
    PyObject *py_file;
    FILE *fp;
    int close_file;
    mpl_off_t offset;
    FT_StreamRec stream;
    FT_Byte *mem;
    size_t mem_size;
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];
    Py_ssize_t suboffsets[2];
} PyFT2Font;

static unsigned long read_from_file_callback(FT_Stream stream,
                                             unsigned long offset,
                                             unsigned char *buffer,
                                             unsigned long count)
{

    PyFT2Font *def = (PyFT2Font *)stream->descriptor.pointer;

    if (fseek(def->fp, offset, SEEK_SET) == -1) {
        return 0;
    }

    if (count > 0) {
        return fread(buffer, 1, count, def->fp);
    }

    return 0;
}

static void close_file_callback(FT_Stream stream)
{
    PyFT2Font *def = (PyFT2Font *)stream->descriptor.pointer;

    if (mpl_PyFile_DupClose(def->py_file, def->fp, def->offset)) {
        throw std::runtime_error("Couldn't close file");
    }

    if (def->close_file) {
        mpl_PyFile_CloseFile(def->py_file);
    }

    Py_DECREF(def->py_file);
    def->py_file = NULL;
}

static int convert_open_args(PyFT2Font *self, PyObject *py_file_arg, FT_Open_Args *open_args)
{
    PyObject *py_file = NULL;
    int close_file = 0;
    FILE *fp;
    PyObject *data = NULL;
    char *data_ptr;
    Py_ssize_t data_len;
    long file_size;
    FT_Byte *new_memory;
    mpl_off_t offset = 0;

    int result = 0;

    memset((void *)open_args, 0, sizeof(FT_Open_Args));

    if (PyBytes_Check(py_file_arg) || PyUnicode_Check(py_file_arg)) {
        if ((py_file = mpl_PyFile_OpenFile(py_file_arg, (char *)"rb")) == NULL) {
            goto exit;
        }
        close_file = 1;
    } else {
        Py_INCREF(py_file_arg);
        py_file = py_file_arg;
    }

    if ((fp = mpl_PyFile_Dup(py_file, (char *)"rb", &offset))) {
        Py_INCREF(py_file);
        self->py_file = py_file;
        self->close_file = close_file;
        self->fp = fp;
        self->offset = offset;
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        self->stream.base = NULL;
        self->stream.size = (unsigned long)file_size;
        self->stream.pos = 0;
        self->stream.descriptor.pointer = self;
        self->stream.read = &read_from_file_callback;
        self->stream.close = &close_file_callback;

        open_args->flags = FT_OPEN_STREAM;
        open_args->stream = &self->stream;
    } else {
        if (PyObject_HasAttrString(py_file_arg, "read") &&
            (data = PyObject_CallMethod(py_file_arg, (char *)"read", (char *)""))) {
            if (PyBytes_AsStringAndSize(data, &data_ptr, &data_len)) {
                goto exit;
            }

            if (self->mem) {
                free(self->mem);
            }
            self->mem = (FT_Byte *)malloc((self->mem_size + data_len) * sizeof(FT_Byte));
            if (self->mem == NULL) {
                goto exit;
            }
            new_memory = self->mem + self->mem_size;
            self->mem_size += data_len;

            memcpy(new_memory, data_ptr, data_len);
            open_args->flags = FT_OPEN_MEMORY;
            open_args->memory_base = new_memory;
            open_args->memory_size = data_len;
            open_args->stream = NULL;
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "First argument must be a path or file object reading bytes");
            goto exit;
        }
    }

    result = 1;

exit:

    Py_XDECREF(py_file);
    Py_XDECREF(data);

    return result;
}

static PyTypeObject PyFT2FontType;

static PyObject *PyFT2Font_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFT2Font *self;
    self = (PyFT2Font *)type->tp_alloc(type, 0);
    self->x = NULL;
    self->fname = NULL;
    self->py_file = NULL;
    self->fp = NULL;
    self->close_file = 0;
    self->offset = 0;
    memset(&self->stream, 0, sizeof(FT_StreamRec));
    self->mem = 0;
    self->mem_size = 0;
    return (PyObject *)self;
}

const char *PyFT2Font_init__doc__ =
    "FT2Font(ttffile)\n"
    "\n"
    "Create a new FT2Font object\n"
    "The following global font attributes are defined:\n"
    "  num_faces              number of faces in file\n"
    "  face_flags             face flags  (int type); see the ft2font constants\n"
    "  style_flags            style flags  (int type); see the ft2font constants\n"
    "  num_glyphs             number of glyphs in the face\n"
    "  family_name            face family name\n"
    "  style_name             face style name\n"
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
    "  postscript_name        PostScript name of the font\n";

static void PyFT2Font_fail(PyFT2Font *self)
{
    free(self->mem);
    self->mem = NULL;
    Py_XDECREF(self->py_file);
    self->py_file = NULL;
}

static int PyFT2Font_init(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyObject *fname;
    FT_Open_Args open_args;
    long hinting_factor = 8;
    int kerning_factor = 0;
    const char *names[] = { "filename", "hinting_factor", "_kerning_factor", NULL };

    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "O|l$i:FT2Font", (char **)names, &fname,
             &hinting_factor, &kerning_factor)) {
        return -1;
    }

    if (!convert_open_args(self, fname, &open_args)) {
        return -1;
    }

    CALL_CPP_FULL(
        "FT2Font", (self->x = new FT2Font(open_args, hinting_factor)), PyFT2Font_fail(self), -1);

    CALL_CPP_INIT("FT2Font->set_kerning_factor", (self->x->set_kerning_factor(kerning_factor)));

    Py_INCREF(fname);
    self->fname = fname;

    return 0;
}

static void PyFT2Font_dealloc(PyFT2Font *self)
{
    delete self->x;
    free(self->mem);
    Py_XDECREF(self->py_file);
    Py_XDECREF(self->fname);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char *PyFT2Font_clear__doc__ =
    "clear()\n"
    "\n"
    "Clear all the glyphs, reset for a new set_text";

static PyObject *PyFT2Font_clear(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    CALL_CPP("clear", (self->x->clear()));

    Py_RETURN_NONE;
}

const char *PyFT2Font_set_size__doc__ =
    "set_size(ptsize, dpi)\n"
    "\n"
    "Set the point size and dpi of the text.\n";

static PyObject *PyFT2Font_set_size(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    double ptsize;
    double dpi;

    if (!PyArg_ParseTuple(args, "dd:set_size", &ptsize, &dpi)) {
        return NULL;
    }

    CALL_CPP("set_size", (self->x->set_size(ptsize, dpi)));

    Py_RETURN_NONE;
}

const char *PyFT2Font_set_charmap__doc__ =
    "set_charmap(i)\n"
    "\n"
    "Make the i-th charmap current\n";

static PyObject *PyFT2Font_set_charmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    int i;

    if (!PyArg_ParseTuple(args, "i:set_charmap", &i)) {
        return NULL;
    }

    CALL_CPP("set_charmap", (self->x->set_charmap(i)));

    Py_RETURN_NONE;
}

const char *PyFT2Font_select_charmap__doc__ =
    "select_charmap(i)\n"
    "\n"
    "select charmap i where i is one of the FT_Encoding number\n";

static PyObject *PyFT2Font_select_charmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    unsigned long i;

    if (!PyArg_ParseTuple(args, "k:select_charmap", &i)) {
        return NULL;
    }

    CALL_CPP("select_charmap", self->x->select_charmap(i));

    Py_RETURN_NONE;
}

const char *PyFT2Font_get_kerning__doc__ =
    "dx = get_kerning(left, right, mode)\n"
    "\n"
    "Get the kerning between left char and right glyph indices\n"
    "mode is a kerning mode constant\n"
    "  KERNING_DEFAULT  - Return scaled and grid-fitted kerning distances\n"
    "  KERNING_UNFITTED - Return scaled but un-grid-fitted kerning distances\n"
    "  KERNING_UNSCALED - Return the kerning vector in original font units\n";

static PyObject *PyFT2Font_get_kerning(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    FT_UInt left, right, mode;
    int result;

    if (!PyArg_ParseTuple(args, "III:get_kerning", &left, &right, &mode)) {
        return NULL;
    }

    CALL_CPP("get_kerning", (result = self->x->get_kerning(left, right, mode)));

    return PyLong_FromLong(result);
}

const char *PyFT2Font_set_text__doc__ =
    "set_text(s, angle)\n"
    "\n"
    "Set the text string and angle.\n"
    "You must call this before draw_glyphs_to_bitmap\n"
    "A sequence of x,y positions is returned";

static PyObject *PyFT2Font_set_text(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyObject *textobj;
    double angle = 0.0;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    std::vector<double> xys;
    const char *names[] = { "string", "angle", "flags", NULL };

    /* This makes a technically incorrect assumption that FT_Int32 is
       int. In theory it can also be long, if the size of int is less
       than 32 bits. This is very unlikely on modern platforms. */
    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "O|di:set_text", (char **)names, &textobj, &angle, &flags)) {
        return NULL;
    }

    std::vector<uint32_t> codepoints;
    size_t size;

    if (PyUnicode_Check(textobj)) {
        size = PyUnicode_GET_SIZE(textobj);
        codepoints.resize(size);
        Py_UNICODE *unistr = PyUnicode_AsUnicode(textobj);
        for (size_t i = 0; i < size; ++i) {
            codepoints[i] = unistr[i];
        }
    } else if (PyBytes_Check(textobj)) {
        size = PyBytes_Size(textobj);
        codepoints.resize(size);
        char *bytestr = PyBytes_AsString(textobj);
        for (size_t i = 0; i < size; ++i) {
            codepoints[i] = bytestr[i];
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "String must be str or bytes");
        return NULL;
    }

    uint32_t* codepoints_array = NULL;
    if (size > 0) {
        codepoints_array = &codepoints[0];
    }
    CALL_CPP("set_text", self->x->set_text(size, codepoints_array, angle, flags, xys));

    return convert_xys_to_array(xys);
}

const char *PyFT2Font_get_num_glyphs__doc__ =
    "get_num_glyphs()\n"
    "\n"
    "Return the number of loaded glyphs\n";

static PyObject *PyFT2Font_get_num_glyphs(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromLong(self->x->get_num_glyphs());
}

const char *PyFT2Font_load_char__doc__ =
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
    "  vertAdvance    # advance height for vertical layout\n";

static PyObject *PyFT2Font_load_char(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    long charcode;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    const char *names[] = { "charcode", "flags", NULL };

    /* This makes a technically incorrect assumption that FT_Int32 is
       int. In theory it can also be long, if the size of int is less
       than 32 bits. This is very unlikely on modern platforms. */
    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "l|i:load_char", (char **)names, &charcode, &flags)) {
        return NULL;
    }

    CALL_CPP("load_char", (self->x->load_char(charcode, flags)));

    return PyGlyph_new(self->x->get_face(),
                       self->x->get_last_glyph(),
                       self->x->get_last_glyph_index(),
                       self->x->get_hinting_factor());
}

const char *PyFT2Font_load_glyph__doc__ =
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
    "  vertAdvance    # advance height for vertical layout\n";

static PyObject *PyFT2Font_load_glyph(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    FT_UInt glyph_index;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    const char *names[] = { "glyph_index", "flags", NULL };

    /* This makes a technically incorrect assumption that FT_Int32 is
       int. In theory it can also be long, if the size of int is less
       than 32 bits. This is very unlikely on modern platforms. */
    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "I|i:load_glyph", (char **)names, &glyph_index, &flags)) {
        return NULL;
    }

    CALL_CPP("load_glyph", (self->x->load_glyph(glyph_index, flags)));

    return PyGlyph_new(self->x->get_face(),
                       self->x->get_last_glyph(),
                       self->x->get_last_glyph_index(),
                       self->x->get_hinting_factor());
}

const char *PyFT2Font_get_width_height__doc__ =
    "w, h = get_width_height()\n"
    "\n"
    "Get the width and height in 26.6 subpixels of the current string set by set_text\n"
    "The rotation of the string is accounted for.  To get width and height\n"
    "in pixels, divide these values by 64\n";

static PyObject *PyFT2Font_get_width_height(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    long width, height;

    CALL_CPP("get_width_height", (self->x->get_width_height(&width, &height)));

    return Py_BuildValue("ll", width, height);
}

const char *PyFT2Font_get_bitmap_offset__doc__ =
    "x, y = get_bitmap_offset()\n"
    "\n"
    "Get the offset in 26.6 subpixels for the bitmap if ink hangs left or below (0, 0).\n"
    "Since matplotlib only supports left-to-right text, y is always 0.\n";

static PyObject *PyFT2Font_get_bitmap_offset(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    long x, y;

    CALL_CPP("get_bitmap_offset", (self->x->get_bitmap_offset(&x, &y)));

    return Py_BuildValue("ll", x, y);
}

const char *PyFT2Font_get_descent__doc__ =
    "d = get_descent()\n"
    "\n"
    "Get the descent of the current string set by set_text in 26.6 subpixels.\n"
    "The rotation of the string is accounted for.  To get the descent\n"
    "in pixels, divide this value by 64.\n";

static PyObject *PyFT2Font_get_descent(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    long descent;

    CALL_CPP("get_descent", (descent = self->x->get_descent()));

    return PyLong_FromLong(descent);
}

const char *PyFT2Font_draw_glyphs_to_bitmap__doc__ =
    "draw_glyphs_to_bitmap()\n"
    "\n"
    "Draw the glyphs that were loaded by set_text to the bitmap\n"
    "The bitmap size will be automatically set to include the glyphs\n";

static PyObject *PyFT2Font_draw_glyphs_to_bitmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    bool antialiased = true;
    const char *names[] = { "antialiased", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&:draw_glyphs_to_bitmap",
                                     (char **)names, &convert_bool, &antialiased)) {
        return NULL;
    }

    CALL_CPP("draw_glyphs_to_bitmap", (self->x->draw_glyphs_to_bitmap(antialiased)));

    Py_RETURN_NONE;
}

const char *PyFT2Font_get_xys__doc__ =
    "get_xys()\n"
    "\n"
    "Get the xy locations of the current glyphs\n";

static PyObject *PyFT2Font_get_xys(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    bool antialiased = true;
    std::vector<double> xys;
    const char *names[] = { "antialiased", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&:get_xys",
                                     (char **)names, &convert_bool, &antialiased)) {
        return NULL;
    }

    CALL_CPP("get_xys", (self->x->get_xys(antialiased, xys)));

    return convert_xys_to_array(xys);
}

const char *PyFT2Font_draw_glyph_to_bitmap__doc__ =
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

static PyObject *PyFT2Font_draw_glyph_to_bitmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyFT2Image *image;
    double xd, yd;
    PyGlyph *glyph;
    bool antialiased = true;
    const char *names[] = { "image", "x", "y", "glyph", "antialiased", NULL };

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O!ddO!|O&:draw_glyph_to_bitmap",
                                     (char **)names,
                                     &PyFT2ImageType,
                                     &image,
                                     &xd,
                                     &yd,
                                     &PyGlyphType,
                                     &glyph,
                                     &convert_bool,
                                     &antialiased)) {
        return NULL;
    }

    CALL_CPP("draw_glyph_to_bitmap",
             self->x->draw_glyph_to_bitmap(*(image->x), xd, yd, glyph->glyphInd, antialiased));

    Py_RETURN_NONE;
}

const char *PyFT2Font_get_glyph_name__doc__ =
    "get_glyph_name(index)\n"
    "\n"
    "Retrieves the ASCII name of a given glyph in a face.\n"
    "\n"
    "Due to Matplotlib's internal design, for fonts that do not contain glyph \n"
    "names (per FT_FACE_FLAG_GLYPH_NAMES), this returns a made-up name which \n"
    "does *not* roundtrip through `.get_name_index`.\n";

static PyObject *PyFT2Font_get_glyph_name(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    unsigned int glyph_number;
    char buffer[128];
    if (!PyArg_ParseTuple(args, "I:get_glyph_name", &glyph_number)) {
        return NULL;
    }
    CALL_CPP("get_glyph_name", (self->x->get_glyph_name(glyph_number, buffer)));
    return PyUnicode_FromString(buffer);
}

const char *PyFT2Font_get_charmap__doc__ =
    "get_charmap()\n"
    "\n"
    "Returns a dictionary that maps the character codes of the selected charmap\n"
    "(Unicode by default) to their corresponding glyph indices.\n";

static PyObject *PyFT2Font_get_charmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyObject *charmap;
    if (!(charmap = PyDict_New())) {
        return NULL;
    }
    FT_UInt index;
    FT_ULong code = FT_Get_First_Char(self->x->get_face(), &index);
    while (index != 0) {
        PyObject *key = NULL, *val = NULL;
        bool error = (!(key = PyLong_FromLong(code))
                      || !(val = PyLong_FromLong(index))
                      || (PyDict_SetItem(charmap, key, val) == -1));
        Py_XDECREF(key);
        Py_XDECREF(val);
        if (error) {
            Py_DECREF(charmap);
            return NULL;
        }
        code = FT_Get_Next_Char(self->x->get_face(), code, &index);
    }
    return charmap;
}


const char *PyFT2Font_get_char_index__doc__ =
    "get_char_index()\n"
    "\n"
    "Given a character code, returns a glyph index.\n";

static PyObject *PyFT2Font_get_char_index(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    FT_UInt index;
    FT_ULong ccode;

    if (!PyArg_ParseTuple(args, "k:get_char_index", &ccode)) {
        return NULL;
    }

    index = FT_Get_Char_Index(self->x->get_face(), ccode);

    return PyLong_FromLong(index);
}


const char *PyFT2Font_get_sfnt__doc__ =
    "get_sfnt(name)\n"
    "\n"
    "Get all values from the SFNT names table.  Result is a dictionary whose "
    "key is the platform-ID, ISO-encoding-scheme, language-code, and "
    "description.\n";

static PyObject *PyFT2Font_get_sfnt(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyObject *names;

    if (!(self->x->get_face()->face_flags & FT_FACE_FLAG_SFNT)) {
        PyErr_SetString(PyExc_ValueError, "No SFNT name table");
        return NULL;
    }

    size_t count = FT_Get_Sfnt_Name_Count(self->x->get_face());

    names = PyDict_New();
    if (names == NULL) {
        return NULL;
    }

    for (FT_UInt j = 0; j < count; ++j) {
        FT_SfntName sfnt;
        FT_Error error = FT_Get_Sfnt_Name(self->x->get_face(), j, &sfnt);

        if (error) {
            Py_DECREF(names);
            PyErr_SetString(PyExc_ValueError, "Could not get SFNT name");
            return NULL;
        }

        PyObject *key = Py_BuildValue(
            "HHHH", sfnt.platform_id, sfnt.encoding_id, sfnt.language_id, sfnt.name_id);
        if (key == NULL) {
            Py_DECREF(names);
            return NULL;
        }

        PyObject *val = PyBytes_FromStringAndSize((const char *)sfnt.string, sfnt.string_len);
        if (val == NULL) {
            Py_DECREF(key);
            Py_DECREF(names);
            return NULL;
        }

        if (PyDict_SetItem(names, key, val)) {
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(names);
            return NULL;
        }

        Py_DECREF(key);
        Py_DECREF(val);
    }

    return names;
}

const char *PyFT2Font_get_name_index__doc__ =
    "get_name_index(name)\n"
    "\n"
    "Returns the glyph index of a given glyph name.\n"
    "The glyph index 0 means `undefined character code'.\n";

static PyObject *PyFT2Font_get_name_index(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    char *glyphname;
    long name_index;
    if (!PyArg_ParseTuple(args, "s:get_name_index", &glyphname)) {
        return NULL;
    }
    CALL_CPP("get_name_index", name_index = self->x->get_name_index(glyphname));
    return PyLong_FromLong(name_index);
}

const char *PyFT2Font_get_ps_font_info__doc__ =
    "get_ps_font_info()\n"
    "\n"
    "Return the information in the PS Font Info structure.\n";

static PyObject *PyFT2Font_get_ps_font_info(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PS_FontInfoRec fontinfo;

    FT_Error error = FT_Get_PS_Font_Info(self->x->get_face(), &fontinfo);
    if (error) {
        PyErr_SetString(PyExc_ValueError, "Could not get PS font info");
        return NULL;
    }

    return Py_BuildValue("ssssslbhH",
                         fontinfo.version ? fontinfo.version : "",
                         fontinfo.notice ? fontinfo.notice : "",
                         fontinfo.full_name ? fontinfo.full_name : "",
                         fontinfo.family_name ? fontinfo.family_name : "",
                         fontinfo.weight ? fontinfo.weight : "",
                         fontinfo.italic_angle,
                         fontinfo.is_fixed_pitch,
                         fontinfo.underline_position,
                         fontinfo.underline_thickness);
}

const char *PyFT2Font_get_sfnt_table__doc__ =
    "get_sfnt_table(name)\n"
    "\n"
    "Return one of the following SFNT tables: head, maxp, OS/2, hhea, "
    "vhea, post, or pclt.\n";

static PyObject *PyFT2Font_get_sfnt_table(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    char *tagname;
    if (!PyArg_ParseTuple(args, "s:get_sfnt_table", &tagname)) {
        return NULL;
    }

    int tag;
    const char *tags[] = { "head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt", NULL };

    for (tag = 0; tags[tag] != NULL; tag++) {
        if (strncmp(tagname, tags[tag], 5) == 0) {
            break;
        }
    }

    void *table = FT_Get_Sfnt_Table(self->x->get_face(), (FT_Sfnt_Tag)tag);
    if (!table) {
        Py_RETURN_NONE;
    }

    switch (tag) {
    case 0: {
        char head_dict[] =
            "{s:(h,H), s:(h,H), s:l, s:l, s:H, s:H,"
            "s:(l,l), s:(l,l), s:h, s:h, s:h, s:h, s:H, s:H, s:h, s:h, s:h}";
        TT_Header *t = (TT_Header *)table;
        return Py_BuildValue(head_dict,
                             "version",
                             FIXED_MAJOR(t->Table_Version),
                             FIXED_MINOR(t->Table_Version),
                             "fontRevision",
                             FIXED_MAJOR(t->Font_Revision),
                             FIXED_MINOR(t->Font_Revision),
                             "checkSumAdjustment",
                             t->CheckSum_Adjust,
                             "magicNumber",
                             t->Magic_Number,
                             "flags",
                             t->Flags,
                             "unitsPerEm",
                             t->Units_Per_EM,
                             "created",
                             t->Created[0],
                             t->Created[1],
                             "modified",
                             t->Modified[0],
                             t->Modified[1],
                             "xMin",
                             t->xMin,
                             "yMin",
                             t->yMin,
                             "xMax",
                             t->xMax,
                             "yMax",
                             t->yMax,
                             "macStyle",
                             t->Mac_Style,
                             "lowestRecPPEM",
                             t->Lowest_Rec_PPEM,
                             "fontDirectionHint",
                             t->Font_Direction,
                             "indexToLocFormat",
                             t->Index_To_Loc_Format,
                             "glyphDataFormat",
                             t->Glyph_Data_Format);
    }
    case 1: {
        char maxp_dict[] =
            "{s:(h,H), s:H, s:H, s:H, s:H, s:H, s:H,"
            "s:H, s:H, s:H, s:H, s:H, s:H, s:H, s:H}";
        TT_MaxProfile *t = (TT_MaxProfile *)table;
        return Py_BuildValue(maxp_dict,
                             "version",
                             FIXED_MAJOR(t->version),
                             FIXED_MINOR(t->version),
                             "numGlyphs",
                             t->numGlyphs,
                             "maxPoints",
                             t->maxPoints,
                             "maxContours",
                             t->maxContours,
                             "maxComponentPoints",
                             t->maxCompositePoints,
                             "maxComponentContours",
                             t->maxCompositeContours,
                             "maxZones",
                             t->maxZones,
                             "maxTwilightPoints",
                             t->maxTwilightPoints,
                             "maxStorage",
                             t->maxStorage,
                             "maxFunctionDefs",
                             t->maxFunctionDefs,
                             "maxInstructionDefs",
                             t->maxInstructionDefs,
                             "maxStackElements",
                             t->maxStackElements,
                             "maxSizeOfInstructions",
                             t->maxSizeOfInstructions,
                             "maxComponentElements",
                             t->maxComponentElements,
                             "maxComponentDepth",
                             t->maxComponentDepth);
    }
    case 2: {
        char os_2_dict[] =
            "{s:H, s:h, s:H, s:H, s:H, s:h, s:h, s:h,"
            "s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:y#, s:(kkkk),"
            "s:y#, s:H, s:H, s:H}";
        TT_OS2 *t = (TT_OS2 *)table;
        return Py_BuildValue(os_2_dict,
                             "version",
                             t->version,
                             "xAvgCharWidth",
                             t->xAvgCharWidth,
                             "usWeightClass",
                             t->usWeightClass,
                             "usWidthClass",
                             t->usWidthClass,
                             "fsType",
                             t->fsType,
                             "ySubscriptXSize",
                             t->ySubscriptXSize,
                             "ySubscriptYSize",
                             t->ySubscriptYSize,
                             "ySubscriptXOffset",
                             t->ySubscriptXOffset,
                             "ySubscriptYOffset",
                             t->ySubscriptYOffset,
                             "ySuperscriptXSize",
                             t->ySuperscriptXSize,
                             "ySuperscriptYSize",
                             t->ySuperscriptYSize,
                             "ySuperscriptXOffset",
                             t->ySuperscriptXOffset,
                             "ySuperscriptYOffset",
                             t->ySuperscriptYOffset,
                             "yStrikeoutSize",
                             t->yStrikeoutSize,
                             "yStrikeoutPosition",
                             t->yStrikeoutPosition,
                             "sFamilyClass",
                             t->sFamilyClass,
                             "panose",
                             t->panose,
                             Py_ssize_t(10),
                             "ulCharRange",
                             t->ulUnicodeRange1,
                             t->ulUnicodeRange2,
                             t->ulUnicodeRange3,
                             t->ulUnicodeRange4,
                             "achVendID",
                             t->achVendID,
                             Py_ssize_t(4),
                             "fsSelection",
                             t->fsSelection,
                             "fsFirstCharIndex",
                             t->usFirstCharIndex,
                             "fsLastCharIndex",
                             t->usLastCharIndex);
    }
    case 3: {
        char hhea_dict[] =
            "{s:(h,H), s:h, s:h, s:h, s:H, s:h, s:h, s:h,"
            "s:h, s:h, s:h, s:h, s:H}";
        TT_HoriHeader *t = (TT_HoriHeader *)table;
        return Py_BuildValue(hhea_dict,
                             "version",
                             FIXED_MAJOR(t->Version),
                             FIXED_MINOR(t->Version),
                             "ascent",
                             t->Ascender,
                             "descent",
                             t->Descender,
                             "lineGap",
                             t->Line_Gap,
                             "advanceWidthMax",
                             t->advance_Width_Max,
                             "minLeftBearing",
                             t->min_Left_Side_Bearing,
                             "minRightBearing",
                             t->min_Right_Side_Bearing,
                             "xMaxExtent",
                             t->xMax_Extent,
                             "caretSlopeRise",
                             t->caret_Slope_Rise,
                             "caretSlopeRun",
                             t->caret_Slope_Run,
                             "caretOffset",
                             t->caret_Offset,
                             "metricDataFormat",
                             t->metric_Data_Format,
                             "numOfLongHorMetrics",
                             t->number_Of_HMetrics);
    }
    case 4: {
        char vhea_dict[] =
            "{s:(h,H), s:h, s:h, s:h, s:H, s:h, s:h, s:h,"
            "s:h, s:h, s:h, s:h, s:H}";
        TT_VertHeader *t = (TT_VertHeader *)table;
        return Py_BuildValue(vhea_dict,
                             "version",
                             FIXED_MAJOR(t->Version),
                             FIXED_MINOR(t->Version),
                             "vertTypoAscender",
                             t->Ascender,
                             "vertTypoDescender",
                             t->Descender,
                             "vertTypoLineGap",
                             t->Line_Gap,
                             "advanceHeightMax",
                             t->advance_Height_Max,
                             "minTopSideBearing",
                             t->min_Top_Side_Bearing,
                             "minBottomSizeBearing",
                             t->min_Bottom_Side_Bearing,
                             "yMaxExtent",
                             t->yMax_Extent,
                             "caretSlopeRise",
                             t->caret_Slope_Rise,
                             "caretSlopeRun",
                             t->caret_Slope_Run,
                             "caretOffset",
                             t->caret_Offset,
                             "metricDataFormat",
                             t->metric_Data_Format,
                             "numOfLongVerMetrics",
                             t->number_Of_VMetrics);
    }
    case 5: {
        char post_dict[] = "{s:(h,H), s:(h,H), s:h, s:h, s:k, s:k, s:k, s:k, s:k}";
        TT_Postscript *t = (TT_Postscript *)table;
        return Py_BuildValue(post_dict,
                             "format",
                             FIXED_MAJOR(t->FormatType),
                             FIXED_MINOR(t->FormatType),
                             "italicAngle",
                             FIXED_MAJOR(t->italicAngle),
                             FIXED_MINOR(t->italicAngle),
                             "underlinePosition",
                             t->underlinePosition,
                             "underlineThickness",
                             t->underlineThickness,
                             "isFixedPitch",
                             t->isFixedPitch,
                             "minMemType42",
                             t->minMemType42,
                             "maxMemType42",
                             t->maxMemType42,
                             "minMemType1",
                             t->minMemType1,
                             "maxMemType1",
                             t->maxMemType1);
    }
    case 6: {
        char pclt_dict[] =
            "{s:(h,H), s:k, s:H, s:H, s:H, s:H, s:H, s:H, s:y#, s:y#, s:b, "
            "s:b, s:b}";
        TT_PCLT *t = (TT_PCLT *)table;
        return Py_BuildValue(pclt_dict,
                             "version",
                             FIXED_MAJOR(t->Version),
                             FIXED_MINOR(t->Version),
                             "fontNumber",
                             t->FontNumber,
                             "pitch",
                             t->Pitch,
                             "xHeight",
                             t->xHeight,
                             "style",
                             t->Style,
                             "typeFamily",
                             t->TypeFamily,
                             "capHeight",
                             t->CapHeight,
                             "symbolSet",
                             t->SymbolSet,
                             "typeFace",
                             t->TypeFace,
                             Py_ssize_t(16),
                             "characterComplement",
                             t->CharacterComplement,
                             Py_ssize_t(8),
                             "strokeWeight",
                             t->StrokeWeight,
                             "widthType",
                             t->WidthType,
                             "serifStyle",
                             t->SerifStyle);
    }
    default:
        Py_RETURN_NONE;
    }
}

const char *PyFT2Font_get_path__doc__ =
    "get_path()\n"
    "\n"
    "Get the path data from the currently loaded glyph as a tuple of vertices, "
    "codes.\n";

static PyObject *PyFT2Font_get_path(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    int count;

    CALL_CPP("get_path", (count = self->x->get_path_count()));

    npy_intp vertices_dims[2] = { count, 2 };
    numpy::array_view<double, 2> vertices(vertices_dims);

    npy_intp codes_dims[1] = { count };
    numpy::array_view<unsigned char, 1> codes(codes_dims);

    self->x->get_path(vertices.data(), codes.data());

    return Py_BuildValue("NN", vertices.pyobj(), codes.pyobj());
}

const char *PyFT2Font_get_image__doc__ =
    "get_image()\n"
    "\n"
    "Returns the underlying image buffer for this font object.\n";

static PyObject *PyFT2Font_get_image(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    FT2Image &im = self->x->get_image();
    npy_intp dims[] = {(npy_intp)im.get_height(), (npy_intp)im.get_width() };
    return PyArray_SimpleNewFromData(2, dims, NPY_UBYTE, im.get_buffer());
}

static PyObject *PyFT2Font_postscript_name(PyFT2Font *self, void *closure)
{
    const char *ps_name = FT_Get_Postscript_Name(self->x->get_face());
    if (ps_name == NULL) {
        ps_name = "UNAVAILABLE";
    }

    return PyUnicode_FromString(ps_name);
}

static PyObject *PyFT2Font_num_faces(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_faces);
}

static PyObject *PyFT2Font_family_name(PyFT2Font *self, void *closure)
{
    const char *name = self->x->get_face()->family_name;
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return PyUnicode_FromString(name);
}

static PyObject *PyFT2Font_style_name(PyFT2Font *self, void *closure)
{
    const char *name = self->x->get_face()->style_name;
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return PyUnicode_FromString(name);
}

static PyObject *PyFT2Font_face_flags(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->face_flags);
}

static PyObject *PyFT2Font_style_flags(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->style_flags);
}

static PyObject *PyFT2Font_num_glyphs(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_glyphs);
}

static PyObject *PyFT2Font_num_fixed_sizes(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_fixed_sizes);
}

static PyObject *PyFT2Font_num_charmaps(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_charmaps);
}

static PyObject *PyFT2Font_scalable(PyFT2Font *self, void *closure)
{
    if (FT_IS_SCALABLE(self->x->get_face())) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *PyFT2Font_units_per_EM(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->units_per_EM);
}

static PyObject *PyFT2Font_get_bbox(PyFT2Font *self, void *closure)
{
    FT_BBox *bbox = &(self->x->get_face()->bbox);

    return Py_BuildValue("llll",
                         bbox->xMin, bbox->yMin, bbox->xMax, bbox->yMax);
}

static PyObject *PyFT2Font_ascender(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->ascender);
}

static PyObject *PyFT2Font_descender(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->descender);
}

static PyObject *PyFT2Font_height(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->height);
}

static PyObject *PyFT2Font_max_advance_width(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->max_advance_width);
}

static PyObject *PyFT2Font_max_advance_height(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->max_advance_height);
}

static PyObject *PyFT2Font_underline_position(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->underline_position);
}

static PyObject *PyFT2Font_underline_thickness(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->underline_thickness);
}

static PyObject *PyFT2Font_fname(PyFT2Font *self, void *closure)
{
    if (self->fname) {
        Py_INCREF(self->fname);
        return self->fname;
    }

    Py_RETURN_NONE;
}

static int PyFT2Font_get_buffer(PyFT2Font *self, Py_buffer *buf, int flags)
{
    FT2Image &im = self->x->get_image();

    Py_INCREF(self);
    buf->obj = (PyObject *)self;
    buf->buf = im.get_buffer();
    buf->len = im.get_width() * im.get_height();
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 2;
    self->shape[0] = im.get_height();
    self->shape[1] = im.get_width();
    buf->shape = self->shape;
    self->strides[0] = im.get_width();
    self->strides[1] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static PyTypeObject *PyFT2Font_init_type(PyObject *m, PyTypeObject *type)
{
    static PyGetSetDef getset[] = {
        {(char *)"postscript_name", (getter)PyFT2Font_postscript_name, NULL, NULL, NULL},
        {(char *)"num_faces", (getter)PyFT2Font_num_faces, NULL, NULL, NULL},
        {(char *)"family_name", (getter)PyFT2Font_family_name, NULL, NULL, NULL},
        {(char *)"style_name", (getter)PyFT2Font_style_name, NULL, NULL, NULL},
        {(char *)"face_flags", (getter)PyFT2Font_face_flags, NULL, NULL, NULL},
        {(char *)"style_flags", (getter)PyFT2Font_style_flags, NULL, NULL, NULL},
        {(char *)"num_glyphs", (getter)PyFT2Font_num_glyphs, NULL, NULL, NULL},
        {(char *)"num_fixed_sizes", (getter)PyFT2Font_num_fixed_sizes, NULL, NULL, NULL},
        {(char *)"num_charmaps", (getter)PyFT2Font_num_charmaps, NULL, NULL, NULL},
        {(char *)"scalable", (getter)PyFT2Font_scalable, NULL, NULL, NULL},
        {(char *)"units_per_EM", (getter)PyFT2Font_units_per_EM, NULL, NULL, NULL},
        {(char *)"bbox", (getter)PyFT2Font_get_bbox, NULL, NULL, NULL},
        {(char *)"ascender", (getter)PyFT2Font_ascender, NULL, NULL, NULL},
        {(char *)"descender", (getter)PyFT2Font_descender, NULL, NULL, NULL},
        {(char *)"height", (getter)PyFT2Font_height, NULL, NULL, NULL},
        {(char *)"max_advance_width", (getter)PyFT2Font_max_advance_width, NULL, NULL, NULL},
        {(char *)"max_advance_height", (getter)PyFT2Font_max_advance_height, NULL, NULL, NULL},
        {(char *)"underline_position", (getter)PyFT2Font_underline_position, NULL, NULL, NULL},
        {(char *)"underline_thickness", (getter)PyFT2Font_underline_thickness, NULL, NULL, NULL},
        {(char *)"fname", (getter)PyFT2Font_fname, NULL, NULL, NULL},
        {NULL}
    };

    static PyMethodDef methods[] = {
        {"clear", (PyCFunction)PyFT2Font_clear, METH_NOARGS, PyFT2Font_clear__doc__},
        {"set_size", (PyCFunction)PyFT2Font_set_size, METH_VARARGS, PyFT2Font_set_size__doc__},
        {"set_charmap", (PyCFunction)PyFT2Font_set_charmap, METH_VARARGS, PyFT2Font_set_charmap__doc__},
        {"select_charmap", (PyCFunction)PyFT2Font_select_charmap, METH_VARARGS, PyFT2Font_select_charmap__doc__},
        {"get_kerning", (PyCFunction)PyFT2Font_get_kerning, METH_VARARGS, PyFT2Font_get_kerning__doc__},
        {"set_text", (PyCFunction)PyFT2Font_set_text, METH_VARARGS|METH_KEYWORDS, PyFT2Font_set_text__doc__},
        {"get_num_glyphs", (PyCFunction)PyFT2Font_get_num_glyphs, METH_NOARGS, PyFT2Font_get_num_glyphs__doc__},
        {"load_char", (PyCFunction)PyFT2Font_load_char, METH_VARARGS|METH_KEYWORDS, PyFT2Font_load_char__doc__},
        {"load_glyph", (PyCFunction)PyFT2Font_load_glyph, METH_VARARGS|METH_KEYWORDS, PyFT2Font_load_glyph__doc__},
        {"get_width_height", (PyCFunction)PyFT2Font_get_width_height, METH_NOARGS, PyFT2Font_get_width_height__doc__},
        {"get_bitmap_offset", (PyCFunction)PyFT2Font_get_bitmap_offset, METH_NOARGS, PyFT2Font_get_bitmap_offset__doc__},
        {"get_descent", (PyCFunction)PyFT2Font_get_descent, METH_NOARGS, PyFT2Font_get_descent__doc__},
        {"draw_glyphs_to_bitmap", (PyCFunction)PyFT2Font_draw_glyphs_to_bitmap, METH_VARARGS|METH_KEYWORDS, PyFT2Font_draw_glyphs_to_bitmap__doc__},
        {"get_xys", (PyCFunction)PyFT2Font_get_xys, METH_VARARGS|METH_KEYWORDS, PyFT2Font_get_xys__doc__},
        {"draw_glyph_to_bitmap", (PyCFunction)PyFT2Font_draw_glyph_to_bitmap, METH_VARARGS|METH_KEYWORDS, PyFT2Font_draw_glyph_to_bitmap__doc__},
        {"get_glyph_name", (PyCFunction)PyFT2Font_get_glyph_name, METH_VARARGS, PyFT2Font_get_glyph_name__doc__},
        {"get_charmap", (PyCFunction)PyFT2Font_get_charmap, METH_NOARGS, PyFT2Font_get_charmap__doc__},
        {"get_char_index", (PyCFunction)PyFT2Font_get_char_index, METH_VARARGS, PyFT2Font_get_char_index__doc__},
        {"get_sfnt", (PyCFunction)PyFT2Font_get_sfnt, METH_NOARGS, PyFT2Font_get_sfnt__doc__},
        {"get_name_index", (PyCFunction)PyFT2Font_get_name_index, METH_VARARGS, PyFT2Font_get_name_index__doc__},
        {"get_ps_font_info", (PyCFunction)PyFT2Font_get_ps_font_info, METH_NOARGS, PyFT2Font_get_ps_font_info__doc__},
        {"get_sfnt_table", (PyCFunction)PyFT2Font_get_sfnt_table, METH_VARARGS, PyFT2Font_get_sfnt_table__doc__},
        {"get_path", (PyCFunction)PyFT2Font_get_path, METH_NOARGS, PyFT2Font_get_path__doc__},
        {"get_image", (PyCFunction)PyFT2Font_get_image, METH_NOARGS, PyFT2Font_get_path__doc__},
        {NULL}
    };

    static PyBufferProcs buffer_procs;
    memset(&buffer_procs, 0, sizeof(PyBufferProcs));
    buffer_procs.bf_getbuffer = (getbufferproc)PyFT2Font_get_buffer;

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib.ft2font.FT2Font";
    type->tp_doc = PyFT2Font_init__doc__;
    type->tp_basicsize = sizeof(PyFT2Font);
    type->tp_dealloc = (destructor)PyFT2Font_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    type->tp_methods = methods;
    type->tp_getset = getset;
    type->tp_new = PyFT2Font_new;
    type->tp_init = (initproc)PyFT2Font_init;
    type->tp_as_buffer = &buffer_procs;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "FT2Font", (PyObject *)type)) {
        return NULL;
    }

    return type;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ft2font",
    NULL,
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_ft2font(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);

    if (m == NULL) {
        return NULL;
    }

    if (!PyFT2Image_init_type(m, &PyFT2ImageType)) {
        return NULL;
    }

    if (!PyGlyph_init_type(m, &PyGlyphType)) {
        return NULL;
    }

    if (!PyFT2Font_init_type(m, &PyFT2FontType)) {
        return NULL;
    }

    PyObject *d = PyModule_GetDict(m);

    if (add_dict_int(d, "SCALABLE", FT_FACE_FLAG_SCALABLE) ||
        add_dict_int(d, "FIXED_SIZES", FT_FACE_FLAG_FIXED_SIZES) ||
        add_dict_int(d, "FIXED_WIDTH", FT_FACE_FLAG_FIXED_WIDTH) ||
        add_dict_int(d, "SFNT", FT_FACE_FLAG_SFNT) ||
        add_dict_int(d, "HORIZONTAL", FT_FACE_FLAG_HORIZONTAL) ||
        add_dict_int(d, "VERTICAL", FT_FACE_FLAG_VERTICAL) ||
        add_dict_int(d, "KERNING", FT_FACE_FLAG_KERNING) ||
        add_dict_int(d, "FAST_GLYPHS", FT_FACE_FLAG_FAST_GLYPHS) ||
        add_dict_int(d, "MULTIPLE_MASTERS", FT_FACE_FLAG_MULTIPLE_MASTERS) ||
        add_dict_int(d, "GLYPH_NAMES", FT_FACE_FLAG_GLYPH_NAMES) ||
        add_dict_int(d, "EXTERNAL_STREAM", FT_FACE_FLAG_EXTERNAL_STREAM) ||
        add_dict_int(d, "ITALIC", FT_STYLE_FLAG_ITALIC) ||
        add_dict_int(d, "BOLD", FT_STYLE_FLAG_BOLD) ||
        add_dict_int(d, "KERNING_DEFAULT", FT_KERNING_DEFAULT) ||
        add_dict_int(d, "KERNING_UNFITTED", FT_KERNING_UNFITTED) ||
        add_dict_int(d, "KERNING_UNSCALED", FT_KERNING_UNSCALED) ||
        add_dict_int(d, "LOAD_DEFAULT", FT_LOAD_DEFAULT) ||
        add_dict_int(d, "LOAD_NO_SCALE", FT_LOAD_NO_SCALE) ||
        add_dict_int(d, "LOAD_NO_HINTING", FT_LOAD_NO_HINTING) ||
        add_dict_int(d, "LOAD_RENDER", FT_LOAD_RENDER) ||
        add_dict_int(d, "LOAD_NO_BITMAP", FT_LOAD_NO_BITMAP) ||
        add_dict_int(d, "LOAD_VERTICAL_LAYOUT", FT_LOAD_VERTICAL_LAYOUT) ||
        add_dict_int(d, "LOAD_FORCE_AUTOHINT", FT_LOAD_FORCE_AUTOHINT) ||
        add_dict_int(d, "LOAD_CROP_BITMAP", FT_LOAD_CROP_BITMAP) ||
        add_dict_int(d, "LOAD_PEDANTIC", FT_LOAD_PEDANTIC) ||
        add_dict_int(d, "LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH", FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH) ||
        add_dict_int(d, "LOAD_NO_RECURSE", FT_LOAD_NO_RECURSE) ||
        add_dict_int(d, "LOAD_IGNORE_TRANSFORM", FT_LOAD_IGNORE_TRANSFORM) ||
        add_dict_int(d, "LOAD_MONOCHROME", FT_LOAD_MONOCHROME) ||
        add_dict_int(d, "LOAD_LINEAR_DESIGN", FT_LOAD_LINEAR_DESIGN) ||
        add_dict_int(d, "LOAD_NO_AUTOHINT", (unsigned long)FT_LOAD_NO_AUTOHINT) ||
        add_dict_int(d, "LOAD_TARGET_NORMAL", (unsigned long)FT_LOAD_TARGET_NORMAL) ||
        add_dict_int(d, "LOAD_TARGET_LIGHT", (unsigned long)FT_LOAD_TARGET_LIGHT) ||
        add_dict_int(d, "LOAD_TARGET_MONO", (unsigned long)FT_LOAD_TARGET_MONO) ||
        add_dict_int(d, "LOAD_TARGET_LCD", (unsigned long)FT_LOAD_TARGET_LCD) ||
        add_dict_int(d, "LOAD_TARGET_LCD_V", (unsigned long)FT_LOAD_TARGET_LCD_V)) {
        return NULL;
    }

    // initialize library
    int error = FT_Init_FreeType(&_ft2Library);

    if (error) {
        PyErr_SetString(PyExc_RuntimeError, "Could not initialize the freetype2 library");
        return NULL;
    }

    {
        FT_Int major, minor, patch;
        char version_string[64];

        FT_Library_Version(_ft2Library, &major, &minor, &patch);
        sprintf(version_string, "%d.%d.%d", major, minor, patch);
        if (PyModule_AddStringConstant(m, "__freetype_version__", version_string)) {
            return NULL;
        }
    }

    if (PyModule_AddStringConstant(m, "__freetype_build_type__", STRINGIFY(FREETYPE_BUILD_TYPE))) {
        return NULL;
    }

    import_array();

    return m;
}

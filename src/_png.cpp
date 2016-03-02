/* -*- mode: c++; c-basic-offset: 4 -*- */

// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS

/* For linux, png.h must be imported before Python.h because
   png.h needs to be the one to define setjmp.
   Undefining _POSIX_C_SOURCE and _XOPEN_SOURCE stops a couple
   of harmless warnings.
*/


extern "C" {
#   include <png.h>
#   ifdef _POSIX_C_SOURCE
#       undef _POSIX_C_SOURCE
#   endif
#   ifdef _XOPEN_SOURCE
#       undef _XOPEN_SOURCE
#   endif
}

#include "numpy_cpp.h"
#include "mplutils.h"
#include "file_compat.h"

#   include <vector>
#   include "Python.h"


// As reported in [3082058] build _png.so on aix
#ifdef _AIX
#undef jmpbuf
#endif

struct buffer_t {
    PyObject *str;
    size_t cursor;
    size_t size;
};


static void write_png_data_buffer(png_structp png_ptr, png_bytep data, png_size_t length)
{
    buffer_t *buff = (buffer_t *)png_get_io_ptr(png_ptr);
    if (buff->cursor + length < buff->size) {
        memcpy(PyBytes_AS_STRING(buff->str) + buff->cursor, data, length);
        buff->cursor += length;
    }
}

static void flush_png_data_buffer(png_structp png_ptr)
{

}

static void write_png_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    PyObject *py_file_obj = (PyObject *)png_get_io_ptr(png_ptr);
    PyObject *write_method = PyObject_GetAttrString(py_file_obj, "write");
    PyObject *result = NULL;
    if (write_method) {
#if PY3K
        result = PyObject_CallFunction(write_method, (char *)"y#", data, length);
#else
        result = PyObject_CallFunction(write_method, (char *)"s#", data, length);
#endif
    }
    Py_XDECREF(write_method);
    Py_XDECREF(result);
}

static void flush_png_data(png_structp png_ptr)
{
    PyObject *py_file_obj = (PyObject *)png_get_io_ptr(png_ptr);
    PyObject *flush_method = PyObject_GetAttrString(py_file_obj, "flush");
    PyObject *result = NULL;
    if (flush_method) {
        result = PyObject_CallFunction(flush_method, (char *)"");
    }
    Py_XDECREF(flush_method);
    Py_XDECREF(result);
}

const char *Py_write_png__doc__ =
    "write_png(buffer, file, dpi=0, compression=6, filter=auto)\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "buffer : numpy array of image data\n"
    "    Must be an MxNxD array of dtype uint8.\n"
    "    - if D is 1, the image is greyscale\n"
    "    - if D is 3, the image is RGB\n"
    "    - if D is 4, the image is RGBA\n"
    "\n"
    "file : str path, file-like object or None\n"
    "    - If a str, must be a file path\n"
    "    - If a file-like object, must write bytes\n"
    "    - If None, a byte string containing the PNG data will be returned\n"
    "\n"
    "dpi : float\n"
    "    The dpi to store in the file metadata.\n"
    "\n"
    "compression : int\n"
    "    The level of lossless zlib compression to apply.  0 indicates no\n"
    "    compression.  Values 1-9 indicate low/fast through high/slow\n"
    "    compression.  Default is 6.\n"
    "\n"
    "filter : int\n"
    "    Filter to apply.  Must be one of the constants: PNG_FILTER_NONE,\n"
    "    PNG_FILTER_SUB, PNG_FILTER_UP, PNG_FILTER_AVG, PNG_FILTER_PAETH.\n"
    "    See the PNG standard for more information.\n"
    "    If not provided, libpng will try to automatically determine the\n"
    "    best filter on a line-by-line basis.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "buffer : bytes or None\n"
    "    Byte string containing the PNG content if None was passed in for\n"
    "    file, otherwise None is returned.\n";

static PyObject *Py_write_png(PyObject *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<unsigned char, 3> buffer;
    PyObject *filein;
    double dpi = 0;
    int compression = 6;
    int filter = -1;
    const char *names[] = { "buffer", "file", "dpi", "compression", "filter", NULL };

    // We don't need strict contiguity, just for each row to be
    // contiguous, and libpng has special handling for getting RGB out
    // of RGBA, ARGB or BGR. But the simplest thing to do is to
    // enforce contiguity using array_view::converter_contiguous.
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O&O|dii:write_png",
                                     (char **)names,
                                     &buffer.converter_contiguous,
                                     &buffer,
                                     &filein,
                                     &dpi,
                                     &compression,
                                     &filter)) {
        return NULL;
    }

    png_uint_32 width = (png_uint_32)buffer.dim(1);
    png_uint_32 height = (png_uint_32)buffer.dim(0);
    int channels = buffer.dim(2);
    std::vector<png_bytep> row_pointers(height);
    for (png_uint_32 row = 0; row < (png_uint_32)height; ++row) {
        row_pointers[row] = (png_bytep)buffer[row].data();
    }

    FILE *fp = NULL;
    mpl_off_t offset = 0;
    bool close_file = false;
    bool close_dup_file = false;
    PyObject *py_file = NULL;

    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    struct png_color_8_struct sig_bit;
    int png_color_type;
    buffer_t buff;
    buff.str = NULL;

    switch (channels) {
    case 1:
	png_color_type = PNG_COLOR_TYPE_GRAY;
	break;
    case 3:
	png_color_type = PNG_COLOR_TYPE_RGB;
	break;
    case 4:
	png_color_type = PNG_COLOR_TYPE_RGB_ALPHA;
	break;
    default:
        PyErr_SetString(PyExc_ValueError,
			"Buffer must be an NxMxD array with D in 1, 3, 4 "
			"(grayscale, RGB, RGBA)");
        goto exit;
    }

    if (compression < 0 || compression > 9) {
        PyErr_Format(PyExc_ValueError,
                     "compression must be in range 0-9, got %d", compression);
        goto exit;
    }

    if (PyBytes_Check(filein) || PyUnicode_Check(filein)) {
        if ((py_file = mpl_PyFile_OpenFile(filein, (char *)"wb")) == NULL) {
            goto exit;
        }
        close_file = true;
    } else {
        py_file = filein;
    }

    if (filein == Py_None) {
        buff.size = width * height * 4 + 1024;
        buff.str = PyBytes_FromStringAndSize(NULL, buff.size);
        if (buff.str == NULL) {
            goto exit;
        }
        buff.cursor = 0;
    } else {
        #if PY3K
        if (close_file) {
        #else
        if (close_file || PyFile_Check(py_file)) {
        #endif
            fp = mpl_PyFile_Dup(py_file, (char *)"wb", &offset);
        }

        if (fp) {
            close_dup_file = true;
        } else {
            PyErr_Clear();
            PyObject *write_method = PyObject_GetAttrString(py_file, "write");
            if (!(write_method && PyCallable_Check(write_method))) {
                Py_XDECREF(write_method);
                PyErr_SetString(PyExc_TypeError,
                                "Object does not appear to be a 8-bit string path or "
                                "a Python file-like object");
                goto exit;
            }
            Py_XDECREF(write_method);
        }
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create write struct");
        goto exit;
    }

    png_set_compression_level(png_ptr, compression);
    if (filter >= 0) {
        png_set_filter(png_ptr, 0, filter);
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create info struct");
        goto exit;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        PyErr_SetString(PyExc_RuntimeError, "libpng signaled error");
        goto exit;
    }

    if (buff.str) {
        png_set_write_fn(png_ptr, (void *)&buff, &write_png_data_buffer, &flush_png_data_buffer);
    } else if (fp) {
        png_init_io(png_ptr, fp);
    } else {
        png_set_write_fn(png_ptr, (void *)py_file, &write_png_data, &flush_png_data);
    }
    png_set_IHDR(png_ptr,
                 info_ptr,
                 width,
                 height,
                 8,
                 png_color_type,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);

    // Save the dpi of the image in the file
    if (dpi > 0.0) {
        png_uint_32 dots_per_meter = (png_uint_32)(dpi / (2.54 / 100.0));
        png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter, PNG_RESOLUTION_METER);
    }

    sig_bit.alpha = 0;
    switch (png_color_type) {
    case PNG_COLOR_TYPE_GRAY:
	sig_bit.gray = 8;
	sig_bit.red = 0;
	sig_bit.green = 0;
	sig_bit.blue = 0;
	break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
	sig_bit.alpha = 8;
	// fall through
    case PNG_COLOR_TYPE_RGB:
	sig_bit.gray = 0;
	sig_bit.red = 8;
	sig_bit.green = 8;
	sig_bit.blue = 8;
	break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "internal error, bad png_color_type");
        goto exit;
    }
    png_set_sBIT(png_ptr, info_ptr, &sig_bit);

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, &row_pointers[0]);
    png_write_end(png_ptr, info_ptr);

exit:

    if (png_ptr && info_ptr) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }

    if (close_dup_file) {
        mpl_PyFile_DupClose(py_file, fp, offset);
    }

    if (close_file) {
        mpl_PyFile_CloseFile(py_file);
        Py_DECREF(py_file);
    }

    if (PyErr_Occurred()) {
        Py_XDECREF(buff.str);
        return NULL;
    } else {
        if (buff.str) {
            _PyBytes_Resize(&buff.str, buff.cursor);
            return buff.str;
        }
        Py_RETURN_NONE;
    }
}

static void _read_png_data(PyObject *py_file_obj, png_bytep data, png_size_t length)
{
    PyObject *read_method = PyObject_GetAttrString(py_file_obj, "read");
    PyObject *result = NULL;
    char *buffer;
    Py_ssize_t bufflen;
    if (read_method) {
        result = PyObject_CallFunction(read_method, (char *)"i", length);
        if (PyBytes_AsStringAndSize(result, &buffer, &bufflen) == 0) {
            if (bufflen == (Py_ssize_t)length) {
                memcpy(data, buffer, length);
            } else {
                PyErr_SetString(PyExc_IOError, "read past end of file");
            }
        }
    }
    Py_XDECREF(read_method);
    Py_XDECREF(result);
}

static void read_png_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    PyObject *py_file_obj = (PyObject *)png_get_io_ptr(png_ptr);
    _read_png_data(py_file_obj, data, length);
}

static PyObject *_read_png(PyObject *filein, bool float_result)
{
    png_byte header[8]; // 8 is the maximum size that can be checked
    FILE *fp = NULL;
    mpl_off_t offset = 0;
    bool close_file = false;
    bool close_dup_file = false;
    PyObject *py_file = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    int num_dims;
    std::vector<png_bytep> row_pointers;
    png_uint_32 width = 0;
    png_uint_32 height = 0;
    int bit_depth;
    PyObject *result = NULL;

    // TODO: Remove direct calls to Numpy API here

    if (PyBytes_Check(filein) || PyUnicode_Check(filein)) {
        if ((py_file = mpl_PyFile_OpenFile(filein, (char *)"rb")) == NULL) {
            goto exit;
        }
        close_file = true;
    } else {
        py_file = filein;
    }

    #if PY3K
    if (close_file) {
    #else
    if (close_file || PyFile_Check(py_file)) {
    #endif
        fp = mpl_PyFile_Dup(py_file, (char *)"rb", &offset);
    }

    if (fp) {
        close_dup_file = true;
        if (fread(header, 1, 8, fp) != 8) {
            PyErr_SetString(PyExc_IOError, "error reading PNG header");
            goto exit;
        }
    } else {
        PyErr_Clear();

        PyObject *read_method = PyObject_GetAttrString(py_file, "read");
        if (!(read_method && PyCallable_Check(read_method))) {
            Py_XDECREF(read_method);
            PyErr_SetString(PyExc_TypeError,
                            "Object does not appear to be a 8-bit string path or "
                            "a Python file-like object");
            goto exit;
        }
        Py_XDECREF(read_method);
        _read_png_data(py_file, header, 8);
    }

    if (png_sig_cmp(header, 0, 8)) {
        PyErr_SetString(PyExc_ValueError, "invalid PNG header");
        goto exit;
    }

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "png_create_read_struct failed");
        goto exit;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "png_create_info_struct failed");
        goto exit;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        PyErr_SetString(PyExc_RuntimeError, "Error setting jump");
        goto exit;
    }

    if (fp) {
        png_init_io(png_ptr, fp);
    } else {
        png_set_read_fn(png_ptr, (void *)py_file, &read_png_data);
    }
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);

    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Unpack 1, 2, and 4-bit images
    if (bit_depth < 8) {
        png_set_packing(png_ptr);
    }

    // If sig bits are set, shift data
    png_color_8p sig_bit;
    if ((png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_PALETTE) &&
        png_get_sBIT(png_ptr, info_ptr, &sig_bit)) {
        png_set_shift(png_ptr, sig_bit);
    }

    // Convert big endian to little
    if (bit_depth == 16) {
        png_set_swap(png_ptr);
    }

    // Convert palletes to full RGB
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
        bit_depth = 8;
    }

    // If there's an alpha channel convert gray to RGB
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }

    png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    row_pointers.resize(height);
    for (png_uint_32 row = 0; row < height; row++) {
        row_pointers[row] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];
    }

    png_read_image(png_ptr, &row_pointers[0]);

    npy_intp dimensions[3];
    dimensions[0] = height; // numrows
    dimensions[1] = width; // numcols
    if (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_ALPHA) {
        dimensions[2] = 4; // RGBA images
    } else if (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_COLOR) {
        dimensions[2] = 3; // RGB images
    } else {
        dimensions[2] = 1; // Greyscale images
    }

    if (float_result) {
        double max_value = (1 << bit_depth) - 1;

        numpy::array_view<float, 3> A(dimensions);

        for (png_uint_32 y = 0; y < height; y++) {
            png_byte *row = row_pointers[y];
            for (png_uint_32 x = 0; x < width; x++) {
                if (bit_depth == 16) {
                    png_uint_16 *ptr = &reinterpret_cast<png_uint_16 *>(row)[x * dimensions[2]];
                    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++) {
                        A(y, x, p) = (float)(ptr[p]) / max_value;
                    }
                } else {
                    png_byte *ptr = &(row[x * dimensions[2]]);
                    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++) {
                        A(y, x, p) = (float)(ptr[p]) / max_value;
                    }
                }
            }
        }

        result = A.pyobj();
    } else if (bit_depth == 16) {
        numpy::array_view<png_uint_16, 3> A(dimensions);

        for (png_uint_32 y = 0; y < height; y++) {
            png_byte *row = row_pointers[y];
            for (png_uint_32 x = 0; x < width; x++) {
                png_uint_16 *ptr = &reinterpret_cast<png_uint_16 *>(row)[x * dimensions[2]];
                for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++) {
                    A(y, x, p) = ptr[p];
                }
            }
        }

        result = A.pyobj();
    } else if (bit_depth == 8) {
        numpy::array_view<png_byte, 3> A(dimensions);

        for (png_uint_32 y = 0; y < height; y++) {
            png_byte *row = row_pointers[y];
            for (png_uint_32 x = 0; x < width; x++) {
                png_byte *ptr = &(row[x * dimensions[2]]);
                for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++) {
                    A(y, x, p) = ptr[p];
                }
            }
        }

        result = A.pyobj();
    } else {
        PyErr_SetString(PyExc_RuntimeError, "image has unknown bit depth");
        goto exit;
    }

    // free the png memory
    png_read_end(png_ptr, info_ptr);

    // For gray, return an x by y array, not an x by y by 1
    num_dims = (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_COLOR) ? 3 : 2;

    if (num_dims == 2) {
        PyArray_Dims dims = {dimensions, 2};
        PyObject *reshaped = PyArray_Newshape((PyArrayObject *)result, &dims, NPY_CORDER);
        Py_DECREF(result);
        result = reshaped;
    }

exit:
    if (png_ptr && info_ptr) {
#ifndef png_infopp_NULL
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
#else
        png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
#endif
    }

    if (close_dup_file) {
        mpl_PyFile_DupClose(py_file, fp, offset);
    }

    if (close_file) {
        mpl_PyFile_CloseFile(py_file);
        Py_DECREF(py_file);
    }

    for (png_uint_32 row = 0; row < height; row++) {
        delete[] row_pointers[row];
    }

    if (PyErr_Occurred()) {
        Py_XDECREF(result);
        return NULL;
    } else {
        return result;
    }
}

const char *Py_read_png_float__doc__ =
    "read_png_float(file)\n"
    "\n"
    "Read in a PNG file, converting values to floating-point doubles\n"
    "in the range (0, 1)\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file : str path or file-like object\n";

static PyObject *Py_read_png_float(PyObject *self, PyObject *args, PyObject *kwds)
{
    return _read_png(args, true);
}

const char *Py_read_png_int__doc__ =
    "read_png_int(file)\n"
    "\n"
    "Read in a PNG file with original integer values.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file : str path or file-like object\n";

static PyObject *Py_read_png_int(PyObject *self, PyObject *args, PyObject *kwds)
{
    return _read_png(args, false);
}

const char *Py_read_png__doc__ =
    "read_png(file)\n"
    "\n"
    "Read in a PNG file, converting values to floating-point doubles\n"
    "in the range (0, 1)\n"
    "\n"
    "Alias for read_png_float()\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file : str path or file-like object\n";

static PyMethodDef module_methods[] = {
    {"write_png", (PyCFunction)Py_write_png, METH_VARARGS|METH_KEYWORDS, Py_write_png__doc__},
    {"read_png", (PyCFunction)Py_read_png_float, METH_O, Py_read_png__doc__},
    {"read_png_float", (PyCFunction)Py_read_png_float, METH_O, Py_read_png_float__doc__},
    {"read_png_int", (PyCFunction)Py_read_png_int, METH_O, Py_read_png_int__doc__},
    {NULL}
};

extern "C" {

#if PY3K
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_png",
        NULL,
        0,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

#define INITERROR return NULL

    PyMODINIT_FUNC PyInit__png(void)

#else
#define INITERROR return

    PyMODINIT_FUNC init_png(void)
#endif

    {
        PyObject *m;

#if PY3K
        m = PyModule_Create(&moduledef);
#else
        m = Py_InitModule3("_png", module_methods, NULL);
#endif

        if (m == NULL) {
            INITERROR;
        }

        import_array();

        if (PyModule_AddIntConstant(m, "PNG_FILTER_NONE", PNG_FILTER_NONE) ||
            PyModule_AddIntConstant(m, "PNG_FILTER_SUB", PNG_FILTER_SUB) ||
            PyModule_AddIntConstant(m, "PNG_FILTER_UP", PNG_FILTER_UP) ||
            PyModule_AddIntConstant(m, "PNG_FILTER_AVG", PNG_FILTER_AVG) ||
            PyModule_AddIntConstant(m, "PNG_FILTER_PAETH", PNG_FILTER_PAETH)) {
            INITERROR;
        }


#if PY3K
        return m;
#endif
    }
}

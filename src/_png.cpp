#include <png.h>

// To remove a gcc warning
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

// TODO: Un CXX-ify this module
#include "CXX/Extensions.hxx"
#include "numpy/arrayobject.h"
#include "mplutils.h"

// the extension module
class _png_module : public Py::ExtensionModule<_png_module>
{
public:
    _png_module()
            : Py::ExtensionModule<_png_module>( "_png" )
    {
        add_varargs_method("write_png", &_png_module::write_png,
                           "write_png(buffer, width, height, fileobj, dpi=None)");
        add_varargs_method("read_png", &_png_module::read_png,
                           "read_png(fileobj)");
        initialize("Module to write PNG files");
    }

    virtual ~_png_module() {}

private:
    Py::Object write_png(const Py::Tuple& args);
    Py::Object read_png(const Py::Tuple& args);
};

static void write_png_data(png_structp png_ptr, png_bytep data, png_size_t length) {
  PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
  PyObject* write_method = PyObject_GetAttrString(py_file_obj, "write");
  PyObject* result = NULL;
  if (write_method)
    result = PyObject_CallFunction(write_method, (char *)"s#", data, length);
  Py_XDECREF(write_method);
  Py_XDECREF(result);
}

static void flush_png_data(png_structp png_ptr) {
  PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
  PyObject* flush_method = PyObject_GetAttrString(py_file_obj, "flush");
  PyObject* result = NULL;
  if (flush_method)
    result = PyObject_CallFunction(flush_method, (char *)"");
  Py_XDECREF(flush_method);
  Py_XDECREF(result);
}

// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
Py::Object _png_module::write_png(const Py::Tuple& args)
{
  args.verify_length(4, 5);

  FILE *fp = NULL;
  bool close_file = false;
  Py::Object buffer_obj = Py::Object(args[0]);
  PyObject* buffer = buffer_obj.ptr();
  if (!PyObject_CheckReadBuffer(buffer)) {
    throw Py::TypeError("First argument must be an rgba buffer.");
  }

  const void* pixBufferPtr = NULL;
  Py_ssize_t pixBufferLength = 0;
  if (PyObject_AsReadBuffer(buffer, &pixBufferPtr, &pixBufferLength)) {
    throw Py::ValueError("Couldn't get data from read buffer.");
  }

  png_byte* pixBuffer = (png_byte*)pixBufferPtr;
  int width = (int)Py::Int(args[1]);
  int height = (int)Py::Int(args[2]);

  if (pixBufferLength < width * height * 4) {
    throw Py::ValueError("Buffer and width, height don't seem to match.");
  }

  Py::Object py_fileobj = Py::Object(args[3]);
  if (py_fileobj.isString()) {
    std::string fileName = Py::String(py_fileobj);
    const char *file_name = fileName.c_str();
    if ((fp = fopen(file_name, "wb")) == NULL)
      throw Py::RuntimeError( Printf("Could not open file %s", file_name).str() );
    close_file = true;
  } else if (PyFile_CheckExact(py_fileobj.ptr())) {
    fp = PyFile_AsFile(py_fileobj.ptr());
  }
  else {
    PyObject* write_method = PyObject_GetAttrString(py_fileobj.ptr(), "write");
    if (!(write_method && PyCallable_Check(write_method))) {
      Py_XDECREF(write_method);
      throw Py::TypeError("Object does not appear to be a 8-bit string path or a Python file-like object");
    }
    Py_XDECREF(write_method);
  }

  png_bytep *row_pointers = NULL;
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;

  try {
    struct png_color_8_struct sig_bit;
    png_uint_32 row;

    row_pointers = new png_bytep[height];
    for (row = 0; row < (png_uint_32)height; ++row) {
      row_pointers[row] = pixBuffer + row * width * 4;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
      throw Py::RuntimeError("Could not create write struct");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
      throw Py::RuntimeError("Could not create info struct");
    }

    if (setjmp(png_ptr->jmpbuf)) {
      throw Py::RuntimeError("Error building image");
    }

    if (fp) {
      png_init_io(png_ptr, fp);
    } else {
      png_set_write_fn(png_ptr, (void*)py_fileobj.ptr(),
		       &write_png_data, &flush_png_data);
    }
    png_set_IHDR(png_ptr, info_ptr,
		 width, height, 8,
		 PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
		 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Save the dpi of the image in the file
    if (args.size() == 5) {
      double dpi = Py::Float(args[4]);
      size_t dots_per_meter = (size_t)(dpi / (2.54 / 100.0));
      png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter, PNG_RESOLUTION_METER);
    }

    // this a a color image!
    sig_bit.gray = 0;
    sig_bit.red = 8;
    sig_bit.green = 8;
    sig_bit.blue = 8;
    /* if the image has an alpha channel then */
    sig_bit.alpha = 8;
    png_set_sBIT(png_ptr, info_ptr, &sig_bit);

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, info_ptr);
  } catch (...) {
      if (fp && close_file) fclose(fp);
      delete [] row_pointers;
      /* Changed calls to png_destroy_write_struct to follow
         http://www.libpng.org/pub/png/libpng-manual.txt.
         This ensures the info_ptr memory is released.
      */
      if (png_ptr && info_ptr) png_destroy_write_struct(&png_ptr, &info_ptr);
      throw;
  }

  png_destroy_write_struct(&png_ptr, &info_ptr);
  delete [] row_pointers;
  if (fp && close_file) fclose(fp);

  return Py::Object();
}


Py::Object
_png_module::read_png(const Py::Tuple& args) {

  args.verify_length(1);
  std::string fname = Py::String(args[0]);

  png_byte header[8];	// 8 is the maximum size that can be checked

  FILE *fp = fopen(fname.c_str(), "rb");
  if (!fp)
    throw Py::RuntimeError(Printf("_image_module::readpng could not open PNG file %s for reading", fname.c_str()).str());

  if (fread(header, 1, 8, fp) != 8)
    throw Py::RuntimeError("_image_module::readpng: error reading PNG header");
  if (png_sig_cmp(header, 0, 8))
    throw Py::RuntimeError("_image_module::readpng: file not recognized as a PNG file");


  /* initialize stuff */
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    throw Py::RuntimeError("_image_module::readpng:  png_create_read_struct failed");

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    throw Py::RuntimeError("_image_module::readpng:  png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    throw Py::RuntimeError("_image_module::readpng:  error during init_io");

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  png_uint_32 width = info_ptr->width;
  png_uint_32 height = info_ptr->height;

  int bit_depth = info_ptr->bit_depth;

  // Unpack 1, 2, and 4-bit images
  if (bit_depth < 8)
    png_set_packing(png_ptr);

  // If sig bits are set, shift data
  png_color_8p sig_bit;
  if ((info_ptr->color_type != PNG_COLOR_TYPE_PALETTE) && png_get_sBIT(png_ptr, info_ptr, &sig_bit))
    png_set_shift(png_ptr, sig_bit);

  // Convert big endian to little
  if (bit_depth == 16)
    png_set_swap(png_ptr);

  // Convert palletes to full RGB
  if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png_ptr);

  // If there's an alpha channel convert gray to RGB
  if (info_ptr->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png_ptr);

  png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  if (setjmp(png_jmpbuf(png_ptr)))
    throw Py::RuntimeError("_image_module::readpng: error during read_image");

  png_bytep *row_pointers = new png_bytep[height];
  png_uint_32 row;

  for (row = 0; row < height; row++)
    row_pointers[row] = new png_byte[png_get_rowbytes(png_ptr,info_ptr)];

  png_read_image(png_ptr, row_pointers);

  npy_intp dimensions[3];
  dimensions[0] = height;  //numrows
  dimensions[1] = width;   //numcols
  if (info_ptr->color_type & PNG_COLOR_MASK_ALPHA)
    dimensions[2] = 4;     //RGBA images
  else if (info_ptr->color_type & PNG_COLOR_MASK_COLOR)
    dimensions[2] = 3;     //RGB images
  else
    dimensions[2] = 1;     //Greyscale images
  //For gray, return an x by y array, not an x by y by 1
  int num_dims  = (info_ptr->color_type & PNG_COLOR_MASK_COLOR) ? 3 : 2;

  double max_value = (1 << ((bit_depth < 8) ? 8 : bit_depth)) - 1;
  PyArrayObject *A = (PyArrayObject *) PyArray_SimpleNew(num_dims, dimensions, PyArray_FLOAT);

  if (A == NULL) {
    throw Py::MemoryError("Could not allocate image array");
  }

  for (png_uint_32 y = 0; y < height; y++) {
    png_byte* row = row_pointers[y];
	for (png_uint_32 x = 0; x < width; x++) {
	  size_t offset = y*A->strides[0] + x*A->strides[1];
	  if (bit_depth == 16) {
	    png_uint_16* ptr = &reinterpret_cast<png_uint_16*> (row)[x * dimensions[2]];
            for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++)
	      *(float*)(A->data + offset + p*A->strides[2]) = (float)(ptr[p]) / max_value;
	  } else {
	    png_byte* ptr = &(row[x * dimensions[2]]);
	    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++)
		{
	      *(float*)(A->data + offset + p*A->strides[2]) = (float)(ptr[p]) / max_value;
	    }
	  }
    }
  }

  //free the png memory
  png_read_end(png_ptr, info_ptr);
  png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
  fclose(fp);
  for (row = 0; row < height; row++)
    delete [] row_pointers[row];
  delete [] row_pointers;
  return Py::asObject((PyObject*)A);
}

extern "C"
    DL_EXPORT(void)
    init_png(void)
{
    import_array();

    static _png_module* _png = NULL;
    _png = new _png_module;
}

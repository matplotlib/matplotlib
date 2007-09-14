#include <cstring>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <utility>
#include <fstream>


#include <pygobject.h>
#include <pygtk/pygtk.h>

#include "agg_basics.h"
#include "_backend_agg.h"

// the extension module
class _gtkagg_module : public Py::ExtensionModule<_gtkagg_module>
{
public:
  _gtkagg_module()
    : Py::ExtensionModule<_gtkagg_module>( "_gtkagg" )
  {
    add_varargs_method("agg_to_gtk_drawable",
		       &_gtkagg_module::agg_to_gtk_drawable,
		       "Draw to a gtk drawable from a agg buffer.");
    initialize( "The _gtkagg module" );
  }

  virtual ~_gtkagg_module() {}

private:

  Py::Object agg_to_gtk_drawable(const Py::Tuple &args) {
    // args are gc, renderer, bbox where bbox is a transforms BBox
    // (possibly None).  If bbox is None, blit the entire agg buffer
    // to gtk.  If bbox is not None, blit only the region defined by
    // the bbox
    args.verify_length(3);

    PyGObject *py_drawable = (PyGObject *)(args[0].ptr());
    RendererAgg* aggRenderer = static_cast<RendererAgg*>(args[1].ptr());

    GdkDrawable *drawable = GDK_DRAWABLE(py_drawable->obj);
    GdkGC* gc = gdk_gc_new(drawable);

    int srcstride = aggRenderer->get_width()*4;
    int srcwidth = (int)aggRenderer->get_width();
    int srcheight = (int)aggRenderer->get_height();

    // these three will be overridden below
    int destx = 0;
    int desty = 0;
    int destwidth = 1;
    int destheight = 1;
    int deststride = 1;


    bool needfree = false;

    agg::int8u *destbuffer = NULL;

    if (args[2].ptr() == Py_None) {
      //bbox is None; copy the entire image
      destbuffer = aggRenderer->pixBuffer;
      destwidth = srcwidth;
      destheight = srcheight;
      deststride = srcstride;
    }
    else {
      //bbox is not None; copy the image in the bbox
      // MGDTODO: Use PyArray rather than buffer interface here

      PyObject* clipbox = args[2].ptr();
      const void* clipbox_buffer;
      Py_ssize_t clipbox_buffer_len;
      if (!PyObject_CheckReadBuffer(clipbox))
	throw Py::TypeError
	  ("Argument 3 to agg_to_gtk_drawable must be a Bbox object.");

      if (PyObject_AsReadBuffer(clipbox, &clipbox_buffer, &clipbox_buffer_len))
	throw Py::Exception();

      if (clipbox_buffer_len != sizeof(double) * 4)
	throw Py::TypeError
	  ("Argument 3 to agg_to_gtk_drawable must be a Bbox object.");

      double* clipbox_values = (double*)clipbox_buffer;
      double l = clipbox_values[0];
      double b = clipbox_values[1];
      double r = clipbox_values[2];
      double t = clipbox_values[3];

      //std::cout << b << " "
      //		<< t << " ";

      destx = (int)l;
      desty = srcheight-(int)t;
      destwidth = (int)(r-l);
      destheight = (int)(t-b);
      deststride = destwidth*4;

      needfree = true;
      destbuffer = new agg::int8u[deststride*destheight];
      if (destbuffer ==NULL) {
	throw Py::MemoryError("_gtkagg could not allocate memory for destbuffer");
      }

      agg::rendering_buffer destrbuf;
      destrbuf.attach(destbuffer, destwidth, destheight, deststride);
      pixfmt destpf(destrbuf);
      renderer_base destrb(destpf);
      //destrb.clear(agg::rgba(1, 0, 0));

      agg::rect_base<int> region(destx, desty, (int)r, srcheight-(int)b);
      destrb.copy_from(*aggRenderer->renderingBuffer, &region,
		       -destx, -desty);


    }


    /*std::cout << desty << " "
	      << destheight << " "
	      << srcheight << std::endl;*/


    //gdk_rgb_init();
    gdk_draw_rgb_32_image(drawable, gc, destx, desty,
			  destwidth,
			  destheight,
			  GDK_RGB_DITHER_NORMAL,
			  destbuffer,
			  deststride);

    if (needfree) delete [] destbuffer;

    return Py::Object();

  }
};


extern "C"
DL_EXPORT(void)
  init_gtkagg(void)
{
  init_pygobject();
  init_pygtk();
  //suppress unused warning by creating in two lines
  static _gtkagg_module* _gtkagg = NULL;
  _gtkagg = new _gtkagg_module;

};








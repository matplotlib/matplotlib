#include <cstring>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <utility>

#include <pygobject.h>
#include <pygtk/pygtk.h>

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

    args.verify_length(2);  

  
    PyGObject *py_drawable = (PyGObject *)(args[0].ptr());
    RendererAgg* aggRenderer = static_cast<RendererAgg*>(args[1].ptr());

    
    GdkDrawable *drawable = GDK_DRAWABLE(py_drawable->obj);
    GdkGC* gc = gdk_gc_new(drawable);

    unsigned int width = aggRenderer->get_width();
    unsigned int height = aggRenderer->get_height();

    gdk_draw_rgb_32_image(drawable, gc, 0, 0, 
			  width, 
			  height, 
			  GDK_RGB_DITHER_NORMAL,
			  aggRenderer->pixBuffer,
			  width*4);

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







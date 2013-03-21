
/* -*- mode: c++; c-basic-offset: 4 -*- */

#include <pygobject.h>
#include <pygtk/pygtk.h>

#include <cstring>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <utility>
#include <fstream>

#include "agg_basics.h"
#include "numpy/arrayobject.h"
#include "_backend_agg.h"
#include "agg_py_transforms.h"

// the extension module
class _gtkagg_module : public Py::ExtensionModule<_gtkagg_module>
{
public:
    _gtkagg_module()
        : Py::ExtensionModule<_gtkagg_module>("_gtkagg")
    {
        add_varargs_method("agg_to_gtk_drawable",
                           &_gtkagg_module::agg_to_gtk_drawable,
                           "Draw to a gtk drawable from a agg buffer.");
        initialize("The _gtkagg module");
    }

    virtual ~_gtkagg_module() {}

private:

    Py::Object agg_to_gtk_drawable(const Py::Tuple &args)
    {
        // args are gc, renderer, bbox where bbox is a transforms BBox
        // (possibly None).  If bbox is None, blit the entire agg buffer
        // to gtk.  If bbox is not None, blit only the region defined by
        // the bbox
        args.verify_length(3);

        PyGObject *py_drawable = (PyGObject *)(args[0].ptr());
        RendererAgg* aggRenderer = static_cast<RendererAgg*>(args[1].ptr());

        GdkDrawable *drawable = GDK_DRAWABLE(py_drawable->obj);
        GdkGC* gc = gdk_gc_new(drawable);

        int srcstride = aggRenderer->get_width() * 4;
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

        if (args[2].ptr() == Py_None)
        {
            //bbox is None; copy the entire image
            destbuffer = aggRenderer->pixBuffer;
            destwidth = srcwidth;
            destheight = srcheight;
            deststride = srcstride;
        }
        else
        {
            //bbox is not None; copy the image in the bbox
            PyObject* clipbox = args[2].ptr();
            double l, b, r, t;

            if (!py_convert_bbox(clipbox, l, b, r, t))
            {
                throw Py::TypeError
                ("Argument 3 to agg_to_gtk_drawable must be a Bbox object.");
            }

            destx = (int)l;
            desty = srcheight - (int)t;
            destwidth = (int)(r - l);
            destheight = (int)(t - b);
            deststride = destwidth * 4;

            needfree = true;
            destbuffer = new agg::int8u[deststride*destheight];
            if (destbuffer == NULL)
            {
                throw Py::MemoryError("_gtkagg could not allocate memory for destbuffer");
            }

            agg::rendering_buffer destrbuf;
            destrbuf.attach(destbuffer, destwidth, destheight, deststride);
            pixfmt destpf(destrbuf);
            renderer_base destrb(destpf);

            //destrb.clear(agg::rgba(1, 1, 1, 0));

            agg::rect_base<int> region(destx, desty, (int)r, srcheight - (int)b);
            destrb.copy_from(aggRenderer->renderingBuffer, &region,
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

        gdk_gc_destroy(gc);
        if (needfree)
        {
            delete [] destbuffer;
        }

        return Py::Object();

    }
};

PyMODINIT_FUNC
init_gtkagg(void)
{
    init_pygobject();
    init_pygtk();

    import_array();
    //suppress unused warning by creating in two lines
    static _gtkagg_module* _gtkagg = NULL;
    _gtkagg = new _gtkagg_module;
}








/* -*- mode: c++; c-basic-offset: 4 -*- */

#include <vector>

#include <pygobject.h>
#include <pygtk/pygtk.h>

#include "agg_basics.h"
#include "agg_pixfmt_rgba.h"
#include "agg_renderer_base.h"
#include "agg_rendering_buffer.h"

#include "numpy_cpp.h"
#include "py_converters.h"

static PyObject *Py_agg_to_gtk_drawable(PyObject *self, PyObject *args, PyObject *kwds)
{
    typedef agg::pixfmt_rgba32_plain pixfmt;
    typedef agg::renderer_base<pixfmt> renderer_base;

    PyGObject *py_drawable;
    numpy::array_view<agg::int8u, 3> buffer;
    agg::rect_d rect;

    // args are gc, renderer, bbox where bbox is a transforms BBox
    // (possibly None).  If bbox is None, blit the entire agg buffer
    // to gtk.  If bbox is not None, blit only the region defined by
    // the bbox

    if (!PyArg_ParseTuple(args,
                          "OO&O&:agg_to_gtk_drawable",
                          &py_drawable,
                          &buffer.converter,
                          &buffer,
                          &convert_rect,
                          &rect)) {
        return NULL;
    }

    if (buffer.dim(2) != 4) {
        PyErr_SetString(PyExc_ValueError, "Invalid image buffer.  Must be NxMx4.");
        return NULL;
    }

    GdkDrawable *drawable = GDK_DRAWABLE(py_drawable->obj);
    GdkGC *gc = gdk_gc_new(drawable);

    int srcstride = buffer.dim(1) * 4;
    int srcwidth = buffer.dim(1);
    int srcheight = buffer.dim(0);

    // these three will be overridden below
    int destx = 0;
    int desty = 0;
    int destwidth = 1;
    int destheight = 1;
    int deststride = 1;

    std::vector<agg::int8u> destbuffer;
    agg::int8u *destbufferptr;

    if (rect.x1 == 0.0 && rect.x2 == 0.0 && rect.y1 == 0.0 && rect.y2 == 0.0) {
        // bbox is None; copy the entire image
        destbufferptr = (agg::int8u *)buffer.data();
        destwidth = srcwidth;
        destheight = srcheight;
        deststride = srcstride;
    } else {
        destx = (int)rect.x1;
        desty = srcheight - (int)rect.y2;
        destwidth = (int)(rect.x2 - rect.x1);
        destheight = (int)(rect.y2 - rect.y1);
        deststride = destwidth * 4;
        destbuffer.reserve(destheight * deststride);
        destbufferptr = &destbuffer[0];

        agg::rendering_buffer destrbuf;
        destrbuf.attach(destbufferptr, destwidth, destheight, deststride);
        pixfmt destpf(destrbuf);
        renderer_base destrb(destpf);

        agg::rendering_buffer srcrbuf;
        srcrbuf.attach((agg::int8u *)buffer.data(), buffer.dim(1), buffer.dim(0), buffer.dim(1) * 4);

        agg::rect_base<int> region(destx, desty, (int)rect.x2, srcheight - (int)rect.y1);
        destrb.copy_from(srcrbuf, &region, -destx, -desty);
    }

    gdk_draw_rgb_32_image(drawable,
                          gc,
                          destx,
                          desty,
                          destwidth,
                          destheight,
                          GDK_RGB_DITHER_NORMAL,
                          destbufferptr,
                          deststride);

    gdk_gc_destroy(gc);

    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"agg_to_gtk_drawable", (PyCFunction)Py_agg_to_gtk_drawable, METH_VARARGS, NULL},
    NULL
};

extern "C" {

#if PY3K
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gtkagg",
        NULL,
        0,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

#define INITERROR return NULL

    PyMODINIT_FUNC PyInit__gtkagg(void)

#else
#define INITERROR return

    PyMODINIT_FUNC init_gtkagg(void)
#endif

    {
        PyObject *m;

#if PY3K
        m = PyModule_Create(&moduledef);
#else
        m = Py_InitModule3("_gtkagg", module_methods, NULL);
#endif

        if (m == NULL) {
            INITERROR;
        }

        init_pygobject();
        init_pygtk();
        import_array();

#if PY3K
        return m;
#endif
    }
}

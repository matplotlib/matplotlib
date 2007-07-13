/*
 * The Python Imaging Library.
 * $Id$
 *
 */

/* This is needed for (at least) Tk 8.4.1, otherwise the signature of
** Tk_PhotoPutBlock changes.
*/
#define USE_COMPOSITELESS_PHOTO_PUT_BLOCK

#include <Python.h>
#include <stdlib.h>

#include "agg_basics.h"
#include "_backend_agg.h"
#include "_transforms.h"

extern "C" {
#ifdef __APPLE__
#  ifdef TK_FRAMEWORK
#     include <Tcl/tcl.h>
#     include <Tk/tk.h>
#  else
#     include <tk.h>
#  endif
#else
#  include <tk.h>
#endif
};



typedef struct {
    PyObject_HEAD
    Tcl_Interp* interp;
} TkappObject;

static int
PyAggImagePhoto(ClientData clientdata, Tcl_Interp* interp,
               int argc, char **argv)
{
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;
    PyObject* aggo;

    // vars for blitting
    PyObject* bboxo;
    Bbox* bbox;
    agg::int8u *destbuffer;
    double l,b,r,t;
    int destx, desty, destwidth, destheight, deststride;

    long mode;
    long nval;
    if (Tk_MainWindow(interp) == NULL) {
        // Will throw a _tkinter.TclError with "this isn't a Tk application"
        return TCL_ERROR;
    }

    if (argc != 5) {
        Tcl_AppendResult(interp, "usage: ", argv[0],
                         " destPhoto srcImage", (char *) NULL);
        return TCL_ERROR;
    }

    /* get Tcl PhotoImage handle */
    photo = Tk_FindPhoto(interp, argv[1]);
    if (photo == NULL) {
        Tcl_AppendResult(interp, "destination photo must exist", (char *) NULL);
        return TCL_ERROR;
    }
    /* get array (or object that can be converted to array) pointer */
    aggo = (PyObject*)atol(argv[2]);
    RendererAgg *aggRenderer = (RendererAgg *)aggo;
    int srcheight = (int)aggRenderer->get_height();

    /* XXX insert aggRenderer type check */

    /* get array mode (0=mono, 1=rgb, 2=rgba) */
    mode = atol(argv[3]);
    if ((mode != 0) && (mode != 1) && (mode != 2)) {
        Tcl_AppendResult(interp, "illegal image mode", (char *) NULL);
        return TCL_ERROR;
    }

    /* check for bbox/blitting */
    bboxo = (PyObject*)atol(argv[4]);
    if (bboxo != Py_None) {
      bbox = (Bbox*)bboxo;
      l = bbox->ll_api()->x_api()->val();
      b = bbox->ll_api()->y_api()->val();
      r = bbox->ur_api()->x_api()->val();
      t = bbox->ur_api()->y_api()->val();

      destx = (int)l;
      desty = srcheight-(int)t;
      destwidth = (int)(r-l);
      destheight = (int)(t-b);
      deststride = 4*destwidth;

      destbuffer = new agg::int8u[deststride*destheight];
      if (destbuffer == NULL) {
	throw Py::MemoryError("_tkagg could not allocate memory for destbuffer");
      }

      agg::rendering_buffer destrbuf;
      destrbuf.attach(destbuffer, destwidth, destheight, deststride);
      pixfmt destpf(destrbuf);
      renderer_base destrb(destpf);

      agg::rect_base<int> region(destx, desty, (int)r, srcheight-(int)b);
      destrb.copy_from(*aggRenderer->renderingBuffer, &region,
		       -destx, -desty);
    } else {
      bbox = NULL;
      destbuffer = NULL;
      destx = desty = destwidth = destheight = deststride = 0;
    }

    /* setup tkblock */
    block.pixelSize = 1;
    if (mode == 0) {
        block.offset[0]= block.offset[1] = block.offset[2] =0;
        nval = 1;
    } else {
        block.offset[0] = 0;
        block.offset[1] = 1;
        block.offset[2] = 2;
        if (mode == 1) {
            block.offset[3] = 0;
            block.pixelSize = 3;
            nval = 3;
        } else {
            block.offset[3] = 3;
            block.pixelSize = 4;
            nval = 4;
        }
    }

    if (bbox) {

      block.width  = destwidth;
      block.height = destheight;
      block.pitch = deststride;
      block.pixelPtr = destbuffer;

      Tk_PhotoPutBlock(photo, &block, destx, desty, destwidth, destheight);
      delete [] destbuffer;

    } else {
      block.width  = aggRenderer->get_width();
      block.height = aggRenderer->get_height();
      block.pitch = block.width * nval;
      block.pixelPtr =  aggRenderer->pixBuffer;

      /* Clear current contents */
      Tk_PhotoBlank(photo);
      /* Copy opaque block to photo image, and leave the rest to TK */
      Tk_PhotoPutBlock(photo, &block, 0, 0, block.width, block.height);
    }

    return TCL_OK;
}


static PyObject *
_pyobj_addr(PyObject* self, PyObject* args)
{
  PyObject *pyobj;
  if (!PyArg_ParseTuple(args, "O", &pyobj))
      return NULL;
  return Py_BuildValue("l", (long) pyobj);
}

static PyObject*
_tkinit(PyObject* self, PyObject* args)
{
    Tcl_Interp* interp;
    TkappObject* app;

    long arg;
    int is_interp;
    if (!PyArg_ParseTuple(args, "li", &arg, &is_interp))
        return NULL;

    if (is_interp) {
        interp = (Tcl_Interp*) arg;
    } else {
        /* Do it the hard way.  This will break if the TkappObject
           layout changes */
        app = (TkappObject*) arg;
        interp = app->interp;
    }

    /* This will bomb if interp is invalid... */

    Tcl_CreateCommand(interp, "PyAggImagePhoto",
		      (Tcl_CmdProc *) PyAggImagePhoto,
                      (ClientData) 0, (Tcl_CmdDeleteProc*) NULL);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef functions[] = {
    /* Tkinter interface stuff */
    {"_pyobj_addr", (PyCFunction)_pyobj_addr, 1},
    {"tkinit", (PyCFunction)_tkinit, 1},
    {NULL, NULL} /* sentinel */
};

extern "C"
DL_EXPORT(void) init_tkagg(void)
{
    Py_InitModule("_tkagg", functions);
}

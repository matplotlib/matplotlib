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

extern "C" {
#ifdef __APPLE__
#  include <Tcl/tcl.h>
#  include <Tk/tk.h>
#else
#  include <tk.h> 
#endif
};

#include "_backend_agg.h"

typedef struct {
    PyObject_HEAD
    Tcl_Interp* interp;
} TkappObject;

// const on win32
#ifdef WIN32
#define argv_t const char
#else
#define argv_t char
#endif
static int
PyAggImagePhoto(ClientData clientdata, Tcl_Interp* interp,
               int argc, argv_t **argv)
{
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;
    RendererAggObject* aggRenderer;
    long mode;
    long nval;
    if (argc != 4) {
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
    aggRenderer = (RendererAggObject *) atol(argv[2]);

    /* XXX insert aggRenderer type check */

    /* get array mode (0=mono, 1=rgb, 2=rgba) */
    mode = atol(argv[3]);
    if ((mode != 0) && (mode != 1) && (mode != 2)) {
        Tcl_AppendResult(interp, "illegal image mode", (char *) NULL);
        return TCL_ERROR;
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
    block.width  = aggRenderer->rbase->width();
    block.height = aggRenderer->rbase->height();
    block.pitch = block.width * nval;
    block.pixelPtr =  aggRenderer->buffer;
    /* Clear current contents */
    Tk_PhotoBlank(photo);
    /* Copy opaque block to photo image, and leave the rest to TK */
    Tk_PhotoPutBlock(photo, &block, 0, 0, block.width, block.height);

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

    Tcl_CreateCommand(interp, "PyAggImagePhoto", PyAggImagePhoto,
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

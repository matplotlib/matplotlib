#include <Python.h>
#include <stdio.h>

#ifdef NUMARRAY
#include "numarray/arrayobject.h"
#else
#include "Numeric/arrayobject.h"
#endif

extern long GcInit1(long, long, double *, double *, int *, short *, int , const double *, double, long *);
extern long GcInit2(long, long, double *, double *, int *, short *, int , const double *, double *,
                                    long, long *);
extern long GcTrace(long *n, double *px, double *py);


static PyObject * GcInit1_wrap(PyObject *self, PyObject *args)
{
  PyObject *ox, *oy, *oreg, *otriangle, *ozz;
  long imax, jmax, nparts=0, ntotal=0;
  int region=0;
  double lev=0.0;
  PyArrayObject *xdata, *ydata, *regdata, *tridata, *zzdata;
  int *zzsize;

  if (!PyArg_ParseTuple(args,"OOOOiOd", &ox, &oy, &oreg, &otriangle, &region, &ozz, &lev))
    return NULL;

  if (!PyArray_Check(ox))
  {
    PyErr_SetString(PyExc_TypeError, "Argument x must be an array");
    return NULL;
  }

  if (!PyArray_Check(oy))
  {
    PyErr_SetString(PyExc_TypeError, "Argument y must be an array");
    return NULL;
  }

  if (!PyArray_Check(oreg))
  {
    PyErr_SetString(PyExc_TypeError, "Argument reg must be an array");
     return NULL;
  }

  if (!PyArray_Check(otriangle))
  {
    PyErr_SetString(PyExc_TypeError, "Argument triangle must be an array");
    return NULL;
  }
  if (!PyArray_Check(ozz))
  {
    PyErr_SetString(PyExc_TypeError, "Argument z must be an array");
    return NULL;
    }

  xdata = (PyArrayObject *) PyArray_ContiguousFromObject(ox, 'd', 2, 2);
  ydata = (PyArrayObject *) PyArray_ContiguousFromObject(oy, 'd', 2, 2);
  regdata = (PyArrayObject *) PyArray_ContiguousFromObject(oreg, 'i', 2, 2);
  tridata = (PyArrayObject *) PyArray_ContiguousFromObject(otriangle, 's' , 2, 2);
  zzdata = (PyArrayObject *) PyArray_ContiguousFromObject(ozz, 'd', 2, 2);

  if (xdata->nd != 2 || ydata->nd != 2 || regdata->nd != 2 || tridata->nd != 2 || zzdata->nd !=2)
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_ValueError, "Argument must be a 2D array");
    return NULL;
  }

  zzsize = zzdata->dimensions;

  if ((zzsize[0] != xdata->dimensions[0]) || (zzsize[1] != xdata->dimensions[1]))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_ValueError, "Arrays x and z must have equal shapes");
    return NULL;
  }

  if ((zzsize[0] != ydata->dimensions[0]) || (zzsize[1] != ydata->dimensions[1]))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
     PyErr_SetString(PyExc_ValueError, "Arrays y and z must have equal shapes");
     return NULL;
  }

  //if ((zzsize[0] != regdata->dimensions[0]) || (zzsize[1] != regdata->dimensions[1]))
  //{
   //  PyErr_SetString(PyExc_ValueError, "Arrays reg and z must have equal shapes");
   //  return NULL;
   // }
  if ((zzsize[0] != tridata->dimensions[0]) || (zzsize[1] != tridata->dimensions[1]))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
     PyErr_SetString(PyExc_ValueError, "Arrays triangle and z must have equal shapes");
     return NULL;
  }

  imax = zzsize[1];
  jmax = zzsize[0];

   ntotal = GcInit1(imax, jmax, (double *) xdata->data, (double *) ydata->data, (int *)regdata->data, (short *) tridata->data, region, (double *) zzdata->data, lev, &nparts);

    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
   return Py_BuildValue("ll", ntotal, nparts);
}

static PyObject * GcInit2_wrap(PyObject *self, PyObject *args)
{
  PyObject *ox, *oy, *oreg, *otriangle, *ozz;
  long imax, jmax, nparts=0, ntotal=0, nchunk=1;   /* nchunk = 1?? */
  int region=0;
  double levs[2];
  PyArrayObject *xdata, *ydata, *regdata, *tridata, *zzdata;
  int *zzsize;

  if (!PyArg_ParseTuple(args,"OOOOiO(dd)l", &ox, &oy, &oreg, &otriangle,
                         &region, &ozz, levs, levs+1, &nchunk))
    return NULL;

  if (!PyArray_Check(ox))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_TypeError, "Argument x must be an array");
    return NULL;
  }

  if (!PyArray_Check(oy))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_TypeError, "Argument y must be an array");
    return NULL;
  }

  if (!PyArray_Check(oreg))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_TypeError, "Argument reg must be an array");
     return NULL;
  }

  if (!PyArray_Check(otriangle))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_TypeError, "Argument triangle must be an array");
    return NULL;
  }
  if (!PyArray_Check(ozz))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_TypeError, "Argument z must be an array");
    return NULL;
    }

  xdata = (PyArrayObject *) PyArray_ContiguousFromObject(ox, 'd', 2, 2);
  ydata = (PyArrayObject *) PyArray_ContiguousFromObject(oy, 'd', 2, 2);
  regdata = (PyArrayObject *) PyArray_ContiguousFromObject(oreg, 'i', 2, 2);
  tridata = (PyArrayObject *) PyArray_ContiguousFromObject(otriangle, 's' , 2, 2);
  zzdata = (PyArrayObject *) PyArray_ContiguousFromObject(ozz, 'd', 2, 2);

  if (xdata->nd != 2 || ydata->nd != 2 || regdata->nd != 2 || tridata->nd != 2 || zzdata->nd !=2)
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_ValueError, "Argument must be a 2D array");
    return NULL;
  }

  zzsize = zzdata->dimensions;

  if ((zzsize[0] != xdata->dimensions[0]) || (zzsize[1] != xdata->dimensions[1]))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
    PyErr_SetString(PyExc_ValueError, "Arrays x and z must have equal shapes");
    return NULL;
  }

  if ((zzsize[0] != ydata->dimensions[0]) || (zzsize[1] != ydata->dimensions[1]))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
     PyErr_SetString(PyExc_ValueError, "Arrays y and z must have equal shapes");
     return NULL;
  }

  //if ((zzsize[0] != regdata->dimensions[0]) || (zzsize[1] != regdata->dimensions[1]))
  //{
   //  PyErr_SetString(PyExc_ValueError, "Arrays reg and z must have equal shapes");
   //  return NULL;
   // }
  if ((zzsize[0] != tridata->dimensions[0]) || (zzsize[1] != tridata->dimensions[1]))
  {
    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
     PyErr_SetString(PyExc_ValueError, "Arrays triangle and z must have equal shapes");
     return NULL;
  }

  imax = zzsize[1];
  jmax = zzsize[0];

   ntotal = GcInit2(imax, jmax, (double *) xdata->data, (double *) ydata->data,
                     (int *)regdata->data, (short *) tridata->data,
                     region, (double *) zzdata->data, levs, nchunk, &nparts);

    Py_XDECREF(xdata); Py_XDECREF(ydata); Py_XDECREF(regdata); Py_XDECREF(tridata); Py_XDECREF(zzdata);       
    Py_XDECREF(ox); Py_XDECREF(oy); Py_XDECREF(ozz); Py_XDECREF(oreg); Py_XDECREF(otriangle);       
   return Py_BuildValue("ll", ntotal, nparts);
}



static PyObject * GcTrace_wrap(PyObject *self, PyObject *args)
{
  long ntotal;
  PyObject *oxp, *oyp, *onp;
  PyArrayObject *xcpdata, *ycpdata, *npdata;
  int npsize, n, p, start = 0, end = 0;
  PyObject *point, *contourList, *all_contours;

  if (!PyArg_ParseTuple(args,"OOO", &onp, &oxp, &oyp))
    return NULL;

  npdata = (PyArrayObject *) PyArray_ContiguousFromObject(onp, 'l', 1, 1);
  xcpdata = (PyArrayObject *) PyArray_ContiguousFromObject(oxp, 'd', 1, 1);
  ycpdata = (PyArrayObject *) PyArray_ContiguousFromObject(oyp, 'd', 1, 1 );

  if (npdata->nd != 1)
  {
    Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
    Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
     PyErr_SetString(PyExc_ValueError, "Argument np must be a 1D array");
     return NULL;
  }

  if (xcpdata->nd != 1)
  {
    Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
    Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
     PyErr_SetString(PyExc_ValueError, "Argument xp must be a 1D array");
     return NULL;
  }

  if (ycpdata->nd != 1)
  {
    Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
    Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
     PyErr_SetString(PyExc_ValueError, "Argument yp must be a 1D array");
     return NULL;
  }


  ntotal = GcTrace( (long *) npdata->data, (double *) xcpdata->data, (double *) ycpdata->data);

  if (ntotal<0)
    {
    Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
    Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
     PyErr_SetString(PyExc_ValueError, "Illegal return value ntotal in GcTrace");
     return NULL;
  }

  npsize = PyArray_Size((PyObject *) npdata);

  all_contours = PyList_New(0);

  for (n = 0; n<npsize; n++)
     {
	start = end;
	end = start + ((long *) npdata->data)[n];
	contourList = PyList_New(0);
	//printf("_c %d, %d\n", start, end);
	for (p = start; p<end; p++)
	{
	  //printf("\t%d, %f, %f\n", p, ((double *) xcpdata->data)[p], ((double *) ycpdata->data)[p]);
	   point = Py_BuildValue("(d,d)", ((double *) xcpdata->data)[p],((double *) ycpdata->data)[p]);
	   if (PyList_Append(contourList, point))
	   {
	     Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
	     Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
	      printf ("Error in appending to list\n");
	      return NULL;
	      }
	  }
	if (PyList_Append(all_contours, contourList))
	  {
	    Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
	    Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
	    printf ("error in appending to all_contours\n");
	    return NULL;
	  }
	}
  Py_XDECREF(npdata); Py_XDECREF(xcpdata); Py_XDECREF(ycpdata);
  Py_XDECREF(oxp); Py_XDECREF(oyp); Py_XDECREF(onp); 
  return Py_BuildValue("O", all_contours);
}




static PyMethodDef contour_methods[] =
  {
    {"GcInit1", GcInit1_wrap, METH_VARARGS, "wrap gcinit1"},
    {"GcInit2", GcInit2_wrap, METH_VARARGS, "wrap gcinit2"},
    {"GcTrace", GcTrace_wrap, METH_VARARGS, "wrap gctrace"},
    {NULL, NULL, 0, NULL}
  };


#if defined(_MSC_VER)
DL_EXPORT(void)
#elif defined(__cplusplus)
  extern "C" void
#else
void
#endif

#if defined(NUMARRAY)
init_na_contour(void)
{
  (void) Py_InitModule("_na_contour", contour_methods);
  import_array();
}
#else
init_nc_contour(void)
{
 (void) Py_InitModule("_nc_contour", contour_methods);
  import_array();
}
#endif





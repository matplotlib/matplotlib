#include <Python.h>
#include <stdio.h>

#ifdef NUMARRAY
#include "numarray/arrayobject.h" 
#else
#include "Numeric/arrayobject.h" 
#endif   

extern long GcInit1(long, long, double *, double *, int *, short *, int , const double *, double, long *);
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
    PyErr_SetString(PyExc_ValueError, "Argument must be a 2D array");
    return NULL;
  }

  zzsize = zzdata->dimensions;
  
  if ((zzsize[0] != xdata->dimensions[0]) || (zzsize[1] != xdata->dimensions[1])) 
  {
    PyErr_SetString(PyExc_ValueError, "Arrays x and z must have equal shapes");   
    return NULL;
  }

  if ((zzsize[0] != ydata->dimensions[0]) || (zzsize[1] != ydata->dimensions[1])) 
  {
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
     PyErr_SetString(PyExc_ValueError, "Arrays triangle and z must have equal shapes");   
     return NULL;
  }
  
  imax = zzsize[1];
  jmax = zzsize[0];

   ntotal = GcInit1(imax, jmax, (double *) xdata->data, (double *) ydata->data, (int *)regdata->data, (short *) tridata->data, region, (double *) zzdata->data, lev, &nparts);

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
     PyErr_SetString(PyExc_ValueError, "Argument np must be a 1D array");   
     return NULL;
  }

  if (xcpdata->nd != 1)
  {
     PyErr_SetString(PyExc_ValueError, "Argument xp must be a 1D array");   
     return NULL;
  }

  if (ycpdata->nd != 1)
  {
     PyErr_SetString(PyExc_ValueError, "Argument yp must be a 1D array");
     return NULL;
  }


  ntotal = GcTrace( (long *) npdata->data, (double *) xcpdata->data, (double *) ycpdata->data);

  if (ntotal<0)
    {
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
	      printf ("Error in appending to list\n");
	      return NULL;
	      }
	  }
	if (PyList_Append(all_contours, contourList))
	  {
	    printf ("error in appending to all_contours\n");
	    return NULL;
	  }
	}
return Py_BuildValue("O", all_contours);  
}




static PyMethodDef contour_methods[] =
  {
    {"GcInit1", GcInit1_wrap, METH_VARARGS, "wrap gcinit1"},
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





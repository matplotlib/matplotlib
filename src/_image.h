/* image.h	
 *
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include "Python.h"

#include "agg_affine_matrix.h"
#include "agg_rendering_buffer.h"


 
typedef struct {
  PyObject_HEAD
  PyObject	*x_attr;	/* Attributes dictionary */
  
  agg::int8u *bufferIn;
  agg::rendering_buffer *rbufIn;
  size_t colsIn, rowsIn;             

  agg::int8u *bufferOut;
  agg::rendering_buffer *rbufOut;
  size_t rowsOut, colsOut;             
  unsigned BPP;

  unsigned interpolation, aspect;

  agg::affine_matrix srcMatrix, imageMatrix;

} ImageObject;



#endif


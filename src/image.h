/* image.h	
 *
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include "Python.h"

#include "agg_rendering_buffer.h"


 
typedef struct {
  PyObject_HEAD
  PyObject	*x_attr;	/* Attributes dictionary */
  
  agg::int8u *bufferIn;
  agg::rendering_buffer *rbufIn;
  size_t widthIn, heightIn;             // the number of bytes in buffer

  agg::int8u *bufferOut;
  agg::rendering_buffer *rbufOut;
  size_t widthOut, heightOut;             // the number of bytes in buffer
  unsigned BPP;

  unsigned interpolation, aspect;

} ImageObject;




#endif


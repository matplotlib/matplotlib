/* A python interface to freetype2 */
#ifndef _FT2FONT_H
#define _FT2FONT_H
#define MAXGLYPHS 1000

#include "Python.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H

// the freetype string renderered into a width, height buffer
typedef struct {
  unsigned char *buffer;
  unsigned int width;
  unsigned int height;
} FT2_Image;


typedef struct {
  PyObject_HEAD
  PyObject	*x_attr;	        /* Attributes dictionary */
  FT_Face       face;

  FT_Matrix     matrix;                 /* transformation matrix */
  FT_UInt       glyph_index;
  FT_Vector     pen;                    /* untransformed origin  */
  FT_Error      error;
  FT_Glyph      glyphs[MAXGLYPHS];
  FT_Vector     pos   [MAXGLYPHS];
  char          *text;
  double angle;
  int num_chars;
  int num_glyphs;
  FT2_Image image;
} FT2FontObject;

static PyTypeObject FT2Font_Type;

#endif

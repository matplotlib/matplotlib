/******************************************************************
Copyright 2000 by Object Craft P/L, Melbourne, Australia.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in
supporting documentation, and that the name of Object Craft
is not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

OBJECT CRAFT DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
EVENT SHALL OBJECT CRAFT BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

******************************************************************/

#include "_backend_agg.h"

#define TT_VALID(handle) ((handle).z != NULL)

void set_error(PyObject* err, char *fmt, ...)
{
  char msg[1024];
  va_list ap;
  
  va_start(ap, fmt);
  vsprintf(msg, fmt, ap);
  va_end(ap);
  PyErr_SetString(err, msg);
}

static TT_Engine engine;
static int engine_initialised;

static double tt2double(TT_F26Dot6 num)
{
  return (num >> 6) + (num & 0x3f) / 64.0;
}

static TT_F26Dot6 double2tt(double num)
{
  return (TT_F26Dot6)(num * 64);
}

static long tt2int(TT_F26Dot6 num)
{
  return num / 64L;
}

static long tt_frac(TT_F26Dot6 num)
{
  return num & 0x3f;
}

static long tt_trunc(TT_F26Dot6 num)
{
  return num & ~0x3f;
}

static int tt_ceil(TT_F26Dot6 num)
{
  return (num + 63) & ~63;
}

static TT_F26Dot6 int2tt(int num)
{
  return num * 64L;
}

static void clear_raster(TT_Raster_Map *bit)
{
  memset(bit->bitmap, 0, bit->size);
}

static void alloc_raster(TT_Raster_Map *bit, int width, int height)
{
  bit->rows = height;
  bit->width = (width + 3) & -4;
  bit->flow = TT_Flow_Up;
  bit->cols = bit->width;
  bit->size = bit->rows * bit->width;
  
  bit->bitmap = malloc(bit->size);
}

static void free_raster(TT_Raster_Map *bit)
{
  free(bit->bitmap);
  bit->bitmap = NULL;
}


static agg::int8u *raster_to_pixbuf(Raster *raster, int color)
{
  agg::int8u *pixels = new agg::int8u[raster->bit.width * raster->bit.rows * 4];
  int i;
  agg::int8u rgb[3], alpha;
  static agg::int8u xlat[] = { 0, 0x3f, 0x7f, 0xbf, 0xff };
  
  rgb[0] = (color >> 24) & 0xff;
  rgb[1] = (color >> 16) & 0xff;
  rgb[2] = (color >>  8) & 0xff;
  alpha = color & 0xff;
  
  for (i = 0; i < raster->bit.rows; i++) {
    agg::int8u *rgba;
    agg::int8u *col;
    int j;
    
    rgba = pixels + i * raster->bit.width * 4;
    col = (agg::int8u *)raster->bit.bitmap + (raster->bit.rows - i - 1) * (raster->bit.width);
    for (j = 0; j < raster->bit.width; j++, col++) {
      if (*col == 0) {
	*rgba++ = 0;
	*rgba++ = 0;
	*rgba++ = 0;
	*rgba++ = 0;
      } else {
	*rgba++ = rgb[0];
	*rgba++ = rgb[1];
	*rgba++ = rgb[2];
	if (*col < sizeof(xlat))
	  *rgba++ = (xlat[*col] * alpha) / 0xff;
	else
	  *rgba++ = alpha;
      }
    }
  }
  return pixels;
}

static void font_free_raster(FontObj *self, Raster *raster)
{
  free_raster(&raster->bit);
  free_raster(&raster->small_bit);
  free(raster);
}

typedef struct {
  int max_advance;
  TT_F26Dot6 pen_x, pen_y;
} Metrics;

static void font_text_width(FontObj *self, char *text, int text_len,
			    Metrics *metrics)
{
  int i;
  TT_F26Dot6 vec_x, vec_y;
  
  metrics->max_advance = 0;
  for (i = 0; i < text_len; i++) {
    unsigned char j = text[i];
    TT_Glyph_Metrics gmetrics;
    
    if (!TT_VALID(self->glyphs[j]))
      continue;
    
    TT_Get_Glyph_Metrics(self->glyphs[j], &gmetrics);
    
    if (gmetrics.advance > metrics->max_advance)
      metrics->max_advance = gmetrics.advance;
    vec_x = gmetrics.advance;
    vec_y = 0;
    if (self->rotate)
      TT_Transform_Vector(&vec_x, &vec_y, &self->matrix);
    metrics->pen_x += vec_x;
    metrics->pen_y += vec_y;
  }
}

static void font_calc_size(FontObj *self, TT_F26Dot6 x, TT_F26Dot6 y,
			   char *text, int text_len, int *width, int *height)
{
  Metrics metrics;
  TT_F26Dot6 height_x, height_y;
  
  height_x = 0;
  height_y = self->ascent - self->descent;
  if (self->rotate)
    TT_Transform_Vector(&height_x, &height_y, &self->matrix);
  
  metrics.pen_x = tt_frac(x);
  metrics.pen_y = tt_frac(y);
  font_text_width(self, text, text_len, &metrics);
  
  *width = tt2int(tt_ceil(labs(metrics.pen_x) + labs(height_x)));
  *height = tt2int(tt_ceil(labs(metrics.pen_y) + labs(height_y)));
}

static Raster *font_build_raster(FontObj *self, TT_F26Dot6 x, TT_F26Dot6 y,
				 char *text, int text_len)
{
  Raster *raster;
  Metrics metrics;
  TT_F26Dot6 vec_x, vec_y;
  TT_F26Dot6 height_x, height_y;
  
  height_x = 0;
  height_y = self->ascent - self->descent;
  if (self->rotate)
    TT_Transform_Vector(&height_x, &height_y, &self->matrix);
  
  raster = (Raster *)malloc(sizeof(*raster));
  memset(raster, 0, sizeof(*raster));
  
  metrics.pen_x = tt_frac(x);
  metrics.pen_y = tt_frac(y);
  font_text_width(self, text, text_len, &metrics);
  
  raster->width = tt2int(tt_ceil(labs(metrics.pen_x) + labs(height_x)));
  raster->height = tt2int(tt_ceil(labs(metrics.pen_y) + labs(height_y)));
  
  alloc_raster(&raster->bit, raster->width, raster->height);
  clear_raster(&raster->bit);
  
  vec_x = metrics.max_advance;
  vec_y = 0;
  if (self->rotate)
    TT_Transform_Vector(&vec_x, &vec_y, &self->matrix);
  alloc_raster(&raster->small_bit,
	       tt2int(tt_ceil(labs(vec_x) + labs(height_x))),
	       tt2int(tt_ceil(labs(vec_y) + labs(height_y))));
  return raster;
}

static void blit_or(TT_Raster_Map *dst, TT_Raster_Map *src,
		    int x_off, int y_off)
{
  int x, y;
  int x1, x2, y1, y2;
  char *s, *d;
  
  x1 = x_off < 0 ? -x_off : 0;
  y1 = y_off < 0 ? -y_off : 0;
  
  x2 = (int)dst->cols - x_off;
  if (x2 > src->cols)
    x2 = src->cols;
  
  y2 = (int)dst->rows - y_off;
  if (y2 > src->rows)
    y2 = src->rows;
  
  if (x1 >= x2)
    return;
  
  for (y = y1; y < y2; y++) {
    s = ((char*)src->bitmap) + y * src->cols + x1;
    d = ((char*)dst->bitmap) + (y + y_off) * dst->cols + x1 + x_off;
    
    for (x = x1; x < x2; ++x)
      *d++ |= *s++;
    
  }
}

static void font_render_glyphs(FontObj *self, Raster *raster,
			       TT_F26Dot6 *x, TT_F26Dot6 *y,
			       char* text, int text_len)
{
  TT_F26Dot6 image_x, image_y; /* bottom left of text pixmap in image */
  TT_F26Dot6 pen_x, pen_y;	/* glyph posn in small pixmap */
  TT_F26Dot6 off_x, off_y;	/* glyph offset for rendering */
  TT_F26Dot6 blit_x = 0, blit_y = 0;
  int i;
  
  /* calculate the bottom left corner of text pixmap in image */
  image_x = tt_trunc(*x + self->offset_x);
  if (self->quadrant == 1 || self->quadrant == 2)
    image_x -= int2tt(raster->bit.width);
  image_y = tt_trunc(*y + self->offset_y);
  if (self->quadrant == 2 || self->quadrant == 3)
    image_y -= int2tt(raster->bit.rows);
  for (i = 0; i < text_len; i++) {
    unsigned char j = text[i];
    TT_Glyph glyph = self->glyphs[j];
    TT_Outline outline;
    TT_Glyph_Metrics gmetrics;
    TT_F26Dot6 vec_x, vec_y; /* advance pen calc */
    
    /* calculate pen and blit position */
    switch (self->quadrant) {
    case 0:
      blit_x = tt_trunc(*x + self->offset_x) - image_x;
      blit_y = tt_trunc(*y + self->offset_y) - image_y;
      break;
      
    case 1:
      blit_x = tt_trunc(*x + self->offset_x) - image_x - int2tt(raster->small_bit.width);
      blit_y = tt_trunc(*y + self->offset_y) - image_y;
      break;
      
    case 2:
      blit_x = tt_trunc(*x + self->offset_x) - image_x - int2tt(raster->small_bit.width);
      blit_y = tt_trunc(*y + self->offset_y) - image_y - int2tt(raster->small_bit.rows);
      break;
      
    case 3:
      blit_x = tt_trunc(*x + self->offset_x) - image_x;
      blit_y = tt_trunc(*y + self->offset_y) - image_y - int2tt(raster->small_bit.rows);
      break;
    }
    pen_x = *x - image_x - blit_x;
    pen_y = *y - image_y - blit_y;
    
    if (!TT_VALID(glyph))
      continue;
    /* offset glyph outline by fractional pixels */
    off_x = tt_frac(pen_x);
    off_y = tt_frac(pen_y);
    TT_Get_Glyph_Outline(glyph, &outline);
    TT_Translate_Outline(&outline, off_x, off_y);
    TT_Get_Glyph_Metrics(glyph, &gmetrics);
    /* render glyph in small pixmap then blit into big pixmap */
    clear_raster(&raster->small_bit);
    TT_Get_Glyph_Pixmap(glyph, &raster->small_bit,
			tt_trunc(pen_x), tt_trunc(pen_y));
    blit_or(&raster->bit, &raster->small_bit,
	    tt2int(blit_x), tt2int(blit_y));
    /* undo glyph offset */
    TT_Translate_Outline(&outline, -off_x, -off_y);
    /* work out pen advance */
    vec_x = gmetrics.advance;
    vec_y = 0;
    if (self->rotate)
      TT_Transform_Vector(&vec_x, &vec_y, &self->matrix);
    /* advance position passed in */
    *x += vec_x;
    *y += vec_y;
  }
}

static TT_Error font_load_glyphs(FontObj *self, char* text, int text_len)
{
  unsigned short i, n, code, load_flags;
  unsigned short num_glyphs = 0, no_cmap = 0;
  unsigned short platform, encoding;
  TT_Error error;
  TT_CharMap char_map;
  
  /* look for a Unicode charmap */
  n = TT_Get_CharMap_Count(self->face);
  for (i = 0; i < n; i++) {
    TT_Get_CharMap_ID(self->face, i, &platform, &encoding);
    if ((platform == 3 && encoding == 1) ||
	(platform == 0 && encoding == 0)) {
      TT_Get_CharMap(self->face, i, &char_map);
      break;
    }
  }
  if (i == n) {
    TT_Face_Properties properties;
    
    TT_Get_Face_Properties(self->face, &properties);
    no_cmap = 1;
    num_glyphs = properties.num_Glyphs;
  }
  
  if (self->glyphs == NULL) {
    self->glyphs = (TT_Glyph*)malloc(256 * sizeof(*self->glyphs));
    memset(self->glyphs, 0, 256 * sizeof(*self->glyphs));
  }
  load_flags = TTLOAD_SCALE_GLYPH;
  if (self->hinted)
    load_flags |= TTLOAD_HINT_GLYPH;
  
  for (i = 0; i < text_len; i++) {
    unsigned char j = text[i];
    TT_Outline outline;
    
    if (TT_VALID(self->glyphs[j]))
      continue;
    
    if (no_cmap) {
      code = (j - ' ' + 1) < 0 ? 0 : (j - ' ' + 1);
      if (code >= num_glyphs)
	code = 0;
    } else
      code = TT_Char_Index(char_map, j);
    
    error = TT_New_Glyph(self->face, &self->glyphs[j]);
    if (error)
      return error;
    error = TT_Load_Glyph(self->instance, self->glyphs[j], code, load_flags);
    error = TT_Get_Glyph_Outline(self->glyphs[j], &outline);
    if (error)
      return error;
    if (self->rotate)
      TT_Transform_Outline(&outline, &self->matrix);
  }
  return 0;
}

static PyObject *make_xy_tuple(TT_F26Dot6 x, TT_F26Dot6 y)
{
  PyObject *tup;
  PyObject *num;
  
  tup = PyTuple_New(2);
  if (tup == NULL)
    return NULL;
  if ((num = PyFloat_FromDouble(tt2double(x))) == NULL) {
    Py_DECREF(tup);
    return NULL;
  }
  if (PyTuple_SetItem(tup, 0, num) < 0)
    return NULL;
  if ((num = PyFloat_FromDouble(tt2double(y))) == NULL) {
    Py_DECREF(tup);
    return NULL;
  }
  if (PyTuple_SetItem(tup, 1, num) < 0)
    return NULL;
  return tup;
}

static char font_advance__doc__[] = 
"advance(text)\n"
"\n"
"Return the (x, y) pen advance for drawing text";

static PyObject *font_advance(FontObj *self, PyObject *args)
{
  char *text;
  int text_len;
  TT_Error error;
  Metrics metrics;
  
  if (!PyArg_ParseTuple(args, "s#", &text, &text_len))
    return NULL;
  
  error = font_load_glyphs(self, text, text_len);
  if (error) {
    set_error(PyExc_RuntimeError,
	      "freetype error 0x%x; loading glyphs", error);
    return NULL;
  }
  metrics.pen_x = metrics.pen_y = 0;
  font_text_width(self, text, text_len, &metrics);
  
  return make_xy_tuple(metrics.pen_x, -metrics.pen_y);
}

static char font_transform__doc__[] = 
"transform(x, y)\n"
"\n"
"Return (x, y) modified by the font transformation matrix";

static PyObject *font_transform(FontObj *self, PyObject *args)
{
  double x, y;
  TT_F26Dot6 tt_x, tt_y;
  
  if (!PyArg_ParseTuple(args, "dd", &x, &y))
    return NULL;
  
  tt_x = double2tt(x);
  tt_y = double2tt(y);
  if (self->rotate)
    TT_Transform_Vector(&tt_x, &tt_y, &self->matrix);
  
  return make_xy_tuple(tt_x, -tt_y);
}

static char font_textsize__doc__[] = 
"textsize(text)\n"
"\n"
"Return (width, height) of the text which would be drawn";

static PyObject *font_textsize(FontObj *self, PyObject *args)
{
  char *text;
  int text_len;
  int width, height;
  TT_Error error;
  
  if (!PyArg_ParseTuple(args, "s#",
			&text, &text_len))
    return NULL;
  
  error = font_load_glyphs(self, text, text_len);
  if (error) {
    set_error(PyExc_RuntimeError,
	      "freetype error 0x%x; loading glyphs", error);
    return NULL;
  }
  
  font_calc_size(self, 0, 0, text, text_len, &width, &height);
  
  return make_xy_tuple(int2tt(width), int2tt(height));
}

static struct PyMethodDef font_methods[] = {
  { "advance", (PyCFunction)font_advance, METH_VARARGS, font_advance__doc__ },
  { "transform", (PyCFunction)font_transform, METH_VARARGS, font_transform__doc__ },
  { "textsize",(PyCFunction)font_textsize, METH_VARARGS, font_textsize__doc__ },
  { NULL, NULL }		/* sentinel */
};

static FontObj *new_FontObj(char *filename, double point_size, double rotate)
{
  FontObj *self;
  TT_Error error;
  int upm;
  TT_Instance_Metrics imetrics;
  
  if (!engine_initialised) {
    error = TT_Init_FreeType(&engine);
    if (error) {
      set_error(PyExc_RuntimeError,
		"freetype error 0x%x; initializing freetype engine", error);
      return NULL;
    }
    engine_initialised = 1;
  }
  
  self = PyObject_NEW(FontObj, &Font_Type);
  if (self == NULL)
    return NULL;
  
  self->point_size = double2tt(point_size);
  self->dpi = 96;
  self->hinted = 1;
  
  self->face.z = NULL;
  memset(&self->properties, 0, sizeof(self->properties));
  self->instance.z = NULL;
  self->glyphs = NULL;
  
  error = TT_Open_Face(engine, filename, &self->face);
  if (error) {
    if (error == TT_Err_Could_Not_Open_File)
      set_error(PyExc_IOError, "could not open file");
    else
      set_error(PyExc_RuntimeError,
		"freetype error 0x%x; opening %s", error, filename);
    Py_DECREF(self);
    return NULL;
  }
  TT_Get_Face_Properties(self->face, &self->properties);
  error = TT_New_Instance(self->face, &self->instance);
  if (!error)
    error = TT_Set_Instance_Resolutions(self->instance,
					self->dpi, self->dpi);
  if (!error)
    error = TT_Set_Instance_CharSize(self->instance, self->point_size);
  
  TT_Set_Instance_Transform_Flags(self->instance, 1, 0);
  
  if (rotate) {
    rotate = fmod(rotate, 360.0);
    if (rotate < 0.0)
      rotate += 360.0;
    if (rotate < 180.0) {
      if (rotate < 90.0)
	self->quadrant = 0;
      else
	self->quadrant = 1;
    } else {
      if (rotate < 270.0)
	self->quadrant = 2;
      else
	self->quadrant = 3;
    }
    self->quadrant = 3 - self->quadrant;
    self->rotate = (rotate * M_PI / 180.0);
    
    self->matrix.xx = (TT_Fixed)(cos(self->rotate) * (1L << 16));
    self->matrix.xy = (TT_Fixed)(sin(self->rotate) * (1L << 16));
    self->matrix.yx = -self->matrix.xy;
    self->matrix.yy = self->matrix.xx;
  } else {
    self->rotate = 0;
    self->quadrant = 0;
  }
  
  TT_Get_Instance_Metrics(self->instance, &imetrics);
  upm = self->properties.header->Units_Per_EM;
  self->ascent = int2tt(self->properties.horizontal->Ascender * imetrics.y_ppem) / upm;
  self->descent = int2tt(self->properties.horizontal->Descender * imetrics.y_ppem) / upm;
  self->line_gap = int2tt(self->properties.horizontal->Line_Gap * imetrics.y_ppem) / upm;
  
  self->offset_x = 0;
  self->offset_y = self->descent;
  if (self->rotate) {
    TT_F26Dot6 ascent_x, ascent_y, descent_x, descent_y;
    
    ascent_x = 0;
    ascent_y = self->ascent;
    TT_Transform_Vector(&ascent_x, &ascent_y, &self->matrix);
    descent_x = 0;
    descent_y = self->descent;
    TT_Transform_Vector(&descent_x, &descent_y, &self->matrix);
    switch (self->quadrant) {
    case 0:
    case 2:
      self->offset_x = ascent_x;
      self->offset_y = descent_y;
      break;
    case 1:
    case 3:
      self->offset_x = descent_x;
      self->offset_y = ascent_y;
      break;
    }
  }
  if (error) {
    set_error(PyExc_RuntimeError,
	      "freetype error 0x%x; initialising font instance", error);
    Py_DECREF(self);
    return NULL;
  }
  
  return self;
}

static void dealloc_FontObj(FontObj *self)
{
  int  i;
  
  if (self->glyphs != NULL) {
    for (i = 0; i < 256; i++)
      TT_Done_Glyph(self->glyphs[i]);
    free(self->glyphs);
  }
  if (TT_VALID(self->instance))
    TT_Done_Instance(self->instance);
  if (TT_VALID(self->face))
    TT_Close_Face(self->face);
  PyMem_DEL(self);
}

static PyObject *font_getattr(FontObj *self, char *name)
{
  switch (name[0]) {
  case 'a':
    if (strcmp(name, "ascent") == 0)
      return PyFloat_FromDouble(tt2double(self->ascent));
    break;
  case 'd':
    if (strcmp(name, "descent") == 0)
      return PyFloat_FromDouble(tt2double(self->descent));
    break;
  case 'h':
    if (strcmp(name, "height") == 0)
      return PyFloat_FromDouble(tt2double(self->ascent - self->descent));
    break;
  case 'l':
    if (strcmp(name, "line_gap") == 0)
      return PyFloat_FromDouble(tt2double(self->line_gap));
    break;
  case 'p':
    if (strcmp(name, "point_size") == 0)
      return PyFloat_FromDouble(tt2double(self->point_size));
    break;
  case 'q':
    if (strcmp(name, "quadrant") == 0)
      return PyInt_FromLong(3 - self->quadrant);
    break;
  case 'r':
    if (strcmp(name, "rotate") == 0)
      return PyFloat_FromDouble(self->rotate);
    break;
  }
  return Py_FindMethod(font_methods, (PyObject *)self, name);
}

static char Font_Type__doc__[] = 
"";

PyTypeObject Font_Type = {
  PyObject_HEAD_INIT(0)
  0,				/*ob_size*/
  "Font",			/*tp_name*/
  sizeof(FontObj),		/*tp_basicsize*/
  0,				/*tp_itemsize*/
  /* methods */
  (destructor)dealloc_FontObj, /*tp_dealloc*/
  (printfunc)0,		/*tp_print*/
  (getattrfunc)font_getattr,	/*tp_getattr*/
  (setattrfunc)0,		/*tp_setattr*/
  (cmpfunc)0,			/*tp_compare*/
  (reprfunc)0,		/*tp_repr*/
  0,				/*tp_as_number*/
  0,				/*tp_as_sequence*/
  0,				/*tp_as_mapping*/
  (hashfunc)0,		/*tp_hash*/
  (ternaryfunc)0,		/*tp_call*/
  (reprfunc)0,		/*tp_str*/
  
  /* Space for future expansion */
  0L, 0L, 0L, 0L,
  Font_Type__doc__ /* Documentation string */
};


PyObject *font_draw_text(RendererAggObject *renderer, PyObject *args)
{
  //printf("font_draw_text\n");
  FontObj *font_obj;
  double image_x, image_y;
  TT_F26Dot6 x, y, orig_y;
  char *text;
  int color, text_len;
  TT_Error error;
  Raster *raster;
  double affine[6];
  agg::int8u *pixels;
  
  if (!PyArg_ParseTuple(args, "O!ddis#",
			&Font_Type, (PyObject*)&font_obj,
			&image_x, &image_y, &color, &text, &text_len))
    return NULL;

  //printf("font_draw_text args parsed\n");  
  error = font_load_glyphs(font_obj, text, text_len);
  if (error) {
    set_error(PyExc_RuntimeError,
	      "freetype error 0x%x; loading glyphs", error);
    return NULL;
  }
  
  x = double2tt(image_x);
  orig_y = y = double2tt(image_y);
  raster = font_build_raster(font_obj, x, y, text, text_len);
  //printf("font_draw_text built raster\n");
  affine[0] = 1;
  affine[1] = 0;
  affine[2] = 0;
  affine[3] = 1;
  affine[4] = tt2int(tt_trunc(x + font_obj->offset_x));
  affine[5] = tt2int(tt_trunc(y - font_obj->offset_y)) - raster->bit.rows;
  if (font_obj->quadrant == 1 || font_obj->quadrant == 2)
    affine[4] -= raster->bit.width;
  if (font_obj->quadrant == 2 || font_obj->quadrant == 3)
    affine[5] += raster->bit.rows;
  font_render_glyphs(font_obj, raster, &x, &y, text, text_len);
  //printf("font_draw_text built rastering to pixbuf\n");
  pixels = raster_to_pixbuf(raster, color);
  //printf("font_draw_text built rastering to pixbuf done\n");

  agg::rendering_buffer rbuf;
  rbuf.attach(pixels, raster->bit.width, raster->bit.rows, raster->bit.width * 4);

  int xint((int)image_x);
  int yint((int)image_y);
  // blend the raster text in with the buffer
  pixfmt pixf(rbuf);
  pixfmt::color_type p;
  for (int i=0; i<raster->bit.width; ++i)
    for (int j=0; j<raster->bit.rows; ++j) {
      p = pixf.pixel(i,j);
      //printf("%d, %d, %d\n", i, j, p.a);
      renderer->pixf->blend_pixel(i+xint, j+yint, p, p.a);
    }

  delete pixels;
  
  font_free_raster(font_obj, raster);
  
  return make_xy_tuple(x, orig_y + (orig_y - y));
}

PyObject *font_new(PyObject *args)
{
  char *filename;
  double point_size = 12.0;
  double rotate = 0.0;
  
  if (!PyArg_ParseTuple(args, "s|dd", &filename, &point_size, &rotate))
    return NULL;
  return (PyObject*)new_FontObj(filename, point_size, -rotate);
}

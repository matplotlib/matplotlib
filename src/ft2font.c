#include "ft2font.h"

static PyObject *ErrorObject;

static PyTypeObject FT2Font_Type;
static PyTypeObject Glyph_Type;
static int FT2Font_setattr(FT2FontObject *self, char *name, PyObject *v);

int           _initLib = 0;
FT_Library    _ft2Library;
#define FT2FontObject_Check(v)	((v)->ob_type == &FT2Font_Type)
#define GlyphObject_Check(v)	((v)->ob_type == &Glyph_Type)


static PyMethodDef Glyph_methods[] = {
  {NULL,		NULL}		/* sentinel */
};

static void
Glyph_dealloc(GlyphObject *self)
{
  // do I need to dealloc all the ints?
  Py_XDECREF(self->x_attr);
  PyObject_Del(self);
}

static PyObject *
Glyph_getattr(GlyphObject *self, char *name)
{
  if (self->x_attr != NULL) {
    PyObject *v = PyDict_GetItemString(self->x_attr, name);
    if (v != NULL) {
      Py_INCREF(v);
      return v;
    }
  }
  return Py_FindMethod(Glyph_methods, (PyObject *)self, name);
}

static int
Glyph_setattr(GlyphObject *self, char *name, PyObject *v)
{
  if (self->x_attr == NULL) {
    self->x_attr = PyDict_New();
    if (self->x_attr == NULL)
      return -1;
  }

  if (v == NULL) {
    int rv = PyDict_DelItemString(self->x_attr, name);
    if (rv < 0)
      PyErr_SetString(PyExc_AttributeError,
		      "delete non-existing Glyph attribute");
    return rv;
  }
  else {
    return PyDict_SetItemString(self->x_attr, name, v);
  }
}

static PyTypeObject Glyph_Type = {
  /* The ob_type field must be initialized in the module init function
   * to be portable to Windows without using C++. */
  PyObject_HEAD_INIT(NULL)
  0,			/*ob_size*/
  "ft2font.Glyph",		/*tp_name*/
  sizeof(GlyphObject),	/*tp_basicsize*/
  0,			/*tp_itemsize*/
  /* methods */
  (destructor)Glyph_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)Glyph_getattr, /*tp_getattr*/
  (setattrfunc)Glyph_setattr, /*tp_setattr*/
  0,			/*tp_compare*/
  0,			/*tp_repr*/
  0,			/*tp_as_number*/
  0,			/*tp_as_sequence*/
  0,			/*tp_as_mapping*/
  0,			/*tp_hash*/
  0,                      /*tp_call*/
  0,                      /*tp_str*/
  0,                      /*tp_getattro*/
  0,                      /*tp_setattro*/
  0,                      /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,     /*tp_flags*/
  0,                      /*tp_doc*/
  0,                      /*tp_traverse*/
  0,                      /*tp_clear*/
  0,                      /*tp_richcompare*/
  0,                      /*tp_weaklistoffset*/
  0,                      /*tp_iter*/
  0,                      /*tp_iternext*/
  0,                      /*tp_methods*/
  0,                      /*tp_members*/
  0,                      /*tp_getset*/
  0,                      /*tp_base*/
  0,                      /*tp_dict*/
  0,                      /*tp_descr_get*/
  0,                      /*tp_descr_set*/
  0,                      /*tp_dictoffset*/
  0,                      /*tp_init*/
  0,                      /*tp_alloc*/
  0,                      /*tp_new*/
  0,                      /*tp_free*/
  0,                      /*tp_is_gc*/
};
/* --------------------------------------------------------------------- */




static FT2FontObject *
newFT2FontObject(PyObject *args)
{
  int error;
  char* facefile;
  if (! _initLib) { 
    error = FT_Init_FreeType( &_ft2Library ); //initialize library 

    if (error) {
      PyErr_SetString(PyExc_RuntimeError, 
		      "Could not find initialize the freetype2 library");
      return NULL;
    }
    _initLib = 1;

  }

  
  if (!PyArg_ParseTuple(args, "s:FT2Font", &facefile))
    return NULL;

  

  FT2FontObject *self;
  self = PyObject_New(FT2FontObject, &FT2Font_Type);
  self->image.buffer = NULL;
  self->image.width = 0;
  self->image.height = 0;
  self->text = NULL;
  self->angle = 0.0;
  self->num_chars = 0;
  self->num_glyphs = 0;
  self->pen.x = 0;
  self->pen.y = 0;


  error = FT_New_Face( _ft2Library, facefile, 0, &self->face );
  if (error) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not load the facefile");
    return NULL;
  }
  
  // set a default fontsize 12 pt at 72dpi
  error = FT_Set_Char_Size( self->face, 12 * 64, 0, 72, 72 );
  if (error) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not set the fontsize");
    return NULL;
  }
  

  if (self == NULL)
    return NULL;
  self->x_attr = NULL;
  
  // set some face props as attributes
  const char*  ps_name;
  ps_name = FT_Get_Postscript_Name( self->face );
  if ( ps_name == NULL )
    ps_name = "UNAVAILABLE";
    
  FT2Font_setattr(self, "postscript_name", PyString_FromString(ps_name));
  FT2Font_setattr(self, "num_faces", PyInt_FromLong(self->face->num_faces));  
  FT2Font_setattr(self, "family_name", PyString_FromString(self->face->family_name));  
  FT2Font_setattr(self, "style_name", PyString_FromString(self->face->style_name));  
  FT2Font_setattr(self, "face_flags", PyInt_FromLong(self->face->face_flags));  
  FT2Font_setattr(self, "style_flags", PyInt_FromLong(self->face->style_flags));  
  FT2Font_setattr(self, "num_glyphs", PyInt_FromLong(self->face->num_glyphs));  
  FT2Font_setattr(self, "num_fixed_sizes", PyInt_FromLong(self->face->num_fixed_sizes));  
  FT2Font_setattr(self, "num_charmaps", PyInt_FromLong(self->face->num_charmaps));  

  int scalable;
  scalable = FT_IS_SCALABLE( self->face );
  FT2Font_setattr(self, "scalable", PyInt_FromLong(scalable));  

  if (scalable) {
    FT2Font_setattr(self, "units_per_EM", PyInt_FromLong(self->face->units_per_EM));  
    PyObject *bbox = Py_BuildValue
      ("(llll)", 
       self->face->bbox.xMin, self->face->bbox.yMin, 
       self->face->bbox.xMax, self->face->bbox.yMax );
    FT2Font_setattr(self, "bbox", bbox);
    FT2Font_setattr(self, "ascender", PyInt_FromLong(self->face->ascender));  
    FT2Font_setattr(self, "descender", PyInt_FromLong(self->face->descender));  
    FT2Font_setattr(self, "height", PyInt_FromLong(self->face->height));  
    FT2Font_setattr(self, "max_advance_width", PyInt_FromLong(self->face->max_advance_width));  
    FT2Font_setattr(self, "max_advance_height", PyInt_FromLong(self->face->max_advance_height));  
    FT2Font_setattr(self, "underline_position", PyInt_FromLong(self->face->underline_position));  
    FT2Font_setattr(self, "underline_thickness", PyInt_FromLong(self->face->underline_thickness));  
       
    
  }

  //printf("set font props\n");
  return self;
}



char FT2Font_set_bitmap_size__doc__[] = 
"set_bitmap_size(w, h)\n"
"\n"
"Manually set the bitmap size to render the glyps to.  This is useful"
"in cases where you want to render several different glyphs to the bitmap"
;

static PyObject *
FT2Font_set_bitmap_size(FT2FontObject *self, PyObject *args)
{
  long width, height, numBytes, n;
  
  if (!PyArg_ParseTuple(args, "ll:set_bitmap_size", &width, &height))
    return NULL;
  
  self->image.width   = (unsigned)width;
  self->image.height  = (unsigned)height;


  numBytes = self->image.width*self->image.height;
  free(self->image.buffer);
  self->image.buffer = (unsigned char *)malloc(numBytes);
  for (n=0; n<numBytes; n++) 
    self->image.buffer[n] = 0;

  //printf("set bitmap size to %lu %lu\n", self->image.width, self->image.height);
  Py_INCREF(Py_None);
  return Py_None;

}

/* FT2Font methods */

static void
FT2Font_dealloc(FT2FontObject *self)
{

  FT_Done_Face    ( self->face );

  //todo: how to free the library, ie, when all fonts are done
  //FT_Done_FreeType( _ft2Library );

  //todo: how to free buffer - this seqfaults
  free(self->image.buffer );
  self->image.buffer = NULL;

  //printf("deallocating font %d\n", self->num_glyphs);
  Py_XDECREF(self->x_attr);
  
  PyObject_Del(self);

}

static PyObject *
FT2Font_set_size(FT2FontObject *self, PyObject *args)
{
  int error;
  double ptsize, dpi;
  if (!PyArg_ParseTuple(args, "dd:set_size", &ptsize, &dpi))
    return NULL;

  error = FT_Set_Char_Size( self->face, ptsize * 64, 0, dpi, dpi );
  if (error) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not set the fontsize");
    return NULL;
  }

  
  Py_INCREF(Py_None);
  return Py_None;
}




void compute_string_bbox( FT2FontObject *self, FT_BBox *abbox ) { 

  int n;
  FT_BBox bbox; 
  /* initialize string bbox to "empty" values */ 
  bbox.xMin = bbox.yMin = 32000; 
  bbox.xMax = bbox.yMax = -32000; 

  for ( n = 0; n < self->num_glyphs; n++ ) { 
    FT_BBox glyph_bbox;
    FT_Glyph_Get_CBox( self->glyphs[n], ft_glyph_bbox_subpixels, &glyph_bbox );
    if ( glyph_bbox.xMin < bbox.xMin ) bbox.xMin = glyph_bbox.xMin;
    if ( glyph_bbox.yMin < bbox.yMin ) bbox.yMin = glyph_bbox.yMin;
    if ( glyph_bbox.xMax > bbox.xMax ) bbox.xMax = glyph_bbox.xMax;
    if ( glyph_bbox.yMax > bbox.yMax ) bbox.yMax = glyph_bbox.yMax;
  } /* check that we really grew the string bbox */ 
  if ( bbox.xMin > bbox.xMax ) { 
    bbox.xMin = 0; bbox.yMin = 0; bbox.xMax = 0; bbox.yMax = 0; 
  } /* return string bbox */ 
  //printf("computed string bbox from %d glyphs\n", self->num_glyphs);
  *abbox = bbox; 

} 


void load_glyphs(FT2FontObject *self) {


  /* a small shortcut */ 
  FT_UInt glyph_index;
  FT_Bool use_kerning;
  FT_UInt previous;
  int n, error;

  use_kerning = FT_HAS_KERNING( self->face ); 
  previous = 0; 
  

  self->num_glyphs = 0;
  self->pen.x = 0;
  self->pen.y = 0;

  for ( n = 0; n < self->num_chars; n++ ) { 

    glyph_index = FT_Get_Char_Index( self->face, self->text[n] );
    /* retrieve kerning distance and move pen position */ 
    if ( use_kerning && previous && glyph_index ) { 
      FT_Vector delta;
      FT_Get_Kerning( self->face, previous, glyph_index, 
		      FT_KERNING_DEFAULT, &delta );
      self->pen.x += delta.x;  
    } 


    error = FT_Load_Glyph( self->face, glyph_index, FT_LOAD_DEFAULT ); 
    if ( error ) {
      printf("\tcould not load glyph for %c\n", self->text[n]);
      continue; 
    }
    /* ignore errors, jump to next glyph */ 

    /* extract glyph image and store it in our table */ 
    error = FT_Get_Glyph( self->face->glyph, &self->glyphs[self->num_glyphs] ); 
    if ( error ) {
      printf("\tcould not get glyph for %c\n", self->text[n]);
      continue; 
    }
    /* ignore errors, jump to next glyph */ 


    FT_Glyph_Transform( self->glyphs[self->num_glyphs], 0, &self->pen);
    self->pen.x += self->face->glyph->advance.x;

    previous = glyph_index; 
    //printf("loaded glyph %d\n", self->num_glyphs);
    self->num_glyphs++; 


  }

  
  // now apply the rotation
  for (n=0; n<self->num_glyphs; n++) { 
    FT_Glyph_Transform(self->glyphs[n], &self->matrix, 0);
  }
  
  

}



static PyObject *
FT2Font_set_text(FT2FontObject *self, PyObject *args)
{
  double angle;
  if (!PyArg_ParseTuple(args, "s#d:set_text", &self->text, &self->num_chars,
			&angle))
    return NULL;
  self->angle = angle/360.0*2*3.14159;
  //this computes width and height in subpixels so we have to divide by 64
  self->matrix.xx = (FT_Fixed)( cos( self->angle ) * 0x10000L );
  self->matrix.xy = (FT_Fixed)(-sin( self->angle ) * 0x10000L );
  self->matrix.yx = (FT_Fixed)( sin( self->angle ) * 0x10000L );
  self->matrix.yy = (FT_Fixed)( cos( self->angle ) * 0x10000L );

  load_glyphs(self);

  Py_INCREF(Py_None);
  return Py_None;

}

char FT2Font_load_char__doc__[] = 
"load_char(charcode)\n"
"\n"
"Load character with charcode in current fontfile and set glyph.\n"
"Return value is a Glyph object, with attributes\n"
"  width          # glyph width\n"
"  height         # glyph height\n"
"\n"
"  horiBearingX   # left side bearing in horizontal layouts\n"
"  horiBearingY   # top side bearing in horizontal layouts\n"
"  horiAdvance    # advance width for horizontal layout\n"
"\n"
"  vertBearingX   # left side bearing in vertical layouts\n"
"  vertBearingY  #top side bearing in vertical layouts\n"
"  vertAdvance    # advance height for vertical layout\n"
;
static  GlyphObject *
FT2Font_load_char(FT2FontObject *self, PyObject *args)
{
  //load a char using the unsigned long charcode
  long charcode;
  int error;
  if (!PyArg_ParseTuple(args, "l:load_char", &charcode))
    return NULL;

  error = FT_Load_Char( self->face, (unsigned long)charcode, FT_LOAD_DEFAULT);

  if (error) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not load charcode");
    return NULL;
  }
    
  error = FT_Get_Glyph( self->face->glyph, &self->glyphs[self->num_glyphs] ); 
  if (error) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not get glyph for char");
    return NULL;
  }
  
  
  

  GlyphObject *gm;
  gm = PyObject_New(GlyphObject, &Glyph_Type);
  

  if (gm == NULL) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not create glyph metrix object");
    
    return NULL;
  }

  gm->glyph_num = self->num_glyphs++;

  gm->x_attr = NULL;

  Glyph_setattr(gm, "width", PyInt_FromLong(self->face->glyph->metrics.width));
  Glyph_setattr(gm, "height", PyInt_FromLong(self->face->glyph->metrics.height));
  Glyph_setattr(gm, "horiBearingX", PyInt_FromLong(self->face->glyph->metrics.horiBearingX));
  Glyph_setattr(gm, "horiBearingY", PyInt_FromLong(self->face->glyph->metrics.horiBearingY));
  Glyph_setattr(gm, "horiAdvance", PyInt_FromLong(self->face->glyph->metrics.horiAdvance));
  Glyph_setattr(gm, "vertBearingX", PyInt_FromLong(self->face->glyph->metrics.vertBearingX));
  Glyph_setattr(gm, "vertBearingY", PyInt_FromLong(self->face->glyph->metrics.vertBearingY));
  Glyph_setattr(gm, "vertAdvance", PyInt_FromLong(self->face->glyph->metrics.vertAdvance));
    
  return gm;
}

static PyObject *
FT2Font_get_width_height(FT2FontObject *self, PyObject *args)
{
  

  FT_BBox bbox;
  
  compute_string_bbox(self, &bbox);
  return Py_BuildValue("(ll)", 
		       (bbox.xMax - bbox.xMin)/64, 
		       (bbox.yMax - bbox.yMin)/64);
}


void
draw_bitmap( FT_Bitmap*  bitmap,
	     FT2_Image* image, 
             FT_Int      x,
             FT_Int      y)
{
  FT_Int  i, j, p, q;
  FT_Int  x_max = x + bitmap->width;
  FT_Int  y_max = y + bitmap->rows;

  for ( i = x, p = 0; i < x_max; i++, p++ )
  {
    for ( j = y, q = 0; j < y_max; j++, q++ )
    {
      if ( i >= image->width || j >= image->height )
        continue;
      image->buffer[i + j*image->width] |= bitmap->buffer[q * bitmap->width + p]; 


    }
  }
}



static PyObject *
FT2Font_write_bitmap(FT2FontObject *self, PyObject *args)
{

  char *filename;
  FT_Int  i, j;

 
  if (!PyArg_ParseTuple(args, "s:write_bitmap", &filename))
    return NULL;
  
  FILE *fh = fopen(filename, "w");
  for ( i = 0; i< self->image.height; i++)
  {
    for ( j = 0; j < self->image.width; ++j)
    {
      fputc(self->image.buffer[j + i*self->image.width], fh);
    }
  }


  fclose(fh);
  Py_INCREF(Py_None);
  return Py_None;

}


static PyObject *
FT2Font_draw_glyphs_to_bitmap(FT2FontObject *self, PyObject *args)
{

  int error;
  FT_BBox string_bbox;
  int n;
  unsigned long numBytes;
  
  if (!PyArg_ParseTuple(args, ":draw_glyphs_to_bitmap"))
    return NULL;
  
  compute_string_bbox(self, &string_bbox);


  self->image.width   = (string_bbox.xMax-string_bbox.xMin) / 64;
  self->image.height  = (string_bbox.yMax-string_bbox.yMin) / 64;


  numBytes = self->image.width*self->image.height;
  free(self->image.buffer);
  self->image.buffer = (unsigned char *)malloc(numBytes);
  for (n=0; n<numBytes; n++) 
    self->image.buffer[n] = 0;

  
  for ( n = 0; n < self->num_glyphs; n++ )
    {
      FT_BBox bbox;

      FT_Glyph_Get_CBox(self->glyphs[n], ft_glyph_bbox_pixels, &bbox);

      error = FT_Glyph_To_Bitmap(&self->glyphs[n],
				 FT_RENDER_MODE_NORMAL,
				 0,
				 //&self->pos[n],
				 0  //don't destroy image
				 );

      if (error) {
	PyErr_SetString(PyExc_RuntimeError, 
			"Could not convert glyph to bitmap");
	return NULL;
      }
      
      FT_BitmapGlyph bitmap = (FT_BitmapGlyph)self->glyphs[n];
      /* now, draw to our target surface (convert position) */
      
      
      //bitmap left and top in pixel, string bbok in subpixel
      //printf("%ld, %ld\n", string_bbox.yMin, string_bbox.yMax);
      draw_bitmap( &bitmap->bitmap, 
		   &self->image, 		   	       
		   bitmap->left-string_bbox.xMin/64.0,
		   string_bbox.yMax/64.0-bitmap->top
		   );
      
    }
  
  //show_image();
  

  Py_INCREF(Py_None);
  return Py_None;
}


char FT2Font_draw_glyph_to_bitmap__doc__[] = 
"draw_glyph_to_bitmap(x, y, glyph)\n"
"\n"
"Draw a single glyph to the bitmap at pixel locations x,y\n"
"Note it is your responsibility to set up the bitmap manually\n"
"with set_bitmap_size(w,h) before this call is made.\n"
"\n"
"If you want automatic layout, use set_text in combinations with\n"
"draw_glyphs_to_bitmap.  This function is intended for people who\n"
"want to render individual glyphs at precise locations, eg, a\n"
"a glyph returned by load_char\n";
;

static PyObject *
FT2Font_draw_glyph_to_bitmap(FT2FontObject *self, PyObject *args)
{

  if (self->image.width==0 || self->image.height==0) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "You must first set the size of the bitmap with set_bitmap_size");
    return NULL;
    
  }

  int error;
  GlyphObject *glyph;
  long x, y;  
  if (!PyArg_ParseTuple(args, "llO!:draw_glyph_to_bitmap", &x, &y, 
			&Glyph_Type, (PyObject*)&glyph))
    return NULL;

  
  //printf("draw_glyph_to_bitmap parsed args: %ld %ld\n", x, y);
  error = FT_Glyph_To_Bitmap(&self->glyphs[glyph->glyph_num],
			     FT_RENDER_MODE_NORMAL,
			     0,  //no additional translation			     
			     0  //don't destroy image
			     );

  if (error) {
    PyErr_SetString(PyExc_RuntimeError, 
		    "Could not convert glyph to bitmap");
    return NULL;
  }

  //printf("draw_glyph_to_bitmap cast\n");      
  FT_BitmapGlyph bitmap = (FT_BitmapGlyph)self->glyphs[glyph->glyph_num];

  //printf("draw_glyph_to_bitmap to image at %ld, %lu, %lu, %lu\n", 
  //x, y, self->image.width, self->image.height);            
  //printf("bitmap props %d, %d\n",  bitmap->left, bitmap->top);            


  draw_bitmap( &bitmap->bitmap, 
	       &self->image,
	       //x + bitmap->left,
	       x,
	       //y+bitmap->top
	       y
	       );
  //	       start.y-bitmap->top);
  
  //printf("draw_glyph_to_bitmap all dont\n");      
  Py_INCREF(Py_None);
  return Py_None;

}
  
static PyMethodDef FT2Font_methods[] = {
  {"write_bitmap",  (PyCFunction)FT2Font_write_bitmap,	METH_VARARGS},
  {"set_bitmap_size",  (PyCFunction)FT2Font_set_bitmap_size,	METH_VARARGS, FT2Font_load_char__doc__},
  {"draw_glyph_to_bitmap",  (PyCFunction)FT2Font_draw_glyph_to_bitmap,	METH_VARARGS},
  {"draw_glyphs_to_bitmap",  (PyCFunction)FT2Font_draw_glyphs_to_bitmap,	METH_VARARGS},
  {"load_char",	   (PyCFunction)FT2Font_load_char,	METH_VARARGS, FT2Font_load_char__doc__},
  {"set_text",	   (PyCFunction)FT2Font_set_text,	METH_VARARGS},
  {"set_size",	   (PyCFunction)FT2Font_set_size,	METH_VARARGS},
  {"get_width_height",	   (PyCFunction)FT2Font_get_width_height,	METH_VARARGS},
  {NULL,		NULL}		/* sentinel */
};

static PyObject *
FT2Font_getattr(FT2FontObject *self, char *name)
{
  if (self->x_attr != NULL) {
    PyObject *v = PyDict_GetItemString(self->x_attr, name);
    if (v != NULL) {
      Py_INCREF(v);
      return v;
    }
  }
  return Py_FindMethod(FT2Font_methods, (PyObject *)self, name);
}

static int
FT2Font_setattr(FT2FontObject *self, char *name, PyObject *v)
{
  if (self->x_attr == NULL) {
    self->x_attr = PyDict_New();
    if (self->x_attr == NULL)
      return -1;
  }
  if (v == NULL) {
    int rv = PyDict_DelItemString(self->x_attr, name);
    if (rv < 0)
      PyErr_SetString(PyExc_AttributeError,
		      "delete non-existing FT2Font attribute");
    return rv;
  }
  else
    return PyDict_SetItemString(self->x_attr, name, v);
}

static PyTypeObject FT2Font_Type = {
  /* The ob_type field must be initialized in the module init function
   * to be portable to Windows without using C++. */
  PyObject_HEAD_INIT(NULL)
  0,			/*ob_size*/
  "ft2font.FT2Font",		/*tp_name*/
  sizeof(FT2FontObject),	/*tp_basicsize*/
  0,			/*tp_itemsize*/
  /* methods */
  (destructor)FT2Font_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)FT2Font_getattr, /*tp_getattr*/
  (setattrfunc)FT2Font_setattr, /*tp_setattr*/
  0,			/*tp_compare*/
  0,			/*tp_repr*/
  0,			/*tp_as_number*/
  0,			/*tp_as_sequence*/
  0,			/*tp_as_mapping*/
  0,			/*tp_hash*/
  0,                      /*tp_call*/
  0,                      /*tp_str*/
  0,                      /*tp_getattro*/
  0,                      /*tp_setattro*/
  0,                      /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,     /*tp_flags*/
  0,                      /*tp_doc*/
  0,                      /*tp_traverse*/
  0,                      /*tp_clear*/
  0,                      /*tp_richcompare*/
  0,                      /*tp_weaklistoffset*/
  0,                      /*tp_iter*/
  0,                      /*tp_iternext*/
  0,                      /*tp_methods*/
  0,                      /*tp_members*/
  0,                      /*tp_getset*/
  0,                      /*tp_base*/
  0,                      /*tp_dict*/
  0,                      /*tp_descr_get*/
  0,                      /*tp_descr_set*/
  0,                      /*tp_dictoffset*/
  0,                      /*tp_init*/
  0,                      /*tp_alloc*/
  0,                      /*tp_new*/
  0,                      /*tp_free*/
  0,                      /*tp_is_gc*/
};
/* --------------------------------------------------------------------- */



/* Function of no arguments returning new FT2Font object */
char ft2font_new__doc__[] = 
"FT2Font(ttffile)\n"
"\n"
"Create a new FT2Font object; the following global font attributes\n"
"are defined\n"
"num_faces        # number of faces in file\n"
"face_flags       # the face flags as python int\n"
"style_flags      # the style flags as python int\n"
"num_glyphs       # the number of glyphs in the face\n"
"family_name      # the face family name\n"
"style_name       # the face syle name\n"
"num_fixed_sizes  # the number of embedded bitmap strikes in the current face\n"
"\n"
"# the following are only available if face.scalable\n"
"bbox                 # the face global bounding box (xmin, ymin, xmax, ymax)\n"
"units_per_EM         # number of font units covered by the EM\n"
"ascender             # the ascender in 26.6 units\n"
"descender            # the descender in 26.6 units\n"
"height               # the height in 26.6 units; used to compute a default\n"
"                       line spacing (i.e., the baseline-to-baseline distance)  \n"
"max_advance_width    # maximum horizontal cursor advance for all glyphs\n"
"max_advance_height   # same for vertical layout\n"
"underline_position   # vertical position of the underline bar\n"
"underline_thickness  # vertical thickness of the underline\n"
;
static PyObject *
ft2font_new(PyObject *self, PyObject *args)
{

  FT2FontObject *rv;
  

  rv = newFT2FontObject(args);

  if ( rv == NULL )
    return NULL;

  return (PyObject *)rv;
}


/* List of functions defined in the module */

static PyMethodDef ft2font_methods[] = {  
  {"FT2Font",		ft2font_new,		METH_VARARGS, ft2font_new__doc__},
  {NULL,		NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initft2font) */

DL_EXPORT(void)
     initft2font(void)
{
  PyObject *m, *d;
  
  /* Initialize the type of the new type object here; doing it here
   * is required for portability to Windows without requiring C++. */
  FT2Font_Type.ob_type = &PyType_Type;
  Glyph_Type.ob_type = &PyType_Type;
  
  /* Create the module and add the functions */
  m = Py_InitModule("ft2font", ft2font_methods);
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(m);
  ErrorObject = PyErr_NewException("ft2font.error", NULL, NULL);
  PyDict_SetItemString(d, "error", ErrorObject);
}

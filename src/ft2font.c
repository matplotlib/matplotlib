#include "ft2font.h"

static PyObject *ErrorObject;



int           _initLib = 0;
FT_Library    _ft2Library;
#define FT2FontObject_Check(v)	((v)->ob_type == &FT2Font_Type)


static FT2FontObject *
newFT2FontObject(PyObject *args)
{
  char* facefile;
  int error;
  if (! _initLib) { 
    error = FT_Init_FreeType( &_ft2Library ); //initialize library 
    _initLib = 1;
    // TODO handle errors
  }
  

  FT2FontObject *self;
  self = PyObject_New(FT2FontObject, &FT2Font_Type);
  self->image.buffer = NULL;
  self->image.width = 0;
  self->image.height = 0;
  
  if (!PyArg_ParseTuple(args, "s:FT2Font", &facefile))
    return NULL;
  error = FT_New_Face( _ft2Library, facefile, 0, &self->face );

  if (self == NULL)
    return NULL;
  self->x_attr = NULL;

  return self;
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
  /* TODO: error handling omitted */
  
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
  *abbox = bbox; 

} 



void load_glyphs(FT2FontObject *self) {

  FT_GlyphSlot slot = self->face->glyph;
  /* a small shortcut */ 
  FT_UInt glyph_index;
  FT_Bool use_kerning;
  FT_UInt previous;
  int n, pen_x, pen_y, error;

  /* start at (0,0) */ 
  pen_x = 0;  pen_y = 0; 

  use_kerning = FT_HAS_KERNING( self->face ); 
  previous = 0; 
  

  self->num_glyphs = 0;
  for ( n = 0; n < self->num_chars; n++ ) { 

    glyph_index = FT_Get_Char_Index( self->face, self->text[n] );
    /* retrieve kerning distance and move pen position */ 
    if ( use_kerning && previous && glyph_index ) { 
      FT_Vector delta;
      FT_Get_Kerning( self->face, previous, glyph_index, 
		      FT_KERNING_DEFAULT, &delta );
      pen_x += delta.x;  
    } 

    self->pos[self->num_glyphs].x = pen_x; 
    self->pos[self->num_glyphs].y = pen_y; 

    error = FT_Load_Glyph( self->face, glyph_index, FT_LOAD_DEFAULT ); 
    if ( error ) continue; 
    /* ignore errors, jump to next glyph */ 
    /* extract glyph image and store it in our table */ 
    error = FT_Get_Glyph( slot, &self->glyphs[self->num_glyphs] ); 
    if ( error ) continue; 
    /* ignore errors, jump to next glyph */ 

    FT_Glyph_Transform( self->glyphs[n], 0, &self->pos[self->num_glyphs]);
    pen_x += slot->advance.x;

    previous = glyph_index; 
    self->num_glyphs++; 

  }
  FT_Vector delta;
  delta.x = 0; delta.y = 0;
  for (n=0; n<self->num_glyphs; n++) { 
    FT_Glyph_Transform(self->glyphs[n], &self->matrix, &delta);
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
FT2Font_draw_bitmap(FT2FontObject *self, PyObject *args)
{

  int error;
  FT_BBox string_bbox;
  int n, numBytes;
  FT_Vector pen;
  
  if (!PyArg_ParseTuple(args, ":draw_bitmap"))
    return NULL;
  
  compute_string_bbox(self, &string_bbox);


  self->image.width   = (string_bbox.xMax-string_bbox.xMin) / 64;
  self->image.height  = (string_bbox.yMax-string_bbox.yMin) / 64;


  numBytes = self->image.width*self->image.height;
  free(self->image.buffer);
  self->image.buffer = (unsigned char *)malloc(numBytes);
  for (n=0; n<numBytes; n++) 
    self->image.buffer[n] = 0;
  
    
  /* the pen position in 26.6 cartesian space coordinates; */
  /* start at (0,0) relative to the upper left corner  */
  pen.x = 0; //what should these be?
  pen.y = 0;

  
  for ( n = 0; n < self->num_glyphs; n++ )
    {
      FT_BBox bbox;

      //FT_Glyph_Transform(self->glyphs[n], 0, &pen);
      FT_Glyph_Get_CBox(self->glyphs[n], ft_glyph_bbox_pixels, &bbox);

      error = FT_Glyph_To_Bitmap(&self->glyphs[n],
				 FT_RENDER_MODE_NORMAL,
				 0,
				 //&self->pos[n],
				 0  //don't destroy image
				 );

      if (error) {
	printf("\tError creating bitmap for %c\n", self->text[n]);
	continue;
      }
      
      FT_BitmapGlyph bitmap = (FT_BitmapGlyph)self->glyphs[n];
      /* now, draw to our target surface (convert position) */
      
      
      //bitmap left and top in pixel, string bbok in subpixel
      //printf("%ld, %ld\n", string_bbox.yMin, string_bbox.yMax);
      if (1) {
	draw_bitmap( &bitmap->bitmap, 
		     &self->image, 		   	       
		     bitmap->left-string_bbox.xMin/64.0,
		     string_bbox.yMax/64.0-bitmap->top);
      }
      else {      
	draw_bitmap( &bitmap->bitmap, 
		     &self->image, 		   	       
		     bitmap->left-string_bbox.xMin/64,
		     string_bbox.yMax/64.0-(bitmap->top-string_bbox.yMin/64.0));
      }
    }
  
  //show_image();
  

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef FT2Font_methods[] = {
  {"write_bitmap",  (PyCFunction)FT2Font_write_bitmap,	METH_VARARGS},
  {"draw_bitmap",  (PyCFunction)FT2Font_draw_bitmap,	METH_VARARGS},
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
  {"FT2Font",		ft2font_new,		METH_VARARGS},
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
  
  /* Create the module and add the functions */
  m = Py_InitModule("ft2font", ft2font_methods);
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(m);
  ErrorObject = PyErr_NewException("ft2font.error", NULL, NULL);
  PyDict_SetItemString(d, "error", ErrorObject);
}

#include <sstream>
#include "ft2font.h"

#define _DEBUG 0
 
FT_Library _ft2Library;

FT2Image::FT2Image() : buffer(NULL) {}
FT2Image::~FT2Image() {delete [] buffer; buffer=NULL;}

Glyph::Glyph( const FT_Face& face, const FT_Glyph& glyph, size_t ind) :
  glyphInd(ind) {
  if (_DEBUG) std::cout << "Glyph::Glyph" <<std::endl;
  
  FT_BBox bbox;
  FT_Glyph_Get_CBox( glyph, ft_glyph_bbox_subpixels, &bbox );
  
  
  setattr("width", Py::Int( face->glyph->metrics.width) );
  setattr("height", Py::Int( face->glyph->metrics.height) );
  setattr("horiBearingX", Py::Int( face->glyph->metrics.horiBearingX) );
  setattr("horiBearingY", Py::Int( face->glyph->metrics.horiBearingY) );
  setattr("horiAdvance", Py::Int( face->glyph->metrics.horiAdvance) );
  setattr("vertBearingX", Py::Int( face->glyph->metrics.vertBearingX) );
  
  setattr("vertBearingY", Py::Int( face->glyph->metrics.vertBearingY) );
  setattr("vertAdvance", Py::Int( face->glyph->metrics.vertAdvance) );
  
  Py::Tuple abbox(4);

  abbox[0] = Py::Int(bbox.xMin);
  abbox[1] = Py::Int(bbox.yMin);
  abbox[2] = Py::Int(bbox.xMax);
  abbox[3] = Py::Int(bbox.yMax);
  setattr("bbox", abbox);
  
  
}


Glyph::~Glyph() {
  if (_DEBUG) std::cout << "Glyph::~Glyph" <<std::endl;  
}

int 
Glyph::setattr( const char *name, const Py::Object &value ) {
  if (_DEBUG) std::cout << "Glyph::setattr" << std::endl;
  __dict__[name] = value;
  return 0; 
}

Py::Object 
Glyph::getattr( const char *name ) {
  if (_DEBUG) std::cout << "Glyph::getattr" << std::endl;
  if ( __dict__.hasKey(name) ) return __dict__[name];
  else return getattr_default( name );

}

FT2Font::FT2Font(std::string facefile) 
{
  if (_DEBUG) std::cout << "FT2Font::FT2Font" << std::endl;
  clear(Py::Tuple(0));
  
  int error = FT_New_Face( _ft2Library, facefile.c_str(), 0, &face );
  
  if (error == FT_Err_Unknown_File_Format ) {
    std::ostringstream s;
    s << "Could not load facefile " << facefile << "; Unknown_File_Format" << std::endl;
    throw Py::RuntimeError(s.str());
    
  }
  else if (error == FT_Err_Cannot_Open_Resource) {
    std::ostringstream s;
    s << "Could not open facefile " << facefile << "; Cannot_Open_Resource" << std::endl;
    throw Py::RuntimeError(s.str());
  }
  else if (error == FT_Err_Invalid_File_Format) {
    std::ostringstream s;
    s << "Could not open facefile " << facefile << "; Invalid_File_Format" << std::endl;
    throw Py::RuntimeError(s.str());
  }
  else if (error) {
    std::ostringstream s;
    s << "Could not open facefile " << facefile << "; freetype error code " << error<< std::endl;
    throw Py::RuntimeError(s.str());
    
  }
  
  // set a default fontsize 12 pt at 72dpi
  error = FT_Set_Char_Size( face, 12 * 64, 0, 72, 72 );
  //error = FT_Set_Char_Size( face, 20 * 64, 0, 80, 80 );
  if (error) {
    std::ostringstream s;
    s << "Could not set the fontsize for facefile  " << facefile << std::endl;
    throw Py::RuntimeError(s.str());
  }
  
  // set some face props as attributes
  //small memory leak fixed after 2.1.8 
  const char* ps_name = FT_Get_Postscript_Name( face );
  //const char* ps_name = "jdh";
  if ( ps_name == NULL )
    ps_name = "UNAVAILABLE";
  
  setattr("postscript_name", Py::String(ps_name));
  setattr("num_faces",       Py::Int(face->num_faces));
  setattr("family_name",     Py::String(face->family_name));
  setattr("style_name",      Py::String(face->style_name));
  setattr("face_flags",      Py::Int(face->face_flags));
  setattr("style_flags",     Py::Int(face->style_flags));
  setattr("num_glyphs",      Py::Int(face->num_glyphs));
  setattr("num_fixed_sizes", Py::Int(face->num_fixed_sizes));
  setattr("num_charmaps",    Py::Int(face->num_charmaps));
  
   
  int scalable = FT_IS_SCALABLE( face );

  setattr("scalable", Py::Int(scalable));
  
  if (scalable) {
    setattr("units_per_EM", Py::Int(face->units_per_EM));
    
    Py::Tuple bbox(4);
    bbox[0] = Py::Int(face->bbox.xMin);
    bbox[1] = Py::Int(face->bbox.yMin);
    bbox[2] = Py::Int(face->bbox.xMax);
    bbox[3] = Py::Int(face->bbox.yMax);
    setattr("bbox",  bbox);
    setattr("ascender",            Py::Int(face->ascender));
    setattr("descender",           Py::Int(face->descender));
    setattr("height",              Py::Int(face->height));
    setattr("max_advance_width",   Py::Int(face->max_advance_width));
    setattr("max_advance_height",  Py::Int(face->max_advance_height));
    setattr("underline_position",  Py::Int(face->underline_position));
    setattr("underline_thickness", Py::Int(face->underline_thickness));
    
  }
  
  
}

FT2Font::~FT2Font()
{
  if (_DEBUG) std::cout << "FT2Font::~FT2Font" << std::endl;
  FT_Done_Face    ( face );
  
  delete [] image.buffer ;
  image.buffer = NULL;

  for (size_t i=0; i<glyphs.size(); i++) {
    FT_Done_Glyph( glyphs[i] );    
  }

  for (size_t i=0; i<gms.size(); i++) {
    Py_DECREF(gms[i]);
  }
}

int 
FT2Font::setattr( const char *name, const Py::Object &value ) {
  if (_DEBUG) std::cout << "FT2Font::setattr" << std::endl;
  __dict__[name] = value;
  return 1; 
}

Py::Object 
FT2Font::getattr( const char *name ) {
  if (_DEBUG) std::cout << "FT2Font::getattr" << std::endl;
  if ( __dict__.hasKey(name) ) return __dict__[name];
  else return getattr_default( name );

}



char FT2Font::set_bitmap_size__doc__[] = 
"set_bitmap_size(w, h)\n"
"\n"
"Manually set the bitmap size to render the glyps to.  This is useful"
"in cases where you want to render several different glyphs to the bitmap"
;

Py::Object
FT2Font::set_bitmap_size(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::set_bitmap_size" << std::endl;
  args.verify_length(2);
  
  long width = Py::Int(args[0]);
  long height = Py::Int(args[1]);
  
  image.width   = (unsigned)width;
  image.height  = (unsigned)height;
  
  
  long numBytes = image.width * image.height;
  
  delete [] image.buffer;
  image.buffer = new unsigned char [numBytes];
  for (long n=0; n<numBytes; n++) 
    image.buffer[n] = 0;
  
  return Py::Object();
  
}





char FT2Font::clear__doc__[] = 
"clear()\n"
"\n"
"Clear all the glyphs, reset for a new set_text"
;

Py::Object
FT2Font::clear(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::clear" << std::endl;
  args.verify_length(0);

  //todo: move to image method?
  delete [] image.buffer ;
  image.buffer = NULL;
  image.width = 0;
  image.height = 0;
  image.offsetx = 0;
  image.offsety = 0;
  
  text = "";
  angle = 0.0;
  
  pen.x = 0;
  pen.y = 0;

  for (size_t i=0; i<glyphs.size(); i++) {
    FT_Done_Glyph( glyphs[i] );    
  }

  for (size_t i=0; i<gms.size(); i++) {
    Py_DECREF(gms[i]);
  }

  glyphs.resize(0);
  gms.resize(0);

  return Py::Object();
  
}

char FT2Font::set_size__doc__[] = 
"set_size(ptsize, dpi)\n"
"\n"
"Set the point size and dpi of the text.\n"
;

Py::Object
FT2Font::set_size(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::set_size" << std::endl;
  args.verify_length(2);
  
  
  double ptsize = Py::Float(args[0]);
  double dpi = Py::Float(args[1]);
  

  int error = FT_Set_Char_Size( face, (long)(ptsize * 64), 0, 
				(unsigned int)dpi, 
				(unsigned int)dpi );
  if (error) 
    throw Py::RuntimeError("Could not set the fontsize");  
  return Py::Object();
}




FT_BBox
FT2Font::compute_string_bbox( ) { 
  if (_DEBUG) std::cout << "FT2Font::compute_string_bbox" << std::endl;
  
  FT_BBox bbox; 
  /* initialize string bbox to "empty" values */ 
  bbox.xMin = bbox.yMin = 32000; 
  bbox.xMax = bbox.yMax = -32000; 
  
  for ( size_t n = 0; n < glyphs.size(); n++ ) { 
    FT_BBox glyph_bbox;
    FT_Glyph_Get_CBox( glyphs[n], ft_glyph_bbox_subpixels, &glyph_bbox );
    if ( glyph_bbox.xMin < bbox.xMin ) bbox.xMin = glyph_bbox.xMin;
    if ( glyph_bbox.yMin < bbox.yMin ) bbox.yMin = glyph_bbox.yMin;
    if ( glyph_bbox.xMax > bbox.xMax ) bbox.xMax = glyph_bbox.xMax;
    if ( glyph_bbox.yMax > bbox.yMax ) bbox.yMax = glyph_bbox.yMax;
  } /* check that we really grew the string bbox */ 
  if ( bbox.xMin > bbox.xMax ) { 
    bbox.xMin = 0; bbox.yMin = 0; bbox.xMax = 0; bbox.yMax = 0; 
  } /* return string bbox */ 
  return bbox; 
} 


void 
FT2Font::load_glyphs() {
  if (_DEBUG) std::cout << "FT2Font::load_glyphs" << std::endl;
  
  /* a small shortcut */ 
  
  
  FT_Bool use_kerning = FT_HAS_KERNING( face ); 
  FT_UInt previous = 0; 
  
  
  glyphs.resize(0);
  pen.x = 0;
  pen.y = 0;
  
  for ( unsigned int n = 0; n < text.size(); n++ ) { 
    
    FT_UInt glyph_index = FT_Get_Char_Index( face, text[n] );
    /* retrieve kerning distance and move pen position */ 
    if ( use_kerning && previous && glyph_index ) { 
      FT_Vector delta;
      FT_Get_Kerning( face, previous, glyph_index, 
		      ft_kerning_default, &delta );
      pen.x += delta.x;  
    } 
    

    error = FT_Load_Glyph( face, glyph_index, FT_LOAD_DEFAULT ); 
    if ( error ) {
      std::cerr << "\tcould not load glyph for " << text[n] << std::endl;
      continue; 
    }
    /* ignore errors, jump to next glyph */ 
    
    /* extract glyph image and store it in our table */

    FT_Glyph thisGlyph; 
    error = FT_Get_Glyph( face->glyph, &thisGlyph ); 

    if ( error ) {
      std::cerr << "\tcould not get glyph for " << text[n] << std::endl;
      continue; 
    }
    /* ignore errors, jump to next glyph */ 
    

    FT_Glyph_Transform( thisGlyph, 0, &pen);
    pen.x += face->glyph->advance.x;
    
    previous = glyph_index; 
    glyphs.push_back(thisGlyph);
    
  }
  
  
  // now apply the rotation
  for (unsigned int n=0; n<glyphs.size(); n++) 
    FT_Glyph_Transform(glyphs[n], &matrix, 0);
}



char FT2Font::set_text__doc__[] = 
"set_text(s, angle)\n"
"\n"
"Set the text string and angle.\n"
"You must call this before draw_glyphs_to_bitmap\n"
;

Py::Object
FT2Font::set_text(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::set_text" << std::endl;
  args.verify_length(2);
  text = Py::String(args[0]);
  double angle = Py::Float(args[1]);
  
  angle = angle/360.0*2*3.14159;
  //this computes width and height in subpixels so we have to divide by 64
  matrix.xx = (FT_Fixed)( cos( angle ) * 0x10000L );
  matrix.xy = (FT_Fixed)(-sin( angle ) * 0x10000L );
  matrix.yx = (FT_Fixed)( sin( angle ) * 0x10000L );
  matrix.yy = (FT_Fixed)( cos( angle ) * 0x10000L );
  
  load_glyphs();
  
  
  return Py::Object();
  
}

char FT2Font::get_glyph__doc__[] = 
"get_glyph(num)\n"
"\n"
"Return the glyph object with num num\n"
;
Py::Object
FT2Font::get_glyph(const Py::Tuple & args){
  if (_DEBUG) std::cout << "FT2Font::get_glyph" << std::endl;
  
  args.verify_length(1);
  int num = Py::Int(args[0]);
  
  if ( (size_t)num >= gms.size()) 
    throw Py::ValueError("Glyph index out of range");
  
  //todo: refcount?
  return Py::asObject(gms[num]);
}

char FT2Font::get_num_glyphs__doc__[] = 
"get_num_glyphs()\n"
"\n"
"Return the number of loaded glyphs\n"
;
Py::Object
FT2Font::get_num_glyphs(const Py::Tuple & args){
  if (_DEBUG) std::cout << "FT2Font::get_num_glyphs" << std::endl;
  
  args.verify_length(0);
  return Py::Int( (long)glyphs.size());
  
}

char FT2Font::load_char__doc__[] = 
"load_char(charcode)\n"
"\n"
"Load character with charcode in current fontfile and set glyph.\n"
"Return value is a Glyph object, with attributes\n"
"  width          # glyph width\n"
"  height         # glyph height\n"
"  bbox           # the glyph bbox (xmin, ymin, xmax, ymax)\n" 
"  horiBearingX   # left side bearing in horizontal layouts\n"
"  horiBearingY   # top side bearing in horizontal layouts\n"
"  horiAdvance    # advance width for horizontal layout\n"
"  vertBearingX   # left side bearing in vertical layouts\n"
"  vertBearingY   # top side bearing in vertical layouts\n"
"  vertAdvance    # advance height for vertical layout\n"
;
Py::Object
FT2Font::load_char(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::load_char" << std::endl;
  //load a char using the unsigned long charcode
  args.verify_length(1);
  long charcode = Py::Int(args[0]);
  
  int error = FT_Load_Char( face, (unsigned long)charcode, FT_LOAD_DEFAULT);
  
  if (error) 
    throw Py::RuntimeError("Could not load charcode");
  
  
  FT_Glyph thisGlyph; 
  error = FT_Get_Glyph( face->glyph, &thisGlyph ); 

  if (error) 
    throw Py::RuntimeError("Could not get glyph for char");
  
  size_t num = glyphs.size();  //the index into the glyphs list
  glyphs.push_back(thisGlyph);  
  Glyph* gm = new Glyph(face, thisGlyph, num);
  gms.push_back(gm);
  Py_INCREF(gm); //todo: refcount correct?
  return Py::asObject( gm);
}




char FT2Font::get_width_height__doc__[] = 
"w, h = get_width_height()\n"
"\n"
"Get the width and height in 26.6 subpixels of the current string set by set_text\n"
"The rotation of the string is accounted for.  To get width and height\n"
"in pixels, divide these values by 64\n"
;
Py::Object
FT2Font::get_width_height(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::get_width_height" << std::endl;
  
  args.verify_length(0);
  FT_BBox bbox =  compute_string_bbox();
  
  Py::Tuple ret(2);
  ret[0] = Py::Int(bbox.xMax - bbox.xMin);
  ret[1] = Py::Int(bbox.yMax - bbox.yMin);
  return ret;
}


void
FT2Font::draw_bitmap( FT_Bitmap*  bitmap,
		      FT_Int      x,
		      FT_Int      y) {
  if (_DEBUG) std::cout << "FT2Font::draw_bitmap" << std::endl;
  FT_Int  i, j, p, q;
  FT_Int  x_max = x + bitmap->width;
  FT_Int  y_max = y + bitmap->rows;
  
  FT_Int width = (FT_Int)image.width;
  FT_Int height = (FT_Int)image.height;
  for ( i = x, p = 0; i < x_max; i++, p++ )
    {
      for ( j = y, q = 0; j < y_max; j++, q++ )
	{
	  if ( i >= width || j >= height )
	    continue;
	  image.buffer[i + j*width] |= bitmap->buffer[q*bitmap->width + p];
	}
    }
}


char FT2Font::write_bitmap__doc__[] = 
"write_bitmap(fname)\n"
"\n"
"Write the bitmap to file fname\n"
;

Py::Object
FT2Font::write_bitmap(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::write_bitmap" << std::endl;
  
  args.verify_length(1);
  
  FT_Int  i, j;
  
  std::string filename = Py::String(args[0]);
  
  FILE *fh = fopen(filename.c_str(), "w");
  FT_Int width = (FT_Int)image.width;
  FT_Int height = (FT_Int)image.height;
  
  for ( i = 0; i< height; i++)   
      for ( j = 0; j < width; ++j) 
	  fputc(image.buffer[j + i*width], fh);
    
  fclose(fh);
  
  return Py::Object();
  
}


char FT2Font::draw_rect__doc__[] = 
"draw_bbox(x0, y0, x1, y1)\n"
"\n"
"Draw a rect to the image.  It is your responsibility to set the dimensions\n"
"of the image, eg, with set_bitmap_size\n"
"\n"
;
Py::Object
FT2Font::draw_rect(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::draw_rect" << std::endl;
  
  args.verify_length(4);
  
  long x0 = Py::Int(args[0]);
  long y0 = Py::Int(args[1]);
  long x1 = Py::Int(args[2]);
  long y1 = Py::Int(args[3]);
  
  
  FT_Int iwidth = (FT_Int)image.width;
  FT_Int iheight = (FT_Int)image.height;
  
  if ( x0<0 || y0<0 || x1<0 || y1<0 || 
       x0>iwidth || x1>iwidth ||
       y0>iheight || y1>iheight ) 
    throw Py::ValueError("rect coords outside image bounds");
  
  
  for (long i=x0; i<x1; ++i) {
    image.buffer[i + y0*iwidth] = 255;
    image.buffer[i + y1*iwidth] = 255;
  }
  
  for (long j=y0; j<y1; ++j) {
    image.buffer[x0 + j*iwidth] = 255;
    image.buffer[x1 + j*iwidth] = 255;
  }
  
  
  return Py::Object();
}


char FT2Font::image_as_str__doc__[] = 
"width, height, s = image_as_str()\n"
"\n"
"Return the image buffer as a string\n"
"\n"
;
Py::Object
FT2Font::image_as_str(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::image_as_str" << std::endl;
  args.verify_length(0);
  
  return Py::Object(
		    Py_BuildValue("lls#", 
				  image.width, 
				  image.height, 
				  image.buffer, 
				  image.width*image.height)
		    );
}


char FT2Font::draw_glyphs_to_bitmap__doc__[] = 
"draw_glyphs_to_bitmap()\n"
"\n"
"Draw the glyphs that were loaded by set_text to the bitmap\n"
"The bitmap size will be automatically set to include the glyphs\n"
;
Py::Object
FT2Font::draw_glyphs_to_bitmap(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::draw_glyphs_to_bitmap" << std::endl;
  args.verify_length(0);
  
  FT_BBox string_bbox = compute_string_bbox();
  
  image.width   = (string_bbox.xMax-string_bbox.xMin) / 64+2;
  image.height  = (string_bbox.yMax-string_bbox.yMin) / 64+2;
  
  image.offsetx = (int)(string_bbox.xMin/64.0);
  if (angle==0)  
    image.offsety =    -image.height;
  else 
    image.offsety =   (int)(-string_bbox.yMax/64.0);
  
  
  size_t numBytes = image.width*image.height;
  delete [] image.buffer;
  image.buffer = new unsigned char [numBytes];
  for (size_t n=0; n<numBytes; n++) 
    image.buffer[n] = 0;
  
  for ( size_t n = 0; n < glyphs.size(); n++ )
    {
      FT_BBox bbox;
      
      FT_Glyph_Get_CBox(glyphs[n], ft_glyph_bbox_pixels, &bbox);
      
      
      error = FT_Glyph_To_Bitmap(&glyphs[n],
				 ft_render_mode_normal,
				 0,
				 //&pos[n],
				 1  //destroy image; 
				 );

      
      if (error) 
	throw Py::RuntimeError("Could not convert glyph to bitmap");
      
      FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[n];
      /* now, draw to our target surface (convert position) */
      
      
      //bitmap left and top in pixel, string bbox in subpixel
      
      draw_bitmap( &bitmap->bitmap, 
		   bitmap->left-string_bbox.xMin/64,
		   string_bbox.yMax/64-bitmap->top+1
		   );
      
      
    }
  
  return Py::Object();
}


char FT2Font::draw_glyph_to_bitmap__doc__[] = 
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

Py::Object
FT2Font::draw_glyph_to_bitmap(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::draw_glyph_to_bitmap" << std::endl;
  args.verify_length(3);
  
  if (image.width==0 || image.height==0)
    throw Py::RuntimeError("You must first set the size of the bitmap with set_bitmap_size");
  
  
  
  long x = Py::Int(args[0]);
  long y = Py::Int(args[1]);
  if (!Glyph::check(args[2].ptr()))
    throw Py::TypeError("Usage: draw_glyph_to_bitmap(x,y,glyph)");
  Glyph* glyph = static_cast<Glyph*>(args[2].ptr());
  
  if ((size_t)glyph->glyphInd >= glyphs.size())
    throw Py::ValueError("glyph num is out of range");

  error = FT_Glyph_To_Bitmap(&glyphs[glyph->glyphInd],
			     ft_render_mode_normal,
			     0,  //no additional translation
			     1   //destroy image; 
			     );

  
  if (error) 
    throw Py::RuntimeError("Could not convert glyph to bitmap");
  
  FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[glyph->glyphInd];

  draw_bitmap( &bitmap->bitmap, 
	       //x + bitmap->left,
	       x,
	       //y+bitmap->top
	       y
	       );

  return Py::Object();
  
}

// ID        Platform       Encoding
// 0         Unicode        Reserved (set to 0)
// 1         Macintoch      The Script Manager code
// 2         ISO            ISO encoding
// 3         Microsoft      Microsoft encoding
// 240-255   User-defined   Reserved for all nonregistered platforms

// Code      ISO encoding scheme
// 0         7-bit ASCII
// 1         ISO 10646
// 2         ISO 8859-1

// Code      Language       Code      Language       Code
// 0         English        10        Hebrew         20        Urdu
// 1         French         11        Japanese       21        Hindi
// 2         German         12        Arabic         22        Thai
// 3         Italian        13        Finnish
// 4         Dutch          14        Greek
// 5         Swedish        15        Icelandic
// 6         Spanish        16        Maltese
// 7         Danish         17        Turkish
// 8         Portuguese     18        Yugoslavian
// 9         Norwegian      19        Chinese

// Code      Meaning        Description
// 0         Copyright notice     e.g. "Copyright Apple Computer, Inc. 1992
// 1         Font family name     e.g. "New York"
// 2         Font style           e.g. "Bold"
// 3         Font identification  e.g. "Apple Computer New York Bold Ver 1"
// 4         Full font name       e.g. "New York Bold"
// 5         Version string       e.g. "August 10, 1991, 1.08d21"
// 6         Postscript name      e.g. "Times-Bold"
// 7         Trademark            
// 8         Designer             e.g. "Apple Computer"

char FT2Font::get_sfnt_name__doc__[] =
"get_sfnt_name(name)\n"
"\n"
"Get a value from the SFNT names table, if present\n"
"The font name identifier codes are:\n"
"\n"
"  0    Copyright notice     e.g. Copyright Apple Computer, Inc. 1992\n"
"  1    Font family name     e.g. New York\n"
"  2    Font style           e.g. Bold\n"
"  3    Font identification  e.g. Apple Computer New York Bold Ver 1\n"
"  4    Full font name       e.g. New York Bold\n"
"  5    Version string       e.g. August 10, 1991, 1.08d21\n"
"  6    Postscript name      e.g. Times-Bold\n"
"  7    Trademark            \n"
"  8    Designer             e.g. Apple Computer\n"
"  11   URL                  e.g. http://www.apple.com\n"
"  13   Copyright license    \n"
;

Py::Object
FT2Font::get_sfnt_name(const Py::Tuple & args) {
  if (_DEBUG) std::cout << "FT2Font::get_sfnt_name" << std::endl;
  args.verify_length(1);
  
  if (!(face->face_flags & FT_FACE_FLAG_SFNT)) 
    throw Py::RuntimeError("No SFNT name table");
  
  int name_id = Py::Int(args[0]);
  
  for (size_t j = 0; j < FT_Get_Sfnt_Name_Count(face); j++) {
    FT_Error error;
    FT_SfntName sfnt;
    
    error = FT_Get_Sfnt_Name(face, j, &sfnt);
    
    if (error) 
      throw Py::RuntimeError("Could not get SFNT name");
    
    //printf("%d %d %d %d\n", sfnt.platform_id, sfnt.encoding_id,
    //       sfnt.language_id, sfnt.name_id, name_id);
    // Test for Microsoft, 7-bit ASCII, English first
    if (sfnt.platform_id == 1 && sfnt.encoding_id == 0 &&
	sfnt.language_id == 0 && sfnt.name_id == name_id) {
      //printf("%d\n", sfnt.string_len);
      return Py::Object(Py_BuildValue("s#", sfnt.string, sfnt.string_len));
    }
  }
  
  
  return Py::Object();
}


Py::Object 
ft2font_module::new_ft2font (const Py::Tuple &args) {
  if (_DEBUG) std::cout << "ft2font_module::new_ft2font " << std::endl;
  args.verify_length(1);
  
  std::string facefile = Py::String(args[0]);
  return Py::asObject( new FT2Font(facefile) );
}   


void 
Glyph::init_type() {
  if (_DEBUG) std::cout << "Glyph::init_type" << std::endl;
  behaviors().name("Glyph");
  behaviors().doc("Glyph");
  behaviors().supportGetattr();
  behaviors().supportSetattr();

}

void 
FT2Font::init_type() {
  if (_DEBUG) std::cout << "FT2Font::init_type" << std::endl;
  behaviors().name("FT2Font");
  behaviors().doc("FT2Font");
  
  
  add_varargs_method("clear", &FT2Font::clear, 
		     FT2Font::clear__doc__);
  add_varargs_method("write_bitmap", &FT2Font::write_bitmap, 
		     FT2Font::write_bitmap__doc__);
  add_varargs_method("set_bitmap_size", &FT2Font::set_bitmap_size, 
		     FT2Font::load_char__doc__);
  add_varargs_method("draw_rect",&FT2Font::draw_rect,
		     FT2Font::draw_rect__doc__);
  add_varargs_method("draw_glyph_to_bitmap", &FT2Font::draw_glyph_to_bitmap, 
		     FT2Font::draw_glyph_to_bitmap__doc__);
  add_varargs_method("draw_glyphs_to_bitmap", &FT2Font::draw_glyphs_to_bitmap, 
		     FT2Font::draw_glyphs_to_bitmap__doc__);
  add_varargs_method("get_glyph", &FT2Font::get_glyph, 
		     FT2Font::get_glyph__doc__);
  add_varargs_method("get_num_glyphs", &FT2Font::get_num_glyphs, 
		     FT2Font::get_num_glyphs__doc__);
  add_varargs_method("image_as_str", &FT2Font::image_as_str, 
		     FT2Font::image_as_str__doc__);
  add_varargs_method("load_char", &FT2Font::load_char, 
		     FT2Font::load_char__doc__);
  add_varargs_method("set_text", &FT2Font::set_text, 
		     FT2Font::set_text__doc__);
  add_varargs_method("set_size", &FT2Font::set_size, 
		     FT2Font::set_size__doc__);
  add_varargs_method("get_width_height", &FT2Font::get_width_height, 
		     FT2Font::get_width_height__doc__);
  add_varargs_method("get_sfnt_name", &FT2Font::get_sfnt_name,  
		     FT2Font::get_sfnt_name__doc__);
  
  behaviors().supportGetattr();
  behaviors().supportSetattr();
};

//todo add module docs strings


char ft2font__doc__[] =
"ft2font\n"
"\n"
"Methods:\n"
"  FT2Font(ttffile)\n"
"Face Constants\n"
"  SCALABLE               scalable\n"
"  FIXED_SIZES            \n"
"  FIXED_WIDTH            \n"
"  SFNT                   \n"
"  HORIZONTAL             \n"
"  VERTICAL               \n"
"  KERNING                \n"
"  FAST_GLYPHS            \n"
"  MULTIPLE_MASTERS       \n"
"  GLYPH_NAMES            \n"
"  EXTERNAL_STREAM        \n"
"Style Constants\n"
"  ITALIC                 \n"
"  BOLD                   \n"
;

/* Function of no arguments returning new FT2Font object */
char ft2font_new__doc__[] = 
"FT2Font(ttffile)\n"
"\n"
"Create a new FT2Font object\n"
"The following global font attributes are defined:\n"
"  num_faces              number of faces in file\n"
"  face_flags             face flags  (int type); see the ft2font constants\n"
"  style_flags            style flags  (int type); see the ft2font constants\n"
"  num_glyphs             number of glyphs in the face\n"
"  family_name            face family name\n"
"  style_name             face syle name\n"
"  num_fixed_sizes        number of bitmap in the face\n"
"  scalable               face is scalable\n"
"\n"
"The following are available, if scalable is true:\n"
"  bbox                   face global bounding box (xmin, ymin, xmax, ymax)\n"
"  units_per_EM           number of font units covered by the EM\n"
"  ascender               ascender in 26.6 units\n"
"  descender              descender in 26.6 units\n"
"  height                 height in 26.6 units; used to compute a default\n"
"                         line spacing (baseline-to-baseline distance)\n"
"  max_advance_width      maximum horizontal cursor advance for all glyphs\n"
"  max_advance_height     same for vertical layout\n"
"  underline_position     vertical position of the underline bar\n"
"  underline_thickness    vertical thickness of the underline\n"
;


#if defined(_MSC_VER)
DL_EXPORT(void)
#elif defined(__cplusplus)
  extern "C" void
#else
void
#endif
initft2font(void)
{
  static ft2font_module* ft2font = new ft2font_module;
  
  Py::Dict d = ft2font->moduleDictionary();
  d["SCALABLE"] 	= Py::Int(FT_FACE_FLAG_SCALABLE);
  d["FIXED_SIZES"] 	= Py::Int(FT_FACE_FLAG_FIXED_SIZES);
  d["FIXED_WIDTH"] 	= Py::Int(FT_FACE_FLAG_FIXED_WIDTH);
  d["SFNT"] 		= Py::Int(FT_FACE_FLAG_SFNT);
  d["HORIZONTAL"] 	= Py::Int(FT_FACE_FLAG_HORIZONTAL);
  d["VERTICAL"] 	= Py::Int(FT_FACE_FLAG_SCALABLE);
  d["KERNING"] 		= Py::Int(FT_FACE_FLAG_KERNING);
  d["FAST_GLYPHS"] 	= Py::Int(FT_FACE_FLAG_FAST_GLYPHS);
  d["MULTIPLE_MASTERS"] = Py::Int(FT_FACE_FLAG_MULTIPLE_MASTERS);
  d["GLYPH_NAMES"] 	= Py::Int(FT_FACE_FLAG_GLYPH_NAMES);
  d["EXTERNAL_STREAM"] 	= Py::Int(FT_FACE_FLAG_EXTERNAL_STREAM);
  d["ITALIC"] 		= Py::Int(FT_STYLE_FLAG_ITALIC);
  d["BOLD"] 		= Py::Int(FT_STYLE_FLAG_BOLD);
  
  
  //initialize library    
  int error = FT_Init_FreeType( &_ft2Library ); 
  
  if (error) 
    throw Py::RuntimeError("Could not find initialize the freetype2 library");
  
};

ft2font_module::~ft2font_module() {
  
    FT_Done_FreeType( _ft2Library );
}

//----------------------------------------------------------------------------
//
//----------------------------------------------------------------------------
// Contact: mcseemagg@yahoo.com
//          baer@karto.baug.ethz.ch
//----------------------------------------------------------------------------
//
// class pixel_map
//
//----------------------------------------------------------------------------

#include <string.h>
#include <Carbon.h>
#include <QuickTimeComponents.h>
#include <ImageCompression.h>
#include "platform/mac/agg_mac_pmap.h"
#include "agg_basics.h"

namespace agg
{

    //------------------------------------------------------------------------
    pixel_map::~pixel_map()
    {
        destroy();
    }


    //------------------------------------------------------------------------
    pixel_map::pixel_map() :
        m_pmap(0),
        m_buf(0),
        m_bpp(0),
        m_img_size(0)

    {
    }


    //------------------------------------------------------------------------
    void pixel_map::destroy()
    {
		delete[] m_buf;
		m_buf = NULL;
		if (m_pmap != nil)
		{
			DisposeGWorld(m_pmap);
			m_pmap = nil;
		}
    }


    //------------------------------------------------------------------------
    void pixel_map::create(unsigned width, 
                           unsigned height, 
                           org_e    org,
                           unsigned clear_val)
    {
        destroy();
        if(width == 0)  width = 1;
        if(height == 0) height = 1;
        m_bpp = org;
        
        Rect	r;
    	int		row_bytes = calc_row_len (width, m_bpp);
    	MacSetRect(&r, 0, 0, width, height);
    	m_buf = new unsigned char[m_img_size = row_bytes * height];
 		// The Quicktime version for creating GWorlds is more flexible than the classical function.
    	QTNewGWorldFromPtr (&m_pmap, m_bpp, &r, nil, nil, 0, m_buf, row_bytes);

        // create_gray_scale_palette(m_pmap);  I didn't care about gray scale palettes so far.
        if(clear_val <= 255)
        {
            memset(m_buf, clear_val, m_img_size);
        }
    }



    //------------------------------------------------------------------------
    void pixel_map::clear(unsigned clear_val)
    {
        if(m_buf) memset(m_buf, clear_val, m_img_size);
    }


    //static
    //This function is just copied from the Win32 plattform support.
    //Is also seems to be appropriate for MacOS as well, but it is not
    //thouroughly tested so far.
    //------------------------------------------------------------------------

    unsigned pixel_map::calc_row_len(unsigned width, unsigned bits_per_pixel)
    {
        unsigned n = width;
        unsigned k;

        switch(bits_per_pixel)
        {
            case  1: k = n;
                     n = n >> 3;
                     if(k & 7) n++; 
                     break;

            case  4: k = n;
                     n = n >> 1;
                     if(k & 3) n++; 
                     break;

            case  8:
                     break;

            case 16: n = n << 1;
                     break;

            case 24: n = (n << 1) + n; 
                     break;

            case 32: n = n << 2;
                     break;

            default: n = 0;
                     break;
        }
        return ((n + 3) >> 2) << 2;
    }




    //------------------------------------------------------------------------
    void pixel_map::draw(WindowRef window, const Rect *device_rect, const Rect *pmap_rect) const
    {
        if(m_pmap == nil || m_buf == NULL) return;

		PixMapHandle	pm = GetGWorldPixMap (m_pmap);
		CGrafPtr		port = GetWindowPort (window);
		Rect			dest_rect;

		// Again, I used the Quicktime version.
		// Good old 'CopyBits' does better interpolation when scaling
		// but does not support all pixel depths.
		MacSetRect (&dest_rect, 0, 0, this->width(), this->height());
		ImageDescriptionHandle		image_description;
		MakeImageDescriptionForPixMap (pm, &image_description);	   
		if (image_description != nil)
		{
			DecompressImage (GetPixBaseAddr (pm), image_description, GetPortPixMap (port), nil, &dest_rect, ditherCopy, nil);	   
			DisposeHandle ((Handle) image_description);
		}
	}


    //------------------------------------------------------------------------
    void pixel_map::draw(WindowRef window, int x, int y, double scale) const
    {
        if(m_pmap == nil || m_buf == NULL) return;
        unsigned width  = (unsigned)(this->width() * scale);
        unsigned height = (unsigned)(this->height() * scale);
        Rect rect;
        SetRect (&rect, x, y, x + width, y + height);
        draw(window, &rect);
    }



    //------------------------------------------------------------------------
    void pixel_map::blend(WindowRef window, const Rect *device_rect, const Rect *bmp_rect) const
    {
		draw (window, device_rect, bmp_rect);	// currently just mapped to drawing method
    }
    

    //------------------------------------------------------------------------
    void pixel_map::blend(WindowRef window, int x, int y, double scale) const
    {
        draw(window, x, y, scale);	// currently just mapped to drawing method
    }


    // I let Quicktime handle image import since it supports most popular
    // image formats such as:
    // *.psd, *.bmp, *.tif, *.png, *.jpg, *.gif, *.pct, *.pcx
    //------------------------------------------------------------------------
    bool pixel_map::load_from_qt(const char *filename)
    {
		FSSpec						fss;
		OSErr						err;
		
		// get file specification to application directory
		err = HGetVol(nil, &fss.vRefNum, &fss.parID);
		if (err == noErr)
		{
			CopyCStringToPascal(filename, fss.name);
			GraphicsImportComponent		gi;
			err = GetGraphicsImporterForFile (&fss, &gi);
			if (err == noErr)
			{
				ImageDescriptionHandle	desc;
				GraphicsImportGetImageDescription(gi, &desc);
// For simplicity, all images are currently converted to 32 bit.
				// create an empty pixelmap
				short depth = 32;
				create ((**desc).width, (**desc).height, (org_e)depth, 0xff);
				DisposeHandle ((Handle)desc);
				// let Quicktime draw to pixelmap
				GraphicsImportSetGWorld(gi, m_pmap, nil);
				GraphicsImportDraw(gi);
// Well, this is a hack. The graphics importer sets the alpha channel of the pixelmap to 0x00
// for imported images without alpha channel but this would cause agg to draw an invisible image.
				// set alpha channel to 0xff
				unsigned char * buf = m_buf;
				for (unsigned int size = 0; size < m_img_size; size += 4)
				{
					*buf = 0xff;
					buf += 4;
				}
			}
		}
        return err == noErr;
    }



    //------------------------------------------------------------------------
    bool pixel_map::save_as_qt(const char *filename) const
    {
		FSSpec						fss;
 		OSErr						err;
		
		// get file specification to application directory
		err = HGetVol(nil, &fss.vRefNum, &fss.parID);
		if (err == noErr)
		{
     		GraphicsExportComponent		ge;
  			CopyCStringToPascal(filename, fss.name);
  			// I decided to use PNG as output image file type.
  			// There are a number of other available formats.
  			// Should I check the file suffix to choose the image file format?
			err = OpenADefaultComponent(GraphicsExporterComponentType, kQTFileTypePNG, &ge);
			if (err == noErr)
			{
    			err = GraphicsExportSetInputGWorld(ge, m_pmap);
	    		if (err == noErr)
    			{
    				err = GraphicsExportSetOutputFile (ge, &fss);
    				if (err == noErr)
    				{
    					GraphicsExportDoExport(ge, nil);
    				}
    			}
    			CloseComponent(ge);
    		}
    	}
    	
        return err == noErr;
    }

    //------------------------------------------------------------------------
    unsigned char* pixel_map::buf()
    {
        return m_buf;
    }

    //------------------------------------------------------------------------
    unsigned pixel_map::width() const
    {
        if(m_pmap == nil) return 0;
		PixMapHandle	pm = GetGWorldPixMap (m_pmap);
		Rect			bounds;
		GetPixBounds (pm, &bounds);
		return bounds.right - bounds.left;
    }

    //------------------------------------------------------------------------
    unsigned pixel_map::height() const
    {
        if(m_pmap == nil) return 0;
		PixMapHandle	pm = GetGWorldPixMap (m_pmap);
		Rect			bounds;
		GetPixBounds (pm, &bounds);
		return bounds.bottom - bounds.top;
    }

    //------------------------------------------------------------------------
    int pixel_map::row_bytes() const
    {
        if(m_pmap == nil) return 0;
 		PixMapHandle	pm = GetGWorldPixMap (m_pmap);
        return calc_row_len(width(), GetPixDepth(pm));
    }



}




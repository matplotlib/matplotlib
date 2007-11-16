//----------------------------------------------------------------------------
//
//----------------------------------------------------------------------------
// Contact: mcseemagg@yahoo.com
//----------------------------------------------------------------------------
//
// class pixel_map
//
//----------------------------------------------------------------------------

#include "platform/win32/agg_win32_bmp.h"
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
        m_bmp(0),
        m_buf(0),
        m_bpp(0),
        m_is_internal(false),
        m_img_size(0),
        m_full_size(0)

    {
    }


    //------------------------------------------------------------------------
    void pixel_map::destroy()
    {
        if(m_bmp && m_is_internal) delete [] (unsigned char*)m_bmp;
        m_bmp  = 0;
        m_is_internal = false;
        m_buf = 0;
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
        create_from_bmp(create_bitmap_info(width, height, m_bpp));
        create_gray_scale_palette(m_bmp);
        m_is_internal = true;
        if(clear_val <= 255)
        {
            memset(m_buf, clear_val, m_img_size);
        }
    }


    //------------------------------------------------------------------------
    HBITMAP pixel_map::create_dib_section(HDC h_dc,
                                          unsigned width, 
                                          unsigned height, 
                                          org_e    org,
                                          unsigned clear_val)
    {
        destroy();
        if(width == 0)  width = 1;
        if(height == 0) height = 1;
        m_bpp = org;
        HBITMAP h_bitmap = create_dib_section_from_args(h_dc, width, height, m_bpp);
        create_gray_scale_palette(m_bmp);
        m_is_internal = true;
        if(clear_val <= 255)
        {
            memset(m_buf, clear_val, m_img_size);
        }
        return h_bitmap;
    }



    //------------------------------------------------------------------------
    void pixel_map::clear(unsigned clear_val)
    {
        if(m_buf) memset(m_buf, clear_val, m_img_size);
    }


    //------------------------------------------------------------------------
    void pixel_map::attach_to_bmp(BITMAPINFO *bmp)
    {
        if(bmp)
        {
            destroy();
            create_from_bmp(bmp);
            m_is_internal = false;
        }
    }



    //static
    //------------------------------------------------------------------------
    unsigned pixel_map::calc_full_size(BITMAPINFO *bmp)
    {
        if(bmp == 0) return 0;

        return sizeof(BITMAPINFOHEADER) +
               sizeof(RGBQUAD) * calc_palette_size(bmp) +
               bmp->bmiHeader.biSizeImage;
    }

    //static
    //------------------------------------------------------------------------
    unsigned pixel_map::calc_header_size(BITMAPINFO *bmp)
    {
        if(bmp == 0) return 0;
        return sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * calc_palette_size(bmp);
    }


    //static
    //------------------------------------------------------------------------
    unsigned  pixel_map::calc_palette_size(unsigned  clr_used, unsigned bits_per_pixel)
    {
        int palette_size = 0;

        if(bits_per_pixel <= 8)
        {
            palette_size = clr_used;
            if(palette_size == 0)
            {
                palette_size = 1 << bits_per_pixel;
            }
        }
        return palette_size;
    }

    //static
    //------------------------------------------------------------------------
    unsigned pixel_map::calc_palette_size(BITMAPINFO *bmp)
    {
        if(bmp == 0) return 0;
        return calc_palette_size(bmp->bmiHeader.biClrUsed, bmp->bmiHeader.biBitCount);
    }


    //static
    //------------------------------------------------------------------------
    unsigned char * pixel_map::calc_img_ptr(BITMAPINFO *bmp)
    {
        if(bmp == 0) return 0;
        return ((unsigned char*)bmp) + calc_header_size(bmp);
    }

    //static
    //------------------------------------------------------------------------
    BITMAPINFO* pixel_map::create_bitmap_info(unsigned width, 
                                              unsigned height, 
                                              unsigned bits_per_pixel)
    {
        unsigned line_len = calc_row_len(width, bits_per_pixel);
        unsigned img_size = line_len * height;
        unsigned rgb_size = calc_palette_size(0, bits_per_pixel) * sizeof(RGBQUAD);
        unsigned full_size = sizeof(BITMAPINFOHEADER) + rgb_size + img_size;

        BITMAPINFO *bmp = (BITMAPINFO *) new unsigned char[full_size];

        bmp->bmiHeader.biSize   = sizeof(BITMAPINFOHEADER);
        bmp->bmiHeader.biWidth  = width;
        bmp->bmiHeader.biHeight = height;
        bmp->bmiHeader.biPlanes = 1;
        bmp->bmiHeader.biBitCount = (unsigned short)bits_per_pixel;
        bmp->bmiHeader.biCompression = 0;
        bmp->bmiHeader.biSizeImage = img_size;
        bmp->bmiHeader.biXPelsPerMeter = 0;
        bmp->bmiHeader.biYPelsPerMeter = 0;
        bmp->bmiHeader.biClrUsed = 0;
        bmp->bmiHeader.biClrImportant = 0;

        return bmp;
    }


    //static
    //------------------------------------------------------------------------
    void pixel_map::create_gray_scale_palette(BITMAPINFO *bmp)
    {
        if(bmp == 0) return;

        unsigned rgb_size = calc_palette_size(bmp);
        RGBQUAD *rgb = (RGBQUAD*)(((unsigned char*)bmp) + sizeof(BITMAPINFOHEADER));
        unsigned brightness;
        unsigned i;

        for(i = 0; i < rgb_size; i++)
        {
            brightness = (255 * i) / (rgb_size - 1);
            rgb->rgbBlue =
            rgb->rgbGreen =  
            rgb->rgbRed = (unsigned char)brightness; 
            rgb->rgbReserved = 0;
            rgb++;
        }
    }



    //static
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

            case 16: n *= 2;
                     break;

            case 24: n *= 3; 
                     break;

            case 32: n *= 4;
                     break;

            case 48: n *= 6; 
                     break;

            case 64: n *= 8; 
                     break;

            default: n = 0;
                     break;
        }
        return ((n + 3) >> 2) << 2;
    }





    //------------------------------------------------------------------------
    void pixel_map::draw(HDC h_dc, const RECT *device_rect, const RECT *bmp_rect) const
    {
        if(m_bmp == 0 || m_buf == 0) return;

        unsigned bmp_x = 0;
        unsigned bmp_y = 0;
        unsigned bmp_width  = m_bmp->bmiHeader.biWidth;
        unsigned bmp_height = m_bmp->bmiHeader.biHeight;
        unsigned dvc_x = 0;
        unsigned dvc_y = 0; 
        unsigned dvc_width  = m_bmp->bmiHeader.biWidth;
        unsigned dvc_height = m_bmp->bmiHeader.biHeight;
        
        if(bmp_rect) 
        {
            bmp_x      = bmp_rect->left;
            bmp_y      = bmp_rect->top;
            bmp_width  = bmp_rect->right  - bmp_rect->left;
            bmp_height = bmp_rect->bottom - bmp_rect->top;
        } 

        dvc_x      = bmp_x;
        dvc_y      = bmp_y;
        dvc_width  = bmp_width;
        dvc_height = bmp_height;

        if(device_rect) 
        {
            dvc_x      = device_rect->left;
            dvc_y      = device_rect->top;
            dvc_width  = device_rect->right  - device_rect->left;
            dvc_height = device_rect->bottom - device_rect->top;
        }

        if(dvc_width != bmp_width || dvc_height != bmp_height)
        {
            ::SetStretchBltMode(h_dc, COLORONCOLOR);
            ::StretchDIBits(
                h_dc,            // handle of device context 
                dvc_x,           // x-coordinate of upper-left corner of source rect. 
                dvc_y,           // y-coordinate of upper-left corner of source rect. 
                dvc_width,       // width of source rectangle 
                dvc_height,      // height of source rectangle 
                bmp_x,
                bmp_y,           // x, y -coordinates of upper-left corner of dest. rect. 
                bmp_width,       // width of destination rectangle 
                bmp_height,      // height of destination rectangle 
                m_buf,           // address of bitmap bits 
                m_bmp,           // address of bitmap data 
                DIB_RGB_COLORS,  // usage 
                SRCCOPY          // raster operation code 
            );
        }
        else
        {
            ::SetDIBitsToDevice(
                h_dc,            // handle to device context
                dvc_x,           // x-coordinate of upper-left corner of 
                dvc_y,           // y-coordinate of upper-left corner of 
                dvc_width,       // source rectangle width
                dvc_height,      // source rectangle height
                bmp_x,           // x-coordinate of lower-left corner of 
                bmp_y,           // y-coordinate of lower-left corner of 
                0,               // first scan line in array
                bmp_height,      // number of scan lines
                m_buf,           // address of array with DIB bits
                m_bmp,           // address of structure with bitmap info.
                DIB_RGB_COLORS   // RGB or palette indexes
            );
        }
    }


    //------------------------------------------------------------------------
    void pixel_map::draw(HDC h_dc, int x, int y, double scale) const
    {
        if(m_bmp == 0 || m_buf == 0) return;

        unsigned width  = unsigned(m_bmp->bmiHeader.biWidth * scale);
        unsigned height = unsigned(m_bmp->bmiHeader.biHeight * scale);
        RECT rect;
        rect.left   = x;
        rect.top    = y;
        rect.right  = x + width;
        rect.bottom = y + height;
        draw(h_dc, &rect);
    }




    //------------------------------------------------------------------------
    void pixel_map::blend(HDC h_dc, const RECT *device_rect, const RECT *bmp_rect) const
    {
#if !defined(AGG_BMP_ALPHA_BLEND)
        draw(h_dc, device_rect, bmp_rect);
        return;
#else
        if(m_bpp != 32)
        {
            draw(h_dc, device_rect, bmp_rect);
            return;
        }

        if(m_bmp == 0 || m_buf == 0) return;

        unsigned bmp_x = 0;
        unsigned bmp_y = 0;
        unsigned bmp_width  = m_bmp->bmiHeader.biWidth;
        unsigned bmp_height = m_bmp->bmiHeader.biHeight;
        unsigned dvc_x = 0;
        unsigned dvc_y = 0; 
        unsigned dvc_width  = m_bmp->bmiHeader.biWidth;
        unsigned dvc_height = m_bmp->bmiHeader.biHeight;
        
        if(bmp_rect) 
        {
            bmp_x      = bmp_rect->left;
            bmp_y      = bmp_rect->top;
            bmp_width  = bmp_rect->right  - bmp_rect->left;
            bmp_height = bmp_rect->bottom - bmp_rect->top;
        } 

        dvc_x      = bmp_x;
        dvc_y      = bmp_y;
        dvc_width  = bmp_width;
        dvc_height = bmp_height;

        if(device_rect) 
        {
            dvc_x      = device_rect->left;
            dvc_y      = device_rect->top;
            dvc_width  = device_rect->right  - device_rect->left;
            dvc_height = device_rect->bottom - device_rect->top;
        }

        HDC mem_dc = ::CreateCompatibleDC(h_dc);
        void* buf = 0;
        HBITMAP bmp = ::CreateDIBSection(
            mem_dc, 
            m_bmp,  
            DIB_RGB_COLORS,
            &buf,
            0,
            0
        );
        memcpy(buf, m_buf, m_bmp->bmiHeader.biSizeImage);

        HBITMAP temp = (HBITMAP)::SelectObject(mem_dc, bmp);

        BLENDFUNCTION blend;
        blend.BlendOp = AC_SRC_OVER;
        blend.BlendFlags = 0;

#if defined(AC_SRC_ALPHA)
        blend.AlphaFormat = AC_SRC_ALPHA;
//#elif defined(AC_SRC_NO_PREMULT_ALPHA)
//        blend.AlphaFormat = AC_SRC_NO_PREMULT_ALPHA;
#else 
#error "No appropriate constant for alpha format. Check version of wingdi.h, There must be AC_SRC_ALPHA or AC_SRC_NO_PREMULT_ALPHA"
#endif

        blend.SourceConstantAlpha = 255;
        ::AlphaBlend(
          h_dc,      
          dvc_x,      
          dvc_y,      
          dvc_width,  
          dvc_height, 
          mem_dc,
          bmp_x,
          bmp_y,     
          bmp_width, 
          bmp_height,
          blend
        );

        ::SelectObject(mem_dc, temp);
        ::DeleteObject(bmp);
        ::DeleteObject(mem_dc);
#endif //defined(AGG_BMP_ALPHA_BLEND)
    }


    //------------------------------------------------------------------------
    void pixel_map::blend(HDC h_dc, int x, int y, double scale) const
    {
        if(m_bmp == 0 || m_buf == 0) return;
        unsigned width  = unsigned(m_bmp->bmiHeader.biWidth * scale);
        unsigned height = unsigned(m_bmp->bmiHeader.biHeight * scale);
        RECT rect;
        rect.left   = x;
        rect.top    = y;
        rect.right  = x + width;
        rect.bottom = y + height;
        blend(h_dc, &rect);
    }


    //------------------------------------------------------------------------
    bool pixel_map::load_from_bmp(FILE *fd)
    {
        BITMAPFILEHEADER  bmf;
        BITMAPINFO       *bmi = 0;
        unsigned          bmp_size;

        fread(&bmf, sizeof(bmf), 1, fd);
        if(bmf.bfType != 0x4D42) goto bmperr;

        bmp_size = bmf.bfSize - sizeof(BITMAPFILEHEADER);

        bmi = (BITMAPINFO*) new unsigned char [bmp_size];
        if(fread(bmi, 1, bmp_size, fd) != bmp_size) goto bmperr;
        destroy();
        m_bpp = bmi->bmiHeader.biBitCount;
        create_from_bmp(bmi);
        m_is_internal = 1;
        return true;

    bmperr:
        if(bmi) delete [] (unsigned char*) bmi;
        return false;
    }



    //------------------------------------------------------------------------
    bool pixel_map::load_from_bmp(const char *filename)
    {
        FILE *fd = fopen(filename, "rb");
        bool ret = false;
        if(fd)
        {
            ret = load_from_bmp(fd);
            fclose(fd);
        }
        return ret;
    }



    //------------------------------------------------------------------------
    bool pixel_map::save_as_bmp(FILE *fd) const
    {
        if(m_bmp == 0) return 0;

        BITMAPFILEHEADER bmf;

        bmf.bfType      = 0x4D42;
        bmf.bfOffBits   = calc_header_size(m_bmp) + sizeof(bmf);
        bmf.bfSize      = bmf.bfOffBits + m_img_size;
        bmf.bfReserved1 = 0;
        bmf.bfReserved2 = 0;

        fwrite(&bmf, sizeof(bmf), 1, fd);
        fwrite(m_bmp, m_full_size, 1, fd);
        return true;
    }



    //------------------------------------------------------------------------
    bool pixel_map::save_as_bmp(const char *filename) const
    {
        FILE *fd = fopen(filename, "wb");
        bool ret = false;
        if(fd)
        {
            ret = save_as_bmp(fd);
            fclose(fd);
        }
        return ret;
    }


    //------------------------------------------------------------------------
    unsigned char* pixel_map::buf()
    {
        return m_buf;
    }

    //------------------------------------------------------------------------
    unsigned pixel_map::width() const
    {
        return m_bmp->bmiHeader.biWidth;
    }

    //------------------------------------------------------------------------
    unsigned pixel_map::height() const
    {
        return m_bmp->bmiHeader.biHeight;
    }

    //------------------------------------------------------------------------
    int pixel_map::stride() const
    {
        return calc_row_len(m_bmp->bmiHeader.biWidth, 
                            m_bmp->bmiHeader.biBitCount);
    }


    //private
    //------------------------------------------------------------------------
    void pixel_map::create_from_bmp(BITMAPINFO *bmp)
    {
        if(bmp)
        {
            m_img_size  = calc_row_len(bmp->bmiHeader.biWidth, 
                                       bmp->bmiHeader.biBitCount) * 
                          bmp->bmiHeader.biHeight;

            m_full_size = calc_full_size(bmp);
            m_bmp       = bmp;
            m_buf       = calc_img_ptr(bmp);
        }
    }


    //private
    //------------------------------------------------------------------------
    HBITMAP pixel_map::create_dib_section_from_args(HDC h_dc,
                                                    unsigned width, 
                                                    unsigned height, 
                                                    unsigned bits_per_pixel)
    {
        unsigned line_len  = calc_row_len(width, bits_per_pixel);
        unsigned img_size  = line_len * height;
        unsigned rgb_size  = calc_palette_size(0, bits_per_pixel) * sizeof(RGBQUAD);
        unsigned full_size = sizeof(BITMAPINFOHEADER) + rgb_size;
        
        BITMAPINFO *bmp = (BITMAPINFO *) new unsigned char[full_size];
        
        bmp->bmiHeader.biSize   = sizeof(BITMAPINFOHEADER);
        bmp->bmiHeader.biWidth  = width;
        bmp->bmiHeader.biHeight = height;
        bmp->bmiHeader.biPlanes = 1;
        bmp->bmiHeader.biBitCount = (unsigned short)bits_per_pixel;
        bmp->bmiHeader.biCompression = 0;
        bmp->bmiHeader.biSizeImage = img_size;
        bmp->bmiHeader.biXPelsPerMeter = 0;
        bmp->bmiHeader.biYPelsPerMeter = 0;
        bmp->bmiHeader.biClrUsed = 0;
        bmp->bmiHeader.biClrImportant = 0;
        
        void*   img_ptr  = 0;
        HBITMAP h_bitmap = ::CreateDIBSection(h_dc, bmp, DIB_RGB_COLORS, &img_ptr, NULL, 0);
        
        if(img_ptr)
        {
            m_img_size  = calc_row_len(width, bits_per_pixel) * height;
            m_full_size = 0;
            m_bmp       = bmp;
            m_buf       = (unsigned char *) img_ptr;
        }
        
        return h_bitmap;
    }
}




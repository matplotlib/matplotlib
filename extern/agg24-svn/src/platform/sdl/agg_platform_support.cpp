//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// class platform_support. SDL version.
//
//----------------------------------------------------------------------------

#include <string.h>
#include "platform/agg_platform_support.h"
#include "SDL.h"
#include "SDL_byteorder.h"


namespace agg
{

    //------------------------------------------------------------------------
    class platform_specific
    {
    public:
        platform_specific(pix_format_e format, bool flip_y);
        ~platform_specific();

        pix_format_e  m_format;
        pix_format_e  m_sys_format;
        bool          m_flip_y;
        unsigned      m_bpp;
        unsigned      m_sys_bpp;
        unsigned      m_rmask;
        unsigned      m_gmask;
        unsigned      m_bmask;
        unsigned      m_amask;
        bool          m_update_flag;
        bool          m_resize_flag;
        bool          m_initialized;
        SDL_Surface*  m_surf_screen;
        SDL_Surface*  m_surf_window;
        SDL_Surface*  m_surf_img[platform_support::max_images];
        int           m_cur_x;
        int           m_cur_y;
	int          m_sw_start;
    };



    //------------------------------------------------------------------------
    platform_specific::platform_specific(pix_format_e format, bool flip_y) :
        m_format(format),
        m_sys_format(pix_format_undefined),
        m_flip_y(flip_y),
        m_bpp(0),
        m_sys_bpp(0),
        m_update_flag(true), 
        m_resize_flag(true),
        m_initialized(false),
        m_surf_screen(0),
        m_surf_window(0),
        m_cur_x(0),
        m_cur_y(0)
    {
        memset(m_surf_img, 0, sizeof(m_surf_img));

        switch(m_format)
        {
			case pix_format_gray8:
            m_bpp = 8;
            break;

        case pix_format_rgb565:
            m_rmask = 0xF800;
            m_gmask = 0x7E0;
            m_bmask = 0x1F;
            m_amask = 0;
            m_bpp = 16;
            break;

        case pix_format_rgb555:
            m_rmask = 0x7C00;
            m_gmask = 0x3E0;
            m_bmask = 0x1F;
            m_amask = 0;
            m_bpp = 16;
            break;
			
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        case pix_format_rgb24:
			m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgr24:
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgra32:
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;

        case pix_format_abgr32:
            m_rmask = 0xFF000000;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF00;
            m_amask = 0xFF;
            m_bpp = 32;
            break;

        case pix_format_argb32:
            m_rmask = 0xFF00;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF000000;
            m_amask = 0xFF;
            m_bpp = 32;
            break;

        case pix_format_rgba32:
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;
#else //SDL_BIG_ENDIAN (PPC)
        case pix_format_rgb24:
			m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgr24:
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0;
            m_bpp = 24;
            break;

        case pix_format_bgra32:
            m_rmask = 0xFF00;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF000000;
            m_amask = 0xFF;
            m_bpp = 32;
            break;

        case pix_format_abgr32:
            m_rmask = 0xFF;
            m_gmask = 0xFF00;
            m_bmask = 0xFF0000;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;

        case pix_format_argb32:
            m_rmask = 0xFF0000;
            m_gmask = 0xFF00;
            m_bmask = 0xFF;
            m_amask = 0xFF000000;
            m_bpp = 32;
            break;

        case pix_format_rgba32:
            m_rmask = 0xFF000000;
            m_gmask = 0xFF0000;
            m_bmask = 0xFF00;
            m_amask = 0xFF;
            m_bpp = 32;
            break;
#endif
        }
    }

    //------------------------------------------------------------------------
    platform_specific::~platform_specific()
    {
        int i;
        for(i = platform_support::max_images - 1; i >= 0; --i)
        {
            if(m_surf_img[i]) SDL_FreeSurface(m_surf_img[i]);
        }
        if(m_surf_window) SDL_FreeSurface(m_surf_window);
        if(m_surf_screen) SDL_FreeSurface(m_surf_screen);
    }



    //------------------------------------------------------------------------
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        m_specific(new platform_specific(format, flip_y)),
        m_format(format),
        m_bpp(m_specific->m_bpp),
        m_window_flags(0),
        m_wait_mode(true),
        m_flip_y(flip_y)
    {
        SDL_Init(SDL_INIT_VIDEO);
        strcpy(m_caption, "Anti-Grain Geometry Application");
    }


    //------------------------------------------------------------------------
    platform_support::~platform_support()
    {
        delete m_specific;
    }



    //------------------------------------------------------------------------
    void platform_support::caption(const char* cap)
    {
        strcpy(m_caption, cap);
        if(m_specific->m_initialized)
        {
            SDL_WM_SetCaption(cap, 0);
        }
    }
    




    //------------------------------------------------------------------------
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        m_window_flags = flags;
        unsigned wflags = SDL_SWSURFACE;

        if(m_window_flags & window_hw_buffer)
        {
            wflags = SDL_HWSURFACE;
        }

        if(m_window_flags & window_resize)
        {
            wflags |= SDL_RESIZABLE;
        }

        if(m_specific->m_surf_screen) SDL_FreeSurface(m_specific->m_surf_screen);

        m_specific->m_surf_screen = SDL_SetVideoMode(width, height, m_bpp, wflags);
        if(m_specific->m_surf_screen == 0) 
        {
            fprintf(stderr, 
                    "Unable to set %dx%d %d bpp video: %s\n", 
                    width, 
                    height, 
                    m_bpp, 
                    ::SDL_GetError());
            return false;
        }

        SDL_WM_SetCaption(m_caption, 0);

        if(m_specific->m_surf_window) SDL_FreeSurface(m_specific->m_surf_window);

        m_specific->m_surf_window = 
            SDL_CreateRGBSurface(SDL_HWSURFACE, 
                                 m_specific->m_surf_screen->w, 
                                 m_specific->m_surf_screen->h,
                                 m_specific->m_surf_screen->format->BitsPerPixel,
                                 m_specific->m_rmask, 
                                 m_specific->m_gmask, 
                                 m_specific->m_bmask, 
                                 m_specific->m_amask);

        if(m_specific->m_surf_window == 0) 
        {
            fprintf(stderr, 
                    "Unable to create image buffer %dx%d %d bpp: %s\n", 
                    width, 
                    height, 
                    m_bpp, 
                    SDL_GetError());
            return false;
        }

        m_rbuf_window.attach((unsigned char*)m_specific->m_surf_window->pixels, 
                             m_specific->m_surf_window->w, 
                             m_specific->m_surf_window->h, 
                             m_flip_y ? -m_specific->m_surf_window->pitch : 
                                         m_specific->m_surf_window->pitch);

        if(!m_specific->m_initialized)
        {
            m_initial_width = width;
            m_initial_height = height;
            on_init();
            m_specific->m_initialized = true;
        }
        on_resize(m_rbuf_window.width(), m_rbuf_window.height());
        m_specific->m_update_flag = true;
        return true;
    }



    //------------------------------------------------------------------------
    void platform_support::update_window()
    {
        SDL_BlitSurface(m_specific->m_surf_window, 0, m_specific->m_surf_screen, 0);
        SDL_UpdateRect(m_specific->m_surf_screen, 0, 0, 0, 0);
    }


    //------------------------------------------------------------------------
    int platform_support::run()
    {
        SDL_Event event;
        bool ev_flag = false;

        for(;;)
        {
            if(m_specific->m_update_flag)
            {
                on_draw();
                update_window();
                m_specific->m_update_flag = false;
            }

            ev_flag = false;
            if(m_wait_mode)
            {
                SDL_WaitEvent(&event);
                ev_flag = true;
            }
            else
            {
                if(SDL_PollEvent(&event))
                {
                    ev_flag = true;
                }
                else
                {
                    on_idle();
                }
            }

            if(ev_flag)
            {
                if(event.type == SDL_QUIT)
                {
                    break;
                }

                int y;
                unsigned flags = 0;

                switch (event.type) 
                {
                case SDL_VIDEORESIZE:
                    if(!init(event.resize.w, event.resize.h, m_window_flags)) return false;
                    on_resize(m_rbuf_window.width(), m_rbuf_window.height());
                    trans_affine_resizing(event.resize.w, event.resize.h);
                    m_specific->m_update_flag = true;
                    break;

                case SDL_KEYDOWN:
                    {
                        flags = 0;
                        if(event.key.keysym.mod & KMOD_SHIFT) flags |= kbd_shift;
                        if(event.key.keysym.mod & KMOD_CTRL)  flags |= kbd_ctrl;

                        bool left  = false;
                        bool up    = false;
                        bool right = false;
                        bool down  = false;

                        switch(event.key.keysym.sym)
                        {
                        case key_left:
                            left = true;
                            break;

                        case key_up:
                            up = true;
                            break;

                        case key_right:
                            right = true;
                            break;

                        case key_down:
                            down = true;
                            break;
                        }

                        if(m_ctrls.on_arrow_keys(left, right, down, up))
                        {
                            on_ctrl_change();
                            force_redraw();
                        }
                        else
                        {
                            on_key(m_specific->m_cur_x,
                                   m_specific->m_cur_y,
                                   event.key.keysym.sym,
                                   flags);
                        }
                    }
                    break;

                case SDL_MOUSEMOTION:
                    y = m_flip_y ? 
                        m_rbuf_window.height() - event.motion.y : 
                        event.motion.y;

                    m_specific->m_cur_x = event.motion.x;
                    m_specific->m_cur_y = y;
                    flags = 0;
                    if(event.motion.state & SDL_BUTTON_LMASK) flags |= mouse_left;
                    if(event.motion.state & SDL_BUTTON_RMASK) flags |= mouse_right;

                    if(m_ctrls.on_mouse_move(m_specific->m_cur_x, 
                                             m_specific->m_cur_y,
                                             (flags & mouse_left) != 0))
                    {
                        on_ctrl_change();
                        force_redraw();
                    }
                    else
                    {
                        on_mouse_move(m_specific->m_cur_x, 
                                      m_specific->m_cur_y, 
                                      flags);
                    }
		    SDL_Event eventtrash;
		    while (SDL_PeepEvents(&eventtrash, 1, SDL_GETEVENT, SDL_EVENTMASK(SDL_MOUSEMOTION))!=0){;}
                    break;

		case SDL_MOUSEBUTTONDOWN:
                    y = m_flip_y
                        ? m_rbuf_window.height() - event.button.y
                        : event.button.y;

                    m_specific->m_cur_x = event.button.x;
                    m_specific->m_cur_y = y;
                    flags = 0;
                    switch(event.button.button)
                    {
                    case SDL_BUTTON_LEFT:
                        {
                            flags = mouse_left;

if(m_ctrls.on_mouse_button_down(m_specific->m_cur_x,
                                m_specific->m_cur_y))
                            {
                                m_ctrls.set_cur(m_specific->m_cur_x, 
                                    m_specific->m_cur_y);
                                on_ctrl_change();
                                force_redraw();
                            }
                            else
                            {
                                if(m_ctrls.in_rect(m_specific->m_cur_x, 
                                    m_specific->m_cur_y))
                                {
                                    if(m_ctrls.set_cur(m_specific->m_cur_x, 
                                        m_specific->m_cur_y))
                                    {
                                        on_ctrl_change();
                                        force_redraw();
                                    }
                                }
                                else
                                {
                                    on_mouse_button_down(m_specific->m_cur_x, 
                                        m_specific->m_cur_y, 
                                        flags);
                                }
                            }
                        }
                        break;
                    case SDL_BUTTON_RIGHT:
                        flags = mouse_right;
                        on_mouse_button_down(m_specific->m_cur_x, 
                            m_specific->m_cur_y, 
                            flags);
                        break;
                    } //switch(event.button.button)
                    break;
		    
                case SDL_MOUSEBUTTONUP:
                    y = m_flip_y
                        ? m_rbuf_window.height() - event.button.y
                        : event.button.y;

                    m_specific->m_cur_x = event.button.x;
                    m_specific->m_cur_y = y;
                    flags = 0;
                    if(m_ctrls.on_mouse_button_up(m_specific->m_cur_x, 
                                                  m_specific->m_cur_y))
                    {
                        on_ctrl_change();
                        force_redraw();
                    }
                    on_mouse_button_up(m_specific->m_cur_x, 
                                       m_specific->m_cur_y, 
                                       flags);
                    break;
                }
            }
        }
        return 0;
    }



    //------------------------------------------------------------------------
    const char* platform_support::img_ext() const { return ".bmp"; }

    //------------------------------------------------------------------------
    const char* platform_support::full_file_name(const char* file_name)
    {
        return file_name;
    }

    //------------------------------------------------------------------------
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        if(idx < max_images)
        {
            if(m_specific->m_surf_img[idx]) SDL_FreeSurface(m_specific->m_surf_img[idx]);

            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            if(len < 4 || strcmp(fn + len - 4, ".bmp") != 0)
            {
                strcat(fn, ".bmp");
            }

            SDL_Surface* tmp_surf = SDL_LoadBMP(fn);
            if (tmp_surf == 0) 
            {
                fprintf(stderr, "Couldn't load %s: %s\n", fn, SDL_GetError());
                return false;
            }

            SDL_PixelFormat format;
            format.palette = 0;
            format.BitsPerPixel = m_bpp;
            format.BytesPerPixel = m_bpp >> 8;
            format.Rmask = m_specific->m_rmask;
            format.Gmask = m_specific->m_gmask;
            format.Bmask = m_specific->m_bmask;
            format.Amask = m_specific->m_amask;
            format.Rshift = 0;
            format.Gshift = 0;
            format.Bshift = 0;
            format.Ashift = 0;
            format.Rloss = 0;
            format.Gloss = 0;
            format.Bloss = 0;
            format.Aloss = 0;
            format.colorkey = 0;
            format.alpha = 0;

            m_specific->m_surf_img[idx] = 
                SDL_ConvertSurface(tmp_surf, 
                                   &format, 
                                   SDL_SWSURFACE);

            SDL_FreeSurface(tmp_surf);
            
            if(m_specific->m_surf_img[idx] == 0) return false;

            m_rbuf_img[idx].attach((unsigned char*)m_specific->m_surf_img[idx]->pixels, 
                                   m_specific->m_surf_img[idx]->w, 
                                   m_specific->m_surf_img[idx]->h, 
                                   m_flip_y ? -m_specific->m_surf_img[idx]->pitch : 
                                               m_specific->m_surf_img[idx]->pitch);
            return true;

        }
        return false;
    }




    //------------------------------------------------------------------------
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        if(idx < max_images && m_specific->m_surf_img[idx])
        {
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            if(len < 4 || strcmp(fn + len - 4, ".bmp") != 0)
            {
                strcat(fn, ".bmp");
            }
            return SDL_SaveBMP(m_specific->m_surf_img[idx], fn) == 0;
        }
        return false;
    }



    //------------------------------------------------------------------------
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        if(idx < max_images)
        {

            if(m_specific->m_surf_img[idx]) SDL_FreeSurface(m_specific->m_surf_img[idx]);

             m_specific->m_surf_img[idx] = 
                 SDL_CreateRGBSurface(SDL_SWSURFACE, 
                                      width, 
                                      height,
                                      m_specific->m_surf_screen->format->BitsPerPixel,
                                      m_specific->m_rmask, 
                                      m_specific->m_gmask, 
                                      m_specific->m_bmask, 
                                      m_specific->m_amask);
            if(m_specific->m_surf_img[idx] == 0) 
            {
                fprintf(stderr, "Couldn't create image: %s\n", SDL_GetError());
                return false;
            }

            m_rbuf_img[idx].attach((unsigned char*)m_specific->m_surf_img[idx]->pixels, 
                                   m_specific->m_surf_img[idx]->w, 
                                   m_specific->m_surf_img[idx]->h, 
                                   m_flip_y ? -m_specific->m_surf_img[idx]->pitch : 
                                               m_specific->m_surf_img[idx]->pitch);

            return true;
        }

        return false;
    }
    
    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        m_specific->m_sw_start = SDL_GetTicks();
    }

    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        int stop = SDL_GetTicks();
        return double(stop - m_specific->m_sw_start);
    }

    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
        fprintf(stderr, "%s\n", msg);
    }

    //------------------------------------------------------------------------
    void platform_support::force_redraw()
    {
        m_specific->m_update_flag = true;
    }


    //------------------------------------------------------------------------
    void platform_support::on_init() {}
    void platform_support::on_resize(int sx, int sy) {}
    void platform_support::on_idle() {}
    void platform_support::on_mouse_move(int x, int y, unsigned flags) {}
    void platform_support::on_mouse_button_down(int x, int y, unsigned flags) {}
    void platform_support::on_mouse_button_up(int x, int y, unsigned flags) {}
    void platform_support::on_key(int x, int y, unsigned key, unsigned flags) {}
    void platform_support::on_ctrl_change() {}
    void platform_support::on_draw() {}
    void platform_support::on_post_draw(void* raw_handler) {}


}


int agg_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    return agg_main(argc, argv);
}


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
// class platform_support
//
//----------------------------------------------------------------------------

#include <windows.h>
#include <string.h>
#include "platform/agg_platform_support.h"
#include "platform/win32/agg_win32_bmp.h"
#include "util/agg_color_conv_rgb8.h"
#include "util/agg_color_conv_rgb16.h"


namespace agg
{
    
    //------------------------------------------------------------------------
    HINSTANCE g_windows_instance = 0;
    int       g_windows_cmd_show = 0;


    //------------------------------------------------------------------------
    class platform_specific
    {
    public:
        platform_specific(pix_format_e format, bool flip_y);

        void create_pmap(unsigned width, unsigned height, 
                         rendering_buffer* wnd);

        void display_pmap(HDC dc, const rendering_buffer* src);
        bool load_pmap(const char* fn, unsigned idx, 
                       rendering_buffer* dst);

        bool save_pmap(const char* fn, unsigned idx, 
                       const rendering_buffer* src);

        unsigned translate(unsigned keycode);

        pix_format_e  m_format;
        pix_format_e  m_sys_format;
        bool          m_flip_y;
        unsigned      m_bpp;
        unsigned      m_sys_bpp;
        HWND          m_hwnd;
        pixel_map     m_pmap_window;
        pixel_map     m_pmap_img[platform_support::max_images];
        unsigned      m_keymap[256];
        unsigned      m_last_translated_key;
        int           m_cur_x;
        int           m_cur_y;
        unsigned      m_input_flags;
        bool          m_redraw_flag;
        HDC           m_current_dc;
        LARGE_INTEGER m_sw_freq;
        LARGE_INTEGER m_sw_start;
    };


    //------------------------------------------------------------------------
    platform_specific::platform_specific(pix_format_e format, bool flip_y) :
        m_format(format),
        m_sys_format(pix_format_undefined),
        m_flip_y(flip_y),
        m_bpp(0),
        m_sys_bpp(0),
        m_hwnd(0),
        m_last_translated_key(0),
        m_cur_x(0),
        m_cur_y(0),
        m_input_flags(0),
        m_redraw_flag(true),
        m_current_dc(0)
    {
        memset(m_keymap, 0, sizeof(m_keymap));

        m_keymap[VK_PAUSE]      = key_pause;
        m_keymap[VK_CLEAR]      = key_clear;

        m_keymap[VK_NUMPAD0]    = key_kp0;
        m_keymap[VK_NUMPAD1]    = key_kp1;
        m_keymap[VK_NUMPAD2]    = key_kp2;
        m_keymap[VK_NUMPAD3]    = key_kp3;
        m_keymap[VK_NUMPAD4]    = key_kp4;
        m_keymap[VK_NUMPAD5]    = key_kp5;
        m_keymap[VK_NUMPAD6]    = key_kp6;
        m_keymap[VK_NUMPAD7]    = key_kp7;
        m_keymap[VK_NUMPAD8]    = key_kp8;
        m_keymap[VK_NUMPAD9]    = key_kp9;
        m_keymap[VK_DECIMAL]    = key_kp_period;
        m_keymap[VK_DIVIDE]     = key_kp_divide;
        m_keymap[VK_MULTIPLY]   = key_kp_multiply;
        m_keymap[VK_SUBTRACT]   = key_kp_minus;
        m_keymap[VK_ADD]        = key_kp_plus;

        m_keymap[VK_UP]         = key_up;
        m_keymap[VK_DOWN]       = key_down;
        m_keymap[VK_RIGHT]      = key_right;
        m_keymap[VK_LEFT]       = key_left;
        m_keymap[VK_INSERT]     = key_insert;
        m_keymap[VK_DELETE]     = key_delete;
        m_keymap[VK_HOME]       = key_home;
        m_keymap[VK_END]        = key_end;
        m_keymap[VK_PRIOR]      = key_page_up;
        m_keymap[VK_NEXT]       = key_page_down;

        m_keymap[VK_F1]         = key_f1;
        m_keymap[VK_F2]         = key_f2;
        m_keymap[VK_F3]         = key_f3;
        m_keymap[VK_F4]         = key_f4;
        m_keymap[VK_F5]         = key_f5;
        m_keymap[VK_F6]         = key_f6;
        m_keymap[VK_F7]         = key_f7;
        m_keymap[VK_F8]         = key_f8;
        m_keymap[VK_F9]         = key_f9;
        m_keymap[VK_F10]        = key_f10;
        m_keymap[VK_F11]        = key_f11;
        m_keymap[VK_F12]        = key_f12;
        m_keymap[VK_F13]        = key_f13;
        m_keymap[VK_F14]        = key_f14;
        m_keymap[VK_F15]        = key_f15;

        m_keymap[VK_NUMLOCK]    = key_numlock;
        m_keymap[VK_CAPITAL]    = key_capslock;
        m_keymap[VK_SCROLL]     = key_scrollock;


        switch(m_format)
        {
        case pix_format_bw:
            m_sys_format = pix_format_bw;
            m_bpp = 1;
            m_sys_bpp = 1;
            break;

        case pix_format_gray8:
            m_sys_format = pix_format_gray8;
            m_bpp = 8;
            m_sys_bpp = 8;
            break;

        case pix_format_gray16:
            m_sys_format = pix_format_gray8;
            m_bpp = 16;
            m_sys_bpp = 8;
            break;

        case pix_format_rgb565:
        case pix_format_rgb555:
            m_sys_format = pix_format_rgb555;
            m_bpp = 16;
            m_sys_bpp = 16;
            break;

        case pix_format_rgbAAA:
        case pix_format_bgrAAA:
        case pix_format_rgbBBA:
        case pix_format_bgrABB:
            m_sys_format = pix_format_bgr24;
            m_bpp = 32;
            m_sys_bpp = 24;
            break;

        case pix_format_rgb24:
        case pix_format_bgr24:
            m_sys_format = pix_format_bgr24;
            m_bpp = 24;
            m_sys_bpp = 24;
            break;

        case pix_format_rgb48:
        case pix_format_bgr48:
            m_sys_format = pix_format_bgr24;
            m_bpp = 48;
            m_sys_bpp = 24;
            break;

        case pix_format_bgra32:
        case pix_format_abgr32:
        case pix_format_argb32:
        case pix_format_rgba32:
            m_sys_format = pix_format_bgra32;
            m_bpp = 32;
            m_sys_bpp = 32;
            break;

        case pix_format_bgra64:
        case pix_format_abgr64:
        case pix_format_argb64:
        case pix_format_rgba64:
            m_sys_format = pix_format_bgra32;
            m_bpp = 64;
            m_sys_bpp = 32;
            break;
        }
        ::QueryPerformanceFrequency(&m_sw_freq);
        ::QueryPerformanceCounter(&m_sw_start);
    }


    //------------------------------------------------------------------------
    void platform_specific::create_pmap(unsigned width, 
                                        unsigned height,
                                        rendering_buffer* wnd)
    {
        m_pmap_window.create(width, height, org_e(m_bpp));
        wnd->attach(m_pmap_window.buf(), 
                    m_pmap_window.width(),
                    m_pmap_window.height(),
                      m_flip_y ?
                      m_pmap_window.stride() :
                     -m_pmap_window.stride());
    }


    //------------------------------------------------------------------------
    static void convert_pmap(rendering_buffer* dst, 
                             const rendering_buffer* src, 
                             pix_format_e format)
    {
        switch(format)
        {
        case pix_format_gray8:
            break;

        case pix_format_gray16:
            color_conv(dst, src, color_conv_gray16_to_gray8());
            break;

        case pix_format_rgb565:
            color_conv(dst, src, color_conv_rgb565_to_rgb555());
            break;

        case pix_format_rgbAAA:
            color_conv(dst, src, color_conv_rgbAAA_to_bgr24());
            break;

        case pix_format_bgrAAA:
            color_conv(dst, src, color_conv_bgrAAA_to_bgr24());
            break;

        case pix_format_rgbBBA:
            color_conv(dst, src, color_conv_rgbBBA_to_bgr24());
            break;

        case pix_format_bgrABB:
            color_conv(dst, src, color_conv_bgrABB_to_bgr24());
            break;

        case pix_format_rgb24:
            color_conv(dst, src, color_conv_rgb24_to_bgr24());
            break;

        case pix_format_rgb48:
            color_conv(dst, src, color_conv_rgb48_to_bgr24());
            break;

        case pix_format_bgr48:
            color_conv(dst, src, color_conv_bgr48_to_bgr24());
            break;

        case pix_format_abgr32:
            color_conv(dst, src, color_conv_abgr32_to_bgra32());
            break;

        case pix_format_argb32:
            color_conv(dst, src, color_conv_argb32_to_bgra32());
            break;

        case pix_format_rgba32:
            color_conv(dst, src, color_conv_rgba32_to_bgra32());
            break;

        case pix_format_bgra64:
            color_conv(dst, src, color_conv_bgra64_to_bgra32());
            break;

        case pix_format_abgr64:
            color_conv(dst, src, color_conv_abgr64_to_bgra32());
            break;

        case pix_format_argb64:
            color_conv(dst, src, color_conv_argb64_to_bgra32());
            break;

        case pix_format_rgba64:
            color_conv(dst, src, color_conv_rgba64_to_bgra32());
            break;
        }
    }


    //------------------------------------------------------------------------
    void platform_specific::display_pmap(HDC dc, const rendering_buffer* src)
    {
        if(m_sys_format == m_format)
        {
            m_pmap_window.draw(dc);
        }
        else
        {
            pixel_map pmap_tmp;
            pmap_tmp.create(m_pmap_window.width(), 
                            m_pmap_window.height(),
                            org_e(m_sys_bpp));

            rendering_buffer rbuf_tmp;
            rbuf_tmp.attach(pmap_tmp.buf(),
                            pmap_tmp.width(),
                            pmap_tmp.height(),
                            m_flip_y ?
                              pmap_tmp.stride() :
                             -pmap_tmp.stride());

            convert_pmap(&rbuf_tmp, src, m_format);
            pmap_tmp.draw(dc);
        }
    }



    //------------------------------------------------------------------------
    bool platform_specific::save_pmap(const char* fn, unsigned idx, 
                                      const rendering_buffer* src)
    {
        if(m_sys_format == m_format)
        {
            return m_pmap_img[idx].save_as_bmp(fn);
        }

        pixel_map pmap_tmp;
        pmap_tmp.create(m_pmap_img[idx].width(), 
                          m_pmap_img[idx].height(),
                          org_e(m_sys_bpp));

        rendering_buffer rbuf_tmp;
        rbuf_tmp.attach(pmap_tmp.buf(),
                          pmap_tmp.width(),
                          pmap_tmp.height(),
                          m_flip_y ?
                          pmap_tmp.stride() :
                          -pmap_tmp.stride());

        convert_pmap(&rbuf_tmp, src, m_format);
        return pmap_tmp.save_as_bmp(fn);
    }



    //------------------------------------------------------------------------
    bool platform_specific::load_pmap(const char* fn, unsigned idx, 
                                      rendering_buffer* dst)
    {
        pixel_map pmap_tmp;
        if(!pmap_tmp.load_from_bmp(fn)) return false;

        rendering_buffer rbuf_tmp;
        rbuf_tmp.attach(pmap_tmp.buf(),
                        pmap_tmp.width(),
                        pmap_tmp.height(),
                        m_flip_y ?
                          pmap_tmp.stride() :
                         -pmap_tmp.stride());

        m_pmap_img[idx].create(pmap_tmp.width(), 
                               pmap_tmp.height(), 
                               org_e(m_bpp),
                               0);

        dst->attach(m_pmap_img[idx].buf(),
                    m_pmap_img[idx].width(),
                    m_pmap_img[idx].height(),
                    m_flip_y ?
                       m_pmap_img[idx].stride() :
                      -m_pmap_img[idx].stride());

        switch(m_format)
        {
        case pix_format_gray8:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_gray8()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_gray8()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_gray8()); break;
            }
            break;

        case pix_format_gray16:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_gray16()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_gray16()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_gray16()); break;
            }
            break;

        case pix_format_rgb555:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb555()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_rgb555()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb555()); break;
            }
            break;

        case pix_format_rgb565:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb565()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_rgb565()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb565()); break;
            }
            break;

        case pix_format_rgb24:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb24()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_rgb24()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb24()); break;
            }
            break;

        case pix_format_bgr24:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_bgr24()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_bgr24()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgr24()); break;
            }
            break;

        case pix_format_rgb48:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb48()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_rgb48()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb48()); break;
            }
            break;

        case pix_format_bgr48:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_bgr48()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_bgr48()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgr48()); break;
            }
            break;

        case pix_format_abgr32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_abgr32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_abgr32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_abgr32()); break;
            }
            break;

        case pix_format_argb32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_argb32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_argb32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_argb32()); break;
            }
            break;

        case pix_format_bgra32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_bgra32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_bgra32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgra32()); break;
            }
            break;

        case pix_format_rgba32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgba32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_rgba32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgba32()); break;
            }
            break;

        case pix_format_abgr64:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_abgr64()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_abgr64()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_abgr64()); break;
            }
            break;

        case pix_format_argb64:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_argb64()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_argb64()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_argb64()); break;
            }
            break;

        case pix_format_bgra64:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_bgra64()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_bgra64()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgra64()); break;
            }
            break;

        case pix_format_rgba64:
            switch(pmap_tmp.bpp())
            {
            //case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgba64()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_bgr24_to_rgba64()); break;
            //case 32: color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgba64()); break;
            }
            break;

        }

        return true;
    }








    //------------------------------------------------------------------------
    unsigned platform_specific::translate(unsigned keycode)
    {
        return m_last_translated_key = (keycode > 255) ? 0 : m_keymap[keycode];
    }



    //------------------------------------------------------------------------
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        m_specific(new platform_specific(format, flip_y)),
        m_format(format),
        m_bpp(m_specific->m_bpp),
        m_window_flags(0),
        m_wait_mode(true),
        m_flip_y(flip_y),
        m_initial_width(10),
        m_initial_height(10)
    {
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
        if(m_specific->m_hwnd)
        {
            SetWindowText(m_specific->m_hwnd, m_caption);
        }
    }

    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        ::QueryPerformanceCounter(&(m_specific->m_sw_start));
    }

    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        LARGE_INTEGER stop;
        ::QueryPerformanceCounter(&stop);
        return double(stop.QuadPart - 
                      m_specific->m_sw_start.QuadPart) * 1000.0 / 
                      double(m_specific->m_sw_freq.QuadPart);
    }



    //------------------------------------------------------------------------
    static unsigned get_key_flags(int wflags)
    {
        unsigned flags = 0;
        if(wflags & MK_LBUTTON) flags |= mouse_left;
        if(wflags & MK_RBUTTON) flags |= mouse_right;
        if(wflags & MK_SHIFT)   flags |= kbd_shift;
        if(wflags & MK_CONTROL) flags |= kbd_ctrl;
        return flags;
    }


    void* platform_support::raw_display_handler()
    {
        return m_specific->m_current_dc;
    }


    //------------------------------------------------------------------------
    LRESULT CALLBACK window_proc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        PAINTSTRUCT ps;
        HDC paintDC;


        void* user_data = reinterpret_cast<void*>(::GetWindowLong(hWnd, GWL_USERDATA));
        platform_support* app = 0;

        if(user_data)
        {
            app = reinterpret_cast<platform_support*>(user_data);
        }

        if(app == 0)
        {
            if(msg == WM_DESTROY)
            {
                ::PostQuitMessage(0);
                return 0;
            }
            return ::DefWindowProc(hWnd, msg, wParam, lParam);
        }

        HDC dc = ::GetDC(app->m_specific->m_hwnd);
        app->m_specific->m_current_dc = dc;
        LRESULT ret = 0;

        switch(msg) 
        {
        //--------------------------------------------------------------------
        case WM_CREATE:
            break;
        
        //--------------------------------------------------------------------
        case WM_SIZE:
            app->m_specific->create_pmap(LOWORD(lParam), 
                                         HIWORD(lParam),
                                         &app->rbuf_window());

            app->trans_affine_resizing(LOWORD(lParam), HIWORD(lParam));
            app->on_resize(LOWORD(lParam), HIWORD(lParam));
            app->force_redraw();
            break;
        
        //--------------------------------------------------------------------
        case WM_ERASEBKGND:
            break;
        
        //--------------------------------------------------------------------
        case WM_LBUTTONDOWN:
            ::SetCapture(app->m_specific->m_hwnd);
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            if(app->flip_y())
            {
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            app->m_specific->m_input_flags = mouse_left | get_key_flags(wParam);
            
            app->m_ctrls.set_cur(app->m_specific->m_cur_x, 
                                 app->m_specific->m_cur_y);
            if(app->m_ctrls.on_mouse_button_down(app->m_specific->m_cur_x, 
                                                 app->m_specific->m_cur_y))
            {
                app->on_ctrl_change();
                app->force_redraw();
            }
            else
            {
                if(app->m_ctrls.in_rect(app->m_specific->m_cur_x, 
                                        app->m_specific->m_cur_y))
                {
                    if(app->m_ctrls.set_cur(app->m_specific->m_cur_x, 
                                            app->m_specific->m_cur_y))
                    {
                        app->on_ctrl_change();
                        app->force_redraw();
                    }
                }
                else
                {
                    app->on_mouse_button_down(app->m_specific->m_cur_x, 
                                              app->m_specific->m_cur_y, 
                                              app->m_specific->m_input_flags);
                }
            }
/*
            if(!app->wait_mode())
            {
                app->on_idle();
            }
*/
            break;

        //--------------------------------------------------------------------
        case WM_LBUTTONUP:
            ::ReleaseCapture();
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            if(app->flip_y())
            {
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            app->m_specific->m_input_flags = mouse_left | get_key_flags(wParam);

            if(app->m_ctrls.on_mouse_button_up(app->m_specific->m_cur_x, 
                                               app->m_specific->m_cur_y))
            {
                app->on_ctrl_change();
                app->force_redraw();
            }
            app->on_mouse_button_up(app->m_specific->m_cur_x, 
                                    app->m_specific->m_cur_y, 
                                    app->m_specific->m_input_flags);
/*
            if(!app->wait_mode())
            {
                app->on_idle();
            }
*/
            break;


        //--------------------------------------------------------------------
        case WM_RBUTTONDOWN:
            ::SetCapture(app->m_specific->m_hwnd);
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            if(app->flip_y())
            {
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            app->m_specific->m_input_flags = mouse_right | get_key_flags(wParam);
            app->on_mouse_button_down(app->m_specific->m_cur_x, 
                                      app->m_specific->m_cur_y, 
                                      app->m_specific->m_input_flags);
/*
            if(!app->wait_mode())
            {
                app->on_idle();
            }
*/
            break;

        //--------------------------------------------------------------------
        case WM_RBUTTONUP:
            ::ReleaseCapture();
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            if(app->flip_y())
            {
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            app->m_specific->m_input_flags = mouse_right | get_key_flags(wParam);
            app->on_mouse_button_up(app->m_specific->m_cur_x, 
                                    app->m_specific->m_cur_y, 
                                    app->m_specific->m_input_flags);
/*
            if(!app->wait_mode())
            {
                app->on_idle();
            }
*/
            break;

        //--------------------------------------------------------------------
        case WM_MOUSEMOVE:
            app->m_specific->m_cur_x = int16(LOWORD(lParam));
            if(app->flip_y())
            {
                app->m_specific->m_cur_y = app->rbuf_window().height() - int16(HIWORD(lParam));
            }
            else
            {
                app->m_specific->m_cur_y = int16(HIWORD(lParam));
            }
            app->m_specific->m_input_flags = get_key_flags(wParam);


            if(app->m_ctrls.on_mouse_move(
                app->m_specific->m_cur_x, 
                app->m_specific->m_cur_y,
                (app->m_specific->m_input_flags & mouse_left) != 0))
            {
                app->on_ctrl_change();
                app->force_redraw();
            }
            else
            {
                if(!app->m_ctrls.in_rect(app->m_specific->m_cur_x, 
                                         app->m_specific->m_cur_y))
                {
                    app->on_mouse_move(app->m_specific->m_cur_x, 
                                       app->m_specific->m_cur_y, 
                                       app->m_specific->m_input_flags);
                }
            }
/*
            if(!app->wait_mode())
            {
                app->on_idle();
            }
*/
            break;

        //--------------------------------------------------------------------
        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            app->m_specific->m_last_translated_key = 0;
            switch(wParam) 
            {
                case VK_CONTROL:
                    app->m_specific->m_input_flags |= kbd_ctrl;
                    break;

                case VK_SHIFT:
                    app->m_specific->m_input_flags |= kbd_shift;
                    break;

                default:
                    app->m_specific->translate(wParam);
                    break;
            }
        
            if(app->m_specific->m_last_translated_key)
            {
                bool left  = false;
                bool up    = false;
                bool right = false;
                bool down  = false;

                switch(app->m_specific->m_last_translated_key)
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

                case key_f2:                        
                    app->copy_window_to_img(agg::platform_support::max_images - 1);
                    app->save_img(agg::platform_support::max_images - 1, "screenshot");
                    break;
                }

                if(app->window_flags() & window_process_all_keys)
                {
                    app->on_key(app->m_specific->m_cur_x,
                                app->m_specific->m_cur_y,
                                app->m_specific->m_last_translated_key,
                                app->m_specific->m_input_flags);
                }
                else
                {
                    if(app->m_ctrls.on_arrow_keys(left, right, down, up))
                    {
                        app->on_ctrl_change();
                        app->force_redraw();
                    }
                    else
                    {
                        app->on_key(app->m_specific->m_cur_x,
                                    app->m_specific->m_cur_y,
                                    app->m_specific->m_last_translated_key,
                                    app->m_specific->m_input_flags);
                    }
                }
            }
/*
            if(!app->wait_mode())
            {
                app->on_idle();
            }
*/
            break;

        //--------------------------------------------------------------------
        case WM_SYSKEYUP:
        case WM_KEYUP:
            app->m_specific->m_last_translated_key = 0;
            switch(wParam) 
            {
                case VK_CONTROL:
                    app->m_specific->m_input_flags &= ~kbd_ctrl;
                    break;

                case VK_SHIFT:
                    app->m_specific->m_input_flags &= ~kbd_shift;
                    break;
            }
            break;

        //--------------------------------------------------------------------
        case WM_CHAR:
        case WM_SYSCHAR:
            if(app->m_specific->m_last_translated_key == 0)
            {
                app->on_key(app->m_specific->m_cur_x,
                            app->m_specific->m_cur_y,
                            wParam,
                            app->m_specific->m_input_flags);
            }
            break;
        
        //--------------------------------------------------------------------
        case WM_PAINT:
            paintDC = ::BeginPaint(hWnd, &ps);
            app->m_specific->m_current_dc = paintDC;
            if(app->m_specific->m_redraw_flag)
            {
                app->on_draw();
                app->m_specific->m_redraw_flag = false;
            }
            app->m_specific->display_pmap(paintDC, &app->rbuf_window());
            app->on_post_draw(paintDC);
            app->m_specific->m_current_dc = 0;
            ::EndPaint(hWnd, &ps);
            break;
        
        //--------------------------------------------------------------------
        case WM_COMMAND:
            break;
        
        //--------------------------------------------------------------------
        case WM_DESTROY:
            ::PostQuitMessage(0);
            break;
        
        //--------------------------------------------------------------------
        default:
            ret = ::DefWindowProc(hWnd, msg, wParam, lParam);
            break;
        }
        app->m_specific->m_current_dc = 0;
        ::ReleaseDC(app->m_specific->m_hwnd, dc);
        return ret;
    }


    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
        ::MessageBox(m_specific->m_hwnd, msg, "AGG Message", MB_OK);
    }


    //------------------------------------------------------------------------
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        if(m_specific->m_sys_format == pix_format_undefined)
        {
            return false;
        }

        m_window_flags = flags;

        int wflags = CS_OWNDC | CS_VREDRAW | CS_HREDRAW;

        WNDCLASS wc;
        wc.lpszClassName = "AGGAppClass";
        wc.lpfnWndProc = window_proc;
        wc.style = wflags;
        wc.hInstance = g_windows_instance;
        wc.hIcon = LoadIcon(0, IDI_APPLICATION);
        wc.hCursor = LoadCursor(0, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
        wc.lpszMenuName = "AGGAppMenu";
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        ::RegisterClass(&wc);

        wflags = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;

        if(m_window_flags & window_resize)
        {
            wflags |= WS_THICKFRAME | WS_MAXIMIZEBOX;
        }

        m_specific->m_hwnd = ::CreateWindow("AGGAppClass",
                                            m_caption,
                                            wflags,
                                            100,
                                            100,
                                            width,
                                            height,
                                            0,
                                            0,
                                            g_windows_instance,
                                            0);

        if(m_specific->m_hwnd == 0)
        {
            return false;
        }


        RECT rct;
        ::GetClientRect(m_specific->m_hwnd, &rct);

        ::MoveWindow(m_specific->m_hwnd,   // handle to window
                     100,                  // horizontal position
                     100,                  // vertical position
                     width + (width - (rct.right - rct.left)),
                     height + (height - (rct.bottom - rct.top)),
                     FALSE);
   
        ::SetWindowLong(m_specific->m_hwnd, GWL_USERDATA, (LONG)this);
        m_specific->create_pmap(width, height, &m_rbuf_window);
        m_initial_width = width;
        m_initial_height = height;
        on_init();
        m_specific->m_redraw_flag = true;
        ::ShowWindow(m_specific->m_hwnd, g_windows_cmd_show);
        return true;
    }



    //------------------------------------------------------------------------
    int platform_support::run()
    {
        MSG msg;

        for(;;)
        {
            if(m_wait_mode)
            {
                if(!::GetMessage(&msg, 0, 0, 0))
                {
                    break;
                }
                ::TranslateMessage(&msg);
                ::DispatchMessage(&msg);
            }
            else
            {
                if(::PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
                {
                    ::TranslateMessage(&msg);
                    if(msg.message == WM_QUIT)
                    {
                        break;
                    }
                    ::DispatchMessage(&msg);
                }
                else
                {
                    on_idle();
                }
            }
        }
        return (int)msg.wParam;
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
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            if(len < 4 || stricmp(fn + len - 4, ".BMP") != 0)
            {
                strcat(fn, ".bmp");
            }
            return m_specific->load_pmap(fn, idx, &m_rbuf_img[idx]);
        }
        return true;
    }



    //------------------------------------------------------------------------
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        if(idx < max_images)
        {
            char fn[1024];
            strcpy(fn, file);
            int len = strlen(fn);
            if(len < 4 || stricmp(fn + len - 4, ".BMP") != 0)
            {
                strcat(fn, ".bmp");
            }
            return m_specific->save_pmap(fn, idx, &m_rbuf_img[idx]);
        }
        return true;
    }



    //------------------------------------------------------------------------
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        if(idx < max_images)
        {
            if(width  == 0) width  = m_specific->m_pmap_window.width();
            if(height == 0) height = m_specific->m_pmap_window.height();
            m_specific->m_pmap_img[idx].create(width, height, org_e(m_specific->m_bpp));
            m_rbuf_img[idx].attach(m_specific->m_pmap_img[idx].buf(), 
                                   m_specific->m_pmap_img[idx].width(),
                                   m_specific->m_pmap_img[idx].height(),
                                   m_flip_y ?
                                    m_specific->m_pmap_img[idx].stride() :
                                   -m_specific->m_pmap_img[idx].stride());
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    void platform_support::force_redraw()
    {
        m_specific->m_redraw_flag = true;
        ::InvalidateRect(m_specific->m_hwnd, 0, FALSE);
    }



    //------------------------------------------------------------------------
    void platform_support::update_window()
    {
        HDC dc = ::GetDC(m_specific->m_hwnd);
        m_specific->display_pmap(dc, &m_rbuf_window);
        ::ReleaseDC(m_specific->m_hwnd, dc);
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




namespace agg
{
    // That's ridiculous. I have to parse the command line by myself
    // because Windows doesn't provide a method of getting the command
    // line arguments in a form of argc, argv. Of course, there's 
    // CommandLineToArgv() but first, it returns Unicode that I don't
    // need to deal with, but most of all, it's not compatible with Win98.
    //-----------------------------------------------------------------------
    class tokenizer
    {
    public:
        enum sep_flag
        {
            single,
            multiple,
            whole_str
        };

        struct token
        {
            const char* ptr;
            unsigned    len;
        };

    public:
        tokenizer(const char* sep, 
                  const char* trim=0,
                  const char* quote="\"",
                  char mask_chr='\\',
                  sep_flag sf=multiple);

        void  set_str(const char* str);
        token next_token();

    private:
        int  check_chr(const char *str, char chr);

    private:
        const char* m_src_string;
        int         m_start;
        const char* m_sep;
        const char* m_trim;
        const char* m_quote;
        char        m_mask_chr;
        unsigned    m_sep_len;
        sep_flag    m_sep_flag;
    };



    //-----------------------------------------------------------------------
    inline void tokenizer::set_str(const char* str) 
    { 
        m_src_string = str; 
        m_start = 0;
    }


    //-----------------------------------------------------------------------
    inline int tokenizer::check_chr(const char *str, char chr)
    {
        return int(strchr(str, chr));
    }


    //-----------------------------------------------------------------------
    tokenizer::tokenizer(const char* sep, 
                         const char* trim,
                         const char* quote,
                         char mask_chr,
                         sep_flag sf) :
        m_src_string(0),
        m_start(0),
        m_sep(sep),
        m_trim(trim),
        m_quote(quote),
        m_mask_chr(mask_chr),
        m_sep_len(sep ? strlen(sep) : 0),
        m_sep_flag(sep ? sf : single)
    {
    }


    //-----------------------------------------------------------------------
    tokenizer::token tokenizer::next_token()
    {
        unsigned count = 0;
        char quote_chr = 0;
        token tok;

        tok.ptr = 0;
        tok.len = 0;
        if(m_src_string == 0 || m_start == -1) return tok;

        register const char *pstr = m_src_string + m_start;

        if(*pstr == 0) 
        {
            m_start = -1;
            return tok;
        }

        int sep_len = 1;
        if(m_sep_flag == whole_str) sep_len = m_sep_len;

        if(m_sep_flag == multiple)
        {
            //Pass all the separator symbols at the begin of the string
            while(*pstr && check_chr(m_sep, *pstr)) 
            {
                ++pstr;
                ++m_start;
            }
        }

        if(*pstr == 0) 
        {
            m_start = -1;
            return tok;
        }

        for(count = 0;; ++count) 
        {
            char c = *pstr;
            int found = 0;

            //We are outside of qotation: find one of separator symbols
            if(quote_chr == 0)
            {
                if(sep_len == 1)
                {
                    found = check_chr(m_sep, c);
                }
                else
                {
                    found = strncmp(m_sep, pstr, m_sep_len) == 0; 
                }
            }

            ++pstr;

            if(c == 0 || found) 
            {
                if(m_trim)
                {
                    while(count && 
                          check_chr(m_trim, m_src_string[m_start]))
                    {
                        ++m_start;
                        --count;
                    }

                    while(count && 
                          check_chr(m_trim, m_src_string[m_start + count - 1]))
                    {
                        --count;
                    }
                }

                tok.ptr = m_src_string + m_start;
                tok.len = count;

                //Next time it will be the next separator character
                //But we must check, whether it is NOT the end of the string.
                m_start += count;
                if(c) 
                {
                    m_start += sep_len;
                    if(m_sep_flag == multiple)
                    {
                        //Pass all the separator symbols 
                        //after the end of the string
                        while(check_chr(m_sep, m_src_string[m_start])) 
                        {
                            ++m_start;
                        }
                    }
                }
                break;
            }

            //Switch quote. If it is not a quote yet, try to check any of
            //quote symbols. Otherwise quote must be finished with quote_symb
            if(quote_chr == 0)
            {
                if(check_chr(m_quote, c)) 
                {
                    quote_chr = c;
                    continue;
                }
            }
            else
            {
                //We are inside quote: pass all the mask symbols
                if(m_mask_chr && c == m_mask_chr)
                {
                    if(*pstr) 
                    {
                        ++count;
                        ++pstr;
                    }
                    continue; 
                }
                if(c == quote_chr) 
                {
                    quote_chr = 0;
                    continue;
                }
            }
        }
        return tok;
    }


}



//----------------------------------------------------------------------------
int agg_main(int argc, char* argv[]);



//----------------------------------------------------------------------------
int PASCAL WinMain(HINSTANCE hInstance,
                   HINSTANCE hPrevInstance,
                   LPSTR lpszCmdLine,
                   int nCmdShow)
{
    agg::g_windows_instance = hInstance;
    agg::g_windows_cmd_show = nCmdShow;

    char* argv_str = new char [strlen(lpszCmdLine) + 3];
    char* argv_ptr = argv_str;

    char* argv[64];
    memset(argv, 0, sizeof(argv));

    agg::tokenizer cmd_line(" ", "\"' ", "\"'", '\\', agg::tokenizer::multiple);
    cmd_line.set_str(lpszCmdLine);

    int argc = 0;
    argv[argc++] = argv_ptr;
    *argv_ptr++ = 0;

    while(argc < 64)
    {
        agg::tokenizer::token tok = cmd_line.next_token();
        if(tok.ptr == 0) break;
        if(tok.len)
        {
            memcpy(argv_ptr, tok.ptr, tok.len);
            argv[argc++] = argv_ptr;
            argv_ptr += tok.len;
            *argv_ptr++ = 0;
        }
    }

    int ret = agg_main(argc, argv);
    delete [] argv_str;

    return ret;
}





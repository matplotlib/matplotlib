//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4 
// Copyright (C) 2002-2005 Maxim Shemanarev (McSeem)
// Copyright (C) 2003 Hansruedi Baer (MacOS support)
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
//			baer@karto.baug.eth.ch
//----------------------------------------------------------------------------
//
// class platform_support
//
//----------------------------------------------------------------------------
//
// Note:
// I tried to retain the original structure for the Win32 platform as far
// as possible. Currently, not all features are implemented but the examples
// should work properly.
// HB
//----------------------------------------------------------------------------

#include <Carbon.h>
#if defined(__MWERKS__)
#include "console.h"
#endif
#include <string.h>
#include <unistd.h>
#include "platform/agg_platform_support.h"
#include "platform/mac/agg_mac_pmap.h"
#include "util/agg_color_conv_rgb8.h"


namespace agg
{
    
pascal OSStatus DoWindowClose (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoWindowDrawContent (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoAppQuit (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoMouseDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoMouseUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoMouseDragged (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoKeyDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal OSStatus DoKeyUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData);
pascal void DoPeriodicTask (EventLoopTimerRef theTimer, void* userData);


    //------------------------------------------------------------------------
    class platform_specific
    {
    public:
        platform_specific(pix_format_e format, bool flip_y);

        void create_pmap(unsigned width, unsigned height, 
                         rendering_buffer* wnd);

        void display_pmap(WindowRef window, const rendering_buffer* src);
        bool load_pmap(const char* fn, unsigned idx, 
                       rendering_buffer* dst);

        bool save_pmap(const char* fn, unsigned idx, 
                       const rendering_buffer* src);

        unsigned translate(unsigned keycode);

        pix_format_e     m_format;
        pix_format_e     m_sys_format;
        bool             m_flip_y;
        unsigned         m_bpp;
        unsigned         m_sys_bpp;
        WindowRef        m_window;
        pixel_map   	 m_pmap_window;
        pixel_map        m_pmap_img[platform_support::max_images];
        unsigned         m_keymap[256];
        unsigned         m_last_translated_key;
        int              m_cur_x;
        int              m_cur_y;
        unsigned         m_input_flags;
        bool             m_redraw_flag;
        UnsignedWide     m_sw_freq;
        UnsignedWide     m_sw_start;
    };


    //------------------------------------------------------------------------
    platform_specific::platform_specific(pix_format_e format, bool flip_y) :
       	m_format(format),
        m_sys_format(pix_format_undefined),
        m_flip_y(flip_y),
        m_bpp(0),
        m_sys_bpp(0),
        m_window(nil),
        m_last_translated_key(0),
        m_cur_x(0),
        m_cur_y(0),
        m_input_flags(0),
        m_redraw_flag(true)
    {
        memset(m_keymap, 0, sizeof(m_keymap));

        //Keyboard input is not yet fully supported nor tested
        //m_keymap[VK_PAUSE]       = key_pause;
        m_keymap[kClearCharCode]      = key_clear;

        //m_keymap[VK_NUMPAD0]    = key_kp0;
        //m_keymap[VK_NUMPAD1]    = key_kp1;
        //m_keymap[VK_NUMPAD2]    = key_kp2;
        //m_keymap[VK_NUMPAD3]    = key_kp3;
        //m_keymap[VK_NUMPAD4]    = key_kp4;
        //m_keymap[VK_NUMPAD5]    = key_kp5;
        //m_keymap[VK_NUMPAD6]    = key_kp6;
        //m_keymap[VK_NUMPAD7]    = key_kp7;
        //m_keymap[VK_NUMPAD8]    = key_kp8;
        //m_keymap[VK_NUMPAD9]    = key_kp9;
        //m_keymap[VK_DECIMAL]    = key_kp_period;
        //m_keymap[VK_DIVIDE]     = key_kp_divide;
        //m_keymap[VK_MULTIPLY]   = key_kp_multiply;
        //m_keymap[VK_SUBTRACT]   = key_kp_minus;
        //m_keymap[VK_ADD]        = key_kp_plus;

        m_keymap[kUpArrowCharCode]    = key_up;
        m_keymap[kDownArrowCharCode]  = key_down;
        m_keymap[kRightArrowCharCode] = key_right;
        m_keymap[kLeftArrowCharCode]  = key_left;
        //m_keymap[VK_INSERT]     = key_insert;
        m_keymap[kDeleteCharCode]     = key_delete;
        m_keymap[kHomeCharCode]       = key_home;
        m_keymap[kEndCharCode]        = key_end;
        m_keymap[kPageUpCharCode]     = key_page_up;
        m_keymap[kPageDownCharCode]   = key_page_down;

        //m_keymap[VK_F1]         = key_f1;
        //m_keymap[VK_F2]         = key_f2;
        //m_keymap[VK_F3]         = key_f3;
        //m_keymap[VK_F4]         = key_f4;
        //m_keymap[VK_F5]         = key_f5;
        //m_keymap[VK_F6]         = key_f6;
        //m_keymap[VK_F7]         = key_f7;
        //m_keymap[VK_F8]         = key_f8;
        //m_keymap[VK_F9]         = key_f9;
        //m_keymap[VK_F10]        = key_f10;
        //m_keymap[VK_F11]        = key_f11;
        //m_keymap[VK_F12]        = key_f12;
        //m_keymap[VK_F13]        = key_f13;
        //m_keymap[VK_F14]        = key_f14;
        //m_keymap[VK_F15]        = key_f15;

        //m_keymap[VK_NUMLOCK]    = key_numlock;
        //m_keymap[VK_CAPITAL]    = key_capslock;
        //m_keymap[VK_SCROLL]     = key_scrollock;

        switch(m_format)
        {
        case pix_format_gray8:
            m_sys_format = pix_format_gray8;
            m_bpp = 8;
            m_sys_bpp = 8;
            break;

        case pix_format_rgb565:
        case pix_format_rgb555:
            m_sys_format = pix_format_rgb555;
            m_bpp = 16;
            m_sys_bpp = 16;
            break;

        case pix_format_rgb24:
        case pix_format_bgr24:
            m_sys_format = pix_format_rgb24;
            m_bpp = 24;
            m_sys_bpp = 24;
            break;

        case pix_format_bgra32:
        case pix_format_abgr32:
        case pix_format_argb32:
        case pix_format_rgba32:
            m_sys_format = pix_format_argb32;
            m_bpp = 32;
            m_sys_bpp = 32;
            break;
        }
        ::Microseconds(&m_sw_freq);
        ::Microseconds(&m_sw_start);
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
                     -m_pmap_window.row_bytes() :
                      m_pmap_window.row_bytes());
    }


    //------------------------------------------------------------------------
    void platform_specific::display_pmap(WindowRef window, const rendering_buffer* src)
    {
        if(m_sys_format == m_format)
        {
            m_pmap_window.draw(window);
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
                             -pmap_tmp.row_bytes() :
                              pmap_tmp.row_bytes());

            switch(m_format)
            {
            case pix_format_gray8:
                return;

            case pix_format_rgb565:
                color_conv(&rbuf_tmp, src, color_conv_rgb565_to_rgb555());
                break;

            case pix_format_bgr24:
                color_conv(&rbuf_tmp, src, color_conv_bgr24_to_rgb24());
                break;

            case pix_format_abgr32:
                color_conv(&rbuf_tmp, src, color_conv_abgr32_to_argb32());
                break;

            case pix_format_bgra32:
                color_conv(&rbuf_tmp, src, color_conv_bgra32_to_argb32());
                break;

            case pix_format_rgba32:
                color_conv(&rbuf_tmp, src, color_conv_rgba32_to_argb32());
                break;
            }
            pmap_tmp.draw(window);
        }
    }


    //------------------------------------------------------------------------
    bool platform_specific::save_pmap(const char* fn, unsigned idx, 
                                      const rendering_buffer* src)
    {
        if(m_sys_format == m_format)
        {
            return m_pmap_img[idx].save_as_qt(fn);
        }
        else
        {
            pixel_map pmap_tmp;
            pmap_tmp.create(m_pmap_img[idx].width(), 
                            m_pmap_img[idx].height(),
                            org_e(m_sys_bpp));

            rendering_buffer rbuf_tmp;
            rbuf_tmp.attach(pmap_tmp.buf(),
                            pmap_tmp.width(),
                            pmap_tmp.height(),
                            m_flip_y ?
                             -pmap_tmp.row_bytes() :
                              pmap_tmp.row_bytes());
            switch(m_format)
            {
            case pix_format_gray8:
                return false;

            case pix_format_rgb565:
                color_conv(&rbuf_tmp, src, color_conv_rgb565_to_rgb555());
                break;

            case pix_format_rgb24:
                color_conv(&rbuf_tmp, src, color_conv_rgb24_to_bgr24());
                break;

            case pix_format_abgr32:
                color_conv(&rbuf_tmp, src, color_conv_abgr32_to_bgra32());
                break;

            case pix_format_argb32:
                color_conv(&rbuf_tmp, src, color_conv_argb32_to_bgra32());
                break;

            case pix_format_rgba32:
                color_conv(&rbuf_tmp, src, color_conv_rgba32_to_bgra32());
                break;
            }
            return pmap_tmp.save_as_qt(fn);
        }
        return true;
    }



    //------------------------------------------------------------------------
    bool platform_specific::load_pmap(const char* fn, unsigned idx, 
                                      rendering_buffer* dst)
    {
        pixel_map pmap_tmp;
        if(!pmap_tmp.load_from_qt(fn)) return false;

        rendering_buffer rbuf_tmp;
        rbuf_tmp.attach(pmap_tmp.buf(),
                        pmap_tmp.width(),
                        pmap_tmp.height(),
                        m_flip_y ?
                         -pmap_tmp.row_bytes() :
                          pmap_tmp.row_bytes());

        m_pmap_img[idx].create(pmap_tmp.width(), 
                               pmap_tmp.height(), 
                               org_e(m_bpp),
                               0);

        dst->attach(m_pmap_img[idx].buf(),
                    m_pmap_img[idx].width(),
                    m_pmap_img[idx].height(),
                    m_flip_y ?
                      -m_pmap_img[idx].row_bytes() :
                       m_pmap_img[idx].row_bytes());

        switch(m_format)
        {
        case pix_format_gray8:
            return false;
            break;

        case pix_format_rgb555:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb555()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_rgb555()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_rgb555()); break;
            }
            break;

        case pix_format_rgb565:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb565()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_rgb565()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_rgb565()); break;
            }
            break;

        case pix_format_rgb24:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgb24()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_rgb24()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_rgb24()); break;
            }
            break;

        case pix_format_bgr24:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_bgr24()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_bgr24()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_bgr24()); break;
            }
            break;

        case pix_format_abgr32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_abgr32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_abgr32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_abgr32()); break;
            }
            break;

        case pix_format_argb32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_argb32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_argb32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_argb32()); break;
            }
            break;

        case pix_format_bgra32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_bgra32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_bgra32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_bgra32()); break;
            }
            break;

        case pix_format_rgba32:
            switch(pmap_tmp.bpp())
            {
            case 16: color_conv(dst, &rbuf_tmp, color_conv_rgb555_to_rgba32()); break;
            case 24: color_conv(dst, &rbuf_tmp, color_conv_rgb24_to_rgba32()); break;
            case 32: color_conv(dst, &rbuf_tmp, color_conv_argb32_to_rgba32()); break;
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
        if(m_specific->m_window)
        {
        	SetWindowTitleWithCFString (m_specific->m_window, CFStringCreateWithCStringNoCopy (nil, cap, kCFStringEncodingASCII, nil));
        }
    }



    //------------------------------------------------------------------------
    static unsigned get_key_flags(UInt32 wflags)
    {
        unsigned flags = 0;
        
         if(wflags & shiftKey)   flags |= kbd_shift;
         if(wflags & controlKey) flags |= kbd_ctrl;

        return flags;
    }


    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
		SInt16 item;
		Str255 p_msg;
		
		::CopyCStringToPascal (msg, p_msg);
		::StandardAlert (kAlertPlainAlert, (const unsigned char*) "\013AGG Message", p_msg, NULL, &item);
		//::StandardAlert (kAlertPlainAlert, (const unsigned char*) "\pAGG Message", p_msg, NULL, &item);
    }


    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
		::Microseconds (&(m_specific->m_sw_start));
    }


    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        UnsignedWide stop;
        ::Microseconds(&stop);
        return double(stop.lo - 
                      m_specific->m_sw_start.lo) * 1e6 / 
                      double(m_specific->m_sw_freq.lo);
    }


    //------------------------------------------------------------------------
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        if(m_specific->m_sys_format == pix_format_undefined)
        {
            return false;
        }

        m_window_flags = flags;

		// application
		EventTypeSpec		eventType;
		EventHandlerUPP		handlerUPP;

		eventType.eventClass = kEventClassApplication;
		eventType.eventKind = kEventAppQuit;

		handlerUPP = NewEventHandlerUPP(DoAppQuit);

		InstallApplicationEventHandler (handlerUPP, 1, &eventType, nil, nil);

		eventType.eventClass = kEventClassMouse;
		eventType.eventKind = kEventMouseDown;
		handlerUPP = NewEventHandlerUPP(DoMouseDown);
		InstallApplicationEventHandler (handlerUPP, 1, &eventType, this, nil);

		eventType.eventKind = kEventMouseUp;
		handlerUPP = NewEventHandlerUPP(DoMouseUp);
		InstallApplicationEventHandler (handlerUPP, 1, &eventType, this, nil);
		
		eventType.eventKind = kEventMouseDragged;
		handlerUPP = NewEventHandlerUPP(DoMouseDragged);
		InstallApplicationEventHandler (handlerUPP, 1, &eventType, this, nil);

		eventType.eventClass = kEventClassKeyboard;
		eventType.eventKind = kEventRawKeyDown;
		handlerUPP = NewEventHandlerUPP(DoKeyDown);
		InstallApplicationEventHandler (handlerUPP, 1, &eventType, this, nil);

		eventType.eventKind = kEventRawKeyUp;
		handlerUPP = NewEventHandlerUPP(DoKeyUp);
		InstallApplicationEventHandler (handlerUPP, 1, &eventType, this, nil);

		eventType.eventKind = kEventRawKeyRepeat;
		handlerUPP = NewEventHandlerUPP(DoKeyDown);		// 'key repeat' is translated to 'key down'
		InstallApplicationEventHandler (handlerUPP, 1, &eventType, this, nil);

		WindowAttributes	windowAttrs;
		Rect				bounds;

		// window
		windowAttrs = kWindowCloseBoxAttribute | kWindowCollapseBoxAttribute | kWindowStandardHandlerAttribute;
		SetRect (&bounds, 0, 0, width, height);
		OffsetRect (&bounds, 100, 100);
		CreateNewWindow (kDocumentWindowClass, windowAttrs, &bounds, &m_specific->m_window);

        if(m_specific->m_window == nil)
        {
            return false;
        }

		// I assume the text is ASCII.
		// Change to kCFStringEncodingMacRoman, kCFStringEncodingISOLatin1, kCFStringEncodingUTF8 or what else you need.
        SetWindowTitleWithCFString (m_specific->m_window, CFStringCreateWithCStringNoCopy (nil, m_caption, kCFStringEncodingASCII, nil));
		
		eventType.eventClass = kEventClassWindow;
		eventType.eventKind = kEventWindowClose;

		handlerUPP = NewEventHandlerUPP(DoWindowClose);
		InstallWindowEventHandler (m_specific->m_window, handlerUPP, 1, &eventType, this, NULL);

		eventType.eventKind = kEventWindowDrawContent;
		handlerUPP = NewEventHandlerUPP(DoWindowDrawContent);
		InstallWindowEventHandler (m_specific->m_window, handlerUPP, 1, &eventType, this, NULL);
		
		// Periodic task
		// Instead of an idle function I use the Carbon event timer.
		// You may decide to change the wait value which is currently 50 milliseconds.
		EventLoopRef		mainLoop;
		EventLoopTimerUPP	timerUPP;
		EventLoopTimerRef	theTimer;

		mainLoop = GetMainEventLoop();
		timerUPP = NewEventLoopTimerUPP (DoPeriodicTask);
		InstallEventLoopTimer (mainLoop, 0, 50 * kEventDurationMillisecond, timerUPP, this, &theTimer);

        m_specific->create_pmap(width, height, &m_rbuf_window);
        m_initial_width = width;
        m_initial_height = height;
        on_init();
        on_resize(width, height);
        m_specific->m_redraw_flag = true;
		
  		ShowWindow (m_specific->m_window);
  		SetPortWindowPort (m_specific->m_window);
		
      return true;
    }


    //------------------------------------------------------------------------
    int platform_support::run()
    {
		
		RunApplicationEventLoop ();
        return true;
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
#if defined(__MWERKS__)
            if(len < 4 || stricmp(fn + len - 4, ".BMP") != 0)
#else
	        if(len < 4 || strncasecmp(fn + len - 4, ".BMP", 4) != 0)
#endif
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
#if defined(__MWERKS__)
            if(len < 4 || stricmp(fn + len - 4, ".BMP") != 0)
#else
	        if(len < 4 || strncasecmp(fn + len - 4, ".BMP", 4) != 0)
#endif
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
                                   -m_specific->m_pmap_img[idx].row_bytes() :
                                    m_specific->m_pmap_img[idx].row_bytes());
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    void platform_support::force_redraw()
    {
    	Rect	bounds;
    	
        m_specific->m_redraw_flag = true;
        // on_ctrl_change ();
		on_draw();

    	SetRect(&bounds, 0, 0, m_rbuf_window.width(), m_rbuf_window.height());
    	InvalWindowRect(m_specific->m_window, &bounds);
    }



    //------------------------------------------------------------------------
    void platform_support::update_window()
    {
        m_specific->display_pmap(m_specific->m_window, &m_rbuf_window);
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


//------------------------------------------------------------------------
pascal OSStatus DoWindowClose (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	userData;
	
	QuitApplicationEventLoop ();

	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoAppQuit (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	userData;
	
	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoMouseDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	Point wheresMyMouse;
	UInt32 modifier;
	
	GetEventParameter (theEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &wheresMyMouse);
	GlobalToLocal (&wheresMyMouse);
	GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    platform_support * app = reinterpret_cast<platform_support*>(userData);

    app->m_specific->m_cur_x = wheresMyMouse.h;
    if(app->flip_y())
    {
        app->m_specific->m_cur_y = app->rbuf_window().height() - wheresMyMouse.v;
    }
    else
    {
        app->m_specific->m_cur_y = wheresMyMouse.v;
    }
    app->m_specific->m_input_flags = mouse_left | get_key_flags(modifier);
    
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

	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoMouseUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	Point wheresMyMouse;
	UInt32 modifier;
	
	GetEventParameter (theEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &wheresMyMouse);
	GlobalToLocal (&wheresMyMouse);
	GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    platform_support * app = reinterpret_cast<platform_support*>(userData);

    app->m_specific->m_cur_x = wheresMyMouse.h;
    if(app->flip_y())
    {
        app->m_specific->m_cur_y = app->rbuf_window().height() - wheresMyMouse.v;
    }
    else
    {
        app->m_specific->m_cur_y = wheresMyMouse.v;
    }
    app->m_specific->m_input_flags = mouse_left | get_key_flags(modifier);

    if(app->m_ctrls.on_mouse_button_up(app->m_specific->m_cur_x, 
                                       app->m_specific->m_cur_y))
    {
        app->on_ctrl_change();
        app->force_redraw();
    }
    app->on_mouse_button_up(app->m_specific->m_cur_x, 
                            app->m_specific->m_cur_y, 
                            app->m_specific->m_input_flags);

	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoMouseDragged (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	Point wheresMyMouse;
	UInt32 modifier;
	
	GetEventParameter (theEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &wheresMyMouse);
	GlobalToLocal (&wheresMyMouse);
	GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

    platform_support * app = reinterpret_cast<platform_support*>(userData);

    app->m_specific->m_cur_x = wheresMyMouse.h;
    if(app->flip_y())
    {
        app->m_specific->m_cur_y = app->rbuf_window().height() - wheresMyMouse.v;
    }
    else
    {
        app->m_specific->m_cur_y = wheresMyMouse.v;
    }
    app->m_specific->m_input_flags = mouse_left | get_key_flags(modifier);


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
        app->on_mouse_move(app->m_specific->m_cur_x, 
                           app->m_specific->m_cur_y, 
                           app->m_specific->m_input_flags);
    }

	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoKeyDown (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	char key_code;
	UInt32 modifier;
	
	GetEventParameter (theEvent, kEventParamKeyMacCharCodes, typeChar, NULL, sizeof(char), NULL, &key_code);
	GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

	platform_support * app = reinterpret_cast<platform_support*>(userData);

	app->m_specific->m_last_translated_key = 0;
    switch(modifier) 
    {
        case controlKey:
            app->m_specific->m_input_flags |= kbd_ctrl;
            break;

        case shiftKey:
            app->m_specific->m_input_flags |= kbd_shift;
            break;

        default:
            app->m_specific->translate(key_code);
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

		//On a Mac, screenshots are handled by the system.
        case key_f2:                        
            app->copy_window_to_img(agg::platform_support::max_images - 1);
            app->save_img(agg::platform_support::max_images - 1, "screenshot");
            break;
        }


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

	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoKeyUp (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
	char key_code;
	UInt32 modifier;
	
	GetEventParameter (theEvent, kEventParamKeyMacCharCodes, typeChar, NULL, sizeof(char), NULL, &key_code);
	GetEventParameter (theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(UInt32), NULL, &modifier);

	platform_support * app = reinterpret_cast<platform_support*>(userData);

    app->m_specific->m_last_translated_key = 0;
    switch(modifier) 
    {
        case controlKey:
            app->m_specific->m_input_flags &= ~kbd_ctrl;
            break;

        case shiftKey:
            app->m_specific->m_input_flags &= ~kbd_shift;
            break;
    }
    
	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal OSStatus DoWindowDrawContent (EventHandlerCallRef nextHandler, EventRef theEvent, void* userData)
{
    platform_support * app = reinterpret_cast<platform_support*>(userData);

    if(app)
    {
        if(app->m_specific->m_redraw_flag)
        {
            app->on_draw();
            app->m_specific->m_redraw_flag = false;
        }
        app->m_specific->display_pmap(app->m_specific->m_window, &app->rbuf_window());
    }

	return CallNextEventHandler (nextHandler, theEvent);
}


//------------------------------------------------------------------------
pascal void DoPeriodicTask (EventLoopTimerRef theTimer, void* userData)
{
    platform_support * app = reinterpret_cast<platform_support*>(userData);
    
    if(!app->wait_mode())
		app->on_idle();
}


}




//----------------------------------------------------------------------------
int agg_main(int argc, char* argv[]);


// Hm. Classic MacOS does not know command line input.
// CodeWarrior provides a way to mimic command line input.
// The function 'ccommand' can be used to get the command
// line arguments.
//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
#if defined(__MWERKS__)
	// argc = ccommand (&argv);
#endif
    
    // Check if we are launched by double-clicking under OSX 
	// Get rid of extra argument, this will confuse the standard argument parsing
	// calls used in the examples to get the name of the image file to be used
    if ( argc >= 2 && strncmp (argv[1], "-psn", 4) == 0 ) {
        argc = 1;
    } 

launch:
    return agg_main(argc, argv);
}
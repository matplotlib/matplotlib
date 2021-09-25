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
// It's not a part of the AGG library, it's just a helper class to create 
// interactive demo examples. Since the examples should not be too complex
// this class is provided to support some very basic interactive graphical
// functionality, such as putting the rendered image to the window, simple 
// keyboard and mouse input, window resizing, setting the window title,
// and catching the "idle" events.
// 
// The idea is to have a single header file that does not depend on any 
// platform (I hate these endless #ifdef/#elif/#elif.../#endif) and a number
// of different implementations depending on the concrete platform. 
// The most popular platforms are:
//
// Windows-32 API
// X-Window API
// SDL library (see http://www.libsdl.org/)
// MacOS C/C++ API
// 
// This file does not include any system dependent .h files such as
// windows.h or X11.h, so, your demo applications do not depend on the
// platform. The only file that can #include system dependend headers
// is the implementation file agg_platform_support.cpp. Different
// implementations are placed in different directories, such as
// ~/agg/src/platform/win32
// ~/agg/src/platform/sdl
// ~/agg/src/platform/X11
// and so on.
//
// All the system dependent stuff sits in the platform_specific 
// class which is forward-declared here but not defined. 
// The platform_support class has just a pointer to it and it's 
// the responsibility of the implementation to create/delete it.
// This class being defined in the implementation file can have 
// any platform dependent stuff such as HWND, X11 Window and so on.
//
//----------------------------------------------------------------------------


#ifndef AGG_PLATFORM_SUPPORT_INCLUDED
#define AGG_PLATFORM_SUPPORT_INCLUDED


#include "agg_basics.h"
#include "agg_rendering_buffer.h"
#include "agg_trans_viewport.h"
#include "ctrl/agg_ctrl.h"

namespace agg
{

    //----------------------------------------------------------window_flag_e
    // These are flags used in method init(). Not all of them are
    // applicable on different platforms, for example the win32_api
    // cannot use a hardware buffer (window_hw_buffer).
    // The implementation should simply ignore unsupported flags.
    enum window_flag_e
    {
        window_resize            = 1,
        window_hw_buffer         = 2,
        window_keep_aspect_ratio = 4,
        window_process_all_keys  = 8
    };

    //-----------------------------------------------------------pix_format_e
    // Possible formats of the rendering buffer. Initially I thought that it's
    // reasonable to create the buffer and the rendering functions in 
    // accordance with the native pixel format of the system because it 
    // would have no overhead for pixel format conersion. 
    // But eventually I came to a conclusion that having a possibility to 
    // convert pixel formats on demand is a good idea. First, it was X11 where 
    // there lots of different formats and visuals and it would be great to 
    // render everything in, say, RGB-24 and display it automatically without
    // any additional efforts. The second reason is to have a possibility to 
    // debug renderers for different pixel formats and colorspaces having only 
    // one computer and one system.
    //
    // This stuff is not included into the basic AGG functionality because the 
    // number of supported pixel formats (and/or colorspaces) can be great and 
    // if one needs to add new format it would be good only to add new 
    // rendering files without having to modify any existing ones (a general 
    // principle of incapsulation and isolation).
    //
    // Using a particular pixel format doesn't obligatory mean the necessity
    // of software conversion. For example, win32 API can natively display 
    // gray8, 15-bit RGB, 24-bit BGR, and 32-bit BGRA formats. 
    // This list can be (and will be!) extended in future.
    enum pix_format_e
    {
        pix_format_undefined = 0,  // By default. No conversions are applied 
        pix_format_bw,             // 1 bit per color B/W
        pix_format_gray8,          // Simple 256 level grayscale
        pix_format_sgray8,         // Simple 256 level grayscale (sRGB)
        pix_format_gray16,         // Simple 65535 level grayscale
        pix_format_gray32,         // Grayscale, one 32-bit float per pixel
        pix_format_rgb555,         // 15 bit rgb. Depends on the byte ordering!
        pix_format_rgb565,         // 16 bit rgb. Depends on the byte ordering!
        pix_format_rgbAAA,         // 30 bit rgb. Depends on the byte ordering!
        pix_format_rgbBBA,         // 32 bit rgb. Depends on the byte ordering!
        pix_format_bgrAAA,         // 30 bit bgr. Depends on the byte ordering!
        pix_format_bgrABB,         // 32 bit bgr. Depends on the byte ordering!
        pix_format_rgb24,          // R-G-B, one byte per color component
        pix_format_srgb24,         // R-G-B, one byte per color component (sRGB)
        pix_format_bgr24,          // B-G-R, one byte per color component
        pix_format_sbgr24,         // B-G-R, native win32 BMP format (sRGB)
        pix_format_rgba32,         // R-G-B-A, one byte per color component
        pix_format_srgba32,        // R-G-B-A, one byte per color component (sRGB)
        pix_format_argb32,         // A-R-G-B, native MAC format
        pix_format_sargb32,        // A-R-G-B, native MAC format (sRGB)
        pix_format_abgr32,         // A-B-G-R, one byte per color component
        pix_format_sabgr32,        // A-B-G-R, one byte per color component (sRGB)
        pix_format_bgra32,         // B-G-R-A, native win32 BMP format
        pix_format_sbgra32,        // B-G-R-A, native win32 BMP format (sRGB)
        pix_format_rgb48,          // R-G-B, 16 bits per color component
        pix_format_bgr48,          // B-G-R, native win32 BMP format.
        pix_format_rgb96,          // R-G-B, one 32-bit float per color component
        pix_format_bgr96,          // B-G-R, one 32-bit float per color component
        pix_format_rgba64,         // R-G-B-A, 16 bits byte per color component
        pix_format_argb64,         // A-R-G-B, native MAC format
        pix_format_abgr64,         // A-B-G-R, one byte per color component
        pix_format_bgra64,         // B-G-R-A, native win32 BMP format
        pix_format_rgba128,        // R-G-B-A, one 32-bit float per color component
        pix_format_argb128,        // A-R-G-B, one 32-bit float per color component
        pix_format_abgr128,        // A-B-G-R, one 32-bit float per color component
        pix_format_bgra128,        // B-G-R-A, one 32-bit float per color component
  
        end_of_pix_formats
    };

    //-------------------------------------------------------------input_flag_e
    // Mouse and keyboard flags. They can be different on different platforms
    // and the ways they are obtained are also different. But in any case
    // the system dependent flags should be mapped into these ones. The meaning
    // of that is as follows. For example, if kbd_ctrl is set it means that the 
    // ctrl key is pressed and being held at the moment. They are also used in 
    // the overridden methods such as on_mouse_move(), on_mouse_button_down(),
    // on_mouse_button_dbl_click(), on_mouse_button_up(), on_key(). 
    // In the method on_mouse_button_up() the mouse flags have different
    // meaning. They mean that the respective button is being released, but
    // the meaning of the keyboard flags remains the same.
    // There's absolut minimal set of flags is used because they'll be most
    // probably supported on different platforms. Even the mouse_right flag
    // is restricted because Mac's mice have only one button, but AFAIK
    // it can be simulated with holding a special key on the keydoard.
    enum input_flag_e
    {
        mouse_left  = 1,
        mouse_right = 2,
        kbd_shift   = 4,
        kbd_ctrl    = 8
    };

    //--------------------------------------------------------------key_code_e
    // Keyboard codes. There's also a restricted set of codes that are most 
    // probably supported on different platforms. Any platform dependent codes
    // should be converted into these ones. There're only those codes are
    // defined that cannot be represented as printable ASCII-characters. 
    // All printable ASCII-set can be used in a regular C/C++ manner: 
    // ' ', 'A', '0' '+' and so on.
    // Since the class is used for creating very simple demo-applications
    // we don't need very rich possibilities here, just basic ones. 
    // Actually the numeric key codes are taken from the SDL library, so,
    // the implementation of the SDL support does not require any mapping.
    enum key_code_e
    {
        // ASCII set. Should be supported everywhere
        key_backspace      = 8,
        key_tab            = 9,
        key_clear          = 12,
        key_return         = 13,
        key_pause          = 19,
        key_escape         = 27,

        // Keypad 
        key_delete         = 127,
        key_kp0            = 256,
        key_kp1            = 257,
        key_kp2            = 258,
        key_kp3            = 259,
        key_kp4            = 260,
        key_kp5            = 261,
        key_kp6            = 262,
        key_kp7            = 263,
        key_kp8            = 264,
        key_kp9            = 265,
        key_kp_period      = 266,
        key_kp_divide      = 267,
        key_kp_multiply    = 268,
        key_kp_minus       = 269,
        key_kp_plus        = 270,
        key_kp_enter       = 271,
        key_kp_equals      = 272,

        // Arrow-keys and stuff
        key_up             = 273,
        key_down           = 274,
        key_right          = 275,
        key_left           = 276,
        key_insert         = 277,
        key_home           = 278,
        key_end            = 279,
        key_page_up        = 280,
        key_page_down      = 281,

        // Functional keys. You'd better avoid using
        // f11...f15 in your applications if you want 
        // the applications to be portable
        key_f1             = 282,
        key_f2             = 283,
        key_f3             = 284,
        key_f4             = 285,
        key_f5             = 286,
        key_f6             = 287,
        key_f7             = 288,
        key_f8             = 289,
        key_f9             = 290,
        key_f10            = 291,
        key_f11            = 292,
        key_f12            = 293,
        key_f13            = 294,
        key_f14            = 295,
        key_f15            = 296,

        // The possibility of using these keys is 
        // very restricted. Actually it's guaranteed 
        // only in win32_api and win32_sdl implementations
        key_numlock        = 300,
        key_capslock       = 301,
        key_scrollock      = 302,

        // Phew!
        end_of_key_codes
    };


    //------------------------------------------------------------------------
    // A predeclaration of the platform dependent class. Since we do not
    // know anything here the only we can have is just a pointer to this
    // class as a data member. It should be created and destroyed explicitly
    // in the constructor/destructor of the platform_support class. 
    // Although the pointer to platform_specific is public the application 
    // cannot have access to its members or methods since it does not know
    // anything about them and it's a perfect incapsulation :-)
    class platform_specific;

    
    //----------------------------------------------------------ctrl_container
    // A helper class that contains pointers to a number of controls.
    // This class is used to ease the event handling with controls.
    // The implementation should simply call the appropriate methods
    // of this class when appropriate events occur.
    class ctrl_container
    {
        enum max_ctrl_e { max_ctrl = 64 };

    public:
        //--------------------------------------------------------------------
        ctrl_container() : m_num_ctrl(0), m_cur_ctrl(-1) {}

        //--------------------------------------------------------------------
        void add(ctrl& c)
        {
            if(m_num_ctrl < max_ctrl)
            {
                m_ctrl[m_num_ctrl++] = &c;
            }
        }

        //--------------------------------------------------------------------
        bool in_rect(double x, double y)
        {
            unsigned i;
            for(i = 0; i < m_num_ctrl; i++)
            {
                if(m_ctrl[i]->in_rect(x, y)) return true;
            }
            return false;
        }

        //--------------------------------------------------------------------
        bool on_mouse_button_down(double x, double y)
        {
            unsigned i;
            for(i = 0; i < m_num_ctrl; i++)
            {
                if(m_ctrl[i]->on_mouse_button_down(x, y)) return true;
            }
            return false;
        }

        //--------------------------------------------------------------------
        bool on_mouse_button_up(double x, double y)
        {
            unsigned i;
            bool flag = false;
            for(i = 0; i < m_num_ctrl; i++)
            {
                if(m_ctrl[i]->on_mouse_button_up(x, y)) flag = true;
            }
            return flag;
        }

        //--------------------------------------------------------------------
        bool on_mouse_move(double x, double y, bool button_flag)
        {
            unsigned i;
            for(i = 0; i < m_num_ctrl; i++)
            {
                if(m_ctrl[i]->on_mouse_move(x, y, button_flag)) return true;
            }
            return false;
        }

        //--------------------------------------------------------------------
        bool on_arrow_keys(bool left, bool right, bool down, bool up)
        {
            if(m_cur_ctrl >= 0)
            {
                return m_ctrl[m_cur_ctrl]->on_arrow_keys(left, right, down, up);
            }
            return false;
        }

        //--------------------------------------------------------------------
        bool set_cur(double x, double y)
        {
            unsigned i;
            for(i = 0; i < m_num_ctrl; i++)
            {
                if(m_ctrl[i]->in_rect(x, y)) 
                {
                    if(m_cur_ctrl != int(i))
                    {
                        m_cur_ctrl = i;
                        return true;
                    }
                    return false;
                }
            }
            if(m_cur_ctrl != -1)
            {
                m_cur_ctrl = -1;
                return true;
            }
            return false;
        }

    private:
        ctrl*         m_ctrl[max_ctrl];
        unsigned      m_num_ctrl;
        int           m_cur_ctrl;
    };



    //---------------------------------------------------------platform_support
    // This class is a base one to the apllication classes. It can be used 
    // as follows:
    //
    //  class the_application : public agg::platform_support
    //  {
    //  public:
    //      the_application(unsigned bpp, bool flip_y) :
    //          platform_support(bpp, flip_y) 
    //      . . .
    //
    //      //override stuff . . .
    //      virtual void on_init()
    //      {
    //         . . .
    //      }
    //
    //      virtual void on_draw()
    //      {
    //          . . .
    //      }
    //
    //      virtual void on_resize(int sx, int sy)
    //      {
    //          . . .
    //      }
    //      // . . . and so on, see virtual functions
    //
    //
    //      //any your own stuff . . .
    //  };
    //
    //
    //  int agg_main(int argc, char* argv[])
    //  {
    //      the_application app(pix_format_rgb24, true);
    //      app.caption("AGG Example. Lion");
    //
    //      if(app.init(500, 400, agg::window_resize))
    //      {
    //          return app.run();
    //      }
    //      return 1;
    //  }
    //
    // The reason to have agg_main() instead of just main() is that SDL
    // for Windows requires including SDL.h if you define main(). Since
    // the demo applications cannot rely on any platform/library specific
    // stuff it's impossible to include SDL.h into the application files.
    // The demo applications are simple and their use is restricted, so, 
    // this approach is quite reasonable.
    // 
    class platform_support
    {
    public:
        enum max_images_e { max_images = 16 };

        // format - see enum pix_format_e {};
        // flip_y - true if you want to have the Y-axis flipped vertically.
        platform_support(pix_format_e format, bool flip_y);
        virtual ~platform_support();

        // Setting the windows caption (title). Should be able
        // to be called at least before calling init(). 
        // It's perfect if they can be called anytime.
        void        caption(const char* cap);
        const char* caption() const { return m_caption; }

        //--------------------------------------------------------------------
        // These 3 methods handle working with images. The image
        // formats are the simplest ones, such as .BMP in Windows or 
        // .ppm in Linux. In the applications the names of the files
        // should not have any file extensions. Method load_img() can
        // be called before init(), so, the application could be able 
        // to determine the initial size of the window depending on 
        // the size of the loaded image. 
        // The argument "idx" is the number of the image 0...max_images-1
        bool load_img(unsigned idx, const char* file);
        bool save_img(unsigned idx, const char* file);
        bool create_img(unsigned idx, unsigned width=0, unsigned height=0);

        //--------------------------------------------------------------------
        // init() and run(). See description before the class for details.
        // The necessity of calling init() after creation is that it's 
        // impossible to call the overridden virtual function (on_init()) 
        // from the constructor. On the other hand it's very useful to have
        // some on_init() event handler when the window is created but 
        // not yet displayed. The rbuf_window() method (see below) is 
        // accessible from on_init().
        bool init(unsigned width, unsigned height, unsigned flags);
        int  run();

        //--------------------------------------------------------------------
        // The very same parameters that were used in the constructor
        pix_format_e format() const { return m_format; }
        bool flip_y() const { return m_flip_y; }
        unsigned bpp() const { return m_bpp; }

        //--------------------------------------------------------------------
        // The following provides a very simple mechanism of doing someting
        // in background. It's not multithreading. When wait_mode is true
        // the class waits for the events and it does not ever call on_idle().
        // When it's false it calls on_idle() when the event queue is empty.
        // The mode can be changed anytime. This mechanism is satisfactory
        // to create very simple animations.
        bool wait_mode() const { return m_wait_mode; }
        void wait_mode(bool wait_mode) { m_wait_mode = wait_mode; }

        //--------------------------------------------------------------------
        // These two functions control updating of the window. 
        // force_redraw() is an analog of the Win32 InvalidateRect() function.
        // Being called it sets a flag (or sends a message) which results
        // in calling on_draw() and updating the content of the window 
        // when the next event cycle comes.
        // update_window() results in just putting immediately the content 
        // of the currently rendered buffer to the window without calling
        // on_draw().
        void force_redraw();
        void update_window();

        //--------------------------------------------------------------------
        // So, finally, how to draw anythig with AGG? Very simple.
        // rbuf_window() returns a reference to the main rendering 
        // buffer which can be attached to any rendering class.
        // rbuf_img() returns a reference to the previously created
        // or loaded image buffer (see load_img()). The image buffers 
        // are not displayed directly, they should be copied to or 
        // combined somehow with the rbuf_window(). rbuf_window() is
        // the only buffer that can be actually displayed.
        rendering_buffer& rbuf_window()          { return m_rbuf_window; } 
        rendering_buffer& rbuf_img(unsigned idx) { return m_rbuf_img[idx]; } 
        

        //--------------------------------------------------------------------
        // Returns file extension used in the implementation for the particular
        // system.
        const char* img_ext() const;

        //--------------------------------------------------------------------
        void copy_img_to_window(unsigned idx)
        {
            if(idx < max_images && rbuf_img(idx).buf())
            {
                rbuf_window().copy_from(rbuf_img(idx));
            }
        }
        
        //--------------------------------------------------------------------
        void copy_window_to_img(unsigned idx)
        {
            if(idx < max_images)
            {
                create_img(idx, rbuf_window().width(), rbuf_window().height());
                rbuf_img(idx).copy_from(rbuf_window());
            }
        }
       
        //--------------------------------------------------------------------
        void copy_img_to_img(unsigned idx_to, unsigned idx_from)
        {
            if(idx_from < max_images && 
               idx_to < max_images && 
               rbuf_img(idx_from).buf())
            {
                create_img(idx_to, 
                           rbuf_img(idx_from).width(), 
                           rbuf_img(idx_from).height());
                rbuf_img(idx_to).copy_from(rbuf_img(idx_from));
            }
        }

        //--------------------------------------------------------------------
        // Event handlers. They are not pure functions, so you don't have
        // to override them all.
        // In my demo applications these functions are defined inside
        // the the_application class (implicit inlining) which is in general 
        // very bad practice, I mean vitual inline methods. At least it does
        // not make sense. 
        // But in this case it's quite appropriate bacause we have the only
        // instance of the the_application class and it is in the same file 
        // where this class is defined.
        virtual void on_init();
        virtual void on_resize(int sx, int sy);
        virtual void on_idle();
        virtual void on_mouse_move(int x, int y, unsigned flags);
        virtual void on_mouse_button_down(int x, int y, unsigned flags);
        virtual void on_mouse_button_up(int x, int y, unsigned flags);
        virtual void on_key(int x, int y, unsigned key, unsigned flags);
        virtual void on_ctrl_change();
        virtual void on_draw();
        virtual void on_post_draw(void* raw_handler);

        //--------------------------------------------------------------------
        // Adding control elements. A control element once added will be 
        // working and reacting to the mouse and keyboard events. Still, you
        // will have to render them in the on_draw() using function 
        // render_ctrl() because platform_support doesn't know anything about 
        // renderers you use. The controls will be also scaled automatically 
        // if they provide a proper scaling mechanism (all the controls 
        // included into the basic AGG package do).
        // If you don't need a particular control to be scaled automatically 
        // call ctrl::no_transform() after adding.
        void add_ctrl(ctrl& c) { m_ctrls.add(c); c.transform(m_resize_mtx); }

        //--------------------------------------------------------------------
        // Auxiliary functions. trans_affine_resizing() modifier sets up the resizing 
        // matrix on the basis of the given width and height and the initial
        // width and height of the window. The implementation should simply 
        // call this function every time when it catches the resizing event
        // passing in the new values of width and height of the window.
        // Nothing prevents you from "cheating" the scaling matrix if you
        // call this function from somewhere with wrong arguments. 
        // trans_affine_resizing() accessor simply returns current resizing matrix 
        // which can be used to apply additional scaling of any of your 
        // stuff when the window is being resized.
        // width(), height(), initial_width(), and initial_height() must be
        // clear to understand with no comments :-)
        void trans_affine_resizing(int width, int height)
        {
            if(m_window_flags & window_keep_aspect_ratio)
            {
                //double sx = double(width) / double(m_initial_width);
                //double sy = double(height) / double(m_initial_height);
                //if(sy < sx) sx = sy;
                //m_resize_mtx = trans_affine_scaling(sx, sx);
                trans_viewport vp;
                vp.preserve_aspect_ratio(0.5, 0.5, aspect_ratio_meet);
                vp.device_viewport(0, 0, width, height);
                vp.world_viewport(0, 0, m_initial_width, m_initial_height);
                m_resize_mtx = vp.to_affine();
            }
            else
            {
                m_resize_mtx = trans_affine_scaling(
                    double(width) / double(m_initial_width),
                    double(height) / double(m_initial_height));
            }
        }
        trans_affine& trans_affine_resizing() { return m_resize_mtx; }
        const    trans_affine& trans_affine_resizing() const { return m_resize_mtx; }
        double   width()  const { return m_rbuf_window.width(); }
        double   height() const { return m_rbuf_window.height(); }
        double   initial_width()  const { return m_initial_width; }
        double   initial_height() const { return m_initial_height; }
        unsigned window_flags() const { return m_window_flags; }

        //--------------------------------------------------------------------
        // Get raw display handler depending on the system. 
        // For win32 its an HDC, for other systems it can be a pointer to some
        // structure. See the implementation files for detals.
        // It's provided "as is", so, first you should check if it's not null.
        // If it's null the raw_display_handler is not supported. Also, there's 
        // no guarantee that this function is implemented, so, in some 
        // implementations you may have simply an unresolved symbol when linking.
        void* raw_display_handler();

        //--------------------------------------------------------------------
        // display message box or print the message to the console 
        // (depending on implementation)
        void message(const char* msg);

        //--------------------------------------------------------------------
        // Stopwatch functions. Function elapsed_time() returns time elapsed 
        // since the latest start_timer() invocation in millisecods. 
        // The resolutoin depends on the implementation. 
        // In Win32 it uses QueryPerformanceFrequency() / QueryPerformanceCounter().
        void   start_timer();
        double elapsed_time() const;

        //--------------------------------------------------------------------
        // Get the full file name. In most cases it simply returns
        // file_name. As it's appropriate in many systems if you open
        // a file by its name without specifying the path, it tries to 
        // open it in the current directory. The demos usually expect 
        // all the supplementary files to be placed in the current 
        // directory, that is usually coincides with the directory where
        // the executable is. However, in some systems (BeOS) it's not so.
        // For those kinds of systems full_file_name() can help access files 
        // preserving commonly used policy.
        // So, it's a good idea to use in the demos the following:
        // FILE* fd = fopen(full_file_name("some.file"), "r"); 
        // instead of
        // FILE* fd = fopen("some.file", "r"); 
        const char* full_file_name(const char* file_name);

    public:
        platform_specific* m_specific;
        ctrl_container m_ctrls;

        // Sorry, I'm too tired to describe the private 
        // data membders. See the implementations for different
        // platforms for details.
    private:
        platform_support(const platform_support&);
        const platform_support& operator = (const platform_support&);

        pix_format_e     m_format;
        unsigned         m_bpp;
        rendering_buffer m_rbuf_window;
        rendering_buffer m_rbuf_img[max_images];
        unsigned         m_window_flags;
        bool             m_wait_mode;
        bool             m_flip_y;
        char             m_caption[256];
        int              m_initial_width;
        int              m_initial_height;
        trans_affine     m_resize_mtx;
    };


}



#endif


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
// Contact: superstippi@gmx.de
//----------------------------------------------------------------------------
//
// class platform_support
//
//----------------------------------------------------------------------------

#include <new>
#include <stdio.h>

#include <Alert.h>
#include <Application.h>
#include <Bitmap.h>
#include <Message.h>
#include <MessageRunner.h>
#include <Messenger.h>
#include <Path.h>
#include <Roster.h>
#include <TranslationUtils.h>
#include <View.h>
#include <Window.h>

#include <string.h>
#include "platform/agg_platform_support.h"
#include "util/agg_color_conv_rgb8.h"

using std::nothrow;


static void
attach_buffer_to_BBitmap(agg::rendering_buffer& buffer, BBitmap* bitmap, bool flipY)
{
    uint8* bits = (uint8*)bitmap->Bits();
    uint32 width = bitmap->Bounds().IntegerWidth() + 1;
    uint32 height = bitmap->Bounds().IntegerHeight() + 1;
    int32 bpr = bitmap->BytesPerRow();
    if (flipY) {
// XXX: why don't I have to do this?!?
//        bits += bpr * (height - 1);
        bpr = -bpr;
    }
    buffer.attach(bits, width, height, bpr);
}


static color_space
pix_format_to_color_space(agg::pix_format_e format)
{
    color_space bitmapFormat = B_NO_COLOR_SPACE;
    switch (format) {
        case agg::pix_format_rgb555:

            bitmapFormat = B_RGB15;
            break;

        case agg::pix_format_rgb565:

            bitmapFormat = B_RGB16;
            break;

        case agg::pix_format_rgb24:
        case agg::pix_format_bgr24:

            bitmapFormat = B_RGB24;
            break;

        case agg::pix_format_rgba32:
        case agg::pix_format_argb32:
        case agg::pix_format_abgr32:
        case agg::pix_format_bgra32:

            bitmapFormat = B_RGBA32;
            break;
    }
    return bitmapFormat;
}


// #pragma mark -


class AGGView : public BView {
 public:
                            AGGView(BRect frame, agg::platform_support* agg,
                                    agg::pix_format_e format, bool flipY);
    virtual                 ~AGGView();

    virtual void            AttachedToWindow();
    virtual void            DetachedFromWindow();

    virtual void            MessageReceived(BMessage* message);
    virtual void            Draw(BRect updateRect);
    virtual void            FrameResized(float width, float height);

    virtual void            KeyDown(const char* bytes, int32 numBytes);

    virtual void            MouseDown(BPoint where);
    virtual void            MouseMoved(BPoint where, uint32 transit,
                               const BMessage* dragMesage);
    virtual void            MouseUp(BPoint where);

            BBitmap*        Bitmap() const;

            uint8           LastKeyDown() const;
            uint32          MouseButtons();

            void            Update();
            void            ForceRedraw();

            unsigned        GetKeyFlags();

 private:
    BBitmap*                fBitmap;
    agg::pix_format_e       fFormat;
    bool                    fFlipY;

    agg::platform_support*  fAGG;

    uint32                  fMouseButtons;
    int32                   fMouseX;
    int32                   fMouseY;

    uint8                   fLastKeyDown;

    bool                    fRedraw;

    BMessageRunner*         fPulse;
    bigtime_t               fLastPulse;
    bool                    fEnableTicks;
};

AGGView::AGGView(BRect frame,
                 agg::platform_support* agg,
                 agg::pix_format_e format,
                 bool flipY)
    : BView(frame, "AGG View", B_FOLLOW_ALL,
            B_FRAME_EVENTS | B_WILL_DRAW),
      fFormat(format),
      fFlipY(flipY),

      fAGG(agg),

      fMouseButtons(0),
      fMouseX(-1),
      fMouseY(-1),
      
      fLastKeyDown(0),

      fRedraw(true),

      fPulse(NULL),
      fLastPulse(0),
      fEnableTicks(true)
{
    SetViewColor(B_TRANSPARENT_32_BIT);
    
    frame.OffsetTo(0.0, 0.0);
    fBitmap = new BBitmap(frame, 0, pix_format_to_color_space(fFormat));
    if (fBitmap->IsValid()) {
        attach_buffer_to_BBitmap(fAGG->rbuf_window(), fBitmap, fFlipY);
    } else {
        delete fBitmap;
        fBitmap = NULL;
    }
}


AGGView::~AGGView()
{
    delete fBitmap;
    delete fPulse;
}


void
AGGView::AttachedToWindow()
{
    BMessage message('tick');
    BMessenger target(this, Looper());
    delete fPulse;
//    BScreen screen;
//    TODO: calc screen retrace
    fPulse = new BMessageRunner(target, &message, 40000);

    // make sure we call this once
    fAGG->on_resize(Bounds().IntegerWidth() + 1,
                    Bounds().IntegerHeight() + 1);
    MakeFocus();
}


void
AGGView::DetachedFromWindow()
{
    delete fPulse;
    fPulse = NULL;
}


void
AGGView::MessageReceived(BMessage* message)
{
    bigtime_t now = system_time();
    switch (message->what) {
        case 'tick':
            // drop messages that have piled up
            if (/*now - fLastPulse > 30000*/fEnableTicks) {
                fLastPulse = now;
                if (!fAGG->wait_mode())
                    fAGG->on_idle();
                Window()->PostMessage('entk', this);
                fEnableTicks = false;
            } else {
//                printf("dropping tick message (%lld)\n", now - fLastPulse);
            }
            break;
        case 'entk':
            fEnableTicks = true;
            if (now - fLastPulse > 30000) {
                fLastPulse = now;
                if (!fAGG->wait_mode())
                    fAGG->on_idle();
            }
            break;
        default:
            BView::MessageReceived(message);
            break;
    }
}


void
AGGView::Draw(BRect updateRect)
{
    if (fBitmap) {
        if (fRedraw) {
            fAGG->on_draw();
            fRedraw = false;
        }
        if (fFormat == agg::pix_format_bgra32) {
            DrawBitmap(fBitmap, updateRect, updateRect);
        } else {
            BBitmap* bitmap = new BBitmap(fBitmap->Bounds(), 0, B_RGBA32);

            agg::rendering_buffer rbufSrc;
            attach_buffer_to_BBitmap(rbufSrc, fBitmap, false);

            agg::rendering_buffer rbufDst;
            attach_buffer_to_BBitmap(rbufDst, bitmap, false);

            switch(fFormat) {
                case agg::pix_format_rgb555:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgb555_to_bgra32());
                    break;
                case agg::pix_format_rgb565:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgb565_to_bgra32());
                    break;
                case agg::pix_format_rgb24:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgb24_to_bgra32());
                    break;
                case agg::pix_format_bgr24:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_bgr24_to_bgra32());
                    break;
                case agg::pix_format_rgba32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_rgba32_to_bgra32());
                    break;
                case agg::pix_format_argb32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_argb32_to_bgra32());
                    break;
                case agg::pix_format_abgr32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_abgr32_to_bgra32());
                    break;
                case agg::pix_format_bgra32:
                    agg::color_conv(&rbufDst, &rbufSrc,
                                    agg::color_conv_bgra32_to_bgra32());
                    break;
            }
            DrawBitmap(bitmap, updateRect, updateRect);
            delete bitmap;
        }
    } else {
        FillRect(updateRect);
    }
}


void
AGGView::FrameResized(float width, float height)
{
    BRect r(0.0, 0.0, width, height);
    BBitmap* bitmap = new BBitmap(r, 0, pix_format_to_color_space(fFormat));
    if (bitmap->IsValid()) {
        delete fBitmap;
        fBitmap = bitmap;
           attach_buffer_to_BBitmap(fAGG->rbuf_window(), fBitmap, fFlipY);

        fAGG->trans_affine_resizing((int)width + 1,
                                    (int)height + 1);

        // pass the event on to AGG
        fAGG->on_resize((int)width + 1, (int)height + 1);
        
        fRedraw = true;
        Invalidate();
    } else
        delete bitmap;
}


void
AGGView::KeyDown(const char* bytes, int32 numBytes)
{
    if (bytes && numBytes > 0) {
        fLastKeyDown = bytes[0];

        bool left  = false;
        bool up    = false;
        bool right = false;
        bool down  = false;

        switch (fLastKeyDown) {

            case B_LEFT_ARROW:
                left = true;
                break;

            case B_UP_ARROW:
                up = true;
                break;

            case B_RIGHT_ARROW:
                right = true;
                break;

            case B_DOWN_ARROW:
                down = true;
                break;
        }

/*            case key_f2:                        
fAGG->copy_window_to_img(agg::platform_support::max_images - 1);
fAGG->save_img(agg::platform_support::max_images - 1, "screenshot");
break;
}*/


        if (fAGG->m_ctrls.on_arrow_keys(left, right, down, up)) {
            fAGG->on_ctrl_change();
            fAGG->force_redraw();
        } else {
            fAGG->on_key(fMouseX, fMouseY, fLastKeyDown, GetKeyFlags());
        }
//        fAGG->on_key(fMouseX, fMouseY, fLastKeyDown, GetKeyFlags());

    }
}


void
AGGView::MouseDown(BPoint where)
{
    BMessage* currentMessage = Window()->CurrentMessage();
    if (currentMessage) {
        if (currentMessage->FindInt32("buttons", (int32*)&fMouseButtons) < B_OK)
            fMouseButtons = B_PRIMARY_MOUSE_BUTTON;
    } else
        fMouseButtons = B_PRIMARY_MOUSE_BUTTON;

    fMouseX = (int)where.x;
    fMouseY = fFlipY ? (int)(Bounds().Height() - where.y) : (int)where.y;

    // pass the event on to AGG
    if (fMouseButtons == B_PRIMARY_MOUSE_BUTTON) {
        // left mouse button -> see if to handle in controls
        fAGG->m_ctrls.set_cur(fMouseX, fMouseY);
        if (fAGG->m_ctrls.on_mouse_button_down(fMouseX, fMouseY)) {
            fAGG->on_ctrl_change();
            fAGG->force_redraw();
        } else {
            if (fAGG->m_ctrls.in_rect(fMouseX, fMouseY)) {
                if (fAGG->m_ctrls.set_cur(fMouseX, fMouseY)) {
                    fAGG->on_ctrl_change();
                    fAGG->force_redraw();
                }
            } else {
                fAGG->on_mouse_button_down(fMouseX, fMouseY, GetKeyFlags());
            }
        }
    } else if (fMouseButtons & B_SECONDARY_MOUSE_BUTTON) {
        // right mouse button -> simple
        fAGG->on_mouse_button_down(fMouseX, fMouseY, GetKeyFlags());
    }
    SetMouseEventMask(B_POINTER_EVENTS, B_LOCK_WINDOW_FOCUS);
}


void
AGGView::MouseMoved(BPoint where, uint32 transit, const BMessage* dragMesage)
{
    // workarround missed mouse up events
    // (if we react too slowly, app_server might have dropped events)
    BMessage* currentMessage = Window()->CurrentMessage();
    int32 buttons = 0;
    if (currentMessage->FindInt32("buttons", &buttons) < B_OK) {
        buttons = 0;
    }
    if (!buttons)
        MouseUp(where);

    fMouseX = (int)where.x;
    fMouseY = fFlipY ? (int)(Bounds().Height() - where.y) : (int)where.y;

    // pass the event on to AGG
    if (fAGG->m_ctrls.on_mouse_move(fMouseX, fMouseY,
                                    (GetKeyFlags() & agg::mouse_left) != 0)) {
        fAGG->on_ctrl_change();
        fAGG->force_redraw();
    } else {
        if (!fAGG->m_ctrls.in_rect(fMouseX, fMouseY)) {
            fAGG->on_mouse_move(fMouseX, fMouseY, GetKeyFlags());
        }
    }
}


void
AGGView::MouseUp(BPoint where)
{
    fMouseX = (int)where.x;
    fMouseY = fFlipY ? (int)(Bounds().Height() - where.y) : (int)where.y;

    // pass the event on to AGG
    if (fMouseButtons == B_PRIMARY_MOUSE_BUTTON) {
        fMouseButtons = 0;

        if (fAGG->m_ctrls.on_mouse_button_up(fMouseX, fMouseY)) {
            fAGG->on_ctrl_change();
            fAGG->force_redraw();
        }
        fAGG->on_mouse_button_up(fMouseX, fMouseY, GetKeyFlags());
    } else if (fMouseButtons == B_SECONDARY_MOUSE_BUTTON) {
        fMouseButtons = 0;

        fAGG->on_mouse_button_up(fMouseX, fMouseY, GetKeyFlags());
    }
}


BBitmap*
AGGView::Bitmap() const
{
    return fBitmap;
}


uint8
AGGView::LastKeyDown() const
{
    return fLastKeyDown;
}


uint32
AGGView::MouseButtons()
{
    uint32 buttons = 0;
    if (LockLooper()) {
        buttons = fMouseButtons;
        UnlockLooper();
    }
    return buttons;
}


void
AGGView::Update()
{
    // trigger display update
    if (LockLooper()) {
        Invalidate();
        UnlockLooper();
    }
}


void
AGGView::ForceRedraw()
{
    // force a redraw (fRedraw = true;)
    // and trigger display update
    if (LockLooper()) {
        fRedraw = true;
        Invalidate();
        UnlockLooper();
    }
}


unsigned
AGGView::GetKeyFlags()
{
    uint32 buttons = fMouseButtons;
    uint32 mods = modifiers();
    unsigned flags = 0;
    if (buttons & B_PRIMARY_MOUSE_BUTTON)   flags |= agg::mouse_left;
    if (buttons & B_SECONDARY_MOUSE_BUTTON) flags |= agg::mouse_right;
    if (mods & B_SHIFT_KEY)                 flags |= agg::kbd_shift;
    if (mods & B_COMMAND_KEY)               flags |= agg::kbd_ctrl;
    return flags;
}

// #pragma mark -


class AGGWindow : public BWindow {
 public:
                    AGGWindow()
                    : BWindow(BRect(-50.0, -50.0, -10.0, -10.0),
                              "AGG Application", B_TITLED_WINDOW, B_ASYNCHRONOUS_CONTROLS)
                    {
                    }

    virtual bool    QuitRequested()
                    {
                        be_app->PostMessage(B_QUIT_REQUESTED);
                        return true;
                    }

            bool    Init(BRect frame, agg::platform_support* agg, agg::pix_format_e format,
                              bool flipY, uint32 flags)
                    {
                        MoveTo(frame.LeftTop());
                        ResizeTo(frame.Width(), frame.Height());

                        SetFlags(flags);

                        frame.OffsetTo(0.0, 0.0);
                        fView = new AGGView(frame, agg, format, flipY);
                        AddChild(fView);

                        return fView->Bitmap() != NULL;
                    }


        AGGView*    View() const
                    {
                        return fView;
                    }
 private:
    AGGView*        fView;
};

// #pragma mark -


class AGGApplication : public BApplication {
 public:
                    AGGApplication()
                    : BApplication("application/x-vnd.AGG-AGG")
                    {
                        fWindow = new AGGWindow();
                    }

    virtual void    ReadyToRun()
                    {
                        if (fWindow) {
                            fWindow->Show();
                        }
                    }

    virtual bool    Init(agg::platform_support* agg, int width, int height,
                         agg::pix_format_e format, bool flipY, uint32 flags)
                    {
                        BRect r(50.0, 50.0,
                                50.0 + width - 1.0,
                                50.0 + height - 1.0);
                        uint32 windowFlags = B_ASYNCHRONOUS_CONTROLS;
                        if (!(flags & agg::window_resize))
                            windowFlags |= B_NOT_RESIZABLE;

                        return fWindow->Init(r, agg, format, flipY, windowFlags);;
                    }


        AGGWindow*  Window() const
                    {
                        return fWindow;
                    }

 private:
    AGGWindow*      fWindow;
};


// #pragma mark -


namespace agg
{

class platform_specific {
 public:
                    platform_specific(agg::platform_support* agg,
                                      agg::pix_format_e format, bool flip_y)
                        : fAGG(agg),
                          fApp(NULL),
                          fFormat(format),
                          fFlipY(flip_y),
                          fTimerStart(system_time())
                    {
                        memset(fImages, 0, sizeof(fImages));
                        fApp = new AGGApplication();
                        fAppPath[0] = 0;
                        // figure out where we're running from
                        app_info info;
                        status_t ret = fApp->GetAppInfo(&info);
                        if (ret >= B_OK) {
                            BPath path(&info.ref);
                            ret = path.InitCheck();
                            if (ret >= B_OK) {
                                ret = path.GetParent(&path);
                                if (ret >= B_OK) {
                                    sprintf(fAppPath, "%s", path.Path());
                                } else {
                                    fprintf(stderr, "getting app parent folder failed: %s\n", strerror(ret));
                                }
                            } else {
                                fprintf(stderr, "making app path failed: %s\n", strerror(ret));
                            }
                        } else {
                            fprintf(stderr, "GetAppInfo() failed: %s\n", strerror(ret));
                        }
                    }
                    ~platform_specific()
                    {
                        for (int32 i = 0; i < agg::platform_support::max_images; i++)
                            delete fImages[i];
                        delete fApp;
                    }

    bool            Init(int width, int height, unsigned flags)
                    {
                        return fApp->Init(fAGG, width, height, fFormat, fFlipY, flags);
                    }

    int             Run()
                    {
                        status_t ret = B_NO_INIT;
                        if (fApp) {
                            fApp->Run();
                            ret = B_OK;
                        }
                        return ret;
                    }

    void            SetTitle(const char* title)
                    {
                        if (fApp && fApp->Window() && fApp->Window()->Lock()) {
                            fApp->Window()->SetTitle(title);
                            fApp->Window()->Unlock();
                        }
                    }
    void            StartTimer()
                    {
                        fTimerStart = system_time();
                    }
    double          ElapsedTime() const
                    {
                        return (system_time() - fTimerStart) / 1000.0;
                    }

    void            ForceRedraw()
                    {
                        fApp->Window()->View()->ForceRedraw();
                    }
    void            UpdateWindow()
                    {
                        fApp->Window()->View()->Update();
                    }


    agg::platform_support*  fAGG;
    AGGApplication*     fApp;
    agg::pix_format_e    fFormat;
    bool                fFlipY;
    bigtime_t           fTimerStart;
    BBitmap*            fImages[agg::platform_support::max_images];

    char                fAppPath[B_PATH_NAME_LENGTH];
    char                fFilePath[B_PATH_NAME_LENGTH];
};


    //------------------------------------------------------------------------
    platform_support::platform_support(pix_format_e format, bool flip_y) :
        m_specific(new platform_specific(this, format, flip_y)),
        m_format(format),
        m_bpp(32/*m_specific->m_bpp*/),
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
        m_specific->SetTitle(cap);
    }

    //------------------------------------------------------------------------
    void platform_support::start_timer()
    {
        m_specific->StartTimer();
    }

    //------------------------------------------------------------------------
    double platform_support::elapsed_time() const
    {
        return m_specific->ElapsedTime();
    }

    //------------------------------------------------------------------------
    void* platform_support::raw_display_handler()
    {
        // TODO: if we ever support BDirectWindow here, that would
        // be the frame buffer pointer with offset to the window top left
        return NULL;
    }

    //------------------------------------------------------------------------
    void platform_support::message(const char* msg)
    {
        BAlert* alert = new BAlert("AGG Message", msg, "Ok");
        alert->Go(/*NULL*/);
    }


    //------------------------------------------------------------------------
    bool platform_support::init(unsigned width, unsigned height, unsigned flags)
    {
        m_initial_width = width;
        m_initial_height = height;
        m_window_flags = flags;

        if (m_specific->Init(width, height, flags)) {
            on_init();
            return true;
        }

        return false;
    }


    //------------------------------------------------------------------------
    int platform_support::run()
    {
        return m_specific->Run();
    }


    //------------------------------------------------------------------------
    const char* platform_support::img_ext() const { return ".ppm"; }


    const char* platform_support::full_file_name(const char* file_name)
    {
        sprintf(m_specific->fFilePath, "%s/%s", m_specific->fAppPath, file_name);
        return m_specific->fFilePath;
    }


    //------------------------------------------------------------------------
    bool platform_support::load_img(unsigned idx, const char* file)
    {
        if (idx < max_images)
        {
            char path[B_PATH_NAME_LENGTH];
            sprintf(path, "%s/%s%s", m_specific->fAppPath, file, img_ext());
            BBitmap* transBitmap = BTranslationUtils::GetBitmap(path);
            if (transBitmap && transBitmap->IsValid()) {
                if(transBitmap->ColorSpace() != B_RGB32 && transBitmap->ColorSpace() != B_RGBA32) {
                    // ups we got a smart ass Translator making our live harder
                    delete transBitmap;
                    return false;
                }

                color_space format = B_RGB24;

                switch (m_format) {
                    case pix_format_gray8:
                        format = B_GRAY8;
                        break;
                    case pix_format_rgb555:
                        format = B_RGB15;
                        break;
                    case pix_format_rgb565:
                        format = B_RGB16;
                        break;
                    case pix_format_rgb24:
                        format = B_RGB24_BIG;
                        break;
                    case pix_format_bgr24:
                        format = B_RGB24;
                        break;
                    case pix_format_abgr32:
                    case pix_format_argb32:
                    case pix_format_bgra32:
                        format = B_RGB32;
                        break;
                    case pix_format_rgba32:
                        format = B_RGB32_BIG;
                        break;
                }
                BBitmap* bitmap = new (nothrow) BBitmap(transBitmap->Bounds(), 0, format);
                if (!bitmap || !bitmap->IsValid()) {
                    fprintf(stderr, "failed to allocate temporary bitmap!\n");
                    delete transBitmap;
                    delete bitmap;
                    return false;
                }

                delete m_specific->fImages[idx];

                rendering_buffer rbuf_tmp;
                attach_buffer_to_BBitmap(rbuf_tmp, transBitmap, m_flip_y);
        
                m_specific->fImages[idx] = bitmap;
        
                attach_buffer_to_BBitmap(m_rbuf_img[idx], bitmap, m_flip_y);
        
                rendering_buffer* dst = &m_rbuf_img[idx];

                switch(m_format)
                {
                case pix_format_gray8:
                    return false;
//                  color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_gray8()); break;
                    break;
        
                case pix_format_rgb555:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb555()); break;
                    break;
        
                case pix_format_rgb565:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb565()); break;
                    break;
        
                case pix_format_rgb24:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgb24()); break;
                    break;
        
                case pix_format_bgr24:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgr24()); break;
                    break;
        
                case pix_format_abgr32:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_abgr32()); break;
                    break;
        
                case pix_format_argb32:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_argb32()); break;
                    break;
        
                case pix_format_bgra32:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_bgra32()); break;
                    break;
        
                case pix_format_rgba32:
                    color_conv(dst, &rbuf_tmp, color_conv_bgra32_to_rgba32()); break;
                    break;
                }
                delete transBitmap;
                
                return true;

            } else {
                fprintf(stderr, "failed to load bitmap: '%s'\n", full_file_name(file));
            }
        }
        return false;
    }



    //------------------------------------------------------------------------
    bool platform_support::save_img(unsigned idx, const char* file)
    {
        // TODO: implement using BTranslatorRoster and friends
        return false;
    }



    //------------------------------------------------------------------------
    bool platform_support::create_img(unsigned idx, unsigned width, unsigned height)
    {
        if(idx < max_images)
        {
            if(width  == 0) width  = m_specific->fApp->Window()->View()->Bitmap()->Bounds().IntegerWidth() + 1;
            if(height == 0) height = m_specific->fApp->Window()->View()->Bitmap()->Bounds().IntegerHeight() + 1;
            BBitmap* bitmap = new BBitmap(BRect(0.0, 0.0, width - 1, height - 1), 0, B_RGBA32);;
            if (bitmap && bitmap->IsValid()) {
                delete m_specific->fImages[idx];
                m_specific->fImages[idx] = bitmap;
                attach_buffer_to_BBitmap(m_rbuf_img[idx], bitmap, m_flip_y);
                return true;
            } else {
                delete bitmap;
            }
        }
        return false;
    }


    //------------------------------------------------------------------------
    void platform_support::force_redraw()
    {
        m_specific->ForceRedraw();
    }



    //------------------------------------------------------------------------
    void platform_support::update_window()
    {
        m_specific->UpdateWindow();
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






//----------------------------------------------------------------------------
int agg_main(int argc, char* argv[]);



int
main(int argc, char* argv[])
{
    return agg_main(argc, argv);
}





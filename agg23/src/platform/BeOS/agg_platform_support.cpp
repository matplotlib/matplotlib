//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.3
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

class AGGView : public BView {
 public:
                    AGGView(BRect frame, agg::platform_support* agg, agg::pix_format_e format, bool flipY)
                    : BView(frame, "AGG View", B_FOLLOW_ALL,
                            B_FRAME_EVENTS | B_WILL_DRAW),
                      fAGG(agg),
                      fFormat(format),
                      fMouseButtons(0),
                      fFlipY(flipY),
                      fRedraw(true),
                      fPulse(NULL),
                      fLastPulse(0),
                      fEnableTicks(true)
                    {
                        SetViewColor(B_TRANSPARENT_32_BIT);

                        frame.OffsetTo(0.0, 0.0);
                        fBitmap = new BBitmap(frame, 0, B_RGBA32);
                        if (fBitmap->IsValid()) {
                            memset(fBitmap->Bits(), 0, fBitmap->BitsLength());
                            fAGG->rbuf_window().attach((uint8*)fBitmap->Bits(),
                                                       fBitmap->Bounds().IntegerWidth() + 1,
                                                       fBitmap->Bounds().IntegerHeight() + 1,
                                                       fFlipY ? -fBitmap->BytesPerRow() : fBitmap->BytesPerRow());
                        } else {
                            delete fBitmap;
                            fBitmap = NULL;
                        }
                    }

    virtual         ~AGGView()
                    {
                        delete fBitmap;
                        delete fPulse;
                    }

    virtual void    AttachedToWindow()
                    {
                        BMessage message('tick');
                        BMessenger target(this, Looper());
                        delete fPulse;
//                      BScreen screen;
//                      TODO: calc screen retrace
                        fPulse = new BMessageRunner(target, &message, 40000);

                        // make sure we call this once
                        fAGG->on_resize(Bounds().IntegerWidth() + 1,
                                        Bounds().IntegerHeight() + 1);
                    }
    virtual void    DetachedFromWindow()
                    {
                        delete fPulse;
                        fPulse = NULL;
                    }

    virtual void    MessageReceived(BMessage* message)
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
//                                  fprintf(stderr, "dropping tick message (%lld)\n", now - fLastPulse);
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

    virtual void    Draw(BRect updateRect)
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

                                agg::rendering_buffer rbuf_src;
                                rbuf_src.attach((uint8*)fBitmap->Bits(), 
                                                fBitmap->Bounds().IntegerWidth() + 1, 
                                                fBitmap->Bounds().IntegerHeight() + 1, 
                                                fFlipY ? -fBitmap->BytesPerRow() : fBitmap->BytesPerRow());

                                agg::rendering_buffer rbuf_dst;
                                rbuf_dst.attach((uint8*)bitmap->Bits(), 
                                                bitmap->Bounds().IntegerWidth() + 1, 
                                                bitmap->Bounds().IntegerHeight() + 1, 
                                                fFlipY ? -bitmap->BytesPerRow() : bitmap->BytesPerRow());
                                switch(fFormat) {
                                    case agg::pix_format_rgb555: agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_rgb555_to_bgra32()); break;
                                    case agg::pix_format_rgb565: agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_rgb565_to_bgra32()); break;
                                    case agg::pix_format_rgb24:  agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_rgb24_to_bgra32());  break;
                                    case agg::pix_format_bgr24:  agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_bgr24_to_bgra32());  break;
                                    case agg::pix_format_rgba32: agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_rgba32_to_bgra32()); break;
                                    case agg::pix_format_argb32: agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_argb32_to_bgra32()); break;
                                    case agg::pix_format_abgr32: agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_abgr32_to_bgra32()); break;
                                    case agg::pix_format_bgra32: agg::color_conv(&rbuf_dst, &rbuf_src, agg::color_conv_bgra32_to_bgra32()); break;
                                }
                                DrawBitmap(bitmap, updateRect, updateRect);
                                delete bitmap;
                            }
                        } else {
                            FillRect(updateRect);
                        }
                    }

    virtual void    FrameResized(float width, float height)
                    {
                        BRect r(0.0, 0.0, width, height);
                        BBitmap* bitmap = new BBitmap(r, 0, B_RGBA32);
                        if (bitmap->IsValid()) {
                        	delete fBitmap;
                        	fBitmap = bitmap;
                            fAGG->rbuf_window().attach((uint8*)fBitmap->Bits(),
                                                       fBitmap->Bounds().IntegerWidth() + 1,
                                                       fBitmap->Bounds().IntegerHeight() + 1,
                                                       fFlipY ? -fBitmap->BytesPerRow() : fBitmap->BytesPerRow());
    
                            fAGG->trans_affine_resizing((int)width + 1,
                                                        (int)height + 1);
    
                            // pass the event on to AGG
                            fAGG->on_resize((int)width + 1,
                                            (int)height + 1);
                            
                            fRedraw = true;
                            Invalidate();
                        } else
                        	delete bitmap;
                    }

    virtual void    KeyDown(const char* bytes, int32 numBytes)
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

/*                case key_f2:                        
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
//                          fAGG->on_key(fMouseX, fMouseY, fLastKeyDown, GetKeyFlags());

                        }
                    }
    virtual void    MouseDown(BPoint where)
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
    virtual void    MouseMoved(BPoint where, uint32 transit, const BMessage* dragMesage)
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

    virtual void    MouseUp(BPoint where)
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

    BBitmap*        Bitmap() const
                    {
                        return fBitmap;
                    }

    uint8           LastKeyDown() const
                    {
                        return fLastKeyDown;
                    }
    uint32          MouseButtons()
                    {
                        uint32 buttons = 0;
                        if (LockLooper()) {
                            buttons = fMouseButtons;
                            UnlockLooper();
                        }
                        return buttons;
                    }

    void            Update()
                    {
                        // trigger display update
                        if (LockLooper()) {
                            Invalidate();
                            UnlockLooper();
                        }
                    }

    void            ForceRedraw()
                    {
                        // force a redraw (fRedraw = true;)
                        // and trigger display update
                        if (LockLooper()) {
                            fRedraw = true;
                            Invalidate();
                            UnlockLooper();
                        }
                    }

    unsigned        GetKeyFlags()
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

 private:
    BBitmap*            fBitmap;
    uint8               fLastKeyDown;
    agg::platform_support*  fAGG;
    agg::pix_format_e       fFormat;

    uint32              fMouseButtons;
    int32               fMouseX;
    int32               fMouseY;
    bool                fFlipY;
    bool                fRedraw;
    BMessageRunner*     fPulse;
    bigtime_t           fLastPulse;
    bool                fEnableTicks;
};

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
                        // ignore flip_y for now
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
			            status_t ret = be_roster->GetRunningAppInfo(be_app->Team(), &info);
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
			                fprintf(stderr, "GetRunningAppInfo() failed: %s\n", strerror(ret));
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
    agg::pix_format_e	fFormat;
    bool                fFlipY;
    bigtime_t           fTimerStart;
    BBitmap*            fImages[agg::platform_support::max_images];

    char				fAppPath[B_PATH_NAME_LENGTH];
    char				fFilePath[B_PATH_NAME_LENGTH];
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
        return NULL;//m_specific->m_current_dc;
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
        bool success = m_specific->Init(width, height, flags);

        m_window_flags = flags;

//      m_specific->create_pmap(width, height, &m_rbuf_window);
        m_initial_width = width;
        m_initial_height = height;
        on_init();
//      m_specific->m_redraw_flag = true;
        return true;
    }



    //------------------------------------------------------------------------
    int platform_support::run()
    {
        return m_specific->Run();
    }


    //------------------------------------------------------------------------
    const char* platform_support::img_ext() const { return ""; }

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
            BBitmap* transBitmap = BTranslationUtils::GetBitmap(full_file_name(file));
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
                BBitmap* bitmap = new BBitmap(transBitmap->Bounds(), 0, format);
                if (!bitmap || !bitmap->IsValid()) {
                    fprintf(stderr, "failed to allocate temporary bitmap!\n");
                    delete transBitmap;
                    delete bitmap;
                    return false;
                }

                delete m_specific->fImages[idx];

                rendering_buffer rbuf_tmp;
                rbuf_tmp.attach((uint8*)transBitmap->Bits(),
                                transBitmap->Bounds().IntegerWidth() + 1,
                                transBitmap->Bounds().IntegerHeight() + 1,
                                m_flip_y ? -transBitmap->BytesPerRow() : transBitmap->BytesPerRow());
        
                m_specific->fImages[idx] = bitmap;
        
                m_rbuf_img[idx].attach((uint8*)bitmap->Bits(),
                                       bitmap->Bounds().IntegerWidth() + 1,
                                       bitmap->Bounds().IntegerHeight() + 1,
                                       m_flip_y ? -bitmap->BytesPerRow() : bitmap->BytesPerRow());
        
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
                fprintf(stderr, "failed to load bitmap: '%s'\n",full_file_name(file));
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
                m_rbuf_img[idx].attach((uint8*)bitmap->Bits(), 
                                        width, height,
                                        m_flip_y ? -bitmap->BytesPerRow() : bitmap->BytesPerRow());
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





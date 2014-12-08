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

#include "platform/agg_platform_support.h"
#include "util/agg_color_conv_rgb8.h"

#include <sys/time.h>
#include <cstring>

#include <classes/requester.h>
#include <classes/window.h>
#include <datatypes/pictureclass.h>
#include <proto/exec.h>
#include <proto/datatypes.h>
#include <proto/dos.h>
#include <proto/graphics.h>
#include <proto/intuition.h>
#include <proto/keymap.h>
#include <proto/Picasso96API.h>
#include <proto/utility.h>

Library* DataTypesBase = 0;
Library* GraphicsBase = 0;
Library* IntuitionBase = 0;
Library* KeymapBase = 0;
Library* P96Base = 0;

DataTypesIFace* IDataTypes = 0;
GraphicsIFace* IGraphics = 0;
IntuitionIFace* IIntuition = 0;
KeymapIFace* IKeymap = 0;
P96IFace* IP96 = 0;

Class* RequesterClass = 0;
Class* WindowClass = 0;


namespace agg
{
	void handle_idcmp(Hook* hook, APTR win, IntuiMessage* msg);

	//------------------------------------------------------------------------
	class platform_specific
	{
	public:
		platform_specific(platform_support& support, pix_format_e format,
			bool flip_y);
		~platform_specific();
		bool handle_input();
		bool load_img(const char* file, unsigned idx, rendering_buffer* rbuf);
		bool create_img(unsigned idx, rendering_buffer* rbuf, unsigned width,
			unsigned height);
		bool make_bitmap();
	public:
		platform_support& m_support;
		RGBFTYPE m_ftype;
		pix_format_e m_format;
		unsigned m_bpp;
		BitMap* m_bitmap;
		bool m_flip_y;
		uint16 m_width;
		uint16 m_height;
		APTR m_window_obj;
		Window* m_window;
		Hook* m_idcmp_hook;
		unsigned m_input_flags;
		bool m_dragging;
		double m_start_time;
		uint16 m_last_key;
		BitMap* m_img_bitmaps[platform_support::max_images];
	};

	//------------------------------------------------------------------------
	platform_specific::platform_specific(platform_support& support,
		pix_format_e format, bool flip_y) :
		m_support(support),
		m_ftype(RGBFB_NONE),
		m_format(format),
		m_bpp(0),
		m_bitmap(0),
		m_flip_y(flip_y),
		m_width(0),
		m_height(0),
		m_window_obj(0),
		m_window(0),
		m_idcmp_hook(0),
		m_input_flags(0),
		m_dragging(false),
		m_start_time(0.0),
		m_last_key(0)
	{
		switch ( format )
		{
		case pix_format_gray8:
			// Not supported.
			break;
		case pix_format_rgb555:
			m_ftype = RGBFB_R5G5B5;
			m_bpp = 15;
			break;
		case pix_format_rgb565:
			m_ftype = RGBFB_R5G6B5;
			m_bpp = 16;
			break;
		case pix_format_rgb24:
			m_ftype = RGBFB_R8G8B8;
			m_bpp = 24;
			break;
		case pix_format_bgr24:
			m_ftype = RGBFB_B8G8R8;
			m_bpp = 24;
			break;
		case pix_format_bgra32:
			m_ftype = RGBFB_B8G8R8A8;
			m_bpp = 32;
			break;
		case pix_format_abgr32:
			m_ftype = RGBFB_A8B8G8R8;
			m_bpp = 32;
			break;
		case pix_format_argb32:
			m_ftype = RGBFB_A8R8G8B8;
			m_bpp = 32;
			break;
        case pix_format_rgba32:
			m_ftype = RGBFB_R8G8B8A8;
			m_bpp = 32;
			break;
		}

		for ( unsigned i = 0; i < platform_support::max_images; ++i )
		{
			m_img_bitmaps[i] = 0;
		}
	}

	//------------------------------------------------------------------------
	platform_specific::~platform_specific()
	{
		IIntuition->DisposeObject(m_window_obj);

		IP96->p96FreeBitMap(m_bitmap);

		for ( unsigned i = 0; i < platform_support::max_images; ++i )
		{
			IP96->p96FreeBitMap(m_img_bitmaps[i]);
		}

		if ( m_idcmp_hook != 0 )
		{
			IExec->FreeSysObject(ASOT_HOOK, m_idcmp_hook);
		}
	}

	//------------------------------------------------------------------------
	bool platform_specific::handle_input()
	{
		int16 code = 0;
		uint32 result = 0;
		Object* obj = reinterpret_cast<Object*>(m_window_obj);

		while ( (result = IIntuition->IDoMethod(obj, WM_HANDLEINPUT,
				&code)) != WMHI_LASTMSG )
		{
			switch ( result & WMHI_CLASSMASK )
			{
			case WMHI_CLOSEWINDOW:
				return true;
				break;
			case WMHI_INTUITICK:
				if ( !m_support.wait_mode() )
				{
					m_support.on_idle();
				}
				break;
			case WMHI_NEWSIZE:
				if ( make_bitmap() )
				{
					m_support.trans_affine_resizing(m_width, m_height);
					m_support.on_resize(m_width, m_height);
					m_support.force_redraw();
				}
				break;
			}
		}

		return false;
	}		

	//------------------------------------------------------------------------
	bool platform_specific::load_img(const char* file, unsigned idx,
		rendering_buffer* rbuf)
	{
		if ( m_img_bitmaps[idx] != 0 )
		{
			IP96->p96FreeBitMap(m_img_bitmaps[idx]);
			m_img_bitmaps[idx] = 0;
		}

		bool result = false;

		Object* picture = IDataTypes->NewDTObject(const_cast<STRPTR>(file),
			DTA_GroupID, GID_PICTURE,
			PDTA_DestMode, PMODE_V43,
			PDTA_Remap, FALSE,
			TAG_END);
		if ( picture != 0 )
		{
			gpLayout layout;
			layout.MethodID = DTM_PROCLAYOUT;
			layout.gpl_GInfo = 0;
			layout.gpl_Initial = 1;
			ULONG loaded = IDataTypes->DoDTMethodA(picture, 0, 0,
				reinterpret_cast<Msg>(&layout));
			if ( loaded != 0 )
			{
				BitMap* src_bitmap = 0;
				IDataTypes->GetDTAttrs(picture,
					PDTA_ClassBitMap, &src_bitmap,
					TAG_END);

				bool supported = false;

				RGBFTYPE ftype = static_cast<RGBFTYPE>(IP96->p96GetBitMapAttr(
					src_bitmap, P96BMA_RGBFORMAT));

				switch ( ftype )
				{
				case RGBFB_R8G8B8:
					supported = true;
					break;
				default:
					m_support.message("File uses unsupported graphics mode.");
					break;
				}

				if ( supported )  {
					uint16 width = IP96->p96GetBitMapAttr(src_bitmap,
						P96BMA_WIDTH);
					uint16 height = IP96->p96GetBitMapAttr(src_bitmap,
						P96BMA_HEIGHT);

					m_img_bitmaps[idx] = IP96->p96AllocBitMap(width, height,
						m_bpp, BMF_USERPRIVATE, 0, m_ftype);
					if ( m_img_bitmaps[idx] != 0 )
					{
						int8u* buf = reinterpret_cast<int8u*>(
							IP96->p96GetBitMapAttr(m_img_bitmaps[idx],
							P96BMA_MEMORY));
						int bpr = IP96->p96GetBitMapAttr(m_img_bitmaps[idx],
							P96BMA_BYTESPERROW);
						int stride = (m_flip_y) ? -bpr : bpr;
						rbuf->attach(buf, width, height, stride);

						// P96 sets the alpha to zero so it can't be used to
						// color convert true color modes.
						if ( m_bpp == 32 )
						{
							RenderInfo ri;
							int32 lock = IP96->p96LockBitMap(src_bitmap,
								reinterpret_cast<uint8*>(&ri),
								sizeof(RenderInfo));

							rendering_buffer rbuf_src;
							rbuf_src.attach(
								reinterpret_cast<int8u*>(ri.Memory),
								width, height, (m_flip_y) ?
									-ri.BytesPerRow : ri.BytesPerRow);

							switch ( m_format )
							{
							case pix_format_bgra32:
								color_conv(rbuf, &rbuf_src,
									color_conv_rgb24_to_bgra32());
								break;
							case pix_format_abgr32:
								color_conv(rbuf, &rbuf_src,
									color_conv_rgb24_to_abgr32());
								break;
							case pix_format_argb32:
								color_conv(rbuf, &rbuf_src,
									color_conv_rgb24_to_argb32());
								break;
							case pix_format_rgba32:
								color_conv(rbuf, &rbuf_src,
									color_conv_rgb24_to_rgba32());
								break;
							}

							IP96->p96UnlockBitMap(src_bitmap, lock);
						}
						else
						{
							IGraphics->BltBitMap(src_bitmap, 0, 0,
								m_img_bitmaps[idx], 0, 0, width, height,
								ABC|ABNC, 0xFF, 0);
						}

						result = true;
					}
				}
			}
		}

		IGraphics->WaitBlit();
		IDataTypes->DisposeDTObject(picture);

		return result;
	}

	//------------------------------------------------------------------------
	bool platform_specific::create_img(unsigned idx, rendering_buffer* rbuf,
		unsigned width, unsigned height)
	{
		if ( m_img_bitmaps[idx] != 0 )
		{
			IP96->p96FreeBitMap(m_img_bitmaps[idx]);
			m_img_bitmaps[idx] = 0;
		}

		m_img_bitmaps[idx] = IP96->p96AllocBitMap(width, height,
			m_bpp, BMF_USERPRIVATE, m_bitmap, m_ftype);
		if ( m_img_bitmaps[idx] != 0 )
		{
			int8u* buf = reinterpret_cast<int8u*>(
				IP96->p96GetBitMapAttr(m_img_bitmaps[idx],
				P96BMA_MEMORY));
			int bpr = IP96->p96GetBitMapAttr(m_img_bitmaps[idx],
				P96BMA_BYTESPERROW);
			int stride = (m_flip_y) ? -bpr : bpr;

			rbuf->attach(buf, width, height, stride);

			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------
	bool platform_specific::make_bitmap()
	{
		uint32 width = 0;
		uint32 height = 0;
		IIntuition->GetWindowAttrs(m_window,
			WA_InnerWidth, &width,
			WA_InnerHeight, &height,
			TAG_END);

		BitMap* bm = IP96->p96AllocBitMap(width, height, m_bpp,
			BMF_USERPRIVATE|BMF_CLEAR, 0, m_ftype);
		if ( bm == 0 )
		{
			return false;
		}

		int8u* buf = reinterpret_cast<int8u*>(
			IP96->p96GetBitMapAttr(bm, P96BMA_MEMORY));
		int bpr = IP96->p96GetBitMapAttr(bm, P96BMA_BYTESPERROW);
		int stride = (m_flip_y) ? -bpr : bpr;

		m_support.rbuf_window().attach(buf, width, height, stride);

		if ( m_bitmap != 0 )
		{
			IP96->p96FreeBitMap(m_bitmap);
			m_bitmap = 0;
		}

		m_bitmap = bm;
		m_width = width;
		m_height = height;

		return true;
	}

	//------------------------------------------------------------------------
	platform_support::platform_support(pix_format_e format, bool flip_y) :
		m_specific(new platform_specific(*this, format, flip_y)),
		m_format(format),
		m_bpp(m_specific->m_bpp),
		m_window_flags(0),
		m_wait_mode(true),
		m_flip_y(flip_y),
		m_initial_width(10),
		m_initial_height(10)
	{
		std::strncpy(m_caption, "Anti-Grain Geometry", 256);
	}

	//------------------------------------------------------------------------
	platform_support::~platform_support()
	{
		delete m_specific;
	}

	//------------------------------------------------------------------------
	void platform_support::caption(const char* cap)
	{
		std::strncpy(m_caption, cap, 256);
		if ( m_specific->m_window != 0 )
		{
			const char* ignore = reinterpret_cast<const char*>(-1);
			IIntuition->SetWindowAttr(m_specific->m_window,
				WA_Title, m_caption, sizeof(char*));
		}
	}

	//------------------------------------------------------------------------
	void platform_support::start_timer()
	{
		timeval tv;
		gettimeofday(&tv, 0);
		m_specific->m_start_time = tv.tv_secs + tv.tv_micro/1e6;
	}

	//------------------------------------------------------------------------
	double platform_support::elapsed_time() const
	{
		timeval tv;
		gettimeofday(&tv, 0);
		double end_time = tv.tv_secs + tv.tv_micro/1e6;

		double elasped_seconds = end_time - m_specific->m_start_time;
		double elasped_millis = elasped_seconds*1e3;

		return elasped_millis;
	}

	//------------------------------------------------------------------------
	void* platform_support::raw_display_handler()
	{
		return 0;	// Not available.
	}

	//------------------------------------------------------------------------
	void platform_support::message(const char* msg)
	{
		APTR req = IIntuition->NewObject(RequesterClass, 0,
			REQ_TitleText, "Anti-Grain Geometry",
			REQ_Image, REQIMAGE_INFO,
			REQ_BodyText, msg,
			REQ_GadgetText, "_Ok",
			TAG_END);
		if ( req == 0 )
		{
			IDOS->Printf("Message: %s\n", msg);
			return;
		}

		orRequest reqmsg;
		reqmsg.MethodID = RM_OPENREQ;
		reqmsg.or_Attrs = 0;
		reqmsg.or_Window = m_specific->m_window;
		reqmsg.or_Screen = 0;
		
		IIntuition->IDoMethodA(reinterpret_cast<Object*>(req),
			reinterpret_cast<Msg>(&reqmsg));
		IIntuition->DisposeObject(req);
	}

	//------------------------------------------------------------------------
	bool platform_support::init(unsigned width, unsigned height,
		unsigned flags)
	{
		if( m_specific->m_ftype == RGBFB_NONE )
		{
			message("Unsupported mode requested.");
			return false;
		}

		m_window_flags = flags;

		m_specific->m_idcmp_hook = reinterpret_cast<Hook*>(
			IExec->AllocSysObjectTags(ASOT_HOOK,
				ASOHOOK_Entry, handle_idcmp,
				ASOHOOK_Data, this,
				TAG_END));
		if ( m_specific->m_idcmp_hook == 0 )
		{
			return false;
		}

		m_specific->m_window_obj = IIntuition->NewObject(WindowClass, 0,
				WA_Title, m_caption,
				WA_AutoAdjustDClip, TRUE,
				WA_InnerWidth, width,
				WA_InnerHeight, height,
				WA_Activate, TRUE,
				WA_SmartRefresh, TRUE,
				WA_NoCareRefresh, TRUE,
				WA_CloseGadget, TRUE,
				WA_DepthGadget, TRUE,
				WA_SizeGadget, (flags & agg::window_resize) ? TRUE : FALSE,
				WA_DragBar, TRUE,
				WA_AutoAdjust, TRUE,
				WA_ReportMouse, TRUE,
				WA_RMBTrap, TRUE,
				WA_MouseQueue, 1,
				WA_IDCMP,
					IDCMP_NEWSIZE |
					IDCMP_MOUSEBUTTONS |
					IDCMP_MOUSEMOVE |
					IDCMP_RAWKEY |
					IDCMP_INTUITICKS,
				WINDOW_IDCMPHook, m_specific->m_idcmp_hook,
				WINDOW_IDCMPHookBits,
					IDCMP_MOUSEBUTTONS |
					IDCMP_MOUSEMOVE |
					IDCMP_RAWKEY,
				TAG_END);
		if ( m_specific->m_window_obj == 0 )
		{
			return false;
		}

		Object* obj = reinterpret_cast<Object*>(m_specific->m_window_obj);
		m_specific->m_window =
			reinterpret_cast<Window*>(IIntuition->IDoMethod(obj, WM_OPEN));
		if ( m_specific->m_window == 0 )
		{
			return false;
		}

		RGBFTYPE ftype = static_cast<RGBFTYPE>(IP96->p96GetBitMapAttr(
			m_specific->m_window->RPort->BitMap, P96BMA_RGBFORMAT));

		switch ( ftype )
		{
		case RGBFB_A8R8G8B8:
		case RGBFB_B8G8R8A8:
		case RGBFB_R5G6B5PC:
			break;
		default:
			message("Unsupported screen mode.\n");
			return false;
		}

		if ( !m_specific->make_bitmap() )
		{
			return false;
		}

		m_initial_width = width;
		m_initial_height = height;

		on_init();
		on_resize(width, height);
		force_redraw();

		return true;
	}

	//------------------------------------------------------------------------
	int platform_support::run()
	{
		uint32 window_mask = 0;
		IIntuition->GetAttr(WINDOW_SigMask, m_specific->m_window_obj,
			&window_mask);
		uint32 wait_mask = window_mask | SIGBREAKF_CTRL_C;

		bool done = false;

		while ( !done )
		{
			uint32 sig_mask = IExec->Wait(wait_mask);
			if ( sig_mask & SIGBREAKF_CTRL_C )
			{
				done = true;
			}
			else
			{
				done = m_specific->handle_input();
			}
		}

		return 0;
	}

	//------------------------------------------------------------------------
	const char* platform_support::img_ext() const
	{
		return ".bmp";
	}

	//------------------------------------------------------------------------
	const char* platform_support::full_file_name(const char* file_name)
	{
		return file_name;
	}

	//------------------------------------------------------------------------
	bool platform_support::load_img(unsigned idx, const char* file)
	{
		if ( idx < max_images )
		{
			static char fn[1024];
			std::strncpy(fn, file, 1024);
			int len = std::strlen(fn);
			if ( len < 4 || std::strcmp(fn + len - 4, ".bmp") != 0 )
			{
				std::strncat(fn, ".bmp", 1024);
			}

			return m_specific->load_img(fn, idx, &m_rbuf_img[idx]);
		}

		return false;
	}

	//------------------------------------------------------------------------
	bool platform_support::save_img(unsigned idx, const char* file)
	{
		message("Not supported");
		return false;
	}

	//------------------------------------------------------------------------
	bool platform_support::create_img(unsigned idx, unsigned width,
		unsigned height)
	{
		if ( idx < max_images )
		{
			if ( width == 0 )
			{
				width = m_specific->m_width;
			}

			if ( height == 0 )
			{
				height = m_specific->m_height;
			}

			return m_specific->create_img(idx, &m_rbuf_img[idx], width,
				height);
		}

		return false;
	}

	//------------------------------------------------------------------------
	void platform_support::force_redraw()
	{
		on_draw();
		update_window();
	}

	//------------------------------------------------------------------------
	void platform_support::update_window()
	{
		// Note this function does automatic color conversion.
		IGraphics->BltBitMapRastPort(m_specific->m_bitmap, 0, 0,
			m_specific->m_window->RPort, m_specific->m_window->BorderLeft,
			m_specific->m_window->BorderTop, m_specific->m_width,
			m_specific->m_height, ABC|ABNC);
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
	void handle_idcmp(Hook* hook, APTR obj, IntuiMessage* msg)
	{
		platform_support* app =
			reinterpret_cast<platform_support*>(hook->h_Data);
		Window* window = app->m_specific->m_window;

		int16 x = msg->MouseX - window->BorderLeft;

		int16 y = 0;
		if ( app->flip_y() )
		{
			y = window->Height - window->BorderBottom - msg->MouseY;
		}
		else
		{
			y = msg->MouseY - window->BorderTop;
		}

		switch ( msg->Class )
		{
		case IDCMP_MOUSEBUTTONS:
			if ( msg->Code & IECODE_UP_PREFIX )
			{
				if ( msg->Code == SELECTUP )
				{
					app->m_specific->m_input_flags = mouse_left;
					app->m_specific->m_dragging = false;
				}
				else if ( msg->Code == MENUUP )
				{
					app->m_specific->m_input_flags = mouse_right;
					app->m_specific->m_dragging = false;
				}
				else
				{
					return;
				}


				if ( app->m_ctrls.on_mouse_button_up(x, y) )
				{
					app->on_ctrl_change();
					app->force_redraw();
				}

				app->on_mouse_button_up(x, y, app->m_specific->m_input_flags);
			}
			else
			{
				if ( msg->Code == SELECTDOWN )
				{
					app->m_specific->m_input_flags = mouse_left;
					app->m_specific->m_dragging = true;
				}
				else if ( msg->Code == MENUDOWN )
				{
					app->m_specific->m_input_flags = mouse_right;
					app->m_specific->m_dragging = true;
				}
				else
				{
					return;
				}

				app->m_ctrls.set_cur(x, y);
				if ( app->m_ctrls.on_mouse_button_down(x, y) )
				{
					app->on_ctrl_change();
					app->force_redraw();
				}
				else
				{
					if ( app->m_ctrls.in_rect(x, y) )
					{
						if ( app->m_ctrls.set_cur(x, y) )
						{
							app->on_ctrl_change();
							app->force_redraw();
						}
					}
					else
					{
						app->on_mouse_button_down(x, y,
							app->m_specific->m_input_flags);
					}
				}
			}
			break;
		case IDCMP_MOUSEMOVE:
			if ( app->m_specific->m_dragging )  {
				if ( app->m_ctrls.on_mouse_move(x, y,
					 app->m_specific->m_input_flags & mouse_left) != 0 )
				{
					app->on_ctrl_change();
					app->force_redraw();
				}
				else
				{
					if ( !app->m_ctrls.in_rect(x, y) )
					{
						app->on_mouse_move(x, y,
							app->m_specific->m_input_flags);
					}
				}
			}
			break;
		case IDCMP_RAWKEY:
		{
			static InputEvent ie = { 0 };
			ie.ie_Class = IECLASS_RAWKEY;
			ie.ie_Code = msg->Code;
			ie.ie_Qualifier = msg->Qualifier;

			static const unsigned BUF_SIZE = 16;
			static char key_buf[BUF_SIZE];
			int16 num_chars = IKeymap->MapRawKey(&ie, key_buf, BUF_SIZE, 0);

			uint32 code = 0x00000000;
			switch ( num_chars )
			{
			case 1:
				code = key_buf[0];
				break;
			case 2:
				code = key_buf[0]<<8 | key_buf[1];
				break;
			case 3:
				code = key_buf[0]<<16 | key_buf[1]<<8 | key_buf[2];
				break;
			}

			uint16 key_code = 0;

			if ( num_chars == 1 )
			{
				if ( code >= IECODE_ASCII_FIRST && code <= IECODE_ASCII_LAST )
				{
					key_code = code;
				}
			}

			if ( key_code == 0 )
			{
				switch ( code )
				{
				case 0x00000008: key_code = key_backspace;	break;
				case 0x00000009: key_code = key_tab;		break;
				case 0x0000000D: key_code = key_return;		break;
				case 0x0000001B: key_code = key_escape;		break;
				case 0x0000007F: key_code = key_delete;		break;
				case 0x00009B41:
				case 0x00009B54: key_code = key_up;			break;
				case 0x00009B42:
				case 0x00009B53: key_code = key_down;		break;
				case 0x00009B43:
				case 0x009B2040: key_code = key_right;		break;
				case 0x00009B44:
				case 0x009B2041: key_code = key_left;		break;
				case 0x009B307E: key_code = key_f1;			break;
				case 0x009B317E: key_code = key_f2;			break;
				case 0x009B327E: key_code = key_f3;			break;
				case 0x009B337E: key_code = key_f4;			break;
				case 0x009B347E: key_code = key_f5;			break;
				case 0x009B357E: key_code = key_f6;			break;
				case 0x009B367E: key_code = key_f7;			break;
				case 0x009B377E: key_code = key_f8;			break;
				case 0x009B387E: key_code = key_f9;			break;
				case 0x009B397E: key_code = key_f10;		break;
				case 0x009B3F7E: key_code = key_scrollock;	break;
				}
			}

			if ( ie.ie_Code & IECODE_UP_PREFIX )
			{
				if ( app->m_specific->m_last_key != 0 )
				{
					bool left = (key_code == key_left) ? true : false;
					bool right = (key_code == key_right) ? true : false;
					bool down = (key_code == key_down) ? true : false;
					bool up = (key_code == key_up) ? true : false;

					if ( app->m_ctrls.on_arrow_keys(left, right, down, up) )
					{
						app->on_ctrl_change();
						app->force_redraw();
					}
					else
					{
						app->on_key(x, y, app->m_specific->m_last_key, 0);
					}

					app->m_specific->m_last_key = 0;
				}
			}
			else
			{
				app->m_specific->m_last_key = key_code;
			}
			break;
		}
		default:
			break;
		}
	}
}

//----------------------------------------------------------------------------
int agg_main(int argc, char* argv[]);
bool open_libs();
void close_libs();

//----------------------------------------------------------------------------
bool open_libs()
{
	DataTypesBase = IExec->OpenLibrary("datatypes.library", 51);
	GraphicsBase = IExec->OpenLibrary("graphics.library", 51);
	IntuitionBase = IExec->OpenLibrary("intuition.library", 51);
	KeymapBase = IExec->OpenLibrary("keymap.library", 51);
	P96Base = IExec->OpenLibrary("Picasso96API.library", 2);

	IDataTypes = reinterpret_cast<DataTypesIFace*>(
		IExec->GetInterface(DataTypesBase, "main", 1, 0));
	IGraphics = reinterpret_cast<GraphicsIFace*>(
		IExec->GetInterface(GraphicsBase, "main", 1, 0));
	IIntuition = reinterpret_cast<IntuitionIFace*>(
		IExec->GetInterface(IntuitionBase, "main", 1, 0));
	IKeymap = reinterpret_cast<KeymapIFace*>(
		IExec->GetInterface(KeymapBase, "main", 1, 0));
	IP96 = reinterpret_cast<P96IFace*>(
		IExec->GetInterface(P96Base, "main", 1, 0));

	if ( IDataTypes == 0 ||
		 IGraphics == 0 ||
		 IIntuition == 0 ||
		 IKeymap == 0 ||
		 IP96 == 0 )
	{
		close_libs();
		return false;
	}
	else
	{
		return true;
	}
}

//----------------------------------------------------------------------------
void close_libs()
{
	IExec->DropInterface(reinterpret_cast<Interface*>(IP96));
	IExec->DropInterface(reinterpret_cast<Interface*>(IKeymap));
	IExec->DropInterface(reinterpret_cast<Interface*>(IIntuition));
	IExec->DropInterface(reinterpret_cast<Interface*>(IGraphics));
	IExec->DropInterface(reinterpret_cast<Interface*>(IDataTypes));

	IExec->CloseLibrary(P96Base);
	IExec->CloseLibrary(KeymapBase);
	IExec->CloseLibrary(IntuitionBase);
	IExec->CloseLibrary(GraphicsBase);
	IExec->CloseLibrary(DataTypesBase);
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	if ( !open_libs() )  {
		IDOS->Printf("Can't open libraries.\n");
		return -1;
	}

	ClassLibrary* requester =
		IIntuition->OpenClass("requester.class", 51, &RequesterClass);
	ClassLibrary* window =
		IIntuition->OpenClass("window.class", 51, &WindowClass);
	if ( requester == 0 || window == 0 )
	{
		IDOS->Printf("Can't open classes.\n");
		IIntuition->CloseClass(requester);
		IIntuition->CloseClass(window);
		close_libs();
		return -1;
	}

	int rc = agg_main(argc, argv);

	IIntuition->CloseClass(window);
	IIntuition->CloseClass(requester);
	close_libs();

	return rc;
}

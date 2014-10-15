//----------------------------------------------------------------------------
// AGG Contribution Pack - Gradients 1 (AGG CP - Gradients 1)
// http://milan.marusinec.sk/aggcp
//
// For Anti-Grain Geometry - Version 2.4 
// http://www.antigrain.org
//
// Contribution Created By:
//  Milan Marusinec alias Milano
//  milan@marusinec.sk
//  Copyright (c) 2007-2008
//
// Permission to copy, use, modify, sell and distribute this software
// is granted provided this copyright notice appears in all copies.
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
// [History] -----------------------------------------------------------------
//
// 03.02.2008-Milano: Ported from Object Pascal code of AggPas
//
#ifndef AGG_SPAN_GRADIENT_IMAGE_INCLUDED
#define AGG_SPAN_GRADIENT_IMAGE_INCLUDED

#include "agg_basics.h"
#include "agg_span_gradient.h"
#include "agg_color_rgba.h"
#include "agg_rendering_buffer.h"
#include "agg_pixfmt_rgba.h"

namespace agg
{

	//==========================================================one_color_function
	template<class ColorT> class one_color_function
	{
	public:
		typedef ColorT color_type;

		color_type m_color;

		one_color_function() :
			m_color()
		{
		}

		static unsigned size() { return 1; }

		const color_type& operator [] (unsigned i) const 
		{
			return m_color;
		}

		color_type* operator [] (unsigned i)
		{
			return &m_color;
		}	        
	};

	//==========================================================gradient_image
	template<class ColorT> class gradient_image
	{
	private:
		//------------ fields
		typedef ColorT color_type;
		typedef agg::pixfmt_rgba32 pixfmt_type;

		agg::rgba8* m_buffer;

		int m_alocdx;
		int m_alocdy;
		int m_width;
		int m_height;

		color_type* m_color;

		one_color_function<color_type> m_color_function;

	public:
		gradient_image() :
			m_color_function(),
			m_buffer(NULL),
			m_alocdx(0),
			m_alocdy(0),
			m_width(0),
			m_height(0)
		{
			m_color = m_color_function[0 ];
		}

		~gradient_image()
		{
			if (m_buffer) { delete [] m_buffer; }
		}

		void* image_create(int width, int height )
		{
			void* result = NULL;

			if (width > m_alocdx || height > m_alocdy)
			{
				if (m_buffer) { delete [] m_buffer; }

				m_buffer = NULL;
				m_buffer = new agg::rgba8[width * height];

				if (m_buffer)
				{
					m_alocdx = width;
					m_alocdy = height;
				}
				else
				{
					m_alocdx = 0;
					m_alocdy = 0;
				};
			};

			if (m_buffer)
			{
				m_width  = width;
				m_height = height;

				for (int rows = 0; rows < height; rows++)
				{
					agg::rgba8* row = &m_buffer[rows * m_alocdx ];
					memset(row ,0 ,m_width * 4 );
				};

				result = m_buffer;
			};
			return result;
		}

		void* image_buffer() { return m_buffer; }
		int   image_width()  { return m_width; }
		int   image_height() { return m_height; }
		int   image_stride() { return m_alocdx * 4; }

		int calculate(int x, int y, int d) const
		{
			if (m_buffer)
			{
				int px = x >> agg::gradient_subpixel_shift;
				int py = y >> agg::gradient_subpixel_shift;

				px %= m_width;

				if (px < 0)
				{
					px += m_width;
				}

				py %= m_height;

				if (py < 0 )
				{
					py += m_height;
				}

				rgba8* pixel = &m_buffer[py * m_alocdx + px ];

				m_color->r = pixel->r;
				m_color->g = pixel->g;
				m_color->b = pixel->b;
				m_color->a = pixel->a;

			}
			else
			{
				m_color->r = 0;
				m_color->g = 0;
				m_color->b = 0;
				m_color->a = 0;
			}
			return 0;
		}

		const one_color_function<color_type>& color_function() const
		{
			return m_color_function;
		}

	};
	
}

#endif

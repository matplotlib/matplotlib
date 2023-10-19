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
// 02.02.2008-Milano: Ported from Object Pascal code of AggPas
//
#ifndef AGG_SPAN_GRADIENT_CONTOUR_INCLUDED
#define AGG_SPAN_GRADIENT_CONTOUR_INCLUDED

#include "agg_basics.h"
#include "agg_trans_affine.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_gray.h"
#include "agg_conv_transform.h"
#include "agg_conv_curve.h"
#include "agg_bounding_rect.h"
#include "agg_renderer_base.h"
#include "agg_renderer_primitives.h"
#include "agg_rasterizer_outline.h"
#include "agg_span_gradient.h"

#define infinity 1E20

namespace agg
{

	//==========================================================gradient_contour
	class gradient_contour
	{
	private:
		int8u* m_buffer;
		int	   m_width;
		int    m_height;
		int    m_frame;

		double m_d1;
		double m_d2;

	public:
		gradient_contour() :
			m_buffer(NULL),
			m_width(0),
			m_height(0),
			m_frame(10),
			m_d1(0),
			m_d2(100)
		{
		}

		gradient_contour(double d1, double d2) :
			m_buffer(NULL),
			m_width(0),
			m_height(0),
			m_frame(10),
			m_d1(d1),
			m_d2(d2)
		{
		}

		~gradient_contour()
		{
			if (m_buffer)
			{
				delete [] m_buffer;
			}
		}

		int8u* contour_create(path_storage* ps );

		int    contour_width() { return m_width; }
		int    contour_height() { return m_height; }

		void   d1(double d ) { m_d1 = d; }
		void   d2(double d ) { m_d2 = d; }

		void   frame(int f ) { m_frame = f; }
		int    frame() { return m_frame; }

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

				return iround(m_buffer[py * m_width + px ] * (m_d2 / 256 ) + m_d1 ) << gradient_subpixel_shift;

			}
			else
			{
				return 0;
			}
		}

	};

	static AGG_INLINE int square(int x ) { return x * x; }

	// DT algorithm by: Pedro Felzenszwalb
	void dt(float* spanf, float* spang, float* spanr, int* spann ,int length )
	{
		int k = 0;
		float s;

		spann[0 ] = 0;
		spang[0 ] = float(-infinity );
		spang[1 ] = float(+infinity );

		for (int q = 1; q <= length - 1; q++)
		{
			s = ((spanf[q ] + square(q ) ) - (spanf[spann[k ] ] + square(spann[k ] ) ) ) / (2 * q - 2 * spann[k ] );

			while (s <= spang[k ])
			{
				k--;
				s = ((spanf[q ] + square(q ) ) - (spanf[spann[k ] ] + square(spann[k ] ) ) ) / (2 * q - 2 * spann[k ] );
			}

			k++;
			spann[k ] = q;
			spang[k ] = s;
			spang[k + 1 ] = float(+infinity);

		}

		k = 0;

		for (int q = 0; q <= length - 1; q++)
		{
			while (spang[k + 1 ] < q )
			{
				k++;
			}

			spanr[q ] = square(q - spann[k ] ) + spanf[spann[k ] ];
		}
	}

	// DT algorithm by: Pedro Felzenszwalb
	int8u* gradient_contour::contour_create(path_storage* ps )
	{
		int8u* result = NULL;

		if (ps)
		{
		   // I. Render Black And White NonAA Stroke of the Path
		   // Path Bounding Box + Some Frame Space Around [configurable]
			agg::conv_curve<agg::path_storage> conv(*ps);

			double x1, y1, x2, y2;

			if (agg::bounding_rect_single(conv ,0 ,&x1 ,&y1 ,&x2 ,&y2 ))
			{
			   // Create BW Rendering Surface
				int width  = int(ceil(x2 - x1 ) ) + m_frame * 2 + 1;
				int height = int(ceil(y2 - y1 ) ) + m_frame * 2 + 1;

				int8u* buffer = new int8u[width * height];

				if (buffer)
				{
					memset(buffer ,255 ,width * height );

				   // Setup VG Engine & Render
					agg::rendering_buffer rb;
					rb.attach(buffer ,width ,height ,width );

					agg::pixfmt_gray8 pf(rb);
					agg::renderer_base<agg::pixfmt_gray8> renb(pf );

					agg::renderer_primitives<agg::renderer_base<agg::pixfmt_gray8> > prim(renb );
					agg::rasterizer_outline<renderer_primitives<agg::renderer_base<agg::pixfmt_gray8> > > ras(prim );

					agg::trans_affine mtx;
					mtx *= agg::trans_affine_translation(-x1 + m_frame, -y1 + m_frame );

					agg::conv_transform<agg::conv_curve<agg::path_storage> > trans(conv ,mtx );

					prim.line_color(agg::rgba8(0 ,0 ,0 ,255 ) );
					ras.add_path(trans );

				   // II. Distance Transform
				   // Create Float Buffer + 0 vs. infinity (1e20) assignment
					float* image = new float[width * height];

					if (image)
					{
						for (int y = 0, l = 0; y < height; y++ )
						{
							for (int x = 0; x < width; x++, l++ )
							{
								if (buffer[l ] == 0)
								{
									image[l ] = 0.0;
								}
								else
								{
									image[l ] = float(infinity );
								}
							}

						}

					   // DT of 2d
					   // SubBuff<float> max width,height
						int length = width;

						if (height > length)
						{
							length = height;
						}

						float* spanf = new float[length];
						float* spang = new float[length + 1];
						float* spanr = new float[length];
						int* spann = new int[length];

						if ((spanf) && (spang) && (spanr) && (spann))
						{
						   // Transform along columns
							for (int x = 0; x < width; x++ )
							{
								for (int y = 0; y < height; y++ )
								{
									spanf[y] = image[y * width + x];
								}

							   // DT of 1d
								dt(spanf ,spang ,spanr ,spann ,height );

								for (int y = 0; y < height; y++ )
								{
									image[y * width + x] = spanr[y];
								}
							}

						   // Transform along rows
							for (int y = 0; y < height; y++ )
							{
								for (int x = 0; x < width; x++ )
								{
									spanf[x] = image[y * width + x];
								}

							   // DT of 1d
								dt(spanf ,spang ,spanr ,spann ,width );

								for (int x = 0; x < width; x++ )
								{
									image[y * width + x] = spanr[x];
								}
							}

						   // Take Square Roots, Min & Max
							float min = sqrt(image[0] );
							float max = min;

							for (int y = 0, l = 0; y < height; y++ )
							{
								for (int x = 0; x < width; x++, l++ )
								{
									image[l] = sqrt(image[l]);

									if (min > image[l])
									{
										min = image[l];
									}

									if (max < image[l])
									{
										max = image[l];
									}

								}
							}

						   // III. Convert To Grayscale
							if (min == max)
							{
								memset(buffer ,0 ,width * height );
							}
							else
							{
								float scale = 255 / (max - min );

								for (int y = 0, l = 0; y < height; y++ )
								{
									for (int x = 0; x < width; x++ ,l++ )
									{
										buffer[l] = int8u(int((image[l] - min ) * scale ));
									}
								}
							}

						   // OK
							if (m_buffer)
							{
								delete [] m_buffer;
							}

							m_buffer = buffer;
							m_width  = width;
							m_height = height;

							buffer = NULL;
							result = m_buffer;

						}

						if (spanf) { delete [] spanf; }
						if (spang) { delete [] spang; }
						if (spanr) { delete [] spanr; }
						if (spann) { delete [] spann; }

						delete [] image;

					}
				}

				if (buffer)
				{
					delete [] buffer;
				}

			}

		}
		return result;
	}

}

#endif

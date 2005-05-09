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
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Filtering class image_filter_lut implemantation
//
//----------------------------------------------------------------------------


#include "agg_image_filters.h"


namespace agg
{

    //--------------------------------------------------------------------
    image_filter_lut::~image_filter_lut()
    {
        delete [] m_weight_array;
    }


    //--------------------------------------------------------------------
    image_filter_lut::image_filter_lut() :
        m_weight_array(0),
        m_max_size(0)
    {}

    //--------------------------------------------------------------------
    void image_filter_lut::realloc(double radius)
    {
        m_radius = radius;
        m_diameter = unsigned(ceil(radius)) * 2;
        m_start = -int(m_diameter / 2 - 1);
        unsigned size = m_diameter << image_subpixel_shift;
        if(size > m_max_size)
        {
            delete [] m_weight_array;
            m_weight_array = new int16 [size];
            m_max_size = size;
        }
    }



    //--------------------------------------------------------------------
    // This function normalizes integer values and corrects the rounding 
    // errors. It doesn't do anything with the source floating point values
    // (m_weight_array_dbl), it corrects only integers according to the rule 
    // of 1.0 which means that any sum of pixel weights must be equal to 1.0.
    // So, the filter function must produce a graph of the proper shape.
    //--------------------------------------------------------------------
    void image_filter_lut::normalize()
    {
        unsigned i;
        int flip = 1;

        for(i = 0; i < image_subpixel_size; i++)
        {
            for(;;)
            {
                int sum = 0;
                unsigned j;
                for(j = 0; j < m_diameter; j++)
                {
                    sum += m_weight_array[j * image_subpixel_size + i];
                }

                if(sum == image_filter_size) break;

                double k = double(image_filter_size) / double(sum);
                sum = 0;
                for(j = 0; j < m_diameter; j++)
                {
                    sum += m_weight_array[j * image_subpixel_size + i] = 
                        int(m_weight_array[j * image_subpixel_size + i] * k);
                }

                sum -= image_filter_size;
                int inc = (sum > 0) ? -1 : 1;

                for(j = 0; j < m_diameter && sum; j++)
                {
                    flip ^= 1;
                    unsigned idx = flip ? m_diameter/2 + j/2 : m_diameter/2 - j/2;
                    int v = m_weight_array[idx * image_subpixel_size + i];
                    if(v < image_filter_size)
                    {
                        m_weight_array[idx * image_subpixel_size + i] += inc;
                        sum += inc;
                    }
                }
            }
        }

        unsigned pivot = m_diameter << (image_subpixel_shift - 1);

        for(i = 0; i < pivot; i++)
        {
            m_weight_array[pivot + i] = m_weight_array[pivot - i];
        }
        unsigned end = (diameter() << image_subpixel_shift) - 1;
        m_weight_array[0] = m_weight_array[end];
    }


}


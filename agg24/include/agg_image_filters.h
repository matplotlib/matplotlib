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
// Image transformation filters,
// Filtering classes (image_filter_lut, image_filter),
// Basic filter shape classes
//----------------------------------------------------------------------------
#ifndef AGG_IMAGE_FILTERS_INCLUDED
#define AGG_IMAGE_FILTERS_INCLUDED

#include "agg_array.h"
#include "agg_math.h"

namespace agg
{

    // See Implementation agg_image_filters.cpp 

    enum image_filter_scale_e
    {
        image_filter_shift = 14,                      //----image_filter_shift
        image_filter_scale = 1 << image_filter_shift, //----image_filter_scale 
        image_filter_mask  = image_filter_scale - 1   //----image_filter_mask 
    };

    enum image_subpixel_scale_e
    {
        image_subpixel_shift = 8,                         //----image_subpixel_shift
        image_subpixel_scale = 1 << image_subpixel_shift, //----image_subpixel_scale 
        image_subpixel_mask  = image_subpixel_scale - 1   //----image_subpixel_mask 
    };


    //-----------------------------------------------------image_filter_lut
    class image_filter_lut
    {
    public:
        template<class FilterF> void calculate(const FilterF& filter,
                                               bool normalization=true)
        {
            double r = filter.radius();
            realloc_lut(r);
            unsigned i;
            unsigned pivot = diameter() << (image_subpixel_shift - 1);
            for(i = 0; i < pivot; i++)
            {
                double x = double(i) / double(image_subpixel_scale);
                double y = filter.calc_weight(x);
                m_weight_array[pivot + i] = 
                m_weight_array[pivot - i] = (int16)iround(y * image_filter_scale);
            }
            unsigned end = (diameter() << image_subpixel_shift) - 1;
            m_weight_array[0] = m_weight_array[end];
            if(normalization) 
            {
                normalize();
            }
        }

        image_filter_lut() : m_radius(0), m_diameter(0), m_start(0) {}

        template<class FilterF> image_filter_lut(const FilterF& filter, 
                                                 bool normalization=true)
        {
            calculate(filter, normalization);
        }

        double       radius()       const { return m_radius;   }
        unsigned     diameter()     const { return m_diameter; }
        int          start()        const { return m_start;    }
        const int16* weight_array() const { return &m_weight_array[0]; }
        void         normalize();

    private:
        void realloc_lut(double radius);
        image_filter_lut(const image_filter_lut&);
        const image_filter_lut& operator = (const image_filter_lut&);

        double           m_radius;
        unsigned         m_diameter;
        int              m_start;
        pod_array<int16> m_weight_array;
    };



    //--------------------------------------------------------image_filter
    template<class FilterF> class image_filter : public image_filter_lut
    {
    public:
        image_filter()
        {
            calculate(m_filter_function);
        }
    private:
        FilterF m_filter_function;
    };


    //-----------------------------------------------image_filter_bilinear
    struct image_filter_bilinear
    {
        static double radius() { return 1.0; }
        static double calc_weight(double x)
        {
            return 1.0 - x;
        }
    };


    //-----------------------------------------------image_filter_hanning
    struct image_filter_hanning
    {
        static double radius() { return 1.0; }
        static double calc_weight(double x)
        {
            return 0.5 + 0.5 * cos(pi * x);
        }
    };


    //-----------------------------------------------image_filter_hamming
    struct image_filter_hamming
    {
        static double radius() { return 1.0; }
        static double calc_weight(double x)
        {
            return 0.54 + 0.46 * cos(pi * x);
        }
    };

    //-----------------------------------------------image_filter_hermite
    struct image_filter_hermite
    {
        static double radius() { return 1.0; }
        static double calc_weight(double x)
        {
            return (2.0 * x - 3.0) * x * x + 1.0;
        }
    };
   
    //------------------------------------------------image_filter_quadric
    struct image_filter_quadric
    {
        static double radius() { return 1.5; }
        static double calc_weight(double x)
        {
            double t;
            if(x <  0.5) return 0.75 - x * x;
            if(x <  1.5) {t = x - 1.5; return 0.5 * t * t;}
            return 0.0;
        }
    };

    //------------------------------------------------image_filter_bicubic
    class image_filter_bicubic
    {
        static double pow3(double x)
        {
            return (x <= 0.0) ? 0.0 : x * x * x;
        }

    public:
        static double radius() { return 2.0; }
        static double calc_weight(double x)
        {
            return
                (1.0/6.0) * 
                (pow3(x + 2) - 4 * pow3(x + 1) + 6 * pow3(x) - 4 * pow3(x - 1));
        }
    };

    //-------------------------------------------------image_filter_kaiser
    class image_filter_kaiser
    {
        double a;
        double i0a;
        double epsilon;

    public:
        image_filter_kaiser(double b = 6.33) :
            a(b), epsilon(1e-12)
        {
            i0a = 1.0 / bessel_i0(b);
        }

        static double radius() { return 1.0; }
        double calc_weight(double x) const
        {
            return bessel_i0(a * sqrt(1. - x * x)) * i0a;
        }

    private:
        double bessel_i0(double x) const
        {
            int i;
            double sum, y, t;

            sum = 1.;
            y = x * x / 4.;
            t = y;
        
            for(i = 2; t > epsilon; i++)
            {
                sum += t;
                t *= (double)y / (i * i);
            }
            return sum;
        }
    };

    //----------------------------------------------image_filter_catrom
    struct image_filter_catrom
    {
        static double radius() { return 2.0; }
        static double calc_weight(double x)
        {
            if(x <  1.0) return 0.5 * (2.0 + x * x * (-5.0 + x * 3.0));
            if(x <  2.0) return 0.5 * (4.0 + x * (-8.0 + x * (5.0 - x)));
            return 0.;
        }
    };

    //---------------------------------------------image_filter_mitchell
    class image_filter_mitchell
    {
        double p0, p2, p3;
        double q0, q1, q2, q3;

    public:
        image_filter_mitchell(double b = 1.0/3.0, double c = 1.0/3.0) :
            p0((6.0 - 2.0 * b) / 6.0),
            p2((-18.0 + 12.0 * b + 6.0 * c) / 6.0),
            p3((12.0 - 9.0 * b - 6.0 * c) / 6.0),
            q0((8.0 * b + 24.0 * c) / 6.0),
            q1((-12.0 * b - 48.0 * c) / 6.0),
            q2((6.0 * b + 30.0 * c) / 6.0),
            q3((-b - 6.0 * c) / 6.0)
        {}

        static double radius() { return 2.0; }
        double calc_weight(double x) const
        {
            if(x < 1.0) return p0 + x * x * (p2 + x * p3);
            if(x < 2.0) return q0 + x * (q1 + x * (q2 + x * q3));
            return 0.0;
        }
    };


    //----------------------------------------------image_filter_spline16
    struct image_filter_spline16
    {
        static double radius() { return 2.0; }
        static double calc_weight(double x)
        {
            if(x < 1.0)
            {
                return ((x - 9.0/5.0 ) * x - 1.0/5.0 ) * x + 1.0;
            }
            return ((-1.0/3.0 * (x-1) + 4.0/5.0) * (x-1) - 7.0/15.0 ) * (x-1);
        }
    };


    //---------------------------------------------image_filter_spline36
    struct image_filter_spline36
    {
        static double radius() { return 3.0; }
        static double calc_weight(double x)
        {
           if(x < 1.0)
           {
              return ((13.0/11.0 * x - 453.0/209.0) * x - 3.0/209.0) * x + 1.0;
           }
           if(x < 2.0)
           {
              return ((-6.0/11.0 * (x-1) + 270.0/209.0) * (x-1) - 156.0/ 209.0) * (x-1);
           }
           return ((1.0/11.0 * (x-2) - 45.0/209.0) * (x-2) +  26.0/209.0) * (x-2);
        }
    };


    //----------------------------------------------image_filter_gaussian
    struct image_filter_gaussian
    {
        static double radius() { return 2.0; }
        static double calc_weight(double x) 
        {
            return exp(-2.0 * x * x) * sqrt(2.0 / pi);
        }
    };


    //------------------------------------------------image_filter_bessel
    struct image_filter_bessel
    {
        static double radius() { return 3.2383; } 
        static double calc_weight(double x)
        {
            return (x == 0.0) ? pi / 4.0 : besj(pi * x, 1) / (2.0 * x);
        }
    };


    //-------------------------------------------------image_filter_sinc
    class image_filter_sinc
    {
    public:
        image_filter_sinc(double r) : m_radius(r < 2.0 ? 2.0 : r) {}
        double radius() const { return m_radius; }
        double calc_weight(double x) const
        {
            if(x == 0.0) return 1.0;
            x *= pi;
            return sin(x) / x;
        }
    private:
        double m_radius;
    };


    //-----------------------------------------------image_filter_lanczos
    class image_filter_lanczos
    {
    public:
        image_filter_lanczos(double r) : m_radius(r < 2.0 ? 2.0 : r) {}
        double radius() const { return m_radius; }
        double calc_weight(double x) const
        {
           if(x == 0.0) return 1.0;
           if(x > m_radius) return 0.0;
           x *= pi;
           double xr = x / m_radius;
           return (sin(x) / x) * (sin(xr) / xr);
        }
    private:
        double m_radius;
    };


    //----------------------------------------------image_filter_blackman
    class image_filter_blackman
    {
    public:
        image_filter_blackman(double r) : m_radius(r < 2.0 ? 2.0 : r) {}
        double radius() const { return m_radius; }
        double calc_weight(double x) const
        {
           if(x == 0.0) return 1.0;
           if(x > m_radius) return 0.0;
           x *= pi;
           double xr = x / m_radius;
           return (sin(x) / x) * (0.42 + 0.5*cos(xr) + 0.08*cos(2*xr));
        }
    private:
        double m_radius;
    };

    //------------------------------------------------image_filter_sinc36
    class image_filter_sinc36 : public image_filter_sinc
    { public: image_filter_sinc36() : image_filter_sinc(3.0){} };

    //------------------------------------------------image_filter_sinc64
    class image_filter_sinc64 : public image_filter_sinc
    { public: image_filter_sinc64() : image_filter_sinc(4.0){} };

    //-----------------------------------------------image_filter_sinc100
    class image_filter_sinc100 : public image_filter_sinc
    { public: image_filter_sinc100() : image_filter_sinc(5.0){} };

    //-----------------------------------------------image_filter_sinc144
    class image_filter_sinc144 : public image_filter_sinc
    { public: image_filter_sinc144() : image_filter_sinc(6.0){} };

    //-----------------------------------------------image_filter_sinc196
    class image_filter_sinc196 : public image_filter_sinc
    { public: image_filter_sinc196() : image_filter_sinc(7.0){} };

    //-----------------------------------------------image_filter_sinc256
    class image_filter_sinc256 : public image_filter_sinc
    { public: image_filter_sinc256() : image_filter_sinc(8.0){} };

    //---------------------------------------------image_filter_lanczos36
    class image_filter_lanczos36 : public image_filter_lanczos
    { public: image_filter_lanczos36() : image_filter_lanczos(3.0){} };

    //---------------------------------------------image_filter_lanczos64
    class image_filter_lanczos64 : public image_filter_lanczos
    { public: image_filter_lanczos64() : image_filter_lanczos(4.0){} };

    //--------------------------------------------image_filter_lanczos100
    class image_filter_lanczos100 : public image_filter_lanczos
    { public: image_filter_lanczos100() : image_filter_lanczos(5.0){} };

    //--------------------------------------------image_filter_lanczos144
    class image_filter_lanczos144 : public image_filter_lanczos
    { public: image_filter_lanczos144() : image_filter_lanczos(6.0){} };

    //--------------------------------------------image_filter_lanczos196
    class image_filter_lanczos196 : public image_filter_lanczos
    { public: image_filter_lanczos196() : image_filter_lanczos(7.0){} };

    //--------------------------------------------image_filter_lanczos256
    class image_filter_lanczos256 : public image_filter_lanczos
    { public: image_filter_lanczos256() : image_filter_lanczos(8.0){} };

    //--------------------------------------------image_filter_blackman36
    class image_filter_blackman36 : public image_filter_blackman
    { public: image_filter_blackman36() : image_filter_blackman(3.0){} };

    //--------------------------------------------image_filter_blackman64
    class image_filter_blackman64 : public image_filter_blackman
    { public: image_filter_blackman64() : image_filter_blackman(4.0){} };

    //-------------------------------------------image_filter_blackman100
    class image_filter_blackman100 : public image_filter_blackman
    { public: image_filter_blackman100() : image_filter_blackman(5.0){} };

    //-------------------------------------------image_filter_blackman144
    class image_filter_blackman144 : public image_filter_blackman
    { public: image_filter_blackman144() : image_filter_blackman(6.0){} };

    //-------------------------------------------image_filter_blackman196
    class image_filter_blackman196 : public image_filter_blackman
    { public: image_filter_blackman196() : image_filter_blackman(7.0){} };

    //-------------------------------------------image_filter_blackman256
    class image_filter_blackman256 : public image_filter_blackman
    { public: image_filter_blackman256() : image_filter_blackman(8.0){} };


}

#endif

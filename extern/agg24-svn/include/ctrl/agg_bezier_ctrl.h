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
// classes bezier_ctrl_impl, bezier_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_BEZIER_CTRL_INCLUDED
#define AGG_BEZIER_CTRL_INCLUDED

#include "agg_math.h"
#include "agg_ellipse.h"
#include "agg_trans_affine.h"
#include "agg_color_rgba.h"
#include "agg_conv_stroke.h"
#include "agg_conv_curve.h"
#include "agg_polygon_ctrl.h"


namespace agg
{

    //--------------------------------------------------------bezier_ctrl_impl
    class bezier_ctrl_impl : public ctrl
    {
    public:
        bezier_ctrl_impl();

        void curve(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3,
                   double x4, double y4);
        curve4& curve();

        double x1() const { return m_poly.xn(0); }
        double y1() const { return m_poly.yn(0); }
        double x2() const { return m_poly.xn(1); }
        double y2() const { return m_poly.yn(1); }
        double x3() const { return m_poly.xn(2); }
        double y3() const { return m_poly.yn(2); }
        double x4() const { return m_poly.xn(3); }
        double y4() const { return m_poly.yn(3); }

        void x1(double x) { m_poly.xn(0) = x; }
        void y1(double y) { m_poly.yn(0) = y; }
        void x2(double x) { m_poly.xn(1) = x; }
        void y2(double y) { m_poly.yn(1) = y; }
        void x3(double x) { m_poly.xn(2) = x; }
        void y3(double y) { m_poly.yn(2) = y; }
        void x4(double x) { m_poly.xn(3) = x; }
        void y4(double y) { m_poly.yn(3) = y; }

        void   line_width(double w) { m_stroke.width(w); }
        double line_width() const   { return m_stroke.width(); }

        void   point_radius(double r) { m_poly.point_radius(r); }
        double point_radius() const   { return m_poly.point_radius(); }

        virtual bool in_rect(double x, double y) const;
        virtual bool on_mouse_button_down(double x, double y);
        virtual bool on_mouse_button_up(double x, double y);
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // Vertex source interface
        unsigned num_paths() { return 7; };
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);


    private:
        curve4              m_curve;
        ellipse             m_ellipse;
        conv_stroke<curve4> m_stroke;
        polygon_ctrl_impl   m_poly;
        unsigned            m_idx;
    };



    //----------------------------------------------------------bezier_ctrl
    template<class ColorT> class bezier_ctrl : public bezier_ctrl_impl
    {
    public:
        bezier_ctrl() :
            m_color(rgba(0.0, 0.0, 0.0))
        {
        }
          
        void line_color(const ColorT& c) { m_color = c; }
        const ColorT& color(unsigned i) const { return m_color; } 

    private:
        bezier_ctrl(const bezier_ctrl<ColorT>&);
        const bezier_ctrl<ColorT>& operator = (const bezier_ctrl<ColorT>&);

        ColorT m_color;
    };





    //--------------------------------------------------------curve3_ctrl_impl
    class curve3_ctrl_impl : public ctrl
    {
    public:
        curve3_ctrl_impl();

        void curve(double x1, double y1, 
                   double x2, double y2, 
                   double x3, double y3);
        curve3& curve();

        double x1() const { return m_poly.xn(0); }
        double y1() const { return m_poly.yn(0); }
        double x2() const { return m_poly.xn(1); }
        double y2() const { return m_poly.yn(1); }
        double x3() const { return m_poly.xn(2); }
        double y3() const { return m_poly.yn(2); }

        void x1(double x) { m_poly.xn(0) = x; }
        void y1(double y) { m_poly.yn(0) = y; }
        void x2(double x) { m_poly.xn(1) = x; }
        void y2(double y) { m_poly.yn(1) = y; }
        void x3(double x) { m_poly.xn(2) = x; }
        void y3(double y) { m_poly.yn(2) = y; }

        void   line_width(double w) { m_stroke.width(w); }
        double line_width() const   { return m_stroke.width(); }

        void   point_radius(double r) { m_poly.point_radius(r); }
        double point_radius() const   { return m_poly.point_radius(); }

        virtual bool in_rect(double x, double y) const;
        virtual bool on_mouse_button_down(double x, double y);
        virtual bool on_mouse_button_up(double x, double y);
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // Vertex source interface
        unsigned num_paths() { return 6; };
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);


    private:
        curve3              m_curve;
        ellipse             m_ellipse;
        conv_stroke<curve3> m_stroke;
        polygon_ctrl_impl   m_poly;
        unsigned            m_idx;
    };



    //----------------------------------------------------------curve3_ctrl
    template<class ColorT> class curve3_ctrl : public curve3_ctrl_impl
    {
    public:
        curve3_ctrl() :
            m_color(rgba(0.0, 0.0, 0.0))
        {
        }
          
        void line_color(const ColorT& c) { m_color = c; }
        const ColorT& color(unsigned i) const { return m_color; } 

    private:
        curve3_ctrl(const curve3_ctrl<ColorT>&);
        const curve3_ctrl<ColorT>& operator = (const curve3_ctrl<ColorT>&);

        ColorT m_color;
    };




}



#endif


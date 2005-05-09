#include <stdio.h>
#include <stdlib.h>
#include "agg_basics.h"
#include "agg_rendering_buffer.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_scanline_p.h"
#include "agg_renderer_scanline.h"
#include "agg_pixfmt_rgba.h"
#include "platform/agg_platform_support.h"
#include "ctrl/agg_slider_ctrl.h"
#include "agg_svg_parser.h"

//#include "agg_gamma_lut.h"

enum { flip_y = false };


class the_application : public agg::platform_support
{
    agg::svg::path_renderer m_path;

    agg::slider_ctrl<agg::rgba8> m_expand;
    agg::slider_ctrl<agg::rgba8> m_gamma;
    agg::slider_ctrl<agg::rgba8> m_scale;
    agg::slider_ctrl<agg::rgba8> m_rotate;

    double m_min_x;
    double m_min_y;
    double m_max_x;
    double m_max_y;

    double m_x;
    double m_y;
    double m_dx;
    double m_dy;
    bool   m_drag_flag;

public:

    the_application(agg::pix_format_e format, bool flip_y) :
        agg::platform_support(format, flip_y),
        m_path(),
        m_expand(5,     5,    256-5, 11,    !flip_y),
        m_gamma (5,     5+15, 256-5, 11+15, !flip_y),
        m_scale (256+5, 5,    512-5, 11,    !flip_y),
        m_rotate(256+5, 5+15, 512-5, 11+15, !flip_y),
        m_min_x(0.0),
        m_min_y(0.0),
        m_max_x(0.0),
        m_max_y(0.0),
        m_x(0.0),
        m_y(0.0),
        m_dx(0.0),
        m_dy(0.0),
        m_drag_flag(false)
    {
        add_ctrl(m_expand);
        add_ctrl(m_gamma);
        add_ctrl(m_scale);
        add_ctrl(m_rotate);

        m_expand.label("Expand=%3.2f");
        m_expand.range(-1, 1.2);
        m_expand.value(0.0);

        m_gamma.label("Gamma=%3.2f");
        m_gamma.range(0.0, 3.0);
        m_gamma.value(1.0);

        m_scale.label("Scale=%3.2f");
        m_scale.range(0.2, 10.0);
        m_scale.value(1.0);

        m_rotate.label("Rotate=%3.2f");
        m_rotate.range(-180.0, 180.0);
        m_rotate.value(0.0);
    }

    void parse_svg(const char* fname)
    {
        agg::svg::parser p(m_path);
        p.parse(fname);
        m_path.arrange_orientations();
        m_path.bounding_rect(&m_min_x, &m_min_y, &m_max_x, &m_max_y);
        caption(p.title());
    }

    virtual void on_resize(int cx, int cy)
    {
    }

    virtual void on_draw()
    {
        typedef agg::pixfmt_bgra32 pixfmt;
        typedef agg::renderer_base<pixfmt> renderer_base;
        typedef agg::renderer_scanline_aa_solid<renderer_base> renderer_solid;

        pixfmt pixf(rbuf_window());
        renderer_base rb(pixf);
        renderer_solid ren(rb);

        rb.clear(agg::rgba(1,1,1));

        agg::rasterizer_scanline_aa<> ras;
        agg::scanline_p8 sl;
        agg::trans_affine mtx;

        ras.gamma(agg::gamma_power(m_gamma.value()));
        mtx *= agg::trans_affine_translation((m_min_x + m_max_x) * -0.5, (m_min_y + m_max_y) * -0.5);
        mtx *= agg::trans_affine_scaling(m_scale.value());
        mtx *= agg::trans_affine_rotation(agg::deg2rad(m_rotate.value()));
        mtx *= agg::trans_affine_translation((m_min_x + m_max_x) * 0.5 + m_x, (m_min_y + m_max_y) * 0.5 + m_y + 30);
        
        m_path.expand(m_expand.value());
        m_path.render(ras, sl, ren, mtx, rb.clip_box(), 1.0);

        ras.gamma(agg::gamma_none());
        agg::render_ctrl(ras, sl, ren, m_expand);
        agg::render_ctrl(ras, sl, ren, m_gamma);
        agg::render_ctrl(ras, sl, ren, m_scale);
        agg::render_ctrl(ras, sl, ren, m_rotate);


        //agg::gamma_lut<> gl(m_gamma.value());
        //unsigned x, y;
        //unsigned w = unsigned(width());
        //unsigned h = unsigned(height());
        //for(y = 0; y < h; y++)
        //{
        //    for(x = 0; x < w; x++)
        //    {
        //        agg::rgba8 c = rb.pixel(x, y);
        //        c.r = gl.inv(c.r);
        //        c.g = gl.inv(c.g);
        //        c.b = gl.inv(c.b);
        //        rb.copy_pixel(x, y, c);
        //    }
        //}
    }

    virtual void on_mouse_button_down(int x, int y, unsigned flags)
    {
        m_dx = x - m_x;
        m_dy = y - m_y;
        m_drag_flag = true;
    }

    virtual void on_mouse_move(int x, int y, unsigned flags)
    {
        if(flags == 0)
        {
            m_drag_flag = false;
        }

        if(m_drag_flag)
        {
            m_x = x - m_dx;
            m_y = y - m_dy;
            force_redraw();
        }
    }

    virtual void on_mouse_button_up(int x, int y, unsigned flags)
    {
        m_drag_flag = false;
    }

    virtual void on_key(int x, int y, unsigned key, unsigned flags)
    {
        if(key == ' ')
        {

            agg::trans_affine mtx;
            mtx *= agg::trans_affine_translation((m_min_x + m_max_x) * -0.5, (m_min_y + m_max_y) * -0.5);
            mtx *= agg::trans_affine_scaling(m_scale.value());
            mtx *= agg::trans_affine_rotation(agg::deg2rad(m_rotate.value()));
            mtx *= agg::trans_affine_translation((m_min_x + m_max_x) * 0.5, (m_min_y + m_max_y) * 0.5);
            mtx *= agg::trans_affine_translation(m_x, m_y);

            double m[6];
            mtx.store_to(m);

            char buf[128];
            sprintf(buf, "%3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f",
                         m[0], m[1], m[2], m[3], m[4], m[5]);

            message(buf);
            FILE* fd = fopen("transform.txt", "a");
            fprintf(fd, "%s\n", buf);
            fclose(fd);
        }
    }



};




int agg_main(int argc, char* argv[])
{
    the_application app(agg::pix_format_bgra32, flip_y);

    const char* fname = "../tiger.svg";
    if(argc <= 1)
    {
        FILE* fd = fopen(fname, "r");
        if(fd == 0)
        {
            app.message("Usage: svg_test <svg_file>");
            return 1;
        }
        fclose(fd);
    }
    else
    {
        fname = argv[1];
    }

    try
    {
        app.parse_svg(fname);
    }
    catch(agg::svg::exception& e)
    {
        app.message(e.msg());
        return 1;
    }

    if(app.init(512, 600, agg::window_resize))
    {
        return app.run();
    }
    return 1;
}




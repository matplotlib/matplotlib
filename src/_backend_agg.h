/* -*- mode: c++; c-basic-offset: 4 -*- */

/* _backend_agg.h - A rewrite of _backend_agg using PyCXX to handle
   ref counting, etc..
*/

#ifndef __BACKEND_AGG_H
#define __BACKEND_AGG_H
#include <utility>
#include "CXX/Extensions.hxx"

#include "agg_arrowhead.h"
#include "agg_basics.h"
#include "agg_bezier_arc.h"
#include "agg_color_rgba.h"
#include "agg_conv_concat.h"
#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_conv_dash.h"
#include "agg_conv_marker.h"
#include "agg_conv_marker_adaptor.h"
#include "agg_math_stroke.h"
#include "agg_conv_stroke.h"
#include "agg_ellipse.h"
#include "agg_embedded_raster_fonts.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_pixfmt_gray.h"
#include "agg_alpha_mask_u8.h"
#include "agg_pixfmt_amask_adaptor.h"
#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_raster_text.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_u.h"
#include "agg_scanline_p.h"
#include "agg_vcgen_markers_term.h"

#include "agg_py_path_iterator.h"
#include "path_converters.h"

// These are copied directly from path.py, and must be kept in sync
#define STOP   0
#define MOVETO 1
#define LINETO 2
#define CURVE3 3
#define CURVE4 4
#define CLOSEPOLY 5

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3, 1 };

typedef agg::pixfmt_rgba32_plain pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_aa_solid<renderer_base> renderer_aa;
typedef agg::renderer_scanline_bin_solid<renderer_base> renderer_bin;
typedef agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl> rasterizer;

typedef agg::scanline_p8 scanline_p8;
typedef agg::scanline_bin scanline_bin;
typedef agg::amask_no_clip_gray8 alpha_mask_type;
typedef agg::scanline_u8_am<alpha_mask_type> scanline_am;

typedef agg::renderer_base<agg::pixfmt_gray8> renderer_base_alpha_mask_type;
typedef agg::renderer_scanline_aa_solid<renderer_base_alpha_mask_type> renderer_alpha_mask_type;

// a helper class to pass agg::buffer objects around.  agg::buffer is
// a class in the swig wrapper
class BufferRegion : public Py::PythonExtension<BufferRegion>
{
public:
    BufferRegion(const agg::rect_i &r, bool freemem = true) :
        rect(r), freemem(freemem)
    {
        width = r.x2 - r.x1;
        height = r.y2 - r.y1;
        stride = width * 4;
        data = new agg::int8u[stride * height];
    }

    agg::int8u* data;
    agg::rect_i rect;
    int width;
    int height;
    int stride;

    bool freemem;

    // set the x and y corners of the rectangle
    Py::Object set_x(const Py::Tuple &args);
    Py::Object set_y(const Py::Tuple &args);

    Py::Object get_extents(const Py::Tuple &args);

    Py::Object to_string(const Py::Tuple &args);
    Py::Object to_string_argb(const Py::Tuple &args);
    static void init_type(void);

    virtual ~BufferRegion()
    {
        if (freemem)
        {
            delete [] data;
            data = NULL;
        }
    };

private:
    // prevent copying
    BufferRegion(const BufferRegion&);
    BufferRegion& operator=(const BufferRegion&);
};

class GCAgg
{
public:
    GCAgg(const Py::Object& gc, double dpi);

    double dpi;
    bool isaa;

    agg::line_cap_e cap;
    agg::line_join_e join;

    double linewidth;
    double alpha;
    bool forced_alpha;
    agg::rgba color;

    Py::Object cliprect;
    Py::Object clippath;
    agg::trans_affine clippath_trans;

    //dashes
    typedef std::vector<std::pair<double, double> > dash_t;
    double dashOffset;
    dash_t dashes;
    e_snap_mode snap_mode;

    Py::Object hatchpath;

    double sketch_scale;
    double sketch_length;
    double sketch_randomness;

protected:
    agg::rgba get_color(const Py::Object& gc);
    double points_to_pixels(const Py::Object& points);
    void _set_linecap(const Py::Object& gc) ;
    void _set_joinstyle(const Py::Object& gc) ;
    void _set_dashes(const Py::Object& gc) ;
    void _set_clip_rectangle(const Py::Object& gc);
    void _set_clip_path(const Py::Object& gc);
    void _set_antialiased(const Py::Object& gc);
    void _set_snap(const Py::Object& gc);
    void _set_hatch_path(const Py::Object& gc);
    void _set_sketch_params(const Py::Object& gc);
};


//struct AMRenderer {
//
//}

// the renderer
class RendererAgg: public Py::PythonExtension<RendererAgg>
{
    typedef std::pair<bool, agg::rgba> facepair_t;
public:
    RendererAgg(unsigned int width, unsigned int height, double dpi, int debug);
    static void init_type(void);

    unsigned int get_width()
    {
        return width;
    }

    unsigned int get_height()
    {
        return height;
    }

    // the drawing methods
    //Py::Object _draw_markers_nocache(const Py::Tuple & args);
    //Py::Object _draw_markers_cache(const Py::Tuple & args);
    Py::Object draw_markers(const Py::Tuple & args);
    Py::Object draw_text_image(const Py::Tuple & args);
    Py::Object draw_image(const Py::Tuple & args);
    Py::Object draw_path(const Py::Tuple & args);
    Py::Object draw_path_collection(const Py::Tuple & args);
    Py::Object draw_quad_mesh(const Py::Tuple& args);
    Py::Object draw_gouraud_triangle(const Py::Tuple& args);
    Py::Object draw_gouraud_triangles(const Py::Tuple& args);

    Py::Object write_rgba(const Py::Tuple & args);
    Py::Object tostring_rgb(const Py::Tuple & args);
    Py::Object tostring_argb(const Py::Tuple & args);
    Py::Object tostring_bgra(const Py::Tuple & args);
    Py::Object tostring_rgba_minimized(const Py::Tuple & args);
    Py::Object buffer_rgba(const Py::Tuple & args);
    Py::Object clear(const Py::Tuple & args);

    Py::Object copy_from_bbox(const Py::Tuple & args);
    Py::Object restore_region(const Py::Tuple & args);
    Py::Object restore_region2(const Py::Tuple & args);

    #if PY3K
    virtual int buffer_get( Py_buffer *, int flags );
    #endif

    virtual ~RendererAgg();

    static const size_t PIXELS_PER_INCH;
    unsigned int width, height;
    double dpi;
    size_t NUMBYTES;  //the number of bytes in buffer

    agg::int8u *pixBuffer;
    agg::rendering_buffer renderingBuffer;

    agg::int8u *alphaBuffer;
    agg::rendering_buffer alphaMaskRenderingBuffer;
    alpha_mask_type alphaMask;
    agg::pixfmt_gray8 pixfmtAlphaMask;
    renderer_base_alpha_mask_type rendererBaseAlphaMask;
    renderer_alpha_mask_type rendererAlphaMask;
    scanline_am scanlineAlphaMask;

    scanline_p8 slineP8;
    scanline_bin slineBin;
    pixfmt pixFmt;
    renderer_base rendererBase;
    renderer_aa rendererAA;
    renderer_bin rendererBin;
    rasterizer theRasterizer;

    Py::Object lastclippath;
    agg::trans_affine lastclippath_transform;

    static const size_t HATCH_SIZE = 72;
    agg::int8u hatchBuffer[HATCH_SIZE * HATCH_SIZE * 4];
    agg::rendering_buffer hatchRenderingBuffer;

    const int debug;

    agg::rgba _fill_color;


protected:
    double points_to_pixels(const Py::Object& points);
    agg::rgba rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha);
    facepair_t _get_rgba_face(const Py::Object& rgbFace, double alpha, bool forced_alpha);

    template<class R>
    void set_clipbox(const Py::Object& cliprect, R& rasterizer);

    bool render_clippath(const Py::Object& clippath, const agg::trans_affine& clippath_trans);

    template<class PathIteratorType>
    void _draw_path(PathIteratorType& path, bool has_clippath,
                    const facepair_t& face, const GCAgg& gc);

    template<class PathGenerator, int check_snap, int has_curves>
    Py::Object
    _draw_path_collection_generic
    (GCAgg&                         gc,
     agg::trans_affine              master_transform,
     const Py::Object&              cliprect,
     const Py::Object&              clippath,
     const agg::trans_affine&       clippath_trans,
     const PathGenerator&           path_generator,
     const Py::Object&              transforms_obj,
     const Py::Object&              offsets_obj,
     const agg::trans_affine&       offset_trans,
     const Py::Object&              facecolors_obj,
     const Py::Object&              edgecolors_obj,
     const Py::SeqBase<Py::Float>&  linewidths,
     const Py::SeqBase<Py::Object>& linestyles_obj,
     const Py::SeqBase<Py::Int>&    antialiaseds,
     const bool                     data_offsets);

    void
    _draw_gouraud_triangle(
        const double* points, const double* colors,
        agg::trans_affine trans, bool has_clippath);

private:
    void create_alpha_buffers();

    // prevent copying
    RendererAgg(const RendererAgg&);
    RendererAgg& operator=(const RendererAgg&);
};

// the extension module
class _backend_agg_module : public Py::ExtensionModule<_backend_agg_module>
{
public:
    _backend_agg_module()
        : Py::ExtensionModule<_backend_agg_module>("_backend_agg")
    {
        RendererAgg::init_type();
        BufferRegion::init_type();

        add_keyword_method("RendererAgg", &_backend_agg_module::new_renderer,
                           "RendererAgg(width, height, dpi)");
        initialize("The agg rendering backend");
    }

    virtual ~_backend_agg_module() {}

private:

    Py::Object new_renderer(const Py::Tuple &args, const Py::Dict &kws);

    // prevent copying
    _backend_agg_module(const _backend_agg_module&);
    _backend_agg_module& operator=(const _backend_agg_module&);
};



#endif

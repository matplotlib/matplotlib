#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mplutils.h"
#include "numpy_cpp.h"
#include "py_converters.h"
#include "_backend_agg.h"
#include "py_converters_11.h"

using namespace pybind11::literals;

/**********************************************************************
 * BufferRegion
 * */

/* TODO: This doesn't seem to be used internally.  Remove? */

static void
PyBufferRegion_set_x(BufferRegion *self, int x)
{
    self->get_rect().x1 = x;
}

static void
PyBufferRegion_set_y(BufferRegion *self, int y)
{
    self->get_rect().y1 = y;
}

static pybind11::object
PyBufferRegion_get_extents(BufferRegion *self)
{
    agg::rect_i rect = self->get_rect();

    return pybind11::make_tuple(rect.x1, rect.y1, rect.x2, rect.y2);
}

/**********************************************************************
 * RendererAgg
 * */

static void
PyRendererAgg_draw_path(RendererAgg *self,
                        pybind11::object gc_obj,
                        mpl::PathIterator path,
                        agg::trans_affine trans,
                        agg::rgba face)
{
    GCAgg gc;

    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }

    self->draw_path(gc, path, trans, face);
}

static void
PyRendererAgg_draw_text_image(RendererAgg *self,
                              pybind11::array_t<agg::int8u, pybind11::array::c_style> image_obj,
                              double x,
                              double y,
                              double angle,
                              pybind11::object gc_obj)
{
    numpy::array_view<agg::int8u, 2> image;
    GCAgg gc;

    if (!image.converter_contiguous(image_obj.ptr(), &image)) {
        throw pybind11::error_already_set();
    }
    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }

    self->draw_text_image(gc, image, x, y, angle);
}

static void
PyRendererAgg_draw_markers(RendererAgg *self,
                           pybind11::object gc_obj,
                           mpl::PathIterator marker_path,
                           agg::trans_affine marker_path_trans,
                           mpl::PathIterator path,
                           agg::trans_affine trans,
                           agg::rgba face)
{
    GCAgg gc;

    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }

    self->draw_markers(gc, marker_path, marker_path_trans, path, trans, face);
}

static void
PyRendererAgg_draw_image(RendererAgg *self,
                         pybind11::object gc_obj,
                         double x,
                         double y,
                         pybind11::array_t<agg::int8u, pybind11::array::c_style> image_obj)
{
    GCAgg gc;
    numpy::array_view<agg::int8u, 3> image;

    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }
    if (!image.set(image_obj.ptr())) {
        throw pybind11::error_already_set();
    }

    x = mpl_round(x);
    y = mpl_round(y);

    gc.alpha = 1.0;
    self->draw_image(gc, x, y, image);
}

static void
PyRendererAgg_draw_path_collection(RendererAgg *self,
                                   pybind11::object gc_obj,
                                   agg::trans_affine master_transform,
                                   pybind11::object paths_obj,
                                   pybind11::object transforms_obj,
                                   pybind11::object offsets_obj,
                                   agg::trans_affine offset_trans,
                                   pybind11::object facecolors_obj,
                                   pybind11::object edgecolors_obj,
                                   pybind11::object linewidths_obj,
                                   pybind11::object dashes_obj,
                                   pybind11::object antialiaseds_obj,
                                   pybind11::object Py_UNUSED(ignored_obj),
                                   // offset position is no longer used
                                   pybind11::object Py_UNUSED(offset_position_obj))
{
    GCAgg gc;
    mpl::PathGenerator paths;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    numpy::array_view<const double, 2> facecolors;
    numpy::array_view<const double, 2> edgecolors;
    numpy::array_view<const double, 1> linewidths;
    DashesVector dashes;
    numpy::array_view<const uint8_t, 1> antialiaseds;

    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }
    if (!convert_pathgen(paths_obj.ptr(), &paths)) {
        throw pybind11::error_already_set();
    }
    if (!convert_transforms(transforms_obj.ptr(), &transforms)) {
        throw pybind11::error_already_set();
    }
    if (!convert_points(offsets_obj.ptr(), &offsets)) {
        throw pybind11::error_already_set();
    }
    if (!convert_colors(facecolors_obj.ptr(), &facecolors)) {
        throw pybind11::error_already_set();
    }
    if (!convert_colors(edgecolors_obj.ptr(), &edgecolors)) {
        throw pybind11::error_already_set();
    }
    if (!linewidths.converter(linewidths_obj.ptr(), &linewidths)) {
        throw pybind11::error_already_set();
    }
    if (!convert_dashes_vector(dashes_obj.ptr(), &dashes)) {
        throw pybind11::error_already_set();
    }
    if (!antialiaseds.converter(antialiaseds_obj.ptr(), &antialiaseds)) {
        throw pybind11::error_already_set();
    }

    self->draw_path_collection(gc,
            master_transform,
            paths,
            transforms,
            offsets,
            offset_trans,
            facecolors,
            edgecolors,
            linewidths,
            dashes,
            antialiaseds);
}

static void
PyRendererAgg_draw_quad_mesh(RendererAgg *self,
                             pybind11::object gc_obj,
                             agg::trans_affine master_transform,
                             unsigned int mesh_width,
                             unsigned int mesh_height,
                             pybind11::object coordinates_obj,
                             pybind11::object offsets_obj,
                             agg::trans_affine offset_trans,
                             pybind11::object facecolors_obj,
                             bool antialiased,
                             pybind11::object edgecolors_obj)
{
    GCAgg gc;
    numpy::array_view<const double, 3> coordinates;
    numpy::array_view<const double, 2> offsets;
    numpy::array_view<const double, 2> facecolors;
    numpy::array_view<const double, 2> edgecolors;

    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }
    if (!coordinates.converter(coordinates_obj.ptr(), &coordinates)) {
        throw pybind11::error_already_set();
    }
    if (!convert_points(offsets_obj.ptr(), &offsets)) {
        throw pybind11::error_already_set();
    }
    if (!convert_colors(facecolors_obj.ptr(), &facecolors)) {
        throw pybind11::error_already_set();
    }
    if (!convert_colors(edgecolors_obj.ptr(), &edgecolors)) {
        throw pybind11::error_already_set();
    }

    self->draw_quad_mesh(gc,
            master_transform,
            mesh_width,
            mesh_height,
            coordinates,
            offsets,
            offset_trans,
            facecolors,
            antialiased,
            edgecolors);
}

static void
PyRendererAgg_draw_gouraud_triangles(RendererAgg *self,
                                     pybind11::object gc_obj,
                                     pybind11::object points_obj,
                                     pybind11::object colors_obj,
                                     agg::trans_affine trans)
{
    GCAgg gc;
    numpy::array_view<const double, 3> points;
    numpy::array_view<const double, 3> colors;

    if (!convert_gcagg(gc_obj.ptr(), &gc)) {
        throw pybind11::error_already_set();
    }
    if (!points.converter(points_obj.ptr(), &points)) {
        throw pybind11::error_already_set();
    }
    if (!colors.converter(colors_obj.ptr(), &colors)) {
        throw pybind11::error_already_set();
    }

    self->draw_gouraud_triangles(gc, points, colors, trans);
}

PYBIND11_MODULE(_backend_agg, m)
{
    _import_array();
    pybind11::class_<RendererAgg>(m, "RendererAgg", pybind11::buffer_protocol())
        .def(pybind11::init<unsigned int, unsigned int, double>(),
             "width"_a, "height"_a, "dpi"_a)

        .def("draw_path", &PyRendererAgg_draw_path,
             "gc"_a, "path"_a, "trans"_a, "face"_a = nullptr)
        .def("draw_markers", &PyRendererAgg_draw_markers,
             "gc"_a, "marker_path"_a, "marker_path_trans"_a, "path"_a, "trans"_a,
             "face"_a = nullptr)
        .def("draw_text_image", &PyRendererAgg_draw_text_image,
             "image"_a, "x"_a, "y"_a, "angle"_a, "gc"_a)
        .def("draw_image", &PyRendererAgg_draw_image,
             "gc"_a, "x"_a, "y"_a, "image"_a)
        .def("draw_path_collection", &PyRendererAgg_draw_path_collection,
             "gc"_a, "master_transform"_a, "paths"_a, "transforms"_a, "offsets"_a,
             "offset_trans"_a, "facecolors"_a, "edgecolors"_a, "linewidths"_a,
             "dashes"_a, "antialiaseds"_a, "ignored"_a, "offset_position"_a)
        .def("draw_quad_mesh", &PyRendererAgg_draw_quad_mesh,
             "gc"_a, "master_transform"_a, "mesh_width"_a, "mesh_height"_a,
             "coordinates"_a, "offsets"_a, "offset_trans"_a, "facecolors"_a,
             "antialiased"_a, "edgecolors"_a)
        .def("draw_gouraud_triangles", &PyRendererAgg_draw_gouraud_triangles,
             "gc"_a, "points"_a, "colors"_a, "trans"_a = nullptr)

        .def("clear", &RendererAgg::clear)

        .def("copy_from_bbox", &RendererAgg::copy_from_bbox,
             "bbox"_a)
        .def("restore_region",
             pybind11::overload_cast<BufferRegion&>(&RendererAgg::restore_region),
             "region"_a)
        .def("restore_region",
             pybind11::overload_cast<BufferRegion&, int, int, int, int, int, int>(&RendererAgg::restore_region),
             "region"_a, "xx1"_a, "yy1"_a, "xx2"_a, "yy2"_a, "x"_a, "y"_a)

        .def_buffer([](RendererAgg *renderer) -> pybind11::buffer_info {
            std::vector<pybind11::ssize_t> shape {
                renderer->get_height(),
                renderer->get_width(),
                4
            };
            std::vector<pybind11::ssize_t> strides {
                renderer->get_width() * 4,
                4,
                1
            };
            return pybind11::buffer_info(renderer->pixBuffer, shape, strides);
        });

    pybind11::class_<BufferRegion>(m, "BufferRegion", pybind11::buffer_protocol())
        // BufferRegion is not constructible from Python, thus no py::init is added.
        .def("set_x", &PyBufferRegion_set_x)
        .def("set_y", &PyBufferRegion_set_y)
        .def("get_extents", &PyBufferRegion_get_extents)
        .def_buffer([](BufferRegion *buffer) -> pybind11::buffer_info {
            std::vector<pybind11::ssize_t> shape {
                buffer->get_height(),
                buffer->get_width(),
                4
            };
            std::vector<pybind11::ssize_t> strides {
                buffer->get_width() * 4,
                4,
                1
            };
            return pybind11::buffer_info(buffer->get_data(), shape, strides);
        });
}

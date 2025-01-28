#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mplutils.h"
#include "py_converters.h"
#include "_backend_agg.h"

namespace py = pybind11;
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

static py::object
PyBufferRegion_get_extents(BufferRegion *self)
{
    agg::rect_i rect = self->get_rect();

    return py::make_tuple(rect.x1, rect.y1, rect.x2, rect.y2);
}

/**********************************************************************
 * RendererAgg
 * */

static void
PyRendererAgg_draw_path(RendererAgg *self,
                        GCAgg &gc,
                        mpl::PathIterator path,
                        agg::trans_affine trans,
                        py::object rgbFace)
{
    agg::rgba face = rgbFace.cast<agg::rgba>();
    if (!rgbFace.is_none()) {
        if (gc.forced_alpha || rgbFace.cast<py::sequence>().size() == 3) {
            face.a = gc.alpha;
        }
    }

    self->draw_path(gc, path, trans, face);
}

static void
PyRendererAgg_draw_text_image(RendererAgg *self,
                              py::array_t<agg::int8u, py::array::c_style | py::array::forcecast> image_obj,
                              std::variant<double, int> vx,
                              std::variant<double, int> vy,
                              double angle,
                              GCAgg &gc)
{
    int x, y;

    if (auto value = std::get_if<double>(&vx)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="x", "obj_type"_a="parameter as float",
             "alternative"_a="int(x)");
        x = static_cast<int>(*value);
    } else if (auto value = std::get_if<int>(&vx)) {
        x = *value;
    } else {
        throw std::runtime_error("Should not happen");
    }

    if (auto value = std::get_if<double>(&vy)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="y", "obj_type"_a="parameter as float",
             "alternative"_a="int(y)");
        y = static_cast<int>(*value);
    } else if (auto value = std::get_if<int>(&vy)) {
        y = *value;
    } else {
        throw std::runtime_error("Should not happen");
    }

    // TODO: This really shouldn't be mutable, but Agg's renderer buffers aren't const.
    auto image = image_obj.mutable_unchecked<2>();

    self->draw_text_image(gc, image, x, y, angle);
}

static void
PyRendererAgg_draw_markers(RendererAgg *self,
                           GCAgg &gc,
                           mpl::PathIterator marker_path,
                           agg::trans_affine marker_path_trans,
                           mpl::PathIterator path,
                           agg::trans_affine trans,
                           py::object rgbFace)
{
    agg::rgba face = rgbFace.cast<agg::rgba>();
    if (!rgbFace.is_none()) {
        if (gc.forced_alpha || rgbFace.cast<py::sequence>().size() == 3) {
            face.a = gc.alpha;
        }
    }

    self->draw_markers(gc, marker_path, marker_path_trans, path, trans, face);
}

static void
PyRendererAgg_draw_image(RendererAgg *self,
                         GCAgg &gc,
                         double x,
                         double y,
                         py::array_t<agg::int8u, py::array::c_style | py::array::forcecast> image_obj)
{
    // TODO: This really shouldn't be mutable, but Agg's renderer buffers aren't const.
    auto image = image_obj.mutable_unchecked<3>();

    x = mpl_round(x);
    y = mpl_round(y);

    gc.alpha = 1.0;
    self->draw_image(gc, x, y, image);
}

static void
PyRendererAgg_draw_path_collection(RendererAgg *self,
                                   GCAgg &gc,
                                   agg::trans_affine master_transform,
                                   mpl::PathGenerator paths,
                                   py::array_t<double> transforms_obj,
                                   py::array_t<double> offsets_obj,
                                   agg::trans_affine offset_trans,
                                   py::array_t<double> facecolors_obj,
                                   py::array_t<double> edgecolors_obj,
                                   py::array_t<double> linewidths_obj,
                                   DashesVector dashes,
                                   py::array_t<uint8_t> antialiaseds_obj,
                                   py::object Py_UNUSED(ignored_obj),
                                   // offset position is no longer used
                                   py::object Py_UNUSED(offset_position_obj))
{
    auto transforms = convert_transforms(transforms_obj);
    auto offsets = convert_points(offsets_obj);
    auto facecolors = convert_colors(facecolors_obj);
    auto edgecolors = convert_colors(edgecolors_obj);
    auto linewidths = linewidths_obj.unchecked<1>();
    auto antialiaseds = antialiaseds_obj.unchecked<1>();

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
                             GCAgg &gc,
                             agg::trans_affine master_transform,
                             unsigned int mesh_width,
                             unsigned int mesh_height,
                             py::array_t<double, py::array::c_style | py::array::forcecast> coordinates_obj,
                             py::array_t<double> offsets_obj,
                             agg::trans_affine offset_trans,
                             py::array_t<double> facecolors_obj,
                             bool antialiased,
                             py::array_t<double> edgecolors_obj)
{
    auto coordinates = coordinates_obj.mutable_unchecked<3>();
    auto offsets = convert_points(offsets_obj);
    auto facecolors = convert_colors(facecolors_obj);
    auto edgecolors = convert_colors(edgecolors_obj);

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
                                     GCAgg &gc,
                                     py::array_t<double> points_obj,
                                     py::array_t<double> colors_obj,
                                     agg::trans_affine trans)
{
    auto points = points_obj.unchecked<3>();
    auto colors = colors_obj.unchecked<3>();

    self->draw_gouraud_triangles(gc, points, colors, trans);
}

PYBIND11_MODULE(_backend_agg, m, py::mod_gil_not_used())
{
    py::class_<RendererAgg>(m, "RendererAgg", py::buffer_protocol())
        .def(py::init<unsigned int, unsigned int, double>(),
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
             py::overload_cast<BufferRegion&>(&RendererAgg::restore_region),
             "region"_a)
        .def("restore_region",
             py::overload_cast<BufferRegion&, int, int, int, int, int, int>(&RendererAgg::restore_region),
             "region"_a, "xx1"_a, "yy1"_a, "xx2"_a, "yy2"_a, "x"_a, "y"_a)

        .def_buffer([](RendererAgg *renderer) -> py::buffer_info {
            std::vector<py::ssize_t> shape {
                renderer->get_height(),
                renderer->get_width(),
                4
            };
            std::vector<py::ssize_t> strides {
                renderer->get_width() * 4,
                4,
                1
            };
            return py::buffer_info(renderer->pixBuffer, shape, strides);
        });

    py::class_<BufferRegion>(m, "BufferRegion", py::buffer_protocol())
        // BufferRegion is not constructible from Python, thus no py::init is added.
        .def("set_x", &PyBufferRegion_set_x)
        .def("set_y", &PyBufferRegion_set_y)
        .def("get_extents", &PyBufferRegion_get_extents)
        .def_buffer([](BufferRegion *buffer) -> py::buffer_info {
            std::vector<py::ssize_t> shape {
                buffer->get_height(),
                buffer->get_width(),
                4
            };
            std::vector<py::ssize_t> strides {
                buffer->get_width() * 4,
                4,
                1
            };
            return py::buffer_info(buffer->get_data(), shape, strides);
        });
}

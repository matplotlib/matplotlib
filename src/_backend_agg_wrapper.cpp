#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <optional>

#include "mpl_utils.h"
#include "mpl_buffer.h"
#include "agg_convert_overloads.h"
#include "_backend_agg.h"

namespace nb = nanobind;
using namespace nanobind::literals;

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

static nb::object
PyBufferRegion_get_extents(BufferRegion *self)
{
    agg::rect_i rect = self->get_rect();

    return nb::make_tuple(rect.x1, rect.y1, rect.x2, rect.y2);
}

/**********************************************************************
 * RendererAgg
 * */

static void
PyRendererAgg_draw_path(RendererAgg *self,
                        const mpl::defer<GCAgg> &gc_obj,
                        const mpl::defer<mpl::PathIterator> &path_obj,
                        const mpl::defer<agg::trans_affine> &trans_obj,
                        const mpl::defer<agg::rgba> &face_obj)
{
    GCAgg gc;  gc_obj.assign(gc, "gc");
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");
    auto face = face_obj.value("value");

    if (!face_obj.is_none()) {
        if (gc.forced_alpha || nb::len(face_obj) == 3) {
            face.a = gc.alpha;
        }
    }

    self->draw_path(gc, path, trans, face);
}

static void
PyRendererAgg_draw_text_image(RendererAgg *self,
                              const mpl::defer<TextImage> &image_obj,
                              int x, int y, double angle,
                              const mpl::defer<GCAgg> &gc_obj)
{
    auto image = image_obj.value("image");
    GCAgg gc; gc_obj.assign(gc, "gc");

    self->draw_text_image(gc, image, x, y, angle);
}

static void
PyRendererAgg_draw_text_images(
    RendererAgg *self,
    const mpl::defer<GlyphRunBuffer> &buffer_obj,
    const mpl::defer<GlyphRunPositionArray> &positions_obj,
    const mpl::defer<GCAgg> &gc_obj)
{
    auto buffer = buffer_obj.value("buffer");
    auto positions = positions_obj.value("positions");
    GCAgg gc; gc_obj.assign(gc, "gc");

    // The renderer indexes into the buffer with these.
    auto const& size = buffer.shape(0);
    for (auto i = 0; i < positions.shape(0); i++) {
        auto const& offset = positions(i, 0);
        auto const& rows = positions(i, 1);
        auto const& cols = positions(i, 2);
        if (offset < 0 || rows < 0 || cols < 0 || offset > size ||
                (rows > 0 && cols > (size - offset) / rows)) {
            throw nb::value_error(mpl::format(
                "positions describes a bitmap outside of buffer at row {}", i));
        }
    }

    self->draw_text_images(gc, buffer, positions);
}

static void
PyRendererAgg_draw_markers(RendererAgg *self,
                           const mpl::defer<GCAgg> &gc_obj,
                           const mpl::defer<mpl::PathIterator> &marker_path_obj,
                           const mpl::defer<agg::trans_affine> &marker_path_trans_obj,
                           const mpl::defer<mpl::PathIterator> &path_obj,
                           const mpl::defer<agg::trans_affine> &trans_obj,
                           const mpl::defer<agg::rgba> &face_obj)
{
    GCAgg gc; gc_obj.assign(gc, "gc");
    auto marker_path = marker_path_obj.value("marker_path");
    auto marker_path_trans = marker_path_trans_obj.value("marker_path_trans");
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");
    auto face = face_obj.value("face");

    if (!face_obj.is_none()) {
        if (gc.forced_alpha || nb::len(face_obj) == 3) {
            face.a = gc.alpha;
        }
    }

    self->draw_markers(gc, marker_path, marker_path_trans, path, trans, face);
}

static void
PyRendererAgg_draw_image(RendererAgg *self,
                         const mpl::defer<GCAgg> &gc_obj,
                         double x,
                         double y,
                         const mpl::defer<RGBAImage> &image_obj)
{
    GCAgg gc; gc_obj.assign(gc, "gc");
    auto image = image_obj.value("image");

    x = mpl::round(x);
    y = mpl::round(y);

    self->draw_image(gc, x, y, image);
}

static void
PyRendererAgg_draw_path_collection(
    RendererAgg *self,
    const mpl::defer<GCAgg> &gc_obj,
    const mpl::defer<agg::trans_affine> &master_transform_obj,
    const mpl::defer<mpl::PathGenerator> &paths_obj,
    const mpl::defer<AffineTransformArray> &all_transforms_obj,
    const mpl::defer<PointArray> &offsets_obj,
    const mpl::defer<agg::trans_affine> &offset_trans_obj,
    const mpl::defer<ColorArray> &facecolors_obj,
    const mpl::defer<ColorArray> &edgecolors_obj,
    const mpl::defer<LineWidthArray> &linewidths_obj,
    const mpl::defer<LinestyleVector> &linestyles_obj,
    const mpl::defer<AntialiasedArray> &antialiaseds_obj,
    nb::object Py_UNUSED(urls_obj),
    // offset position is no longer used
    nb::object Py_UNUSED(offset_position_obj),
    std::optional<mpl::defer<ColorArray>> &hatchcolors_obj)
{
    GCAgg gc; gc_obj.assign(gc, "gc");
    auto master_transform = master_transform_obj.value("master_transform");
    auto paths = paths_obj.value("paths");
    auto all_transforms = all_transforms_obj.value("all_transforms");
    auto offsets = offsets_obj.value("offsets");
    auto offset_trans = offset_trans_obj.value("offset_trans");
    auto facecolors = facecolors_obj.value("facecolors");
    auto edgecolors = edgecolors_obj.value("edgecolors");
    auto linewidths = linewidths_obj.value("linewidths");
    auto linestyles = linestyles_obj.value("linestyles");
    auto antialiaseds = antialiaseds_obj.value("antialiaseds");

    auto hatchcolors = hatchcolors_obj ? hatchcolors_obj.value().value("hatchcolors") :
        ColorArray::unmanaged_empty();

    self->draw_path_collection(gc,
            master_transform,
            paths,
            all_transforms,
            offsets,
            offset_trans,
            facecolors,
            edgecolors,
            linewidths,
            linestyles,
            antialiaseds,
            hatchcolors);
}

static void
PyRendererAgg_draw_quad_mesh(
    RendererAgg *self,
    const mpl::defer<GCAgg> &gc_obj,
    const mpl::defer<agg::trans_affine> &master_transform_obj,
    unsigned int mesh_width,
    unsigned int mesh_height,
    const mpl::defer<QuadMeshCoordinateArray> &coordinates_obj,
    const mpl::defer<PointArray> &offsets_obj,
    const mpl::defer<agg::trans_affine> &offset_trans_obj,
    const mpl::defer<ColorArray> &facecolors_obj,
    bool antialiased,
    std::optional<mpl::defer<ColorArray>> &edgecolors_obj)
{
    GCAgg gc; gc_obj.assign(gc, "gc");
    auto master_transform = master_transform_obj.value("master_transform");
    auto coordinates = coordinates_obj.value("coordinates");
    auto offsets = offsets_obj.value("offsets");
    auto offset_trans = offset_trans_obj.value("offset_trans");
    auto facecolors = facecolors_obj.value("facecolors");
    auto edgecolors = edgecolors_obj ? edgecolors_obj.value().value("edgecolors") :
        ColorArray::unmanaged_empty();

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
                                     const mpl::defer<GCAgg> &gc_obj,
                                     const mpl::defer<GouraudPointArray> &points_obj,
                                     const mpl::defer<GouraudColorArray> &colors_obj,
                                     const mpl::defer<agg::trans_affine> &trans_obj)
{
    GCAgg gc; gc_obj.assign(gc, "gc");
    auto points = points_obj.value("points");
    auto colors = colors_obj.value("colors");
    auto trans = trans_obj.value("trans");

    self->draw_gouraud_triangles(gc, points, colors, trans);
}

static BufferRegion *
PyRendererAgg_copy_from_bbox(RendererAgg *self,
                             const mpl::defer<agg::rect_d> &bbox_obj)
{
    auto bbox = bbox_obj.value("bbox");

    return self->copy_from_bbox(bbox);
}

NB_MODULE(_backend_agg, m)
{
    nb::class_<RendererAgg>(m, "RendererAgg", nb::type_slots(mpl::buffer_slots<RendererAgg>))
        .def(nb::init<unsigned int, unsigned int, double>(),
             "width"_a, "height"_a, "dpi"_a)

        .def("draw_path", &PyRendererAgg_draw_path,
             "gc"_a, "path"_a, "trans"_a, "face"_a.none() = nb::none())
        .def("draw_markers", &PyRendererAgg_draw_markers,
             "gc"_a, "marker_path"_a, "marker_path_trans"_a, "path"_a, "trans"_a,
             "face"_a.none() = nullptr)
        .def("draw_text_image", &PyRendererAgg_draw_text_image,
             "image"_a, "x"_a, "y"_a, "angle"_a, "gc"_a)
        .def("draw_text_images", &PyRendererAgg_draw_text_images,
             "buffer"_a, "positions"_a, "gc"_a)
        .def("draw_image", &PyRendererAgg_draw_image,
             "gc"_a, "x"_a, "y"_a, "image"_a)
        .def("draw_path_collection", &PyRendererAgg_draw_path_collection,
             "gc"_a, "master_transform"_a, "paths"_a, "all_transforms"_a, "offsets"_a,
             "offset_trans"_a, "facecolors"_a, "edgecolors"_a, "linewidths"_a,
             "linestyles"_a, "antialiaseds"_a, "urls"_a, "offset_position"_a,
             nb::kw_only(), "hatchcolors"_a.none() = nb::none())
        .def("draw_quad_mesh", &PyRendererAgg_draw_quad_mesh,
             "gc"_a, "master_transform"_a, "mesh_width"_a, "mesh_height"_a,
             "coordinates"_a, "offsets"_a, "offset_trans"_a, "facecolors"_a,
             "antialiased"_a, "edgecolors"_a)
        .def("_draw_gouraud_triangles", &PyRendererAgg_draw_gouraud_triangles,
             "gc"_a, "points"_a, "colors"_a, "trans"_a = nb::none())

        .def("clear", &RendererAgg::clear)

        .def("copy_from_bbox", &PyRendererAgg_copy_from_bbox, "bbox"_a)
        .def("restore_region",
             nb::overload_cast<BufferRegion&>(&RendererAgg::restore_region),
             "region"_a)
        .def("restore_region",
             nb::overload_cast<BufferRegion&, int, int, int, int, int, int>(&RendererAgg::restore_region),
             "region"_a, "xx1"_a, "yy1"_a, "xx2"_a, "yy2"_a, "x"_a, "y"_a);

    nb::class_<BufferRegion>(m, "BufferRegion", nb::type_slots(mpl::buffer_slots<BufferRegion>))
        // BufferRegion is not constructible from Python, thus no nb::init is added.
        .def("set_x", &PyBufferRegion_set_x)
        .def("set_y", &PyBufferRegion_set_y)
        .def("get_extents", &PyBufferRegion_get_extents);
}

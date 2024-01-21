#include "_tri.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_tri, m) {
    py::class_<Triangulation>(m, "Triangulation")
        .def(py::init<const Triangulation::CoordinateArray&,
                      const Triangulation::CoordinateArray&,
                      const Triangulation::TriangleArray&,
                      const Triangulation::MaskArray&,
                      const Triangulation::EdgeArray&,
                      const Triangulation::NeighborArray&,
                      bool>(),
            "x"_a,
            "y"_a,
            "triangles"_a,
            "mask"_a,
            "edges"_a,
            "neighbors"_a,
            "correct_triangle_orientations"_a,
            "Create a new C++ Triangulation object.\n"
            "This should not be called directly, use the python class\n"
            "matplotlib.tri.Triangulation instead.\n")
        .def("calculate_plane_coefficients", &Triangulation::calculate_plane_coefficients,
            "Calculate plane equation coefficients for all unmasked triangles.")
        .def("get_edges", &Triangulation::get_edges,
            "Return edges array.")
        .def("get_neighbors", &Triangulation::get_neighbors,
            "Return neighbors array.")
        .def("set_mask", &Triangulation::set_mask,
            "Set or clear the mask array.");

    py::class_<TriContourGenerator>(m, "TriContourGenerator")
        .def(py::init<Triangulation&,
                      const TriContourGenerator::CoordinateArray&>(),
            "triangulation"_a,
            "z"_a,
            "Create a new C++ TriContourGenerator object.\n"
            "This should not be called directly, use the functions\n"
            "matplotlib.axes.tricontour and tricontourf instead.\n")
        .def("create_contour", &TriContourGenerator::create_contour,
            "Create and return a non-filled contour.")
        .def("create_filled_contour", &TriContourGenerator::create_filled_contour,
            "Create and return a filled contour.");

    py::class_<TrapezoidMapTriFinder>(m, "TrapezoidMapTriFinder")
        .def(py::init<Triangulation&>(),
            "triangulation"_a,
            "Create a new C++ TrapezoidMapTriFinder object.\n"
            "This should not be called directly, use the python class\n"
            "matplotlib.tri.TrapezoidMapTriFinder instead.\n")
        .def("find_many", &TrapezoidMapTriFinder::find_many,
            "Find indices of triangles containing the point coordinates (x, y).")
        .def("get_tree_stats", &TrapezoidMapTriFinder::get_tree_stats,
            "Return statistics about the tree used by the trapezoid map.")
        .def("initialize", &TrapezoidMapTriFinder::initialize,
            "Initialize this object, creating the trapezoid map from the triangulation.")
        .def("print_tree", &TrapezoidMapTriFinder::print_tree,
            "Print the search tree as text to stdout; useful for debug purposes.");
}

#include "_tri.h"

PYBIND11_MODULE(_tri, m) {
    py::class_<Triangulation>(m, "Triangulation",
        .def(py::init<const Triangulation::CoordinateArray&,
                      const Triangulation::CoordinateArray&,
                      const Triangulation::TriangleArray&,
                      const Triangulation::MaskArray&,
                      const Triangulation::EdgeArray&,
                      const Triangulation::NeighborArray&,
                      bool>(),
            py::arg("x"),
            py::arg("y"),
            py::arg("triangles"),
            py::arg("mask"),
            py::arg("edges"),
            py::arg("neighbors"),
            py::arg("correct_triangle_orientations"),
            "Triangulation(x, y, triangles, mask, edges, neighbors)\n"
            "--\n\n"
            "Create a new C++ Triangulation object.\n"
            "This should not be called directly, instead use the python class\n"
            "matplotlib.tri.Triangulation instead.\n")
        .def("calculate_plane_coefficients", &Triangulation::calculate_plane_coefficients,
            "calculate_plane_coefficients(self, z, plane_coefficients)\n"
            "--\n\n"
            "Calculate plane equation coefficients for all unmasked triangles.")
        .def("get_edges", &Triangulation::get_edges,
            "get_edges(self)\n"
            "--\n\n"
            "Return edges array.")
        .def("get_neighbors", &Triangulation::get_neighbors,
            "get_neighbors(self)\n"
            "--\n\n"
            "Return neighbors array.")
        .def("set_mask", &Triangulation::set_mask,
            "set_mask(self, mask)\n"
            "--\n\n"
            "Set or clear the mask array.");

    py::class_<TriContourGenerator>(m, "TriContourGenerator",
        .def(py::init<Triangulation&,
                      const TriContourGenerator::CoordinateArray&>(),
            py::arg("triangulation"),
            py::arg("z"),
            "TriContourGenerator(triangulation, z)\n"
            "--\n\n"
            "Create a new C++ TriContourGenerator object.\n"
            "This should not be called directly, instead use the functions\n"
            "matplotlib.axes.tricontour and tricontourf instead.\n")
        .def("create_contour", &TriContourGenerator::create_contour,
            "create_contour(self, level)\n"
            "--\n\n"
            "Create and return a non-filled contour.")
        .def("create_filled_contour", &TriContourGenerator::create_filled_contour,
            "create_filled_contour(self, lower_level, upper_level)\n"
            "--\n\n"
            "Create and return a filled contour.");

    py::class_<TrapezoidMapTriFinder>(m, "TrapezoidMapTriFinder",
        .def(py::init<Triangulation&>(),
            py::arg("triangulation"),
            "TrapezoidMapTriFinder(triangulation)\n"
            "--\n\n"
            "Create a new C++ TrapezoidMapTriFinder object.\n"
            "This should not be called directly, instead use the python class\n"
            "matplotlib.tri.TrapezoidMapTriFinder instead.\n")
        .def("find_many", &TrapezoidMapTriFinder::find_many,
            "find_many(self, x, y)\n"
            "--\n\n"
            "Find indices of triangles containing the point coordinates (x, y).")
        .def("get_tree_stats", &TrapezoidMapTriFinder::get_tree_stats,
            "get_tree_stats(self)\n"
            "--\n\n"
            "Return statistics about the tree used by the trapezoid map.")
        .def("initialize", &TrapezoidMapTriFinder::initialize,
            "initialize(self)\n"
            "--\n\n"
            "Initialize this object, creating the trapezoid map from the triangulation.")
        .def("print_tree", &TrapezoidMapTriFinder::print_tree,
            "print_tree(self)\n"
            "--\n\n"
            "Print the search tree as text to stdout; useful for debug purposes.");
}

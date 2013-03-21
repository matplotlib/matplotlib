/* This file contains liberal use of asserts to assist code development and
 * debugging.  Standard matplotlib builds disable asserts so they cause no
 * performance reduction.  To enable the asserts, you need to undefine the
 * NDEBUG macro, which is achieved by adding the following
 *     undef_macros=['NDEBUG']
 * to the appropriate make_extension call in setupext.py, and then rebuilding.
 */
#include "_tri.h"
#include "src/mplutils.h"

#include <algorithm>
#include <iostream>
#include <set>

#define MOVETO 1
#define LINETO 2



TriEdge::TriEdge()
    : tri(-1), edge(-1)
{}

TriEdge::TriEdge(int tri_, int edge_)
    : tri(tri_), edge(edge_)
{}

bool TriEdge::operator<(const TriEdge& other) const
{
    if (tri != other.tri)
        return tri < other.tri;
    else
        return edge < other.edge;
}

bool TriEdge::operator==(const TriEdge& other) const
{
    return tri == other.tri && edge == other.edge;
}

bool TriEdge::operator!=(const TriEdge& other) const
{
    return !operator==(other);
}

std::ostream& operator<<(std::ostream& os, const TriEdge& tri_edge)
{
    return os << tri_edge.tri << ' ' << tri_edge.edge;
}



XY::XY()
{}

XY::XY(const double& x_, const double& y_)
    : x(x_), y(y_)
{}

double XY::angle() const
{
    return atan2(y, x);
}

double XY::cross_z(const XY& other) const
{
    return x*other.y - y*other.x;
}

bool XY::is_right_of(const XY& other) const
{
    if (x == other.x)
        return y > other.y;
    else
        return x > other.x;
}

bool XY::operator==(const XY& other) const
{
    return x == other.x && y == other.y;
}

bool XY::operator!=(const XY& other) const
{
    return x != other.x || y != other.y;
}

XY XY::operator*(const double& multiplier) const
{
    return XY(x*multiplier, y*multiplier);
}

const XY& XY::operator+=(const XY& other)
{
    x += other.x;
    y += other.y;
    return *this;
}

const XY& XY::operator-=(const XY& other)
{
    x -= other.x;
    y -= other.y;
    return *this;
}

XY XY::operator+(const XY& other) const
{
    return XY(x + other.x, y + other.y);
}

XY XY::operator-(const XY& other) const
{
    return XY(x - other.x, y - other.y);
}

std::ostream& operator<<(std::ostream& os, const XY& xy)
{
    return os << '(' << xy.x << ' ' << xy.y << ')';
}



XYZ::XYZ(const double& x_, const double& y_, const double& z_)
    : x(x_), y(y_), z(z_)
{}

XYZ XYZ::cross(const XYZ& other) const
{
    return XYZ(y*other.z - z*other.y,
               z*other.x - x*other.z,
               x*other.y - y*other.x);
}

double XYZ::dot(const XYZ& other) const
{
    return x*other.x + y*other.y + z*other.z;
}

double XYZ::length_squared() const
{
    return x*x + y*y + z*z;
}

XYZ XYZ::operator-(const XYZ& other) const
{
    return XYZ(x - other.x, y - other.y, z - other.z);
}

std::ostream& operator<<(std::ostream& os, const XYZ& xyz)
{
    return os << '(' << xyz.x << ' ' << xyz.y << ' ' << xyz.z << ')';
}



BoundingBox::BoundingBox()
    : empty(true)
{}

void BoundingBox::add(const XY& point)
{
    if (empty) {
        empty = false;
        lower = upper = point;
    } else {
        if      (point.x < lower.x) lower.x = point.x;
        else if (point.x > upper.x) upper.x = point.x;

        if      (point.y < lower.y) lower.y = point.y;
        else if (point.y > upper.y) upper.y = point.y;
    }
}

void BoundingBox::expand(const XY& delta)
{
    if (!empty)
    {
        lower -= delta;
        upper += delta;
    }
}



ContourLine::ContourLine()
    : std::vector<XY>()
{}

void ContourLine::insert_unique(iterator pos, const XY& point)
{
    if (empty() || pos == end() || point != *pos)
        std::vector<XY>::insert(pos, point);
}

void ContourLine::push_back(const XY& point)
{
    if (empty() || point != back())
        std::vector<XY>::push_back(point);
}

void ContourLine::write() const
{
    std::cout << "ContourLine of " << size() << " points:";
    for (const_iterator it = begin(); it != end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;
}



void write_contour(const Contour& contour)
{
    std::cout << "Contour of " << contour.size() << " lines." << std::endl;
    for (Contour::const_iterator it = contour.begin(); it != contour.end(); ++it)
        it->write();
}




Triangulation::Triangulation(PyArrayObject* x,
                             PyArrayObject* y,
                             PyArrayObject* triangles,
                             PyArrayObject* mask,
                             PyArrayObject* edges,
                             PyArrayObject* neighbors)
    : _npoints(PyArray_DIM(x,0)),
      _ntri(PyArray_DIM(triangles,0)),
      _x(x),
      _y(y),
      _triangles(triangles),
      _mask(mask),
      _edges(edges),
      _neighbors(neighbors)
{
    _VERBOSE("Triangulation::Triangulation");
    correct_triangles();
}

Triangulation::~Triangulation()
{
    _VERBOSE("Triangulation::~Triangulation");
    Py_XDECREF(_x);
    Py_XDECREF(_y);
    Py_XDECREF(_triangles);
    Py_XDECREF(_mask);
    Py_XDECREF(_edges);
    Py_XDECREF(_neighbors);
}

void Triangulation::calculate_boundaries()
{
    _VERBOSE("Triangulation::calculate_boundaries");

    get_neighbors();  // Ensure _neighbors has been created.

    // Create set of all boundary TriEdges, which are those which do not
    // have a neighbor triangle.
    typedef std::set<TriEdge> BoundaryEdges;
    BoundaryEdges boundary_edges;
    for (int tri = 0; tri < _ntri; ++tri) {
        if (!is_masked(tri)) {
            for (int edge = 0; edge < 3; ++edge) {
                if (get_neighbor(tri, edge) == -1) {
                    boundary_edges.insert(TriEdge(tri, edge));
                }
            }
        }
    }

    // Take any boundary edge and follow the boundary until return to start
    // point, removing edges from boundary_edges as they are used.  At the same
    // time, initialise the _tri_edge_to_boundary_map.
    while (!boundary_edges.empty()) {
        // Start of new boundary.
        BoundaryEdges::iterator it = boundary_edges.begin();
        int tri = it->tri;
        int edge = it->edge;
        _boundaries.push_back(Boundary());
        Boundary& boundary = _boundaries.back();

        while (true) {
            boundary.push_back(TriEdge(tri, edge));
            boundary_edges.erase(it);
            _tri_edge_to_boundary_map[TriEdge(tri, edge)] =
                BoundaryEdge(_boundaries.size()-1, boundary.size()-1);

            // Move to next edge of current triangle.
            edge = (edge+1) % 3;

            // Find start point index of boundary edge.
            int point = get_triangle_point(tri, edge);

            // Find next TriEdge by traversing neighbors until find one
            // without a neighbor.
            while (get_neighbor(tri, edge) != -1) {
                tri = get_neighbor(tri, edge);
                edge = get_edge_in_triangle(tri, point);
            }

            if (TriEdge(tri,edge) == boundary.front())
                break;  // Reached beginning of this boundary, so finished it.
            else
                it = boundary_edges.find(TriEdge(tri, edge));
        }
    }
}

void Triangulation::calculate_edges()
{
    _VERBOSE("Triangulation::calculate_edges");
    Py_XDECREF(_edges);

    // Create set of all edges, storing them with start point index less than
    // end point index.
    typedef std::set<Edge> EdgeSet;
    EdgeSet edge_set;
    for (int tri = 0; tri < _ntri; ++tri) {
        if (!is_masked(tri)) {
            for (int edge = 0; edge < 3; edge++) {
                int start = get_triangle_point(tri, edge);
                int end   = get_triangle_point(tri, (edge+1)%3);
                edge_set.insert(start > end ? Edge(start,end) : Edge(end,start));
            }
        }
    }

    // Convert to python _edges array.
    npy_intp dims[2] = {static_cast<npy_intp>(edge_set.size()), 2};
    _edges = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_INT);
    int* edges_ptr = (int*)PyArray_DATA(_edges);
    for (EdgeSet::const_iterator it = edge_set.begin(); it != edge_set.end(); ++it) {
        *edges_ptr++ = it->start;
        *edges_ptr++ = it->end;
    }
}

void Triangulation::calculate_neighbors()
{
    _VERBOSE("Triangulation::calculate_neighbors");
    Py_XDECREF(_neighbors);

    // Create _neighbors array with shape (ntri,3) and initialise all to -1.
    npy_intp dims[2] = {_ntri,3};
    _neighbors = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_INT);
    int* neighbors_ptr = (int*)PyArray_DATA(_neighbors);
    std::fill(neighbors_ptr, neighbors_ptr + 3*_ntri, -1);

    // For each triangle edge (start to end point), find corresponding neighbor
    // edge from end to start point.  Do this by traversing all edges and
    // storing them in a map from edge to TriEdge.  If corresponding neighbor
    // edge is already in the map, don't need to store new edge as neighbor
    // already found.
    typedef std::map<Edge, TriEdge> EdgeToTriEdgeMap;
    EdgeToTriEdgeMap edge_to_tri_edge_map;
    for (int tri = 0; tri < _ntri; ++tri) {
        if (!is_masked(tri)) {
            for (int edge = 0; edge < 3; ++edge) {
                int start = get_triangle_point(tri, edge);
                int end   = get_triangle_point(tri, (edge+1)%3);
                EdgeToTriEdgeMap::iterator it =
                    edge_to_tri_edge_map.find(Edge(end,start));
                if (it == edge_to_tri_edge_map.end()) {
                    // No neighbor edge exists in the edge_to_tri_edge_map, so
                    // add this edge to it.
                    edge_to_tri_edge_map[Edge(start,end)] = TriEdge(tri,edge);
                } else {
                    // Neighbor edge found, set the two elements of _neighbors
                    // and remove edge from edge_to_tri_edge_map.
                    neighbors_ptr[3*tri + edge] = it->second.tri;
                    neighbors_ptr[3*it->second.tri + it->second.edge] = tri;
                    edge_to_tri_edge_map.erase(it);
                }
            }
        }
    }

    // Note that remaining edges in the edge_to_tri_edge_map correspond to
    // boundary edges, but the boundaries are calculated separately elsewhere.
}

Py::Object Triangulation::calculate_plane_coefficients(const Py::Tuple &args)
{
    _VERBOSE("Triangulation::calculate_plane_coefficients");
    args.verify_length(1);

    PyArrayObject* z = (PyArrayObject*)PyArray_ContiguousFromObject(
                           args[0].ptr(), PyArray_DOUBLE, 1, 1);
    if (z == 0 || PyArray_DIM(z,0) != PyArray_DIM(_x,0)) {
        Py_XDECREF(z);
        throw Py::ValueError(
            "z array must have same length as triangulation x and y arrays");
    }

    PyArrayObject* planes_array = 0;  // Array to return.

    try
    {
        const double* zs = (const double*)PyArray_DATA(z);

        npy_intp dims[2] = {_ntri, 3};
        planes_array = (PyArrayObject*)PyArray_SimpleNew(2, dims,
                                                         PyArray_DOUBLE);
        double* planes = (double*)PyArray_DATA(planes_array);
        const int* tris = get_triangles_ptr();
        const double* xs = (const double*)PyArray_DATA(_x);
        const double* ys = (const double*)PyArray_DATA(_y);
        for (int tri = 0; tri < _ntri; ++tri)
        {
            if (is_masked(tri))
            {
                *planes++ = 0.0;
                *planes++ = 0.0;
                *planes++ = 0.0;
                tris += 3;
            }
            else
            {
                // Equation of plane for all points r on plane is r.normal = p
                // where normal is vector normal to the plane, and p is a
                // constant.  Rewrite as
                // r_x*normal_x + r_y*normal_y + r_z*normal_z = p
                // and rearrange to give
                // r_z = (-normal_x/normal_z)*r_x + (-normal_y/normal_z)*r_y +
                //       p/normal_z
                XYZ point0(xs[*tris], ys[*tris], zs[*tris]);
                tris++;
                XYZ side01 = XYZ(xs[*tris], ys[*tris], zs[*tris]) - point0;
                tris++;
                XYZ side02 = XYZ(xs[*tris], ys[*tris], zs[*tris]) - point0;
                tris++;

                XYZ normal = side01.cross(side02);

                if (normal.z == 0.0)
                {
                    // Normal is in x-y plane which means triangle consists of
                    // colinear points. To avoid dividing by zero, we use the
                    // Moore-Penrose pseudo-inverse.
                    double sum2 = (side01.x*side01.x + side01.y*side01.y +
                                   side02.x*side02.x + side02.y*side02.y);
                    double a = (side01.x*side01.z + side02.x*side02.z) / sum2;
                    double b = (side01.y*side01.z + side02.y*side02.z) / sum2;
                    *planes++ = a;
                    *planes++ = b;
                    *planes++ = point0.z - a*point0.x - b*point0.y;
                }
                else
                {
                    *planes++ = -normal.x / normal.z;           // x
                    *planes++ = -normal.y / normal.z;           // y
                    *planes++ = normal.dot(point0) / normal.z;  // constant
                }
            }
        }
    }
    catch (...)
    {
        Py_DECREF(z);
        Py_XDECREF(planes_array);
        throw;
    }

    Py_DECREF(z);
    return Py::asObject((PyObject*)planes_array);
}

void Triangulation::correct_triangles()
{
    int* triangles_ptr = (int*)PyArray_DATA(_triangles);
    int* neighbors_ptr = _neighbors != 0 ? (int*)PyArray_DATA(_neighbors) : 0;
    for (int tri = 0; tri < _ntri; ++tri) {
        XY point0 = get_point_coords(*triangles_ptr++);
        XY point1 = get_point_coords(*triangles_ptr++);
        XY point2 = get_point_coords(*triangles_ptr++);
        if ( (point1 - point0).cross_z(point2 - point0) < 0.0) {
            // Triangle points are clockwise, so change them to anticlockwise.
            std::swap(*(triangles_ptr-2), *(triangles_ptr-1));
            if (neighbors_ptr)
                std::swap(*(neighbors_ptr+3*tri+1), *(neighbors_ptr+3*tri+2));
        }
    }
}

const Triangulation::Boundaries& Triangulation::get_boundaries() const
{
    _VERBOSE("Triangulation::get_boundaries");
    if (_boundaries.empty())
        const_cast<Triangulation*>(this)->calculate_boundaries();
    return _boundaries;
}

void Triangulation::get_boundary_edge(const TriEdge& triEdge,
                                      int& boundary,
                                      int& edge) const
{
    get_boundaries();  // Ensure _tri_edge_to_boundary_map has been created.
    TriEdgeToBoundaryMap::const_iterator it =
        _tri_edge_to_boundary_map.find(triEdge);
    assert(it != _tri_edge_to_boundary_map.end() &&
           "TriEdge is not on a boundary");
    boundary = it->second.boundary;
    edge = it->second.edge;
}

int Triangulation::get_edge_in_triangle(int tri, int point) const
{
    assert(tri >= 0 && tri < _ntri && "Triangle index out of bounds");
    assert(point >= 0 && point < _npoints && "Point index out of bounds.");
    const int* triangles_ptr = get_triangles_ptr() + 3*tri;
    for (int edge = 0; edge < 3; ++edge) {
        if (*triangles_ptr++ == point) return edge;
    }
    return -1;  // point is not in triangle.
}

Py::Object Triangulation::get_edges()
{
    _VERBOSE("Triangulation::get_edges");
    if (_edges == 0)
        calculate_edges();
    return Py::asObject(Py::new_reference_to((PyObject*)_edges));
}

int Triangulation::get_neighbor(int tri, int edge) const
{
    assert(tri >= 0 && tri < _ntri && "Triangle index out of bounds");
    assert(edge >= 0 && edge < 3 && "Edge index out of bounds");
    return get_neighbors_ptr()[3*tri + edge];
}

TriEdge Triangulation::get_neighbor_edge(int tri, int edge) const
{
    int neighbor_tri = get_neighbor(tri, edge);
    if (neighbor_tri == -1)
        return TriEdge(-1,-1);
    else
        return TriEdge(neighbor_tri,
                       get_edge_in_triangle(neighbor_tri,
                                            get_triangle_point(tri,
                                                               (edge+1)%3)));
}

Py::Object Triangulation::get_neighbors()
{
    _VERBOSE("Triangulation::get_neighbors");
    if (_neighbors == 0) calculate_neighbors();
    return Py::asObject(Py::new_reference_to((PyObject*)_neighbors));
}

const int* Triangulation::get_neighbors_ptr() const
{
    if (_neighbors == 0)
        const_cast<Triangulation*>(this)->calculate_neighbors();
    return (const int*)PyArray_DATA(_neighbors);
}

int Triangulation::get_npoints() const
{
    return _npoints;
}

int Triangulation::get_ntri() const
{
    return _ntri;
}

XY Triangulation::get_point_coords(int point) const
{
    assert(point >= 0 && point < _npoints && "Point index out of bounds.");
    return XY( ((const double*)PyArray_DATA(_x))[point],
               ((const double*)PyArray_DATA(_y))[point] );
}

int Triangulation::get_triangle_point(int tri, int edge) const
{
    assert(tri >= 0 && tri < _ntri && "Triangle index out of bounds");
    assert(edge >= 0 && edge < 3 && "Edge index out of bounds");
    return get_triangles_ptr()[3*tri + edge];
}

int Triangulation::get_triangle_point(const TriEdge& tri_edge) const
{
    return get_triangle_point(tri_edge.tri, tri_edge.edge);
}

const int* Triangulation::get_triangles_ptr() const
{
    return (const int*)PyArray_DATA(_triangles);
}

void Triangulation::init_type()
{
    _VERBOSE("Triangulation::init_type");

    behaviors().name("Triangulation");
    behaviors().doc("Triangulation");

    add_varargs_method("calculate_plane_coefficients",
                       &Triangulation::calculate_plane_coefficients,
                       "calculate_plane_coefficients(z)");
    add_noargs_method("get_edges", &Triangulation::get_edges,
                      "get_edges()");
    add_noargs_method("get_neighbors", &Triangulation::get_neighbors,
                      "get_neighbors()");
    add_varargs_method("set_mask", &Triangulation::set_mask,
                       "set_mask(mask)");
}

bool Triangulation::is_masked(int tri) const
{
    assert(tri >= 0 && tri < _ntri && "Triangle index out of bounds.");
    return _mask && *((bool*)PyArray_DATA(_mask) + tri);
}

Py::Object Triangulation::set_mask(const Py::Tuple &args)
{
    _VERBOSE("Triangulation::set_mask");
    args.verify_length(1);

    Py_XDECREF(_mask);
    _mask = 0;
    if (args[0] != Py::None())
    {
        _mask = (PyArrayObject*)PyArray_ContiguousFromObject(
                    args[0].ptr(), PyArray_BOOL, 1, 1);
        if (_mask == 0 || PyArray_DIM(_mask,0) != PyArray_DIM(_triangles,0)) {
            Py_XDECREF(_mask);
            throw Py::ValueError(
                "mask must be a 1D array with the same length as the triangles array");
        }
    }

    // Clear derived fields so they are recalculated when needed.
     Py_XDECREF(_edges);
    _edges = 0;
    Py_XDECREF(_neighbors);
    _neighbors = 0;
    _boundaries.clear();

    return Py::None();
}

void Triangulation::write_boundaries() const
{
    const Boundaries& bs = get_boundaries();
    std::cout << "Number of boundaries: " << bs.size() << std::endl;
    for (Boundaries::const_iterator it = bs.begin(); it != bs.end(); ++it) {
        const Boundary& b = *it;
        std::cout << "  Boundary of " << b.size() << " points: ";
        for (Boundary::const_iterator itb = b.begin(); itb != b.end(); ++itb) {
            std::cout << *itb << ", ";
        }
        std::cout << std::endl;
    }
}




TriContourGenerator::TriContourGenerator(Py::Object triangulation,
                                         PyArrayObject* z)
    : _triangulation(triangulation),
      _z(z),
      _interior_visited(2*get_triangulation().get_ntri()),
      _boundaries_visited(0),
      _boundaries_used(0)
{
    _VERBOSE("TriContourGenerator::TriContourGenerator");
}

TriContourGenerator::~TriContourGenerator()
{
    _VERBOSE("TriContourGenerator::~TriContourGenerator");
    Py_XDECREF(_z);
}

void TriContourGenerator::clear_visited_flags(bool include_boundaries)
{
    // Clear _interiorVisited.
    std::fill(_interior_visited.begin(), _interior_visited.end(), false);

    if (include_boundaries) {
        if (_boundaries_visited.empty()) {
            const Boundaries& boundaries = get_boundaries();

            // Initialise _boundaries_visited.
            _boundaries_visited.reserve(boundaries.size());
            for (Boundaries::const_iterator it = boundaries.begin();
                    it != boundaries.end(); ++it)
                _boundaries_visited.push_back(BoundaryVisited(it->size()));

            // Initialise _boundaries_used.
            _boundaries_used = BoundariesUsed(boundaries.size());
        }

        // Clear _boundaries_visited.
        for (BoundariesVisited::iterator it = _boundaries_visited.begin();
                it != _boundaries_visited.end(); ++it)
            std::fill(it->begin(), it->end(), false);

        // Clear _boundaries_used.
        std::fill(_boundaries_used.begin(), _boundaries_used.end(), false);
    }
}

Py::Object TriContourGenerator::contour_to_segs(const Contour& contour)
{
    Py::List segs(contour.size());
    for (Contour::size_type i = 0; i < contour.size(); ++i) {
        const ContourLine& line = contour[i];
        npy_intp dims[2] = {static_cast<npy_intp>(line.size()),2};
        PyArrayObject* py_line = (PyArrayObject*)PyArray_SimpleNew(
                                                     2, dims, PyArray_DOUBLE);
        double* p = (double*)PyArray_DATA(py_line);
        for (ContourLine::const_iterator it = line.begin(); it != line.end(); ++it) {
            *p++ = it->x;
            *p++ = it->y;
        }
        segs[i] = Py::asObject((PyObject*)py_line);
    }
    return segs;
}

Py::Object TriContourGenerator::contour_to_segs_and_kinds(const Contour& contour)
{
    Contour::const_iterator line;
    ContourLine::const_iterator point;

    // Find total number of points in all contour lines.
    int n_points = 0;
    for (line = contour.begin(); line != contour.end(); ++line)
        n_points += line->size();

    // Create segs array for point coordinates.
    npy_intp segs_dims[2] = {n_points, 2};
    PyArrayObject* segs = (PyArrayObject*)PyArray_SimpleNew(
                                                2, segs_dims, PyArray_DOUBLE);
    double* segs_ptr = (double*)PyArray_DATA(segs);

    // Create kinds array for code types.
    npy_intp kinds_dims[1] = {n_points};
    PyArrayObject* kinds = (PyArrayObject*)PyArray_SimpleNew(
                                                1, kinds_dims, PyArray_UBYTE);
    unsigned char* kinds_ptr = (unsigned char*)PyArray_DATA(kinds);

    for (line = contour.begin(); line != contour.end(); ++line) {
        for (point = line->begin(); point != line->end(); point++) {
            *segs_ptr++ = point->x;
            *segs_ptr++ = point->y;
            *kinds_ptr++ = (point == line->begin() ? MOVETO : LINETO);
        }
    }

    Py::Tuple result(2);
    result[0] = Py::asObject((PyObject*)segs);
    result[1] = Py::asObject((PyObject*)kinds);
    return result;
}

Py::Object TriContourGenerator::create_contour(const Py::Tuple &args)
{
    _VERBOSE("TriContourGenerator::create_contour");
    args.verify_length(1);

    double level = (Py::Float)args[0];

    clear_visited_flags(false);
    Contour contour;

    find_boundary_lines(contour, level);
    find_interior_lines(contour, level, false, false);

    return contour_to_segs(contour);
}

Py::Object TriContourGenerator::create_filled_contour(const Py::Tuple &args)
{
    _VERBOSE("TriContourGenerator::create_filled_contour");
    args.verify_length(2);

    double lower_level = (Py::Float)args[0];
    double upper_level = (Py::Float)args[1];

    clear_visited_flags(true);
    Contour contour;

    find_boundary_lines_filled(contour, lower_level, upper_level);
    find_interior_lines(contour, lower_level, false, true);
    find_interior_lines(contour, upper_level, true,  true);

    return contour_to_segs_and_kinds(contour);
}

XY TriContourGenerator::edge_interp(int tri, int edge, const double& level)
{
    return interp(get_triangulation().get_triangle_point(tri, edge),
                  get_triangulation().get_triangle_point(tri, (edge+1)%3),
                  level);
}

void TriContourGenerator::find_boundary_lines(Contour& contour,
                                              const double& level)
{
    // Traverse boundaries to find starting points for all contour lines that
    // intersect the boundaries.  For each starting point found, follow the
    // line to its end before continuing.
    const Triangulation& triang = get_triangulation();
    const Boundaries& boundaries = get_boundaries();
    for (Boundaries::const_iterator it = boundaries.begin();
            it != boundaries.end(); ++it) {
        const Boundary& boundary = *it;
        bool startAbove, endAbove = false;
        for (Boundary::const_iterator itb = boundary.begin();
                itb != boundary.end(); ++itb) {
            if (itb == boundary.begin())
                startAbove = get_z(triang.get_triangle_point(*itb)) >= level;
            else
                startAbove = endAbove;
            endAbove = get_z(triang.get_triangle_point(itb->tri,
                                                       (itb->edge+1)%3)) >= level;
            if (startAbove && !endAbove) {
                // This boundary edge is the start point for a contour line,
                // so follow the line.
                contour.push_back(ContourLine());
                ContourLine& contour_line = contour.back();
                TriEdge tri_edge = *itb;
                follow_interior(contour_line, tri_edge, true, level, false);
            }
        }
    }
}

void TriContourGenerator::find_boundary_lines_filled(Contour& contour,
                                                     const double& lower_level,
                                                     const double& upper_level)
{
    // Traverse boundaries to find starting points for all contour lines that
    // intersect the boundaries.  For each starting point found, follow the
    // line to its end before continuing.
    const Triangulation& triang = get_triangulation();
    const Boundaries& boundaries = get_boundaries();
    for (Boundaries::size_type i = 0; i < boundaries.size(); ++i) {
        const Boundary& boundary = boundaries[i];
        for (Boundary::size_type j = 0; j < boundary.size(); ++j) {
            if (!_boundaries_visited[i][j]) {
                // z values of start and end of this boundary edge.
                double z_start = get_z(triang.get_triangle_point(boundary[j]));
                double z_end = get_z(triang.get_triangle_point(
                                   boundary[j].tri, (boundary[j].edge+1)%3));

                // Does this boundary edge's z increase through upper level
                // and/or decrease through lower level?
                bool incr_upper = (z_start < upper_level && z_end >= upper_level);
                bool decr_lower = (z_start >= lower_level && z_end < lower_level);

                if (decr_lower || incr_upper) {
                    // Start point for contour line, so follow it.
                    contour.push_back(ContourLine());
                    ContourLine& contour_line = contour.back();
                    TriEdge start_tri_edge = boundary[j];
                    TriEdge tri_edge = start_tri_edge;

                    // Traverse interior and boundaries until return to start.
                    bool on_upper = incr_upper;
                    do {
                        follow_interior(contour_line, tri_edge, true,
                            on_upper ? upper_level : lower_level, on_upper);
                        on_upper = follow_boundary(contour_line, tri_edge,
                                       lower_level, upper_level, on_upper);
                    } while (tri_edge != start_tri_edge);

                    // Filled contour lines must not have same first and last
                    // points.
                    if (contour_line.size() > 1 &&
                            contour_line.front() == contour_line.back())
                        contour_line.pop_back();
                }
            }
        }
    }

    // Add full boundaries that lie between the lower and upper levels.  These
    // are boundaries that have not been touched by an internal contour line
    // which are stored in _boundaries_used.
    for (Boundaries::size_type i = 0; i < boundaries.size(); ++i) {
        if (!_boundaries_used[i]) {
            const Boundary& boundary = boundaries[i];
            double z = get_z(triang.get_triangle_point(boundary[0]));
            if (z >= lower_level && z < upper_level) {
                contour.push_back(ContourLine());
                ContourLine& contour_line = contour.back();
                for (Boundary::size_type j = 0; j < boundary.size(); ++j)
                    contour_line.push_back(triang.get_point_coords(
                                      triang.get_triangle_point(boundary[j])));
            }
        }
    }
}

void TriContourGenerator::find_interior_lines(Contour& contour,
                                              const double& level,
                                              bool on_upper,
                                              bool filled)
{
    const Triangulation& triang = get_triangulation();
    int ntri = triang.get_ntri();
    for (int tri = 0; tri < ntri; ++tri) {
        int visited_index = (on_upper ? tri+ntri : tri);

        if (_interior_visited[visited_index] || triang.is_masked(tri))
            continue;  // Triangle has already been visited or is masked.

        _interior_visited[visited_index] = true;

        // Determine edge via which to leave this triangle.
        int edge = get_exit_edge(tri, level, on_upper);
        assert(edge >= -1 && edge < 3 && "Invalid exit edge");
        if (edge == -1)
            continue;  // Contour does not pass through this triangle.

        // Found start of new contour line loop.
        contour.push_back(ContourLine());
        ContourLine& contour_line = contour.back();
        TriEdge tri_edge = triang.get_neighbor_edge(tri, edge);
        follow_interior(contour_line, tri_edge, false, level, on_upper);

        if (!filled)
            // Non-filled contour lines must be closed.
            contour_line.push_back(contour_line.front());
        else if (contour_line.size() > 1 &&
                 contour_line.front() == contour_line.back())
            // Filled contour lines must not have same first and last points.
            contour_line.pop_back();
    }
}

bool TriContourGenerator::follow_boundary(ContourLine& contour_line,
                                          TriEdge& tri_edge,
                                          const double& lower_level,
                                          const double& upper_level,
                                          bool on_upper)
{
    const Triangulation& triang = get_triangulation();
    const Boundaries& boundaries = get_boundaries();

    // Have TriEdge to start at, need equivalent boundary edge.
    int boundary, edge;
    triang.get_boundary_edge(tri_edge, boundary, edge);
    _boundaries_used[boundary] = true;

    bool stop = false;
    bool first_edge = true;
    double z_start, z_end = 0;
    while (!stop)
    {
        assert(!_boundaries_visited[boundary][edge] && "Boundary already visited");
        _boundaries_visited[boundary][edge] = true;

        // z values of start and end points of boundary edge.
        if (first_edge)
            z_start = get_z(triang.get_triangle_point(tri_edge));
        else
            z_start = z_end;
        z_end = get_z(triang.get_triangle_point(tri_edge.tri,
                                                (tri_edge.edge+1)%3));

        if (z_end > z_start) {  // z increasing.
            if (!(!on_upper && first_edge) &&
                z_end >= lower_level && z_start < lower_level) {
                stop = true;
                on_upper = false;
            } else if (z_end >= upper_level && z_start < upper_level) {
                stop = true;
                on_upper = true;
            }
        } else {  // z decreasing.
            if (!(on_upper && first_edge) &&
                z_start >= upper_level && z_end < upper_level) {
                stop = true;
                on_upper = true;
            } else if (z_start >= lower_level && z_end < lower_level) {
                stop = true;
                on_upper = false;
            }
        }

        first_edge = false;

        if (!stop) {
            // Move to next boundary edge, adding point to contour line.
            edge = (edge+1) % (int)boundaries[boundary].size();
            tri_edge = boundaries[boundary][edge];
            contour_line.push_back(triang.get_point_coords(
                                       triang.get_triangle_point(tri_edge)));
        }
    }

    return on_upper;
}

void TriContourGenerator::follow_interior(ContourLine& contour_line,
                                          TriEdge& tri_edge,
                                          bool end_on_boundary,
                                          const double& level,
                                          bool on_upper)
{
    int& tri = tri_edge.tri;
    int& edge = tri_edge.edge;

    // Initial point.
    contour_line.push_back(edge_interp(tri, edge, level));

    while (true) {
        int visited_index = tri;
        if (on_upper)
            visited_index += get_triangulation().get_ntri();

        // Check for end not on boundary.
        if (!end_on_boundary && _interior_visited[visited_index])
            break;  // Reached start point, so return.

        // Determine edge by which to leave this triangle.
        edge = get_exit_edge(tri, level, on_upper);
        assert(edge >= 0 && edge < 3 && "Invalid exit edge");

        _interior_visited[visited_index] = true;

        // Append new point to point set.
        assert(edge >= 0 && edge < 3 && "Invalid triangle edge");
        contour_line.push_back(edge_interp(tri, edge, level));

        // Move to next triangle.
        TriEdge next_tri_edge = get_triangulation().get_neighbor_edge(tri,edge);

        // Check if ending on a boundary.
        if (end_on_boundary && next_tri_edge.tri == -1)
            break;

        tri_edge = next_tri_edge;
        assert(tri_edge.tri != -1 && "Invalid triangle for internal loop");
    }
}

const TriContourGenerator::Boundaries& TriContourGenerator::get_boundaries() const
{
    return get_triangulation().get_boundaries();
}

int TriContourGenerator::get_exit_edge(int tri,
                                       const double& level,
                                       bool on_upper) const
{
    assert(tri >= 0 && tri < get_triangulation().get_ntri() &&
           "Triangle index out of bounds.");

    unsigned int config =
        (get_z(get_triangulation().get_triangle_point(tri, 0)) >= level) |
        (get_z(get_triangulation().get_triangle_point(tri, 1)) >= level) << 1 |
        (get_z(get_triangulation().get_triangle_point(tri, 2)) >= level) << 2;

    if (on_upper) config = 7-config;

    switch (config) {
        case 0: return -1;
        case 1: return  2;
        case 2: return  0;
        case 3: return  2;
        case 4: return  1;
        case 5: return  1;
        case 6: return  0;
        case 7: return -1;
        default: assert(0 && "Invalid config value"); return -1;
    }
}

const Triangulation& TriContourGenerator::get_triangulation() const
{
    return *(Triangulation*)_triangulation.ptr();
}

const double& TriContourGenerator::get_z(int point) const
{
    assert(point >= 0 && point < get_triangulation().get_npoints() &&
           "Point index out of bounds.");
    return ((const double*)PyArray_DATA(_z))[point];
}

void TriContourGenerator::init_type()
{
    _VERBOSE("TriContourGenerator::init_type");

    behaviors().name("TriContourGenerator");
    behaviors().doc("TriContourGenerator");

    add_varargs_method("create_contour",
                       &TriContourGenerator::create_contour,
                       "create_contour(level)");
    add_varargs_method("create_filled_contour",
                       &TriContourGenerator::create_filled_contour,
                       "create_filled_contour(lower_level, upper_level)");
}

XY TriContourGenerator::interp(int point1,
                               int point2,
                               const double& level) const
{
    assert(point1 >= 0 && point1 < get_triangulation().get_npoints() &&
           "Point index 1 out of bounds.");
    assert(point2 >= 0 && point2 < get_triangulation().get_npoints() &&
           "Point index 2 out of bounds.");
    assert(point1 != point2 && "Identical points");
    double fraction = (get_z(point2) - level) / (get_z(point2) - get_z(point1));
    return get_triangulation().get_point_coords(point1)*fraction +
           get_triangulation().get_point_coords(point2)*(1.0 - fraction);
}




TrapezoidMapTriFinder::TrapezoidMapTriFinder(Py::Object triangulation)
    : _triangulation(triangulation),
      _points(0),
      _tree(0)
{
    _VERBOSE("TrapezoidMapTriFinder::TrapezoidMapTriFinder");
}

TrapezoidMapTriFinder::~TrapezoidMapTriFinder()
{
    _VERBOSE("TrapezoidMapTriFinder::~TrapezoidMapTriFinder");
    clear();
}

bool
TrapezoidMapTriFinder::add_edge_to_tree(const Edge& edge)
{
    std::vector<Trapezoid*> trapezoids;
    if (!find_trapezoids_intersecting_edge(edge, trapezoids))
        return false;
    assert(!trapezoids.empty() && "No trapezoids intersect edge");

    const Point* p = edge.left;
    const Point* q = edge.right;
    Trapezoid* left_old = 0;    // old trapezoid to the left.
    Trapezoid* left_below = 0;  // below trapezoid to the left.
    Trapezoid* left_above = 0;  // above trapezoid to the left.

    // Iterate through trapezoids intersecting edge from left to right.
    // Replace each old trapezoid with 2+ new trapezoids, and replace its
    // corresponding nodes in the search tree with new nodes.
    unsigned int ntraps = trapezoids.size();
    for (unsigned int i = 0; i < ntraps; ++i) {
        Trapezoid* old = trapezoids[i];  // old trapezoid to replace.
        bool start_trap = (i == 0);
        bool end_trap = (i == ntraps-1);
        bool have_left = (start_trap && edge.left != old->left);
        bool have_right = (end_trap && edge.right != old->right);

        // Old trapezoid is replaced by up to 4 new trapezoids: left is to the
        // left of the start point p, below/above are below/above the edge
        // inserted, and right is to the right of the end point q.
        Trapezoid* left = 0;
        Trapezoid* below = 0;
        Trapezoid* above = 0;
        Trapezoid* right = 0;

        // There are 4 different cases here depending on whether the old
        // trapezoid in question is the start and/or end trapezoid of those
        // that intersect the edge inserted.  There is some code duplication
        // here but it is much easier to understand this way rather than
        // interleave the 4 different cases with many more if-statements.
        if (start_trap && end_trap) {
            // Edge intersects a single trapezoid.
            if (have_left)
                left = new Trapezoid(old->left, p, old->below, old->above);
            below = new Trapezoid(p, q, old->below, edge);
            above = new Trapezoid(p, q, edge, old->above);
            if (have_right)
                right = new Trapezoid(q, old->right, old->below, old->above);

            // Set pairs of trapezoid neighbours.
            if (have_left) {
                left->set_lower_left(old->lower_left);
                left->set_upper_left(old->upper_left);
                left->set_lower_right(below);
                left->set_upper_right(above);
            }
            else {
                below->set_lower_left(old->lower_left);
                above->set_upper_left(old->upper_left);
            }

            if (have_right) {
                right->set_lower_right(old->lower_right);
                right->set_upper_right(old->upper_right);
                below->set_lower_right(right);
                above->set_upper_right(right);
            }
            else {
                below->set_lower_right(old->lower_right);
                above->set_upper_right(old->upper_right);
            }
        }
        else if (start_trap) {
            // Old trapezoid is the first of 2+ trapezoids that the edge
            // intersects.
            if (have_left)
                left = new Trapezoid(old->left, p, old->below, old->above);
            below = new Trapezoid(p, old->right, old->below, edge);
            above = new Trapezoid(p, old->right, edge, old->above);

            // Set pairs of trapezoid neighbours.
            if (have_left) {
                left->set_lower_left(old->lower_left);
                left->set_upper_left(old->upper_left);
                left->set_lower_right(below);
                left->set_upper_right(above);
            }
            else {
                below->set_lower_left(old->lower_left);
                above->set_upper_left(old->upper_left);
            }

            below->set_lower_right(old->lower_right);
            above->set_upper_right(old->upper_right);
        }
        else if (end_trap) {
            // Old trapezoid is the last of 2+ trapezoids that the edge
            // intersects.
            if (left_below->below == old->below) {
                below = left_below;
                below->right = q;
            }
            else
                below = new Trapezoid(old->left, q, old->below, edge);

            if (left_above->above == old->above) {
                above = left_above;
                above->right = q;
            }
            else
                above = new Trapezoid(old->left, q, edge, old->above);

            if (have_right)
                right = new Trapezoid(q, old->right, old->below, old->above);

            // Set pairs of trapezoid neighbours.
            if (have_right) {
                right->set_lower_right(old->lower_right);
                right->set_upper_right(old->upper_right);
                below->set_lower_right(right);
                above->set_upper_right(right);
            }
            else {
                below->set_lower_right(old->lower_right);
                above->set_upper_right(old->upper_right);
            }

            // Connect to new trapezoids replacing prevOld.
            if (below != left_below) {
                below->set_upper_left(left_below);
                if (old->lower_left == left_old)
                    below->set_lower_left(left_below);
                else
                    below->set_lower_left(old->lower_left);
            }

            if (above != left_above) {
                above->set_lower_left(left_above);
                if (old->upper_left == left_old)
                    above->set_upper_left(left_above);
                else
                    above->set_upper_left(old->upper_left);
            }
        }
        else {  // Middle trapezoid.
            // Old trapezoid is neither the first nor last of the 3+ trapezoids
            // that the edge intersects.
            if (left_below->below == old->below) {
                below = left_below;
                below->right = old->right;
            }
            else
                below = new Trapezoid(old->left, old->right, old->below, edge);

            if (left_above->above == old->above) {
                above = left_above;
                above->right = old->right;
            }
            else
                above = new Trapezoid(old->left, old->right, edge, old->above);

            // Connect to new trapezoids replacing prevOld.
            if (below != left_below) {  // below is new.
                below->set_upper_left(left_below);
                if (old->lower_left == left_old)
                    below->set_lower_left(left_below);
                else
                    below->set_lower_left(old->lower_left);
            }

            if (above != left_above) {  // above is new.
                above->set_lower_left(left_above);
                if (old->upper_left == left_old)
                    above->set_upper_left(left_above);
                else
                    above->set_upper_left(old->upper_left);
            }

            below->set_lower_right(old->lower_right);
            above->set_upper_right(old->upper_right);
        }

        // Create new nodes to add to search tree.  Below and above trapezoids
        // may already have owning trapezoid nodes, in which case reuse them.
        Node* new_top_node = new Node(
            &edge,
            below == left_below ? below->trapezoid_node : new Node(below),
            above == left_above ? above->trapezoid_node : new Node(above));
        if (have_right)
            new_top_node = new Node(q, new_top_node, new Node(right));
        if (have_left)
            new_top_node = new Node(p, new Node(left), new_top_node);

        // Insert new_top_node in correct position or positions in search tree.
        Node* old_node = old->trapezoid_node;
        if (old_node == _tree)
            _tree = new_top_node;
        else
            old_node->replace_with(new_top_node);

        // old_node has been removed from all of its parents and is no longer
        // needed.
        assert(old_node->has_no_parents() && "Node should have no parents");
        delete old_node;

        // Clearing up.
        if (!end_trap) {
            // Prepare for next loop.
            left_old = old;
            left_above = above;
            left_below = below;
        }
    }

    return true;
}

void
TrapezoidMapTriFinder::clear()
{
    delete [] _points;
    _points = 0;

    _edges.clear();

    delete _tree;
    _tree = 0;
}

Py::Object
TrapezoidMapTriFinder::find_many(const Py::Tuple& args)
{
    args.verify_length(2);

    // Check input arguments.
    PyArrayObject* x = (PyArrayObject*)PyArray_ContiguousFromObject(
                           args[0].ptr(), PyArray_DOUBLE, 0, 0);
    PyArrayObject* y = (PyArrayObject*)PyArray_ContiguousFromObject(
                           args[1].ptr(), PyArray_DOUBLE, 0, 0);
    bool ok = (x != 0 && y != 0 && PyArray_NDIM(x) == PyArray_NDIM(y));
    int ndim = PyArray_NDIM(x);
    for (int i = 0; ok && i < ndim; ++i)
        ok = (PyArray_DIM(x,i) == PyArray_DIM(y,i));

    if (!ok) {
        Py_XDECREF(x);
        Py_XDECREF(y);
        throw Py::ValueError("x and y must be array_like with same shape");
    }

    // Create integer array to return.
    PyArrayObject* tri = (PyArrayObject*)PyArray_SimpleNew(
                             ndim, PyArray_DIMS(x), PyArray_INT);

    // Fill returned array.
    double* x_ptr = (double*)PyArray_DATA(x);
    double* y_ptr = (double*)PyArray_DATA(y);
    int* tri_ptr = (int*)PyArray_DATA(tri);
    int* tri_end = tri_ptr + PyArray_SIZE(tri);
    while (tri_ptr < tri_end)
        *tri_ptr++ = find_one(XY(*x_ptr++, *y_ptr++));

    Py_XDECREF(x);
    Py_XDECREF(y);
    return Py::asObject((PyObject*)tri);
}

int
TrapezoidMapTriFinder::find_one(const XY& xy)
{
    const Node* node = _tree->search(xy);
    assert(node != 0 && "Search tree for point returned null node");
    return node->get_tri();
}

bool
TrapezoidMapTriFinder::find_trapezoids_intersecting_edge(
    const Edge& edge,
    std::vector<Trapezoid*>& trapezoids)
{
    // This is the FollowSegment algorithm of de Berg et al, with some extra
    // checks to deal with simple colinear (i.e. invalid) triangles.
    trapezoids.clear();
    Trapezoid* trapezoid = _tree->search(edge);
    if (trapezoid == 0) {
        assert(trapezoid != 0 && "search(edge) returns null trapezoid");
        return false;
    }

    trapezoids.push_back(trapezoid);
    while (edge.right->is_right_of(*trapezoid->right)) {
        int orient = edge.get_point_orientation(*trapezoid->right);
        if (orient == 0) {
            if (edge.point_below == trapezoid->right)
                orient = +1;
            else if (edge.point_above == trapezoid->right)
                orient = -1;
            else {
                assert(0 && "Unable to deal with point on edge");
                return false;
            }
        }

        if (orient == -1)
            trapezoid = trapezoid->lower_right;
        else if (orient == +1)
            trapezoid = trapezoid->upper_right;

        if (trapezoid == 0) {
            assert(0 && "Expected trapezoid neighbor");
            return false;
        }
        trapezoids.push_back(trapezoid);
    }

    return true;
}

Py::Object
TrapezoidMapTriFinder::get_tree_stats()
{
    _VERBOSE("TrapezoidMapTriFinder::get_tree_stats");

    NodeStats stats;
    _tree->get_stats(0, stats);

    Py::List list(7);
    list[0] = Py::Int(stats.node_count);
    list[1] = Py::Int(static_cast<long>(stats.unique_nodes.size()));
    list[2] = Py::Int(stats.trapezoid_count);
    list[3] = Py::Int(static_cast<long>(stats.unique_trapezoid_nodes.size()));
    list[4] = Py::Int(stats.max_parent_count);
    list[5] = Py::Int(stats.max_depth);
    list[6] = Py::Float(stats.sum_trapezoid_depth / stats.trapezoid_count);
    return list;
}

const Triangulation&
TrapezoidMapTriFinder::get_triangulation() const
{
    return *(Triangulation*)_triangulation.ptr();
}

void
TrapezoidMapTriFinder::init_type()
{
    _VERBOSE("TrapezoidMapTriFinder::init_type");

    behaviors().name("TrapezoidMapTriFinder");
    behaviors().doc("TrapezoidMapTriFinder");

    add_varargs_method("find_many",
                       &TrapezoidMapTriFinder::find_many,
                       "find_many(x,y)");
    add_noargs_method("get_tree_stats",
                      &TrapezoidMapTriFinder::get_tree_stats,
                      "get_tree_stats()");
    add_noargs_method("initialize",
                      &TrapezoidMapTriFinder::initialize,
                      "initialize()");
    add_noargs_method("print_tree",
                      &TrapezoidMapTriFinder::print_tree,
                      "print_tree()");
}

Py::Object
TrapezoidMapTriFinder::initialize()
{
    _VERBOSE("TrapezoidMapTriFinder::initialize");

    clear();
    const Triangulation& triang = get_triangulation();

    // Set up points array, which contains all of the points in the
    // triangulation plus the 4 corners of the enclosing rectangle.
    int npoints = triang.get_npoints();
    _points = new Point[npoints + 4];
    BoundingBox bbox;
    for (int i = 0; i < npoints; ++i) {
        XY xy = triang.get_point_coords(i);
        // Avoid problems with -0.0 values different from 0.0
        if (xy.x == -0.0)
            xy.x = 0.0;
        if (xy.y == -0.0)
            xy.y = 0.0;
        _points[i] = Point(xy);
        bbox.add(xy);
    }

    // Last 4 points are corner points of enclosing rectangle.  Enclosing
    // rectangle made slightly larger in case corner points are already in the
    // triangulation.
    if (bbox.empty) {
        bbox.add(XY(0.0, 0.0));
        bbox.add(XY(1.0, 1.0));
    }
    else {
        const double small = 0.1;  // Any value > 0.0
        bbox.expand( (bbox.upper - bbox.lower)*small );
    }
    _points[npoints  ] = Point(bbox.lower);                  // SW point.
    _points[npoints+1] = Point(bbox.upper.x, bbox.lower.y);  // SE point.
    _points[npoints+2] = Point(bbox.lower.x, bbox.upper.y);  // NW point.
    _points[npoints+3] = Point(bbox.upper);                  // NE point.

    // Set up edges array.
    // First the bottom and top edges of the enclosing rectangle.
    _edges.push_back(Edge(&_points[npoints],  &_points[npoints+1],-1,-1,0,0));
    _edges.push_back(Edge(&_points[npoints+2],&_points[npoints+3],-1,-1,0,0));

    // Add all edges in the triangulation that point to the right.  Do not
    // explicitly include edges that point to the left as the neighboring
    // triangle will supply that, unless there is no such neighbor.
    int ntri = triang.get_ntri();
    for (int tri = 0; tri < ntri; ++tri) {
        if (!triang.is_masked(tri)) {
            for (int edge = 0; edge < 3; ++edge) {
                Point* start = _points + triang.get_triangle_point(tri,edge);
                Point* end   = _points +
                               triang.get_triangle_point(tri,(edge+1)%3);
                Point* other = _points +
                               triang.get_triangle_point(tri,(edge+2)%3);
                TriEdge neighbor = triang.get_neighbor_edge(tri,edge);
                if (end->is_right_of(*start)) {
                    const Point* neighbor_point_below = (neighbor.tri == -1) ?
                        0 : _points + triang.get_triangle_point(
                                          neighbor.tri, (neighbor.edge+2)%3);
                    _edges.push_back(Edge(start, end, neighbor.tri, tri,
                                          neighbor_point_below, other));
                }
                else if (neighbor.tri == -1)
                    _edges.push_back(Edge(end, start, tri, -1, other, 0));

                // Set triangle associated with start point if not already set.
                if (start->tri == -1)
                    start->tri = tri;
            }
        }
    }

    // Initial trapezoid is enclosing rectangle.
    _tree = new Node(new Trapezoid(&_points[npoints], &_points[npoints+1],
                                   _edges[0], _edges[1]));
    _tree->assert_valid(false);

    // Randomly shuffle all edges other than first 2.
    RandomNumberGenerator rng(1234);
    std::random_shuffle(_edges.begin()+2, _edges.end(), rng);

    // Add edges, one at a time, to tree.
    unsigned int nedges = _edges.size();
    for (unsigned int index = 2; index < nedges; ++index) {
        if (!add_edge_to_tree(_edges[index]))
            throw Py::RuntimeError("Triangulation is invalid");
        _tree->assert_valid(index == nedges-1);
    }

    return Py::None();
}

Py::Object
TrapezoidMapTriFinder::print_tree()
{
    _VERBOSE("TrapezoidMapTriFinder::print_tree");

    assert(_tree != 0 && "Null Node tree");
    _tree->print();

    return Py::None();
}

TrapezoidMapTriFinder::Edge::Edge(const Point* left_,
                                  const Point* right_,
                                  int triangle_below_,
                                  int triangle_above_,
                                  const Point* point_below_,
                                  const Point* point_above_)
    : left(left_),
      right(right_),
      triangle_below(triangle_below_),
      triangle_above(triangle_above_),
      point_below(point_below_),
      point_above(point_above_)
{
    assert(left != 0 && "Null left point");
    assert(right != 0 && "Null right point");
    assert(right->is_right_of(*left) && "Incorrect point order");
    assert(triangle_below >= -1 && "Invalid triangle below index");
    assert(triangle_above >= -1 && "Invalid triangle above index");
}

int
TrapezoidMapTriFinder::Edge::get_point_orientation(const XY& xy) const
{
    double cross_z = (xy - *left).cross_z(*right - *left);
    return (cross_z > 0.0) ? +1 : ((cross_z < 0.0) ? -1 : 0);
}

double
TrapezoidMapTriFinder::Edge::get_slope() const
{
    // Divide by zero is acceptable here.
    XY diff = *right - *left;
    return diff.y / diff.x;
}

double
TrapezoidMapTriFinder::Edge::get_y_at_x(const double& x) const
{
    if (left->x == right->x) {
        // If edge is vertical, return lowest y from left point.
        assert(x == left->x && "x outside of edge");
        return left->y;
    }
    else {
        // Equation of line: left + lambda*(right - left) = xy.
        // i.e. left.x + lambda(right.x - left.x) = x and similar for y.
        double lambda = (x - left->x) / (right->x - left->x);
        assert(lambda >= 0 && lambda <= 1.0 && "Lambda out of bounds");
        return left->y + lambda*(right->y - left->y);
    }
}

bool
TrapezoidMapTriFinder::Edge::has_point(const Point* point) const
{
    assert(point != 0 && "Null point");
    return (left == point || right == point);
}

bool
TrapezoidMapTriFinder::Edge::operator==(const Edge& other) const
{
    return this == &other;
}

void
TrapezoidMapTriFinder::Edge::print_debug() const
{
    std::cout << "Edge " << *this << " tri_below=" << triangle_below
        << " tri_above=" << triangle_above << std::endl;
}

TrapezoidMapTriFinder::Node::Node(const Point* point, Node* left, Node* right)
    : _type(Type_XNode)
{
    assert(point != 0 && "Invalid point");
    assert(left != 0 && "Invalid left node");
    assert(right != 0 && "Invalid right node");
    _union.xnode.point = point;
    _union.xnode.left = left;
    _union.xnode.right = right;
    left->add_parent(this);
    right->add_parent(this);
}

TrapezoidMapTriFinder::Node::Node(const Edge* edge, Node* below, Node* above)
    : _type(Type_YNode)
{
    assert(edge != 0 && "Invalid edge");
    assert(below != 0 && "Invalid below node");
    assert(above != 0 && "Invalid above node");
    _union.ynode.edge = edge;
    _union.ynode.below = below;
    _union.ynode.above = above;
    below->add_parent(this);
    above->add_parent(this);
}

TrapezoidMapTriFinder::Node::Node(Trapezoid* trapezoid)
    : _type(Type_TrapezoidNode)
{
    assert(trapezoid != 0 && "Null Trapezoid");
    _union.trapezoid = trapezoid;
    trapezoid->trapezoid_node = this;
}

TrapezoidMapTriFinder::Node::~Node()
{
    switch (_type) {
        case Type_XNode:
            if (_union.xnode.left->remove_parent(this))
                delete _union.xnode.left;
            if (_union.xnode.right->remove_parent(this))
                delete _union.xnode.right;
            break;
        case Type_YNode:
            if (_union.ynode.below->remove_parent(this))
                delete _union.ynode.below;
            if (_union.ynode.above->remove_parent(this))
                delete _union.ynode.above;
            break;
        case Type_TrapezoidNode:
            delete _union.trapezoid;
            break;
    }
}

void
TrapezoidMapTriFinder::Node::add_parent(Node* parent)
{
    assert(parent != 0 && "Null parent");
    assert(parent != this && "Cannot be parent of self");
    assert(!has_parent(parent) && "Parent already in collection");
    _parents.push_back(parent);
}

void
TrapezoidMapTriFinder::Node::assert_valid(bool tree_complete) const
{
#ifndef NDEBUG
    // Check parents.
    for (Parents::const_iterator it = _parents.begin();
         it != _parents.end(); ++it) {
        Node* parent = *it;
        assert(parent != this && "Cannot be parent of self");
        assert(parent->has_child(this) && "Parent missing child");
    }

    // Check children, and recurse.
    switch (_type) {
        case Type_XNode:
            assert(_union.xnode.left != 0 && "Null left child");
            assert(_union.xnode.left->has_parent(this) && "Incorrect parent");
            assert(_union.xnode.right != 0 && "Null right child");
            assert(_union.xnode.right->has_parent(this) && "Incorrect parent");
            _union.xnode.left->assert_valid(tree_complete);
            _union.xnode.right->assert_valid(tree_complete);
            break;
        case Type_YNode:
            assert(_union.ynode.below != 0 && "Null below child");
            assert(_union.ynode.below->has_parent(this) && "Incorrect parent");
            assert(_union.ynode.above != 0 && "Null above child");
            assert(_union.ynode.above->has_parent(this) && "Incorrect parent");
            _union.ynode.below->assert_valid(tree_complete);
            _union.ynode.above->assert_valid(tree_complete);
            break;
        case Type_TrapezoidNode:
            assert(_union.trapezoid != 0 && "Null trapezoid");
            assert(_union.trapezoid->trapezoid_node == this &&
                   "Incorrect trapezoid node");
            _union.trapezoid->assert_valid(tree_complete);
            break;
    }
#endif
}

void
TrapezoidMapTriFinder::Node::get_stats(int depth,
                                       NodeStats& stats) const
{
    stats.node_count++;
    if (depth > stats.max_depth)
        stats.max_depth = depth;
    bool new_node = stats.unique_nodes.insert(this).second;
    if (new_node)
        stats.max_parent_count = std::max(stats.max_parent_count,
                                          static_cast<long>(_parents.size()));

    switch (_type) {
        case Type_XNode:
            _union.xnode.left->get_stats(depth+1, stats);
            _union.xnode.right->get_stats(depth+1, stats);
            break;
        case Type_YNode:
            _union.ynode.below->get_stats(depth+1, stats);
            _union.ynode.above->get_stats(depth+1, stats);
            break;
        default:  // Type_TrapezoidNode:
            stats.unique_trapezoid_nodes.insert(this);
            stats.trapezoid_count++;
            stats.sum_trapezoid_depth += depth;
            break;
    }
}

int
TrapezoidMapTriFinder::Node::get_tri() const
{
    switch (_type) {
        case Type_XNode:
            return _union.xnode.point->tri;
        case Type_YNode:
            if (_union.ynode.edge->triangle_above != -1)
                return _union.ynode.edge->triangle_above;
            else
                return _union.ynode.edge->triangle_below;
        default:  // Type_TrapezoidNode:
            assert(_union.trapezoid->below.triangle_above ==
                   _union.trapezoid->above.triangle_below &&
                   "Inconsistent triangle indices from trapezoid edges");
            return _union.trapezoid->below.triangle_above;
    }
}

bool
TrapezoidMapTriFinder::Node::has_child(const Node* child) const
{
    assert(child != 0 && "Null child node");
    switch (_type) {
        case Type_XNode:
            return (_union.xnode.left == child || _union.xnode.right == child);
        case Type_YNode:
            return (_union.ynode.below == child ||
                    _union.ynode.above == child);
        default:  // Type_TrapezoidNode:
            return false;
    }
}

bool
TrapezoidMapTriFinder::Node::has_no_parents() const
{
    return _parents.empty();
}

bool
TrapezoidMapTriFinder::Node::has_parent(const Node* parent) const
{
    return (std::find(_parents.begin(), _parents.end(), parent) !=
            _parents.end());
}

void
TrapezoidMapTriFinder::Node::print(int depth /* = 0 */) const
{
    for (int i = 0; i < depth; ++i) std::cout << "  ";
    switch (_type) {
        case Type_XNode:
            std::cout << "XNode " << *_union.xnode.point << std::endl;
            _union.xnode.left->print(depth + 1);
            _union.xnode.right->print(depth + 1);
            break;
        case Type_YNode:
            std::cout << "YNode " << *_union.ynode.edge << std::endl;
            _union.ynode.below->print(depth + 1);
            _union.ynode.above->print(depth + 1);
            break;
        case Type_TrapezoidNode:
            std::cout << "Trapezoid ll="
                << _union.trapezoid->get_lower_left_point()  << " lr="
                << _union.trapezoid->get_lower_right_point() << " ul="
                << _union.trapezoid->get_upper_left_point()  << " ur="
                << _union.trapezoid->get_upper_right_point() << std::endl;
            break;
    }
}

bool
TrapezoidMapTriFinder::Node::remove_parent(Node* parent)
{
    assert(parent != 0 && "Null parent");
    assert(parent != this && "Cannot be parent of self");
    Parents::iterator it = std::find(_parents.begin(), _parents.end(), parent);
    assert(it != _parents.end() && "Parent not in collection");
    _parents.erase(it);
    return _parents.empty();
}

void
TrapezoidMapTriFinder::Node::replace_child(Node* old_child, Node* new_child)
{
    switch (_type) {
        case Type_XNode:
            assert((_union.xnode.left == old_child ||
                    _union.xnode.right == old_child) && "Not a child Node");
            assert(new_child != 0 && "Null child node");
            if (_union.xnode.left == old_child)
                _union.xnode.left = new_child;
            else
                _union.xnode.right = new_child;
            break;
        case Type_YNode:
            assert((_union.ynode.below == old_child ||
                    _union.ynode.above == old_child) && "Not a child node");
            assert(new_child != 0 && "Null child node");
            if (_union.ynode.below == old_child)
                _union.ynode.below = new_child;
            else
                _union.ynode.above = new_child;
            break;
        case Type_TrapezoidNode:
            assert(0 && "Invalid type for this operation");
            break;
    }
    old_child->remove_parent(this);
    new_child->add_parent(this);
}

void
TrapezoidMapTriFinder::Node::replace_with(Node* new_node)
{
    assert(new_node != 0 && "Null replacement node");
    // Replace child of each parent with new_node.  As each has parent has its
    // child replaced it is removed from the _parents collection.
    while (!_parents.empty())
        _parents.front()->replace_child(this, new_node);
}

const TrapezoidMapTriFinder::Node*
TrapezoidMapTriFinder::Node::search(const XY& xy)
{
    switch (_type) {
        case Type_XNode:
            if (xy == *_union.xnode.point)
                return this;
            else if (xy.is_right_of(*_union.xnode.point))
                return _union.xnode.right->search(xy);
            else
                return _union.xnode.left->search(xy);
        case Type_YNode: {
            int orient = _union.ynode.edge->get_point_orientation(xy);
            if (orient == 0)
                return this;
            else if (orient < 0)
                return _union.ynode.above->search(xy);
            else
                return _union.ynode.below->search(xy);
        }
        default:  // Type_TrapezoidNode:
            return this;
    }
}

TrapezoidMapTriFinder::Trapezoid*
TrapezoidMapTriFinder::Node::search(const Edge& edge)
{
    switch (_type) {
        case Type_XNode:
            if (edge.left == _union.xnode.point)
                return _union.xnode.right->search(edge);
            else {
                if (edge.left->is_right_of(*_union.xnode.point))
                    return _union.xnode.right->search(edge);
                else
                    return _union.xnode.left->search(edge);
            }
        case Type_YNode:
            if (edge.left == _union.ynode.edge->left) {
                // Coinciding left edge points.
                if (edge.get_slope() == _union.ynode.edge->get_slope()) {
                    if (_union.ynode.edge->triangle_above ==
                        edge.triangle_below)
                        return _union.ynode.above->search(edge);
                    else if (_union.ynode.edge->triangle_below ==
                             edge.triangle_above)
                        return _union.ynode.below->search(edge);
                    else {
                        assert(0 &&
                               "Invalid triangulation, common left points");
                        return 0;
                    }
                }
                if (edge.get_slope() > _union.ynode.edge->get_slope())
                    return _union.ynode.above->search(edge);
                else
                    return _union.ynode.below->search(edge);
            }
            else if (edge.right == _union.ynode.edge->right) {
                // Coinciding right edge points.
                if (edge.get_slope() == _union.ynode.edge->get_slope()) {
                    if (_union.ynode.edge->triangle_above ==
                        edge.triangle_below)
                        return _union.ynode.above->search(edge);
                    else if (_union.ynode.edge->triangle_below ==
                             edge.triangle_above)
                        return _union.ynode.below->search(edge);
                    else {
                        assert(0 &&
                               "Invalid triangulation, common right points");
                        return 0;
                    }
                }
                if (edge.get_slope() > _union.ynode.edge->get_slope())
                    return _union.ynode.below->search(edge);
                else
                    return _union.ynode.above->search(edge);
            }
            else {
                int orient =
                    _union.ynode.edge->get_point_orientation(*edge.left);
                if (orient == 0) {
                    // edge.left lies on _union.ynode.edge
                    if (_union.ynode.edge->point_above != 0 &&
                        edge.has_point(_union.ynode.edge->point_above))
                        orient = -1;
                    else if (_union.ynode.edge->point_below != 0 &&
                             edge.has_point(_union.ynode.edge->point_below))
                        orient = +1;
                    else {
                        assert(0 && "Invalid triangulation, point on edge");
                        return 0;
                    }
                }
                if (orient < 0)
                    return _union.ynode.above->search(edge);
                else
                    return _union.ynode.below->search(edge);
            }
        default:  // Type_TrapezoidNode:
            return _union.trapezoid;
    }
}

TrapezoidMapTriFinder::Trapezoid::Trapezoid(const Point* left_,
                                            const Point* right_,
                                            const Edge& below_,
                                            const Edge& above_)
    : left(left_), right(right_), below(below_), above(above_),
      lower_left(0), lower_right(0), upper_left(0), upper_right(0),
      trapezoid_node(0)
{
    assert(left != 0 && "Null left point");
    assert(right != 0 && "Null right point");
    assert(right->is_right_of(*left) && "Incorrect point order");
}

void
TrapezoidMapTriFinder::Trapezoid::assert_valid(bool tree_complete) const
{
#ifndef NDEBUG
    assert(left != 0 && "Null left point");
    assert(right != 0 && "Null right point");

    if (lower_left != 0) {
        assert(lower_left->below == below &&
               lower_left->lower_right == this &&
               "Incorrect lower_left trapezoid");
        assert(get_lower_left_point() == lower_left->get_lower_right_point() &&
               "Incorrect lower left point");
    }

    if (lower_right != 0) {
        assert(lower_right->below == below &&
               lower_right->lower_left == this &&
               "Incorrect lower_right trapezoid");
        assert(get_lower_right_point() == lower_right->get_lower_left_point() &&
               "Incorrect lower right point");
    }

    if (upper_left != 0) {
        assert(upper_left->above == above &&
               upper_left->upper_right == this &&
               "Incorrect upper_left trapezoid");
        assert(get_upper_left_point() == upper_left->get_upper_right_point() &&
               "Incorrect upper left point");
    }

    if (upper_right != 0) {
        assert(upper_right->above == above &&
               upper_right->upper_left == this &&
               "Incorrect upper_right trapezoid");
        assert(get_upper_right_point() == upper_right->get_upper_left_point() &&
               "Incorrect upper right point");
    }

    assert(trapezoid_node != 0 && "Null trapezoid_node");

    if (tree_complete) {
        assert(below.triangle_above == above.triangle_below &&
               "Inconsistent triangle indices from trapezoid edges");
    }
#endif
}

XY
TrapezoidMapTriFinder::Trapezoid::get_lower_left_point() const
{
    double x = left->x;
    return XY(x, below.get_y_at_x(x));
}

XY
TrapezoidMapTriFinder::Trapezoid::get_lower_right_point() const
{
    double x = right->x;
    return XY(x, below.get_y_at_x(x));
}

XY
TrapezoidMapTriFinder::Trapezoid::get_upper_left_point() const
{
    double x = left->x;
    return XY(x, above.get_y_at_x(x));
}

XY
TrapezoidMapTriFinder::Trapezoid::get_upper_right_point() const
{
    double x = right->x;
    return XY(x, above.get_y_at_x(x));
}

void
TrapezoidMapTriFinder::Trapezoid::print_debug() const
{
    std::cout << "Trapezoid " << this
        << " left=" << *left
        << " right=" << *right
        << " below=" << below
        << " above=" << above
        << " ll=" << lower_left
        << " lr=" << lower_right
        << " ul=" << upper_left
        << " ur=" << upper_right
        << " node=" << trapezoid_node
        << " llp=" << get_lower_left_point()
        << " lrp=" << get_lower_right_point()
        << " ulp=" << get_upper_left_point()
        << " urp=" << get_upper_right_point() << std::endl;
}

void
TrapezoidMapTriFinder::Trapezoid::set_lower_left(Trapezoid* lower_left_)
{
    lower_left = lower_left_;
    if (lower_left != 0)
        lower_left->lower_right = this;
}

void
TrapezoidMapTriFinder::Trapezoid::set_lower_right(Trapezoid* lower_right_)
{
    lower_right = lower_right_;
    if (lower_right != 0)
        lower_right->lower_left = this;
}

void
TrapezoidMapTriFinder::Trapezoid::set_upper_left(Trapezoid* upper_left_)
{
    upper_left = upper_left_;
    if (upper_left != 0)
        upper_left->upper_right = this;
}

void
TrapezoidMapTriFinder::Trapezoid::set_upper_right(Trapezoid* upper_right_)
{
    upper_right = upper_right_;
    if (upper_right != 0)
        upper_right->upper_left = this;
}


RandomNumberGenerator::RandomNumberGenerator(unsigned long seed)
    : _M(21870), _A(1291), _C(4621), _seed(seed % _M)
{}

unsigned long
RandomNumberGenerator::operator()(unsigned long max_value)
{
    _seed = (_seed*_A + _C) % _M;
    return (_seed*max_value) / _M;
}





#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__tri(void)
#else
PyMODINIT_FUNC
init_tri(void)
#endif
{
    import_array();

    static TriModule* triModule = NULL;
    triModule = new TriModule();

    #if PY_MAJOR_VERSION >= 3
    return triModule->module().ptr();
    #endif
}

TriModule::TriModule()
    : Py::ExtensionModule<TriModule>("tri")
{
    Triangulation::init_type();
    TriContourGenerator::init_type();
    TrapezoidMapTriFinder::init_type();

    add_varargs_method("Triangulation", &TriModule::new_triangulation,
                       "Create and return new C++ Triangulation object");
    add_varargs_method("TriContourGenerator", &TriModule::new_tricontourgenerator,
                       "Create and return new C++ TriContourGenerator object");
    add_varargs_method("TrapezoidMapTriFinder",
                       &TriModule::new_TrapezoidMapTriFinder,
                       "Create and return new C++ TrapezoidMapTriFinder object");

    initialize("Module for unstructured triangular grids");
}

Py::Object TriModule::new_triangulation(const Py::Tuple &args)
{
    _VERBOSE("TriModule::new_triangulation");
    args.verify_length(6);

    // x and y.
    PyArrayObject* x = (PyArrayObject*)PyArray_ContiguousFromObject(
                           args[0].ptr(), PyArray_DOUBLE, 1, 1);
    PyArrayObject* y = (PyArrayObject*)PyArray_ContiguousFromObject(
                           args[1].ptr(), PyArray_DOUBLE, 1, 1);
    if (x == 0 || y == 0 || PyArray_DIM(x,0) != PyArray_DIM(y,0)) {
        Py_XDECREF(x);
        Py_XDECREF(y);
        throw Py::ValueError("x and y must be 1D arrays of the same length");
    }

    // triangles.
    PyArrayObject* triangles = (PyArrayObject*)PyArray_ContiguousFromObject(
                                   args[2].ptr(), PyArray_INT, 2, 2);
    if (triangles == 0 || PyArray_DIM(triangles,1) != 3) {
        Py_XDECREF(x);
        Py_XDECREF(y);
        Py_XDECREF(triangles);
        throw Py::ValueError("triangles must be a 2D array of shape (?,3)");
    }

    // Optional mask.
    PyArrayObject* mask = 0;
    if (args[3].ptr() != 0 && args[3] != Py::None())
    {
        mask = (PyArrayObject*)PyArray_ContiguousFromObject(
                   args[3].ptr(), PyArray_BOOL, 1, 1);
        if (mask == 0 || PyArray_DIM(mask,0) != PyArray_DIM(triangles,0)) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(triangles);
            Py_XDECREF(mask);
            throw Py::ValueError(
                "mask must be a 1D array with the same length as the triangles array");
        }
    }

    // Optional edges.
    PyArrayObject* edges = 0;
    if (args[4].ptr() != 0 && args[4] != Py::None())
    {
        edges = (PyArrayObject*)PyArray_ContiguousFromObject(
                    args[4].ptr(), PyArray_INT, 2, 2);
        if (edges == 0 || PyArray_DIM(edges,1) != 2) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(triangles);
            Py_XDECREF(mask);
            Py_XDECREF(edges);
            throw Py::ValueError("edges must be a 2D array with shape (?,2)");
        }
    }

    // Optional neighbors.
    PyArrayObject* neighbors = 0;
    if (args[5].ptr() != 0 && args[5] != Py::None())
    {
        neighbors = (PyArrayObject*)PyArray_ContiguousFromObject(
                        args[5].ptr(), PyArray_INT, 2, 2);
        if (neighbors == 0 ||
            PyArray_DIM(neighbors,0) != PyArray_DIM(triangles,0) ||
            PyArray_DIM(neighbors,1) != PyArray_DIM(triangles,1)) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(triangles);
            Py_XDECREF(mask);
            Py_XDECREF(edges);
            Py_XDECREF(neighbors);
            throw Py::ValueError(
                "neighbors must be a 2D array with the same shape as the triangles array");
        }
    }

    return Py::asObject(new Triangulation(x, y, triangles, mask, edges, neighbors));
}

Py::Object TriModule::new_tricontourgenerator(const Py::Tuple &args)
{
    _VERBOSE("TriModule::new_tricontourgenerator");
    args.verify_length(2);

    Py::Object tri = args[0];
    if (!Triangulation::check(tri))
        throw Py::ValueError("Expecting a C++ Triangulation object");

    PyArrayObject* z = (PyArrayObject*)PyArray_ContiguousFromObject(
                           args[1].ptr(), PyArray_DOUBLE, 1, 1);
    if (z == 0 ||
        PyArray_DIM(z,0) != ((Triangulation*)tri.ptr())->get_npoints()) {
        Py_XDECREF(z);
        throw Py::ValueError(
            "z must be a 1D array with the same length as the x and y arrays");
    }

    return Py::asObject(new TriContourGenerator(tri, z));
}

Py::Object
TriModule::new_TrapezoidMapTriFinder(const Py::Tuple &args)
{
    _VERBOSE("TriModule::new_TrapezoidMapTriFinder");
    args.verify_length(1);

    Py::Object triangulation = args[0];
    if (!Triangulation::check(triangulation))
        throw Py::ValueError("Expecting a C++ Triangulation object");

    return Py::asObject(new TrapezoidMapTriFinder(triangulation));
}

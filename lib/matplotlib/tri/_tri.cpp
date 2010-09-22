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

bool XY::operator==(const XY& other) const
{
    return x == other.x && y == other.y;
}

bool XY::operator!=(const XY& other) const
{
    return x != other.x || y != other.y;
}

bool XY::operator<(const XY& other) const
{
    if (y == other.y)
        return x < other.x;
    else
        return y < other.y;
}

XY XY::operator*(const double& multiplier) const
{
    return XY(x*multiplier, y*multiplier);
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

bool BoundingBox::contains_x(const double& x) const
{
    return !empty && x >= lower.x && x <= upper.x;
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& box)
{
    if (box.empty)
        return os << "<empty>";
    else
        return os << box.lower << " -> " << box.upper;
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
    npy_intp dims[2] = {edge_set.size(), 2};
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
        npy_intp dims[2] = {line.size(),2};
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





#if defined(_MSC_VER)
DL_EXPORT(void)
#elif defined(__cplusplus)
extern "C" void
#else
void
#endif
init_tri()
{
    static TriModule* triModule = NULL;
    triModule = new TriModule();
    import_array();
}

TriModule::TriModule()
    : Py::ExtensionModule<TriModule>("tri")
{
    Triangulation::init_type();
    TriContourGenerator::init_type();

    add_varargs_method("Triangulation", &TriModule::new_triangulation,
                       "Create and return new C++ Triangulation object");
    add_varargs_method("TriContourGenerator", &TriModule::new_tricontourgenerator,
                       "Create and return new C++ TriContourGenerator object");

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


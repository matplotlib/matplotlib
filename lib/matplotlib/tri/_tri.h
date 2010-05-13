/*
 * Unstructured triangular grid functions, particularly contouring.
 *
 * There are two main classes: Triangulation and TriContourGenerator.
 *
 * Triangulation
 * -------------
 * Triangulation is an unstructured triangular grid with npoints and ntri
 * triangles.  It consists of point x and y coordinates, and information about
 * the triangulation stored in an integer array of shape (ntri,3) called
 * triangles.  Each triangle is represented by three point indices (in the
 * range 0 to npoints-1) that comprise the triangle, ordered anticlockwise.
 * There is an optional mask of length ntri which can be used to mask out
 * triangles and has the same result as removing those triangles from the
 * 'triangles' array.
 *
 * A particular edge of a triangulation is termed a TriEdge, which is a
 * triangle index and an edge index in the range 0 to 2.  TriEdge(tri,edge)
 * refers to the edge that starts at point index triangles(tri,edge) and ends
 * at point index triangles(tri,(edge+1)%3).
 *
 * Various derived fields are calculated when they are first needed.  The
 * triangle connectivity is stored in a neighbors array of shape (ntri,3) such
 * that neighbors(tri,edge) is the index of the triangle that adjoins the
 * TriEdge(tri,edge), or -1 if there is no such neighbor.
 *
 * A triangulation has one or more boundaries, each of which is a 1D array of
 * the TriEdges that comprise the boundary, in order following the boundary
 * with non-masked triangles on the left.
 *
 * TriContourGenerator
 * -------------------
 * A TriContourGenerator generates contours for a particular Triangulation.
 * The process followed is different for non-filled and filled contours, with
 * one and two contour levels respectively.  In both cases boundary contour
 * lines are found first, then interior lines.
 *
 * Boundary lines start and end on a boundary.  They are found by traversing
 * the triangulation boundary edges until a suitable start point is found, and
 * then the contour line is followed across the interior of the triangulation
 * until it ends on another boundary edge.  For a non-filled contour this
 * completes a line, whereas a filled contour continues by following the
 * boundary around until either another boundary start point is found or the
 * start of the contour line is reached.  Filled contour generation stores
 * boolean flags to indicate which boundary edges have already been traversed
 * so that they are not dealt with twice.  Similar flags are used to indicate
 * which triangles have been used when following interior lines.
 *
 * Interior lines do not intersect any boundaries.  They are found by
 * traversing all triangles that have not yet been visited until a suitable
 * starting point is found, and then the contour line is followed across the
 * interior of the triangulation until it returns to the start point.  For
 * filled contours this process is repeated for both lower and upper contour
 * levels, and the direction of traversal is reversed for upper contours.
 *
 * Working out in which direction a contour line leaves a triangle uses the
 * a lookup table.  A triangle has three points, each of which has a z-value
 * which is either less than the contour level or not.  Hence there are 8
 * configurations to deal with, 2 of which do not have a contour line (all
 * points below or above (including the same as) the contour level) and 6 that
 * do.  See the function get_exit_edge for details.
 */
#ifndef _TRI_H
#define _TRI_H

#include "CXX/Extensions.hxx"
#include "CXX/Objects.hxx"
#include "numpy/arrayobject.h"

#include <map>
#include <vector>



/* An edge of a triangle consisting of an triangle index in the range 0 to
 * ntri-1 and an edge index in the range 0 to 2.  Edge i goes from the
 * triangle's point i to point (i+1)%3. */
struct TriEdge
{
    TriEdge();
    TriEdge(int tri_, int edge_);
    bool operator<(const TriEdge& other) const;
    bool operator==(const TriEdge& other) const;
    bool operator!=(const TriEdge& other) const;
    friend std::ostream& operator<<(std::ostream& os, const TriEdge& tri_edge);

    int tri, edge;
};

// 2D point with x,y coordinates.
struct XY
{
    XY();
    XY(const double& x_, const double& y_);
    double angle() const;           // Angle in radians with respect to x-axis.
    double cross_z(const XY& other) const;     // z-component of cross product.
    bool operator==(const XY& other) const;
    bool operator!=(const XY& other) const;
    bool operator<(const XY& other) const;
    XY operator*(const double& multiplier) const;
    XY operator+(const XY& other) const;
    XY operator-(const XY& other) const;
    friend std::ostream& operator<<(std::ostream& os, const XY& xy);

    double x, y;
};

// 2D bounding box, which may be empty.
class BoundingBox
{
public:
    BoundingBox();
    void add(const XY& point);
    bool contains_x(const double& x) const;
    friend std::ostream& operator<<(std::ostream& os, const BoundingBox& box);

    // Consider these member variables read-only.
    bool empty;
    XY lower, upper;
};

/* A single line of a contour, which may be a closed line loop or an open line
 * strip.  Identical adjacent points are avoided using insert_unique() and
 * push_back(), and a closed line loop should also not have identical first and
 * last points. */
class ContourLine : public std::vector<XY>
{
public:
    ContourLine();
    void insert_unique(iterator pos, const XY& point);
    void push_back(const XY& point);
    void write() const;
};

// A Contour is a collection of zero or more ContourLines.
typedef std::vector<ContourLine> Contour;

// Debug contour writing function.
void write_contour(const Contour& contour);




/* Triangulation with npoints points and ntri triangles.  Derived fields are
 * calculated when they are first needed. */
class Triangulation : public Py::PythonExtension<Triangulation>
{
public:
    /* A single boundary is a vector of the TriEdges that make up that boundary
     * following it around with unmasked triangles on the left. */
    typedef std::vector<TriEdge> Boundary;
    typedef std::vector<Boundary> Boundaries;

    /* Constructor with optional mask, edges and neighbors.  The latter two
     * are calculated when first needed.
     *   x: double array of shape (npoints) of points' x-coordinates.
     *   y: double array of shape (npoints) of points' y-coordinates.
     *   triangles: int array of shape (ntri,3) of triangle point indices.
     *              Those ordered clockwise are changed to be anticlockwise.
     *   mask: Optional bool array of shape (ntri) indicating which triangles
     *         are masked.
     *   edges: Optional int array of shape (?,2) of start and end point
     *          indices, each edge (start,end and end,start) appearing only
     *          once.
     *   neighbors: Optional int array of shape (ntri,3) indicating which
     *              triangles are the neighbors of which TriEdges, or -1 if
     *              there is no such neighbor. */
    Triangulation(PyArrayObject* x,
                  PyArrayObject* y,
                  PyArrayObject* triangles,
                  PyArrayObject* mask,
                  PyArrayObject* edges,
                  PyArrayObject* neighbors);

    virtual ~Triangulation();

    // Return the boundaries collection, creating it if necessary.
    const Boundaries& get_boundaries() const;

    // Return which boundary and boundary edge the specified TriEdge is.
    void get_boundary_edge(const TriEdge& triEdge,
                           int& boundary,
                           int& edge) const;

    // Return the edges array, creating it if necessary.
    Py::Object get_edges();

    /* Return the triangle index of the neighbor of the specified triangle
     * edge. */
    int get_neighbor(int tri, int edge) const;

    /* Return the TriEdge that is the neighbor of the specified triangle edge,
     * or TriEdge(-1,-1) if there is no such neighbor. */
    TriEdge get_neighbor_edge(int tri, int edge) const;

    // Return the neighbors array, creating it if necessary.
    Py::Object get_neighbors();

    // Return the number of points in this triangulation.
    int get_npoints() const;

    // Return the number of triangles in this triangulation.
    int get_ntri() const;

    /* Return the index of the point that is at the start of the specified
     * triangle edge. */
    int get_triangle_point(int tri, int edge) const;
    int get_triangle_point(const TriEdge& tri_edge) const;

    // Return the coordinates of the specified point index.
    XY get_point_coords(int point) const;

    // CXX initialisation function.
    static void init_type();

    // Indicates if the specified triangle is masked or not.
    bool is_masked(int tri) const;

    /* Set or clear the mask array.  Clears various derived fields so they are
     * recalculated when next needed.
     *   args[0]: mask, either Py::None or boolean array of shape (ntri). */
    Py::Object set_mask(const Py::Tuple &args);

    // Debug function to write boundaries.
    void write_boundaries() const;

private:
    // An edge of a triangulation, composed of start and end point indices.
    struct Edge
    {
        Edge() : start(-1), end(-1) {}
        Edge(int start_, int end_) : start(start_), end(end_) {}
        bool operator<(const Edge& other) const {
            return start != other.start ? start < other.start : end < other.end;
        }
        int start, end;
    };

    /* An edge of a boundary of a triangulation, composed of a boundary index
     * and an edge index within that boundary.  Used to index into the
     * boundaries collection to obtain the corresponding TriEdge. */
    struct BoundaryEdge
    {
        BoundaryEdge() : boundary(-1), edge(-1) {}
        BoundaryEdge(int boundary_, int edge_)
            : boundary(boundary_), edge(edge_) {}
        int boundary, edge;
    };

    /* Calculate the boundaries collection.  Should normally be accessed via
     * get_boundaries(), which will call this function if necessary. */
    void calculate_boundaries();

    /* Calculate the edges array.  Should normally be accessed via
     * get_edges(), which will call this function if necessary. */
    void calculate_edges();

    /* Calculate the neighbors array. Should normally be accessed via
     * get_neighbors(), which will call this function if necessary. */
    void calculate_neighbors();

    /* Correct each triangle so that the vertices are ordered in an
     * anticlockwise manner. */
    void correct_triangles();

    /* Determine which edge index (0,1 or 2) the specified point index is in
     * the specified triangle, or -1 if the point is not in the triangle. */
    int get_edge_in_triangle(int tri, int point) const;

    // Return pointer to contents of neighbors array.
    const int* get_neighbors_ptr() const;

    // Return pointer to contents of triangles array.
    const int* get_triangles_ptr() const;



    int _npoints, _ntri;

    // Variables shared with python, always set.
    PyArrayObject* _x;         // double array (npoints).
    PyArrayObject* _y;         // double array (npoints).
    PyArrayObject* _triangles; // int array (ntri,3) of triangle point indices,
                               //     ordered anticlockwise.

    // Variables shared with python, may be zero.
    PyArrayObject* _mask;      // bool array (ntri).

    // Derived variables shared with python, may be zero.  If zero, are
    // recalculated when needed.
    PyArrayObject* _edges;     // int array (?,2) of start & end point indices.
    PyArrayObject* _neighbors; // int array (ntri,3), neighbor triangle indices
                               //     or -1 if no neighbor.

    // Variables internal to C++ only.
    Boundaries _boundaries;

    // Map used to look up BoundaryEdges from TriEdges.  Normally accessed via
    // get_boundary_edge().
    typedef std::map<TriEdge, BoundaryEdge> TriEdgeToBoundaryMap;
    TriEdgeToBoundaryMap _tri_edge_to_boundary_map;
};




// Contour generator for a triangulation.
class TriContourGenerator : public Py::PythonExtension<TriContourGenerator>
{
public:
    /* Constructor.
     *   triangulation: Triangulation to generate contours for.
     *   z: Double array of shape (npoints) of z-values at triangulation
     *      points. */
    TriContourGenerator(Py::Object triangulation,
                        PyArrayObject* z);

    virtual ~TriContourGenerator();

    /* Create and return a non-filled contour.
     *   level: Contour level.
     * Returns python list [segs0, segs1, ...] where
     *   segs0: double array of shape (?,2) of point coordinates of first
     *   contour line, etc. */
    Py::Object create_contour(const Py::Tuple &args);

    /* Create and return a filled contour.
     *   lower_level: Lower contour level.
     *   upper_level: Upper contour level.
     * Returns python tuple (segs, kinds) where
     *   segs: double array of shape (n_points,2) of all point coordinates,
     *   kinds: ubyte array of shape (n_points) of all point code types. */
    Py::Object create_filled_contour(const Py::Tuple &args);

    // CXX initialisation function.
    static void init_type();

private:
    typedef Triangulation::Boundary Boundary;
    typedef Triangulation::Boundaries Boundaries;

    /* Clear visited flags.
     *   include_boundaries: Whether to clear boundary flags or not, which are
     *                       only used for filled contours. */
    void clear_visited_flags(bool include_boundaries);

    /* Convert a non-filled Contour from C++ to Python.
     * Returns python list [segs0, segs1, ...] where
     *   segs0: double array of shape (?,2) of point coordinates of first
     *   contour line, etc. */
    Py::Object contour_to_segs(const Contour& contour);

    /* Convert a filled Contour from C++ to Python.
     * Returns python tuple (segs, kinds) where
     *   segs: double array of shape (n_points,2) of all point coordinates,
     *   kinds: ubyte array of shape (n_points) of all point code types. */
    Py::Object contour_to_segs_and_kinds(const Contour& contour);

    /* Return the point on the specified TriEdge that intersects the specified
     * level. */
    XY edge_interp(int tri, int edge, const double& level);

    /* Find and follow non-filled contour lines that start and end on a
     * boundary of the Triangulation.
     *   contour: Contour to add new lines to.
     *   level: Contour level. */
    void find_boundary_lines(Contour& contour,
                             const double& level);

    /* Find and follow filled contour lines at either of the specified contour
     * levels that start and end of a boundary of the Triangulation.
     *   contour: Contour to add new lines to.
     *   lower_level: Lower contour level.
     *   upper_level: Upper contour level. */
    void find_boundary_lines_filled(Contour& contour,
                                    const double& lower_level,
                                    const double& upper_level);

    /* Find and follow lines at the specified contour level that are
     * completely in the interior of the Triangulation and hence do not
     * intersect any boundary.
     *   contour: Contour to add new lines to.
     *   level: Contour level.
     *   on_upper: Whether on upper or lower contour level.
     *   filled: Whether contours are filled or not. */
    void find_interior_lines(Contour& contour,
                             const double& level,
                             bool on_upper,
                             bool filled);

    /* Follow contour line around boundary of the Triangulation from the
     * specified TriEdge to its end which can be on either the lower or upper
     * levels.  Only used for filled contours.
     *   contour_line: Contour line to append new points to.
     *   tri_edge: On entry, TriEdge to start from.  On exit, TriEdge that is
     *             finished on.
     *   lower_level: Lower contour level.
     *   upper_level: Upper contour level.
     *   on_upper: Whether starts on upper level or not.
     * Return true if finishes on upper level, false if lower. */
    bool follow_boundary(ContourLine& contour_line,
                         TriEdge& tri_edge,
                         const double& lower_level,
                         const double& upper_level,
                         bool on_upper);

    /* Follow contour line across interior of Triangulation.
     *   contour_line: Contour line to append new points to.
     *   tri_edge: On entry, TriEdge to start from.  On exit, TriEdge that is
     *             finished on.
     *   end_on_boundary: Whether this line ends on a boundary, or loops back
     *                    upon itself.
     *   level: Contour level to follow.
     *   on_upper: Whether following upper or lower contour level. */
    void follow_interior(ContourLine& contour_line,
                         TriEdge& tri_edge,
                         bool end_on_boundary,
                         const double& level,
                         bool on_upper);

    // Return the Triangulation boundaries.
    const Boundaries& get_boundaries() const;

    /* Return the edge by which the a level leaves a particular triangle,
     * which is 0, 1 or 2 if the contour passes through the triangle or -1
     * otherwise.
     *   tri: Triangle index.
     *   level: Contour level to follow.
     *   on_upper: Whether following upper or lower contour level. */
    int get_exit_edge(int tri, const double& level, bool on_upper) const;

    // Return the Triangulation object.
    const Triangulation& get_triangulation() const;

    // Return the z-value at the specified point index.
    const double& get_z(int point) const;

    /* Return the point at which the a level intersects the line connecting the
     * two specified point indices. */
    XY interp(int point1, int point2, const double& level) const;



    // Variables shared with python, always set.
    Py::Object _triangulation;
    PyArrayObject* _z;        // double array (npoints).

    // Variables internal to C++ only.
    typedef std::vector<bool> InteriorVisited;    // Size 2*ntri
    typedef std::vector<bool> BoundaryVisited;
    typedef std::vector<BoundaryVisited> BoundariesVisited;
    typedef std::vector<bool> BoundariesUsed;

    InteriorVisited _interior_visited;
    BoundariesVisited _boundaries_visited;  // Only used for filled contours.
    BoundariesUsed _boundaries_used;        // Only used for filled contours.
};




// The extension module.
class TriModule : public Py::ExtensionModule<TriModule>
{
public:
    TriModule();

private:
    Py::Object new_triangulation(const Py::Tuple &args);
    Py::Object new_tricontourgenerator(const Py::Tuple &args);
};

#endif

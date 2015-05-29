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

#include "src/numpy_cpp.h"

#include <iostream>
#include <list>
#include <map>
#include <set>
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
    bool is_right_of(const XY& other) const;   // Compares x then y.
    bool operator==(const XY& other) const;
    bool operator!=(const XY& other) const;
    XY operator*(const double& multiplier) const;
    const XY& operator+=(const XY& other);
    const XY& operator-=(const XY& other);
    XY operator+(const XY& other) const;
    XY operator-(const XY& other) const;
    friend std::ostream& operator<<(std::ostream& os, const XY& xy);

    double x, y;
};

// 3D point with x,y,z coordinates.
struct XYZ
{
    XYZ(const double& x_, const double& y_, const double& z_);
    XYZ cross(const XYZ& other) const;
    double dot(const XYZ& other) const;
    double length_squared() const;
    XYZ operator-(const XYZ& other) const;
    friend std::ostream& operator<<(std::ostream& os, const XYZ& xyz);

    double x, y, z;
};

// 2D bounding box, which may be empty.
class BoundingBox
{
public:
    BoundingBox();
    void add(const XY& point);
    void expand(const XY& delta);

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
class Triangulation
{
public:
    typedef numpy::array_view<const double, 1> CoordinateArray;
    typedef numpy::array_view<double, 2> TwoCoordinateArray;
    typedef numpy::array_view<int, 2> TriangleArray;
    typedef numpy::array_view<const bool, 1> MaskArray;
    typedef numpy::array_view<int, 2> EdgeArray;
    typedef numpy::array_view<int, 2> NeighborArray;

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
     *              there is no such neighbor.
     *   correct_triangle_orientations: Whether or not should correct triangle
     *                                  orientations so that vertices are
     *                                  ordered anticlockwise. */
    Triangulation(const CoordinateArray& x,
                  const CoordinateArray& y,
                  const TriangleArray& triangles,
                  const MaskArray& mask,
                  const EdgeArray& edges,
                  const NeighborArray& neighbors,
                  int correct_triangle_orientations);

    /* Calculate plane equation coefficients for all unmasked triangles from
     * the point (x,y) coordinates and point z-array of shape (npoints) passed
     * in via the args.  Returned array has shape (npoints,3) and allows
     * z-value at (x,y) coordinates in triangle tri to be calculated using
     *      z = array[tri,0]*x + array[tri,1]*y + array[tri,2]. */
    TwoCoordinateArray calculate_plane_coefficients(const CoordinateArray& z);

    // Return the boundaries collection, creating it if necessary.
    const Boundaries& get_boundaries() const;

    // Return which boundary and boundary edge the specified TriEdge is.
    void get_boundary_edge(const TriEdge& triEdge,
                           int& boundary,
                           int& edge) const;

    /* Return the edges array, creating it if necessary. */
    EdgeArray& get_edges();

    /* Return the triangle index of the neighbor of the specified triangle
     * edge. */
    int get_neighbor(int tri, int edge) const;

    /* Return the TriEdge that is the neighbor of the specified triangle edge,
     * or TriEdge(-1,-1) if there is no such neighbor. */
    TriEdge get_neighbor_edge(int tri, int edge) const;

    /* Return the neighbors array, creating it if necessary. */
    NeighborArray& get_neighbors();

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

    // Indicates if the specified triangle is masked or not.
    bool is_masked(int tri) const;

    /* Set or clear the mask array.  Clears various derived fields so they are
     * recalculated when next needed.
     *   mask: bool array of shape (ntri) indicating which triangles are
     *         masked, or an empty array to clear mask. */
    void set_mask(const MaskArray& mask);

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




    // Variables shared with python, always set.
    CoordinateArray _x, _y;    // double array (npoints).
    TriangleArray _triangles;  // int array (ntri,3) of triangle point indices,
                               //     ordered anticlockwise.

    // Variables shared with python, may be zero.
    MaskArray _mask;           // bool array (ntri).

    // Derived variables shared with python, may be zero.  If zero, are
    // recalculated when needed.
    EdgeArray _edges;          // int array (?,2) of start & end point indices.
    NeighborArray _neighbors;  // int array (ntri,3), neighbor triangle indices
                               //     or -1 if no neighbor.

    // Variables internal to C++ only.
    Boundaries _boundaries;

    // Map used to look up BoundaryEdges from TriEdges.  Normally accessed via
    // get_boundary_edge().
    typedef std::map<TriEdge, BoundaryEdge> TriEdgeToBoundaryMap;
    TriEdgeToBoundaryMap _tri_edge_to_boundary_map;
};



// Contour generator for a triangulation.
class TriContourGenerator
{
public:
    typedef Triangulation::CoordinateArray CoordinateArray;

    /* Constructor.
     *   triangulation: Triangulation to generate contours for.
     *   z: Double array of shape (npoints) of z-values at triangulation
     *      points. */
    TriContourGenerator(Triangulation& triangulation,
                        const CoordinateArray& z);

    /* Create and return a non-filled contour.
     *   level: Contour level.
     * Returns new python list [segs0, segs1, ...] where
     *   segs0: double array of shape (?,2) of point coordinates of first
     *   contour line, etc. */
    PyObject* create_contour(const double& level);

    /* Create and return a filled contour.
     *   lower_level: Lower contour level.
     *   upper_level: Upper contour level.
     * Returns new python tuple (segs, kinds) where
     *   segs: double array of shape (n_points,2) of all point coordinates,
     *   kinds: ubyte array of shape (n_points) of all point code types. */
    PyObject* create_filled_contour(const double& lower_level,
                                    const double& upper_level);

private:
    typedef Triangulation::Boundary Boundary;
    typedef Triangulation::Boundaries Boundaries;

    /* Clear visited flags.
     *   include_boundaries: Whether to clear boundary flags or not, which are
     *                       only used for filled contours. */
    void clear_visited_flags(bool include_boundaries);

    /* Convert a non-filled Contour from C++ to Python.
     * Returns new python list [segs0, segs1, ...] where
     *   segs0: double array of shape (?,2) of point coordinates of first
     *   contour line, etc. */
    PyObject* contour_to_segs(const Contour& contour);

    /* Convert a filled Contour from C++ to Python.
     * Returns new python tuple (segs, kinds) where
     *   segs: double array of shape (n_points,2) of all point coordinates,
     *   kinds: ubyte array of shape (n_points) of all point code types. */
    PyObject* contour_to_segs_and_kinds(const Contour& contour);

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

    // Return the z-value at the specified point index.
    const double& get_z(int point) const;

    /* Return the point at which the a level intersects the line connecting the
     * two specified point indices. */
    XY interp(int point1, int point2, const double& level) const;



    // Variables shared with python, always set.
    Triangulation& _triangulation;
    CoordinateArray _z;        // double array (npoints).

    // Variables internal to C++ only.
    typedef std::vector<bool> InteriorVisited;    // Size 2*ntri
    typedef std::vector<bool> BoundaryVisited;
    typedef std::vector<BoundaryVisited> BoundariesVisited;
    typedef std::vector<bool> BoundariesUsed;

    InteriorVisited _interior_visited;
    BoundariesVisited _boundaries_visited;  // Only used for filled contours.
    BoundariesUsed _boundaries_used;        // Only used for filled contours.
};



/* TriFinder class implemented using the trapezoid map algorithm from the book
 * "Computational Geometry, Algorithms and Applications", second edition, by
 * M. de Berg, M. van Kreveld, M. Overmars and O. Schwarzkopf.
 *
 * The domain of interest is composed of vertical-sided trapezoids that are
 * bounded to the left and right by points of the triangulation, and below and
 * above by edges of the triangulation.  Each triangle is represented by 1 or
 * more of these trapezoids.  Edges are inserted one a time in a random order.
 *
 * As the trapezoid map is created, a search tree is also created which allows
 * fast lookup O(log N) of the trapezoid containing the point of interest.
 * There are 3 types of node in the search tree: all leaf nodes represent
 * trapezoids and all branch nodes have 2 child nodes and are either x-nodes or
 * y-nodes.  X-nodes represent points in the triangulation, and their 2 children
 * refer to those parts of the search tree to the left and right of the point.
 * Y-nodes represent edges in the triangulation, and their 2 children refer to
 * those parts of the search tree below and above the edge.
 *
 * Nodes can be repeated throughout the search tree, and each is reference
 * counted through the multiple parent nodes it is a child of.
 *
 * The algorithm is only intended to work with valid triangulations, i.e. it
 * must not contain duplicate points, triangles formed from colinear points, or
 * overlapping triangles.  It does have some tolerance to triangles formed from
 * colinear points but only in the simplest of cases.  No explicit testing of
 * the validity of the triangulation is performed as this is a computationally
 * more complex task than the trifinding itself. */
class TrapezoidMapTriFinder
{
public:
    typedef Triangulation::CoordinateArray CoordinateArray;
    typedef numpy::array_view<int, 1> TriIndexArray;

    /* Constructor.  A separate call to initialize() is required to initialize
     * the object before use.
     *   triangulation: Triangulation to find triangles in. */
    TrapezoidMapTriFinder(Triangulation& triangulation);

    ~TrapezoidMapTriFinder();

    /* Return an array of triangle indices.  Takes 1D arrays x and y of
     * point coordinates, and returns an array of the same size containing the
     * indices of the triangles at those points. */
    TriIndexArray find_many(const CoordinateArray& x, const CoordinateArray& y);

    /* Return a reference to a new python list containing the following
     * statistics about the tree:
     *   0: number of nodes (tree size)
     *   1: number of unique nodes (number of unique Node objects in tree)
     *   2: number of trapezoids (tree leaf nodes)
     *   3: number of unique trapezoids
     *   4: maximum parent count (max number of times a node is repeated in
     *          tree)
     *   5: maximum depth of tree (one more than the maximum number of
     *          comparisons needed to search through the tree)
     *   6: mean of all trapezoid depths (one more than the average number of
     *          comparisons needed to search through the tree) */
    PyObject* get_tree_stats();

    /* Initialize this object before use.  May be called multiple times, if,
     * for example, the triangulation is changed by setting the mask. */
    void initialize();

    // Print the search tree as text to stdout; useful for debug purposes.
    void print_tree();

private:
    /* A Point consists of x,y coordinates as well as the index of a triangle
     * associated with the point, so that a search at this point's coordinates
     * can return a valid triangle index. */
    struct Point : XY
    {
        Point() : XY(), tri(-1) {}
        Point(const double& x, const double& y) : XY(x,y), tri(-1) {}
        explicit Point(const XY& xy) : XY(xy), tri(-1) {}

        int tri;
    };

    /* An Edge connects two Points, left and right.  It is always true that
     * right->is_right_of(*left).  Stores indices of triangles below and above
     * the Edge which are used to map from trapezoid to triangle index.  Also
     * stores pointers to the 3rd points of the below and above triangles,
     * which are only used to disambiguate triangles with colinear points. */
    struct Edge
    {
        Edge(const Point* left_,
             const Point* right_,
             int triangle_below_,
             int triangle_above_,
             const Point* point_below_,
             const Point* point_above_);

        // Return -1 if point to left of edge, 0 if on edge, +1 if to right.
        int get_point_orientation(const XY& xy) const;

        // Return slope of edge, even if vertical (divide by zero is OK here).
        double get_slope() const;

        /* Return y-coordinate of point on edge with specified x-coordinate.
         * x must be within the x-limits of this edge. */
        double get_y_at_x(const double& x) const;

        // Return true if the specified point is either of the edge end points.
        bool has_point(const Point* point) const;

        bool operator==(const Edge& other) const;

        friend std::ostream& operator<<(std::ostream& os, const Edge& edge)
        {
            return os << *edge.left << "->" << *edge.right;
        }

        void print_debug() const;


        const Point* left;        // Not owned.
        const Point* right;       // Not owned.
        int triangle_below;       // Index of triangle below (to right of) Edge.
        int triangle_above;       // Index of triangle above (to left of) Edge.
        const Point* point_below; // Used only for resolving ambiguous cases;
        const Point* point_above; //     is 0 if corresponding triangle is -1
    };

    class Node;  // Forward declaration.

    // Helper structure used by TrapezoidMapTriFinder::get_tree_stats.
    struct NodeStats
    {
        NodeStats()
            : node_count(0), trapezoid_count(0), max_parent_count(0),
              max_depth(0), sum_trapezoid_depth(0.0)
        {}

        long node_count, trapezoid_count, max_parent_count, max_depth;
        double sum_trapezoid_depth;
        std::set<const Node*> unique_nodes, unique_trapezoid_nodes;
    };

    struct Trapezoid;  // Forward declaration.

    /* Node of the trapezoid map search tree.  There are 3 possible types:
     * Type_XNode, Type_YNode and Type_TrapezoidNode.  Data members are
     * represented using a union: an XNode has a Point and 2 child nodes
     * (left and right of the point), a YNode has an Edge and 2 child nodes
     * (below and above the edge), and a TrapezoidNode has a Trapezoid.
     * Each Node has multiple parents so it can appear in the search tree
     * multiple times without having to create duplicate identical Nodes.
     * The parent collection acts as a reference count to the number of times
     * a Node occurs in the search tree.  When the parent count is reduced to
     * zero a Node can be safely deleted. */
    class Node
    {
    public:
        Node(const Point* point, Node* left, Node* right);// Type_XNode.
        Node(const Edge* edge, Node* below, Node* above); // Type_YNode.
        Node(Trapezoid* trapezoid);                       // Type_TrapezoidNode.

        ~Node();

        void add_parent(Node* parent);

        /* Recurse through the search tree and assert that everything is valid.
         * Reduces to a no-op if NDEBUG is defined. */
        void assert_valid(bool tree_complete) const;

        // Recurse through the tree to return statistics about it.
        void get_stats(int depth, NodeStats& stats) const;

        // Return the index of the triangle corresponding to this node.
        int get_tri() const;

        bool has_child(const Node* child) const;
        bool has_no_parents() const;
        bool has_parent(const Node* parent) const;

        /* Recurse through the tree and print a textual representation to
         * stdout.  Argument depth used to indent for readability. */
        void print(int depth = 0) const;

        /* Remove a parent from this Node.  Return true if no parents remain
         * so that this Node can be deleted. */
        bool remove_parent(Node* parent);

        void replace_child(Node* old_child, Node* new_child);

        // Replace this node with the specified new_node in all parents.
        void replace_with(Node* new_node);

        /* Recursive search through the tree to find the Node containing the
         * specified XY point. */
        const Node* search(const XY& xy);

        /* Recursive search through the tree to find the Trapezoid containing
         * the left endpoint of the specified Edge.  Return 0 if fails, which
         * can only happen if the triangulation is invalid. */
        Trapezoid* search(const Edge& edge);

        /* Copy constructor and assignment operator defined but not implemented
         * to prevent objects being copied. */
        Node(const Node& other);
        Node& operator=(const Node& other);

    private:
        typedef enum {
            Type_XNode,
            Type_YNode,
            Type_TrapezoidNode
        } Type;
        Type _type;

        union {
            struct {
                const Point* point;  // Not owned.
                Node* left;          // Owned.
                Node* right;         // Owned.
            } xnode;
            struct {
                const Edge* edge;    // Not owned.
                Node* below;         // Owned.
                Node* above;         // Owned.
            } ynode;
            Trapezoid* trapezoid;    // Owned.
        } _union;

        typedef std::list<Node*> Parents;
        Parents _parents;            // Not owned.
    };

    /* A Trapezoid is bounded by Points to left and right, and Edges below and
     * above.  Has up to 4 neighboring Trapezoids to lower/upper left/right.
     * Lower left neighbor is Trapezoid to left that shares the below Edge, or
     * is 0 if there is no such Trapezoid (and similar for other neighbors).
     * To obtain the index of the triangle corresponding to a particular
     * Trapezoid, use the Edge member variables below.triangle_above or
     * above.triangle_below. */
    struct Trapezoid
    {
        Trapezoid(const Point* left_,
                  const Point* right_,
                  const Edge& below_,
                  const Edge& above_);

        /* Assert that this Trapezoid is valid.  Reduces to a no-op if NDEBUG
         * is defined. */
        void assert_valid(bool tree_complete) const;

        /* Return one of the 4 corner points of this Trapezoid.  Only used for
         * debugging purposes. */
        XY get_lower_left_point() const;
        XY get_lower_right_point() const;
        XY get_upper_left_point() const;
        XY get_upper_right_point() const;

        void print_debug() const;

        /* Set one of the 4 neighbor trapezoids and the corresponding reverse
         * Trapezoid of the new neighbor (if it is not 0), so that they are
         * consistent. */
        void set_lower_left(Trapezoid* lower_left_);
        void set_lower_right(Trapezoid* lower_right_);
        void set_upper_left(Trapezoid* upper_left_);
        void set_upper_right(Trapezoid* upper_right_);

        /* Copy constructor and assignment operator defined but not implemented
         * to prevent objects being copied. */
        Trapezoid(const Trapezoid& other);
        Trapezoid& operator=(const Trapezoid& other);


        const Point* left;     // Not owned.
        const Point* right;    // Not owned.
        const Edge& below;
        const Edge& above;

        // 4 neighboring trapezoids, can be 0, not owned.
        Trapezoid* lower_left;   // Trapezoid to left  that shares below
        Trapezoid* lower_right;  // Trapezoid to right that shares below
        Trapezoid* upper_left;   // Trapezoid to left  that shares above
        Trapezoid* upper_right;  // Trapezoid to right that shares above

        Node* trapezoid_node;    // Node that owns this Trapezoid.
    };


    // Add the specified Edge to the search tree, returning true if successful.
    bool add_edge_to_tree(const Edge& edge);

    // Clear all memory allocated by this object.
    void clear();

    // Return the triangle index at the specified point, or -1 if no triangle.
    int find_one(const XY& xy);

    /* Determine the trapezoids that the specified Edge intersects, returning
     * true if successful. */
    bool find_trapezoids_intersecting_edge(const Edge& edge,
                                           std::vector<Trapezoid*>& trapezoids);



    // Variables shared with python, always set.
    Triangulation& _triangulation;

    // Variables internal to C++ only.
    Point* _points;    // Array of all points in triangulation plus corners of
                       // enclosing rectangle.  Owned.

    typedef std::vector<Edge> Edges;
    Edges _edges;   // All Edges in triangulation plus bottom and top Edges of
                    // enclosing rectangle.

    Node* _tree;    // Root node of the trapezoid map search tree.  Owned.
};



/* Linear congruential random number generator.  Edges in the triangulation are
 * randomly shuffled before being added to the trapezoid map.  Want the
 * shuffling to be identical across different operating systems and the same
 * regardless of previous random number use.  Would prefer to use a STL or
 * Boost random number generator, but support is not consistent across
 * different operating systems so implementing own here.
 *
 * This is not particularly random, but is perfectly adequate for the use here.
 * Coefficients taken from Numerical Recipes in C. */
class RandomNumberGenerator
{
public:
    RandomNumberGenerator(unsigned long seed);

    // Return random integer in the range 0 to max_value-1.
    unsigned long operator()(unsigned long max_value);

private:
    const unsigned long _m, _a, _c;
    unsigned long _seed;
};

#endif

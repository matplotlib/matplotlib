/*
 * QuadContourGenerator
 * --------------------
 * A QuadContourGenerator generates contours for scalar fields defined on
 * quadrilateral grids.  A single QuadContourGenerator object can create both
 * line contours (at single levels) and filled contours (between pairs of
 * levels) for the same field.
 *
 * A field to be contoured has nx, ny points in the x- and y-directions
 * respectively.  The quad grid is defined by x and y arrays of shape(ny, nx),
 * and the field itself is the z array also of shape(ny, nx).  There is an
 * optional boolean mask; if it exists then it also has shape(ny, nx).  The
 * mask applies to grid points rather than quads.
 *
 * How quads are masked based on the point mask is determined by the boolean
 * 'corner_mask' flag.  If false then any quad that has one or more of its four
 * corner points masked is itself masked.  If true the behaviour is the same
 * except that any quad which has exactly one of its four corner points masked
 * has only the triangular corner (half of the quad) adjacent to that point
 * masked; the opposite triangular corner has three unmasked points and is not
 * masked.
 *
 * By default the entire domain of nx*ny points is contoured together which can
 * result in some very long polygons.  The alternative is to break up the
 * domain into subdomains or 'chunks' of smaller size, each of which is
 * independently contoured.  The size of these chunks is controlled by the
 * 'nchunk' (or 'chunk_size') parameter.  Chunking not only results in shorter
 * polygons but also requires slightly less RAM.  It can result in rendering
 * artifacts though, depending on backend, antialiased flag and alpha value.
 *
 * Notation
 * --------
 * i and j are array indices in the x- and y-directions respectively.  Although
 * a single element of an array z can be accessed using z[j][i] or z(j,i), it
 * is often convenient to use the single quad index z[quad], where
 *     quad = i + j*nx
 * and hence
 *     i = quad % nx
 *     j = quad / nx
 *
 * Rather than referring to x- and y-directions, compass directions are used
 * instead such that W, E, S, N refer to the -x, +x, -y, +y directions
 * respectively.  To move one quad to the E you would therefore add 1 to the
 * quad index, to move one quad to the N you would add nx to the quad index.
 *
 * Cache
 * -----
 * Lots of information that is reused during contouring is stored as single
 * bits in a mesh-sized cache, indexed by quad.  Each quad's cache entry stores
 * information about the quad itself such as if it is masked, and about the
 * point at the SW corner of the quad, and about the W and S edges.  Hence
 * information about each point and each edge is only stored once in the cache.
 *
 * Cache information is divided into two types: that which is constant over the
 * lifetime of the QuadContourGenerator, and that which changes for each
 * contouring operation.  The former is all grid-specific information such
 * as quad and corner masks, and which edges are boundaries, either between
 * masked and non-masked regions or between adjacent chunks.  The latter
 * includes whether points lie above or below the current contour levels, plus
 * some flags to indicate how the contouring is progressing.
 *
 * Line Contours
 * -------------
 * A line contour connects points with the same z-value.  Each point of such a
 * contour occurs on an edge of the grid, at a point linearly interpolated to
 * the contour z-level from the z-values at the end points of the edge.  The
 * direction of a line contour is such that higher values are to the left of
 * the contour, so any edge that the contour passes through will have a left-
 * hand end point with z > contour level and a right-hand end point with
 * z <= contour level.
 *
 * Line contours are of two types.  Firstly there are open line strips that
 * start on a boundary, traverse the interior of the domain and end on a
 * boundary.  Secondly there are closed line loops that occur completely within
 * the interior of the domain and do not touch a boundary.
 *
 * The QuadContourGenerator makes two sweeps through the grid to generate line
 * contours for a particular level.  In the first sweep it looks only for start
 * points that occur on boundaries, and when it finds one it follows the
 * contour through the interior until it finishes on another boundary edge.
 * Each quad that is visited by the algorithm has a 'visited' flag set in the
 * cache to indicate that the quad does not need to be visited again.  In the
 * second sweep all non-visited quads are checked to see if they contain part
 * of an interior closed loop, and again each time one is found it is followed
 * through the domain interior until it returns back to its start quad and is
 * therefore completed.
 *
 * The situation is complicated by saddle quads that have two opposite corners
 * with z >= contour level and the other two corners with z < contour level.
 * These therefore contain two segments of a line contour, and the visited
 * flags take account of this by only being set on the second visit.  On the
 * first visit a number of saddle flags are set in the cache to indicate which
 * one of the two segments has been completed so far.
 *
 * Filled Contours
 * ---------------
 * Filled contours are produced between two contour levels and are always
 * closed polygons.  They can occur completely within the interior of the
 * domain without touching a boundary, following either the lower or upper
 * contour levels.  Those on the lower level are exactly like interior line
 * contours with higher values on the left.  Those on the upper level are
 * reversed such that higher values are on the right.
 *
 * Filled contours can also involve a boundary in which case they consist of
 * one or more sections along a boundary and one or more sections through the
 * interior.  Interior sections can be on either level, and again those on the
 * upper level have higher values on the right.  Boundary sections can remain
 * on either contour level or switch between the two.
 *
 * Once the start of a filled contour is found, the algorithm is similar to
 * that for line contours in that it follows the contour to its end, which
 * because filled contours are always closed polygons will be by returning
 * back to the start.  However, because two levels must be considered, each
 * level has its own set of saddle and visited flags and indeed some extra
 * visited flags for boundary edges.
 *
 * The major complication for filled contours is that some polygons can be
 * holes (with points ordered clockwise) within other polygons (with points
 * ordered anticlockwise).  When it comes to rendering filled contours each
 * non-hole polygon must be rendered along with its zero or more contained
 * holes or the rendering will not be correct.  The filled contour finding
 * algorithm could progress pretty much as the line contour algorithm does,
 * taking each polygon as it is found, but then at the end there would have to
 * be an extra step to identify the parent non-hole polygon for each hole.
 * This is not a particularly onerous task but it does not scale well and can
 * easily dominate the execution time of the contour finding for even modest
 * problems.  It is much better to identity each hole's parent non-hole during
 * the sweep algorithm.
 *
 * This requirement dictates the order that filled contours are identified.  As
 * the algorithm sweeps up through the grid, every time a polygon passes
 * through a quad a ParentCache object is updated with the new possible parent.
 * When a new hole polygon is started, the ParentCache is used to find the
 * first possible parent in the same quad or to the S of it.  Great care is
 * needed each time a new quad is checked to see if a new polygon should be
 * started, as a single quad can have multiple polygon starts, e.g. a quad
 * could be a saddle quad for both lower and upper contour levels, meaning it
 * has four contour line segments passing through it which could all be from
 * different polygons.  The S-most polygon must be started first, then the next
 * S-most and so on until the N-most polygon is started in that quad.
 */
#ifndef MPL_CONTOUR_H
#define MPL_CONTOUR_H

#include "src/numpy_cpp.h"
#include <stdint.h>
#include <list>
#include <iostream>
#include <vector>


// Edge of a quad including diagonal edges of masked quads if _corner_mask true.
typedef enum
{
    // Listing values here so easier to check for debug purposes.
    Edge_None = -1,
    Edge_E = 0,
    Edge_N = 1,
    Edge_W = 2,
    Edge_S = 3,
    // The following are only used if _corner_mask is true.
    Edge_NE = 4,
    Edge_NW = 5,
    Edge_SW = 6,
    Edge_SE = 7
} Edge;

// Combination of a quad and an edge of that quad.
// An invalid quad edge has quad of -1.
struct QuadEdge
{
    QuadEdge();
    QuadEdge(long quad_, Edge edge_);
    bool operator<(const QuadEdge& other) const;
    bool operator==(const QuadEdge& other) const;
    bool operator!=(const QuadEdge& other) const;
    friend std::ostream& operator<<(std::ostream& os,
                                    const QuadEdge& quad_edge);

    long quad;
    Edge edge;
};

// 2D point with x,y coordinates.
struct XY
{
    XY();
    XY(const double& x_, const double& y_);
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

// A single line of a contour, which may be a closed line loop or an open line
// strip.  Identical adjacent points are avoided using push_back().
// A ContourLine is either a hole (points ordered clockwise) or it is not
// (points ordered anticlockwise).  Each hole has a parent ContourLine that is
// not a hole; each non-hole contains zero or more child holes.  A non-hole and
// its child holes must be rendered together to obtain the correct results.
class ContourLine : public std::vector<XY>
{
public:
    typedef std::list<ContourLine*> Children;

    ContourLine(bool is_hole);
    void add_child(ContourLine* child);
    void clear_parent();
    const Children& get_children() const;
    const ContourLine* get_parent() const;
    ContourLine* get_parent();
    bool is_hole() const;
    void push_back(const XY& point);
    void set_parent(ContourLine* parent);
    void write() const;

private:
    bool _is_hole;
    ContourLine* _parent;  // Only set if is_hole, not owned.
    Children _children;    // Only set if !is_hole, not owned.
};


// A Contour is a collection of zero or more ContourLines.
class Contour : public std::vector<ContourLine*>
{
public:
    Contour();
    virtual ~Contour();
    void delete_contour_lines();
    void write() const;
};


// Single chunk of ContourLine parents, indexed by quad.  As a chunk's filled
// contours are created, the ParentCache is updated each time a ContourLine
// passes through each quad.  When a new ContourLine is created, if it is a
// hole its parent ContourLine is read from the ParentCache by looking at the
// start quad, then each quad to the S in turn until a non-zero ContourLine is
// found.
class ParentCache
{
public:
    ParentCache(long nx, long x_chunk_points, long y_chunk_points);
    ContourLine* get_parent(long quad);
    void set_chunk_starts(long istart, long jstart);
    void set_parent(long quad, ContourLine& contour_line);

private:
    long quad_to_index(long quad) const;

    long _nx;
    long _x_chunk_points, _y_chunk_points;  // Number of points not quads.
    std::vector<ContourLine*> _lines;       // Not owned.
    long _istart, _jstart;
};


// See overview of algorithm at top of file.
class QuadContourGenerator
{
public:
    typedef numpy::array_view<const double, 2> CoordinateArray;
    typedef numpy::array_view<const bool, 2> MaskArray;

    // Constructor with optional mask.
    //   x, y, z: double arrays of shape (ny,nx).
    //   mask: boolean array, ether empty (if no mask), or of shape (ny,nx).
    //   corner_mask: flag for different masking behaviour.
    //   chunk_size: 0 for no chunking, or +ve integer for size of chunks that
    //     the domain is subdivided into.
    QuadContourGenerator(const CoordinateArray& x,
                         const CoordinateArray& y,
                         const CoordinateArray& z,
                         const MaskArray& mask,
                         bool corner_mask,
                         long chunk_size);

    // Destructor.
    ~QuadContourGenerator();

    // Create and return polygons for a line (i.e. non-filled) contour at the
    // specified level.
    PyObject* create_contour(const double& level);

    // Create and return polygons for a filled contour between the two
    // specified levels.
    PyObject* create_filled_contour(const double& lower_level,
                                    const double& upper_level);

private:
    // Typedef for following either a boundary of the domain or the interior;
    // clearer than using a boolean.
    typedef enum
    {
        Boundary,
        Interior
    } BoundaryOrInterior;

    // Typedef for direction of movement from one quad to the next.
    typedef enum
    {
        Dir_Right    = -1,
        Dir_Straight =  0,
        Dir_Left     = +1
    } Dir;

    // Typedef for a polygon being a hole or not; clearer than using a boolean.
    typedef enum
    {
        NotHole,
        Hole
    } HoleOrNot;

    // Append a C++ ContourLine to the end of a python list.  Used for line
    // contours where each ContourLine is converted to a separate numpy array
    // of (x,y) points.
    // Clears the ContourLine too.
    void append_contour_line_to_vertices(ContourLine& contour_line,
                                         PyObject* vertices_list) const;

    // Append a C++ Contour to the end of two python lists.  Used for filled
    // contours where each non-hole ContourLine and its child holes are
    // represented by a numpy array of (x,y) points and a second numpy array of
    // 'kinds' or 'codes' that indicates where the points array is split into
    // individual polygons.
    // Clears the Contour too, freeing each ContourLine as soon as possible
    // for minimum RAM usage.
    void append_contour_to_vertices_and_codes(Contour& contour,
                                              PyObject* vertices_list,
                                              PyObject* codes_list) const;

    // Return number of chunks that fit in the specified point_count.
    long calc_chunk_count(long point_count) const;

    // Return the point on the specified QuadEdge that intersects the specified
    // level.
    XY edge_interp(const QuadEdge& quad_edge, const double& level);

    // Follow a contour along a boundary, appending points to the ContourLine
    // as it progresses.  Only called for filled contours.  Stops when the
    // contour leaves the boundary to move into the interior of the domain, or
    // when the start_quad_edge is reached in which case the ContourLine is a
    // completed closed loop.  Always adds the end point of each boundary edge
    // to the ContourLine, regardless of whether moving to another boundary
    // edge or leaving the boundary into the interior.  Never adds the start
    // point of the first boundary edge to the ContourLine.
    //   contour_line: ContourLine to append points to.
    //   quad_edge: on entry the QuadEdge to start from, on exit the QuadEdge
    //     that is stopped on.
    //   lower_level: lower contour z-value.
    //   upper_level: upper contour z-value.
    //   level_index: level index started on (1 = lower, 2 = upper level).
    //   start_quad_edge: QuadEdge that the ContourLine started from, which is
    //     used to check if the ContourLine is finished.
    // Returns the end level_index.
    unsigned int follow_boundary(ContourLine& contour_line,
                                 QuadEdge& quad_edge,
                                 const double& lower_level,
                                 const double& upper_level,
                                 unsigned int level_index,
                                 const QuadEdge& start_quad_edge);

    // Follow a contour across the interior of the domain, appending points to
    // the ContourLine as it progresses.  Called for both line and filled
    // contours.  Stops when the contour reaches a boundary or, if the
    // start_quad_edge is specified, when quad_edge == start_quad_edge and
    // level_index == start_level_index.  Always adds the end point of each
    // quad traversed to the ContourLine; only adds the start point of the
    // first quad if want_initial_point flag is true.
    //   contour_line: ContourLine to append points to.
    //   quad_edge: on entry the QuadEdge to start from, on exit the QuadEdge
    //     that is stopped on.
    //   level_index: level index started on (1 = lower, 2 = upper level).
    //   level: contour z-value.
    //   want_initial_point: whether want to append the initial point to the
    //     ContourLine or not.
    //   start_quad_edge: the QuadEdge that the ContourLine started from to
    //     check if the ContourLine is finished, or 0 if no check should occur.
    //   start_level_index: the level_index that the ContourLine started from.
    //   set_parents: whether should set ParentCache as it progresses or not.
    //     This is true for filled contours, false for line contours.
    void follow_interior(ContourLine& contour_line,
                         QuadEdge& quad_edge,
                         unsigned int level_index,
                         const double& level,
                         bool want_initial_point,
                         const QuadEdge* start_quad_edge,
                         unsigned int start_level_index,
                         bool set_parents);

    // Return the index limits of a particular chunk.
    void get_chunk_limits(long ijchunk,
                          long& ichunk,
                          long& jchunk,
                          long& istart,
                          long& iend,
                          long& jstart,
                          long& jend);

    // Check if a contour starts within the specified corner quad on the
    // specified level_index, and if so return the start edge.  Otherwise
    // return Edge_None.
    Edge get_corner_start_edge(long quad, unsigned int level_index) const;

    // Return index of point at start or end of specified QuadEdge, assuming
    // anticlockwise ordering around non-masked quads.
    long get_edge_point_index(const QuadEdge& quad_edge, bool start) const;

    // Return the edge to exit a quad from, given the specified entry quad_edge
    // and direction to move in.
    Edge get_exit_edge(const QuadEdge& quad_edge, Dir dir) const;

    // Return the (x,y) coordinates of the specified point index.
    XY get_point_xy(long point) const;

    // Return the z-value of the specified point index.
    const double& get_point_z(long point) const;

    // Check if a contour starts within the specified non-corner quad on the
    // specified level_index, and if so return the start edge.  Otherwise
    // return Edge_None.
    Edge get_quad_start_edge(long quad, unsigned int level_index) const;

    // Check if a contour starts within the specified quad, whether it is a
    // corner or a full quad, and if so return the start edge.  Otherwise
    // return Edge_None.
    Edge get_start_edge(long quad, unsigned int level_index) const;

    // Initialise the cache to contain grid information that is constant
    // across the lifetime of this object, i.e. does not vary between calls to
    // create_contour() and create_filled_contour().
    void init_cache_grid(const MaskArray& mask);

    // Initialise the cache with information that is specific to contouring the
    // specified two levels.  The levels are the same for contour lines,
    // different for filled contours.
    void init_cache_levels(const double& lower_level,
                           const double& upper_level);

    // Return the (x,y) point at which the level intersects the line connecting
    // the two specified point indices.
    XY interp(long point1, long point2, const double& level) const;

    // Return true if the specified QuadEdge is a boundary, i.e. is either an
    // edge between a masked and non-masked quad/corner or is a chunk boundary.
    bool is_edge_a_boundary(const QuadEdge& quad_edge) const;

    // Follow a boundary from one QuadEdge to the next in an anticlockwise
    // manner around the non-masked region.
    void move_to_next_boundary_edge(QuadEdge& quad_edge) const;

    // Move from the quad specified by quad_edge.quad to the neighbouring quad
    // by crossing the edge specified by quad_edge.edge.
    void move_to_next_quad(QuadEdge& quad_edge) const;

    // Check for filled contours starting within the specified quad and
    // complete any that are found, appending them to the specified Contour.
    void single_quad_filled(Contour& contour,
                            long quad,
                            const double& lower_level,
                            const double& upper_level);

    // Start and complete a filled contour line.
    //   quad: index of quad to start ContourLine in.
    //   edge: edge of quad to start ContourLine from.
    //   start_level_index: the level_index that the ContourLine starts from.
    //   hole_or_not: whether the ContourLine is a hole or not.
    //   boundary_or_interior: whether the ContourLine starts on a boundary or
    //     the interior.
    //   lower_level: lower contour z-value.
    //   upper_level: upper contour z-value.
    // Returns newly created ContourLine.
    ContourLine* start_filled(long quad,
                              Edge edge,
                              unsigned int start_level_index,
                              HoleOrNot hole_or_not,
                              BoundaryOrInterior boundary_or_interior,
                              const double& lower_level,
                              const double& upper_level);

    // Start and complete a line contour that both starts and end on a
    // boundary, traversing the interior of the domain.
    //   vertices_list: Python list that the ContourLine should be appended to.
    //   quad: index of quad to start ContourLine in.
    //   edge: boundary edge to start ContourLine from.
    //   level: contour z-value.
    // Returns true if the start quad does not need to be visited again, i.e.
    // VISITED(quad,1).
    bool start_line(PyObject* vertices_list,
                    long quad,
                    Edge edge,
                    const double& level);

    // Debug function that writes the cache status to stdout.
    void write_cache(bool grid_only = false) const;

    // Debug function that writes that cache status for a single quad to
    // stdout.
    void write_cache_quad(long quad, bool grid_only) const;



    // Note that mask is not stored as once it has been used to initialise the
    // cache it is no longer needed.
    CoordinateArray _x, _y, _z;
    long _nx, _ny;             // Number of points in each direction.
    long _n;                   // Total number of points (and hence quads).

    bool _corner_mask;
    long _chunk_size;          // Number of quads per chunk (not points).
                               // Always > 0, unlike python nchunk which is 0
                               //     for no chunking.

    long _nxchunk, _nychunk;   // Number of chunks in each direction.
    long _chunk_count;         // Total number of chunks.

    typedef uint32_t CacheItem;
    CacheItem* _cache;

    ParentCache _parent_cache; // On W quad sides.
};

#endif // _CONTOUR_H

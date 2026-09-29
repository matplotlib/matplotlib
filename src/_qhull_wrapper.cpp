/*
 * Wrapper module for libqhull, providing Delaunay triangulation.
 *
 * This module's methods should not be accessed directly.  To obtain a Delaunay
 * triangulation, construct an instance of the matplotlib.tri.Triangulation
 * class without specifying a triangles array.
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nanobind::literals;

#ifdef _MSC_VER
/* The Qhull header does not declare this as extern "C", but only MSVC seems to
 * do name mangling on global variables. We thus need to declare this before
 * the header so that it treats it correctly, and doesn't mangle the name. */
extern "C" {
extern const char qh_version[];
}
#endif

#include "mplutils.h"
#include "libqhull_r/qhull_ra.h"
#include <cstdio>
#include <vector>

#ifndef MPL_DEVNULL
#error "MPL_DEVNULL must be defined as the OS-equivalent of /dev/null"
#endif

#define STRINGIFY(x) STR(x)
#define STR(x) #x

// Input numpy array class.
using CoordArray = nb::ndarray<double, nb::ndim<1>, nb::c_contig>;

// Output numpy array class.
using IndexArray = nb::ndarray<int, nb::shape<-1, 3>, nb::numpy, nb::c_contig>;

static const char* qhull_error_msg[6] = {
    "",                     /* 0 = qh_ERRnone */
    "input inconsistency",  /* 1 = qh_ERRinput */
    "singular input data",  /* 2 = qh_ERRsingular */
    "precision error",      /* 3 = qh_ERRprec */
    "insufficient memory",  /* 4 = qh_ERRmem */
    "internal error"};      /* 5 = qh_ERRqhull */


/* Return the indices of the 3 vertices that comprise the specified facet (i.e.
 * triangle). */
static void
get_facet_vertices(qhT* qh, const facetT* facet, int indices[3])
{
    vertexT *vertex, **vertexp;
    // After qh_triangulate() every non-upperdelaunay facet is a triangle, so
    // this macro iterates exactly 3 times; the analyzer cannot prove this.
    FOREACHvertex_(facet->vertices) {  // NOLINT(clang-analyzer-security.ArrayBound)
        *indices++ = qh_pointid(qh, vertex->point);
    }
}

/* Return the indices of the 3 triangles that are neighbors of the specified
 * facet (triangle). */
static void
get_facet_neighbours(const facetT* facet, std::vector<int>& tri_indices,
                     int indices[3])
{
    facetT *neighbor, **neighborp;
    // After qh_triangulate() every non-upperdelaunay facet is a triangle with
    // exactly 3 neighbors; the analyzer cannot prove this.
    FOREACHneighbor_(facet) {  // NOLINT(clang-analyzer-security.ArrayBound)
        *indices++ = (neighbor->upperdelaunay ? -1 : tri_indices[neighbor->id]);
    }
}

/* Return true if the specified points arrays contain at least 3 unique points,
 * or false otherwise. */
static bool
at_least_3_unique_points(nb::ssize_t npoints, const double* x, const double* y)
{
    const nb::ssize_t unique1 = 0;  /* First unique point has index 0. */
    nb::ssize_t unique2 = 0;        /* Second unique point index is 0 until set. */

    if (npoints < 3) {
        return false;
    }

    for (nb::ssize_t i = 1; i < npoints; ++i) {
        if (unique2 == 0) {
            /* Looking for second unique point. */
            if (x[i] != x[unique1] || y[i] != y[unique1]) {
                unique2 = i;
            }
        }
        else {
            /* Looking for third unique point. */
            if ( (x[i] != x[unique1] || y[i] != y[unique1]) &&
                 (x[i] != x[unique2] || y[i] != y[unique2]) ) {
                /* 3 unique points found, with indices 0, unique2 and i. */
                return true;
            }
        }
    }

    /* Run out of points before 3 unique points found. */
    return false;
}

/* Holds on to info from Qhull so that it can be destructed automatically. */
class QhullInfo {
public:
    QhullInfo(FILE *error_file, qhT* qh) {
        this->error_file = error_file;
        this->qh = qh;
    }

    ~QhullInfo() {
        qh_freeqhull(this->qh, !qh_ALL);
        int curlong, totlong;  /* Memory remaining. */
        qh_memfreeshort(this->qh, &curlong, &totlong);
        if (curlong || totlong) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Qhull could not free all allocated memory", 1);
        }

        if (this->error_file != stderr) {
            fclose(error_file);
        }
    }

private:
    FILE* error_file;
    qhT* qh;
};

/* Delaunay implementation method.
 * If hide_qhull_errors is true then qhull error messages are discarded;
 * if it is false then they are written to stderr. */
static nb::tuple
delaunay_impl(nb::ssize_t npoints, const double* x, const double* y,
              bool hide_qhull_errors)
{
    qhT qh_qh;                  /* qh variable type and name must be like */
    qhT* qh = &qh_qh;           /* this for Qhull macros to work correctly. */
    facetT* facet;
    size_t i, ntri;
    int max_facet_id;
    int exitcode;               /* Value returned from qh_new_qhull(). */
    const int ndim = 2;
    double x_mean = 0.0;
    double y_mean = 0.0;

    QHULL_LIB_CHECK

    /* Allocate points. */
    std::vector<coordT> points(npoints * ndim);

    /* Determine mean x, y coordinates. */
    for (i = 0; i < npoints; ++i) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= npoints;
    y_mean /= npoints;

    /* Prepare points array to pass to qhull. */
    for (i = 0; i < npoints; ++i) {
        points[2*i  ] = x[i] - x_mean;
        points[2*i+1] = y[i] - y_mean;
    }

    /* qhull expects a FILE* to write errors to. */
    FILE* error_file = nullptr;
    if (hide_qhull_errors) {
        /* qhull errors are ignored by writing to OS-equivalent of /dev/null.
         * Rather than have OS-specific code here, instead it is determined by
         * meson.build and passed in via the macro MPL_DEVNULL. */
        error_file = fopen(STRINGIFY(MPL_DEVNULL), "w");
        if (error_file == nullptr) {
            throw std::runtime_error("Could not open devnull");
        }
    }
    else {
        /* qhull errors written to stderr. */
        error_file = stderr;
    }

    /* Perform Delaunay triangulation. */
    QhullInfo info(error_file, qh);
    qh_zero(qh, error_file);
    exitcode = qh_new_qhull(qh, ndim, (int)npoints, points.data(), False,
                            (char*)"qhull d Qt Qbb Qc Qz", nullptr, error_file);
    if (exitcode != qh_ERRnone) {
        std::string msg = nb::cast<std::string>(
            nb::str("Error in qhull Delaunay triangulation calculation: {} (exitcode={})")
            .format(qhull_error_msg[exitcode], exitcode));
        if (hide_qhull_errors) {
            msg += "; use python verbose option (-v) to see original qhull error.";
        }
        throw std::runtime_error(msg);
    }

    /* Split facets so that they only have 3 points each. */
    qh_triangulate(qh);

    /* Determine ntri and max_facet_id.
       Note that libqhull uses macros to iterate through collections. */
    ntri = 0;
    FORALLfacets {
        if (!facet->upperdelaunay) {
            ++ntri;
        }
    }

    max_facet_id = qh->facet_id - 1;

    /* Create array to map facet id to triangle index. */
    std::vector<int> tri_indices(max_facet_id+1);

    /* Allocate Python arrays to return. */
    auto triangles = mpl_make_numpy_array<IndexArray>({ntri, 3});
    int* triangles_ptr = triangles.data();

    auto neighbors = mpl_make_numpy_array<IndexArray>({ntri, 3});
    int* neighbors_ptr = neighbors.data();

    /* Determine triangles array and set tri_indices array. */
    i = 0;
    FORALLfacets {
        if (!facet->upperdelaunay) {
            int indices[3];
            tri_indices[facet->id] = i++;
            get_facet_vertices(qh, facet, indices);
            // The analyzer takes a path where FOREACHvertex_ executes zero
            // times, leaving indices[] uninitialized.  After qh_triangulate()
            // every non-upperdelaunay facet is a triangle with exactly 3
            // vertices, so this cannot happen.
            *triangles_ptr++ = (facet->toporient ? indices[0] : indices[2]);  // NOLINT(clang-analyzer-core.uninitialized.Assign)
            *triangles_ptr++ = indices[1];
            *triangles_ptr++ = (facet->toporient ? indices[2] : indices[0]);
        }
        else {
            tri_indices[facet->id] = -1;
        }
    }

    /* Determine neighbors array. */
    FORALLfacets {
        if (!facet->upperdelaunay) {
            int indices[3];
            get_facet_neighbours(facet, tri_indices, indices);
            *neighbors_ptr++ = (facet->toporient ? indices[2] : indices[0]);
            *neighbors_ptr++ = (facet->toporient ? indices[0] : indices[2]);
            *neighbors_ptr++ = indices[1];
        }
    }

    return nb::make_tuple(triangles, neighbors);
}

/* Process Python arguments and call Delaunay implementation method. */
static nb::tuple
delaunay(const CoordArray& x, const CoordArray& y, int verbose)
{
    if (x.ndim() != 1 || y.ndim() != 1) {
        throw std::invalid_argument("x and y must be 1D arrays");
    }

    auto npoints = x.shape(0);
    if (npoints != y.shape(0)) {
        throw std::invalid_argument("x and y must be 1D arrays of the same length");
    }

    if (npoints < 3) {
        throw std::invalid_argument("x and y arrays must have a length of at least 3");
    }

    if (!at_least_3_unique_points(npoints, x.data(), y.data())) {
        throw std::invalid_argument("x and y arrays must consist of at least 3 unique points");
    }

    return delaunay_impl(npoints, x.data(), y.data(), verbose == 0);
}

NB_MODULE(_qhull, m)
{
    m.doc() = "Computing Delaunay triangulations.\n";

    m.def("delaunay", &delaunay, "x"_a, "y"_a, "verbose"_a,
        "--\n\n"
        "Compute a Delaunay triangulation.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "x, y : 1d arrays\n"
        "    The coordinates of the point set, which must consist of at least\n"
        "    three unique points.\n"
        "verbose : int\n"
        "    Python's verbosity level.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "triangles, neighbors : int arrays, shape (ntri, 3)\n"
        "    Indices of triangle vertices and indices of triangle neighbors.\n");

    m.def("version", []() { return qh_version; },
        "version()\n--\n\n"
        "Return the qhull version string.");
}

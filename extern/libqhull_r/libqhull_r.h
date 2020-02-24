/*<html><pre>  -<a                             href="qh-qhull_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   libqhull_r.h
   user-level header file for using qhull.a library

   see qh-qhull_r.htm, qhull_ra.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/libqhull_r.h#13 $$Change: 2714 $
   $DateTime: 2019/06/28 16:16:13 $$Author: bbarber $

   includes function prototypes for libqhull_r.c, geom_r.c, global_r.c, io_r.c, user_r.c

   use mem_r.h for mem_r.c
   use qset_r.h for qset_r.c

   see unix_r.c for an example of using libqhull_r.h

   recompile qhull if you change this file
*/

#ifndef qhDEFlibqhull
#define qhDEFlibqhull 1

/*=========================== -included files ==============*/

/* user_r.h first for QHULL_CRTDBG */
#include "user_r.h"      /* user definable constants (e.g., realT). */

#include "mem_r.h"   /* Needed for qhT in libqhull_r.h */
#include "qset_r.h"   /* Needed for QHULL_LIB_CHECK */
/* include stat_r.h after defining boolT.  Needed for qhT in libqhull_r.h */

#include <setjmp.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <stdio.h>

#ifndef __STDC__
#ifndef __cplusplus
#if     !defined(_MSC_VER)
#error  Neither __STDC__ nor __cplusplus is defined.  Please use strict ANSI C or C++ to compile
#error  Qhull.  You may need to turn off compiler extensions in your project configuration.  If
#error  your compiler is a standard C compiler, you can delete this warning from libqhull_r.h
#endif
#endif
#endif

/*============ constants and basic types ====================*/

extern const char qh_version[]; /* defined in global_r.c */
extern const char qh_version2[]; /* defined in global_r.c */

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="coordT">-</a>

  coordT
    coordinates and coefficients are stored as realT (i.e., double)

  notes:
    Qhull works well if realT is 'float'.  If so joggle (QJ) is not effective.

    Could use 'float' for data and 'double' for calculations (realT vs. coordT)
      This requires many type casts, and adjusted error bounds.
      Also C compilers may do expressions in double anyway.
*/
#define coordT realT

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="pointT">-</a>

  pointT
    a point is an array of coordinates, usually qh.hull_dim
    qh_pointid returns
      qh_IDnone if point==0 or qh is undefined
      qh_IDinterior for qh.interior_point
      qh_IDunknown if point is neither in qh.first_point... nor qh.other_points

  notes:
    qh.STOPcone and qh.STOPpoint assume that qh_IDunknown==-1 (other negative numbers indicate points)
    qh_IDunknown is also returned by getid_() for unknown facet, ridge, or vertex
*/
#define pointT coordT
typedef enum
{
    qh_IDnone= -3, qh_IDinterior= -2, qh_IDunknown= -1
}
qh_pointT;

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="flagT">-</a>

  flagT
    Boolean flag as a bit
*/
#define flagT unsigned int

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="boolT">-</a>

  boolT
    boolean value, either True or False

  notes:
    needed for portability
    Use qh_False/qh_True as synonyms
*/
#define boolT unsigned int
#ifdef False
#undef False
#endif
#ifdef True
#undef True
#endif
#define False 0
#define True 1
#define qh_False 0
#define qh_True 1

#include "stat_r.h"  /* needs boolT */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="CENTERtype">-</a>

  qh_CENTER
    to distinguish facet->center
*/
typedef enum
{
    qh_ASnone= 0,    /* If not MERGING and not VORONOI */
    qh_ASvoronoi,    /* Set by qh_clearcenters on qh_prepare_output, or if not MERGING and VORONOI */
    qh_AScentrum     /* If MERGING (assumed during merging) */
}
qh_CENTER;

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="qh_PRINT">-</a>

  qh_PRINT
    output formats for printing (qh.PRINTout).
    'Fa' 'FV' 'Fc' 'FC'


   notes:
   some of these names are similar to qhT names.  The similar names are only
   used in switch statements in qh_printbegin() etc.
*/
typedef enum {qh_PRINTnone= 0,
  qh_PRINTarea, qh_PRINTaverage,           /* 'Fa' 'FV' 'Fc' 'FC' */
  qh_PRINTcoplanars, qh_PRINTcentrums,
  qh_PRINTfacets, qh_PRINTfacets_xridge,   /* 'f' 'FF' 'G' 'FI' 'Fi' 'Fn' */
  qh_PRINTgeom, qh_PRINTids, qh_PRINTinner, qh_PRINTneighbors,
  qh_PRINTnormals, qh_PRINTouter, qh_PRINTmaple, /* 'n' 'Fo' 'i' 'm' 'Fm' 'FM', 'o' */
  qh_PRINTincidences, qh_PRINTmathematica, qh_PRINTmerges, qh_PRINToff,
  qh_PRINToptions, qh_PRINTpointintersect, /* 'FO' 'Fp' 'FP' 'p' 'FQ' 'FS' */
  qh_PRINTpointnearest, qh_PRINTpoints, qh_PRINTqhull, qh_PRINTsize,
  qh_PRINTsummary, qh_PRINTtriangles,      /* 'Fs' 'Ft' 'Fv' 'FN' 'Fx' */
  qh_PRINTvertices, qh_PRINTvneighbors, qh_PRINTextremes,
  qh_PRINTEND} qh_PRINT;

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="qh_ALL">-</a>

  qh_ALL
    argument flag for selecting everything
*/
#define qh_ALL      True
#define qh_NOupper  True      /* argument for qh_findbest */
#define qh_IScheckmax  True   /* argument for qh_findbesthorizon */
#define qh_ISnewfacets  True  /* argument for qh_findbest */
#define qh_RESETvisible  True /* argument for qh_resetlists */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="qh_ERR">-</a>

  qh_ERR...
    Qhull exit status codes, for indicating errors
    See: MSG_ERROR (6000) and MSG_WARNING (7000) [user_r.h]
*/
#define qh_ERRnone  0    /* no error occurred during qhull */
#define qh_ERRinput 1    /* input inconsistency */
#define qh_ERRsingular 2 /* singular input data, calls qh_printhelp_singular */
#define qh_ERRprec  3    /* precision error, calls qh_printhelp_degenerate */
#define qh_ERRmem   4    /* insufficient memory, matches mem_r.h */
#define qh_ERRqhull 5    /* internal error detected, matches mem_r.h, calls qh_printhelp_internal */
#define qh_ERRother 6    /* other error detected */
#define qh_ERRtopology 7 /* topology error, maybe due to nearly adjacent vertices, calls qh_printhelp_topology */
#define qh_ERRwide 8     /* wide facet error, maybe due to nearly adjacent vertices, calls qh_printhelp_wide */
#define qh_ERRdebug 9   /* qh_errexit from debugging code */

/*-<a                             href="qh-qhull_r.htm#TOC"
>--------------------------------</a><a name="qh_FILEstderr">-</a>

qh_FILEstderr
Fake stderr to distinguish error output from normal output
For C++ interface.  Must redefine qh_fprintf_qhull
*/
#define qh_FILEstderr ((FILE *)1)

/* ============ -structures- ====================
   each of the following structures is defined by a typedef
   all realT and coordT fields occur at the beginning of a structure
        (otherwise space may be wasted due to alignment)
   define all flags together and pack into 32-bit number

   DEFqhT and DEFsetT are likewise defined in mem_r.h, qset_r.h, and stat_r.h
*/

typedef struct vertexT vertexT;
typedef struct ridgeT ridgeT;
typedef struct facetT facetT;

#ifndef DEFqhT
#define DEFqhT 1
typedef struct qhT qhT;          /* defined below */
#endif

#ifndef DEFsetT
#define DEFsetT 1
typedef struct setT setT;        /* defined in qset_r.h */
#endif

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="facetT">-</a>

  facetT
    defines a facet

  notes:
   qhull() generates the hull as a list of facets.

  topological information:
    f.previous,next     doubly-linked list of facets, next is always defined
    f.vertices          set of vertices
    f.ridges            set of ridges
    f.neighbors         set of neighbors
    f.toporient         True if facet has top-orientation (else bottom)

  geometric information:
    f.offset,normal     hyperplane equation
    f.maxoutside        offset to outer plane -- all points inside
    f.center            centrum for testing convexity or Voronoi center for output
    f.simplicial        True if facet is simplicial
    f.flipped           True if facet does not include qh.interior_point

  for constructing hull:
    f.visible           True if facet on list of visible facets (will be deleted)
    f.newfacet          True if facet on list of newly created facets
    f.coplanarset       set of points coplanar with this facet
                        (includes near-inside points for later testing)
    f.outsideset        set of points outside of this facet
    f.furthestdist      distance to furthest point of outside set
    f.visitid           marks visited facets during a loop
    f.replace           replacement facet for to-be-deleted, visible facets
    f.samecycle,newcycle cycle of facets for merging into horizon facet

  see below for other flags and fields
*/
/* QhullFacet.cpp -- Update static initializer list for s_empty_facet if add or remove fields */
struct facetT {
#if !qh_COMPUTEfurthest
  coordT   furthestdist;/* distance to furthest point of outsideset */
#endif
#if qh_MAXoutside
  coordT   maxoutside;  /* max computed distance of point to facet
                        Before QHULLfinished this is an approximation
                        since maxdist not always set for qh_mergefacet
                        Actual outer plane is +DISTround and
                        computed outer plane is +2*DISTround.
                        Initial maxoutside is qh.DISTround, otherwise distance tests need to account for DISTround */
#endif
  coordT   offset;      /* exact offset of hyperplane from origin */
  coordT  *normal;      /* normal of hyperplane, hull_dim coefficients */
                        /*   if f.tricoplanar, shared with a neighbor */
  union {               /* in order of testing */
   realT   area;        /* area of facet, only in io_r.c if  f.isarea */
   facetT *replace;     /* replacement facet for qh.NEWfacets with f.visible
                             NULL if qh_mergedegen_redundant, interior, or !NEWfacets */
   facetT *samecycle;   /* cycle of facets from the same visible/horizon intersection,
                             if ->newfacet */
   facetT *newcycle;    /*  in horizon facet, current samecycle of new facets */
   facetT *trivisible;  /* visible facet for ->tricoplanar facets during qh_triangulate() */
   facetT *triowner;    /* owner facet for ->tricoplanar, !isarea facets w/ ->keepcentrum */
  }f;
  coordT  *center;      /* set according to qh.CENTERtype */
                        /*   qh_ASnone:    no center (not MERGING) */
                        /*   qh_AScentrum: centrum for testing convexity (qh_getcentrum) */
                        /*                 assumed qh_AScentrum while merging */
                        /*   qh_ASvoronoi: Voronoi center (qh_facetcenter) */
                        /* after constructing the hull, it may be changed (qh_clearcenter) */
                        /* if tricoplanar and !keepcentrum, shared with a neighbor */
  facetT  *previous;    /* previous facet in the facet_list or NULL, for C++ interface */
  facetT  *next;        /* next facet in the facet_list or facet_tail */
  setT    *vertices;    /* vertices for this facet, inverse sorted by ID
                           if simplicial, 1st vertex was apex/furthest
                           qh_reduce_vertices removes extraneous vertices via qh_remove_extravertices
                           if f.visible, vertices may be on qh.del_vertices */
  setT    *ridges;      /* explicit ridges for nonsimplicial facets or nonsimplicial neighbors.
                           For simplicial facets, neighbors define the ridges
                           qh_makeridges() converts simplicial facets by creating ridges prior to merging
                           If qh.NEWtentative, new facets have horizon ridge, but not vice versa
                           if f.visible && qh.NEWfacets, ridges is empty */
  setT    *neighbors;   /* neighbors of the facet.  Neighbors may be f.visible
                           If simplicial, the kth neighbor is opposite the kth vertex and the
                           first neighbor is the horizon facet for the first vertex.
                           dupridges marked by qh_DUPLICATEridge (0x01) and qh_MERGEridge (0x02)
                           if f.visible && qh.NEWfacets, neighbors is empty */
  setT    *outsideset;  /* set of points outside this facet
                           if non-empty, last point is furthest
                           if NARROWhull, includes coplanars (less than qh.MINoutside) for partitioning*/
  setT    *coplanarset; /* set of points coplanar with this facet
                           >= qh.min_vertex and <= facet->max_outside
                           a point is assigned to the furthest facet
                           if non-empty, last point is furthest away */
  unsigned int visitid; /* visit_id, for visiting all neighbors,
                           all uses are independent */
  unsigned int id;      /* unique identifier from qh.facet_id, 1..qh.facet_id, 0 is sentinel, printed as 'f%d' */
  unsigned int nummerge:9; /* number of merges */
#define qh_MAXnummerge 511 /* 2^9-1 */
                        /* 23 flags (at most 23 due to nummerge), printed by "flags:" in io_r.c */
  flagT    tricoplanar:1; /* True if TRIangulate and simplicial and coplanar with a neighbor */
                          /*   all tricoplanars share the same apex */
                          /*   all tricoplanars share the same ->center, ->normal, ->offset, ->maxoutside */
                          /*     ->keepcentrum is true for the owner.  It has the ->coplanareset */
                          /*   if ->degenerate, does not span facet (one logical ridge) */
                          /*   during qh_triangulate, f.trivisible points to original facet */
  flagT    newfacet:1;  /* True if facet on qh.newfacet_list (new/qh.first_newfacet or merged) */
  flagT    visible:1;   /* True if visible facet (will be deleted) */
  flagT    toporient:1; /* True if created with top orientation
                           after merging, use ridge orientation */
  flagT    simplicial:1;/* True if simplicial facet, ->ridges may be implicit */
  flagT    seen:1;      /* used to perform operations only once, like visitid */
  flagT    seen2:1;     /* used to perform operations only once, like visitid */
  flagT    flipped:1;   /* True if facet is flipped */
  flagT    upperdelaunay:1; /* True if facet is upper envelope of Delaunay triangulation */
  flagT    notfurthest:1; /* True if last point of outsideset is not furthest */

/*-------- flags primarily for output ---------*/
  flagT    good:1;      /* True if a facet marked good for output */
  flagT    isarea:1;    /* True if facet->f.area is defined */

/*-------- flags for merging ------------------*/
  flagT    dupridge:1;  /* True if facet has one or more dupridge in a new facet (qh_matchneighbor),
                             a dupridge has a subridge shared by more than one new facet */
  flagT    mergeridge:1; /* True if facet or neighbor has a qh_MERGEridge (qh_mark_dupridges)
                            ->normal defined for mergeridge and mergeridge2 */
  flagT    mergeridge2:1; /* True if neighbor has a qh_MERGEridge (qh_mark_dupridges) */
  flagT    coplanarhorizon:1;  /* True if horizon facet is coplanar at last use */
  flagT     mergehorizon:1; /* True if will merge into horizon (its first neighbor w/ f.coplanarhorizon). */
  flagT     cycledone:1;/* True if mergecycle_all already done */
  flagT    tested:1;    /* True if facet convexity has been tested (false after merge */
  flagT    keepcentrum:1; /* True if keep old centrum after a merge, or marks owner for ->tricoplanar
                             Set by qh_updatetested if more than qh_MAXnewcentrum extra vertices
                             Set by qh_mergefacet if |maxdist| > qh.WIDEfacet */
  flagT    newmerge:1;  /* True if facet is newly merged for reducevertices */
  flagT    degenerate:1; /* True if facet is degenerate (degen_mergeset or ->tricoplanar) */
  flagT    redundant:1;  /* True if facet is redundant (degen_mergeset)
                         Maybe merge degenerate and redundant to gain another flag */
};


/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="ridgeT">-</a>

  ridgeT
    defines a ridge

  notes:
  a ridge is hull_dim-1 simplex between two neighboring facets.  If the
  facets are non-simplicial, there may be more than one ridge between
  two facets.  E.G. a 4-d hypercube has two triangles between each pair
  of neighboring facets.

  topological information:
    vertices            a set of vertices
    top,bottom          neighboring facets with orientation

  geometric information:
    tested              True if ridge is clearly convex
    nonconvex           True if ridge is non-convex
*/
/* QhullRidge.cpp -- Update static initializer list for s_empty_ridge if add or remove fields */
struct ridgeT {
  setT    *vertices;    /* vertices belonging to this ridge, inverse sorted by ID
                           NULL if a degen ridge (matchsame) */
  facetT  *top;         /* top facet for this ridge */
  facetT  *bottom;      /* bottom facet for this ridge
                        ridge oriented by odd/even vertex order and top/bottom */
  unsigned int id;      /* unique identifier.  Same size as vertex_id, printed as 'r%d' */
  flagT    seen:1;      /* used to perform operations only once */
  flagT    tested:1;    /* True when ridge is tested for convexity by centrum or opposite vertices */
  flagT    nonconvex:1; /* True if getmergeset detected a non-convex neighbor
                           only one ridge between neighbors may have nonconvex */
  flagT    mergevertex:1; /* True if pending qh_appendvertexmerge due to
                             qh_maybe_duplicateridge or qh_maybe_duplicateridges
                             disables check for duplicate vertices in qh_checkfacet */
  flagT    mergevertex2:1; /* True if qh_drop_mergevertex of MRGvertices, printed but not used */
  flagT    simplicialtop:1; /* True if top was simplicial (original vertices) */
  flagT    simplicialbot:1; /* True if bottom was simplicial (original vertices)
                             use qh_test_centrum_merge if top and bot, need to retest since centrum may change */
};

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="vertexT">-</a>

  vertexT
     defines a vertex

  topological information:
    next,previous       doubly-linked list of all vertices
    neighbors           set of adjacent facets (only if qh.VERTEXneighbors)

  geometric information:
    point               array of DIM3 coordinates
*/
/* QhullVertex.cpp -- Update static initializer list for s_empty_vertex if add or remove fields */
struct vertexT {
  vertexT *next;        /* next vertex in vertex_list or vertex_tail */
  vertexT *previous;    /* previous vertex in vertex_list or NULL, for C++ interface */
  pointT  *point;       /* hull_dim coordinates (coordT) */
  setT    *neighbors;   /* neighboring facets of vertex, qh_vertexneighbors()
                           initialized in io_r.c or after first merge
                           qh_update_vertices for qh_addpoint or qh_triangulate
                           updated by merges
                           qh_order_vertexneighbors for 2-d and 3-d */
  unsigned int id;      /* unique identifier, 1..qh.vertex_id,  0 for sentinel, printed as 'r%d' */
  unsigned int visitid; /* for use with qh.vertex_visit, size must match */
  flagT    seen:1;      /* used to perform operations only once */
  flagT    seen2:1;     /* another seen flag */
  flagT    deleted:1;   /* vertex will be deleted via qh.del_vertices */
  flagT    delridge:1;  /* vertex belonged to a deleted ridge, cleared by qh_reducevertices */
  flagT    newfacet:1;  /* true if vertex is in a new facet
                           vertex is on qh.newvertex_list and it has a facet on qh.newfacet_list
                           or vertex is on qh.newvertex_list due to qh_newvertices while merging
                           cleared by qh_resetlists */
  flagT    partitioned:1; /* true if deleted vertex has been partitioned */
};

/*======= -global variables -qh ============================*/

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh">-</a>

  qhT
   All global variables for qhull are in qhT.  It includes qhmemT, qhstatT, and rbox globals

   This version of Qhull is reentrant, but it is not thread-safe.

   Do not run separate threads on the same instance of qhT.

   QHULL_LIB_CHECK checks that a program and the corresponding
   qhull library were built with the same type of header files.

   QHULL_LIB_TYPE is QHULL_NON_REENTRANT, QHULL_QH_POINTER, or QHULL_REENTRANT
*/

#define QHULL_NON_REENTRANT 0
#define QHULL_QH_POINTER 1
#define QHULL_REENTRANT 2

#define QHULL_LIB_TYPE QHULL_REENTRANT

#define QHULL_LIB_CHECK qh_lib_check(QHULL_LIB_TYPE, sizeof(qhT), sizeof(vertexT), sizeof(ridgeT), sizeof(facetT), sizeof(setT), sizeof(qhmemT));
#define QHULL_LIB_CHECK_RBOX qh_lib_check(QHULL_LIB_TYPE, sizeof(qhT), sizeof(vertexT), sizeof(ridgeT), sizeof(facetT), 0, 0);

struct qhT {

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-const">-</a>

  qh constants
    configuration flags and constants for Qhull

  notes:
    The user configures Qhull by defining flags.  They are
    copied into qh by qh_setflags().  qh-quick_r.htm#options defines the flags.
*/
  boolT ALLpoints;        /* true 'Qs' if search all points for initial simplex */
  boolT ALLOWshort;       /* true 'Qa' allow input with fewer or more points than coordinates */
  boolT ALLOWwarning;     /* true 'Qw' if allow option warnings */
  boolT ALLOWwide;        /* true 'Q12' if allow wide facets and wide dupridges, c.f. qh_WIDEmaxoutside */
  boolT ANGLEmerge;       /* true 'Q1' if sort potential merges by type/angle instead of type/distance  */
  boolT APPROXhull;       /* true 'Wn' if MINoutside set */
  realT MINoutside;       /*   Minimum distance for an outside point ('Wn' or 2*qh.MINvisible) */
  boolT ANNOTATEoutput;   /* true 'Ta' if annotate output with message codes */
  boolT ATinfinity;       /* true 'Qz' if point num_points-1 is "at-infinity"
                             for improving precision in Delaunay triangulations */
  boolT AVOIDold;         /* true 'Q4' if avoid old->new merges */
  boolT BESToutside;      /* true 'Qf' if partition points into best outsideset */
  boolT CDDinput;         /* true 'Pc' if input uses CDD format (1.0/offset first) */
  boolT CDDoutput;        /* true 'PC' if print normals in CDD format (offset first) */
  boolT CHECKduplicates;  /* true 'Q15' if qh_maybe_duplicateridges after each qh_mergefacet */
  boolT CHECKfrequently;  /* true 'Tc' if checking frequently */
  realT premerge_cos;     /*   'A-n'   cos_max when pre merging */
  realT postmerge_cos;    /*   'An'    cos_max when post merging */
  boolT DELAUNAY;         /* true 'd' or 'v' if computing DELAUNAY triangulation */
  boolT DOintersections;  /* true 'Gh' if print hyperplane intersections */
  int   DROPdim;          /* drops dim 'GDn' for 4-d -> 3-d output */
  boolT FLUSHprint;       /* true 'Tf' if flush after qh_fprintf for segfaults */
  boolT FORCEoutput;      /* true 'Po' if forcing output despite degeneracies */
  int   GOODpoint;        /* 'QGn' or 'QG-n' (n+1, n-1), good facet if visible from point n (or not) */
  pointT *GOODpointp;     /*   the actual point */
  boolT GOODthreshold;    /* true 'Pd/PD' if qh.lower_threshold/upper_threshold defined
                             set if qh.UPPERdelaunay (qh_initbuild)
                             false if qh.SPLITthreshold */
  int   GOODvertex;       /* 'QVn' or 'QV-n' (n+1, n-1), good facet if vertex for point n (or not) */
  pointT *GOODvertexp;     /*   the actual point */
  boolT HALFspace;        /* true 'Hn,n,n' if halfspace intersection */
  boolT ISqhullQh;        /* Set by Qhull.cpp on initialization */
  int   IStracing;        /* 'Tn' trace execution, 0=none, 1=least, 4=most, -1=events */
  int   KEEParea;         /* 'PAn' number of largest facets to keep */
  boolT KEEPcoplanar;     /* true 'Qc' if keeping nearest facet for coplanar points */
  boolT KEEPinside;       /* true 'Qi' if keeping nearest facet for inside points
                              set automatically if 'd Qc' */
  int   KEEPmerge;        /* 'PMn' number of facets to keep with most merges */
  realT KEEPminArea;      /* 'PFn' minimum facet area to keep */
  realT MAXcoplanar;      /* 'Un' max distance below a facet to be coplanar*/
  int   MAXwide;          /* 'QWn' max ratio for wide facet, otherwise error unless Q12-allow-wide */
  boolT MERGEexact;       /* true 'Qx' if exact merges (concave, degen, dupridge, flipped)
                             tested by qh_checkzero and qh_test_*_merge */
  boolT MERGEindependent; /* true if merging independent sets of coplanar facets. 'Q2' disables */
  boolT MERGING;          /* true if exact-, pre- or post-merging, with angle and centrum tests */
  realT   premerge_centrum;  /*   'C-n' centrum_radius when pre merging.  Default is round-off */
  realT   postmerge_centrum; /*   'Cn' centrum_radius when post merging.  Default is round-off */
  boolT MERGEpinched;     /* true 'Q14' if merging pinched vertices due to dupridge */
  boolT MERGEvertices;    /* true if merging redundant vertices, 'Q3' disables or qh.hull_dim > qh_DIMmergeVertex */
  realT MINvisible;       /* 'Vn' min. distance for a facet to be visible */
  boolT NOnarrow;         /* true 'Q10' if no special processing for narrow distributions */
  boolT NOnearinside;     /* true 'Q8' if ignore near-inside points when partitioning, qh_check_points may fail */
  boolT NOpremerge;       /* true 'Q0' if no defaults for C-0 or Qx */
  boolT ONLYgood;         /* true 'Qg' if process points with good visible or horizon facets */
  boolT ONLYmax;          /* true 'Qm' if only process points that increase max_outside */
  boolT PICKfurthest;     /* true 'Q9' if process furthest of furthest points*/
  boolT POSTmerge;        /* true if merging after buildhull ('Cn' or 'An') */
  boolT PREmerge;         /* true if merging during buildhull ('C-n' or 'A-n') */
                        /* NOTE: some of these names are similar to qh_PRINT names */
  boolT PRINTcentrums;    /* true 'Gc' if printing centrums */
  boolT PRINTcoplanar;    /* true 'Gp' if printing coplanar points */
  int   PRINTdim;         /* print dimension for Geomview output */
  boolT PRINTdots;        /* true 'Ga' if printing all points as dots */
  boolT PRINTgood;        /* true 'Pg' if printing good facets
                             PGood set if 'd', 'PAn', 'PFn', 'PMn', 'QGn', 'QG-n', 'QVn', or 'QV-n' */
  boolT PRINTinner;       /* true 'Gi' if printing inner planes */
  boolT PRINTneighbors;   /* true 'PG' if printing neighbors of good facets */
  boolT PRINTnoplanes;    /* true 'Gn' if printing no planes */
  boolT PRINToptions1st;  /* true 'FO' if printing options to stderr */
  boolT PRINTouter;       /* true 'Go' if printing outer planes */
  boolT PRINTprecision;   /* false 'Pp' if not reporting precision problems */
  qh_PRINT PRINTout[qh_PRINTEND]; /* list of output formats to print */
  boolT PRINTridges;      /* true 'Gr' if print ridges */
  boolT PRINTspheres;     /* true 'Gv' if print vertices as spheres */
  boolT PRINTstatistics;  /* true 'Ts' if printing statistics to stderr */
  boolT PRINTsummary;     /* true 's' if printing summary to stderr */
  boolT PRINTtransparent; /* true 'Gt' if print transparent outer ridges */
  boolT PROJECTdelaunay;  /* true if DELAUNAY, no readpoints() and
                             need projectinput() for Delaunay in qh_init_B */
  int   PROJECTinput;     /* number of projected dimensions 'bn:0Bn:0' */
  boolT RANDOMdist;       /* true 'Rn' if randomly change distplane and setfacetplane */
  realT RANDOMfactor;     /*    maximum random perturbation */
  realT RANDOMa;          /*    qh_randomfactor is randr * RANDOMa + RANDOMb */
  realT RANDOMb;
  boolT RANDOMoutside;    /* true 'Qr' if select a random outside point */
  int   REPORTfreq;       /* 'TFn' buildtracing reports every n facets */
  int   REPORTfreq2;      /* tracemerging reports every REPORTfreq/2 facets */
  int   RERUN;            /* 'TRn' rerun qhull n times (qh.build_cnt) */
  int   ROTATErandom;     /* 'QRn' n<-1 random seed, n==-1 time is seed, n==0 random rotation by time, n>0 rotate input */
  boolT SCALEinput;       /* true 'Qbk' if scaling input */
  boolT SCALElast;        /* true 'Qbb' if scale last coord to max prev coord */
  boolT SETroundoff;      /* true 'En' if qh.DISTround is predefined */
  boolT SKIPcheckmax;     /* true 'Q5' if skip qh_check_maxout, qh_check_points may fail */
  boolT SKIPconvex;       /* true 'Q6' if skip convexity testing during pre-merge */
  boolT SPLITthresholds;  /* true 'Pd/PD' if upper_/lower_threshold defines a region
                               else qh.GOODthresholds
                               set if qh.DELAUNAY (qh_initbuild)
                               used only for printing (!for qh.ONLYgood) */
  int   STOPadd;          /* 'TAn' 1+n for stop after adding n vertices */
  int   STOPcone;         /* 'TCn' 1+n for stopping after cone for point n */
                          /*       also used by qh_build_withresart for err exit*/
  int   STOPpoint;        /* 'TVn' 'TV-n' 1+n for stopping after/before(-)
                                        adding point n */
  int   TESTpoints;       /* 'QTn' num of test points after qh.num_points.  Test points always coplanar. */
  boolT TESTvneighbors;   /*  true 'Qv' if test vertex neighbors at end */
  int   TRACElevel;       /* 'Tn' conditional IStracing level */
  int   TRACElastrun;     /*  qh.TRACElevel applies to last qh.RERUN */
  int   TRACEpoint;       /* 'TPn' start tracing when point n is a vertex, use qh_IDunknown (-1) after qh_buildhull and qh_postmerge */
  realT TRACEdist;        /* 'TWn' start tracing when merge distance too big */
  int   TRACEmerge;       /* 'TMn' start tracing before this merge */
  boolT TRIangulate;      /* true 'Qt' if triangulate non-simplicial facets */
  boolT TRInormals;       /* true 'Q11' if triangulate duplicates ->normal and ->center (sets Qt) */
  boolT UPPERdelaunay;    /* true 'Qu' if computing furthest-site Delaunay */
  boolT USEstdout;        /* true 'Tz' if using stdout instead of stderr */
  boolT VERIFYoutput;     /* true 'Tv' if verify output at end of qhull */
  boolT VIRTUALmemory;    /* true 'Q7' if depth-first processing in buildhull */
  boolT VORONOI;          /* true 'v' if computing Voronoi diagram, also sets qh.DELAUNAY */

  /*--------input constants ---------*/
  realT AREAfactor;       /* 1/(hull_dim-1)! for converting det's to area */
  boolT DOcheckmax;       /* true if calling qh_check_maxout (!qh.SKIPcheckmax && qh.MERGING) */
  char  *feasible_string;  /* feasible point 'Hn,n,n' for halfspace intersection */
  coordT *feasible_point;  /*    as coordinates, both malloc'd */
  boolT GETarea;          /* true 'Fa', 'FA', 'FS', 'PAn', 'PFn' if compute facet area/Voronoi volume in io_r.c */
  boolT KEEPnearinside;   /* true if near-inside points in coplanarset */
  int   hull_dim;         /* dimension of hull, set by initbuffers */
  int   input_dim;        /* dimension of input, set by initbuffers */
  int   num_points;       /* number of input points */
  pointT *first_point;    /* array of input points, see POINTSmalloc */
  boolT POINTSmalloc;     /*   true if qh.first_point/num_points allocated */
  pointT *input_points;   /* copy of original qh.first_point for input points for qh_joggleinput */
  boolT input_malloc;     /* true if qh.input_points malloc'd */
  char  qhull_command[256];/* command line that invoked this program */
  int   qhull_commandsiz2; /*    size of qhull_command at qh_clear_outputflags */
  char  rbox_command[256]; /* command line that produced the input points */
  char  qhull_options[512];/* descriptive list of options */
  int   qhull_optionlen;  /*    length of last line */
  int   qhull_optionsiz;  /*    size of qhull_options at qh_build_withrestart */
  int   qhull_optionsiz2; /*    size of qhull_options at qh_clear_outputflags */
  int   run_id;           /* non-zero, random identifier for this instance of qhull */
  boolT VERTEXneighbors;  /* true if maintaining vertex neighbors */
  boolT ZEROcentrum;      /* true if 'C-0' or 'C-0 Qx' and not post-merging or 'A-n'.  Sets ZEROall_ok */
  realT *upper_threshold; /* don't print if facet->normal[k]>=upper_threshold[k]
                             must set either GOODthreshold or SPLITthreshold
                             if qh.DELAUNAY, default is 0.0 for upper envelope (qh_initbuild) */
  realT *lower_threshold; /* don't print if facet->normal[k] <=lower_threshold[k] */
  realT *upper_bound;     /* scale point[k] to new upper bound */
  realT *lower_bound;     /* scale point[k] to new lower bound
                             project if both upper_ and lower_bound == 0 */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-prec">-</a>

  qh precision constants
    precision constants for Qhull

  notes:
    qh_detroundoff [geom2_r.c] computes the maximum roundoff error for distance
    and other computations.  It also sets default values for the
    qh constants above.
*/
  realT ANGLEround;       /* max round off error for angles */
  realT centrum_radius;   /* max centrum radius for convexity ('Cn' + 2*qh.DISTround) */
  realT cos_max;          /* max cosine for convexity (roundoff added) */
  realT DISTround;        /* max round off error for distances, qh.SETroundoff ('En') overrides qh_distround */
  realT MAXabs_coord;     /* max absolute coordinate */
  realT MAXlastcoord;     /* max last coordinate for qh_scalelast */
  realT MAXoutside;       /* max target for qh.max_outside/f.maxoutside, base for qh_RATIO...
                             recomputed at qh_addpoint, unrelated to qh_MAXoutside */
  realT MAXsumcoord;      /* max sum of coordinates */
  realT MAXwidth;         /* max rectilinear width of point coordinates */
  realT MINdenom_1;       /* min. abs. value for 1/x */
  realT MINdenom;         /*    use divzero if denominator < MINdenom */
  realT MINdenom_1_2;     /* min. abs. val for 1/x that allows normalization */
  realT MINdenom_2;       /*    use divzero if denominator < MINdenom_2 */
  realT MINlastcoord;     /* min. last coordinate for qh_scalelast */
  realT *NEARzero;        /* hull_dim array for near zero in gausselim */
  realT NEARinside;       /* keep points for qh_check_maxout if close to facet */
  realT ONEmerge;         /* max distance for merging simplicial facets */
  realT outside_err;      /* application's epsilon for coplanar points
                             qh_check_bestdist() qh_check_points() reports error if point outside */
  realT WIDEfacet;        /* size of wide facet for skipping ridge in
                             area computation and locking centrum */
  boolT NARROWhull;       /* set in qh_initialhull if angle < qh_MAXnarrow */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-codetern">-</a>

  qh internal constants
    internal constants for Qhull
*/
  char qhull[sizeof("qhull")]; /* "qhull" for checking ownership while debugging */
  jmp_buf errexit;        /* exit label for qh_errexit, defined by setjmp() and NOerrexit */
  char    jmpXtra[40];    /* extra bytes in case jmp_buf is defined wrong by compiler */
  jmp_buf restartexit;    /* restart label for qh_errexit, defined by setjmp() and ALLOWrestart */
  char    jmpXtra2[40];   /* extra bytes in case jmp_buf is defined wrong by compiler*/
  FILE *  fin;            /* pointer to input file, init by qh_initqhull_start2 */
  FILE *  fout;           /* pointer to output file */
  FILE *  ferr;           /* pointer to error file */
  pointT *interior_point; /* center point of the initial simplex*/
  int     normal_size;    /* size in bytes for facet normals and point coords */
  int     center_size;    /* size in bytes for Voronoi centers */
  int     TEMPsize;       /* size for small, temporary sets (in quick mem) */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-lists">-</a>

  qh facet and vertex lists
    defines lists of facets, new facets, visible facets, vertices, and
    new vertices.  Includes counts, next ids, and trace ids.
  see:
    qh_resetlists()
*/
  facetT *facet_list;     /* first facet */
  facetT *facet_tail;     /* end of facet_list (dummy facet with id 0 and next==NULL) */
  facetT *facet_next;     /* next facet for buildhull()
                             previous facets do not have outside sets
                             NARROWhull: previous facets may have coplanar outside sets for qh_outcoplanar */
  facetT *newfacet_list;  /* list of new facets to end of facet_list
                             qh_postmerge sets newfacet_list to facet_list */
  facetT *visible_list;   /* list of visible facets preceding newfacet_list,
                             end of visible list if !facet->visible, same as newfacet_list
                             qh_findhorizon sets visible_list at end of facet_list
                             qh_willdelete prepends to visible_list
                             qh_triangulate appends mirror facets to visible_list at end of facet_list
                             qh_postmerge sets visible_list to facet_list
                             qh_deletevisible deletes the visible facets */
  int       num_visible;  /* current number of visible facets */
  unsigned int tracefacet_id; /* set at init, then can print whenever */
  facetT  *tracefacet;    /*   set in newfacet/mergefacet, undone in delfacet and qh_errexit */
  unsigned int traceridge_id; /* set at init, then can print whenever */
  ridgeT  *traceridge;    /*   set in newridge, undone in delridge, errexit, errexit2, makenew_nonsimplicial, mergecycle_ridges */
  unsigned int tracevertex_id; /* set at buildtracing, can print whenever */
  vertexT *tracevertex;   /*   set in newvertex, undone in delvertex and qh_errexit */
  vertexT *vertex_list;   /* list of all vertices, to vertex_tail */
  vertexT *vertex_tail;   /*      end of vertex_list (dummy vertex with ID 0, next NULL) */
  vertexT *newvertex_list; /* list of vertices in newfacet_list, to vertex_tail
                             all vertices have 'newfacet' set */
  int   num_facets;       /* number of facets in facet_list
                             includes visible faces (num_visible) */
  int   num_vertices;     /* number of vertices in facet_list */
  int   num_outside;      /* number of points in outsidesets (for tracing and RANDOMoutside)
                               includes coplanar outsideset points for NARROWhull/qh_outcoplanar() */
  int   num_good;         /* number of good facets (after qh_findgood_all or qh_markkeep) */
  unsigned int facet_id;  /* ID of next, new facet from newfacet() */
  unsigned int ridge_id;  /* ID of next, new ridge from newridge() */
  unsigned int vertex_id; /* ID of next, new vertex from newvertex() */
  unsigned int first_newfacet; /* ID of first_newfacet for qh_buildcone, or 0 if none */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-var">-</a>

  qh global variables
    defines minimum and maximum distances, next visit ids, several flags,
    and other global variables.
    initialize in qh_initbuild or qh_maxmin if used in qh_buildhull
*/
  unsigned long hulltime; /* ignore time to set up input and randomize */
                          /*   use 'unsigned long' to avoid wrap-around errors */
  boolT ALLOWrestart;     /* true if qh_joggle_restart can use qh.restartexit */
  int   build_cnt;        /* number of calls to qh_initbuild */
  qh_CENTER CENTERtype;   /* current type of facet->center, qh_CENTER */
  int   furthest_id;      /* pointid of furthest point, for tracing */
  int   last_errcode;     /* last errcode from qh_fprintf, reset in qh_build_withrestart */
  facetT *GOODclosest;    /* closest facet to GOODthreshold in qh_findgood */
  pointT *coplanar_apex;  /* last apex declared a coplanar point by qh_getpinchedmerges, prevents infinite loop */
  boolT hasAreaVolume;    /* true if totarea, totvol was defined by qh_getarea */
  boolT hasTriangulation; /* true if triangulation created by qh_triangulate */
  boolT isRenameVertex;   /* true during qh_merge_pinchedvertices, disables duplicate ridge vertices in qh_checkfacet */
  realT JOGGLEmax;        /* set 'QJn' if randomly joggle input. 'QJ'/'QJ0.0' sets default (qh_detjoggle) */
  boolT maxoutdone;       /* set qh_check_maxout(), cleared by qh_addpoint() */
  realT max_outside;      /* maximum distance from a point to a facet,
                               before roundoff, not simplicial vertices
                               actual outer plane is +DISTround and
                               computed outer plane is +2*DISTround */
  realT max_vertex;       /* maximum distance (>0) from vertex to a facet,
                               before roundoff, due to a merge */
  realT min_vertex;       /* minimum distance (<0) from vertex to a facet,
                               before roundoff, due to a merge
                               if qh.JOGGLEmax, qh_makenewplanes sets it
                               recomputed if qh.DOcheckmax, default -qh.DISTround */
  boolT NEWfacets;        /* true while visible facets invalid due to new or merge
                              from qh_makecone/qh_attachnewfacets to qh_resetlists */
  boolT NEWtentative;     /* true while new facets are tentative due to !qh.IGNOREpinched or qh.ONLYgood
                              from qh_makecone to qh_attachnewfacets */
  boolT findbestnew;      /* true if partitioning calls qh_findbestnew */
  boolT findbest_notsharp; /* true if new facets are at least 90 degrees */
  boolT NOerrexit;        /* true if qh.errexit is not available, cleared after setjmp.  See qh.ERREXITcalled */
  realT PRINTcradius;     /* radius for printing centrums */
  realT PRINTradius;      /* radius for printing vertex spheres and points */
  boolT POSTmerging;      /* true when post merging */
  int   printoutvar;      /* temporary variable for qh_printbegin, etc. */
  int   printoutnum;      /* number of facets printed */
  unsigned int repart_facetid; /* previous facetid to prevent recursive qh_partitioncoplanar+qh_partitionpoint */
  int   retry_addpoint;   /* number of retries of qh_addpoint due to merging pinched vertices */
  boolT QHULLfinished;    /* True after qhull() is finished */
  realT totarea;          /* 'FA': total facet area computed by qh_getarea, hasAreaVolume */
  realT totvol;           /* 'FA': total volume computed by qh_getarea, hasAreaVolume */
  unsigned int visit_id;  /* unique ID for searching neighborhoods, */
  unsigned int vertex_visit; /* unique ID for searching vertices, reset with qh_buildtracing */
  boolT WAScoplanar;      /* True if qh_partitioncoplanar (qh_check_maxout) */
  boolT ZEROall_ok;       /* True if qh_checkzero always succeeds */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-set">-</a>

  qh global sets
    defines sets for merging, initial simplex, hashing, extra input points,
    and deleted vertices
*/
  setT *facet_mergeset;   /* temporary set of merges to be done */
  setT *degen_mergeset;   /* temporary set of degenerate and redundant merges */
  setT *vertex_mergeset;  /* temporary set of vertex merges */
  setT *hash_table;       /* hash table for matching ridges in qh_matchfacets
                             size is setsize() */
  setT *other_points;     /* additional points */
  setT *del_vertices;     /* vertices to partition and delete with visible
                             facets.  v.deleted is set for checkfacet */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-buf">-</a>

  qh global buffers
    defines buffers for maxtrix operations, input, and error messages
*/
  coordT *gm_matrix;      /* (dim+1)Xdim matrix for geom_r.c */
  coordT **gm_row;        /* array of gm_matrix rows */
  char* line;             /* malloc'd input line of maxline+1 chars */
  int maxline;
  coordT *half_space;     /* malloc'd input array for halfspace (qh.normal_size+coordT) */
  coordT *temp_malloc;    /* malloc'd input array for points */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-static">-</a>

  qh static variables
    defines static variables for individual functions

  notes:
    do not use 'static' within a function.  Multiple instances of qhull
    may exist.

    do not assume zero initialization, 'QPn' may cause a restart
*/
  boolT ERREXITcalled;    /* true during qh_errexit (prevents duplicate calls).  see qh.NOerrexit */
  boolT firstcentrum;     /* for qh_printcentrum */
  boolT old_randomdist;   /* save RANDOMdist flag during io, tracing, or statistics */
  setT *coplanarfacetset; /* set of coplanar facets for searching qh_findbesthorizon() */
  realT last_low;         /* qh_scalelast parameters for qh_setdelaunay */
  realT last_high;
  realT last_newhigh;
  realT lastcpu;          /* for qh_buildtracing */
  int   lastfacets;       /*   last qh.num_facets */
  int   lastmerges;       /*   last zzval_(Ztotmerge) */ 
  int   lastplanes;       /*   last zzval_(Zsetplane) */ 
  int   lastdist;         /*   last zzval_(Zdistplane) */ 
  unsigned int lastreport; /*  last qh.facet_id */
  int mergereport;        /* for qh_tracemerging */
  setT *old_tempstack;    /* for saving qh->qhmem.tempstack in save_qhull */
  int   ridgeoutnum;      /* number of ridges for 4OFF output (qh_printbegin,etc) */

/*-<a                             href="qh-globa_r.htm#TOC"
  >--------------------------------</a><a name="qh-const">-</a>

  qh memory management, rbox globals, and statistics

  Replaces global data structures defined for libqhull
*/
  int     last_random;    /* Last random number from qh_rand (random_r.c) */
  jmp_buf rbox_errexit;   /* errexit from rboxlib_r.c, defined by qh_rboxpoints() only */
  char    jmpXtra3[40];   /* extra bytes in case jmp_buf is defined wrong by compiler */
  int     rbox_isinteger;
  double  rbox_out_offset;
  void *  cpp_object;     /* C++ pointer.  Currently used by RboxPoints.qh_fprintf_rbox */

  /* Last, otherwise zero'd by qh_initqhull_start2 (global_r.c */
  qhmemT  qhmem;          /* Qhull managed memory (mem_r.h) */
  /* After qhmem because its size depends on the number of statistics */
  qhstatT qhstat;         /* Qhull statistics (stat_r.h) */
};

/*=========== -macros- =========================*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="otherfacet_">-</a>

  otherfacet_(ridge, facet)
    return neighboring facet for a ridge in facet
*/
#define otherfacet_(ridge, facet) \
                        (((ridge)->top == (facet)) ? (ridge)->bottom : (ridge)->top)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="getid_">-</a>

  getid_(p)
    return int ID for facet, ridge, or vertex
    return qh_IDunknown(-1) if NULL
    return 0 if facet_tail or vertex_tail
*/
#define getid_(p)       ((p) ? (int)((p)->id) : qh_IDunknown)

/*============== FORALL macros ===================*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLfacets">-</a>

  FORALLfacets { ... }
    assign 'facet' to each facet in qh.facet_list

  notes:
    uses 'facetT *facet;'
    assumes last facet is a sentinel
    assumes qh defined

  see:
    FORALLfacet_( facetlist )
*/
#define FORALLfacets for (facet=qh->facet_list;facet && facet->next;facet=facet->next)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLpoints">-</a>

  FORALLpoints { ... }
    assign 'point' to each point in qh.first_point, qh.num_points

  notes:
    assumes qh defined

  declare:
    coordT *point, *pointtemp;
*/
#define FORALLpoints FORALLpoint_(qh, qh->first_point, qh->num_points)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLpoint_">-</a>

  FORALLpoint_(qh, points, num) { ... }
    assign 'point' to each point in points array of num points

  declare:
    coordT *point, *pointtemp;
*/
#define FORALLpoint_(qh, points, num) for (point=(points), \
      pointtemp= (points)+qh->hull_dim*(num); point < pointtemp; point += qh->hull_dim)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLvertices">-</a>

  FORALLvertices { ... }
    assign 'vertex' to each vertex in qh.vertex_list

  declare:
    vertexT *vertex;

  notes:
    assumes qh.vertex_list terminated by NULL or a sentinel (v.next==NULL)
    assumes qh defined
*/
#define FORALLvertices for (vertex=qh->vertex_list;vertex && vertex->next;vertex= vertex->next)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHfacet_">-</a>

  FOREACHfacet_( facets ) { ... }
    assign 'facet' to each facet in facets

  declare:
    facetT *facet, **facetp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHfacet_(facets)    FOREACHsetelement_(facetT, facets, facet)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighbor_">-</a>

  FOREACHneighbor_( facet ) { ... }
    assign 'neighbor' to each neighbor in facet->neighbors

  FOREACHneighbor_( vertex ) { ... }
    assign 'neighbor' to each neighbor in vertex->neighbors

  declare:
    facetT *neighbor, **neighborp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHneighbor_(facet)  FOREACHsetelement_(facetT, facet->neighbors, neighbor)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHpoint_">-</a>

  FOREACHpoint_( points ) { ... }
    assign 'point' to each point in points set

  declare:
    pointT *point, **pointp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHpoint_(points)    FOREACHsetelement_(pointT, points, point)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHridge_">-</a>

  FOREACHridge_( ridges ) { ... }
    assign 'ridge' to each ridge in ridges set

  declare:
    ridgeT *ridge, **ridgep;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHridge_(ridges)    FOREACHsetelement_(ridgeT, ridges, ridge)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertex_">-</a>

  FOREACHvertex_( vertices ) { ... }
    assign 'vertex' to each vertex in vertices set

  declare:
    vertexT *vertex, **vertexp;

  notes:
    assumes set is not modified

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertex_(vertices) FOREACHsetelement_(vertexT, vertices,vertex)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHfacet_i_">-</a>

  FOREACHfacet_i_(qh, facets ) { ... }
    assign 'facet' and 'facet_i' for each facet in facets set

  declare:
    facetT *facet;
    int     facet_n, facet_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHfacet_i_(qh, facets)    FOREACHsetelement_i_(qh, facetT, facets, facet)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighbor_i_">-</a>

  FOREACHneighbor_i_(qh, facet ) { ... }
    assign 'neighbor' and 'neighbor_i' for each neighbor in facet->neighbors

  declare:
    facetT *neighbor;
    int     neighbor_n, neighbor_i;

  notes:
    see <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
    for facet neighbors of vertex, need to define a new macro
*/
#define FOREACHneighbor_i_(qh, facet)  FOREACHsetelement_i_(qh, facetT, facet->neighbors, neighbor)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHpoint_i_">-</a>

  FOREACHpoint_i_(qh, points ) { ... }
    assign 'point' and 'point_i' for each point in points set

  declare:
    pointT *point;
    int     point_n, point_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHpoint_i_(qh, points)    FOREACHsetelement_i_(qh, pointT, points, point)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHridge_i_">-</a>

  FOREACHridge_i_(qh, ridges ) { ... }
    assign 'ridge' and 'ridge_i' for each ridge in ridges set

  declare:
    ridgeT *ridge;
    int     ridge_n, ridge_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHridge_i_(qh, ridges)    FOREACHsetelement_i_(qh, ridgeT, ridges, ridge)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertex_i_">-</a>

  FOREACHvertex_i_(qh, vertices ) { ... }
    assign 'vertex' and 'vertex_i' for each vertex in vertices set

  declare:
    vertexT *vertex;
    int     vertex_n, vertex_i;

  see:
    <a href="qset_r.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHvertex_i_(qh, vertices) FOREACHsetelement_i_(qh, vertexT, vertices, vertex)

#ifdef __cplusplus
extern "C" {
#endif

/********* -libqhull_r.c prototypes (duplicated from qhull_ra.h) **********************/

void    qh_qhull(qhT *qh);
boolT   qh_addpoint(qhT *qh, pointT *furthest, facetT *facet, boolT checkdist);
void    qh_errexit2(qhT *qh, int exitcode, facetT *facet, facetT *otherfacet);
void    qh_printsummary(qhT *qh, FILE *fp);

/********* -user_r.c prototypes (alphabetical) **********************/

void    qh_errexit(qhT *qh, int exitcode, facetT *facet, ridgeT *ridge);
void    qh_errprint(qhT *qh, const char* string, facetT *atfacet, facetT *otherfacet, ridgeT *atridge, vertexT *atvertex);
int     qh_new_qhull(qhT *qh, int dim, int numpoints, coordT *points, boolT ismalloc,
                char *qhull_cmd, FILE *outfile, FILE *errfile);
void    qh_printfacetlist(qhT *qh, facetT *facetlist, setT *facets, boolT printall);
void    qh_printhelp_degenerate(qhT *qh, FILE *fp);
void    qh_printhelp_internal(qhT *qh, FILE *fp);
void    qh_printhelp_narrowhull(qhT *qh, FILE *fp, realT minangle);
void    qh_printhelp_singular(qhT *qh, FILE *fp);
void    qh_printhelp_topology(qhT *qh, FILE *fp);
void    qh_printhelp_wide(qhT *qh, FILE *fp);
void    qh_user_memsizes(qhT *qh);

/********* -usermem_r.c prototypes (alphabetical) **********************/
void    qh_exit(int exitcode);
void    qh_fprintf_stderr(int msgcode, const char *fmt, ... );
void    qh_free(void *mem);
void   *qh_malloc(size_t size);

/********* -userprintf_r.c and userprintf_rbox_r.c prototypes **********************/
void    qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );
void    qh_fprintf_rbox(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );

/***** -geom_r.c/geom2_r.c/random_r.c prototypes (duplicated from geom_r.h, random_r.h) ****************/

facetT *qh_findbest(qhT *qh, pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT newfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart);
facetT *qh_findbestnew(qhT *qh, pointT *point, facetT *startfacet,
                     realT *dist, boolT bestoutside, boolT *isoutside, int *numpart);
boolT   qh_gram_schmidt(qhT *qh, int dim, realT **rows);
void    qh_outerinner(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane);
void    qh_printsummary(qhT *qh, FILE *fp);
void    qh_projectinput(qhT *qh);
void    qh_randommatrix(qhT *qh, realT *buffer, int dim, realT **row);
void    qh_rotateinput(qhT *qh, realT **rows);
void    qh_scaleinput(qhT *qh);
void    qh_setdelaunay(qhT *qh, int dim, int count, pointT *points);
coordT  *qh_sethalfspace_all(qhT *qh, int dim, int count, coordT *halfspaces, pointT *feasible);

/***** -global_r.c prototypes (alphabetical) ***********************/

unsigned long qh_clock(qhT *qh);
void    qh_checkflags(qhT *qh, char *command, char *hiddenflags);
void    qh_clear_outputflags(qhT *qh);
void    qh_freebuffers(qhT *qh);
void    qh_freeqhull(qhT *qh, boolT allmem);
void    qh_init_A(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile, int argc, char *argv[]);
void    qh_init_B(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);
void    qh_init_qhull_command(qhT *qh, int argc, char *argv[]);
void    qh_initbuffers(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);
void    qh_initflags(qhT *qh, char *command);
void    qh_initqhull_buffers(qhT *qh);
void    qh_initqhull_globals(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);
void    qh_initqhull_mem(qhT *qh);
void    qh_initqhull_outputflags(qhT *qh);
void    qh_initqhull_start(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile);
void    qh_initqhull_start2(qhT *qh, FILE *infile, FILE *outfile, FILE *errfile);
void    qh_initthresholds(qhT *qh, char *command);
void    qh_lib_check(int qhullLibraryType, int qhTsize, int vertexTsize, int ridgeTsize, int facetTsize, int setTsize, int qhmemTsize);
void    qh_option(qhT *qh, const char *option, int *i, realT *r);
void    qh_zero(qhT *qh, FILE *errfile);

/***** -io_r.c prototypes (duplicated from io_r.h) ***********************/

void    qh_dfacet(qhT *qh, unsigned int id);
void    qh_dvertex(qhT *qh, unsigned int id);
void    qh_printneighborhood(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall);
void    qh_produce_output(qhT *qh);
coordT *qh_readpoints(qhT *qh, int *numpoints, int *dimension, boolT *ismalloc);


/********* -mem_r.c prototypes (duplicated from mem_r.h) **********************/

void qh_meminit(qhT *qh, FILE *ferr);
void qh_memfreeshort(qhT *qh, int *curlong, int *totlong);

/********* -poly_r.c/poly2_r.c prototypes (duplicated from poly_r.h) **********************/

void    qh_check_output(qhT *qh);
void    qh_check_points(qhT *qh);
setT   *qh_facetvertices(qhT *qh, facetT *facetlist, setT *facets, boolT allfacets);
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside,
           realT *bestdist, boolT *isoutside);
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp);
pointT *qh_point(qhT *qh, int id);
setT   *qh_pointfacet(qhT *qh /* qh.facet_list */);
int     qh_pointid(qhT *qh, pointT *point);
setT   *qh_pointvertex(qhT *qh /* qh.facet_list */);
void    qh_setvoronoi_all(qhT *qh);
void    qh_triangulate(qhT *qh /* qh.facet_list */);

/********* -rboxlib_r.c prototypes **********************/
int     qh_rboxpoints(qhT *qh, char* rbox_command);
void    qh_errexit_rbox(qhT *qh, int exitcode);

/********* -stat_r.c prototypes (duplicated from stat_r.h) **********************/

void    qh_collectstatistics(qhT *qh);
void    qh_printallstatistics(qhT *qh, FILE *fp, const char *string);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFlibqhull */

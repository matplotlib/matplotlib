/*<html><pre>  -<a                             href="qh-qhull.htm"
  >-------------------------------</a><a name="TOP">-</a>

   libqhull.h
   user-level header file for using qhull.a library

   see qh-qhull.htm, qhull_a.h

   Copyright (c) 1993-2015 The Geometry Center.
   $Id: //main/2015/qhull/src/libqhull/libqhull.h#7 $$Change: 2066 $
   $DateTime: 2016/01/18 19:29:17 $$Author: bbarber $

   NOTE: access to qh_qh is via the 'qh' macro.  This allows
   qh_qh to be either a pointer or a structure.  An example
   of using qh is "qh.DROPdim" which accesses the DROPdim
   field of qh_qh.  Similarly, access to qh_qhstat is via
   the 'qhstat' macro.

   includes function prototypes for libqhull.c, geom.c, global.c, io.c, user.c

   use mem.h for mem.c
   use qset.h for qset.c

   see unix.c for an example of using libqhull.h

   recompile qhull if you change this file
*/

#ifndef qhDEFlibqhull
#define qhDEFlibqhull 1

/*=========================== -included files ==============*/

/* user_r.h first for QHULL_CRTDBG */
#include "user.h"      /* user definable constants (e.g., qh_QHpointer) */

#include "mem.h"   /* Needed qhT in libqhull_r.h.  Here for compatibility */
#include "qset.h"   /* Needed for QHULL_LIB_CHECK */
/* include stat_r.h after defining boolT.  Needed for qhT in libqhull_r.h.  Here for compatibility and statT */

#include <setjmp.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

#if __MWERKS__ && __POWERPC__
#include  <SIOUX.h>
#include  <Files.h>
#include        <Desk.h>
#endif

#ifndef __STDC__
#ifndef __cplusplus
#if     !_MSC_VER
#error  Neither __STDC__ nor __cplusplus is defined.  Please use strict ANSI C or C++ to compile
#error  Qhull.  You may need to turn off compiler extensions in your project configuration.  If
#error  your compiler is a standard C compiler, you can delete this warning from libqhull.h
#endif
#endif
#endif

/*============ constants and basic types ====================*/

extern const char qh_version[]; /* defined in global.c */
extern const char qh_version2[]; /* defined in global.c */

/*-<a                             href="qh-geom.htm#TOC"
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

/*-<a                             href="qh-geom.htm#TOC"
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
    qh_IDnone = -3, qh_IDinterior = -2, qh_IDunknown = -1
}
qh_pointT;

/*-<a                             href="qh-qhull.htm#TOC"
  >--------------------------------</a><a name="flagT">-</a>

  flagT
    Boolean flag as a bit
*/
#define flagT unsigned int

/*-<a                             href="qh-qhull.htm#TOC"
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

#include "stat.h"  /* after define of boolT */

/*-<a                             href="qh-qhull.htm#TOC"
  >--------------------------------</a><a name="CENTERtype">-</a>

  qh_CENTER
    to distinguish facet->center
*/
typedef enum
{
    qh_ASnone = 0,   /* If not MERGING and not VORONOI */
    qh_ASvoronoi,    /* Set by qh_clearcenters on qh_prepare_output, or if not MERGING and VORONOI */
    qh_AScentrum     /* If MERGING (assumed during merging) */
}
qh_CENTER;

/*-<a                             href="qh-qhull.htm#TOC"
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

/*-<a                             href="qh-qhull.htm#TOC"
  >--------------------------------</a><a name="qh_ALL">-</a>

  qh_ALL
    argument flag for selecting everything
*/
#define qh_ALL      True
#define qh_NOupper  True     /* argument for qh_findbest */
#define qh_IScheckmax  True     /* argument for qh_findbesthorizon */
#define qh_ISnewfacets  True     /* argument for qh_findbest */
#define qh_RESETvisible  True     /* argument for qh_resetlists */

/*-<a                             href="qh-qhull.htm#TOC"
  >--------------------------------</a><a name="qh_ERR">-</a>

  qh_ERR
    Qhull exit codes, for indicating errors
    See: MSG_ERROR and MSG_WARNING [user.h]
*/
#define qh_ERRnone  0    /* no error occurred during qhull */
#define qh_ERRinput 1    /* input inconsistency */
#define qh_ERRsingular 2 /* singular input data */
#define qh_ERRprec  3    /* precision error */
#define qh_ERRmem   4    /* insufficient memory, matches mem.h */
#define qh_ERRqhull 5    /* internal error detected, matches mem.h */

/*-<a                             href="qh-qhull.htm#TOC"
>--------------------------------</a><a name="qh_FILEstderr">-</a>

qh_FILEstderr
Fake stderr to distinguish error output from normal output
For C++ interface.  Must redefine qh_fprintf_qhull
*/
#define qh_FILEstderr ((FILE*)1)

/* ============ -structures- ====================
   each of the following structures is defined by a typedef
   all realT and coordT fields occur at the beginning of a structure
        (otherwise space may be wasted due to alignment)
   define all flags together and pack into 32-bit number
   DEFsetT is likewise defined in
   mem.h and qset.h
*/

typedef struct vertexT vertexT;
typedef struct ridgeT ridgeT;
typedef struct facetT facetT;
#ifndef DEFsetT
#define DEFsetT 1
typedef struct setT setT;          /* defined in qset.h */
#endif

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="facetT">-</a>

  facetT
    defines a facet

  notes:
   qhull() generates the hull as a list of facets.

  topological information:
    f.previous,next     doubly-linked list of facets
    f.vertices          set of vertices
    f.ridges            set of ridges
    f.neighbors         set of neighbors
    f.toporient         True if facet has top-orientation (else bottom)

  geometric information:
    f.offset,normal     hyperplane equation
    f.maxoutside        offset to outer plane -- all points inside
    f.center            centrum for testing convexity
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
struct facetT {
#if !qh_COMPUTEfurthest
  coordT   furthestdist;/* distance to furthest point of outsideset */
#endif
#if qh_MAXoutside
  coordT   maxoutside;  /* max computed distance of point to facet
                        Before QHULLfinished this is an approximation
                        since maxdist not always set for mergefacet
                        Actual outer plane is +DISTround and
                        computed outer plane is +2*DISTround */
#endif
  coordT   offset;      /* exact offset of hyperplane from origin */
  coordT  *normal;      /* normal of hyperplane, hull_dim coefficients */
                        /*   if tricoplanar, shared with a neighbor */
  union {               /* in order of testing */
   realT   area;        /* area of facet, only in io.c if  ->isarea */
   facetT *replace;     /*  replacement facet if ->visible and NEWfacets
                             is NULL only if qh_mergedegen_redundant or interior */
   facetT *samecycle;   /*  cycle of facets from the same visible/horizon intersection,
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
  facetT  *previous;    /* previous facet in the facet_list */
  facetT  *next;        /* next facet in the facet_list */
  setT    *vertices;    /* vertices for this facet, inverse sorted by ID
                           if simplicial, 1st vertex was apex/furthest */
  setT    *ridges;      /* explicit ridges for nonsimplicial facets.
                           for simplicial facets, neighbors define the ridges */
  setT    *neighbors;   /* neighbors of the facet.  If simplicial, the kth
                           neighbor is opposite the kth vertex, and the first
                           neighbor is the horizon facet for the first vertex*/
  setT    *outsideset;  /* set of points outside this facet
                           if non-empty, last point is furthest
                           if NARROWhull, includes coplanars for partitioning*/
  setT    *coplanarset; /* set of points coplanar with this facet
                           > qh.min_vertex and <= facet->max_outside
                           a point is assigned to the furthest facet
                           if non-empty, last point is furthest away */
  unsigned visitid;     /* visit_id, for visiting all neighbors,
                           all uses are independent */
  unsigned id;          /* unique identifier from qh.facet_id */
  unsigned nummerge:9;  /* number of merges */
#define qh_MAXnummerge 511 /*     2^9-1, 32 flags total, see "flags:" in io.c */
  flagT    tricoplanar:1; /* True if TRIangulate and simplicial and coplanar with a neighbor */
                          /*   all tricoplanars share the same apex */
                          /*   all tricoplanars share the same ->center, ->normal, ->offset, ->maxoutside */
                          /*     ->keepcentrum is true for the owner.  It has the ->coplanareset */
                          /*   if ->degenerate, does not span facet (one logical ridge) */
                          /*   during qh_triangulate, f.trivisible points to original facet */
  flagT    newfacet:1;  /* True if facet on qh.newfacet_list (new or merged) */
  flagT    visible:1;   /* True if visible facet (will be deleted) */
  flagT    toporient:1; /* True if created with top orientation
                           after merging, use ridge orientation */
  flagT    simplicial:1;/* True if simplicial facet, ->ridges may be implicit */
  flagT    seen:1;      /* used to perform operations only once, like visitid */
  flagT    seen2:1;     /* used to perform operations only once, like visitid */
  flagT    flipped:1;   /* True if facet is flipped */
  flagT    upperdelaunay:1; /* True if facet is upper envelope of Delaunay triangulation */
  flagT    notfurthest:1; /* True if last point of outsideset is not furthest*/

/*-------- flags primarily for output ---------*/
  flagT    good:1;      /* True if a facet marked good for output */
  flagT    isarea:1;    /* True if facet->f.area is defined */

/*-------- flags for merging ------------------*/
  flagT    dupridge:1;  /* True if duplicate ridge in facet */
  flagT    mergeridge:1; /* True if facet or neighbor contains a qh_MERGEridge
                            ->normal defined (also defined for mergeridge2) */
  flagT    mergeridge2:1; /* True if neighbor contains a qh_MERGEridge (mark_dupridges */
  flagT    coplanar:1;  /* True if horizon facet is coplanar at last use */
  flagT     mergehorizon:1; /* True if will merge into horizon (->coplanar) */
  flagT     cycledone:1;/* True if mergecycle_all already done */
  flagT    tested:1;    /* True if facet convexity has been tested (false after merge */
  flagT    keepcentrum:1; /* True if keep old centrum after a merge, or marks owner for ->tricoplanar */
  flagT    newmerge:1;  /* True if facet is newly merged for reducevertices */
  flagT    degenerate:1; /* True if facet is degenerate (degen_mergeset or ->tricoplanar) */
  flagT    redundant:1;  /* True if facet is redundant (degen_mergeset) */
};


/*-<a                             href="qh-poly.htm#TOC"
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
struct ridgeT {
  setT    *vertices;    /* vertices belonging to this ridge, inverse sorted by ID
                           NULL if a degen ridge (matchsame) */
  facetT  *top;         /* top facet this ridge is part of */
  facetT  *bottom;      /* bottom facet this ridge is part of */
  unsigned id;          /* unique identifier.  Same size as vertex_id and ridge_id */
  flagT    seen:1;      /* used to perform operations only once */
  flagT    tested:1;    /* True when ridge is tested for convexity */
  flagT    nonconvex:1; /* True if getmergeset detected a non-convex neighbor
                           only one ridge between neighbors may have nonconvex */
};

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="vertexT">-</a>

  vertexT
     defines a vertex

  topological information:
    next,previous       doubly-linked list of all vertices
    neighbors           set of adjacent facets (only if qh.VERTEXneighbors)

  geometric information:
    point               array of DIM3 coordinates
*/
struct vertexT {
  vertexT *next;        /* next vertex in vertex_list */
  vertexT *previous;    /* previous vertex in vertex_list */
  pointT  *point;       /* hull_dim coordinates (coordT) */
  setT    *neighbors;   /* neighboring facets of vertex, qh_vertexneighbors()
                           inits in io.c or after first merge */
  unsigned id;          /* unique identifier.  Same size as qh.vertex_id and qh.ridge_id */
  unsigned visitid;     /* for use with qh.vertex_visit, size must match */
  flagT    seen:1;      /* used to perform operations only once */
  flagT    seen2:1;     /* another seen flag */
  flagT    delridge:1;  /* vertex was part of a deleted ridge */
  flagT    deleted:1;   /* true if vertex on qh.del_vertices */
  flagT    newlist:1;   /* true if vertex on qh.newvertex_list */
};

/*======= -global variables -qh ============================*/

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh">-</a>

  qh
   all global variables for qhull are in qh, qhmem, and qhstat

  notes:
   qhmem is defined in mem.h, qhstat is defined in stat.h, qhrbox is defined in rboxpoints.h
   Access to qh_qh is via the "qh" macro.  See qh_QHpointer in user.h

   All global variables for qhull are in qh, qhmem, and qhstat
   qh must be unique for each instance of qhull
   qhstat may be shared between qhull instances.
   qhmem may be shared across multiple instances of Qhull.
   Rbox uses global variables rbox_inuse and rbox, but does not persist data across calls.

   Qhull is not multi-threaded.  Global state could be stored in thread-local storage.

   QHULL_LIB_CHECK checks that a program and the corresponding
   qhull library were built with the same type of header files.
*/

typedef struct qhT qhT;

#define QHULL_NON_REENTRANT 0
#define QHULL_QH_POINTER 1
#define QHULL_REENTRANT 2

#if qh_QHpointer_dllimport
#define qh qh_qh->
__declspec(dllimport) extern qhT *qh_qh;     /* allocated in global.c */
#define QHULL_LIB_TYPE QHULL_QH_POINTER

#elif qh_QHpointer
#define qh qh_qh->
extern qhT *qh_qh;     /* allocated in global.c */
#define QHULL_LIB_TYPE QHULL_QH_POINTER

#elif qh_dllimport
#define qh qh_qh.
__declspec(dllimport) extern qhT qh_qh;      /* allocated in global.c */
#define QHULL_LIB_TYPE QHULL_NON_REENTRANT

#else
#define qh qh_qh.
extern qhT qh_qh;
#define QHULL_LIB_TYPE QHULL_NON_REENTRANT
#endif

#define QHULL_LIB_CHECK qh_lib_check(QHULL_LIB_TYPE, sizeof(qhT), sizeof(vertexT), sizeof(ridgeT), sizeof(facetT), sizeof(setT), sizeof(qhmemT));
#define QHULL_LIB_CHECK_RBOX qh_lib_check(QHULL_LIB_TYPE, sizeof(qhT), sizeof(vertexT), sizeof(ridgeT), sizeof(facetT), 0, 0);

struct qhT {

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-const">-</a>

  qh constants
    configuration flags and constants for Qhull

  notes:
    The user configures Qhull by defining flags.  They are
    copied into qh by qh_setflags().  qh-quick.htm#options defines the flags.
*/
  boolT ALLpoints;        /* true 'Qs' if search all points for initial simplex */
  boolT ANGLEmerge;       /* true 'Qa' if sort potential merges by angle */
  boolT APPROXhull;       /* true 'Wn' if MINoutside set */
  realT   MINoutside;     /*   'Wn' min. distance for an outside point */
  boolT ANNOTATEoutput;   /* true 'Ta' if annotate output with message codes */
  boolT ATinfinity;       /* true 'Qz' if point num_points-1 is "at-infinity"
                             for improving precision in Delaunay triangulations */
  boolT AVOIDold;         /* true 'Q4' if avoid old->new merges */
  boolT BESToutside;      /* true 'Qf' if partition points into best outsideset */
  boolT CDDinput;         /* true 'Pc' if input uses CDD format (1.0/offset first) */
  boolT CDDoutput;        /* true 'PC' if print normals in CDD format (offset first) */
  boolT CHECKfrequently;  /* true 'Tc' if checking frequently */
  realT premerge_cos;     /*   'A-n'   cos_max when pre merging */
  realT postmerge_cos;    /*   'An'    cos_max when post merging */
  boolT DELAUNAY;         /* true 'd' if computing DELAUNAY triangulation */
  boolT DOintersections;  /* true 'Gh' if print hyperplane intersections */
  int   DROPdim;          /* drops dim 'GDn' for 4-d -> 3-d output */
  boolT FORCEoutput;      /* true 'Po' if forcing output despite degeneracies */
  int   GOODpoint;        /* 1+n for 'QGn', good facet if visible/not(-) from point n*/
  pointT *GOODpointp;     /*   the actual point */
  boolT GOODthreshold;    /* true if qh.lower_threshold/upper_threshold defined
                             false if qh.SPLITthreshold */
  int   GOODvertex;       /* 1+n, good facet if vertex for point n */
  pointT *GOODvertexp;     /*   the actual point */
  boolT HALFspace;        /* true 'Hn,n,n' if halfspace intersection */
  boolT ISqhullQh;        /* Set by Qhull.cpp on initialization */
  int   IStracing;        /* trace execution, 0=none, 1=least, 4=most, -1=events */
  int   KEEParea;         /* 'PAn' number of largest facets to keep */
  boolT KEEPcoplanar;     /* true 'Qc' if keeping nearest facet for coplanar points */
  boolT KEEPinside;       /* true 'Qi' if keeping nearest facet for inside points
                              set automatically if 'd Qc' */
  int   KEEPmerge;        /* 'PMn' number of facets to keep with most merges */
  realT KEEPminArea;      /* 'PFn' minimum facet area to keep */
  realT MAXcoplanar;      /* 'Un' max distance below a facet to be coplanar*/
  boolT MERGEexact;       /* true 'Qx' if exact merges (coplanar, degen, dupridge, flipped) */
  boolT MERGEindependent; /* true 'Q2' if merging independent sets */
  boolT MERGING;          /* true if exact-, pre- or post-merging, with angle and centrum tests */
  realT   premerge_centrum;  /*   'C-n' centrum_radius when pre merging.  Default is round-off */
  realT   postmerge_centrum; /*   'Cn' centrum_radius when post merging.  Default is round-off */
  boolT MERGEvertices;    /* true 'Q3' if merging redundant vertices */
  realT MINvisible;       /* 'Vn' min. distance for a facet to be visible */
  boolT NOnarrow;         /* true 'Q10' if no special processing for narrow distributions */
  boolT NOnearinside;     /* true 'Q8' if ignore near-inside points when partitioning */
  boolT NOpremerge;       /* true 'Q0' if no defaults for C-0 or Qx */
  boolT NOwide;           /* true 'Q12' if no error on wide merge due to duplicate ridge */
  boolT ONLYgood;         /* true 'Qg' if process points with good visible or horizon facets */
  boolT ONLYmax;          /* true 'Qm' if only process points that increase max_outside */
  boolT PICKfurthest;     /* true 'Q9' if process furthest of furthest points*/
  boolT POSTmerge;        /* true if merging after buildhull (Cn or An) */
  boolT PREmerge;         /* true if merging during buildhull (C-n or A-n) */
                        /* NOTE: some of these names are similar to qh_PRINT names */
  boolT PRINTcentrums;    /* true 'Gc' if printing centrums */
  boolT PRINTcoplanar;    /* true 'Gp' if printing coplanar points */
  int   PRINTdim;         /* print dimension for Geomview output */
  boolT PRINTdots;        /* true 'Ga' if printing all points as dots */
  boolT PRINTgood;        /* true 'Pg' if printing good facets */
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
  boolT QUICKhelp;        /* true if quick help message for degen input */
  boolT RANDOMdist;       /* true if randomly change distplane and setfacetplane */
  realT RANDOMfactor;     /*    maximum random perturbation */
  realT RANDOMa;          /*    qh_randomfactor is randr * RANDOMa + RANDOMb */
  realT RANDOMb;
  boolT RANDOMoutside;    /* true if select a random outside point */
  int   REPORTfreq;       /* buildtracing reports every n facets */
  int   REPORTfreq2;      /* tracemerging reports every REPORTfreq/2 facets */
  int   RERUN;            /* 'TRn' rerun qhull n times (qh.build_cnt) */
  int   ROTATErandom;     /* 'QRn' seed, 0 time, >= rotate input */
  boolT SCALEinput;       /* true 'Qbk' if scaling input */
  boolT SCALElast;        /* true 'Qbb' if scale last coord to max prev coord */
  boolT SETroundoff;      /* true 'E' if qh.DISTround is predefined */
  boolT SKIPcheckmax;     /* true 'Q5' if skip qh_check_maxout */
  boolT SKIPconvex;       /* true 'Q6' if skip convexity testing during pre-merge */
  boolT SPLITthresholds;  /* true if upper_/lower_threshold defines a region
                               used only for printing (!for qh.ONLYgood) */
  int   STOPcone;         /* 'TCn' 1+n for stopping after cone for point n */
                          /*       also used by qh_build_withresart for err exit*/
  int   STOPpoint;        /* 'TVn' 'TV-n' 1+n for stopping after/before(-)
                                        adding point n */
  int   TESTpoints;       /* 'QTn' num of test points after qh.num_points.  Test points always coplanar. */
  boolT TESTvneighbors;   /*  true 'Qv' if test vertex neighbors at end */
  int   TRACElevel;       /* 'Tn' conditional IStracing level */
  int   TRACElastrun;     /*  qh.TRACElevel applies to last qh.RERUN */
  int   TRACEpoint;       /* 'TPn' start tracing when point n is a vertex */
  realT TRACEdist;        /* 'TWn' start tracing when merge distance too big */
  int   TRACEmerge;       /* 'TMn' start tracing before this merge */
  boolT TRIangulate;      /* true 'Qt' if triangulate non-simplicial facets */
  boolT TRInormals;       /* true 'Q11' if triangulate duplicates ->normal and ->center (sets Qt) */
  boolT UPPERdelaunay;    /* true 'Qu' if computing furthest-site Delaunay */
  boolT USEstdout;        /* true 'Tz' if using stdout instead of stderr */
  boolT VERIFYoutput;     /* true 'Tv' if verify output at end of qhull */
  boolT VIRTUALmemory;    /* true 'Q7' if depth-first processing in buildhull */
  boolT VORONOI;          /* true 'v' if computing Voronoi diagram */

  /*--------input constants ---------*/
  realT AREAfactor;       /* 1/(hull_dim-1)! for converting det's to area */
  boolT DOcheckmax;       /* true if calling qh_check_maxout (qh_initqhull_globals) */
  char  *feasible_string;  /* feasible point 'Hn,n,n' for halfspace intersection */
  coordT *feasible_point;  /*    as coordinates, both malloc'd */
  boolT GETarea;          /* true 'Fa', 'FA', 'FS', 'PAn', 'PFn' if compute facet area/Voronoi volume in io.c */
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
  boolT ZEROcentrum;      /* true if 'C-0' or 'C-0 Qx'.  sets ZEROall_ok */
  realT *upper_threshold; /* don't print if facet->normal[k]>=upper_threshold[k]
                             must set either GOODthreshold or SPLITthreshold
                             if Delaunay, default is 0.0 for upper envelope */
  realT *lower_threshold; /* don't print if facet->normal[k] <=lower_threshold[k] */
  realT *upper_bound;     /* scale point[k] to new upper bound */
  realT *lower_bound;     /* scale point[k] to new lower bound
                             project if both upper_ and lower_bound == 0 */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-prec">-</a>

  qh precision constants
    precision constants for Qhull

  notes:
    qh_detroundoff() computes the maximum roundoff error for distance
    and other computations.  It also sets default values for the
    qh constants above.
*/
  realT ANGLEround;       /* max round off error for angles */
  realT centrum_radius;   /* max centrum radius for convexity (roundoff added) */
  realT cos_max;          /* max cosine for convexity (roundoff added) */
  realT DISTround;        /* max round off error for distances, 'E' overrides qh_distround() */
  realT MAXabs_coord;     /* max absolute coordinate */
  realT MAXlastcoord;     /* max last coordinate for qh_scalelast */
  realT MAXsumcoord;      /* max sum of coordinates */
  realT MAXwidth;         /* max rectilinear width of point coordinates */
  realT MINdenom_1;       /* min. abs. value for 1/x */
  realT MINdenom;         /*    use divzero if denominator < MINdenom */
  realT MINdenom_1_2;     /* min. abs. val for 1/x that allows normalization */
  realT MINdenom_2;       /*    use divzero if denominator < MINdenom_2 */
  realT MINlastcoord;     /* min. last coordinate for qh_scalelast */
  boolT NARROWhull;       /* set in qh_initialhull if angle < qh_MAXnarrow */
  realT *NEARzero;        /* hull_dim array for near zero in gausselim */
  realT NEARinside;       /* keep points for qh_check_maxout if close to facet */
  realT ONEmerge;         /* max distance for merging simplicial facets */
  realT outside_err;      /* application's epsilon for coplanar points
                             qh_check_bestdist() qh_check_points() reports error if point outside */
  realT WIDEfacet;        /* size of wide facet for skipping ridge in
                             area computation and locking centrum */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-codetern">-</a>

  qh internal constants
    internal constants for Qhull
*/
  char qhull[sizeof("qhull")]; /* "qhull" for checking ownership while debugging */
  jmp_buf errexit;        /* exit label for qh_errexit, defined by setjmp() and NOerrexit */
  char jmpXtra[40];       /* extra bytes in case jmp_buf is defined wrong by compiler */
  jmp_buf restartexit;    /* restart label for qh_errexit, defined by setjmp() and ALLOWrestart */
  char jmpXtra2[40];      /* extra bytes in case jmp_buf is defined wrong by compiler*/
  FILE *fin;              /* pointer to input file, init by qh_initqhull_start2 */
  FILE *fout;             /* pointer to output file */
  FILE *ferr;             /* pointer to error file */
  pointT *interior_point; /* center point of the initial simplex*/
  int normal_size;     /* size in bytes for facet normals and point coords*/
  int center_size;     /* size in bytes for Voronoi centers */
  int   TEMPsize;         /* size for small, temporary sets (in quick mem) */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-lists">-</a>

  qh facet and vertex lists
    defines lists of facets, new facets, visible facets, vertices, and
    new vertices.  Includes counts, next ids, and trace ids.
  see:
    qh_resetlists()
*/
  facetT *facet_list;     /* first facet */
  facetT  *facet_tail;     /* end of facet_list (dummy facet) */
  facetT *facet_next;     /* next facet for buildhull()
                             previous facets do not have outside sets
                             NARROWhull: previous facets may have coplanar outside sets for qh_outcoplanar */
  facetT *newfacet_list;  /* list of new facets to end of facet_list */
  facetT *visible_list;   /* list of visible facets preceding newfacet_list,
                             facet->visible set */
  int       num_visible;  /* current number of visible facets */
  unsigned tracefacet_id;  /* set at init, then can print whenever */
  facetT *tracefacet;     /*   set in newfacet/mergefacet, undone in delfacet*/
  unsigned tracevertex_id;  /* set at buildtracing, can print whenever */
  vertexT *tracevertex;     /*   set in newvertex, undone in delvertex*/
  vertexT *vertex_list;     /* list of all vertices, to vertex_tail */
  vertexT  *vertex_tail;    /*      end of vertex_list (dummy vertex) */
  vertexT *newvertex_list; /* list of vertices in newfacet_list, to vertex_tail
                             all vertices have 'newlist' set */
  int   num_facets;       /* number of facets in facet_list
                             includes visible faces (num_visible) */
  int   num_vertices;     /* number of vertices in facet_list */
  int   num_outside;      /* number of points in outsidesets (for tracing and RANDOMoutside)
                               includes coplanar outsideset points for NARROWhull/qh_outcoplanar() */
  int   num_good;         /* number of good facets (after findgood_all) */
  unsigned facet_id;      /* ID of next, new facet from newfacet() */
  unsigned ridge_id;      /* ID of next, new ridge from newridge() */
  unsigned vertex_id;     /* ID of next, new vertex from newvertex() */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-var">-</a>

  qh global variables
    defines minimum and maximum distances, next visit ids, several flags,
    and other global variables.
    initialize in qh_initbuild or qh_maxmin if used in qh_buildhull
*/
  unsigned long hulltime; /* ignore time to set up input and randomize */
                          /*   use unsigned to avoid wrap-around errors */
  boolT ALLOWrestart;     /* true if qh_precision can use qh.restartexit */
  int   build_cnt;        /* number of calls to qh_initbuild */
  qh_CENTER CENTERtype;   /* current type of facet->center, qh_CENTER */
  int   furthest_id;      /* pointid of furthest point, for tracing */
  facetT *GOODclosest;    /* closest facet to GOODthreshold in qh_findgood */
  boolT hasAreaVolume;    /* true if totarea, totvol was defined by qh_getarea */
  boolT hasTriangulation; /* true if triangulation created by qh_triangulate */
  realT JOGGLEmax;        /* set 'QJn' if randomly joggle input */
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
                              from makecone/attachnewfacets to deletevisible */
  boolT findbestnew;      /* true if partitioning calls qh_findbestnew */
  boolT findbest_notsharp; /* true if new facets are at least 90 degrees */
  boolT NOerrexit;        /* true if qh.errexit is not available, cleared after setjmp */
  realT PRINTcradius;     /* radius for printing centrums */
  realT PRINTradius;      /* radius for printing vertex spheres and points */
  boolT POSTmerging;      /* true when post merging */
  int   printoutvar;      /* temporary variable for qh_printbegin, etc. */
  int   printoutnum;      /* number of facets printed */
  boolT QHULLfinished;    /* True after qhull() is finished */
  realT totarea;          /* 'FA': total facet area computed by qh_getarea, hasAreaVolume */
  realT totvol;           /* 'FA': total volume computed by qh_getarea, hasAreaVolume */
  unsigned int visit_id;  /* unique ID for searching neighborhoods, */
  unsigned int vertex_visit; /* unique ID for searching vertices, reset with qh_buildtracing */
  boolT ZEROall_ok;       /* True if qh_checkzero always succeeds */
  boolT WAScoplanar;      /* True if qh_partitioncoplanar (qh_check_maxout) */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-set">-</a>

  qh global sets
    defines sets for merging, initial simplex, hashing, extra input points,
    and deleted vertices
*/
  setT *facet_mergeset;   /* temporary set of merges to be done */
  setT *degen_mergeset;   /* temporary set of degenerate and redundant merges */
  setT *hash_table;       /* hash table for matching ridges in qh_matchfacets
                             size is setsize() */
  setT *other_points;     /* additional points */
  setT *del_vertices;     /* vertices to partition and delete with visible
                             facets.  Have deleted set for checkfacet */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-buf">-</a>

  qh global buffers
    defines buffers for maxtrix operations, input, and error messages
*/
  coordT *gm_matrix;      /* (dim+1)Xdim matrix for geom.c */
  coordT **gm_row;        /* array of gm_matrix rows */
  char* line;             /* malloc'd input line of maxline+1 chars */
  int maxline;
  coordT *half_space;     /* malloc'd input array for halfspace (qh normal_size+coordT) */
  coordT *temp_malloc;    /* malloc'd input array for points */

/*-<a                             href="qh-globa.htm#TOC"
  >--------------------------------</a><a name="qh-static">-</a>

  qh static variables
    defines static variables for individual functions

  notes:
    do not use 'static' within a function.  Multiple instances of qhull
    may exist.

    do not assume zero initialization, 'QPn' may cause a restart
*/
  boolT ERREXITcalled;    /* true during qh_errexit (prevents duplicate calls */
  boolT firstcentrum;     /* for qh_printcentrum */
  boolT old_randomdist;   /* save RANDOMdist flag during io, tracing, or statistics */
  setT *coplanarfacetset;  /* set of coplanar facets for searching qh_findbesthorizon() */
  realT last_low;         /* qh_scalelast parameters for qh_setdelaunay */
  realT last_high;
  realT last_newhigh;
  unsigned lastreport;    /* for qh_buildtracing */
  int mergereport;        /* for qh_tracemerging */
  qhstatT *old_qhstat;    /* for saving qh_qhstat in save_qhull() and UsingLibQhull.  Free with qh_free() */
  setT *old_tempstack;    /* for saving qhmem.tempstack in save_qhull */
  int   ridgeoutnum;      /* number of ridges for 4OFF output (qh_printbegin,etc) */
};

/*=========== -macros- =========================*/

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="otherfacet_">-</a>

  otherfacet_(ridge, facet)
    return neighboring facet for a ridge in facet
*/
#define otherfacet_(ridge, facet) \
                        (((ridge)->top == (facet)) ? (ridge)->bottom : (ridge)->top)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="getid_">-</a>

  getid_(p)
    return int ID for facet, ridge, or vertex
    return qh_IDunknown(-1) if NULL
*/
#define getid_(p)       ((p) ? (int)((p)->id) : qh_IDunknown)

/*============== FORALL macros ===================*/

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLfacets">-</a>

  FORALLfacets { ... }
    assign 'facet' to each facet in qh.facet_list

  notes:
    uses 'facetT *facet;'
    assumes last facet is a sentinel

  see:
    FORALLfacet_( facetlist )
*/
#define FORALLfacets for (facet=qh facet_list;facet && facet->next;facet=facet->next)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLpoints">-</a>

  FORALLpoints { ... }
    assign 'point' to each point in qh.first_point, qh.num_points

  declare:
    coordT *point, *pointtemp;
*/
#define FORALLpoints FORALLpoint_(qh first_point, qh num_points)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLpoint_">-</a>

  FORALLpoint_( points, num) { ... }
    assign 'point' to each point in points array of num points

  declare:
    coordT *point, *pointtemp;
*/
#define FORALLpoint_(points, num) for (point= (points), \
      pointtemp= (points)+qh hull_dim*(num); point < pointtemp; point += qh hull_dim)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLvertices">-</a>

  FORALLvertices { ... }
    assign 'vertex' to each vertex in qh.vertex_list

  declare:
    vertexT *vertex;

  notes:
    assumes qh.vertex_list terminated with a sentinel
*/
#define FORALLvertices for (vertex=qh vertex_list;vertex && vertex->next;vertex= vertex->next)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHfacet_">-</a>

  FOREACHfacet_( facets ) { ... }
    assign 'facet' to each facet in facets

  declare:
    facetT *facet, **facetp;

  see:
    <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHfacet_(facets)    FOREACHsetelement_(facetT, facets, facet)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighbor_">-</a>

  FOREACHneighbor_( facet ) { ... }
    assign 'neighbor' to each neighbor in facet->neighbors

  FOREACHneighbor_( vertex ) { ... }
    assign 'neighbor' to each neighbor in vertex->neighbors

  declare:
    facetT *neighbor, **neighborp;

  see:
    <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHneighbor_(facet)  FOREACHsetelement_(facetT, facet->neighbors, neighbor)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHpoint_">-</a>

  FOREACHpoint_( points ) { ... }
    assign 'point' to each point in points set

  declare:
    pointT *point, **pointp;

  see:
    <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHpoint_(points)    FOREACHsetelement_(pointT, points, point)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHridge_">-</a>

  FOREACHridge_( ridges ) { ... }
    assign 'ridge' to each ridge in ridges set

  declare:
    ridgeT *ridge, **ridgep;

  see:
    <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHridge_(ridges)    FOREACHsetelement_(ridgeT, ridges, ridge)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertex_">-</a>

  FOREACHvertex_( vertices ) { ... }
    assign 'vertex' to each vertex in vertices set

  declare:
    vertexT *vertex, **vertexp;

  see:
    <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertex_(vertices) FOREACHsetelement_(vertexT, vertices,vertex)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHfacet_i_">-</a>

  FOREACHfacet_i_( facets ) { ... }
    assign 'facet' and 'facet_i' for each facet in facets set

  declare:
    facetT *facet;
    int     facet_n, facet_i;

  see:
    <a href="qset.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHfacet_i_(facets)    FOREACHsetelement_i_(facetT, facets, facet)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighbor_i_">-</a>

  FOREACHneighbor_i_( facet ) { ... }
    assign 'neighbor' and 'neighbor_i' for each neighbor in facet->neighbors

  FOREACHneighbor_i_( vertex ) { ... }
    assign 'neighbor' and 'neighbor_i' for each neighbor in vertex->neighbors

  declare:
    facetT *neighbor;
    int     neighbor_n, neighbor_i;

  see:
    <a href="qset.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHneighbor_i_(facet)  FOREACHsetelement_i_(facetT, facet->neighbors, neighbor)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHpoint_i_">-</a>

  FOREACHpoint_i_( points ) { ... }
    assign 'point' and 'point_i' for each point in points set

  declare:
    pointT *point;
    int     point_n, point_i;

  see:
    <a href="qset.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHpoint_i_(points)    FOREACHsetelement_i_(pointT, points, point)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHridge_i_">-</a>

  FOREACHridge_i_( ridges ) { ... }
    assign 'ridge' and 'ridge_i' for each ridge in ridges set

  declare:
    ridgeT *ridge;
    int     ridge_n, ridge_i;

  see:
    <a href="qset.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHridge_i_(ridges)    FOREACHsetelement_i_(ridgeT, ridges, ridge)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertex_i_">-</a>

  FOREACHvertex_i_( vertices ) { ... }
    assign 'vertex' and 'vertex_i' for each vertex in vertices set

  declare:
    vertexT *vertex;
    int     vertex_n, vertex_i;

  see:
    <a href="qset.h#FOREACHsetelement_i_">FOREACHsetelement_i_</a>
*/
#define FOREACHvertex_i_(vertices) FOREACHsetelement_i_(vertexT, vertices,vertex)

/********* -libqhull.c prototypes (duplicated from qhull_a.h) **********************/

void    qh_qhull(void);
boolT   qh_addpoint(pointT *furthest, facetT *facet, boolT checkdist);
void    qh_printsummary(FILE *fp);

/********* -user.c prototypes (alphabetical) **********************/

void    qh_errexit(int exitcode, facetT *facet, ridgeT *ridge);
void    qh_errprint(const char* string, facetT *atfacet, facetT *otherfacet, ridgeT *atridge, vertexT *atvertex);
int     qh_new_qhull(int dim, int numpoints, coordT *points, boolT ismalloc,
                char *qhull_cmd, FILE *outfile, FILE *errfile);
void    qh_printfacetlist(facetT *facetlist, setT *facets, boolT printall);
void    qh_printhelp_degenerate(FILE *fp);
void    qh_printhelp_narrowhull(FILE *fp, realT minangle);
void    qh_printhelp_singular(FILE *fp);
void    qh_user_memsizes(void);

/********* -usermem.c prototypes (alphabetical) **********************/
void    qh_exit(int exitcode);
void    qh_fprintf_stderr(int msgcode, const char *fmt, ... );
void    qh_free(void *mem);
void   *qh_malloc(size_t size);

/********* -userprintf.c and userprintf_rbox.c prototypes **********************/
void    qh_fprintf(FILE *fp, int msgcode, const char *fmt, ... );
void    qh_fprintf_rbox(FILE *fp, int msgcode, const char *fmt, ... );

/***** -geom.c/geom2.c/random.c prototypes (duplicated from geom.h, random.h) ****************/

facetT *qh_findbest(pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT newfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart);
facetT *qh_findbestnew(pointT *point, facetT *startfacet,
                     realT *dist, boolT bestoutside, boolT *isoutside, int *numpart);
boolT   qh_gram_schmidt(int dim, realT **rows);
void    qh_outerinner(facetT *facet, realT *outerplane, realT *innerplane);
void    qh_printsummary(FILE *fp);
void    qh_projectinput(void);
void    qh_randommatrix(realT *buffer, int dim, realT **row);
void    qh_rotateinput(realT **rows);
void    qh_scaleinput(void);
void    qh_setdelaunay(int dim, int count, pointT *points);
coordT  *qh_sethalfspace_all(int dim, int count, coordT *halfspaces, pointT *feasible);

/***** -global.c prototypes (alphabetical) ***********************/

unsigned long qh_clock(void);
void    qh_checkflags(char *command, char *hiddenflags);
void    qh_clear_outputflags(void);
void    qh_freebuffers(void);
void    qh_freeqhull(boolT allmem);
void    qh_freeqhull2(boolT allmem);
void    qh_init_A(FILE *infile, FILE *outfile, FILE *errfile, int argc, char *argv[]);
void    qh_init_B(coordT *points, int numpoints, int dim, boolT ismalloc);
void    qh_init_qhull_command(int argc, char *argv[]);
void    qh_initbuffers(coordT *points, int numpoints, int dim, boolT ismalloc);
void    qh_initflags(char *command);
void    qh_initqhull_buffers(void);
void    qh_initqhull_globals(coordT *points, int numpoints, int dim, boolT ismalloc);
void    qh_initqhull_mem(void);
void    qh_initqhull_outputflags(void);
void    qh_initqhull_start(FILE *infile, FILE *outfile, FILE *errfile);
void    qh_initqhull_start2(FILE *infile, FILE *outfile, FILE *errfile);
void    qh_initthresholds(char *command);
void    qh_lib_check(int qhullLibraryType, int qhTsize, int vertexTsize, int ridgeTsize, int facetTsize, int setTsize, int qhmemTsize);
void    qh_option(const char *option, int *i, realT *r);
#if qh_QHpointer
void    qh_restore_qhull(qhT **oldqh);
qhT    *qh_save_qhull(void);
#endif

/***** -io.c prototypes (duplicated from io.h) ***********************/

void    qh_dfacet(unsigned id);
void    qh_dvertex(unsigned id);
void    qh_printneighborhood(FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall);
void    qh_produce_output(void);
coordT *qh_readpoints(int *numpoints, int *dimension, boolT *ismalloc);


/********* -mem.c prototypes (duplicated from mem.h) **********************/

void qh_meminit(FILE *ferr);
void qh_memfreeshort(int *curlong, int *totlong);

/********* -poly.c/poly2.c prototypes (duplicated from poly.h) **********************/

void    qh_check_output(void);
void    qh_check_points(void);
setT   *qh_facetvertices(facetT *facetlist, setT *facets, boolT allfacets);
facetT *qh_findbestfacet(pointT *point, boolT bestoutside,
           realT *bestdist, boolT *isoutside);
vertexT *qh_nearvertex(facetT *facet, pointT *point, realT *bestdistp);
pointT *qh_point(int id);
setT   *qh_pointfacet(void /*qh.facet_list*/);
int     qh_pointid(pointT *point);
setT   *qh_pointvertex(void /*qh.facet_list*/);
void    qh_setvoronoi_all(void);
void    qh_triangulate(void /*qh.facet_list*/);

/********* -rboxlib.c prototypes **********************/
int     qh_rboxpoints(FILE* fout, FILE* ferr, char* rbox_command);
void    qh_errexit_rbox(int exitcode);

/********* -stat.c prototypes (duplicated from stat.h) **********************/

void    qh_collectstatistics(void);
void    qh_printallstatistics(FILE *fp, const char *string);

#endif /* qhDEFlibqhull */

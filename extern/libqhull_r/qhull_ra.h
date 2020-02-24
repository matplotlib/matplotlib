/*<html><pre>  -<a                             href="qh-qhull_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   qhull_ra.h
   all header files for compiling qhull with reentrant code
   included before C++ headers for user_r.h:QHULL_CRTDBG

   see qh-qhull.htm

   see libqhull_r.h for user-level definitions

   see user_r.h for user-definable constants

   defines internal functions for libqhull_r.c global_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/qhull_ra.h#1 $$Change: 2661 $
   $DateTime: 2019/05/24 20:09:58 $$Author: bbarber $

   Notes:  grep for ((" and (" to catch fprintf("lkasdjf");
           full parens around (x?y:z)
           use '#include "libqhull_r/qhull_ra.h"' to avoid name clashes
*/

#ifndef qhDEFqhulla
#define qhDEFqhulla 1

#include "libqhull_r.h"  /* Includes user_r.h and data types */

#include "stat_r.h"
#include "random_r.h"
#include "mem_r.h"
#include "qset_r.h"
#include "geom_r.h"
#include "merge_r.h"
#include "poly_r.h"
#include "io_r.h"

#include <setjmp.h>
#include <string.h>
#include <math.h>
#include <float.h>    /* some compilers will not need float.h */
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
/*** uncomment here and qset_r.c
     if string.h does not define memcpy()
#include <memory.h>
*/

#if qh_CLOCKtype == 2  /* defined in user_r.h from libqhull_r.h */
#include <sys/types.h>
#include <sys/times.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER  /* Microsoft Visual C++ -- warning level 4 */
#pragma warning( disable : 4100)  /* unreferenced formal parameter */
#pragma warning( disable : 4127)  /* conditional expression is constant */
#pragma warning( disable : 4706)  /* assignment within conditional function */
#pragma warning( disable : 4996)  /* function was declared deprecated(strcpy, localtime, etc.) */
#endif

/* ======= -macros- =========== */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="traceN">-</a>

  traceN((qh, qh->ferr, 0Nnnn, "format\n", vars));
    calls qh_fprintf if qh.IStracing >= N

    Add debugging traps to the end of qh_fprintf

  notes:
    removing tracing reduces code size but doesn't change execution speed
*/
#ifndef qh_NOtrace
#define trace0(args) {if (qh->IStracing) qh_fprintf args;}
#define trace1(args) {if (qh->IStracing >= 1) qh_fprintf args;}
#define trace2(args) {if (qh->IStracing >= 2) qh_fprintf args;}
#define trace3(args) {if (qh->IStracing >= 3) qh_fprintf args;}
#define trace4(args) {if (qh->IStracing >= 4) qh_fprintf args;}
#define trace5(args) {if (qh->IStracing >= 5) qh_fprintf args;}
#else /* qh_NOtrace */
#define trace0(args) {}
#define trace1(args) {}
#define trace2(args) {}
#define trace3(args) {}
#define trace4(args) {}
#define trace5(args) {}
#endif /* qh_NOtrace */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >--------------------------------</a><a name="QHULL_UNUSED">-</a>

  Define an unused variable to avoid compiler warnings

  Derived from Qt's corelib/global/qglobal.h

*/

#if defined(__cplusplus) && defined(__INTEL_COMPILER) && !defined(QHULL_OS_WIN)
template <typename T>
inline void qhullUnused(T &x) { (void)x; }
#  define QHULL_UNUSED(x) qhullUnused(x);
#else
#  define QHULL_UNUSED(x) (void)x;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/***** -libqhull_r.c prototypes (alphabetical after qhull) ********************/

void    qh_qhull(qhT *qh);
boolT   qh_addpoint(qhT *qh, pointT *furthest, facetT *facet, boolT checkdist);
void    qh_build_withrestart(qhT *qh);
vertexT *qh_buildcone(qhT *qh, pointT *furthest, facetT *facet, int goodhorizon, facetT **retryfacet);
boolT   qh_buildcone_mergepinched(qhT *qh, vertexT *apex, facetT *facet, facetT **retryfacet);
boolT   qh_buildcone_onlygood(qhT *qh, vertexT *apex, int goodhorizon);
void    qh_buildhull(qhT *qh);
void    qh_buildtracing(qhT *qh, pointT *furthest, facetT *facet);
void    qh_errexit2(qhT *qh, int exitcode, facetT *facet, facetT *otherfacet);
void    qh_findhorizon(qhT *qh, pointT *point, facetT *facet, int *goodvisible,int *goodhorizon);
pointT *qh_nextfurthest(qhT *qh, facetT **visible);
void    qh_partitionall(qhT *qh, setT *vertices, pointT *points,int npoints);
void    qh_partitioncoplanar(qhT *qh, pointT *point, facetT *facet, realT *dist, boolT allnew);
void    qh_partitionpoint(qhT *qh, pointT *point, facetT *facet);
void    qh_partitionvisible(qhT *qh, boolT allpoints, int *numpoints);
void    qh_joggle_restart(qhT *qh, const char *reason);
void    qh_printsummary(qhT *qh, FILE *fp);

/***** -global_r.c internal prototypes (alphabetical) ***********************/

void    qh_appendprint(qhT *qh, qh_PRINT format);
void    qh_freebuild(qhT *qh, boolT allmem);
void    qh_freebuffers(qhT *qh);
void    qh_initbuffers(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);

/***** -stat_r.c internal prototypes (alphabetical) ***********************/

void    qh_allstatA(qhT *qh);
void    qh_allstatB(qhT *qh);
void    qh_allstatC(qhT *qh);
void    qh_allstatD(qhT *qh);
void    qh_allstatE(qhT *qh);
void    qh_allstatE2(qhT *qh);
void    qh_allstatF(qhT *qh);
void    qh_allstatG(qhT *qh);
void    qh_allstatH(qhT *qh);
void    qh_freebuffers(qhT *qh);
void    qh_initbuffers(qhT *qh, coordT *points, int numpoints, int dim, boolT ismalloc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFqhulla */

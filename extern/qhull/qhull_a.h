/*<html><pre>  -<a                             href="qh-qhull.htm"
  >-------------------------------</a><a name="TOP">-</a>

   qhull_a.h
   all header files for compiling qhull

   see qh-qhull.htm

   see libqhull.h for user-level definitions

   see user.h for user-defineable constants

   defines internal functions for libqhull.c global.c

   Copyright (c) 1993-2012 The Geometry Center.
   $Id: //main/2011/qhull/src/libqhull/qhull_a.h#3 $$Change: 1464 $
   $DateTime: 2012/01/25 22:58:41 $$Author: bbarber $

   Notes:  grep for ((" and (" to catch fprintf("lkasdjf");
           full parens around (x?y:z)
           use '#include qhull/qhull_a.h' to avoid name clashes
*/

#ifndef qhDEFqhulla
#define qhDEFqhulla 1

#include "libqhull.h"  /* Defines data types */

#include "stat.h"
#include "random.h"
#include "mem.h"
#include "qset.h"
#include "geom.h"
#include "merge.h"
#include "poly.h"
#include "io.h"

#include <setjmp.h>
#include <string.h>
#include <math.h>
#include <float.h>    /* some compilers will not need float.h */
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
/*** uncomment here and qset.c
     if string.h does not define memcpy()
#include <memory.h>
*/

#if qh_CLOCKtype == 2  /* defined in user.h from libqhull.h */
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

/*-<a                             href="qh-qhull.htm#TOC"
  >--------------------------------</a><a name="traceN">-</a>

  traceN((qh ferr, 0Nnnn, "format\n", vars));
    calls qh_fprintf if qh.IStracing >= N

    Add debugging traps to the end of qh_fprintf

  notes:
    removing tracing reduces code size but doesn't change execution speed
*/
#ifndef qh_NOtrace
#define trace0(args) {if (qh IStracing) qh_fprintf args;}
#define trace1(args) {if (qh IStracing >= 1) qh_fprintf args;}
#define trace2(args) {if (qh IStracing >= 2) qh_fprintf args;}
#define trace3(args) {if (qh IStracing >= 3) qh_fprintf args;}
#define trace4(args) {if (qh IStracing >= 4) qh_fprintf args;}
#define trace5(args) {if (qh IStracing >= 5) qh_fprintf args;}
#else /* qh_NOtrace */
#define trace0(args) {}
#define trace1(args) {}
#define trace2(args) {}
#define trace3(args) {}
#define trace4(args) {}
#define trace5(args) {}
#endif /* qh_NOtrace */

/*-<a                             href="qh-qhull.htm#TOC"
  >--------------------------------</a><a name="QHULL_UNUSED">-</a>

*/

/* See Qt's qglobal.h */
#if !defined(SAG_COM) && (defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
#   define QHULL_OS_WIN
#elif defined(__MWERKS__) && defined(__INTEL__)
#   define QHULL_OS_WIN
#endif
#if defined(__INTEL_COMPILER) && !defined(QHULL_OS_WIN)
template <typename T>
inline void qhullUnused(T &x) { (void)x; }
#  define QHULL_UNUSED(x) qhullUnused(x);
#else
#  define QHULL_UNUSED(x) (void)x;
#endif

/***** -libqhull.c prototypes (alphabetical after qhull) ********************/

void    qh_qhull(void);
boolT   qh_addpoint(pointT *furthest, facetT *facet, boolT checkdist);
void    qh_buildhull(void);
void    qh_buildtracing(pointT *furthest, facetT *facet);
void    qh_build_withrestart(void);
void    qh_errexit2(int exitcode, facetT *facet, facetT *otherfacet);
void    qh_findhorizon(pointT *point, facetT *facet, int *goodvisible,int *goodhorizon);
pointT *qh_nextfurthest(facetT **visible);
void    qh_partitionall(setT *vertices, pointT *points,int npoints);
void    qh_partitioncoplanar(pointT *point, facetT *facet, realT *dist);
void    qh_partitionpoint(pointT *point, facetT *facet);
void    qh_partitionvisible(boolT allpoints, int *numpoints);
void    qh_precision(const char *reason);
void    qh_printsummary(FILE *fp);

/***** -global.c internal prototypes (alphabetical) ***********************/

void    qh_appendprint(qh_PRINT format);
void    qh_freebuild(boolT allmem);
void    qh_freebuffers(void);
void    qh_initbuffers(coordT *points, int numpoints, int dim, boolT ismalloc);

/***** -stat.c internal prototypes (alphabetical) ***********************/

void    qh_allstatA(void);
void    qh_allstatB(void);
void    qh_allstatC(void);
void    qh_allstatD(void);
void    qh_allstatE(void);
void    qh_allstatE2 (void);
void    qh_allstatF(void);
void    qh_allstatG(void);
void    qh_allstatH(void);
void    qh_freebuffers(void);
void    qh_initbuffers(coordT *points, int numpoints, int dim, boolT ismalloc);

#endif /* qhDEFqhulla */

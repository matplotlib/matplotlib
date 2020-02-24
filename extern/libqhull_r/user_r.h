/*<html><pre>  -<a                             href="qh-user_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   user_r.h
   user redefinable constants

   for each source file, user_r.h is included first

   see qh-user_r.htm.  see COPYING for copyright information.

   See user_r.c for sample code.

   before reading any code, review libqhull_r.h for data structure definitions

Sections:
   ============= qhull library constants ======================
   ============= data types and configuration macros ==========
   ============= performance related constants ================
   ============= memory constants =============================
   ============= joggle constants =============================
   ============= conditional compilation ======================
   ============= merge constants ==============================
   ============= Microsoft DevStudio ==========================

Code flags --
  NOerrors -- the code does not call qh_errexit()
  WARN64 -- the code may be incompatible with 64-bit pointers

*/

#include <float.h>
#include <limits.h>
#include <time.h>

#ifndef qhDEFuser
#define qhDEFuser 1

/* Derived from Qt's corelib/global/qglobal.h */
#if !defined(SAG_COM) && !defined(__CYGWIN__) && (defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
#   define QHULL_OS_WIN
#elif defined(__MWERKS__) && defined(__INTEL__) /* Metrowerks discontinued before the release of Intel Macs */
#   define QHULL_OS_WIN
#endif

/*============================================================*/
/*============= qhull library constants ======================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="filenamelen">-</a>

  FILENAMElen -- max length for TI and TO filenames

*/

#define qh_FILENAMElen 500

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="msgcode">-</a>

  msgcode -- Unique message codes for qh_fprintf

  If add new messages, assign these values and increment in user.h and user_r.h
  See QhullError.h for 10000 error codes.
  Cannot use '0031' since it would be octal

  def counters =  [31/32/33/38, 1067, 2113, 3079, 4097, 5006,
     6428, 7027/7028/7035/7068/7070/7102, 8163, 9428, 10000, 11034]

  See: qh_ERR* [libqhull_r.h]
*/

#define MSG_TRACE0     0   /* always include if logging ('Tn') */
#define MSG_TRACE1  1000
#define MSG_TRACE2  2000
#define MSG_TRACE3  3000
#define MSG_TRACE4  4000
#define MSG_TRACE5  5000
#define MSG_ERROR   6000   /* errors written to qh.ferr */
#define MSG_WARNING 7000
#define MSG_STDERR  8000   /* log messages Written to qh.ferr */
#define MSG_OUTPUT  9000
#define MSG_QHULL_ERROR 10000 /* errors thrown by QhullError.cpp (QHULLlastError is in QhullError.h) */
#define MSG_FIX    11000   /* Document as 'QH11... FIX: ...' */
#define MSG_MAXLEN  3000   /* qh_printhelp_degenerate() in user_r.c */


/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="qh_OPTIONline">-</a>

  qh_OPTIONline -- max length of an option line 'FO'
*/
#define qh_OPTIONline 80

/*============================================================*/
/*============= data types and configuration macros ==========*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="realT">-</a>

  realT
    set the size of floating point numbers

  qh_REALdigits
    maximimum number of significant digits

  qh_REAL_1, qh_REAL_2n, qh_REAL_3n
    format strings for printf

  qh_REALmax, qh_REALmin
    maximum and minimum (near zero) values

  qh_REALepsilon
    machine roundoff.  Maximum roundoff error for addition and multiplication.

  notes:
   Select whether to store floating point numbers in single precision (float)
   or double precision (double).

   Use 'float' to save about 8% in time and 25% in space.  This is particularly
   helpful if high-d where convex hulls are space limited.  Using 'float' also
   reduces the printed size of Qhull's output since numbers have 8 digits of
   precision.

   Use 'double' when greater arithmetic precision is needed.  This is needed
   for Delaunay triangulations and Voronoi diagrams when you are not merging
   facets.

   If 'double' gives insufficient precision, your data probably includes
   degeneracies.  If so you should use facet merging (done by default)
   or exact arithmetic (see imprecision section of manual, qh-impre.htm).
   You may also use option 'Po' to force output despite precision errors.

   You may use 'long double', but many format statements need to be changed
   and you may need a 'long double' square root routine.  S. Grundmann
   (sg@eeiwzb.et.tu-dresden.de) has done this.  He reports that the code runs
   much slower with little gain in precision.

   WARNING: on some machines,    int f(){realT a= REALmax;return (a == REALmax);}
      returns False.  Use (a > REALmax/2) instead of (a == REALmax).

   REALfloat =   1      all numbers are 'float' type
             =   0      all numbers are 'double' type
*/
#define REALfloat 0

#if (REALfloat == 1)
#define realT float
#define REALmax FLT_MAX
#define REALmin FLT_MIN
#define REALepsilon FLT_EPSILON
#define qh_REALdigits 8   /* maximum number of significant digits */
#define qh_REAL_1 "%6.8g "
#define qh_REAL_2n "%6.8g %6.8g\n"
#define qh_REAL_3n "%6.8g %6.8g %6.8g\n"

#elif (REALfloat == 0)
#define realT double
#define REALmax DBL_MAX
#define REALmin DBL_MIN
#define REALepsilon DBL_EPSILON
#define qh_REALdigits 16    /* maximum number of significant digits */
#define qh_REAL_1 "%6.16g "
#define qh_REAL_2n "%6.16g %6.16g\n"
#define qh_REAL_3n "%6.16g %6.16g %6.16g\n"

#else
#error unknown float option
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="countT">-</a>

  countT
    The type for counts and identifiers (e.g., the number of points, vertex identifiers)
    Currently used by C++ code-only.  Decided against using it for setT because most sets are small.

    Defined as 'int' for C-code compatibility and QH11026

    QH11026 FIX: countT may be defined as a 'unsigned int', but several code issues need to be solved first.  See countT in Changes.txt
*/

#ifndef DEFcountT
#define DEFcountT 1
typedef int countT;
#endif
#define COUNTmax INT_MAX

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="qh_POINTSmax">-</a>

  qh_POINTSmax
    Maximum number of points for qh.num_points and point allocation in qh_readpoints
*/
#define qh_POINTSmax (INT_MAX-16)

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="CPUclock">-</a>

  qh_CPUclock
    define the clock() function for reporting the total time spent by Qhull
    returns CPU ticks as a 'long int'
    qh_CPUclock is only used for reporting the total time spent by Qhull

  qh_SECticks
    the number of clock ticks per second

  notes:
    looks for CLOCKS_PER_SEC, CLOCKS_PER_SECOND, or assumes microseconds
    to define a custom clock, set qh_CLOCKtype to 0

    if your system does not use clock() to return CPU ticks, replace
    qh_CPUclock with the corresponding function.  It is converted
    to 'unsigned long' to prevent wrap-around during long runs.  By default,
    <time.h> defines clock_t as 'long'

   Set qh_CLOCKtype to

     1          for CLOCKS_PER_SEC, CLOCKS_PER_SECOND, or microsecond
                Note:  may fail if more than 1 hour elapsed time

     2          use qh_clock() with POSIX times() (see global_r.c)
*/
#define qh_CLOCKtype 1  /* change to the desired number */

#if (qh_CLOCKtype == 1)

#if defined(CLOCKS_PER_SECOND)
#define qh_CPUclock    ((unsigned long)clock())  /* return CPU clock */
#define qh_SECticks CLOCKS_PER_SECOND

#elif defined(CLOCKS_PER_SEC)
#define qh_CPUclock    ((unsigned long)clock())  /* return CPU clock */
#define qh_SECticks CLOCKS_PER_SEC

#elif defined(CLK_TCK)
#define qh_CPUclock    ((unsigned long)clock())  /* return CPU clock */
#define qh_SECticks CLK_TCK

#else
#define qh_CPUclock    ((unsigned long)clock())  /* return CPU clock */
#define qh_SECticks 1E6
#endif

#elif (qh_CLOCKtype == 2)
#define qh_CPUclock    qh_clock()  /* return CPU clock */
#define qh_SECticks 100

#else /* qh_CLOCKtype == ? */
#error unknown clock option
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RANDOM">-</a>

  qh_RANDOMtype, qh_RANDOMmax, qh_RANDOMseed
    define random number generator

    qh_RANDOMint generates a random integer between 0 and qh_RANDOMmax.
    qh_RANDOMseed sets the random number seed for qh_RANDOMint

  Set qh_RANDOMtype (default 5) to:
    1       for random() with 31 bits (UCB)
    2       for rand() with RAND_MAX or 15 bits (system 5)
    3       for rand() with 31 bits (Sun)
    4       for lrand48() with 31 bits (Solaris)
    5       for qh_rand(qh) with 31 bits (included with Qhull, requires 'qh')

  notes:
    Random numbers are used by rbox to generate point sets.  Random
    numbers are used by Qhull to rotate the input ('QRn' option),
    simulate a randomized algorithm ('Qr' option), and to simulate
    roundoff errors ('Rn' option).

    Random number generators differ between systems.  Most systems provide
    rand() but the period varies.  The period of rand() is not critical
    since qhull does not normally use random numbers.

    The default generator is Park & Miller's minimal standard random
    number generator [CACM 31:1195 '88].  It is included with Qhull.

    If qh_RANDOMmax is wrong, qhull will report a warning and Geomview
    output will likely be invisible.
*/
#define qh_RANDOMtype 5   /* *** change to the desired number *** */

#if (qh_RANDOMtype == 1)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, random()/MAX */
#define qh_RANDOMint random()
#define qh_RANDOMseed_(qh, seed) srandom(seed);

#elif (qh_RANDOMtype == 2)
#ifdef RAND_MAX
#define qh_RANDOMmax ((realT)RAND_MAX)
#else
#define qh_RANDOMmax ((realT)32767)   /* 15 bits (System 5) */
#endif
#define qh_RANDOMint  rand()
#define qh_RANDOMseed_(qh, seed) srand((unsigned int)seed);

#elif (qh_RANDOMtype == 3)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, Sun */
#define qh_RANDOMint  rand()
#define qh_RANDOMseed_(qh, seed) srand((unsigned int)seed);

#elif (qh_RANDOMtype == 4)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, lrand38()/MAX */
#define qh_RANDOMint lrand48()
#define qh_RANDOMseed_(qh, seed) srand48(seed);

#elif (qh_RANDOMtype == 5)  /* 'qh' is an implicit parameter */
#define qh_RANDOMmax ((realT)2147483646UL)  /* 31 bits, qh_rand/MAX */
#define qh_RANDOMint qh_rand(qh)
#define qh_RANDOMseed_(qh, seed) qh_srand(qh, seed);
/* unlike rand(), never returns 0 */

#else
#error: unknown random option
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="ORIENTclock">-</a>

  qh_ORIENTclock
    0 for inward pointing normals by Geomview convention
*/
#define qh_ORIENTclock 0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RANDOMdist">-</a>

  qh_RANDOMdist
    define for random perturbation of qh_distplane and qh_setfacetplane (qh.RANDOMdist, 'QRn')

  For testing qh.DISTround.  Qhull should not depend on computations always producing the same roundoff error 

  #define qh_RANDOMdist 1e-13
*/

/*============================================================*/
/*============= joggle constants =============================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEdefault">-</a>

  qh_JOGGLEdefault
    default qh.JOGGLEmax is qh.DISTround * qh_JOGGLEdefault

  notes:
    rbox s r 100 | qhull QJ1e-15 QR0 generates 90% faults at distround 7e-16
    rbox s r 100 | qhull QJ1e-14 QR0 generates 70% faults
    rbox s r 100 | qhull QJ1e-13 QR0 generates 35% faults
    rbox s r 100 | qhull QJ1e-12 QR0 generates 8% faults
    rbox s r 100 | qhull QJ1e-11 QR0 generates 1% faults
    rbox s r 100 | qhull QJ1e-10 QR0 generates 0% faults
    rbox 1000 W0 | qhull QJ1e-12 QR0 generates 86% faults
    rbox 1000 W0 | qhull QJ1e-11 QR0 generates 20% faults
    rbox 1000 W0 | qhull QJ1e-10 QR0 generates 2% faults
    the later have about 20 points per facet, each of which may interfere

    pick a value large enough to avoid retries on most inputs
*/
#define qh_JOGGLEdefault 30000.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEincrease">-</a>

  qh_JOGGLEincrease
    factor to increase qh.JOGGLEmax on qh_JOGGLEretry or qh_JOGGLEagain
*/
#define qh_JOGGLEincrease 10.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEretry">-</a>

  qh_JOGGLEretry
    if ZZretry = qh_JOGGLEretry, increase qh.JOGGLEmax

notes:
try twice at the original value in case of bad luck the first time
*/
#define qh_JOGGLEretry 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEagain">-</a>

  qh_JOGGLEagain
    every following qh_JOGGLEagain, increase qh.JOGGLEmax

  notes:
    1 is OK since it's already failed qh_JOGGLEretry times
*/
#define qh_JOGGLEagain 1

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEmaxincrease">-</a>

  qh_JOGGLEmaxincrease
    maximum qh.JOGGLEmax due to qh_JOGGLEincrease
    relative to qh.MAXwidth

  notes:
    qh.joggleinput will retry at this value until qh_JOGGLEmaxretry
*/
#define qh_JOGGLEmaxincrease 1e-2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="JOGGLEmaxretry">-</a>

  qh_JOGGLEmaxretry
    stop after qh_JOGGLEmaxretry attempts
*/
#define qh_JOGGLEmaxretry 50

/*============================================================*/
/*============= performance related constants ================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="HASHfactor">-</a>

  qh_HASHfactor
    total hash slots / used hash slots.  Must be at least 1.1.

  notes:
    =2 for at worst 50% occupancy for qh.hash_table and normally 25% occupancy
*/
#define qh_HASHfactor 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="VERIFYdirect">-</a>

  qh_VERIFYdirect
    with 'Tv' verify all points against all facets if op count is smaller

  notes:
    if greater, calls qh_check_bestdist() instead
*/
#define qh_VERIFYdirect 1000000

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="INITIALsearch">-</a>

  qh_INITIALsearch
     if qh_INITIALmax, search points up to this dimension
*/
#define qh_INITIALsearch 6

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="INITIALmax">-</a>

  qh_INITIALmax
    if dim >= qh_INITIALmax, use min/max coordinate points for initial simplex

  notes:
    from points with non-zero determinants
    use option 'Qs' to override (much slower)
*/
#define qh_INITIALmax 8

/*============================================================*/
/*============= memory constants =============================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MEMalign">-</a>

  qh_MEMalign
    memory alignment for qh_meminitbuffers() in global_r.c

  notes:
    to avoid bus errors, memory allocation must consider alignment requirements.
    malloc() automatically takes care of alignment.   Since mem_r.c manages
    its own memory, we need to explicitly specify alignment in
    qh_meminitbuffers().

    A safe choice is sizeof(double).  sizeof(float) may be used if doubles
    do not occur in data structures and pointers are the same size.  Be careful
    of machines (e.g., DEC Alpha) with large pointers.

    If using gcc, best alignment is [fmax_() is defined in geom_r.h]
              #define qh_MEMalign fmax_(__alignof__(realT),__alignof__(void *))
*/
#define qh_MEMalign ((int)(fmax_(sizeof(realT), sizeof(void *))))

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MEMbufsize">-</a>

  qh_MEMbufsize
    size of additional memory buffers

  notes:
    used for qh_meminitbuffers() in global_r.c
*/
#define qh_MEMbufsize 0x10000       /* allocate 64K memory buffers */

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MEMinitbuf">-</a>

  qh_MEMinitbuf
    size of initial memory buffer

  notes:
    use for qh_meminitbuffers() in global_r.c
*/
#define qh_MEMinitbuf 0x20000      /* initially allocate 128K buffer */

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="INFINITE">-</a>

  qh_INFINITE
    on output, indicates Voronoi center at infinity
*/
#define qh_INFINITE  -10.101

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="DEFAULTbox">-</a>

  qh_DEFAULTbox
    default box size (Geomview expects 0.5)

  qh_DEFAULTbox
    default box size for integer coorindate (rbox only)
*/
#define qh_DEFAULTbox 0.5
#define qh_DEFAULTzbox 1e6

/*============================================================*/
/*============= conditional compilation ======================*/
/*============================================================*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="compiler">-</a>

  __cplusplus
    defined by C++ compilers

  __MSC_VER
    defined by Microsoft Visual C++

  __MWERKS__ && __INTEL__
    defined by Metrowerks when compiling for Windows (not Intel-based Macintosh)

  __MWERKS__ && __POWERPC__
    defined by Metrowerks when compiling for PowerPC-based Macintosh

  __STDC__
    defined for strict ANSI C
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="COMPUTEfurthest">-</a>

  qh_COMPUTEfurthest
    compute furthest distance to an outside point instead of storing it with the facet
    =1 to compute furthest

  notes:
    computing furthest saves memory but costs time
      about 40% more distance tests for partitioning
      removes facet->furthestdist
*/
#define qh_COMPUTEfurthest 0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="KEEPstatistics">-</a>

  qh_KEEPstatistics
    =0 removes most of statistic gathering and reporting

  notes:
    if 0, code size is reduced by about 4%.
*/
#define qh_KEEPstatistics 1

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXoutside">-</a>

  qh_MAXoutside
    record outer plane for each facet
    =1 to record facet->maxoutside

  notes:
    this takes a realT per facet and slightly slows down qhull
    it produces better outer planes for geomview output
*/
#define qh_MAXoutside 1

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="NOmerge">-</a>

  qh_NOmerge
    disables facet merging if defined
    For MSVC compiles, use qhull_r-exports-nomerge.def instead of qhull_r-exports.def

  notes:
    This saves about 25% space, 30% space in combination with qh_NOtrace, 
    and 36% with qh_NOtrace and qh_KEEPstatistics 0

    Unless option 'Q0' is used
      qh_NOmerge sets 'QJ' to avoid precision errors

  see:
    <a href="mem_r.h#NOmem">qh_NOmem</a> in mem_r.h

    see user_r.c/user_eg.c for removing io_r.o

  #define qh_NOmerge
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="NOtrace">-</a>

  qh_NOtrace
    no tracing if defined
    disables 'Tn', 'TMn', 'TPn' and 'TWn'
    override with 'Qw' for qh_addpoint tracing and various other items

  notes:
    This saves about 15% space.
    Removes all traceN((...)) code and substantial sections of qh.IStracing code

  #define qh_NOtrace
*/

#if 0  /* sample code */
    exitcode= qh_new_qhull(qhT *qh, dim, numpoints, points, ismalloc,
                      flags, outfile, errfile);
    qh_freeqhull(qhT *qh, !qh_ALL); /* frees long memory used by second call */
    qh_memfreeshort(qhT *qh, &curlong, &totlong);  /* frees short memory and memory allocator */
#endif

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="QUICKhelp">-</a>

  qh_QUICKhelp
    =1 to use abbreviated help messages, e.g., for degenerate inputs
*/
#define qh_QUICKhelp    0

/*============================================================*/
/*============= merge constants ==============================*/
/*============================================================*/
/*
   These constants effect facet merging.  You probably will not need
   to modify them.  They effect the performance of facet merging.
*/

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="BESTcentrum">-</a>

  qh_BESTcentrum
     if > 2*dim+n vertices, qh_findbestneighbor() tests centrums (faster)
     else, qh_findbestneighbor() tests all vertices (much better merges)

  qh_BESTcentrum2
     if qh_BESTcentrum2 * DIM3 + BESTcentrum < #vertices tests centrums
*/
#define qh_BESTcentrum 20
#define qh_BESTcentrum2 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="BESTnonconvex">-</a>

  qh_BESTnonconvex
    if > dim+n neighbors, qh_findbestneighbor() tests nonconvex ridges.

  notes:
    It is needed because qh_findbestneighbor is slow for large facets
*/
#define qh_BESTnonconvex 15

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="COPLANARratio">-</a>

  qh_COPLANARratio
    for 3-d+ merging, qh.MINvisible is n*premerge_centrum

  notes:
    for non-merging, it's DISTround
*/
#define qh_COPLANARratio 3

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="DIMmergeVertex">-</a>

  qh_DIMmergeVertex
    max dimension for vertex merging (it is not effective in high-d)
*/
#define qh_DIMmergeVertex 6

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="DIMreduceBuild">-</a>

  qh_DIMreduceBuild
     max dimension for vertex reduction during build (slow in high-d)
*/
#define qh_DIMreduceBuild 5

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="DISToutside">-</a>

  qh_DISToutside
    When is a point clearly outside of a facet?
    Stops search in qh_findbestnew or qh_partitionall
    qh_findbest uses qh.MINoutside since since it is only called if no merges.

  notes:
    'Qf' always searches for best facet
    if !qh.MERGING, same as qh.MINoutside.
    if qh_USEfindbestnew, increase value since neighboring facets may be ill-behaved
      [Note: Zdelvertextot occurs normally with interior points]
            RBOX 1000 s Z1 G1e-13 t1001188774 | QHULL Tv
    When there is a sharp edge, need to move points to a
    clearly good facet; otherwise may be lost in another partitioning.
    if too big then O(n^2) behavior for partitioning in cone
    if very small then important points not processed
    Needed in qh_partitionall for
      RBOX 1000 s Z1 G1e-13 t1001032651 | QHULL Tv
    Needed in qh_findbestnew for many instances of
      RBOX 1000 s Z1 G1e-13 t | QHULL Tv

  See:
    qh_DISToutside -- when is a point clearly outside of a facet
    qh_SEARCHdist -- when is facet coplanar with the best facet?
    qh_USEfindbestnew -- when to use qh_findbestnew for qh_partitionpoint()
*/
#define qh_DISToutside ((qh_USEfindbestnew ? 2 : 1) * \
     fmax_((qh->MERGING ? 2 : 1)*qh->MINoutside, qh->max_outside))

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXcheckpoint">-</a>

  qh_MAXcheckpoint
    Report up to qh_MAXcheckpoint errors per facet in qh_check_point ('Tv')
*/
#define qh_MAXcheckpoint 10

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXcoplanarcentrum">-</a>

  qh_MAXcoplanarcentrum
    if pre-merging with qh.MERGEexact ('Qx') and f.nummerge > qh_MAXcoplanarcentrum
      use f.maxoutside instead of qh.centrum_radius for coplanarity testing

  notes:
    see qh_test_nonsimplicial_merges
    with qh.MERGEexact, a coplanar ridge is ignored until post-merging
    otherwise a large facet with many merges may take all the facets
*/
#define qh_MAXcoplanarcentrum 10

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXnewcentrum">-</a>

  qh_MAXnewcentrum
    if <= dim+n vertices (n approximates the number of merges),
      reset the centrum in qh_updatetested() and qh_mergecycle_facets()

  notes:
    needed to reduce cost and because centrums may move too much if
    many vertices in high-d
*/
#define qh_MAXnewcentrum 5

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXnewmerges">-</a>

  qh_MAXnewmerges
    if >n newmerges, qh_merge_nonconvex() calls qh_reducevertices_centrums.

  notes:
    It is needed because postmerge can merge many facets at once
*/
#define qh_MAXnewmerges 2

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOconcavehorizon">-</a>

  qh_RATIOconcavehorizon
    ratio of horizon vertex distance to max_outside for concave, twisted new facets in qh_test_nonsimplicial_merge
    if too small, end up with vertices far below merged facets
*/
#define qh_RATIOconcavehorizon 20.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOconvexmerge">-</a>

  qh_RATIOconvexmerge
    ratio of vertex distance to qh.min_vertex for clearly convex new facets in qh_test_nonsimplicial_merge

  notes:
    must be convex for MRGtwisted
*/
#define qh_RATIOconvexmerge 10.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOcoplanarapex">-</a>

  qh_RATIOcoplanarapex
    ratio of best distance for coplanar apex vs. vertex merge in qh_getpinchedmerges

  notes:
    A coplanar apex always works, while a vertex merge may fail
*/
#define qh_RATIOcoplanarapex 3.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOcoplanaroutside">-</a>

  qh_RATIOcoplanaroutside
    qh.MAXoutside ratio to repartition a coplanar point in qh_partitioncoplanar and qh_check_maxout

  notes:
    combines several tests, see qh_partitioncoplanar

*/
#define qh_RATIOcoplanaroutside 30.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOmaxsimplex">-</a>

  qh_RATIOmaxsimplex
    ratio of max determinate to estimated determinate for searching all points in qh_maxsimplex

  notes:
    As each point is added to the simplex, the max determinate is should approximate the previous determinate * qh.MAXwidth
    If maxdet is significantly less, the simplex may not be full-dimensional.
    If so, all points are searched, stopping at 10 times qh_RATIOmaxsimplex
*/
#define qh_RATIOmaxsimplex 1.0e-3

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOnearinside">-</a>

  qh_RATIOnearinside
    ratio of qh.NEARinside to qh.ONEmerge for retaining inside points for
    qh_check_maxout().

  notes:
    This is overkill since do not know the correct value.
    It effects whether 'Qc' reports all coplanar points
    Not used for 'd' since non-extreme points are coplanar, nearly incident points
*/
#define qh_RATIOnearinside 5

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOpinchedsubridge">-</a>

  qh_RATIOpinchedsubridge
    ratio to qh.ONEmerge to accept vertices in qh_findbest_pinchedvertex
    skips search of neighboring vertices
    facet width may increase by this ratio
*/
#define qh_RATIOpinchedsubridge 10.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOtrypinched">-</a>

  qh_RATIOtrypinched
    ratio to qh.ONEmerge to try qh_getpinchedmerges in qh_buildcone_mergepinched
    otherwise a duplicate ridge will increase facet width by this amount
*/
#define qh_RATIOtrypinched 4.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="RATIOtwisted">-</a>

  qh_RATIOtwisted
    maximum ratio to qh.ONEmerge to merge twisted facets in qh_merge_twisted
*/
#define qh_RATIOtwisted 20.0

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="SEARCHdist">-</a>

  qh_SEARCHdist
    When is a facet coplanar with the best facet?
    qh_findbesthorizon: all coplanar facets of the best facet need to be searched.
        increases minsearch if ischeckmax and more than 100 neighbors (is_5x_minsearch)
  See:
    qh_DISToutside -- when is a point clearly outside of a facet
    qh_SEARCHdist -- when is facet coplanar with the best facet?
    qh_USEfindbestnew -- when to use qh_findbestnew for qh_partitionpoint()
*/
#define qh_SEARCHdist ((qh_USEfindbestnew ? 2 : 1) * \
      (qh->max_outside + 2 * qh->DISTround + fmax_( qh->MINvisible, qh->MAXcoplanar)));

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="USEfindbestnew">-</a>

  qh_USEfindbestnew
     Always use qh_findbestnew for qh_partitionpoint, otherwise use
     qh_findbestnew if merged new facet or sharpnewfacets.

  See:
    qh_DISToutside -- when is a point clearly outside of a facet
    qh_SEARCHdist -- when is facet coplanar with the best facet?
    qh_USEfindbestnew -- when to use qh_findbestnew for qh_partitionpoint()
*/
#define qh_USEfindbestnew (zzval_(Ztotmerge) > 50)

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="MAXnarrow">-</a>

  qh_MAXnarrow
    max. cosine in initial hull that sets qh.NARROWhull

  notes:
    If qh.NARROWhull, the initial partition does not make
    coplanar points.  If narrow, a coplanar point can be
    coplanar to two facets of opposite orientations and
    distant from the exact convex hull.

    Conservative estimate.  Don't actually see problems until it is -1.0
*/
#define qh_MAXnarrow -0.99999999

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="WARNnarrow">-</a>

  qh_WARNnarrow
    max. cosine in initial hull to warn about qh.NARROWhull

  notes:
    this is a conservative estimate.
    Don't actually see problems until it is -1.0.  See qh-impre.htm
*/
#define qh_WARNnarrow -0.999999999999999

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEcoplanar">-</a>

  qh_WIDEcoplanar
    n*MAXcoplanar or n*MINvisible for a WIDEfacet

    if vertex is further than qh.WIDEfacet from the hyperplane
    then its ridges are not counted in computing the area, and
    the facet's centrum is frozen.

  notes:
    qh.WIDEfacet= max(qh.MAXoutside,qh_WIDEcoplanar*qh.MAXcoplanar,
    qh_WIDEcoplanar * qh.MINvisible);
*/
#define qh_WIDEcoplanar 6

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEduplicate">-</a>

  qh_WIDEduplicate
    merge ratio for errexit from qh_forcedmerges due to duplicate ridge
    Override with option Q12-allow-wide

  Notes:
    Merging a duplicate ridge can lead to very wide facets.
*/
#define qh_WIDEduplicate 100

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEdupridge">-</a>

  qh_WIDEdupridge
    Merge ratio for selecting a forced dupridge merge

  Notes:
    Merging a dupridge can lead to very wide facets.
*/
#define qh_WIDEdupridge 50

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEmaxoutside">-</a>

  qh_WIDEmaxoutside
    Precision ratio for maximum increase for qh.max_outside in qh_check_maxout
    Precision errors while constructing the hull, may lead to very wide facets when checked in qh_check_maxout
    Nearly incident points in 4-d and higher is the most likely culprit
    Skip qh_check_maxout with 'Q5' (no-check-outer)
    Do not error with option 'Q12' (allow-wide)
    Do not warn with options 'Q12 Pp'
*/
#define qh_WIDEmaxoutside 100

/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEmaxoutside2">-</a>

  qh_WIDEmaxoutside2
    Precision ratio for maximum qh.max_outside in qh_check_maxout
    Skip qh_check_maxout with 'Q5' no-check-outer
    Do not error with option 'Q12' allow-wide
*/
#define qh_WIDEmaxoutside2 (10*qh_WIDEmaxoutside)


/*-<a                             href="qh-user_r.htm#TOC"
>--------------------------------</a><a name="WIDEpinched">-</a>

  qh_WIDEpinched
    Merge ratio for distance between pinched vertices compared to current facet width for qh_getpinchedmerges and qh_next_vertexmerge
    Reports warning and merges duplicate ridges instead
    Enable these attempts with option Q14 merge-pinched-vertices

  notes:
    Merging pinched vertices should prevent duplicate ridges (see qh_WIDEduplicate)
    Merging the duplicate ridges may be better than merging the pinched vertices
    Found up to 45x ratio for qh_pointdist -- for ((i=1; i<20; i++)); do rbox 175 C1,6e-13 t | qhull d T4 2>&1 | tee x.1 | grep  -E 'QH|non-simplicial|Statis|pinched'; done
    Actual distance to facets is a third to a tenth of the qh_pointdist (T1)
*/
#define qh_WIDEpinched 100

/*-<a                             href="qh-user_r.htm#TOC"
  >--------------------------------</a><a name="ZEROdelaunay">-</a>

  qh_ZEROdelaunay
    a zero Delaunay facet occurs for input sites coplanar with their convex hull
    the last normal coefficient of a zero Delaunay facet is within
        qh_ZEROdelaunay * qh.ANGLEround of 0

  notes:
    qh_ZEROdelaunay does not allow for joggled input ('QJ').

    You can avoid zero Delaunay facets by surrounding the input with a box.

    Use option 'PDk:-n' to explicitly define zero Delaunay facets
      k= dimension of input sites (e.g., 3 for 3-d Delaunay triangulation)
      n= the cutoff for zero Delaunay facets (e.g., 'PD3:-1e-12')
*/
#define qh_ZEROdelaunay 2

/*============================================================*/
/*============= Microsoft DevStudio ==========================*/
/*============================================================*/

/*
   Finding Memory Leaks Using the CRT Library
   https://msdn.microsoft.com/en-us/library/x98tx3cf(v=vs.100).aspx

   Reports enabled in qh_lib_check for Debug window and stderr

   From 2005=>msvcr80d, 2010=>msvcr100d, 2012=>msvcr110d

   Watch: {,,msvcr80d.dll}_crtBreakAlloc  Value from {n} in the leak report
   _CrtSetBreakAlloc(689); // qh_lib_check() [global_r.c]

   Examples
     http://free-cad.sourceforge.net/SrcDocu/d2/d7f/MemDebug_8cpp_source.html
     https://github.com/illlust/Game/blob/master/library/MemoryLeak.cpp
*/
#if 0   /* off (0) by default for QHULL_CRTDBG */
#define QHULL_CRTDBG
#endif

#if defined(_MSC_VER) && defined(_DEBUG) && defined(QHULL_CRTDBG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#endif /* qh_DEFuser */

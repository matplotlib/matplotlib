/*<html><pre>  -<a                             href="qh-user.htm"
  >-------------------------------</a><a name="TOP">-</a>

   user.h
   user redefinable constants

   see qh-user.htm.  see COPYING for copyright information.

   before reading any code, review libqhull.h for data structure definitions and
   the "qh" macro.

Sections:
   ============= qhull library constants ======================
   ============= data types and configuration macros ==========
   ============= performance related constants ================
   ============= memory constants =============================
   ============= joggle constants =============================
   ============= conditional compilation ======================
   ============= -merge constants- ============================

Code flags --
  NOerrors -- the code does not call qh_errexit()
  WARN64 -- the code may be incompatible with 64-bit pointers

*/

#include <time.h>

#ifndef qhDEFuser
#define qhDEFuser 1

/*============================================================*/
/*============= qhull library constants ======================*/
/*============================================================*/

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="filenamelen">-</a>

  FILENAMElen -- max length for TI and TO filenames

*/

#define qh_FILENAMElen 500

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="msgcode">-</a>

  msgcode -- Unique message codes for qh_fprintf

  If add new messages, assign these values and increment.

  def counters =  [27, 1047, 2059, 3025, 4068, 5003,
     6241, 7079, 8143, 9410, 10000, 11026]

  See: qh_ERR* [libqhull.h]
*/

#define MSG_TRACE0 0
#define MSG_TRACE1 1000
#define MSG_TRACE2 2000
#define MSG_TRACE3 3000
#define MSG_TRACE4 4000
#define MSG_TRACE5 5000
#define MSG_ERROR  6000   /* errors written to qh.ferr */
#define MSG_WARNING 7000
#define MSG_STDERR  8000  /* log messages Written to qh.ferr */
#define MSG_OUTPUT  9000
#define MSG_QHULL_ERROR 10000 /* errors thrown by QhullError [QhullError.h] */
#define MSG_FIXUP  11000  /* FIXUP QH11... */
#define MSG_MAXLEN  3000 /* qh_printhelp_degenerate() in user.c */


/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="qh_OPTIONline">-</a>

  qh_OPTIONline -- max length of an option line 'FO'
*/
#define qh_OPTIONline 80

/*============================================================*/
/*============= data types and configuration macros ==========*/
/*============================================================*/

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
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

     2          use qh_clock() with POSIX times() (see global.c)
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

/*-<a                             href="qh-user.htm#TOC"
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
    5       for qh_rand() with 31 bits (included with Qhull)

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
#define qh_RANDOMseed_(seed) srandom(seed);

#elif (qh_RANDOMtype == 2)
#ifdef RAND_MAX
#define qh_RANDOMmax ((realT)RAND_MAX)
#else
#define qh_RANDOMmax ((realT)32767)   /* 15 bits (System 5) */
#endif
#define qh_RANDOMint  rand()
#define qh_RANDOMseed_(seed) srand((unsigned)seed);

#elif (qh_RANDOMtype == 3)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, Sun */
#define qh_RANDOMint  rand()
#define qh_RANDOMseed_(seed) srand((unsigned)seed);

#elif (qh_RANDOMtype == 4)
#define qh_RANDOMmax ((realT)0x7fffffffUL)  /* 31 bits, lrand38()/MAX */
#define qh_RANDOMint lrand48()
#define qh_RANDOMseed_(seed) srand48(seed);

#elif (qh_RANDOMtype == 5)
#define qh_RANDOMmax ((realT)2147483646UL)  /* 31 bits, qh_rand/MAX */
#define qh_RANDOMint qh_rand()
#define qh_RANDOMseed_(seed) qh_srand(seed);
/* unlike rand(), never returns 0 */

#else
#error: unknown random option
#endif

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="ORIENTclock">-</a>

  qh_ORIENTclock
    0 for inward pointing normals by Geomview convention
*/
#define qh_ORIENTclock 0


/*============================================================*/
/*============= joggle constants =============================*/
/*============================================================*/

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
>--------------------------------</a><a name="JOGGLEincrease">-</a>

qh_JOGGLEincrease
factor to increase qh.JOGGLEmax on qh_JOGGLEretry or qh_JOGGLEagain
*/
#define qh_JOGGLEincrease 10.0

/*-<a                             href="qh-user.htm#TOC"
>--------------------------------</a><a name="JOGGLEretry">-</a>

qh_JOGGLEretry
if ZZretry = qh_JOGGLEretry, increase qh.JOGGLEmax

notes:
try twice at the original value in case of bad luck the first time
*/
#define qh_JOGGLEretry 2

/*-<a                             href="qh-user.htm#TOC"
>--------------------------------</a><a name="JOGGLEagain">-</a>

qh_JOGGLEagain
every following qh_JOGGLEagain, increase qh.JOGGLEmax

notes:
1 is OK since it's already failed qh_JOGGLEretry times
*/
#define qh_JOGGLEagain 1

/*-<a                             href="qh-user.htm#TOC"
>--------------------------------</a><a name="JOGGLEmaxincrease">-</a>

qh_JOGGLEmaxincrease
maximum qh.JOGGLEmax due to qh_JOGGLEincrease
relative to qh.MAXwidth

notes:
qh.joggleinput will retry at this value until qh_JOGGLEmaxretry
*/
#define qh_JOGGLEmaxincrease 1e-2

/*-<a                             href="qh-user.htm#TOC"
>--------------------------------</a><a name="JOGGLEmaxretry">-</a>

qh_JOGGLEmaxretry
stop after qh_JOGGLEmaxretry attempts
*/
#define qh_JOGGLEmaxretry 100

/*============================================================*/
/*============= performance related constants ================*/
/*============================================================*/

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="HASHfactor">-</a>

  qh_HASHfactor
    total hash slots / used hash slots.  Must be at least 1.1.

  notes:
    =2 for at worst 50% occupancy for qh hash_table and normally 25% occupancy
*/
#define qh_HASHfactor 2

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="VERIFYdirect">-</a>

  qh_VERIFYdirect
    with 'Tv' verify all points against all facets if op count is smaller

  notes:
    if greater, calls qh_check_bestdist() instead
*/
#define qh_VERIFYdirect 1000000

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="INITIALsearch">-</a>

  qh_INITIALsearch
     if qh_INITIALmax, search points up to this dimension
*/
#define qh_INITIALsearch 6

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="MEMalign">-</a>

  qh_MEMalign
    memory alignment for qh_meminitbuffers() in global.c

  notes:
    to avoid bus errors, memory allocation must consider alignment requirements.
    malloc() automatically takes care of alignment.   Since mem.c manages
    its own memory, we need to explicitly specify alignment in
    qh_meminitbuffers().

    A safe choice is sizeof(double).  sizeof(float) may be used if doubles
    do not occur in data structures and pointers are the same size.  Be careful
    of machines (e.g., DEC Alpha) with large pointers.

    If using gcc, best alignment is
              #define qh_MEMalign fmax_(__alignof__(realT),__alignof__(void *))
*/
#define qh_MEMalign ((int)(fmax_(sizeof(realT), sizeof(void *))))

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="MEMbufsize">-</a>

  qh_MEMbufsize
    size of additional memory buffers

  notes:
    used for qh_meminitbuffers() in global.c
*/
#define qh_MEMbufsize 0x10000       /* allocate 64K memory buffers */

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="MEMinitbuf">-</a>

  qh_MEMinitbuf
    size of initial memory buffer

  notes:
    use for qh_meminitbuffers() in global.c
*/
#define qh_MEMinitbuf 0x20000      /* initially allocate 128K buffer */

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="INFINITE">-</a>

  qh_INFINITE
    on output, indicates Voronoi center at infinity
*/
#define qh_INFINITE  -10.101

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="compiler">-</a>

  __cplusplus
    defined by C++ compilers

  __MSC_VER
    defined by Microsoft Visual C++

  __MWERKS__ && __POWERPC__
    defined by Metrowerks when compiling for the Power Macintosh

  __STDC__
    defined for strict ANSI C
*/

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="KEEPstatistics">-</a>

  qh_KEEPstatistics
    =0 removes most of statistic gathering and reporting

  notes:
    if 0, code size is reduced by about 4%.
*/
#define qh_KEEPstatistics 1

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="MAXoutside">-</a>

  qh_MAXoutside
    record outer plane for each facet
    =1 to record facet->maxoutside

  notes:
    this takes a realT per facet and slightly slows down qhull
    it produces better outer planes for geomview output
*/
#define qh_MAXoutside 1

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="NOmerge">-</a>

  qh_NOmerge
    disables facet merging if defined

  notes:
    This saves about 10% space.

    Unless 'Q0'
      qh_NOmerge sets 'QJ' to avoid precision errors

    #define qh_NOmerge

  see:
    <a href="mem.h#NOmem">qh_NOmem</a> in mem.c

    see user.c/user_eg.c for removing io.o
*/

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="NOtrace">-</a>

  qh_NOtrace
    no tracing if defined

  notes:
    This saves about 5% space.

    #define qh_NOtrace
*/

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="QHpointer">-</a>

  qh_QHpointer
    access global data with pointer or static structure

  qh_QHpointer  = 1     access globals via a pointer to allocated memory
                        enables qh_saveqhull() and qh_restoreqhull()
                        [2010, gcc] costs about 4% in time and 4% in space
                        [2003, msvc] costs about 8% in time and 2% in space

                = 0     qh_qh and qh_qhstat are static data structures
                        only one instance of qhull() can be active at a time
                        default value

  qh_QHpointer_dllimport and qh_dllimport define qh_qh as __declspec(dllimport) [libqhull.h]
  It is required for msvc-2005.  It is not needed for gcc.

  notes:
    all global variables for qhull are in qh, qhmem, and qhstat
    qh is defined in libqhull.h
    qhmem is defined in mem.h
    qhstat is defined in stat.h
    C++ build defines qh_QHpointer [libqhullp.pro, libqhullcpp.pro]

  see:
    user_eg.c for an example
*/
#ifdef qh_QHpointer
#if qh_dllimport
#error QH6207 Qhull error: Use qh_QHpointer_dllimport instead of qh_dllimport with qh_QHpointer
#endif
#else
#define qh_QHpointer 0
#if qh_QHpointer_dllimport
#error QH6234 Qhull error: Use qh_dllimport instead of qh_QHpointer_dllimport when qh_QHpointer is not defined
#endif
#endif
#if 0  /* sample code */
    qhT *oldqhA, *oldqhB;

    exitcode= qh_new_qhull(dim, numpoints, points, ismalloc,
                      flags, outfile, errfile);
    /* use results from first call to qh_new_qhull */
    oldqhA= qh_save_qhull();
    exitcode= qh_new_qhull(dimB, numpointsB, pointsB, ismalloc,
                      flags, outfile, errfile);
    /* use results from second call to qh_new_qhull */
    oldqhB= qh_save_qhull();
    qh_restore_qhull(&oldqhA);
    /* use results from first call to qh_new_qhull */
    qh_freeqhull(qh_ALL);  /* frees all memory used by first call */
    qh_restore_qhull(&oldqhB);
    /* use results from second call to qh_new_qhull */
    qh_freeqhull(!qh_ALL); /* frees long memory used by second call */
    qh_memfreeshort(&curlong, &totlong);  /* frees short memory and memory allocator */
#endif

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="QUICKhelp">-</a>

  qh_QUICKhelp
    =1 to use abbreviated help messages, e.g., for degenerate inputs
*/
#define qh_QUICKhelp    0

/*============================================================*/
/*============= -merge constants- ============================*/
/*============================================================*/
/*
   These constants effect facet merging.  You probably will not need
   to modify them.  They effect the performance of facet merging.
*/

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="DIMmergeVertex">-</a>

  qh_DIMmergeVertex
    max dimension for vertex merging (it is not effective in high-d)
*/
#define qh_DIMmergeVertex 6

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="DIMreduceBuild">-</a>

  qh_DIMreduceBuild
     max dimension for vertex reduction during build (slow in high-d)
*/
#define qh_DIMreduceBuild 5

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="BESTcentrum">-</a>

  qh_BESTcentrum
     if > 2*dim+n vertices, qh_findbestneighbor() tests centrums (faster)
     else, qh_findbestneighbor() tests all vertices (much better merges)

  qh_BESTcentrum2
     if qh_BESTcentrum2 * DIM3 + BESTcentrum < #vertices tests centrums
*/
#define qh_BESTcentrum 20
#define qh_BESTcentrum2 2

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="BESTnonconvex">-</a>

  qh_BESTnonconvex
    if > dim+n neighbors, qh_findbestneighbor() tests nonconvex ridges.

  notes:
    It is needed because qh_findbestneighbor is slow for large facets
*/
#define qh_BESTnonconvex 15

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="MAXnewmerges">-</a>

  qh_MAXnewmerges
    if >n newmerges, qh_merge_nonconvex() calls qh_reducevertices_centrums.

  notes:
    It is needed because postmerge can merge many facets at once
*/
#define qh_MAXnewmerges 2

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="MAXnewcentrum">-</a>

  qh_MAXnewcentrum
    if <= dim+n vertices (n approximates the number of merges),
      reset the centrum in qh_updatetested() and qh_mergecycle_facets()

  notes:
    needed to reduce cost and because centrums may move too much if
    many vertices in high-d
*/
#define qh_MAXnewcentrum 5

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="COPLANARratio">-</a>

  qh_COPLANARratio
    for 3-d+ merging, qh.MINvisible is n*premerge_centrum

  notes:
    for non-merging, it's DISTround
*/
#define qh_COPLANARratio 3

/*-<a                             href="qh-user.htm#TOC"
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
     fmax_((qh MERGING ? 2 : 1)*qh MINoutside, qh max_outside))

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="RATIOnearinside">-</a>

  qh_RATIOnearinside
    ratio of qh.NEARinside to qh.ONEmerge for retaining inside points for
    qh_check_maxout().

  notes:
    This is overkill since do not know the correct value.
    It effects whether 'Qc' reports all coplanar points
    Not used for 'd' since non-extreme points are coplanar
*/
#define qh_RATIOnearinside 5

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="SEARCHdist">-</a>

  qh_SEARCHdist
    When is a facet coplanar with the best facet?
    qh_findbesthorizon: all coplanar facets of the best facet need to be searched.

  See:
    qh_DISToutside -- when is a point clearly outside of a facet
    qh_SEARCHdist -- when is facet coplanar with the best facet?
    qh_USEfindbestnew -- when to use qh_findbestnew for qh_partitionpoint()
*/
#define qh_SEARCHdist ((qh_USEfindbestnew ? 2 : 1) * \
      (qh max_outside + 2 * qh DISTround + fmax_( qh MINvisible, qh MAXcoplanar)));

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
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

/*-<a                             href="qh-user.htm#TOC"
  >--------------------------------</a><a name="WARNnarrow">-</a>

  qh_WARNnarrow
    max. cosine in initial hull to warn about qh.NARROWhull

  notes:
    this is a conservative estimate.
    Don't actually see problems until it is -1.0.  See qh-impre.htm
*/
#define qh_WARNnarrow -0.999999999999999

/*-<a                             href="qh-user.htm#TOC"
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

#endif /* qh_DEFuser */

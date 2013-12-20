/*<html><pre>  -<a                             href="qh-stat.htm"
  >-------------------------------</a><a name="TOP">-</a>

   stat.h
     contains all statistics that are collected for qhull

   see qh-stat.htm and stat.c

   Copyright (c) 1993-2012 The Geometry Center.
   $Id: //main/2011/qhull/src/libqhull/stat.h#5 $$Change: 1464 $
   $DateTime: 2012/01/25 22:58:41 $$Author: bbarber $

   recompile qhull if you change this file

   Integer statistics are Z* while real statistics are W*.

   define maydebugx to call a routine at every statistic event

*/

#ifndef qhDEFstat
#define qhDEFstat 1

#include "libqhull.h"

/*-<a                             href="qh-stat.htm#TOC"
  >-------------------------------</a><a name="KEEPstatistics">-</a>

  qh_KEEPstatistics
    0 turns off statistic gathering (except zzdef/zzinc/zzadd/zzval/wwval)
*/
#ifndef qh_KEEPstatistics
#define qh_KEEPstatistics 1
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >-------------------------------</a><a name="statistics">-</a>

  Zxxx for integers, Wxxx for reals

  notes:
    be sure that all statistics are defined in stat.c
      otherwise initialization may core dump
    can pick up all statistics by:
      grep '[zw].*_[(][ZW]' *.c >z.x
    remove trailers with query">-</a>
    remove leaders with  query-replace-regexp [ ^I]+  (
*/
#if qh_KEEPstatistics
enum statistics {     /* alphabetical after Z/W */
    Zacoplanar,
    Wacoplanarmax,
    Wacoplanartot,
    Zangle,
    Wangle,
    Wanglemax,
    Wanglemin,
    Zangletests,
    Wareatot,
    Wareamax,
    Wareamin,
    Zavoidold,
    Wavoidoldmax,
    Wavoidoldtot,
    Zback0,
    Zbestcentrum,
    Zbestdist,
    Zbestlower,
    Zbestlowerv,
    Zcentrumtests,
    Zcheckpart,
    Zcomputefurthest,
    Zconcave,
    Wconcavemax,
    Wconcavetot,
    Zconcaveridges,
    Zconcaveridge,
    Zcoplanar,
    Wcoplanarmax,
    Wcoplanartot,
    Zcoplanarangle,
    Zcoplanarcentrum,
    Zcoplanarhorizon,
    Zcoplanarinside,
    Zcoplanarpart,
    Zcoplanarridges,
    Wcpu,
    Zcyclefacetmax,
    Zcyclefacettot,
    Zcyclehorizon,
    Zcyclevertex,
    Zdegen,
    Wdegenmax,
    Wdegentot,
    Zdegenvertex,
    Zdelfacetdup,
    Zdelridge,
    Zdelvertextot,
    Zdelvertexmax,
    Zdetsimplex,
    Zdistcheck,
    Zdistconvex,
    Zdistgood,
    Zdistio,
    Zdistplane,
    Zdiststat,
    Zdistvertex,
    Zdistzero,
    Zdoc1,
    Zdoc2,
    Zdoc3,
    Zdoc4,
    Zdoc5,
    Zdoc6,
    Zdoc7,
    Zdoc8,
    Zdoc9,
    Zdoc10,
    Zdoc11,
    Zdoc12,
    Zdropdegen,
    Zdropneighbor,
    Zdupflip,
    Zduplicate,
    Wduplicatemax,
    Wduplicatetot,
    Zdupridge,
    Zdupsame,
    Zflipped,
    Wflippedmax,
    Wflippedtot,
    Zflippedfacets,
    Zfindbest,
    Zfindbestmax,
    Zfindbesttot,
    Zfindcoplanar,
    Zfindfail,
    Zfindhorizon,
    Zfindhorizonmax,
    Zfindhorizontot,
    Zfindjump,
    Zfindnew,
    Zfindnewmax,
    Zfindnewtot,
    Zfindnewjump,
    Zfindnewsharp,
    Zgauss0,
    Zgoodfacet,
    Zhashlookup,
    Zhashridge,
    Zhashridgetest,
    Zhashtests,
    Zinsidevisible,
    Zintersect,
    Zintersectfail,
    Zintersectmax,
    Zintersectnum,
    Zintersecttot,
    Zmaxneighbors,
    Wmaxout,
    Wmaxoutside,
    Zmaxridges,
    Zmaxvertex,
    Zmaxvertices,
    Zmaxvneighbors,
    Zmemfacets,
    Zmempoints,
    Zmemridges,
    Zmemvertices,
    Zmergeflipdup,
    Zmergehorizon,
    Zmergeinittot,
    Zmergeinitmax,
    Zmergeinittot2,
    Zmergeintohorizon,
    Zmergenew,
    Zmergesettot,
    Zmergesetmax,
    Zmergesettot2,
    Zmergesimplex,
    Zmergevertex,
    Wmindenom,
    Wminvertex,
    Zminnorm,
    Zmultiridge,
    Znearlysingular,
    Zneighbor,
    Wnewbalance,
    Wnewbalance2,
    Znewfacettot,
    Znewfacetmax,
    Znewvertex,
    Wnewvertex,
    Wnewvertexmax,
    Znoarea,
    Znonsimplicial,
    Znowsimplicial,
    Znotgood,
    Znotgoodnew,
    Znotmax,
    Znumfacets,
    Znummergemax,
    Znummergetot,
    Znumneighbors,
    Znumridges,
    Znumvertices,
    Znumvisibility,
    Znumvneighbors,
    Zonehorizon,
    Zpartangle,
    Zpartcoplanar,
    Zpartflip,
    Zparthorizon,
    Zpartinside,
    Zpartition,
    Zpartitionall,
    Zpartnear,
    Zpbalance,
    Wpbalance,
    Wpbalance2,
    Zpostfacets,
    Zpremergetot,
    Zprocessed,
    Zremvertex,
    Zremvertexdel,
    Zrenameall,
    Zrenamepinch,
    Zrenameshare,
    Zretry,
    Wretrymax,
    Zridge,
    Wridge,
    Wridgemax,
    Zridge0,
    Wridge0,
    Wridge0max,
    Zridgemid,
    Wridgemid,
    Wridgemidmax,
    Zridgeok,
    Wridgeok,
    Wridgeokmax,
    Zsearchpoints,
    Zsetplane,
    Ztestvneighbor,
    Ztotcheck,
    Ztothorizon,
    Ztotmerge,
    Ztotpartcoplanar,
    Ztotpartition,
    Ztotridges,
    Ztotvertices,
    Ztotvisible,
    Ztricoplanar,
    Ztricoplanarmax,
    Ztricoplanartot,
    Ztridegen,
    Ztrimirror,
    Ztrinull,
    Wvertexmax,
    Wvertexmin,
    Zvertexridge,
    Zvertexridgetot,
    Zvertexridgemax,
    Zvertices,
    Zvisfacettot,
    Zvisfacetmax,
    Zvisit,
    Zvisit2max,
    Zvisvertextot,
    Zvisvertexmax,
    Zvvisit,
    Zvvisit2max,
    Zwidefacet,
    Zwidevertices,
    ZEND};

/*-<a                             href="qh-stat.htm#TOC"
  >-------------------------------</a><a name="ZZstat">-</a>

  Zxxx/Wxxx statistics that remain defined if qh_KEEPstatistics=0

  notes:
    be sure to use zzdef, zzinc, etc. with these statistics (no double checking!)
*/
#else
enum statistics {     /* for zzdef etc. macros */
  Zback0,
  Zbestdist,
  Zcentrumtests,
  Zcheckpart,
  Zconcaveridges,
  Zcoplanarhorizon,
  Zcoplanarpart,
  Zcoplanarridges,
  Zcyclefacettot,
  Zcyclehorizon,
  Zdelvertextot,
  Zdistcheck,
  Zdistconvex,
  Zdistzero,
  Zdoc1,
  Zdoc2,
  Zdoc3,
  Zdoc11,
  Zflippedfacets,
  Zgauss0,
  Zminnorm,
  Zmultiridge,
  Znearlysingular,
  Wnewvertexmax,
  Znumvisibility,
  Zpartcoplanar,
  Zpartition,
  Zpartitionall,
  Zprocessed,
  Zretry,
  Zridge,
  Wridge,
  Wridgemax,
  Zridge0,
  Wridge0,
  Wridge0max,
  Zridgemid,
  Wridgemid,
  Wridgemidmax,
  Zridgeok,
  Wridgeok,
  Wridgeokmax,
  Zsetplane,
  Ztotcheck,
  Ztotmerge,
    ZEND};
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >-------------------------------</a><a name="ztype">-</a>

  ztype
    the type of a statistic sets its initial value.

  notes:
    The type should be the same as the macro for collecting the statistic
*/
enum ztypes {zdoc,zinc,zadd,zmax,zmin,ZTYPEreal,wadd,wmax,wmin,ZTYPEend};

/*========== macros and constants =============*/

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="MAYdebugx">-</a>

  MAYdebugx
    define as maydebug() to be called frequently for error trapping
*/
#define MAYdebugx

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="zdef_">-</a>

  zzdef_, zdef_( type, name, doc, -1)
    define a statistic (assumes 'qhstat.next= 0;')

  zdef_( type, name, doc, count)
    define an averaged statistic
    printed as name/count
*/
#define zzdef_(stype,name,string,cnt) qhstat id[qhstat next++]=name; \
   qhstat doc[name]= string; qhstat count[name]= cnt; qhstat type[name]= stype
#if qh_KEEPstatistics
#define zdef_(stype,name,string,cnt) qhstat id[qhstat next++]=name; \
   qhstat doc[name]= string; qhstat count[name]= cnt; qhstat type[name]= stype
#else
#define zdef_(type,name,doc,count)
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="zinc_">-</a>

  zzinc_( name ), zinc_( name)
    increment an integer statistic
*/
#define zzinc_(id) {MAYdebugx; qhstat stats[id].i++;}
#if qh_KEEPstatistics
#define zinc_(id) {MAYdebugx; qhstat stats[id].i++;}
#else
#define zinc_(id) {}
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="zadd_">-</a>

  zzadd_( name, value ), zadd_( name, value ), wadd_( name, value )
    add value to an integer or real statistic
*/
#define zzadd_(id, val) {MAYdebugx; qhstat stats[id].i += (val);}
#define wwadd_(id, val) {MAYdebugx; qhstat stats[id].r += (val);}
#if qh_KEEPstatistics
#define zadd_(id, val) {MAYdebugx; qhstat stats[id].i += (val);}
#define wadd_(id, val) {MAYdebugx; qhstat stats[id].r += (val);}
#else
#define zadd_(id, val) {}
#define wadd_(id, val) {}
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="zval_">-</a>

  zzval_( name ), zval_( name ), wwval_( name )
    set or return value of a statistic
*/
#define zzval_(id) ((qhstat stats[id]).i)
#define wwval_(id) ((qhstat stats[id]).r)
#if qh_KEEPstatistics
#define zval_(id) ((qhstat stats[id]).i)
#define wval_(id) ((qhstat stats[id]).r)
#else
#define zval_(id) qhstat tempi
#define wval_(id) qhstat tempr
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="zmax_">-</a>

  zmax_( id, val ), wmax_( id, value )
    maximize id with val
*/
#define wwmax_(id, val) {MAYdebugx; maximize_(qhstat stats[id].r,(val));}
#if qh_KEEPstatistics
#define zmax_(id, val) {MAYdebugx; maximize_(qhstat stats[id].i,(val));}
#define wmax_(id, val) {MAYdebugx; maximize_(qhstat stats[id].r,(val));}
#else
#define zmax_(id, val) {}
#define wmax_(id, val) {}
#endif

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="zmin_">-</a>

  zmin_( id, val ), wmin_( id, value )
    minimize id with val
*/
#if qh_KEEPstatistics
#define zmin_(id, val) {MAYdebugx; minimize_(qhstat stats[id].i,(val));}
#define wmin_(id, val) {MAYdebugx; minimize_(qhstat stats[id].r,(val));}
#else
#define zmin_(id, val) {}
#define wmin_(id, val) {}
#endif

/*================== stat.h types ==============*/


/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="intrealT">-</a>

  intrealT
    union of integer and real, used for statistics
*/
typedef union intrealT intrealT;    /* union of int and realT */
union intrealT {
    int i;
    realT r;
};

/*-<a                             href="qh-stat.htm#TOC"
  >--------------------------------</a><a name="qhstat">-</a>

  qhstat
    global data structure for statistics, similar to qh and qhrbox

  notes:
   access to qh_qhstat is via the "qhstat" macro.  There are two choices
   qh_QHpointer = 1     access globals via a pointer
                        enables qh_saveqhull() and qh_restoreqhull()
                = 0     qh_qhstat is a static data structure
                        only one instance of qhull() can be active at a time
                        default value
   qh_QHpointer is defined in libqhull.h
   qh_QHpointer_dllimport and qh_dllimport define qh_qh as __declspec(dllimport) [libqhull.h]

   allocated in stat.c using qh_malloc()
*/
#ifndef DEFqhstatT
#define DEFqhstatT 1
typedef struct qhstatT qhstatT;
#endif

#if qh_QHpointer_dllimport
#define qhstat qh_qhstat->
__declspec(dllimport) extern qhstatT *qh_qhstat;
#elif qh_QHpointer
#define qhstat qh_qhstat->
extern qhstatT *qh_qhstat;
#elif qh_dllimport
#define qhstat qh_qhstat.
__declspec(dllimport) extern qhstatT qh_qhstat;
#else
#define qhstat qh_qhstat.
extern qhstatT qh_qhstat;
#endif
struct qhstatT {
  intrealT   stats[ZEND];     /* integer and real statistics */
  unsigned   char id[ZEND+10]; /* id's in print order */
  const char *doc[ZEND];       /* array of documentation strings */
  short int  count[ZEND];     /* -1 if none, else index of count to use */
  char       type[ZEND];      /* type, see ztypes above */
  char       printed[ZEND];   /* true, if statistic has been printed */
  intrealT   init[ZTYPEend];  /* initial values by types, set initstatistics */

  int        next;            /* next index for zdef_ */
  int        precision;       /* index for precision problems */
  int        vridges;         /* index for Voronoi ridges */
  int        tempi;
  realT      tempr;
};

/*========== function prototypes ===========*/

void    qh_allstatA(void);
void    qh_allstatB(void);
void    qh_allstatC(void);
void    qh_allstatD(void);
void    qh_allstatE(void);
void    qh_allstatE2(void);
void    qh_allstatF(void);
void    qh_allstatG(void);
void    qh_allstatH(void);
void    qh_allstatI(void);
void    qh_allstatistics(void);
void    qh_collectstatistics(void);
void    qh_freestatistics(void);
void    qh_initstatistics(void);
boolT   qh_newstats(int idx, int *nextindex);
boolT   qh_nostatistic(int i);
void    qh_printallstatistics(FILE *fp, const char *string);
void    qh_printstatistics(FILE *fp, const char *string);
void    qh_printstatlevel(FILE *fp, int id, int start);
void    qh_printstats(FILE *fp, int idx, int *nextindex);
realT   qh_stddev(int num, realT tot, realT tot2, realT *ave);

#endif   /* qhDEFstat */

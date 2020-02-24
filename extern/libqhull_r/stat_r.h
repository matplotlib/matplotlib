/*<html><pre>  -<a                             href="qh-stat_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   stat_r.h
     contains all statistics that are collected for qhull

   see qh-stat_r.htm and stat_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/stat_r.h#3 $$Change: 2711 $
   $DateTime: 2019/06/27 22:34:56 $$Author: bbarber $

   recompile qhull if you change this file

   Integer statistics are Z* while real statistics are W*.

   define MAYdebugx to call a routine at every statistic event

*/

#ifndef qhDEFstat
#define qhDEFstat 1

/* Depends on realT.  Do not include "libqhull_r" to avoid circular dependency */

#ifndef DEFqhT
#define DEFqhT 1
typedef struct qhT qhT;         /* Defined by libqhull_r.h */
#endif

#ifndef DEFqhstatT
#define DEFqhstatT 1
typedef struct qhstatT qhstatT; /* Defined here */
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="KEEPstatistics">-</a>

  qh_KEEPstatistics
    0 turns off statistic reporting and gathering (except zzdef/zzinc/zzadd/zzval/wwval)

  set qh_KEEPstatistics in user_r.h to 0 to turn off statistics
*/
#ifndef qh_KEEPstatistics
#define qh_KEEPstatistics 1
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="statistics">-</a>

  Zxxx for integers, Wxxx for reals

  notes:
    be sure that all statistics are defined in stat_r.c
      otherwise initialization may core dump
    can pick up all statistics by:
      grep '[zw].*_[(][ZW]' *.c >z.x
    remove trailers with query">-</a>
    remove leaders with  query-replace-regexp [ ^I]+  (
*/
#if qh_KEEPstatistics
enum qh_statistics {     /* alphabetical after Z/W */
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
    Zbestlowerall,
    Zbestloweralln,
    Zbestlowerv,
    Zcentrumtests,
    Zcheckpart,
    Zcomputefurthest,
    Zconcave,
    Wconcavemax,
    Wconcavetot,
    Zconcavecoplanar,
    Wconcavecoplanarmax,
    Wconcavecoplanartot,
    Zconcavecoplanarridge,
    Zconcaveridge,
    Zconcaveridges,
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
    Zdetfacetarea,
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
    Zdupsame,
    Zflipped,
    Wflippedmax,
    Wflippedtot,
    Zflippedfacets,
    Zflipridge,
    Zflipridge2,
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
    Zmergeintocoplanar,
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
    Zredundant,
    Wnewbalance,
    Wnewbalance2,
    Znewbesthorizon,
    Znewfacettot,
    Znewfacetmax,
    Znewvertex,
    Wnewvertex,
    Wnewvertexmax,
    Znewvertexridge,
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
    Zpartcorner,
    Zparthidden,
    Zpartinside,
    Zpartition,
    Zpartitionall,
    Zpartnear,
    Zparttwisted,
    Zpbalance,
    Wpbalance,
    Wpbalance2,
    Zpinchduplicate,
    Zpinchedapex,
    Zpinchedvertex,
    Zpostfacets,
    Zpremergetot,
    Zprocessed,
    Zremvertex,
    Zremvertexdel,
    Zredundantmerge,
    Zrenameall,
    Zrenamepinch,
    Zrenameshare,
    Zretry,
    Wretrymax,
    Zretryadd,
    Zretryaddmax,
    Zretryaddtot,
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
    Ztwisted,
    Wtwistedtot,
    Wtwistedmax,
    Ztwistedridge,
    Zvertextests,
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

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="ZZstat">-</a>

  Zxxx/Wxxx statistics that remain defined if qh_KEEPstatistics=0

  notes:
    be sure to use zzdef, zzinc, etc. with these statistics (no double checking!)
*/
#else
enum qh_statistics {     /* for zzdef etc. macros */
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
  Zdistplane,
  Zdistzero,
  Zdoc1,
  Zdoc2,
  Zdoc3,
  Zdoc11,
  Zflippedfacets,
  Zflipridge,
  Zflipridge2,
  Zgauss0,
  Zminnorm,
  Zmultiridge,
  Znearlysingular,
  Wnewvertexmax,
  Znumvisibility,
  Zpartcoplanar,
  Zpartition,
  Zpartitionall,
  Zpinchduplicate,
  Zpinchedvertex,
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
  Zvertextests,
  ZEND};
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="ztype">-</a>

  ztype
    the type of a statistic sets its initial value.

  notes:
    The type should be the same as the macro for collecting the statistic
*/
enum ztypes {zdoc,zinc,zadd,zmax,zmin,ZTYPEreal,wadd,wmax,wmin,ZTYPEend};

/*========== macros and constants =============*/

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="MAYdebugx">-</a>

  MAYdebugx
    define as maydebug() to be called frequently for error trapping
*/
#define MAYdebugx

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zdef_">-</a>

  zzdef_, zdef_( type, name, doc, -1)
    define a statistic (assumes 'qhstat.next= 0;')

  zdef_( type, name, doc, count)
    define an averaged statistic
    printed as name/count
*/
#define zzdef_(stype,name,string,cnt) qh->qhstat.id[qh->qhstat.next++]=name; \
   qh->qhstat.doc[name]= string; qh->qhstat.count[name]= cnt; qh->qhstat.type[name]= stype
#if qh_KEEPstatistics
#define zdef_(stype,name,string,cnt) qh->qhstat.id[qh->qhstat.next++]=name; \
   qh->qhstat.doc[name]= string; qh->qhstat.count[name]= cnt; qh->qhstat.type[name]= stype
#else
#define zdef_(type,name,doc,count)
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zinc_">-</a>

  zzinc_( name ), zinc_( name)
    increment an integer statistic
*/
#define zzinc_(id) {MAYdebugx; qh->qhstat.stats[id].i++;}
#if qh_KEEPstatistics
#define zinc_(id) {MAYdebugx; qh->qhstat.stats[id].i++;}
#else
#define zinc_(id) {}
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zadd_">-</a>

  zzadd_( name, value ), zadd_( name, value ), wadd_( name, value )
    add value to an integer or real statistic
*/
#define zzadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].i += (val);}
#define wwadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].r += (val);}
#if qh_KEEPstatistics
#define zadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].i += (val);}
#define wadd_(id, val) {MAYdebugx; qh->qhstat.stats[id].r += (val);}
#else
#define zadd_(id, val) {}
#define wadd_(id, val) {}
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zval_">-</a>

  zzval_( name ), zval_( name ), wwval_( name )
    set or return value of a statistic
*/
#define zzval_(id) ((qh->qhstat.stats[id]).i)
#define wwval_(id) ((qh->qhstat.stats[id]).r)
#if qh_KEEPstatistics
#define zval_(id) ((qh->qhstat.stats[id]).i)
#define wval_(id) ((qh->qhstat.stats[id]).r)
#else
#define zval_(id) qh->qhstat.tempi
#define wval_(id) qh->qhstat.tempr
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zmax_">-</a>

  zmax_( id, val ), wmax_( id, value )
    maximize id with val
*/
#define wwmax_(id, val) {MAYdebugx; maximize_(qh->qhstat.stats[id].r,(val));}
#if qh_KEEPstatistics
#define zmax_(id, val) {MAYdebugx; maximize_(qh->qhstat.stats[id].i,(val));}
#define wmax_(id, val) {MAYdebugx; maximize_(qh->qhstat.stats[id].r,(val));}
#else
#define zmax_(id, val) {}
#define wmax_(id, val) {}
#endif

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="zmin_">-</a>

  zmin_( id, val ), wmin_( id, value )
    minimize id with val
*/
#if qh_KEEPstatistics
#define zmin_(id, val) {MAYdebugx; minimize_(qh->qhstat.stats[id].i,(val));}
#define wmin_(id, val) {MAYdebugx; minimize_(qh->qhstat.stats[id].r,(val));}
#else
#define zmin_(id, val) {}
#define wmin_(id, val) {}
#endif

/*================== stat_r.h types ==============*/


/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="intrealT">-</a>

  intrealT
    union of integer and real, used for statistics
*/
typedef union intrealT intrealT;    /* union of int and realT */
union intrealT {
    int i;
    realT r;
};

/*-<a                             href="qh-stat_r.htm#TOC"
  >--------------------------------</a><a name="qhstat">-</a>

  qhstat
    Data structure for statistics, similar to qh and qhrbox

    Allocated as part of qhT (libqhull_r.h)
*/

struct qhstatT {
  intrealT   stats[ZEND];     /* integer and real statistics */
  unsigned char id[ZEND+10];  /* id's in print order */
  const char *doc[ZEND];      /* array of documentation strings */
  short int  count[ZEND];     /* -1 if none, else index of count to use */
  char       type[ZEND];      /* type, see ztypes above */
  char       printed[ZEND];   /* true, if statistic has been printed */
  intrealT   init[ZTYPEend];  /* initial values by types, set initstatistics */

  int        next;            /* next index for zdef_ */
  int        precision;       /* index for precision problems, printed on qh_errexit and qh_produce_output2/Q0/QJn */
  int        vridges;         /* index for Voronoi ridges, printed on qh_produce_output2 */
  int        tempi;
  realT      tempr;
};

/*========== function prototypes ===========*/

#ifdef __cplusplus
extern "C" {
#endif

void    qh_allstatA(qhT *qh);
void    qh_allstatB(qhT *qh);
void    qh_allstatC(qhT *qh);
void    qh_allstatD(qhT *qh);
void    qh_allstatE(qhT *qh);
void    qh_allstatE2(qhT *qh);
void    qh_allstatF(qhT *qh);
void    qh_allstatG(qhT *qh);
void    qh_allstatH(qhT *qh);
void    qh_allstatI(qhT *qh);
void    qh_allstatistics(qhT *qh);
void    qh_collectstatistics(qhT *qh);
void    qh_initstatistics(qhT *qh);
boolT   qh_newstats(qhT *qh, int idx, int *nextindex);
boolT   qh_nostatistic(qhT *qh, int i);
void    qh_printallstatistics(qhT *qh, FILE *fp, const char *string);
void    qh_printstatistics(qhT *qh, FILE *fp, const char *string);
void    qh_printstatlevel(qhT *qh, FILE *fp, int id);
void    qh_printstats(qhT *qh, FILE *fp, int idx, int *nextindex);
realT   qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif   /* qhDEFstat */

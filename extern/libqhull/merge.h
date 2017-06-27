/*<html><pre>  -<a                             href="qh-merge.htm"
  >-------------------------------</a><a name="TOP">-</a>

   merge.h
   header file for merge.c

   see qh-merge.htm and merge.c

   Copyright (c) 1993-2015 C.B. Barber.
   $Id: //main/2015/qhull/src/libqhull/merge.h#1 $$Change: 1981 $
   $DateTime: 2015/09/28 20:26:32 $$Author: bbarber $
*/

#ifndef qhDEFmerge
#define qhDEFmerge 1

#include "libqhull.h"


/*============ -constants- ==============*/

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="qh_ANGLEredundant">-</a>

  qh_ANGLEredundant
    indicates redundant merge in mergeT->angle
*/
#define qh_ANGLEredundant 6.0

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="qh_ANGLEdegen">-</a>

  qh_ANGLEdegen
    indicates degenerate facet in mergeT->angle
*/
#define qh_ANGLEdegen     5.0

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="qh_ANGLEconcave">-</a>

  qh_ANGLEconcave
    offset to indicate concave facets in mergeT->angle

  notes:
    concave facets are assigned the range of [2,4] in mergeT->angle
    roundoff error may make the angle less than 2
*/
#define qh_ANGLEconcave  1.5

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="MRG">-</a>

  MRG... (mergeType)
    indicates the type of a merge (mergeT->type)
*/
typedef enum {  /* in sort order for facet_mergeset */
  MRGnone= 0,
  MRGcoplanar,          /* centrum coplanar */
  MRGanglecoplanar,     /* angle coplanar */
                        /* could detect half concave ridges */
  MRGconcave,           /* concave ridge */
  MRGflip,              /* flipped facet. facet1 == facet2 */
  MRGridge,             /* duplicate ridge (qh_MERGEridge) */
                        /* degen and redundant go onto degen_mergeset */
  MRGdegen,             /* degenerate facet (!enough neighbors) facet1 == facet2 */
  MRGredundant,         /* redundant facet (vertex subset) */
                        /* merge_degenredundant assumes degen < redundant */
  MRGmirror,            /* mirror facet from qh_triangulate */
  ENDmrg
} mergeType;

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="qh_MERGEapex">-</a>

  qh_MERGEapex
    flag for qh_mergefacet() to indicate an apex merge
*/
#define qh_MERGEapex     True

/*============ -structures- ====================*/

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="mergeT">-</a>

  mergeT
    structure used to merge facets
*/

typedef struct mergeT mergeT;
struct mergeT {         /* initialize in qh_appendmergeset */
  realT   angle;        /* angle between normals of facet1 and facet2 */
  facetT *facet1;       /* will merge facet1 into facet2 */
  facetT *facet2;
  mergeType type;
};


/*=========== -macros- =========================*/

/*-<a                             href="qh-merge.htm#TOC"
  >--------------------------------</a><a name="FOREACHmerge_">-</a>

  FOREACHmerge_( merges ) {...}
    assign 'merge' to each merge in merges

  notes:
    uses 'mergeT *merge, **mergep;'
    if qh_mergefacet(),
      restart since qh.facet_mergeset may change
    see <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHmerge_( merges ) FOREACHsetelement_(mergeT, merges, merge)

/*============ prototypes in alphabetical order after pre/postmerge =======*/

void    qh_premerge(vertexT *apex, realT maxcentrum, realT maxangle);
void    qh_postmerge(const char *reason, realT maxcentrum, realT maxangle,
             boolT vneighbors);
void    qh_all_merges(boolT othermerge, boolT vneighbors);
void    qh_appendmergeset(facetT *facet, facetT *neighbor, mergeType mergetype, realT *angle);
setT   *qh_basevertices( facetT *samecycle);
void    qh_checkconnect(void /* qh.new_facets */);
boolT   qh_checkzero(boolT testall);
int     qh_compareangle(const void *p1, const void *p2);
int     qh_comparemerge(const void *p1, const void *p2);
int     qh_comparevisit(const void *p1, const void *p2);
void    qh_copynonconvex(ridgeT *atridge);
void    qh_degen_redundant_facet(facetT *facet);
void    qh_degen_redundant_neighbors(facetT *facet, facetT *delfacet);
vertexT *qh_find_newvertex(vertexT *oldvertex, setT *vertices, setT *ridges);
void    qh_findbest_test(boolT testcentrum, facetT *facet, facetT *neighbor,
           facetT **bestfacet, realT *distp, realT *mindistp, realT *maxdistp);
facetT *qh_findbestneighbor(facetT *facet, realT *distp, realT *mindistp, realT *maxdistp);
void    qh_flippedmerges(facetT *facetlist, boolT *wasmerge);
void    qh_forcedmerges( boolT *wasmerge);
void    qh_getmergeset(facetT *facetlist);
void    qh_getmergeset_initial(facetT *facetlist);
void    qh_hashridge(setT *hashtable, int hashsize, ridgeT *ridge, vertexT *oldvertex);
ridgeT *qh_hashridge_find(setT *hashtable, int hashsize, ridgeT *ridge,
              vertexT *vertex, vertexT *oldvertex, int *hashslot);
void    qh_makeridges(facetT *facet);
void    qh_mark_dupridges(facetT *facetlist);
void    qh_maydropneighbor(facetT *facet);
int     qh_merge_degenredundant(void);
void    qh_merge_nonconvex( facetT *facet1, facetT *facet2, mergeType mergetype);
void    qh_mergecycle(facetT *samecycle, facetT *newfacet);
void    qh_mergecycle_all(facetT *facetlist, boolT *wasmerge);
void    qh_mergecycle_facets( facetT *samecycle, facetT *newfacet);
void    qh_mergecycle_neighbors(facetT *samecycle, facetT *newfacet);
void    qh_mergecycle_ridges(facetT *samecycle, facetT *newfacet);
void    qh_mergecycle_vneighbors( facetT *samecycle, facetT *newfacet);
void    qh_mergefacet(facetT *facet1, facetT *facet2, realT *mindist, realT *maxdist, boolT mergeapex);
void    qh_mergefacet2d(facetT *facet1, facetT *facet2);
void    qh_mergeneighbors(facetT *facet1, facetT *facet2);
void    qh_mergeridges(facetT *facet1, facetT *facet2);
void    qh_mergesimplex(facetT *facet1, facetT *facet2, boolT mergeapex);
void    qh_mergevertex_del(vertexT *vertex, facetT *facet1, facetT *facet2);
void    qh_mergevertex_neighbors(facetT *facet1, facetT *facet2);
void    qh_mergevertices(setT *vertices1, setT **vertices);
setT   *qh_neighbor_intersections(vertexT *vertex);
void    qh_newvertices(setT *vertices);
boolT   qh_reducevertices(void);
vertexT *qh_redundant_vertex(vertexT *vertex);
boolT   qh_remove_extravertices(facetT *facet);
vertexT *qh_rename_sharedvertex(vertexT *vertex, facetT *facet);
void    qh_renameridgevertex(ridgeT *ridge, vertexT *oldvertex, vertexT *newvertex);
void    qh_renamevertex(vertexT *oldvertex, vertexT *newvertex, setT *ridges,
                        facetT *oldfacet, facetT *neighborA);
boolT   qh_test_appendmerge(facetT *facet, facetT *neighbor);
boolT   qh_test_vneighbors(void /* qh.newfacet_list */);
void    qh_tracemerge(facetT *facet1, facetT *facet2);
void    qh_tracemerging(void);
void    qh_updatetested( facetT *facet1, facetT *facet2);
setT   *qh_vertexridges(vertexT *vertex);
void    qh_vertexridges_facet(vertexT *vertex, facetT *facet, setT **ridges);
void    qh_willdelete(facetT *facet, facetT *replace);

#endif /* qhDEFmerge */

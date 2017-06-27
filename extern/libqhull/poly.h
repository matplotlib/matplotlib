/*<html><pre>  -<a                             href="qh-poly.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly.h
   header file for poly.c and poly2.c

   see qh-poly.htm, libqhull.h and poly.c

   Copyright (c) 1993-2015 The Geometry Center.
   $Id: //main/2015/qhull/src/libqhull/poly.h#3 $$Change: 2047 $
   $DateTime: 2016/01/04 22:03:18 $$Author: bbarber $
*/

#ifndef qhDEFpoly
#define qhDEFpoly 1

#include "libqhull.h"

/*===============   constants ========================== */

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="ALGORITHMfault">-</a>

  ALGORITHMfault
    use as argument to checkconvex() to report errors during buildhull
*/
#define qh_ALGORITHMfault 0

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="DATAfault">-</a>

  DATAfault
    use as argument to checkconvex() to report errors during initialhull
*/
#define qh_DATAfault 1

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="DUPLICATEridge">-</a>

  DUPLICATEridge
    special value for facet->neighbor to indicate a duplicate ridge

  notes:
    set by matchneighbor, used by matchmatch and mark_dupridge
*/
#define qh_DUPLICATEridge (facetT *)1L

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="MERGEridge">-</a>

  MERGEridge       flag in facet
    special value for facet->neighbor to indicate a merged ridge

  notes:
    set by matchneighbor, used by matchmatch and mark_dupridge
*/
#define qh_MERGEridge (facetT *)2L


/*============ -structures- ====================*/

/*=========== -macros- =========================*/

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLfacet_">-</a>

  FORALLfacet_( facetlist ) { ... }
    assign 'facet' to each facet in facetlist

  notes:
    uses 'facetT *facet;'
    assumes last facet is a sentinel

  see:
    FORALLfacets
*/
#define FORALLfacet_( facetlist ) if (facetlist ) for ( facet=( facetlist ); facet && facet->next; facet= facet->next )

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLnew_facets">-</a>

  FORALLnew_facets { ... }
    assign 'newfacet' to each facet in qh.newfacet_list

  notes:
    uses 'facetT *newfacet;'
    at exit, newfacet==NULL
*/
#define FORALLnew_facets for ( newfacet=qh newfacet_list;newfacet && newfacet->next;newfacet=newfacet->next )

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLvertex_">-</a>

  FORALLvertex_( vertexlist ) { ... }
    assign 'vertex' to each vertex in vertexlist

  notes:
    uses 'vertexT *vertex;'
    at exit, vertex==NULL
*/
#define FORALLvertex_( vertexlist ) for (vertex=( vertexlist );vertex && vertex->next;vertex= vertex->next )

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLvisible_facets">-</a>

  FORALLvisible_facets { ... }
    assign 'visible' to each visible facet in qh.visible_list

  notes:
    uses 'vacetT *visible;'
    at exit, visible==NULL
*/
#define FORALLvisible_facets for (visible=qh visible_list; visible && visible->visible; visible= visible->next)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLsame_">-</a>

  FORALLsame_( newfacet ) { ... }
    assign 'same' to each facet in newfacet->f.samecycle

  notes:
    uses 'facetT *same;'
    stops when it returns to newfacet
*/
#define FORALLsame_(newfacet) for (same= newfacet->f.samecycle; same != newfacet; same= same->f.samecycle)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FORALLsame_cycle_">-</a>

  FORALLsame_cycle_( newfacet ) { ... }
    assign 'same' to each facet in newfacet->f.samecycle

  notes:
    uses 'facetT *same;'
    at exit, same == NULL
*/
#define FORALLsame_cycle_(newfacet) \
     for (same= newfacet->f.samecycle; \
         same; same= (same == newfacet ?  NULL : same->f.samecycle))

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighborA_">-</a>

  FOREACHneighborA_( facet ) { ... }
    assign 'neighborA' to each neighbor in facet->neighbors

  FOREACHneighborA_( vertex ) { ... }
    assign 'neighborA' to each neighbor in vertex->neighbors

  declare:
    facetT *neighborA, **neighborAp;

  see:
    <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHneighborA_(facet)  FOREACHsetelement_(facetT, facet->neighbors, neighborA)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHvisible_">-</a>

  FOREACHvisible_( facets ) { ... }
    assign 'visible' to each facet in facets

  notes:
    uses 'facetT *facet, *facetp;'
    see <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvisible_(facets) FOREACHsetelement_(facetT, facets, visible)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHnewfacet_">-</a>

  FOREACHnewfacet_( facets ) { ... }
    assign 'newfacet' to each facet in facets

  notes:
    uses 'facetT *newfacet, *newfacetp;'
    see <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHnewfacet_(facets) FOREACHsetelement_(facetT, facets, newfacet)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertexA_">-</a>

  FOREACHvertexA_( vertices ) { ... }
    assign 'vertexA' to each vertex in vertices

  notes:
    uses 'vertexT *vertexA, *vertexAp;'
    see <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertexA_(vertices) FOREACHsetelement_(vertexT, vertices, vertexA)

/*-<a                             href="qh-poly.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertexreverse12_">-</a>

  FOREACHvertexreverse12_( vertices ) { ... }
    assign 'vertex' to each vertex in vertices
    reverse order of first two vertices

  notes:
    uses 'vertexT *vertex, *vertexp;'
    see <a href="qset.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertexreverse12_(vertices) FOREACHsetelementreverse12_(vertexT, vertices, vertex)


/*=============== prototypes poly.c in alphabetical order ================*/

void    qh_appendfacet(facetT *facet);
void    qh_appendvertex(vertexT *vertex);
void    qh_attachnewfacets(void /* qh.visible_list, qh.newfacet_list */);
boolT   qh_checkflipped(facetT *facet, realT *dist, boolT allerror);
void    qh_delfacet(facetT *facet);
void    qh_deletevisible(void /*qh.visible_list, qh.horizon_list*/);
setT   *qh_facetintersect(facetT *facetA, facetT *facetB, int *skipAp,int *skipBp, int extra);
int     qh_gethash(int hashsize, setT *set, int size, int firstindex, void *skipelem);
facetT *qh_makenewfacet(setT *vertices, boolT toporient, facetT *facet);
void    qh_makenewplanes(void /* newfacet_list */);
facetT *qh_makenew_nonsimplicial(facetT *visible, vertexT *apex, int *numnew);
facetT *qh_makenew_simplicial(facetT *visible, vertexT *apex, int *numnew);
void    qh_matchneighbor(facetT *newfacet, int newskip, int hashsize,
                          int *hashcount);
void    qh_matchnewfacets(void);
boolT   qh_matchvertices(int firstindex, setT *verticesA, int skipA,
                          setT *verticesB, int *skipB, boolT *same);
facetT *qh_newfacet(void);
ridgeT *qh_newridge(void);
int     qh_pointid(pointT *point);
void    qh_removefacet(facetT *facet);
void    qh_removevertex(vertexT *vertex);
void    qh_updatevertices(void);


/*========== -prototypes poly2.c in alphabetical order ===========*/

void    qh_addhash(void* newelem, setT *hashtable, int hashsize, int hash);
void    qh_check_bestdist(void);
void    qh_check_dupridge(facetT *facet1, realT dist1, facetT *facet2, realT dist2);
void    qh_check_maxout(void);
void    qh_check_output(void);
void    qh_check_point(pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2);
void    qh_check_points(void);
void    qh_checkconvex(facetT *facetlist, int fault);
void    qh_checkfacet(facetT *facet, boolT newmerge, boolT *waserrorp);
void    qh_checkflipped_all(facetT *facetlist);
void    qh_checkpolygon(facetT *facetlist);
void    qh_checkvertex(vertexT *vertex);
void    qh_clearcenters(qh_CENTER type);
void    qh_createsimplex(setT *vertices);
void    qh_delridge(ridgeT *ridge);
void    qh_delvertex(vertexT *vertex);
setT   *qh_facet3vertex(facetT *facet);
facetT *qh_findbestfacet(pointT *point, boolT bestoutside,
           realT *bestdist, boolT *isoutside);
facetT *qh_findbestlower(facetT *upperfacet, pointT *point, realT *bestdistp, int *numpart);
facetT *qh_findfacet_all(pointT *point, realT *bestdist, boolT *isoutside,
                          int *numpart);
int     qh_findgood(facetT *facetlist, int goodhorizon);
void    qh_findgood_all(facetT *facetlist);
void    qh_furthestnext(void /* qh.facet_list */);
void    qh_furthestout(facetT *facet);
void    qh_infiniteloop(facetT *facet);
void    qh_initbuild(void);
void    qh_initialhull(setT *vertices);
setT   *qh_initialvertices(int dim, setT *maxpoints, pointT *points, int numpoints);
vertexT *qh_isvertex(pointT *point, setT *vertices);
vertexT *qh_makenewfacets(pointT *point /*horizon_list, visible_list*/);
void    qh_matchduplicates(facetT *atfacet, int atskip, int hashsize, int *hashcount);
void    qh_nearcoplanar(void /* qh.facet_list */);
vertexT *qh_nearvertex(facetT *facet, pointT *point, realT *bestdistp);
int     qh_newhashtable(int newsize);
vertexT *qh_newvertex(pointT *point);
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp);
void    qh_outcoplanar(void /* facet_list */);
pointT *qh_point(int id);
void    qh_point_add(setT *set, pointT *point, void *elem);
setT   *qh_pointfacet(void /*qh.facet_list*/);
setT   *qh_pointvertex(void /*qh.facet_list*/);
void    qh_prependfacet(facetT *facet, facetT **facetlist);
void    qh_printhashtable(FILE *fp);
void    qh_printlists(void);
void    qh_resetlists(boolT stats, boolT resetVisible /*qh.newvertex_list qh.newfacet_list qh.visible_list*/);
void    qh_setvoronoi_all(void);
void    qh_triangulate(void /*qh.facet_list*/);
void    qh_triangulate_facet(facetT *facetA, vertexT **first_vertex);
void    qh_triangulate_link(facetT *oldfacetA, facetT *facetA, facetT *oldfacetB, facetT *facetB);
void    qh_triangulate_mirror(facetT *facetA, facetT *facetB);
void    qh_triangulate_null(facetT *facetA);
void    qh_vertexintersect(setT **vertexsetA,setT *vertexsetB);
setT   *qh_vertexintersect_new(setT *vertexsetA,setT *vertexsetB);
void    qh_vertexneighbors(void /*qh.facet_list*/);
boolT   qh_vertexsubset(setT *vertexsetA, setT *vertexsetB);


#endif /* qhDEFpoly */

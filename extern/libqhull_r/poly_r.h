/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly_r.h
   header file for poly_r.c and poly2_r.c

   see qh-poly_r.htm, libqhull_r.h and poly_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/poly_r.h#3 $$Change: 2701 $
   $DateTime: 2019/06/25 15:24:47 $$Author: bbarber $
*/

#ifndef qhDEFpoly
#define qhDEFpoly 1

#include "libqhull_r.h"

/*===============   constants ========================== */

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="ALGORITHMfault">-</a>

  qh_ALGORITHMfault
    use as argument to checkconvex() to report errors during buildhull
*/
#define qh_ALGORITHMfault 0

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="DATAfault">-</a>

  qh_DATAfault
    use as argument to checkconvex() to report errors during initialhull
*/
#define qh_DATAfault 1

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="DUPLICATEridge">-</a>

  qh_DUPLICATEridge
    special value for facet->neighbor to indicate a duplicate ridge

  notes:
    set by qh_matchneighbor for qh_matchdupridge
*/
#define qh_DUPLICATEridge (facetT *)1L

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="MERGEridge">-</a>

  qh_MERGEridge       flag in facet
    special value for facet->neighbor to indicate a duplicate ridge that needs merging

  notes:
    set by qh_matchnewfacets..qh_matchdupridge from qh_DUPLICATEridge
    used by qh_mark_dupridges to set facet->mergeridge, facet->mergeridge2 from facet->dupridge
*/
#define qh_MERGEridge (facetT *)2L


/*============ -structures- ====================*/

/*=========== -macros- =========================*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLfacet_">-</a>

  FORALLfacet_( facetlist ) { ... }
    assign 'facet' to each facet in facetlist

  notes:
    uses 'facetT *facet;'
    assumes last facet is a sentinel

  see:
    FORALLfacets
*/
#define FORALLfacet_( facetlist ) if (facetlist) for ( facet=(facetlist); facet && facet->next; facet= facet->next )

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLnew_facets">-</a>

  FORALLnew_facets { ... }
    assign 'newfacet' to each facet in qh.newfacet_list

  notes:
    uses 'facetT *newfacet;'
    at exit, newfacet==NULL
*/
#define FORALLnew_facets for ( newfacet=qh->newfacet_list; newfacet && newfacet->next; newfacet=newfacet->next )

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLvertex_">-</a>

  FORALLvertex_( vertexlist ) { ... }
    assign 'vertex' to each vertex in vertexlist

  notes:
    uses 'vertexT *vertex;'
    at exit, vertex==NULL
*/
#define FORALLvertex_( vertexlist ) for (vertex=( vertexlist );vertex && vertex->next;vertex= vertex->next )

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLvisible_facets">-</a>

  FORALLvisible_facets { ... }
    assign 'visible' to each visible facet in qh.visible_list

  notes:
    uses 'vacetT *visible;'
    at exit, visible==NULL
*/
#define FORALLvisible_facets for (visible=qh->visible_list; visible && visible->visible; visible= visible->next)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FORALLsame_">-</a>

  FORALLsame_( newfacet ) { ... }
    assign 'same' to each facet in newfacet->f.samecycle

  notes:
    uses 'facetT *same;'
    stops when it returns to newfacet
*/
#define FORALLsame_(newfacet) for (same= newfacet->f.samecycle; same != newfacet; same= same->f.samecycle)

/*-<a                             href="qh-poly_r.htm#TOC"
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

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHneighborA_">-</a>

  FOREACHneighborA_( facet ) { ... }
    assign 'neighborA' to each neighbor in facet->neighbors

  FOREACHneighborA_( vertex ) { ... }
    assign 'neighborA' to each neighbor in vertex->neighbors

  declare:
    facetT *neighborA, **neighborAp;

  see:
    <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHneighborA_(facet)  FOREACHsetelement_(facetT, facet->neighbors, neighborA)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvisible_">-</a>

  FOREACHvisible_( facets ) { ... }
    assign 'visible' to each facet in facets

  notes:
    uses 'facetT *facet, *facetp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvisible_(facets) FOREACHsetelement_(facetT, facets, visible)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHnewfacet_">-</a>

  FOREACHnewfacet_( facets ) { ... }
    assign 'newfacet' to each facet in facets

  notes:
    uses 'facetT *newfacet, *newfacetp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHnewfacet_(facets) FOREACHsetelement_(facetT, facets, newfacet)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertexA_">-</a>

  FOREACHvertexA_( vertices ) { ... }
    assign 'vertexA' to each vertex in vertices

  notes:
    uses 'vertexT *vertexA, *vertexAp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertexA_(vertices) FOREACHsetelement_(vertexT, vertices, vertexA)

/*-<a                             href="qh-poly_r.htm#TOC"
  >--------------------------------</a><a name="FOREACHvertexreverse12_">-</a>

  FOREACHvertexreverse12_( vertices ) { ... }
    assign 'vertex' to each vertex in vertices
    reverse order of first two vertices

  notes:
    uses 'vertexT *vertex, *vertexp;'
    see <a href="qset_r.h#FOREACHsetelement_">FOREACHsetelement_</a>
*/
#define FOREACHvertexreverse12_(vertices) FOREACHsetelementreverse12_(vertexT, vertices, vertex)


/*=============== prototypes poly_r.c in alphabetical order ================*/

#ifdef __cplusplus
extern "C" {
#endif

void    qh_appendfacet(qhT *qh, facetT *facet);
void    qh_appendvertex(qhT *qh, vertexT *vertex);
void    qh_attachnewfacets(qhT *qh /* qh.visible_list, qh.newfacet_list */);
boolT   qh_checkflipped(qhT *qh, facetT *facet, realT *dist, boolT allerror);
void    qh_delfacet(qhT *qh, facetT *facet);
void    qh_deletevisible(qhT *qh /* qh.visible_list, qh.horizon_list */);
setT   *qh_facetintersect(qhT *qh, facetT *facetA, facetT *facetB, int *skipAp,int *skipBp, int extra);
int     qh_gethash(qhT *qh, int hashsize, setT *set, int size, int firstindex, void *skipelem);
facetT *qh_getreplacement(qhT *qh, facetT *visible);
facetT *qh_makenewfacet(qhT *qh, setT *vertices, boolT toporient, facetT *facet);
void    qh_makenewplanes(qhT *qh /* qh.newfacet_list */);
facetT *qh_makenew_nonsimplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew);
facetT *qh_makenew_simplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew);
void    qh_matchneighbor(qhT *qh, facetT *newfacet, int newskip, int hashsize,
                          int *hashcount);
coordT  qh_matchnewfacets(qhT *qh);
boolT   qh_matchvertices(qhT *qh, int firstindex, setT *verticesA, int skipA,
                          setT *verticesB, int *skipB, boolT *same);
facetT *qh_newfacet(qhT *qh);
ridgeT *qh_newridge(qhT *qh);
int     qh_pointid(qhT *qh, pointT *point);
void    qh_removefacet(qhT *qh, facetT *facet);
void    qh_removevertex(qhT *qh, vertexT *vertex);
void    qh_update_vertexneighbors(qhT *qh);
void    qh_update_vertexneighbors_cone(qhT *qh);


/*========== -prototypes poly2_r.c in alphabetical order ===========*/

boolT   qh_addfacetvertex(qhT *qh, facetT *facet, vertexT *newvertex);
void    qh_addhash(void *newelem, setT *hashtable, int hashsize, int hash);
void    qh_check_bestdist(qhT *qh);
void    qh_check_maxout(qhT *qh);
void    qh_check_output(qhT *qh);
void    qh_check_point(qhT *qh, pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2, int *errcount);
void    qh_check_points(qhT *qh);
void    qh_checkconvex(qhT *qh, facetT *facetlist, int fault);
void    qh_checkfacet(qhT *qh, facetT *facet, boolT newmerge, boolT *waserrorp);
void    qh_checkflipped_all(qhT *qh, facetT *facetlist);
boolT   qh_checklists(qhT *qh, facetT *facetlist);
void    qh_checkpolygon(qhT *qh, facetT *facetlist);
void    qh_checkvertex(qhT *qh, vertexT *vertex, boolT allchecks, boolT *waserrorp);
void    qh_clearcenters(qhT *qh, qh_CENTER type);
void    qh_createsimplex(qhT *qh, setT *vertices);
void    qh_delridge(qhT *qh, ridgeT *ridge);
void    qh_delvertex(qhT *qh, vertexT *vertex);
setT   *qh_facet3vertex(qhT *qh, facetT *facet);
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside,
           realT *bestdist, boolT *isoutside);
facetT *qh_findbestlower(qhT *qh, facetT *upperfacet, pointT *point, realT *bestdistp, int *numpart);
facetT *qh_findfacet_all(qhT *qh, pointT *point, boolT noupper, realT *bestdist, boolT *isoutside,
                          int *numpart);
int     qh_findgood(qhT *qh, facetT *facetlist, int goodhorizon);
void    qh_findgood_all(qhT *qh, facetT *facetlist);
void    qh_furthestnext(qhT *qh /* qh.facet_list */);
void    qh_furthestout(qhT *qh, facetT *facet);
void    qh_infiniteloop(qhT *qh, facetT *facet);
void    qh_initbuild(qhT *qh);
void    qh_initialhull(qhT *qh, setT *vertices);
setT   *qh_initialvertices(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints);
vertexT *qh_isvertex(pointT *point, setT *vertices);
vertexT *qh_makenewfacets(qhT *qh, pointT *point /* qh.horizon_list, visible_list */);
coordT  qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount);
void    qh_nearcoplanar(qhT *qh /* qh.facet_list */);
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp);
int     qh_newhashtable(qhT *qh, int newsize);
vertexT *qh_newvertex(qhT *qh, pointT *point);
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp);
vertexT *qh_opposite_vertex(qhT *qh, facetT *facetA,  facetT *neighbor);
void    qh_outcoplanar(qhT *qh /* qh.facet_list */);
pointT *qh_point(qhT *qh, int id);
void    qh_point_add(qhT *qh, setT *set, pointT *point, void *elem);
setT   *qh_pointfacet(qhT *qh /* qh.facet_list */);
setT   *qh_pointvertex(qhT *qh /* qh.facet_list */);
void    qh_prependfacet(qhT *qh, facetT *facet, facetT **facetlist);
void    qh_printhashtable(qhT *qh, FILE *fp);
void    qh_printlists(qhT *qh);
void    qh_replacefacetvertex(qhT *qh, facetT *facet, vertexT *oldvertex, vertexT *newvertex);
void    qh_resetlists(qhT *qh, boolT stats, boolT resetVisible /* qh.newvertex_list qh.newfacet_list qh.visible_list */);
void    qh_setvoronoi_all(qhT *qh);
void    qh_triangulate(qhT *qh /* qh.facet_list */);
void    qh_triangulate_facet(qhT *qh, facetT *facetA, vertexT **first_vertex);
void    qh_triangulate_link(qhT *qh, facetT *oldfacetA, facetT *facetA, facetT *oldfacetB, facetT *facetB);
void    qh_triangulate_mirror(qhT *qh, facetT *facetA, facetT *facetB);
void    qh_triangulate_null(qhT *qh, facetT *facetA);
void    qh_vertexintersect(qhT *qh, setT **vertexsetA,setT *vertexsetB);
setT   *qh_vertexintersect_new(qhT *qh, setT *vertexsetA,setT *vertexsetB);
void    qh_vertexneighbors(qhT *qh /* qh.facet_list */);
boolT   qh_vertexsubset(setT *vertexsetA, setT *vertexsetB);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFpoly */

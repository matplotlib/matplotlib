/*<html><pre>  -<a                             href="qh-io_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   io_r.h
   declarations of Input/Output functions

   see README, libqhull_r.h and io_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/io_r.h#2 $$Change: 2671 $
   $DateTime: 2019/06/06 11:24:01 $$Author: bbarber $
*/

#ifndef qhDEFio
#define qhDEFio 1

#include "libqhull_r.h"

/*============ constants and flags ==================*/

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_MAXfirst">-</a>

  qh_MAXfirst
    maximum length of first two lines of stdin
*/
#define qh_MAXfirst  200

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_MINradius">-</a>

  qh_MINradius
    min radius for Gp and Gv, fraction of maxcoord
*/
#define qh_MINradius 0.02

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_GEOMepsilon">-</a>

  qh_GEOMepsilon
    adjust outer planes for 'lines closer' and geomview roundoff.
    This prevents bleed through.
*/
#define qh_GEOMepsilon 2e-3

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="qh_WHITESPACE">-</a>

  qh_WHITESPACE
    possible values of white space
*/
#define qh_WHITESPACE " \n\t\v\r\f"


/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="RIDGE">-</a>

  qh_RIDGE
    to select which ridges to print in qh_eachvoronoi
*/
typedef enum
{
    qh_RIDGEall= 0, qh_RIDGEinner, qh_RIDGEouter
}
qh_RIDGE;

/*-<a                             href="qh-io_r.htm#TOC"
  >--------------------------------</a><a name="printvridgeT">-</a>

  printvridgeT
    prints results of qh_printvdiagram

  see:
    <a href="io_r.c#printvridge">qh_printvridge</a> for an example
*/
typedef void (*printvridgeT)(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);

/*============== -prototypes in alphabetical order =========*/

#ifdef __cplusplus
extern "C" {
#endif

void    qh_dfacet(qhT *qh, unsigned int id);
void    qh_dvertex(qhT *qh, unsigned int id);
int     qh_compare_facetarea(const void *p1, const void *p2);
int     qh_compare_facetvisit(const void *p1, const void *p2);
int     qh_compare_nummerge(const void *p1, const void *p2);
void    qh_copyfilename(qhT *qh, char *filename, int size, const char* source, int length);
void    qh_countfacets(qhT *qh, facetT *facetlist, setT *facets, boolT printall,
              int *numfacetsp, int *numsimplicialp, int *totneighborsp,
              int *numridgesp, int *numcoplanarsp, int *numnumtricoplanarsp);
pointT *qh_detvnorm(qhT *qh, vertexT *vertex, vertexT *vertexA, setT *centers, realT *offsetp);
setT   *qh_detvridge(qhT *qh, vertexT *vertex);
setT   *qh_detvridge3(qhT *qh, vertexT *atvertex, vertexT *vertex);
int     qh_eachvoronoi(qhT *qh, FILE *fp, printvridgeT printvridge, vertexT *atvertex, boolT visitall, qh_RIDGE innerouter, boolT inorder);
int     qh_eachvoronoi_all(qhT *qh, FILE *fp, printvridgeT printvridge, boolT isUpper, qh_RIDGE innerouter, boolT inorder);
void    qh_facet2point(qhT *qh, facetT *facet, pointT **point0, pointT **point1, realT *mindist);
setT   *qh_facetvertices(qhT *qh, facetT *facetlist, setT *facets, boolT allfacets);
void    qh_geomplanes(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane);
void    qh_markkeep(qhT *qh, facetT *facetlist);
setT   *qh_markvoronoi(qhT *qh, facetT *facetlist, setT *facets, boolT printall, boolT *isLowerp, int *numcentersp);
void    qh_order_vertexneighbors(qhT *qh, vertexT *vertex);
void    qh_prepare_output(qhT *qh);
void    qh_printafacet(qhT *qh, FILE *fp, qh_PRINT format, facetT *facet, boolT printall);
void    qh_printbegin(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printcenter(qhT *qh, FILE *fp, qh_PRINT format, const char *string, facetT *facet);
void    qh_printcentrum(qhT *qh, FILE *fp, facetT *facet, realT radius);
void    qh_printend(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printend4geom(qhT *qh, FILE *fp, facetT *facet, int *num, boolT printall);
void    qh_printextremes(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printextremes_2d(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printextremes_d(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printfacet(qhT *qh, FILE *fp, facetT *facet);
void    qh_printfacet2math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format, int notfirst);
void    qh_printfacet2geom(qhT *qh, FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet2geom_points(qhT *qh, FILE *fp, pointT *point1, pointT *point2,
                               facetT *facet, realT offset, realT color[3]);
void    qh_printfacet3math(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format, int notfirst);
void    qh_printfacet3geom_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet3geom_points(qhT *qh, FILE *fp, setT *points, facetT *facet, realT offset, realT color[3]);
void    qh_printfacet3geom_simplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet3vertex(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format);
void    qh_printfacet4geom_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet4geom_simplicial(qhT *qh, FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacetNvertex_nonsimplicial(qhT *qh, FILE *fp, facetT *facet, int id, qh_PRINT format);
void    qh_printfacetNvertex_simplicial(qhT *qh, FILE *fp, facetT *facet, qh_PRINT format);
void    qh_printfacetheader(qhT *qh, FILE *fp, facetT *facet);
void    qh_printfacetridges(qhT *qh, FILE *fp, facetT *facet);
void    qh_printfacets(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printhyperplaneintersection(qhT *qh, FILE *fp, facetT *facet1, facetT *facet2,
                   setT *vertices, realT color[3]);
void    qh_printline3geom(qhT *qh, FILE *fp, pointT *pointA, pointT *pointB, realT color[3]);
void    qh_printneighborhood(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall);
void    qh_printpoint(qhT *qh, FILE *fp, const char *string, pointT *point);
void    qh_printpointid(qhT *qh, FILE *fp, const char *string, int dim, pointT *point, int id);
void    qh_printpoint3(qhT *qh, FILE *fp, pointT *point);
void    qh_printpoints_out(qhT *qh, FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printpointvect(qhT *qh, FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius, realT color[3]);
void    qh_printpointvect2(qhT *qh, FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius);
void    qh_printridge(qhT *qh, FILE *fp, ridgeT *ridge);
void    qh_printspheres(qhT *qh, FILE *fp, setT *vertices, realT radius);
void    qh_printvdiagram(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
int     qh_printvdiagram2(qhT *qh, FILE *fp, printvridgeT printvridge, setT *vertices, qh_RIDGE innerouter, boolT inorder);
void    qh_printvertex(qhT *qh, FILE *fp, vertexT *vertex);
void    qh_printvertexlist(qhT *qh, FILE *fp, const char* string, facetT *facetlist,
                         setT *facets, boolT printall);
void    qh_printvertices(qhT *qh, FILE *fp, const char* string, setT *vertices);
void    qh_printvneighbors(qhT *qh, FILE *fp, facetT* facetlist, setT *facets, boolT printall);
void    qh_printvoronoi(qhT *qh, FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printvnorm(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);
void    qh_printvridge(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);
void    qh_produce_output(qhT *qh);
void    qh_produce_output2(qhT *qh);
void    qh_projectdim3(qhT *qh, pointT *source, pointT *destination);
int     qh_readfeasible(qhT *qh, int dim, const char *curline);
coordT *qh_readpoints(qhT *qh, int *numpoints, int *dimension, boolT *ismalloc);
void    qh_setfeasible(qhT *qh, int dim);
boolT   qh_skipfacet(qhT *qh, facetT *facet);
char   *qh_skipfilename(qhT *qh, char *filename);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFio */

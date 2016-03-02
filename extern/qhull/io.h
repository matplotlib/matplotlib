/*<html><pre>  -<a                             href="qh-io.htm"
  >-------------------------------</a><a name="TOP">-</a>

   io.h
   declarations of Input/Output functions

   see README, libqhull.h and io.c

   Copyright (c) 1993-2012 The Geometry Center.
   $Id: //main/2011/qhull/src/libqhull/io.h#3 $$Change: 1464 $
   $DateTime: 2012/01/25 22:58:41 $$Author: bbarber $
*/

#ifndef qhDEFio
#define qhDEFio 1

#include "libqhull.h"

/*============ constants and flags ==================*/

/*-<a                             href="qh-io.htm#TOC"
  >--------------------------------</a><a name="qh_MAXfirst">-</a>

  qh_MAXfirst
    maximum length of first two lines of stdin
*/
#define qh_MAXfirst  200

/*-<a                             href="qh-io.htm#TOC"
  >--------------------------------</a><a name="qh_MINradius">-</a>

  qh_MINradius
    min radius for Gp and Gv, fraction of maxcoord
*/
#define qh_MINradius 0.02

/*-<a                             href="qh-io.htm#TOC"
  >--------------------------------</a><a name="qh_GEOMepsilon">-</a>

  qh_GEOMepsilon
    adjust outer planes for 'lines closer' and geomview roundoff.
    This prevents bleed through.
*/
#define qh_GEOMepsilon 2e-3

/*-<a                             href="qh-io.htm#TOC"
  >--------------------------------</a><a name="qh_WHITESPACE">-</a>

  qh_WHITESPACE
    possible values of white space
*/
#define qh_WHITESPACE " \n\t\v\r\f"


/*-<a                             href="qh-io.htm#TOC"
  >--------------------------------</a><a name="RIDGE">-</a>

  qh_RIDGE
    to select which ridges to print in qh_eachvoronoi
*/
typedef enum
{
    qh_RIDGEall = 0, qh_RIDGEinner, qh_RIDGEouter
}
qh_RIDGE;

/*-<a                             href="qh-io.htm#TOC"
  >--------------------------------</a><a name="printvridgeT">-</a>

  printvridgeT
    prints results of qh_printvdiagram

  see:
    <a href="io.c#printvridge">qh_printvridge</a> for an example
*/
typedef void (*printvridgeT)(FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);

/*============== -prototypes in alphabetical order =========*/

void    dfacet(unsigned id);
void    dvertex(unsigned id);
int     qh_compare_facetarea(const void *p1, const void *p2);
int     qh_compare_facetmerge(const void *p1, const void *p2);
int     qh_compare_facetvisit(const void *p1, const void *p2);
int     qh_compare_vertexpoint(const void *p1, const void *p2); /* not used */
void    qh_copyfilename(char *filename, int size, const char* source, int length);
void    qh_countfacets(facetT *facetlist, setT *facets, boolT printall,
              int *numfacetsp, int *numsimplicialp, int *totneighborsp,
              int *numridgesp, int *numcoplanarsp, int *numnumtricoplanarsp);
pointT *qh_detvnorm(vertexT *vertex, vertexT *vertexA, setT *centers, realT *offsetp);
setT   *qh_detvridge(vertexT *vertex);
setT   *qh_detvridge3 (vertexT *atvertex, vertexT *vertex);
int     qh_eachvoronoi(FILE *fp, printvridgeT printvridge, vertexT *atvertex, boolT visitall, qh_RIDGE innerouter, boolT inorder);
int     qh_eachvoronoi_all(FILE *fp, printvridgeT printvridge, boolT isUpper, qh_RIDGE innerouter, boolT inorder);
void    qh_facet2point(facetT *facet, pointT **point0, pointT **point1, realT *mindist);
setT   *qh_facetvertices(facetT *facetlist, setT *facets, boolT allfacets);
void    qh_geomplanes(facetT *facet, realT *outerplane, realT *innerplane);
void    qh_markkeep(facetT *facetlist);
setT   *qh_markvoronoi(facetT *facetlist, setT *facets, boolT printall, boolT *isLowerp, int *numcentersp);
void    qh_order_vertexneighbors(vertexT *vertex);
void    qh_prepare_output(void);
void    qh_printafacet(FILE *fp, qh_PRINT format, facetT *facet, boolT printall);
void    qh_printbegin(FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printcenter(FILE *fp, qh_PRINT format, const char *string, facetT *facet);
void    qh_printcentrum(FILE *fp, facetT *facet, realT radius);
void    qh_printend(FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printend4geom(FILE *fp, facetT *facet, int *num, boolT printall);
void    qh_printextremes(FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printextremes_2d(FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printextremes_d(FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printfacet(FILE *fp, facetT *facet);
void    qh_printfacet2math(FILE *fp, facetT *facet, qh_PRINT format, int notfirst);
void    qh_printfacet2geom(FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet2geom_points(FILE *fp, pointT *point1, pointT *point2,
                               facetT *facet, realT offset, realT color[3]);
void    qh_printfacet3math(FILE *fp, facetT *facet, qh_PRINT format, int notfirst);
void    qh_printfacet3geom_nonsimplicial(FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet3geom_points(FILE *fp, setT *points, facetT *facet, realT offset, realT color[3]);
void    qh_printfacet3geom_simplicial(FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet3vertex(FILE *fp, facetT *facet, qh_PRINT format);
void    qh_printfacet4geom_nonsimplicial(FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacet4geom_simplicial(FILE *fp, facetT *facet, realT color[3]);
void    qh_printfacetNvertex_nonsimplicial(FILE *fp, facetT *facet, int id, qh_PRINT format);
void    qh_printfacetNvertex_simplicial(FILE *fp, facetT *facet, qh_PRINT format);
void    qh_printfacetheader(FILE *fp, facetT *facet);
void    qh_printfacetridges(FILE *fp, facetT *facet);
void    qh_printfacets(FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printhyperplaneintersection(FILE *fp, facetT *facet1, facetT *facet2,
                   setT *vertices, realT color[3]);
void    qh_printneighborhood(FILE *fp, qh_PRINT format, facetT *facetA, facetT *facetB, boolT printall);
void    qh_printline3geom(FILE *fp, pointT *pointA, pointT *pointB, realT color[3]);
void    qh_printpoint(FILE *fp, const char *string, pointT *point);
void    qh_printpointid(FILE *fp, const char *string, int dim, pointT *point, int id);
void    qh_printpoint3 (FILE *fp, pointT *point);
void    qh_printpoints_out(FILE *fp, facetT *facetlist, setT *facets, boolT printall);
void    qh_printpointvect(FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius, realT color[3]);
void    qh_printpointvect2 (FILE *fp, pointT *point, coordT *normal, pointT *center, realT radius);
void    qh_printridge(FILE *fp, ridgeT *ridge);
void    qh_printspheres(FILE *fp, setT *vertices, realT radius);
void    qh_printvdiagram(FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
int     qh_printvdiagram2 (FILE *fp, printvridgeT printvridge, setT *vertices, qh_RIDGE innerouter, boolT inorder);
void    qh_printvertex(FILE *fp, vertexT *vertex);
void    qh_printvertexlist(FILE *fp, const char* string, facetT *facetlist,
                         setT *facets, boolT printall);
void    qh_printvertices(FILE *fp, const char* string, setT *vertices);
void    qh_printvneighbors(FILE *fp, facetT* facetlist, setT *facets, boolT printall);
void    qh_printvoronoi(FILE *fp, qh_PRINT format, facetT *facetlist, setT *facets, boolT printall);
void    qh_printvnorm(FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);
void    qh_printvridge(FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded);
void    qh_produce_output(void);
void    qh_produce_output2(void);
void    qh_projectdim3 (pointT *source, pointT *destination);
int     qh_readfeasible(int dim, const char *curline);
coordT *qh_readpoints(int *numpoints, int *dimension, boolT *ismalloc);
void    qh_setfeasible(int dim);
boolT   qh_skipfacet(facetT *facet);
char   *qh_skipfilename(char *filename);

#endif /* qhDEFio */

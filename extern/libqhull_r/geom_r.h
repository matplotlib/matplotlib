/*<html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  geom_r.h
    header file for geometric routines

   see qh-geom_r.htm and geom_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/geom_r.h#1 $$Change: 2661 $
   $DateTime: 2019/05/24 20:09:58 $$Author: bbarber $
*/

#ifndef qhDEFgeom
#define qhDEFgeom 1

#include "libqhull_r.h"

/* ============ -macros- ======================== */

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="fabs_">-</a>

  fabs_(a)
    returns the absolute value of a
*/
#define fabs_( a ) ((( a ) < 0 ) ? -( a ):( a ))

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="fmax_">-</a>

  fmax_(a,b)
    returns the maximum value of a and b
*/
#define fmax_( a,b )  ( ( a ) < ( b ) ? ( b ) : ( a ) )

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="fmin_">-</a>

  fmin_(a,b)
    returns the minimum value of a and b
*/
#define fmin_( a,b )  ( ( a ) > ( b ) ? ( b ) : ( a ) )

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="maximize_">-</a>

  maximize_(maxval, val)
    set maxval to val if val is greater than maxval
*/
#define maximize_( maxval, val ) { if (( maxval ) < ( val )) ( maxval )= ( val ); }

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="minimize_">-</a>

  minimize_(minval, val)
    set minval to val if val is less than minval
*/
#define minimize_( minval, val ) { if (( minval ) > ( val )) ( minval )= ( val ); }

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="det2_">-</a>

  det2_(a1, a2,
        b1, b2)

    compute a 2-d determinate
*/
#define det2_( a1,a2,b1,b2 ) (( a1 )*( b2 ) - ( a2 )*( b1 ))

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="det3_">-</a>

  det3_(a1, a2, a3,
       b1, b2, b3,
       c1, c2, c3)

    compute a 3-d determinate
*/
#define det3_( a1,a2,a3,b1,b2,b3,c1,c2,c3 ) ( ( a1 )*det2_( b2,b3,c2,c3 ) \
                - ( b1 )*det2_( a2,a3,c2,c3 ) + ( c1 )*det2_( a2,a3,b2,b3 ) )

/*-<a                             href="qh-geom_r.htm#TOC"
  >--------------------------------</a><a name="dX">-</a>

  dX( p1, p2 )
  dY( p1, p2 )
  dZ( p1, p2 )

    given two indices into rows[],

    compute the difference between X, Y, or Z coordinates
*/
#define dX( p1,p2 )  ( *( rows[p1] ) - *( rows[p2] ))
#define dY( p1,p2 )  ( *( rows[p1]+1 ) - *( rows[p2]+1 ))
#define dZ( p1,p2 )  ( *( rows[p1]+2 ) - *( rows[p2]+2 ))
#define dW( p1,p2 )  ( *( rows[p1]+3 ) - *( rows[p2]+3 ))

/*============= prototypes in alphabetical order, infrequent at end ======= */

#ifdef __cplusplus
extern "C" {
#endif

void    qh_backnormal(qhT *qh, realT **rows, int numrow, int numcol, boolT sign, coordT *normal, boolT *nearzero);
void    qh_distplane(qhT *qh, pointT *point, facetT *facet, realT *dist);
facetT *qh_findbest(qhT *qh, pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT isnewfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart);
facetT *qh_findbesthorizon(qhT *qh, boolT ischeckmax, pointT *point,
                     facetT *startfacet, boolT noupper, realT *bestdist, int *numpart);
facetT *qh_findbestnew(qhT *qh, pointT *point, facetT *startfacet, realT *dist,
                     boolT bestoutside, boolT *isoutside, int *numpart);
void    qh_gausselim(qhT *qh, realT **rows, int numrow, int numcol, boolT *sign, boolT *nearzero);
realT   qh_getangle(qhT *qh, pointT *vect1, pointT *vect2);
pointT *qh_getcenter(qhT *qh, setT *vertices);
pointT *qh_getcentrum(qhT *qh, facetT *facet);
coordT  qh_getdistance(qhT *qh, facetT *facet, facetT *neighbor, coordT *mindist, coordT *maxdist);
void    qh_normalize(qhT *qh, coordT *normal, int dim, boolT toporient);
void    qh_normalize2(qhT *qh, coordT *normal, int dim, boolT toporient,
            realT *minnorm, boolT *ismin);
pointT *qh_projectpoint(qhT *qh, pointT *point, facetT *facet, realT dist);

void    qh_setfacetplane(qhT *qh, facetT *newfacets);
void    qh_sethyperplane_det(qhT *qh, int dim, coordT **rows, coordT *point0,
              boolT toporient, coordT *normal, realT *offset, boolT *nearzero);
void    qh_sethyperplane_gauss(qhT *qh, int dim, coordT **rows, pointT *point0,
             boolT toporient, coordT *normal, coordT *offset, boolT *nearzero);
boolT   qh_sharpnewfacets(qhT *qh);

/*========= infrequently used code in geom2_r.c =============*/

coordT *qh_copypoints(qhT *qh, coordT *points, int numpoints, int dimension);
void    qh_crossproduct(int dim, realT vecA[3], realT vecB[3], realT vecC[3]);
realT   qh_determinant(qhT *qh, realT **rows, int dim, boolT *nearzero);
realT   qh_detjoggle(qhT *qh, pointT *points, int numpoints, int dimension);
void    qh_detmaxoutside(qhT *qh);
void    qh_detroundoff(qhT *qh);
realT   qh_detsimplex(qhT *qh, pointT *apex, setT *points, int dim, boolT *nearzero);
realT   qh_distnorm(int dim, pointT *point, pointT *normal, realT *offsetp);
realT   qh_distround(qhT *qh, int dimension, realT maxabs, realT maxsumabs);
realT   qh_divzero(realT numer, realT denom, realT mindenom1, boolT *zerodiv);
realT   qh_facetarea(qhT *qh, facetT *facet);
realT   qh_facetarea_simplex(qhT *qh, int dim, coordT *apex, setT *vertices,
          vertexT *notvertex,  boolT toporient, coordT *normal, realT *offset);
pointT *qh_facetcenter(qhT *qh, setT *vertices);
facetT *qh_findgooddist(qhT *qh, pointT *point, facetT *facetA, realT *distp, facetT **facetlist);
vertexT *qh_furthestnewvertex(qhT *qh, unsigned int unvisited, facetT *facet, realT *maxdistp /* qh.newvertex_list */);
vertexT *qh_furthestvertex(qhT *qh, facetT *facetA, facetT *facetB, realT *maxdistp, realT *mindistp);
void    qh_getarea(qhT *qh, facetT *facetlist);
boolT   qh_gram_schmidt(qhT *qh, int dim, realT **rows);
boolT   qh_inthresholds(qhT *qh, coordT *normal, realT *angle);
void    qh_joggleinput(qhT *qh);
realT  *qh_maxabsval(realT *normal, int dim);
setT   *qh_maxmin(qhT *qh, pointT *points, int numpoints, int dimension);
realT   qh_maxouter(qhT *qh);
void    qh_maxsimplex(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints, setT **simplex);
realT   qh_minabsval(realT *normal, int dim);
int     qh_mindiff(realT *vecA, realT *vecB, int dim);
boolT   qh_orientoutside(qhT *qh, facetT *facet);
void    qh_outerinner(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane);
coordT  qh_pointdist(pointT *point1, pointT *point2, int dim);
void    qh_printmatrix(qhT *qh, FILE *fp, const char *string, realT **rows, int numrow, int numcol);
void    qh_printpoints(qhT *qh, FILE *fp, const char *string, setT *points);
void    qh_projectinput(qhT *qh);
void    qh_projectpoints(qhT *qh, signed char *project, int n, realT *points,
             int numpoints, int dim, realT *newpoints, int newdim);
void    qh_rotateinput(qhT *qh, realT **rows);
void    qh_rotatepoints(qhT *qh, realT *points, int numpoints, int dim, realT **rows);
void    qh_scaleinput(qhT *qh);
void    qh_scalelast(qhT *qh, coordT *points, int numpoints, int dim, coordT low,
                   coordT high, coordT newhigh);
void    qh_scalepoints(qhT *qh, pointT *points, int numpoints, int dim,
                realT *newlows, realT *newhighs);
boolT   qh_sethalfspace(qhT *qh, int dim, coordT *coords, coordT **nextp,
              coordT *normal, coordT *offset, coordT *feasible);
coordT *qh_sethalfspace_all(qhT *qh, int dim, int count, coordT *halfspaces, pointT *feasible);
coordT  qh_vertex_bestdist(qhT *qh, setT *vertices);
coordT  qh_vertex_bestdist2(qhT *qh, setT *vertices, vertexT **vertexp, vertexT **vertexp2);
pointT *qh_voronoi_center(qhT *qh, int dim, setT *points);

#ifdef __cplusplus
} /* extern "C"*/
#endif

#endif /* qhDEFgeom */




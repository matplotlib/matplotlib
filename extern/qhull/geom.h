/*<html><pre>  -<a                             href="qh-geom.htm"
  >-------------------------------</a><a name="TOP">-</a>

  geom.h
    header file for geometric routines

   see qh-geom.htm and geom.c

   Copyright (c) 1993-2012 The Geometry Center.
   $Id: //main/2011/qhull/src/libqhull/geom.h#3 $$Change: 1464 $
   $DateTime: 2012/01/25 22:58:41 $$Author: bbarber $
*/

#ifndef qhDEFgeom
#define qhDEFgeom 1

#include "libqhull.h"

/* ============ -macros- ======================== */

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="fabs_">-</a>

  fabs_(a)
    returns the absolute value of a
*/
#define fabs_( a ) ((( a ) < 0 ) ? -( a ):( a ))

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="fmax_">-</a>

  fmax_(a,b)
    returns the maximum value of a and b
*/
#define fmax_( a,b )  ( ( a ) < ( b ) ? ( b ) : ( a ) )

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="fmin_">-</a>

  fmin_(a,b)
    returns the minimum value of a and b
*/
#define fmin_( a,b )  ( ( a ) > ( b ) ? ( b ) : ( a ) )

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="maximize_">-</a>

  maximize_(maxval, val)
    set maxval to val if val is greater than maxval
*/
#define maximize_( maxval, val ) { if (( maxval ) < ( val )) ( maxval )= ( val ); }

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="minimize_">-</a>

  minimize_(minval, val)
    set minval to val if val is less than minval
*/
#define minimize_( minval, val ) { if (( minval ) > ( val )) ( minval )= ( val ); }

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="det2_">-</a>

  det2_(a1, a2,
        b1, b2)

    compute a 2-d determinate
*/
#define det2_( a1,a2,b1,b2 ) (( a1 )*( b2 ) - ( a2 )*( b1 ))

/*-<a                             href="qh-geom.htm#TOC"
  >--------------------------------</a><a name="det3_">-</a>

  det3_(a1, a2, a3,
       b1, b2, b3,
       c1, c2, c3)

    compute a 3-d determinate
*/
#define det3_( a1,a2,a3,b1,b2,b3,c1,c2,c3 ) ( ( a1 )*det2_( b2,b3,c2,c3 ) \
                - ( b1 )*det2_( a2,a3,c2,c3 ) + ( c1 )*det2_( a2,a3,b2,b3 ) )

/*-<a                             href="qh-geom.htm#TOC"
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

void    qh_backnormal(realT **rows, int numrow, int numcol, boolT sign, coordT *normal, boolT *nearzero);
void    qh_distplane(pointT *point, facetT *facet, realT *dist);
facetT *qh_findbest(pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT isnewfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart);
facetT *qh_findbesthorizon(boolT ischeckmax, pointT *point,
                     facetT *startfacet, boolT noupper, realT *bestdist, int *numpart);
facetT *qh_findbestnew(pointT *point, facetT *startfacet, realT *dist,
                     boolT bestoutside, boolT *isoutside, int *numpart);
void    qh_gausselim(realT **rows, int numrow, int numcol, boolT *sign, boolT *nearzero);
realT   qh_getangle(pointT *vect1, pointT *vect2);
pointT *qh_getcenter(setT *vertices);
pointT *qh_getcentrum(facetT *facet);
realT   qh_getdistance(facetT *facet, facetT *neighbor, realT *mindist, realT *maxdist);
void    qh_normalize(coordT *normal, int dim, boolT toporient);
void    qh_normalize2 (coordT *normal, int dim, boolT toporient,
            realT *minnorm, boolT *ismin);
pointT *qh_projectpoint(pointT *point, facetT *facet, realT dist);

void    qh_setfacetplane(facetT *newfacets);
void    qh_sethyperplane_det(int dim, coordT **rows, coordT *point0,
              boolT toporient, coordT *normal, realT *offset, boolT *nearzero);
void    qh_sethyperplane_gauss(int dim, coordT **rows, pointT *point0,
             boolT toporient, coordT *normal, coordT *offset, boolT *nearzero);
boolT   qh_sharpnewfacets(void);

/*========= infrequently used code in geom2.c =============*/

coordT *qh_copypoints(coordT *points, int numpoints, int dimension);
void    qh_crossproduct(int dim, realT vecA[3], realT vecB[3], realT vecC[3]);
realT   qh_determinant(realT **rows, int dim, boolT *nearzero);
realT   qh_detjoggle(pointT *points, int numpoints, int dimension);
void    qh_detroundoff(void);
realT   qh_detsimplex(pointT *apex, setT *points, int dim, boolT *nearzero);
realT   qh_distnorm(int dim, pointT *point, pointT *normal, realT *offsetp);
realT   qh_distround(int dimension, realT maxabs, realT maxsumabs);
realT   qh_divzero(realT numer, realT denom, realT mindenom1, boolT *zerodiv);
realT   qh_facetarea(facetT *facet);
realT   qh_facetarea_simplex(int dim, coordT *apex, setT *vertices,
          vertexT *notvertex,  boolT toporient, coordT *normal, realT *offset);
pointT *qh_facetcenter(setT *vertices);
facetT *qh_findgooddist(pointT *point, facetT *facetA, realT *distp, facetT **facetlist);
void    qh_getarea(facetT *facetlist);
boolT   qh_gram_schmidt(int dim, realT **rows);
boolT   qh_inthresholds(coordT *normal, realT *angle);
void    qh_joggleinput(void);
realT  *qh_maxabsval(realT *normal, int dim);
setT   *qh_maxmin(pointT *points, int numpoints, int dimension);
realT   qh_maxouter(void);
void    qh_maxsimplex(int dim, setT *maxpoints, pointT *points, int numpoints, setT **simplex);
realT   qh_minabsval(realT *normal, int dim);
int     qh_mindiff(realT *vecA, realT *vecB, int dim);
boolT   qh_orientoutside(facetT *facet);
void    qh_outerinner(facetT *facet, realT *outerplane, realT *innerplane);
coordT  qh_pointdist(pointT *point1, pointT *point2, int dim);
void    qh_printmatrix(FILE *fp, const char *string, realT **rows, int numrow, int numcol);
void    qh_printpoints(FILE *fp, const char *string, setT *points);
void    qh_projectinput(void);
void    qh_projectpoints(signed char *project, int n, realT *points,
             int numpoints, int dim, realT *newpoints, int newdim);
void    qh_rotateinput(realT **rows);
void    qh_rotatepoints(realT *points, int numpoints, int dim, realT **rows);
void    qh_scaleinput(void);
void    qh_scalelast(coordT *points, int numpoints, int dim, coordT low,
                   coordT high, coordT newhigh);
void    qh_scalepoints(pointT *points, int numpoints, int dim,
                realT *newlows, realT *newhighs);
boolT   qh_sethalfspace(int dim, coordT *coords, coordT **nextp,
              coordT *normal, coordT *offset, coordT *feasible);
coordT *qh_sethalfspace_all(int dim, int count, coordT *halfspaces, pointT *feasible);
pointT *qh_voronoi_center(int dim, setT *points);

#endif /* qhDEFgeom */

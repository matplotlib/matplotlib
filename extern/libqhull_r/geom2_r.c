/*<html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>


   geom2_r.c
   infrequently used geometric routines of qhull

   see qh-geom_r.htm and geom_r.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/geom2_r.c#15 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $

   frequently used code goes into geom_r.c
*/

#include "qhull_ra.h"

/*================== functions in alphabetic order ============*/

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="copypoints">-</a>

  qh_copypoints(qh, points, numpoints, dimension )
    return qh_malloc'd copy of points

  notes:
    qh_free the returned points to avoid a memory leak
*/
coordT *qh_copypoints(qhT *qh, coordT *points, int numpoints, int dimension) {
  int size;
  coordT *newpoints;

  size= numpoints * dimension * (int)sizeof(coordT);
  if (!(newpoints= (coordT *)qh_malloc((size_t)size))) {
    qh_fprintf(qh, qh->ferr, 6004, "qhull error: insufficient memory to copy %d points\n",
        numpoints);
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  memcpy((char *)newpoints, (char *)points, (size_t)size); /* newpoints!=0 by QH6004 */
  return newpoints;
} /* copypoints */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="crossproduct">-</a>

  qh_crossproduct( dim, vecA, vecB, vecC )
    crossproduct of 2 dim vectors
    C= A x B

  notes:
    from Glasner, Graphics Gems I, p. 639
    only defined for dim==3
*/
void qh_crossproduct(int dim, realT vecA[3], realT vecB[3], realT vecC[3]){

  if (dim == 3) {
    vecC[0]=   det2_(vecA[1], vecA[2],
                     vecB[1], vecB[2]);
    vecC[1]= - det2_(vecA[0], vecA[2],
                     vecB[0], vecB[2]);
    vecC[2]=   det2_(vecA[0], vecA[1],
                     vecB[0], vecB[1]);
  }
} /* vcross */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="determinant">-</a>

  qh_determinant(qh, rows, dim, nearzero )
    compute signed determinant of a square matrix
    uses qh.NEARzero to test for degenerate matrices

  returns:
    determinant
    overwrites rows and the matrix
    if dim == 2 or 3
      nearzero iff determinant < qh->NEARzero[dim-1]
      (!quite correct, not critical)
    if dim >= 4
      nearzero iff diagonal[k] < qh->NEARzero[k]
*/
realT qh_determinant(qhT *qh, realT **rows, int dim, boolT *nearzero) {
  realT det=0;
  int i;
  boolT sign= False;

  *nearzero= False;
  if (dim < 2) {
    qh_fprintf(qh, qh->ferr, 6005, "qhull internal error (qh_determinate): only implemented for dimension >= 2\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }else if (dim == 2) {
    det= det2_(rows[0][0], rows[0][1],
                 rows[1][0], rows[1][1]);
    if (fabs_(det) < 10*qh->NEARzero[1])  /* QH11031 FIX: not really correct, what should this be? */
      *nearzero= True;
  }else if (dim == 3) {
    det= det3_(rows[0][0], rows[0][1], rows[0][2],
                 rows[1][0], rows[1][1], rows[1][2],
                 rows[2][0], rows[2][1], rows[2][2]);
    if (fabs_(det) < 10*qh->NEARzero[2])  /* QH11031 FIX: what should this be?  det 5.5e-12 was flat for qh_maxsimplex of qdelaunay 0,0 27,27 -36,36 -9,63 */
      *nearzero= True;
  }else {
    qh_gausselim(qh, rows, dim, dim, &sign, nearzero);  /* if nearzero, diagonal still ok */
    det= 1.0;
    for (i=dim; i--; )
      det *= (rows[i])[i];
    if (sign)
      det= -det;
  }
  return det;
} /* determinant */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detjoggle">-</a>

  qh_detjoggle(qh, points, numpoints, dimension )
    determine default max joggle for point array
      as qh_distround * qh_JOGGLEdefault

  returns:
    initial value for JOGGLEmax from points and REALepsilon

  notes:
    computes DISTround since qh_maxmin not called yet
    if qh->SCALElast, last dimension will be scaled later to MAXwidth

    loop duplicated from qh_maxmin
*/
realT qh_detjoggle(qhT *qh, pointT *points, int numpoints, int dimension) {
  realT abscoord, distround, joggle, maxcoord, mincoord;
  pointT *point, *pointtemp;
  realT maxabs= -REALmax;
  realT sumabs= 0;
  realT maxwidth= 0;
  int k;

  if (qh->SETroundoff)
    distround= qh->DISTround; /* 'En' */
  else{
    for (k=0; k < dimension; k++) {
      if (qh->SCALElast && k == dimension-1)
        abscoord= maxwidth;
      else if (qh->DELAUNAY && k == dimension-1) /* will qh_setdelaunay() */
        abscoord= 2 * maxabs * maxabs;  /* may be low by qh->hull_dim/2 */
      else {
        maxcoord= -REALmax;
        mincoord= REALmax;
        FORALLpoint_(qh, points, numpoints) {
          maximize_(maxcoord, point[k]);
          minimize_(mincoord, point[k]);
        }
        maximize_(maxwidth, maxcoord-mincoord);
        abscoord= fmax_(maxcoord, -mincoord);
      }
      sumabs += abscoord;
      maximize_(maxabs, abscoord);
    } /* for k */
    distround= qh_distround(qh, qh->hull_dim, maxabs, sumabs);
  }
  joggle= distround * qh_JOGGLEdefault;
  maximize_(joggle, REALepsilon * qh_JOGGLEdefault);
  trace2((qh, qh->ferr, 2001, "qh_detjoggle: joggle=%2.2g maxwidth=%2.2g\n", joggle, maxwidth));
  return joggle;
} /* detjoggle */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detmaxoutside">-</a>

  qh_detmaxoutside(qh);
    determine qh.MAXoutside target for qh_RATIO... tests of distance
    updates option '_max-outside'

  notes:
    called from qh_addpoint and qh_detroundoff
    accounts for qh.ONEmerge, qh.DISTround, qh.MINoutside ('Wn'), qh.max_outside
    see qh_maxout for qh.max_outside with qh.DISTround
*/

void qh_detmaxoutside(qhT *qh) {
  realT maxoutside;

  maxoutside= fmax_(qh->max_outside, qh->ONEmerge + qh->DISTround);
  maximize_(maxoutside, qh->MINoutside);
  qh->MAXoutside= maxoutside;
  trace3((qh, qh->ferr, 3056, "qh_detmaxoutside: MAXoutside %2.2g from qh.max_outside %2.2g, ONEmerge %2.2g, MINoutside %2.2g, DISTround %2.2g\n",
      qh->MAXoutside, qh->max_outside, qh->ONEmerge, qh->MINoutside, qh->DISTround));
} /* detmaxoutside */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detroundoff">-</a>

  qh_detroundoff(qh)
    determine maximum roundoff errors from
      REALepsilon, REALmax, REALmin, qh.hull_dim, qh.MAXabs_coord,
      qh.MAXsumcoord, qh.MAXwidth, qh.MINdenom_1

    accounts for qh.SETroundoff, qh.RANDOMdist, qh->MERGEexact
      qh.premerge_cos, qh.postmerge_cos, qh.premerge_centrum,
      qh.postmerge_centrum, qh.MINoutside,
      qh_RATIOnearinside, qh_COPLANARratio, qh_WIDEcoplanar

  returns:
    sets qh.DISTround, etc. (see below)
    appends precision constants to qh.qhull_options

  see:
    qh_maxmin() for qh.NEARzero

  design:
    determine qh.DISTround for distance computations
    determine minimum denominators for qh_divzero
    determine qh.ANGLEround for angle computations
    adjust qh.premerge_cos,... for roundoff error
    determine qh.ONEmerge for maximum error due to a single merge
    determine qh.NEARinside, qh.MAXcoplanar, qh.MINvisible,
      qh.MINoutside, qh.WIDEfacet
    initialize qh.max_vertex and qh.minvertex
*/
void qh_detroundoff(qhT *qh) {

  qh_option(qh, "_max-width", NULL, &qh->MAXwidth);
  if (!qh->SETroundoff) {
    qh->DISTround= qh_distround(qh, qh->hull_dim, qh->MAXabs_coord, qh->MAXsumcoord);
    qh_option(qh, "Error-roundoff", NULL, &qh->DISTround);
  }
  qh->MINdenom= qh->MINdenom_1 * qh->MAXabs_coord;
  qh->MINdenom_1_2= sqrt(qh->MINdenom_1 * qh->hull_dim) ;  /* if will be normalized */
  qh->MINdenom_2= qh->MINdenom_1_2 * qh->MAXabs_coord;
                                              /* for inner product */
  qh->ANGLEround= 1.01 * qh->hull_dim * REALepsilon;
  if (qh->RANDOMdist) {
    qh->ANGLEround += qh->RANDOMfactor;
    trace4((qh, qh->ferr, 4096, "qh_detroundoff: increase qh.ANGLEround by option 'R%2.2g'\n", qh->RANDOMfactor));
  }
  if (qh->premerge_cos < REALmax/2) {
    qh->premerge_cos -= qh->ANGLEround;
    if (qh->RANDOMdist)
      qh_option(qh, "Angle-premerge-with-random", NULL, &qh->premerge_cos);
  }
  if (qh->postmerge_cos < REALmax/2) {
    qh->postmerge_cos -= qh->ANGLEround;
    if (qh->RANDOMdist)
      qh_option(qh, "Angle-postmerge-with-random", NULL, &qh->postmerge_cos);
  }
  qh->premerge_centrum += 2 * qh->DISTround;    /*2 for centrum and distplane()*/
  qh->postmerge_centrum += 2 * qh->DISTround;
  if (qh->RANDOMdist && (qh->MERGEexact || qh->PREmerge))
    qh_option(qh, "Centrum-premerge-with-random", NULL, &qh->premerge_centrum);
  if (qh->RANDOMdist && qh->POSTmerge)
    qh_option(qh, "Centrum-postmerge-with-random", NULL, &qh->postmerge_centrum);
  { /* compute ONEmerge, max vertex offset for merging simplicial facets */
    realT maxangle= 1.0, maxrho;

    minimize_(maxangle, qh->premerge_cos);
    minimize_(maxangle, qh->postmerge_cos);
    /* max diameter * sin theta + DISTround for vertex to its hyperplane */
    qh->ONEmerge= sqrt((realT)qh->hull_dim) * qh->MAXwidth *
      sqrt(1.0 - maxangle * maxangle) + qh->DISTround;
    maxrho= qh->hull_dim * qh->premerge_centrum + qh->DISTround;
    maximize_(qh->ONEmerge, maxrho);
    maxrho= qh->hull_dim * qh->postmerge_centrum + qh->DISTround;
    maximize_(qh->ONEmerge, maxrho);
    if (qh->MERGING)
      qh_option(qh, "_one-merge", NULL, &qh->ONEmerge);
  }
  qh->NEARinside= qh->ONEmerge * qh_RATIOnearinside; /* only used if qh->KEEPnearinside */
  if (qh->JOGGLEmax < REALmax/2 && (qh->KEEPcoplanar || qh->KEEPinside)) {
    realT maxdist;             /* adjust qh.NEARinside for joggle */
    qh->KEEPnearinside= True;
    maxdist= sqrt((realT)qh->hull_dim) * qh->JOGGLEmax + qh->DISTround;
    maxdist= 2*maxdist;        /* vertex and coplanar point can joggle in opposite directions */
    maximize_(qh->NEARinside, maxdist);  /* must agree with qh_nearcoplanar() */
  }
  if (qh->KEEPnearinside)
    qh_option(qh, "_near-inside", NULL, &qh->NEARinside);
  if (qh->JOGGLEmax < qh->DISTround) {
    qh_fprintf(qh, qh->ferr, 6006, "qhull option error: the joggle for 'QJn', %.2g, is below roundoff for distance computations, %.2g\n",
         qh->JOGGLEmax, qh->DISTround);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  if (qh->MINvisible > REALmax/2) {
    if (!qh->MERGING)
      qh->MINvisible= qh->DISTround;
    else if (qh->hull_dim <= 3)
      qh->MINvisible= qh->premerge_centrum;
    else
      qh->MINvisible= qh_COPLANARratio * qh->premerge_centrum;
    if (qh->APPROXhull && qh->MINvisible > qh->MINoutside)
      qh->MINvisible= qh->MINoutside;
    qh_option(qh, "Visible-distance", NULL, &qh->MINvisible);
  }
  if (qh->MAXcoplanar > REALmax/2) {
    qh->MAXcoplanar= qh->MINvisible;
    qh_option(qh, "U-max-coplanar", NULL, &qh->MAXcoplanar);
  }
  if (!qh->APPROXhull) {             /* user may specify qh->MINoutside */
    qh->MINoutside= 2 * qh->MINvisible;
    if (qh->premerge_cos < REALmax/2)
      maximize_(qh->MINoutside, (1- qh->premerge_cos) * qh->MAXabs_coord);
    qh_option(qh, "Width-outside", NULL, &qh->MINoutside);
  }
  qh->WIDEfacet= qh->MINoutside;
  maximize_(qh->WIDEfacet, qh_WIDEcoplanar * qh->MAXcoplanar);
  maximize_(qh->WIDEfacet, qh_WIDEcoplanar * qh->MINvisible);
  qh_option(qh, "_wide-facet", NULL, &qh->WIDEfacet);
  if (qh->MINvisible > qh->MINoutside + 3 * REALepsilon
  && !qh->BESToutside && !qh->FORCEoutput)
    qh_fprintf(qh, qh->ferr, 7001, "qhull input warning: minimum visibility V%.2g is greater than \nminimum outside W%.2g.  Flipped facets are likely.\n",
             qh->MINvisible, qh->MINoutside);
  qh->max_vertex= qh->DISTround;
  qh->min_vertex= -qh->DISTround;
  /* numeric constants reported in printsummary */
  qh_detmaxoutside(qh);
} /* detroundoff */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="detsimplex">-</a>

  qh_detsimplex(qh, apex, points, dim, nearzero )
    compute determinant of a simplex with point apex and base points

  returns:
     signed determinant and nearzero from qh_determinant

  notes:
     called by qh_maxsimplex and qh_initialvertices
     uses qh.gm_matrix/qh.gm_row (assumes they're big enough)

  design:
    construct qm_matrix by subtracting apex from points
    compute determinate
*/
realT qh_detsimplex(qhT *qh, pointT *apex, setT *points, int dim, boolT *nearzero) {
  pointT *coorda, *coordp, *gmcoord, *point, **pointp;
  coordT **rows;
  int k,  i=0;
  realT det;

  zinc_(Zdetsimplex);
  gmcoord= qh->gm_matrix;
  rows= qh->gm_row;
  FOREACHpoint_(points) {
    if (i == dim)
      break;
    rows[i++]= gmcoord;
    coordp= point;
    coorda= apex;
    for (k=dim; k--; )
      *(gmcoord++)= *coordp++ - *coorda++;
  }
  if (i < dim) {
    qh_fprintf(qh, qh->ferr, 6007, "qhull internal error (qh_detsimplex): #points %d < dimension %d\n",
               i, dim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  det= qh_determinant(qh, rows, dim, nearzero);
  trace2((qh, qh->ferr, 2002, "qh_detsimplex: det=%2.2g for point p%d, dim %d, nearzero? %d\n",
          det, qh_pointid(qh, apex), dim, *nearzero));
  return det;
} /* detsimplex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="distnorm">-</a>

  qh_distnorm( dim, point, normal, offset )
    return distance from point to hyperplane at normal/offset

  returns:
    dist

  notes:
    dist > 0 if point is outside of hyperplane

  see:
    qh_distplane in geom_r.c
*/
realT qh_distnorm(int dim, pointT *point, pointT *normal, realT *offsetp) {
  coordT *normalp= normal, *coordp= point;
  realT dist;
  int k;

  dist= *offsetp;
  for (k=dim; k--; )
    dist += *(coordp++) * *(normalp++);
  return dist;
} /* distnorm */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="distround">-</a>

  qh_distround(qh, dimension, maxabs, maxsumabs )
    compute maximum round-off error for a distance computation
      to a normalized hyperplane
    maxabs is the maximum absolute value of a coordinate
    maxsumabs is the maximum possible sum of absolute coordinate values
    if qh.RANDOMdist ('Qr'), adjusts qh_distround

  returns:
    max dist round for qh.REALepsilon and qh.RANDOMdist

  notes:
    calculate roundoff error according to Golub & van Loan, 1983, Lemma 3.2-1, "Rounding Errors"
    use sqrt(dim) since one vector is normalized
      or use maxsumabs since one vector is < 1
*/
realT qh_distround(qhT *qh, int dimension, realT maxabs, realT maxsumabs) {
  realT maxdistsum, maxround, delta;

  maxdistsum= sqrt((realT)dimension) * maxabs;
  minimize_( maxdistsum, maxsumabs);
  maxround= REALepsilon * (dimension * maxdistsum * 1.01 + maxabs);
              /* adds maxabs for offset */
  if (qh->RANDOMdist) {
    delta= qh->RANDOMfactor * maxabs;
    maxround += delta;
    trace4((qh, qh->ferr, 4092, "qh_distround: increase roundoff by random delta %2.2g for option 'R%2.2g'\n", delta, qh->RANDOMfactor));
  }
  trace4((qh, qh->ferr, 4008, "qh_distround: %2.2g, maxabs %2.2g, maxsumabs %2.2g, maxdistsum %2.2g\n",
            maxround, maxabs, maxsumabs, maxdistsum));
  return maxround;
} /* distround */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="divzero">-</a>

  qh_divzero( numer, denom, mindenom1, zerodiv )
    divide by a number that's nearly zero
    mindenom1= minimum denominator for dividing into 1.0

  returns:
    quotient
    sets zerodiv and returns 0.0 if it would overflow

  design:
    if numer is nearly zero and abs(numer) < abs(denom)
      return numer/denom
    else if numer is nearly zero
      return 0 and zerodiv
    else if denom/numer non-zero
      return numer/denom
    else
      return 0 and zerodiv
*/
realT qh_divzero(realT numer, realT denom, realT mindenom1, boolT *zerodiv) {
  realT temp, numerx, denomx;


  if (numer < mindenom1 && numer > -mindenom1) {
    numerx= fabs_(numer);
    denomx= fabs_(denom);
    if (numerx < denomx) {
      *zerodiv= False;
      return numer/denom;
    }else {
      *zerodiv= True;
      return 0.0;
    }
  }
  temp= denom/numer;
  if (temp > mindenom1 || temp < -mindenom1) {
    *zerodiv= False;
    return numer/denom;
  }else {
    *zerodiv= True;
    return 0.0;
  }
} /* divzero */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="facetarea">-</a>

  qh_facetarea(qh, facet )
    return area for a facet

  notes:
    if non-simplicial,
      uses centrum to triangulate facet and sums the projected areas.
    if (qh->DELAUNAY),
      computes projected area instead for last coordinate
    assumes facet->normal exists
    projecting tricoplanar facets to the hyperplane does not appear to make a difference

  design:
    if simplicial
      compute area
    else
      for each ridge
        compute area from centrum to ridge
    negate area if upper Delaunay facet
*/
realT qh_facetarea(qhT *qh, facetT *facet) {
  vertexT *apex;
  pointT *centrum;
  realT area= 0.0;
  ridgeT *ridge, **ridgep;

  if (facet->simplicial) {
    apex= SETfirstt_(facet->vertices, vertexT);
    area= qh_facetarea_simplex(qh, qh->hull_dim, apex->point, facet->vertices,
                    apex, facet->toporient, facet->normal, &facet->offset);
  }else {
    if (qh->CENTERtype == qh_AScentrum)
      centrum= facet->center;
    else
      centrum= qh_getcentrum(qh, facet);
    FOREACHridge_(facet->ridges)
      area += qh_facetarea_simplex(qh, qh->hull_dim, centrum, ridge->vertices,
                 NULL, (boolT)(ridge->top == facet),  facet->normal, &facet->offset);
    if (qh->CENTERtype != qh_AScentrum)
      qh_memfree(qh, centrum, qh->normal_size);
  }
  if (facet->upperdelaunay && qh->DELAUNAY)
    area= -area;  /* the normal should be [0,...,1] */
  trace4((qh, qh->ferr, 4009, "qh_facetarea: f%d area %2.2g\n", facet->id, area));
  return area;
} /* facetarea */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="facetarea_simplex">-</a>

  qh_facetarea_simplex(qh, dim, apex, vertices, notvertex, toporient, normal, offset )
    return area for a simplex defined by
      an apex, a base of vertices, an orientation, and a unit normal
    if simplicial or tricoplanar facet,
      notvertex is defined and it is skipped in vertices

  returns:
    computes area of simplex projected to plane [normal,offset]
    returns 0 if vertex too far below plane (qh->WIDEfacet)
      vertex can't be apex of tricoplanar facet

  notes:
    if (qh->DELAUNAY),
      computes projected area instead for last coordinate
    uses qh->gm_matrix/gm_row and qh->hull_dim
    helper function for qh_facetarea

  design:
    if Notvertex
      translate simplex to apex
    else
      project simplex to normal/offset
      translate simplex to apex
    if Delaunay
      set last row/column to 0 with -1 on diagonal
    else
      set last row to Normal
    compute determinate
    scale and flip sign for area
*/
realT qh_facetarea_simplex(qhT *qh, int dim, coordT *apex, setT *vertices,
        vertexT *notvertex,  boolT toporient, coordT *normal, realT *offset) {
  pointT *coorda, *coordp, *gmcoord;
  coordT **rows, *normalp;
  int k,  i=0;
  realT area, dist;
  vertexT *vertex, **vertexp;
  boolT nearzero;

  gmcoord= qh->gm_matrix;
  rows= qh->gm_row;
  FOREACHvertex_(vertices) {
    if (vertex == notvertex)
      continue;
    rows[i++]= gmcoord;
    coorda= apex;
    coordp= vertex->point;
    normalp= normal;
    if (notvertex) {
      for (k=dim; k--; )
        *(gmcoord++)= *coordp++ - *coorda++;
    }else {
      dist= *offset;
      for (k=dim; k--; )
        dist += *coordp++ * *normalp++;
      if (dist < -qh->WIDEfacet) {
        zinc_(Znoarea);
        return 0.0;
      }
      coordp= vertex->point;
      normalp= normal;
      for (k=dim; k--; )
        *(gmcoord++)= (*coordp++ - dist * *normalp++) - *coorda++;
    }
  }
  if (i != dim-1) {
    qh_fprintf(qh, qh->ferr, 6008, "qhull internal error (qh_facetarea_simplex): #points %d != dim %d -1\n",
               i, dim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  rows[i]= gmcoord;
  if (qh->DELAUNAY) {
    for (i=0; i < dim-1; i++)
      rows[i][dim-1]= 0.0;
    for (k=dim; k--; )
      *(gmcoord++)= 0.0;
    rows[dim-1][dim-1]= -1.0;
  }else {
    normalp= normal;
    for (k=dim; k--; )
      *(gmcoord++)= *normalp++;
  }
  zinc_(Zdetfacetarea);
  area= qh_determinant(qh, rows, dim, &nearzero);
  if (toporient)
    area= -area;
  area *= qh->AREAfactor;
  trace4((qh, qh->ferr, 4010, "qh_facetarea_simplex: area=%2.2g for point p%d, toporient %d, nearzero? %d\n",
          area, qh_pointid(qh, apex), toporient, nearzero));
  return area;
} /* facetarea_simplex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="facetcenter">-</a>

  qh_facetcenter(qh, vertices )
    return Voronoi center (Voronoi vertex) for a facet's vertices

  returns:
    return temporary point equal to the center

  see:
    qh_voronoi_center()
*/
pointT *qh_facetcenter(qhT *qh, setT *vertices) {
  setT *points= qh_settemp(qh, qh_setsize(qh, vertices));
  vertexT *vertex, **vertexp;
  pointT *center;

  FOREACHvertex_(vertices)
    qh_setappend(qh, &points, vertex->point);
  center= qh_voronoi_center(qh, qh->hull_dim-1, points);
  qh_settempfree(qh, &points);
  return center;
} /* facetcenter */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="findgooddist">-</a>

  qh_findgooddist(qh, point, facetA, dist, facetlist )
    find best good facet visible for point from facetA
    assumes facetA is visible from point

  returns:
    best facet, i.e., good facet that is furthest from point
      distance to best facet
      NULL if none

    moves good, visible facets (and some other visible facets)
      to end of qh->facet_list

  notes:
    uses qh->visit_id

  design:
    initialize bestfacet if facetA is good
    move facetA to end of facetlist
    for each facet on facetlist
      for each unvisited neighbor of facet
        move visible neighbors to end of facetlist
        update best good neighbor
        if no good neighbors, update best facet
*/
facetT *qh_findgooddist(qhT *qh, pointT *point, facetT *facetA, realT *distp,
               facetT **facetlist) {
  realT bestdist= -REALmax, dist;
  facetT *neighbor, **neighborp, *bestfacet=NULL, *facet;
  boolT goodseen= False;

  if (facetA->good) {
    zzinc_(Zcheckpart);  /* calls from check_bestdist occur after print stats */
    qh_distplane(qh, point, facetA, &bestdist);
    bestfacet= facetA;
    goodseen= True;
  }
  qh_removefacet(qh, facetA);
  qh_appendfacet(qh, facetA);
  *facetlist= facetA;
  facetA->visitid= ++qh->visit_id;
  FORALLfacet_(*facetlist) {
    FOREACHneighbor_(facet) {
      if (neighbor->visitid == qh->visit_id)
        continue;
      neighbor->visitid= qh->visit_id;
      if (goodseen && !neighbor->good)
        continue;
      zzinc_(Zcheckpart);
      qh_distplane(qh, point, neighbor, &dist);
      if (dist > 0) {
        qh_removefacet(qh, neighbor);
        qh_appendfacet(qh, neighbor);
        if (neighbor->good) {
          goodseen= True;
          if (dist > bestdist) {
            bestdist= dist;
            bestfacet= neighbor;
          }
        }
      }
    }
  }
  if (bestfacet) {
    *distp= bestdist;
    trace2((qh, qh->ferr, 2003, "qh_findgooddist: p%d is %2.2g above good facet f%d\n",
      qh_pointid(qh, point), bestdist, bestfacet->id));
    return bestfacet;
  }
  trace4((qh, qh->ferr, 4011, "qh_findgooddist: no good facet for p%d above f%d\n",
      qh_pointid(qh, point), facetA->id));
  return NULL;
}  /* findgooddist */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="furthestnewvertex">-</a>

  qh_furthestnewvertex(qh, unvisited, facet, &maxdist )
    return furthest unvisited, new vertex to a facet

  return:
    NULL if no vertex is above facet
    maxdist to facet
    updates v.visitid

  notes:
    Ignores vertices in facetB
    Does not change qh.vertex_visit.  Use in conjunction with qh_furthestvertex
*/
vertexT *qh_furthestnewvertex(qhT *qh, unsigned int unvisited, facetT *facet, realT *maxdistp /* qh.newvertex_list */) {
  vertexT *maxvertex= NULL, *vertex;
  coordT dist, maxdist= 0.0;

  FORALLvertex_(qh->newvertex_list) {
    if (vertex->newfacet && vertex->visitid <= unvisited) {
      vertex->visitid= qh->vertex_visit;
      qh_distplane(qh, vertex->point, facet, &dist);
      if (dist > maxdist) {
        maxdist= dist;
        maxvertex= vertex;
      }
    }
  }
  trace4((qh, qh->ferr, 4085, "qh_furthestnewvertex: v%d dist %2.2g is furthest new vertex for f%d\n",
    getid_(maxvertex), maxdist, facet->id));
  *maxdistp= maxdist;
  return maxvertex;
} /* furthestnewvertex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="furthestvertex">-</a>

  qh_furthestvertex(qh, facetA, facetB, &maxdist, &mindist )
    return furthest vertex in facetA from facetB, or NULL if none

  return:
    maxdist and mindist to facetB or 0.0 if none
    updates qh.vertex_visit

  notes:
    Ignores vertices in facetB
*/
vertexT *qh_furthestvertex(qhT *qh, facetT *facetA, facetT *facetB, realT *maxdistp, realT *mindistp) {
  vertexT *maxvertex= NULL, *vertex, **vertexp;
  coordT dist, maxdist= -REALmax, mindist= REALmax;

  qh->vertex_visit++;
  FOREACHvertex_(facetB->vertices)
    vertex->visitid= qh->vertex_visit;
  FOREACHvertex_(facetA->vertices) {
    if (vertex->visitid != qh->vertex_visit) {
      vertex->visitid= qh->vertex_visit;
      zzinc_(Zvertextests);
      qh_distplane(qh, vertex->point, facetB, &dist);
      if (!maxvertex) {
        maxdist= dist;
        mindist= dist;
        maxvertex= vertex;
      }else if (dist > maxdist) {
        maxdist= dist;
        maxvertex= vertex;
      }else if (dist < mindist)
        mindist= dist;
    }
  }
  if (!maxvertex) {
    trace3((qh, qh->ferr, 3067, "qh_furthestvertex: all vertices of f%d are in f%d.  Returning 0.0 for max and mindist\n",
      facetA->id, facetB->id));
    maxdist= mindist= 0.0;
  }else {
    trace4((qh, qh->ferr, 4084, "qh_furthestvertex: v%d dist %2.2g is furthest (mindist %2.2g) of f%d above f%d\n",
      maxvertex->id, maxdist, mindist, facetA->id, facetB->id));
  }
  *maxdistp= maxdist;
  *mindistp= mindist;
  return maxvertex;
} /* furthestvertex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="getarea">-</a>

  qh_getarea(qh, facetlist )
    set area of all facets in facetlist
    collect statistics
    nop if hasAreaVolume

  returns:
    sets qh->totarea/totvol to total area and volume of convex hull
    for Delaunay triangulation, computes projected area of the lower or upper hull
      ignores upper hull if qh->ATinfinity

  notes:
    could compute outer volume by expanding facet area by rays from interior
    the following attempt at perpendicular projection underestimated badly:
      qh.totoutvol += (-dist + facet->maxoutside + qh->DISTround)
                            * area/ qh->hull_dim;
  design:
    for each facet on facetlist
      compute facet->area
      update qh.totarea and qh.totvol
*/
void qh_getarea(qhT *qh, facetT *facetlist) {
  realT area;
  realT dist;
  facetT *facet;

  if (qh->hasAreaVolume)
    return;
  if (qh->REPORTfreq)
    qh_fprintf(qh, qh->ferr, 8020, "computing area of each facet and volume of the convex hull\n");
  else
    trace1((qh, qh->ferr, 1001, "qh_getarea: computing area for each facet and its volume to qh.interior_point (dist*area/dim)\n"));
  qh->totarea= qh->totvol= 0.0;
  FORALLfacet_(facetlist) {
    if (!facet->normal)
      continue;
    if (facet->upperdelaunay && qh->ATinfinity)
      continue;
    if (!facet->isarea) {
      facet->f.area= qh_facetarea(qh, facet);
      facet->isarea= True;
    }
    area= facet->f.area;
    if (qh->DELAUNAY) {
      if (facet->upperdelaunay == qh->UPPERdelaunay)
        qh->totarea += area;
    }else {
      qh->totarea += area;
      qh_distplane(qh, qh->interior_point, facet, &dist);
      qh->totvol += -dist * area/ qh->hull_dim;
    }
    if (qh->PRINTstatistics) {
      wadd_(Wareatot, area);
      wmax_(Wareamax, area);
      wmin_(Wareamin, area);
    }
  }
  qh->hasAreaVolume= True;
} /* getarea */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="gram_schmidt">-</a>

  qh_gram_schmidt(qh, dim, row )
    implements Gram-Schmidt orthogonalization by rows

  returns:
    false if zero norm
    overwrites rows[dim][dim]

  notes:
    see Golub & van Loan, 1983, Algorithm 6.2-2, "Modified Gram-Schmidt"
    overflow due to small divisors not handled

  design:
    for each row
      compute norm for row
      if non-zero, normalize row
      for each remaining rowA
        compute inner product of row and rowA
        reduce rowA by row * inner product
*/
boolT qh_gram_schmidt(qhT *qh, int dim, realT **row) {
  realT *rowi, *rowj, norm;
  int i, j, k;

  for (i=0; i < dim; i++) {
    rowi= row[i];
    for (norm=0.0, k=dim; k--; rowi++)
      norm += *rowi * *rowi;
    norm= sqrt(norm);
    wmin_(Wmindenom, norm);
    if (norm == 0.0)  /* either 0 or overflow due to sqrt */
      return False;
    for (k=dim; k--; )
      *(--rowi) /= norm;
    for (j=i+1; j < dim; j++) {
      rowj= row[j];
      for (norm=0.0, k=dim; k--; )
        norm += *rowi++ * *rowj++;
      for (k=dim; k--; )
        *(--rowj) -= *(--rowi) * norm;
    }
  }
  return True;
} /* gram_schmidt */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="inthresholds">-</a>

  qh_inthresholds(qh, normal, angle )
    return True if normal within qh.lower_/upper_threshold

  returns:
    estimate of angle by summing of threshold diffs
      angle may be NULL
      smaller "angle" is better

  notes:
    invalid if qh.SPLITthresholds

  see:
    qh.lower_threshold in qh_initbuild()
    qh_initthresholds()

  design:
    for each dimension
      test threshold
*/
boolT qh_inthresholds(qhT *qh, coordT *normal, realT *angle) {
  boolT within= True;
  int k;
  realT threshold;

  if (angle)
    *angle= 0.0;
  for (k=0; k < qh->hull_dim; k++) {
    threshold= qh->lower_threshold[k];
    if (threshold > -REALmax/2) {
      if (normal[k] < threshold)
        within= False;
      if (angle) {
        threshold -= normal[k];
        *angle += fabs_(threshold);
      }
    }
    if (qh->upper_threshold[k] < REALmax/2) {
      threshold= qh->upper_threshold[k];
      if (normal[k] > threshold)
        within= False;
      if (angle) {
        threshold -= normal[k];
        *angle += fabs_(threshold);
      }
    }
  }
  return within;
} /* inthresholds */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="joggleinput">-</a>

  qh_joggleinput(qh)
    randomly joggle input to Qhull by qh.JOGGLEmax
    initial input is qh.first_point/qh.num_points of qh.hull_dim
      repeated calls use qh.input_points/qh.num_points

  returns:
    joggles points at qh.first_point/qh.num_points
    copies data to qh.input_points/qh.input_malloc if first time
    determines qh.JOGGLEmax if it was zero
    if qh.DELAUNAY
      computes the Delaunay projection of the joggled points

  notes:
    if qh.DELAUNAY, unnecessarily joggles the last coordinate
    the initial 'QJn' may be set larger than qh_JOGGLEmaxincrease

  design:
    if qh.DELAUNAY
      set qh.SCALElast for reduced precision errors
    if first call
      initialize qh.input_points to the original input points
      if qh.JOGGLEmax == 0
        determine default qh.JOGGLEmax
    else
      increase qh.JOGGLEmax according to qh.build_cnt
    joggle the input by adding a random number in [-qh.JOGGLEmax,qh.JOGGLEmax]
    if qh.DELAUNAY
      sets the Delaunay projection
*/
void qh_joggleinput(qhT *qh) {
  int i, seed, size;
  coordT *coordp, *inputp;
  realT randr, randa, randb;

  if (!qh->input_points) { /* first call */
    qh->input_points= qh->first_point;
    qh->input_malloc= qh->POINTSmalloc;
    size= qh->num_points * qh->hull_dim * (int)sizeof(coordT);
    if (!(qh->first_point= (coordT *)qh_malloc((size_t)size))) {
      qh_fprintf(qh, qh->ferr, 6009, "qhull error: insufficient memory to joggle %d points\n",
          qh->num_points);
      qh_errexit(qh, qh_ERRmem, NULL, NULL);
    }
    qh->POINTSmalloc= True;
    if (qh->JOGGLEmax == 0.0) {
      qh->JOGGLEmax= qh_detjoggle(qh, qh->input_points, qh->num_points, qh->hull_dim);
      qh_option(qh, "QJoggle", NULL, &qh->JOGGLEmax);
    }
  }else {                 /* repeated call */
    if (!qh->RERUN && qh->build_cnt > qh_JOGGLEretry) {
      if (((qh->build_cnt-qh_JOGGLEretry-1) % qh_JOGGLEagain) == 0) {
        realT maxjoggle= qh->MAXwidth * qh_JOGGLEmaxincrease;
        if (qh->JOGGLEmax < maxjoggle) {
          qh->JOGGLEmax *= qh_JOGGLEincrease;
          minimize_(qh->JOGGLEmax, maxjoggle);
        }
      }
    }
    qh_option(qh, "QJoggle", NULL, &qh->JOGGLEmax);
  }
  if (qh->build_cnt > 1 && qh->JOGGLEmax > fmax_(qh->MAXwidth/4, 0.1)) {
      qh_fprintf(qh, qh->ferr, 6010, "qhull input error (qh_joggleinput): the current joggle for 'QJn', %.2g, is too large for the width\nof the input.  If possible, recompile Qhull with higher-precision reals.\n",
                qh->JOGGLEmax);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  /* for some reason, using qh->ROTATErandom and qh_RANDOMseed does not repeat the run. Use 'TRn' instead */
  seed= qh_RANDOMint;
  qh_option(qh, "_joggle-seed", &seed, NULL);
  trace0((qh, qh->ferr, 6, "qh_joggleinput: joggle input by %4.4g with seed %d\n",
    qh->JOGGLEmax, seed));
  inputp= qh->input_points;
  coordp= qh->first_point;
  randa= 2.0 * qh->JOGGLEmax/qh_RANDOMmax;
  randb= -qh->JOGGLEmax;
  size= qh->num_points * qh->hull_dim;
  for (i=size; i--; ) {
    randr= qh_RANDOMint;
    *(coordp++)= *(inputp++) + (randr * randa + randb);
  }
  if (qh->DELAUNAY) {
    qh->last_low= qh->last_high= qh->last_newhigh= REALmax;
    qh_setdelaunay(qh, qh->hull_dim, qh->num_points, qh->first_point);
  }
} /* joggleinput */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxabsval">-</a>

  qh_maxabsval( normal, dim )
    return pointer to maximum absolute value of a dim vector
    returns NULL if dim=0
*/
realT *qh_maxabsval(realT *normal, int dim) {
  realT maxval= -REALmax;
  realT *maxp= NULL, *colp, absval;
  int k;

  for (k=dim, colp= normal; k--; colp++) {
    absval= fabs_(*colp);
    if (absval > maxval) {
      maxval= absval;
      maxp= colp;
    }
  }
  return maxp;
} /* maxabsval */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxmin">-</a>

  qh_maxmin(qh, points, numpoints, dimension )
    return max/min points for each dimension
    determine max and min coordinates

  returns:
    returns a temporary set of max and min points
      may include duplicate points. Does not include qh.GOODpoint
    sets qh.NEARzero, qh.MAXabs_coord, qh.MAXsumcoord, qh.MAXwidth
         qh.MAXlastcoord, qh.MINlastcoord
    initializes qh.max_outside, qh.min_vertex, qh.WAScoplanar, qh.ZEROall_ok

  notes:
    loop duplicated in qh_detjoggle()

  design:
    initialize global precision variables
    checks definition of REAL...
    for each dimension
      for each point
        collect maximum and minimum point
      collect maximum of maximums and minimum of minimums
      determine qh.NEARzero for Gaussian Elimination
*/
setT *qh_maxmin(qhT *qh, pointT *points, int numpoints, int dimension) {
  int k;
  realT maxcoord, temp;
  pointT *minimum, *maximum, *point, *pointtemp;
  setT *set;

  qh->max_outside= 0.0;
  qh->MAXabs_coord= 0.0;
  qh->MAXwidth= -REALmax;
  qh->MAXsumcoord= 0.0;
  qh->min_vertex= 0.0;
  qh->WAScoplanar= False;
  if (qh->ZEROcentrum)
    qh->ZEROall_ok= True;
  if (REALmin < REALepsilon && REALmin < REALmax && REALmin > -REALmax
  && REALmax > 0.0 && -REALmax < 0.0)
    ; /* all ok */
  else {
    qh_fprintf(qh, qh->ferr, 6011, "qhull error: one or more floating point constants in user_r.h are inconsistent. REALmin %g, -REALmax %g, 0.0, REALepsilon %g, REALmax %g\n",
          REALmin, -REALmax, REALepsilon, REALmax);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  set= qh_settemp(qh, 2*dimension);
  trace1((qh, qh->ferr, 8082, "qh_maxmin: dim             min             max           width    nearzero  min-point  max-point\n"));
  for (k=0; k < dimension; k++) {
    if (points == qh->GOODpointp)
      minimum= maximum= points + dimension;
    else
      minimum= maximum= points;
    FORALLpoint_(qh, points, numpoints) {
      if (point == qh->GOODpointp)
        continue;
      if (maximum[k] < point[k])
        maximum= point;
      else if (minimum[k] > point[k])
        minimum= point;
    }
    if (k == dimension-1) {
      qh->MINlastcoord= minimum[k];
      qh->MAXlastcoord= maximum[k];
    }
    if (qh->SCALElast && k == dimension-1)
      maxcoord= qh->MAXabs_coord;
    else {
      maxcoord= fmax_(maximum[k], -minimum[k]);
      if (qh->GOODpointp) {
        temp= fmax_(qh->GOODpointp[k], -qh->GOODpointp[k]);
        maximize_(maxcoord, temp);
      }
      temp= maximum[k] - minimum[k];
      maximize_(qh->MAXwidth, temp);
    }
    maximize_(qh->MAXabs_coord, maxcoord);
    qh->MAXsumcoord += maxcoord;
    qh_setappend(qh, &set, minimum);
    qh_setappend(qh, &set, maximum);
    /* calculation of qh NEARzero is based on Golub & van Loan, 1983,
       Eq. 4.4-13 for "Gaussian elimination with complete pivoting".
       Golub & van Loan say that n^3 can be ignored and 10 be used in
       place of rho */
    qh->NEARzero[k]= 80 * qh->MAXsumcoord * REALepsilon;
    trace1((qh, qh->ferr, 8106, "           %3d % 14.8e % 14.8e % 14.8e  %4.4e  p%-9d p%-d\n",
            k, minimum[k], maximum[k], maximum[k]-minimum[k], qh->NEARzero[k], qh_pointid(qh, minimum), qh_pointid(qh, maximum)));
    if (qh->SCALElast && k == dimension-1)
      trace1((qh, qh->ferr, 8107, "           last coordinate scaled to (%4.4g, %4.4g), width %4.4e for option 'Qbb'\n",
            qh->MAXabs_coord - qh->MAXwidth, qh->MAXabs_coord, qh->MAXwidth));
  }
  if (qh->IStracing >= 1)
    qh_printpoints(qh, qh->ferr, "qh_maxmin: found the max and min points (by dim):", set);
  return(set);
} /* maxmin */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxouter">-</a>

  qh_maxouter(qh)
    return maximum distance from facet to outer plane
    normally this is qh.max_outside+qh.DISTround
    does not include qh.JOGGLEmax

  see:
    qh_outerinner()

  notes:
    need to add another qh.DISTround if testing actual point with computation
    see qh_detmaxoutside for a qh_RATIO... target

  for joggle:
    qh_setfacetplane() updated qh.max_outer for Wnewvertexmax (max distance to vertex)
    need to use Wnewvertexmax since could have a coplanar point for a high
      facet that is replaced by a low facet
    need to add qh.JOGGLEmax if testing input points
*/
realT qh_maxouter(qhT *qh) {
  realT dist;

  dist= fmax_(qh->max_outside, qh->DISTround);
  dist += qh->DISTround;
  trace4((qh, qh->ferr, 4012, "qh_maxouter: max distance from facet to outer plane is %4.4g, qh.max_outside is %4.4g\n", dist, qh->max_outside));
  return dist;
} /* maxouter */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="maxsimplex">-</a>

  qh_maxsimplex(qh, dim, maxpoints, points, numpoints, simplex )
    determines maximum simplex for a set of points
    maxpoints is the subset of points with a min or max coordinate
    may start with points already in simplex
    skips qh.GOODpointp (assumes that it isn't in maxpoints)

  returns:
    simplex with dim+1 points

  notes:
    called by qh_initialvertices, qh_detvnorm, and qh_voronoi_center
    requires qh.MAXwidth to estimate determinate for each vertex
    assumes at least needed points in points
    maximizes determinate for x,y,z,w, etc.
    uses maxpoints as long as determinate is clearly non-zero

  design:
    initialize simplex with at least two points
      (find points with max or min x coordinate)
    create a simplex of dim+1 vertices as follows
      add point from maxpoints that maximizes the determinate of the point and the simplex vertices  
      if last point and maxdet/prevdet < qh_RATIOmaxsimplex (3.0e-2)
        flag maybe_falsenarrow
      if no maxpoint or maxnearzero or maybe_falsenarrow
        search all points for maximum determinate
        early exit if maybe_falsenarrow and !maxnearzero and maxdet > prevdet
*/
void qh_maxsimplex(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints, setT **simplex) {
  pointT *point, **pointp, *pointtemp, *maxpoint, *minx=NULL, *maxx=NULL;
  boolT nearzero, maxnearzero= False, maybe_falsenarrow;
  int i, sizinit;
  realT maxdet= -1.0, prevdet= -1.0, det, mincoord= REALmax, maxcoord= -REALmax, mindet, ratio, targetdet;

  if (qh->MAXwidth <= 0.0) {
    qh_fprintf(qh, qh->ferr, 6421, "qhull internal error (qh_maxsimplex): qh.MAXwidth required for qh_maxsimplex.  Used to estimate determinate\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  sizinit= qh_setsize(qh, *simplex);
  if (sizinit >= 2) {
    maxdet= pow(qh->MAXwidth, sizinit - 1);
  }else {
    if (qh_setsize(qh, maxpoints) >= 2) {
      FOREACHpoint_(maxpoints) {
        if (maxcoord < point[0]) {
          maxcoord= point[0];
          maxx= point;
        }
        if (mincoord > point[0]) {
          mincoord= point[0];
          minx= point;
        }
      }
    }else {
      FORALLpoint_(qh, points, numpoints) {
        if (point == qh->GOODpointp)
          continue;
        if (maxcoord < point[0]) {
          maxcoord= point[0];
          maxx= point;
        }
        if (mincoord > point[0]) {
          mincoord= point[0];
          minx= point;
        }
      }
    }
    maxdet= maxcoord - mincoord;
    qh_setunique(qh, simplex, minx);
    if (qh_setsize(qh, *simplex) < 2)
      qh_setunique(qh, simplex, maxx);
    sizinit= qh_setsize(qh, *simplex);
    if (sizinit < 2) {
      qh_joggle_restart(qh, "input has same x coordinate");
      if (zzval_(Zsetplane) > qh->hull_dim+1) {
        qh_fprintf(qh, qh->ferr, 6012, "qhull precision error (qh_maxsimplex for voronoi_center): %d points with the same x coordinate %4.4g\n",
                 qh_setsize(qh, maxpoints)+numpoints, mincoord);
        qh_errexit(qh, qh_ERRprec, NULL, NULL);
      }else {
        qh_fprintf(qh, qh->ferr, 6013, "qhull input error: input is less than %d-dimensional since all points have the same x coordinate %4.4g\n",
                 qh->hull_dim, mincoord);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
    }
  }
  for (i=sizinit; i < dim+1; i++) {
    prevdet= maxdet;
    maxpoint= NULL;
    maxdet= -1.0;
    FOREACHpoint_(maxpoints) {
      if (!qh_setin(*simplex, point) && point != maxpoint) {
        det= qh_detsimplex(qh, point, *simplex, i, &nearzero); /* retests maxpoints if duplicate or multiple iterations */
        if ((det= fabs_(det)) > maxdet) {
          maxdet= det;
          maxpoint= point;
          maxnearzero= nearzero;
        }
      }
    }
    maybe_falsenarrow= False;
    ratio= 1.0;
    targetdet= prevdet * qh->MAXwidth;
    mindet= 10 * qh_RATIOmaxsimplex * targetdet;
    if (maxdet > 0.0) {
      ratio= maxdet / targetdet;
      if (ratio < qh_RATIOmaxsimplex)
        maybe_falsenarrow= True;
    }
    if (!maxpoint || maxnearzero || maybe_falsenarrow) {
      zinc_(Zsearchpoints);
      if (!maxpoint) {
        trace0((qh, qh->ferr, 7, "qh_maxsimplex: searching all points for %d-th initial vertex, better than mindet %4.4g, targetdet %4.4g\n",
                i+1, mindet, targetdet));
      }else if (qh->ALLpoints) {
        trace0((qh, qh->ferr, 30, "qh_maxsimplex: searching all points ('Qs') for %d-th initial vertex, better than p%d det %4.4g, targetdet %4.4g, ratio %4.4g\n",
                i+1, qh_pointid(qh, maxpoint), maxdet, targetdet, ratio));
      }else if (maybe_falsenarrow) {
        trace0((qh, qh->ferr, 17, "qh_maxsimplex: searching all points for %d-th initial vertex, better than p%d det %4.4g and mindet %4.4g, ratio %4.4g\n",
                i+1, qh_pointid(qh, maxpoint), maxdet, mindet, ratio));
      }else {
        trace0((qh, qh->ferr, 8, "qh_maxsimplex: searching all points for %d-th initial vertex, better than p%d det %2.2g and mindet %4.4g, targetdet %4.4g\n",
                i+1, qh_pointid(qh, maxpoint), maxdet, mindet, targetdet));
      }
      FORALLpoint_(qh, points, numpoints) {
        if (point == qh->GOODpointp)
          continue;
        if (!qh_setin(maxpoints, point) && !qh_setin(*simplex, point)) {
          det= qh_detsimplex(qh, point, *simplex, i, &nearzero);
          if ((det= fabs_(det)) > maxdet) {
            maxdet= det;
            maxpoint= point;
            maxnearzero= nearzero;
            if (!maxnearzero && !qh->ALLpoints && maxdet > mindet)
              break;
          }
        }
      }
    } /* !maxpoint */
    if (!maxpoint) {
      qh_fprintf(qh, qh->ferr, 6014, "qhull internal error (qh_maxsimplex): not enough points available\n");
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    qh_setappend(qh, simplex, maxpoint);
    trace1((qh, qh->ferr, 1002, "qh_maxsimplex: selected point p%d for %d`th initial vertex, det=%4.4g, targetdet=%4.4g, mindet=%4.4g\n",
            qh_pointid(qh, maxpoint), i+1, maxdet, prevdet * qh->MAXwidth, mindet));
  } /* i */
} /* maxsimplex */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="minabsval">-</a>

  qh_minabsval( normal, dim )
    return minimum absolute value of a dim vector
*/
realT qh_minabsval(realT *normal, int dim) {
  realT minval= 0;
  realT maxval= 0;
  realT *colp;
  int k;

  for (k=dim, colp=normal; k--; colp++) {
    maximize_(maxval, *colp);
    minimize_(minval, *colp);
  }
  return fmax_(maxval, -minval);
} /* minabsval */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="mindiff">-</a>

  qh_mindif(qh, vecA, vecB, dim )
    return index of min abs. difference of two vectors
*/
int qh_mindiff(realT *vecA, realT *vecB, int dim) {
  realT mindiff= REALmax, diff;
  realT *vecAp= vecA, *vecBp= vecB;
  int k, mink= 0;

  for (k=0; k < dim; k++) {
    diff= *vecAp++ - *vecBp++;
    diff= fabs_(diff);
    if (diff < mindiff) {
      mindiff= diff;
      mink= k;
    }
  }
  return mink;
} /* mindiff */



/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="orientoutside">-</a>

  qh_orientoutside(qh, facet  )
    make facet outside oriented via qh.interior_point

  returns:
    True if facet reversed orientation.
*/
boolT qh_orientoutside(qhT *qh, facetT *facet) {
  int k;
  realT dist;

  qh_distplane(qh, qh->interior_point, facet, &dist);
  if (dist > 0) {
    for (k=qh->hull_dim; k--; )
      facet->normal[k]= -facet->normal[k];
    facet->offset= -facet->offset;
    return True;
  }
  return False;
} /* orientoutside */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="outerinner">-</a>

  qh_outerinner(qh, facet, outerplane, innerplane  )
    if facet and qh.maxoutdone (i.e., qh_check_maxout)
      returns outer and inner plane for facet
    else
      returns maximum outer and inner plane
    accounts for qh.JOGGLEmax

  see:
    qh_maxouter(qh), qh_check_bestdist(), qh_check_points()

  notes:
    outerplaner or innerplane may be NULL
    facet is const
    Does not error (QhullFacet)

    includes qh.DISTround for actual points
    adds another qh.DISTround if testing with floating point arithmetic
*/
void qh_outerinner(qhT *qh, facetT *facet, realT *outerplane, realT *innerplane) {
  realT dist, mindist;
  vertexT *vertex, **vertexp;

  if (outerplane) {
    if (!qh_MAXoutside || !facet || !qh->maxoutdone) {
      *outerplane= qh_maxouter(qh);       /* includes qh.DISTround */
    }else { /* qh_MAXoutside ... */
#if qh_MAXoutside
      *outerplane= facet->maxoutside + qh->DISTround;
#endif

    }
    if (qh->JOGGLEmax < REALmax/2)
      *outerplane += qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
  }
  if (innerplane) {
    if (facet) {
      mindist= REALmax;
      FOREACHvertex_(facet->vertices) {
        zinc_(Zdistio);
        qh_distplane(qh, vertex->point, facet, &dist);
        minimize_(mindist, dist);
      }
      *innerplane= mindist - qh->DISTround;
    }else
      *innerplane= qh->min_vertex - qh->DISTround;
    if (qh->JOGGLEmax < REALmax/2)
      *innerplane -= qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
  }
} /* outerinner */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="pointdist">-</a>

  qh_pointdist( point1, point2, dim )
    return distance between two points

  notes:
    returns distance squared if 'dim' is negative
*/
coordT qh_pointdist(pointT *point1, pointT *point2, int dim) {
  coordT dist, diff;
  int k;

  dist= 0.0;
  for (k= (dim > 0 ? dim : -dim); k--; ) {
    diff= *point1++ - *point2++;
    dist += diff * diff;
  }
  if (dim > 0)
    return(sqrt(dist));
  return dist;
} /* pointdist */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="printmatrix">-</a>

  qh_printmatrix(qh, fp, string, rows, numrow, numcol )
    print matrix to fp given by row vectors
    print string as header
    qh may be NULL if fp is defined

  notes:
    print a vector by qh_printmatrix(qh, fp, "", &vect, 1, len)
*/
void qh_printmatrix(qhT *qh, FILE *fp, const char *string, realT **rows, int numrow, int numcol) {
  realT *rowp;
  realT r; /*bug fix*/
  int i,k;

  qh_fprintf(qh, fp, 9001, "%s\n", string);
  for (i=0; i < numrow; i++) {
    rowp= rows[i];
    for (k=0; k < numcol; k++) {
      r= *rowp++;
      qh_fprintf(qh, fp, 9002, "%6.3g ", r);
    }
    qh_fprintf(qh, fp, 9003, "\n");
  }
} /* printmatrix */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="printpoints">-</a>

  qh_printpoints(qh, fp, string, points )
    print pointids to fp for a set of points
    if string, prints string and 'p' point ids
*/
void qh_printpoints(qhT *qh, FILE *fp, const char *string, setT *points) {
  pointT *point, **pointp;

  if (string) {
    qh_fprintf(qh, fp, 9004, "%s", string);
    FOREACHpoint_(points)
      qh_fprintf(qh, fp, 9005, " p%d", qh_pointid(qh, point));
    qh_fprintf(qh, fp, 9006, "\n");
  }else {
    FOREACHpoint_(points)
      qh_fprintf(qh, fp, 9007, " %d", qh_pointid(qh, point));
    qh_fprintf(qh, fp, 9008, "\n");
  }
} /* printpoints */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="projectinput">-</a>

  qh_projectinput(qh)
    project input points using qh.lower_bound/upper_bound and qh->DELAUNAY
    if qh.lower_bound[k]=qh.upper_bound[k]= 0,
      removes dimension k
    if halfspace intersection
      removes dimension k from qh.feasible_point
    input points in qh->first_point, num_points, input_dim

  returns:
    new point array in qh->first_point of qh->hull_dim coordinates
    sets qh->POINTSmalloc
    if qh->DELAUNAY
      projects points to paraboloid
      lowbound/highbound is also projected
    if qh->ATinfinity
      adds point "at-infinity"
    if qh->POINTSmalloc
      frees old point array

  notes:
    checks that qh.hull_dim agrees with qh.input_dim, PROJECTinput, and DELAUNAY


  design:
    sets project[k] to -1 (delete), 0 (keep), 1 (add for Delaunay)
    determines newdim and newnum for qh->hull_dim and qh->num_points
    projects points to newpoints
    projects qh.lower_bound to itself
    projects qh.upper_bound to itself
    if qh->DELAUNAY
      if qh->ATINFINITY
        projects points to paraboloid
        computes "infinity" point as vertex average and 10% above all points
      else
        uses qh_setdelaunay to project points to paraboloid
*/
void qh_projectinput(qhT *qh) {
  int k,i;
  int newdim= qh->input_dim, newnum= qh->num_points;
  signed char *project;
  int projectsize= (qh->input_dim + 1) * (int)sizeof(*project);
  pointT *newpoints, *coord, *infinity;
  realT paraboloid, maxboloid= 0;

  project= (signed char *)qh_memalloc(qh, projectsize);
  memset((char *)project, 0, (size_t)projectsize);
  for (k=0; k < qh->input_dim; k++) {   /* skip Delaunay bound */
    if (qh->lower_bound[k] == 0.0 && qh->upper_bound[k] == 0.0) {
      project[k]= -1;
      newdim--;
    }
  }
  if (qh->DELAUNAY) {
    project[k]= 1;
    newdim++;
    if (qh->ATinfinity)
      newnum++;
  }
  if (newdim != qh->hull_dim) {
    qh_memfree(qh, project, projectsize);
    qh_fprintf(qh, qh->ferr, 6015, "qhull internal error (qh_projectinput): dimension after projection %d != hull_dim %d\n", newdim, qh->hull_dim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (!(newpoints= qh->temp_malloc= (coordT *)qh_malloc((size_t)(newnum * newdim) * sizeof(coordT)))) {
    qh_memfree(qh, project, projectsize);
    qh_fprintf(qh, qh->ferr, 6016, "qhull error: insufficient memory to project %d points\n",
           qh->num_points);
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  /* qh_projectpoints throws error if mismatched dimensions */
  qh_projectpoints(qh, project, qh->input_dim+1, qh->first_point,
                    qh->num_points, qh->input_dim, newpoints, newdim);
  trace1((qh, qh->ferr, 1003, "qh_projectinput: updating lower and upper_bound\n"));
  qh_projectpoints(qh, project, qh->input_dim+1, qh->lower_bound,
                    1, qh->input_dim+1, qh->lower_bound, newdim+1);
  qh_projectpoints(qh, project, qh->input_dim+1, qh->upper_bound,
                    1, qh->input_dim+1, qh->upper_bound, newdim+1);
  if (qh->HALFspace) {
    if (!qh->feasible_point) {
      qh_memfree(qh, project, projectsize);
      qh_fprintf(qh, qh->ferr, 6017, "qhull internal error (qh_projectinput): HALFspace defined without qh.feasible_point\n");
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    qh_projectpoints(qh, project, qh->input_dim, qh->feasible_point,
                      1, qh->input_dim, qh->feasible_point, newdim);
  }
  qh_memfree(qh, project, projectsize);
  if (qh->POINTSmalloc)
    qh_free(qh->first_point);
  qh->first_point= newpoints;
  qh->POINTSmalloc= True;
  qh->temp_malloc= NULL;
  if (qh->DELAUNAY && qh->ATinfinity) {
    coord= qh->first_point;
    infinity= qh->first_point + qh->hull_dim * qh->num_points;
    for (k=qh->hull_dim-1; k--; )
      infinity[k]= 0.0;
    for (i=qh->num_points; i--; ) {
      paraboloid= 0.0;
      for (k=0; k < qh->hull_dim-1; k++) {
        paraboloid += *coord * *coord;
        infinity[k] += *coord;
        coord++;
      }
      *(coord++)= paraboloid;
      maximize_(maxboloid, paraboloid);
    }
    /* coord == infinity */
    for (k=qh->hull_dim-1; k--; )
      *(coord++) /= qh->num_points;
    *(coord++)= maxboloid * 1.1;
    qh->num_points++;
    trace0((qh, qh->ferr, 9, "qh_projectinput: projected points to paraboloid for Delaunay\n"));
  }else if (qh->DELAUNAY)  /* !qh->ATinfinity */
    qh_setdelaunay(qh, qh->hull_dim, qh->num_points, qh->first_point);
} /* projectinput */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="projectpoints">-</a>

  qh_projectpoints(qh, project, n, points, numpoints, dim, newpoints, newdim )
    project points/numpoints/dim to newpoints/newdim
    if project[k] == -1
      delete dimension k
    if project[k] == 1
      add dimension k by duplicating previous column
    n is size of project

  notes:
    newpoints may be points if only adding dimension at end

  design:
    check that 'project' and 'newdim' agree
    for each dimension
      if project == -1
        skip dimension
      else
        determine start of column in newpoints
        determine start of column in points
          if project == +1, duplicate previous column
        copy dimension (column) from points to newpoints
*/
void qh_projectpoints(qhT *qh, signed char *project, int n, realT *points,
        int numpoints, int dim, realT *newpoints, int newdim) {
  int testdim= dim, oldk=0, newk=0, i,j=0,k;
  realT *newp, *oldp;

  for (k=0; k < n; k++)
    testdim += project[k];
  if (testdim != newdim) {
    qh_fprintf(qh, qh->ferr, 6018, "qhull internal error (qh_projectpoints): newdim %d should be %d after projection\n",
      newdim, testdim);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  for (j=0; j<n; j++) {
    if (project[j] == -1)
      oldk++;
    else {
      newp= newpoints+newk++;
      if (project[j] == +1) {
        if (oldk >= dim)
          continue;
        oldp= points+oldk;
      }else
        oldp= points+oldk++;
      for (i=numpoints; i--; ) {
        *newp= *oldp;
        newp += newdim;
        oldp += dim;
      }
    }
    if (oldk >= dim)
      break;
  }
  trace1((qh, qh->ferr, 1004, "qh_projectpoints: projected %d points from dim %d to dim %d\n",
    numpoints, dim, newdim));
} /* projectpoints */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="rotateinput">-</a>

  qh_rotateinput(qh, rows )
    rotate input using row matrix
    input points given by qh->first_point, num_points, hull_dim
    assumes rows[dim] is a scratch buffer
    if qh->POINTSmalloc, overwrites input points, else mallocs a new array

  returns:
    rotated input
    sets qh->POINTSmalloc

  design:
    see qh_rotatepoints
*/
void qh_rotateinput(qhT *qh, realT **rows) {

  if (!qh->POINTSmalloc) {
    qh->first_point= qh_copypoints(qh, qh->first_point, qh->num_points, qh->hull_dim);
    qh->POINTSmalloc= True;
  }
  qh_rotatepoints(qh, qh->first_point, qh->num_points, qh->hull_dim, rows);
}  /* rotateinput */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="rotatepoints">-</a>

  qh_rotatepoints(qh, points, numpoints, dim, row )
    rotate numpoints points by a d-dim row matrix
    assumes rows[dim] is a scratch buffer

  returns:
    rotated points in place

  design:
    for each point
      for each coordinate
        use row[dim] to compute partial inner product
      for each coordinate
        rotate by partial inner product
*/
void qh_rotatepoints(qhT *qh, realT *points, int numpoints, int dim, realT **row) {
  realT *point, *rowi, *coord= NULL, sum, *newval;
  int i,j,k;

  if (qh->IStracing >= 1)
    qh_printmatrix(qh, qh->ferr, "qh_rotatepoints: rotate points by", row, dim, dim);
  for (point=points, j=numpoints; j--; point += dim) {
    newval= row[dim];
    for (i=0; i < dim; i++) {
      rowi= row[i];
      coord= point;
      for (sum=0.0, k=dim; k--; )
        sum += *rowi++ * *coord++;
      *(newval++)= sum;
    }
    for (k=dim; k--; )
      *(--coord)= *(--newval);
  }
} /* rotatepoints */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="scaleinput">-</a>

  qh_scaleinput(qh)
    scale input points using qh->low_bound/high_bound
    input points given by qh->first_point, num_points, hull_dim
    if qh->POINTSmalloc, overwrites input points, else mallocs a new array

  returns:
    scales coordinates of points to low_bound[k], high_bound[k]
    sets qh->POINTSmalloc

  design:
    see qh_scalepoints
*/
void qh_scaleinput(qhT *qh) {

  if (!qh->POINTSmalloc) {
    qh->first_point= qh_copypoints(qh, qh->first_point, qh->num_points, qh->hull_dim);
    qh->POINTSmalloc= True;
  }
  qh_scalepoints(qh, qh->first_point, qh->num_points, qh->hull_dim,
       qh->lower_bound, qh->upper_bound);
}  /* scaleinput */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="scalelast">-</a>

  qh_scalelast(qh, points, numpoints, dim, low, high, newhigh )
    scale last coordinate to [0.0, newhigh], for Delaunay triangulation
    input points given by points, numpoints, dim

  returns:
    changes scale of last coordinate from [low, high] to [0.0, newhigh]
    overwrites last coordinate of each point
    saves low/high/newhigh in qh.last_low, etc. for qh_setdelaunay()

  notes:
    to reduce precision issues, qh_scalelast makes the last coordinate similar to other coordinates
      the last coordinate for Delaunay triangulation is the sum of squares of input coordinates
      note that the range [0.0, newwidth] is wrong for narrow distributions with large positive coordinates (e.g., [995933.64, 995963.48])

    when called by qh_setdelaunay, low/high may not match the data passed to qh_setdelaunay

  design:
    compute scale and shift factors
    apply to last coordinate of each point
*/
void qh_scalelast(qhT *qh, coordT *points, int numpoints, int dim, coordT low,
                   coordT high, coordT newhigh) {
  realT scale, shift;
  coordT *coord, newlow;
  int i;
  boolT nearzero= False;

  newlow= 0.0;
  trace4((qh, qh->ferr, 4013, "qh_scalelast: scale last coordinate from [%2.2g, %2.2g] to [%2.2g, %2.2g]\n",
    low, high, newlow, newhigh));
  qh->last_low= low;
  qh->last_high= high;
  qh->last_newhigh= newhigh;
  scale= qh_divzero(newhigh - newlow, high - low,
                  qh->MINdenom_1, &nearzero);
  if (nearzero) {
    if (qh->DELAUNAY)
      qh_fprintf(qh, qh->ferr, 6019, "qhull input error (qh_scalelast): can not scale last coordinate to [%4.4g, %4.4g].  Input is cocircular or cospherical.   Use option 'Qz' to add a point at infinity.\n",
             newlow, newhigh);
    else
      qh_fprintf(qh, qh->ferr, 6020, "qhull input error (qh_scalelast): can not scale last coordinate to [%4.4g, %4.4g].  New bounds are too wide for compared to existing bounds [%4.4g, %4.4g] (width %4.4g)\n",
             newlow, newhigh, low, high, high-low);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  shift= newlow - low * scale;
  coord= points + dim - 1;
  for (i=numpoints; i--; coord += dim)
    *coord= *coord * scale + shift;
} /* scalelast */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="scalepoints">-</a>

  qh_scalepoints(qh, points, numpoints, dim, newlows, newhighs )
    scale points to new lowbound and highbound
    retains old bound when newlow= -REALmax or newhigh= +REALmax

  returns:
    scaled points
    overwrites old points

  design:
    for each coordinate
      compute current low and high bound
      compute scale and shift factors
      scale all points
      enforce new low and high bound for all points
*/
void qh_scalepoints(qhT *qh, pointT *points, int numpoints, int dim,
        realT *newlows, realT *newhighs) {
  int i,k;
  realT shift, scale, *coord, low, high, newlow, newhigh, mincoord, maxcoord;
  boolT nearzero= False;

  for (k=0; k < dim; k++) {
    newhigh= newhighs[k];
    newlow= newlows[k];
    if (newhigh > REALmax/2 && newlow < -REALmax/2)
      continue;
    low= REALmax;
    high= -REALmax;
    for (i=numpoints, coord=points+k; i--; coord += dim) {
      minimize_(low, *coord);
      maximize_(high, *coord);
    }
    if (newhigh > REALmax/2)
      newhigh= high;
    if (newlow < -REALmax/2)
      newlow= low;
    if (qh->DELAUNAY && k == dim-1 && newhigh < newlow) {
      qh_fprintf(qh, qh->ferr, 6021, "qhull input error: 'Qb%d' or 'QB%d' inverts paraboloid since high bound %.2g < low bound %.2g\n",
               k, k, newhigh, newlow);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    scale= qh_divzero(newhigh - newlow, high - low,
                  qh->MINdenom_1, &nearzero);
    if (nearzero) {
      qh_fprintf(qh, qh->ferr, 6022, "qhull input error: %d'th dimension's new bounds [%2.2g, %2.2g] too wide for\nexisting bounds [%2.2g, %2.2g]\n",
              k, newlow, newhigh, low, high);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    shift= (newlow * high - low * newhigh)/(high-low);
    coord= points+k;
    for (i=numpoints; i--; coord += dim)
      *coord= *coord * scale + shift;
    coord= points+k;
    if (newlow < newhigh) {
      mincoord= newlow;
      maxcoord= newhigh;
    }else {
      mincoord= newhigh;
      maxcoord= newlow;
    }
    for (i=numpoints; i--; coord += dim) {
      minimize_(*coord, maxcoord);  /* because of roundoff error */
      maximize_(*coord, mincoord);
    }
    trace0((qh, qh->ferr, 10, "qh_scalepoints: scaled %d'th coordinate [%2.2g, %2.2g] to [%.2g, %.2g] for %d points by %2.2g and shifted %2.2g\n",
      k, low, high, newlow, newhigh, numpoints, scale, shift));
  }
} /* scalepoints */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="setdelaunay">-</a>

  qh_setdelaunay(qh, dim, count, points )
    project count points to dim-d paraboloid for Delaunay triangulation

    dim is one more than the dimension of the input set
    assumes dim is at least 3 (i.e., at least a 2-d Delaunay triangulation)

    points is a dim*count realT array.  The first dim-1 coordinates
    are the coordinates of the first input point.  array[dim] is
    the first coordinate of the second input point.  array[2*dim] is
    the first coordinate of the third input point.

    if qh.last_low defined (i.e., 'Qbb' called qh_scalelast)
      calls qh_scalelast to scale the last coordinate the same as the other points

  returns:
    for each point
      sets point[dim-1] to sum of squares of coordinates
    scale points to 'Qbb' if needed

  notes:
    to project one point, use
      qh_setdelaunay(qh, qh->hull_dim, 1, point)

    Do not use options 'Qbk', 'QBk', or 'QbB' since they scale
    the coordinates after the original projection.

*/
void qh_setdelaunay(qhT *qh, int dim, int count, pointT *points) {
  int i, k;
  coordT *coordp, coord;
  realT paraboloid;

  trace0((qh, qh->ferr, 11, "qh_setdelaunay: project %d points to paraboloid for Delaunay triangulation\n", count));
  coordp= points;
  for (i=0; i < count; i++) {
    coord= *coordp++;
    paraboloid= coord*coord;
    for (k=dim-2; k--; ) {
      coord= *coordp++;
      paraboloid += coord*coord;
    }
    *coordp++= paraboloid;
  }
  if (qh->last_low < REALmax/2)
    qh_scalelast(qh, points, count, dim, qh->last_low, qh->last_high, qh->last_newhigh);
} /* setdelaunay */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sethalfspace">-</a>

  qh_sethalfspace(qh, dim, coords, nextp, normal, offset, feasible )
    set point to dual of halfspace relative to feasible point
    halfspace is normal coefficients and offset.

  returns:
    false and prints error if feasible point is outside of hull
    overwrites coordinates for point at dim coords
    nextp= next point (coords)
    does not call qh_errexit

  design:
    compute distance from feasible point to halfspace
    divide each normal coefficient by -dist
*/
boolT qh_sethalfspace(qhT *qh, int dim, coordT *coords, coordT **nextp,
         coordT *normal, coordT *offset, coordT *feasible) {
  coordT *normp= normal, *feasiblep= feasible, *coordp= coords;
  realT dist;
  realT r; /*bug fix*/
  int k;
  boolT zerodiv;

  dist= *offset;
  for (k=dim; k--; )
    dist += *(normp++) * *(feasiblep++);
  if (dist > 0)
    goto LABELerroroutside;
  normp= normal;
  if (dist < -qh->MINdenom) {
    for (k=dim; k--; )
      *(coordp++)= *(normp++) / -dist;
  }else {
    for (k=dim; k--; ) {
      *(coordp++)= qh_divzero(*(normp++), -dist, qh->MINdenom_1, &zerodiv);
      if (zerodiv)
        goto LABELerroroutside;
    }
  }
  *nextp= coordp;
#ifndef qh_NOtrace
  if (qh->IStracing >= 4) {
    qh_fprintf(qh, qh->ferr, 8021, "qh_sethalfspace: halfspace at offset %6.2g to point: ", *offset);
    for (k=dim, coordp=coords; k--; ) {
      r= *coordp++;
      qh_fprintf(qh, qh->ferr, 8022, " %6.2g", r);
    }
    qh_fprintf(qh, qh->ferr, 8023, "\n");
  }
#endif
  return True;
LABELerroroutside:
  feasiblep= feasible;
  normp= normal;
  qh_fprintf(qh, qh->ferr, 6023, "qhull input error: feasible point is not clearly inside halfspace\nfeasible point: ");
  for (k=dim; k--; )
    qh_fprintf(qh, qh->ferr, 8024, qh_REAL_1, r=*(feasiblep++));
  qh_fprintf(qh, qh->ferr, 8025, "\n     halfspace: ");
  for (k=dim; k--; )
    qh_fprintf(qh, qh->ferr, 8026, qh_REAL_1, r=*(normp++));
  qh_fprintf(qh, qh->ferr, 8027, "\n     at offset: ");
  qh_fprintf(qh, qh->ferr, 8028, qh_REAL_1, *offset);
  qh_fprintf(qh, qh->ferr, 8029, " and distance: ");
  qh_fprintf(qh, qh->ferr, 8030, qh_REAL_1, dist);
  qh_fprintf(qh, qh->ferr, 8031, "\n");
  return False;
} /* sethalfspace */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sethalfspace_all">-</a>

  qh_sethalfspace_all(qh, dim, count, halfspaces, feasible )
    generate dual for halfspace intersection with feasible point
    array of count halfspaces
      each halfspace is normal coefficients followed by offset
      the origin is inside the halfspace if the offset is negative
    feasible is a point inside all halfspaces (http://www.qhull.org/html/qhalf.htm#notes)

  returns:
    malloc'd array of count X dim-1 points

  notes:
    call before qh_init_B or qh_initqhull_globals
    free memory when done
    unused/untested code: please email bradb@shore.net if this works ok for you
    if using option 'Fp', qh.feasible_point must be set (e.g., to 'feasible')
    qh->feasible_point is a malloc'd array that is freed by qh_freebuffers.

  design:
    see qh_sethalfspace
*/
coordT *qh_sethalfspace_all(qhT *qh, int dim, int count, coordT *halfspaces, pointT *feasible) {
  int i, newdim;
  pointT *newpoints;
  coordT *coordp, *normalp, *offsetp;

  trace0((qh, qh->ferr, 12, "qh_sethalfspace_all: compute dual for halfspace intersection\n"));
  newdim= dim - 1;
  if (!(newpoints= (coordT *)qh_malloc((size_t)(count * newdim) * sizeof(coordT)))){
    qh_fprintf(qh, qh->ferr, 6024, "qhull error: insufficient memory to compute dual of %d halfspaces\n",
          count);
    qh_errexit(qh, qh_ERRmem, NULL, NULL);
  }
  coordp= newpoints;
  normalp= halfspaces;
  for (i=0; i < count; i++) {
    offsetp= normalp + newdim;
    if (!qh_sethalfspace(qh, newdim, coordp, &coordp, normalp, offsetp, feasible)) {
      qh_free(newpoints);  /* feasible is not inside halfspace as reported by qh_sethalfspace */
      qh_fprintf(qh, qh->ferr, 8032, "The halfspace was at index %d\n", i);
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    normalp= offsetp + 1;
  }
  return newpoints;
} /* sethalfspace_all */


/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="sharpnewfacets">-</a>

  qh_sharpnewfacets(qh)

  returns:
    true if could be an acute angle (facets in different quadrants)

  notes:
    for qh_findbest

  design:
    for all facets on qh.newfacet_list
      if two facets are in different quadrants
        set issharp
*/
boolT qh_sharpnewfacets(qhT *qh) {
  facetT *facet;
  boolT issharp= False;
  int *quadrant, k;

  quadrant= (int *)qh_memalloc(qh, qh->hull_dim * (int)sizeof(int));
  FORALLfacet_(qh->newfacet_list) {
    if (facet == qh->newfacet_list) {
      for (k=qh->hull_dim; k--; )
        quadrant[ k]= (facet->normal[ k] > 0);
    }else {
      for (k=qh->hull_dim; k--; ) {
        if (quadrant[ k] != (facet->normal[ k] > 0)) {
          issharp= True;
          break;
        }
      }
    }
    if (issharp)
      break;
  }
  qh_memfree(qh, quadrant, qh->hull_dim * (int)sizeof(int));
  trace3((qh, qh->ferr, 3001, "qh_sharpnewfacets: %d\n", issharp));
  return issharp;
} /* sharpnewfacets */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="vertex_bestdist">-</a>

  qh_vertex_bestdist(qh, vertices )
  qh_vertex_bestdist2(qh, vertices, vertexp, vertexp2 )
    return nearest distance between vertices
    optionally returns vertex and vertex2

  notes:
    called by qh_partitioncoplanar, qh_mergefacet, qh_check_maxout, qh_checkpoint
*/
coordT qh_vertex_bestdist(qhT *qh, setT *vertices) {
  vertexT *vertex, *vertex2;

  return qh_vertex_bestdist2(qh, vertices, &vertex, &vertex2);
} /* vertex_bestdist */

coordT qh_vertex_bestdist2(qhT *qh, setT *vertices, vertexT **vertexp/*= NULL*/, vertexT **vertexp2/*= NULL*/) {
  vertexT *vertex, *vertexA, *bestvertex= NULL, *bestvertex2= NULL;
  coordT dist, bestdist= REALmax;
  int k, vertex_i, vertex_n;

  FOREACHvertex_i_(qh, vertices) {
    for (k= vertex_i+1; k < vertex_n; k++) {
      vertexA= SETelemt_(vertices, k, vertexT);
      dist= qh_pointdist(vertex->point, vertexA->point, -qh->hull_dim);
      if (dist < bestdist) {
        bestdist= dist;
        bestvertex= vertex;
        bestvertex2= vertexA;
      }
    }
  }
  *vertexp= bestvertex;
  *vertexp2= bestvertex2;
  return sqrt(bestdist);
} /* vertex_bestdist */

/*-<a                             href="qh-geom_r.htm#TOC"
  >-------------------------------</a><a name="voronoi_center">-</a>

  qh_voronoi_center(qh, dim, points )
    return Voronoi center for a set of points
    dim is the orginal dimension of the points
    gh.gm_matrix/qh.gm_row are scratch buffers

  returns:
    center as a temporary point (qh_memalloc)
    if non-simplicial,
      returns center for max simplex of points

  notes:
    only called by qh_facetcenter
    from Bowyer & Woodwark, A Programmer's Geometry, 1983, p. 65

  design:
    if non-simplicial
      determine max simplex for points
    translate point0 of simplex to origin
    compute sum of squares of diagonal
    compute determinate
    compute Voronoi center (see Bowyer & Woodwark)
*/
pointT *qh_voronoi_center(qhT *qh, int dim, setT *points) {
  pointT *point, **pointp, *point0;
  pointT *center= (pointT *)qh_memalloc(qh, qh->center_size);
  setT *simplex;
  int i, j, k, size= qh_setsize(qh, points);
  coordT *gmcoord;
  realT *diffp, sum2, *sum2row, *sum2p, det, factor;
  boolT nearzero, infinite;

  if (size == dim+1)
    simplex= points;
  else if (size < dim+1) {
    qh_memfree(qh, center, qh->center_size);
    qh_fprintf(qh, qh->ferr, 6025, "qhull internal error (qh_voronoi_center):  need at least %d points to construct a Voronoi center\n",
             dim+1);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    simplex= points;  /* never executed -- avoids warning */
  }else {
    simplex= qh_settemp(qh, dim+1);
    qh_maxsimplex(qh, dim, points, NULL, 0, &simplex);
  }
  point0= SETfirstt_(simplex, pointT);
  gmcoord= qh->gm_matrix;
  for (k=0; k < dim; k++) {
    qh->gm_row[k]= gmcoord;
    FOREACHpoint_(simplex) {
      if (point != point0)
        *(gmcoord++)= point[k] - point0[k];
    }
  }
  sum2row= gmcoord;
  for (i=0; i < dim; i++) {
    sum2= 0.0;
    for (k=0; k < dim; k++) {
      diffp= qh->gm_row[k] + i;
      sum2 += *diffp * *diffp;
    }
    *(gmcoord++)= sum2;
  }
  det= qh_determinant(qh, qh->gm_row, dim, &nearzero);
  factor= qh_divzero(0.5, det, qh->MINdenom, &infinite);
  if (infinite) {
    for (k=dim; k--; )
      center[k]= qh_INFINITE;
    if (qh->IStracing)
      qh_printpoints(qh, qh->ferr, "qh_voronoi_center: at infinity for ", simplex);
  }else {
    for (i=0; i < dim; i++) {
      gmcoord= qh->gm_matrix;
      sum2p= sum2row;
      for (k=0; k < dim; k++) {
        qh->gm_row[k]= gmcoord;
        if (k == i) {
          for (j=dim; j--; )
            *(gmcoord++)= *sum2p++;
        }else {
          FOREACHpoint_(simplex) {
            if (point != point0)
              *(gmcoord++)= point[k] - point0[k];
          }
        }
      }
      center[i]= qh_determinant(qh, qh->gm_row, dim, &nearzero)*factor + point0[i];
    }
#ifndef qh_NOtrace
    if (qh->IStracing >= 3) {
      qh_fprintf(qh, qh->ferr, 3061, "qh_voronoi_center: det %2.2g factor %2.2g ", det, factor);
      qh_printmatrix(qh, qh->ferr, "center:", &center, 1, dim);
      if (qh->IStracing >= 5) {
        qh_printpoints(qh, qh->ferr, "points", simplex);
        FOREACHpoint_(simplex)
          qh_fprintf(qh, qh->ferr, 8034, "p%d dist %.2g, ", qh_pointid(qh, point),
                   qh_pointdist(point, center, dim));
        qh_fprintf(qh, qh->ferr, 8035, "\n");
      }
    }
#endif
  }
  if (simplex != points)
    qh_settempfree(qh, &simplex);
  return center;
} /* voronoi_center */


/*<html><pre>  -<a                             href="qh-geom.htm"
  >-------------------------------</a><a name="TOP">-</a>

   geom.c
   geometric routines of qhull

   see qh-geom.htm and geom.h

   Copyright (c) 1993-2015 The Geometry Center.
   $Id: //main/2015/qhull/src/libqhull/geom.c#2 $$Change: 1995 $
   $DateTime: 2015/10/13 21:59:42 $$Author: bbarber $

   infrequent code goes into geom2.c
*/

#include "qhull_a.h"

/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="distplane">-</a>

  qh_distplane( point, facet, dist )
    return distance from point to facet

  returns:
    dist
    if qh.RANDOMdist, joggles result

  notes:
    dist > 0 if point is above facet (i.e., outside)
    does not error (for qh_sortfacets, qh_outerinner)

  see:
    qh_distnorm in geom2.c
    qh_distplane [geom.c], QhullFacet::distance, and QhullHyperplane::distance are copies
*/
void qh_distplane(pointT *point, facetT *facet, realT *dist) {
  coordT *normal= facet->normal, *coordp, randr;
  int k;

  switch (qh hull_dim){
  case 2:
    *dist= facet->offset + point[0] * normal[0] + point[1] * normal[1];
    break;
  case 3:
    *dist= facet->offset + point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
    break;
  case 4:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3];
    break;
  case 5:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4];
    break;
  case 6:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4]+point[5]*normal[5];
    break;
  case 7:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4]+point[5]*normal[5]+point[6]*normal[6];
    break;
  case 8:
    *dist= facet->offset+point[0]*normal[0]+point[1]*normal[1]+point[2]*normal[2]+point[3]*normal[3]+point[4]*normal[4]+point[5]*normal[5]+point[6]*normal[6]+point[7]*normal[7];
    break;
  default:
    *dist= facet->offset;
    coordp= point;
    for (k=qh hull_dim; k--; )
      *dist += *coordp++ * *normal++;
    break;
  }
  zinc_(Zdistplane);
  if (!qh RANDOMdist && qh IStracing < 4)
    return;
  if (qh RANDOMdist) {
    randr= qh_RANDOMint;
    *dist += (2.0 * randr / qh_RANDOMmax - 1.0) *
      qh RANDOMfactor * qh MAXabs_coord;
  }
  if (qh IStracing >= 4) {
    qh_fprintf(qh ferr, 8001, "qh_distplane: ");
    qh_fprintf(qh ferr, 8002, qh_REAL_1, *dist);
    qh_fprintf(qh ferr, 8003, "from p%d to f%d\n", qh_pointid(point), facet->id);
  }
  return;
} /* distplane */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="findbest">-</a>

  qh_findbest( point, startfacet, bestoutside, qh_ISnewfacets, qh_NOupper, dist, isoutside, numpart )
    find facet that is furthest below a point
    for upperDelaunay facets
      returns facet only if !qh_NOupper and clearly above

  input:
    starts search at 'startfacet' (can not be flipped)
    if !bestoutside(qh_ALL), stops at qh.MINoutside

  returns:
    best facet (reports error if NULL)
    early out if isoutside defined and bestdist > qh.MINoutside
    dist is distance to facet
    isoutside is true if point is outside of facet
    numpart counts the number of distance tests

  see also:
    qh_findbestnew()

  notes:
    If merging (testhorizon), searches horizon facets of coplanar best facets because
    after qh_distplane, this and qh_partitionpoint are the most expensive in 3-d
      avoid calls to distplane, function calls, and real number operations.
    caller traces result
    Optimized for outside points.   Tried recording a search set for qh_findhorizon.
    Made code more complicated.

  when called by qh_partitionvisible():
    indicated by qh_ISnewfacets
    qh.newfacet_list is list of simplicial, new facets
    qh_findbestnew set if qh_sharpnewfacets returns True (to use qh_findbestnew)
    qh.bestfacet_notsharp set if qh_sharpnewfacets returns False

  when called by qh_findfacet(), qh_partitionpoint(), qh_partitioncoplanar(),
                 qh_check_bestdist(), qh_addpoint()
    indicated by !qh_ISnewfacets
    returns best facet in neighborhood of given facet
      this is best facet overall if dist > -   qh.MAXcoplanar
        or hull has at least a "spherical" curvature

  design:
    initialize and test for early exit
    repeat while there are better facets
      for each neighbor of facet
        exit if outside facet found
        test for better facet
    if point is inside and partitioning
      test for new facets with a "sharp" intersection
      if so, future calls go to qh_findbestnew()
    test horizon facets
*/
facetT *qh_findbest(pointT *point, facetT *startfacet,
                     boolT bestoutside, boolT isnewfacets, boolT noupper,
                     realT *dist, boolT *isoutside, int *numpart) {
  realT bestdist= -REALmax/2 /* avoid underflow */;
  facetT *facet, *neighbor, **neighborp;
  facetT *bestfacet= NULL, *lastfacet= NULL;
  int oldtrace= qh IStracing;
  unsigned int visitid= ++qh visit_id;
  int numpartnew=0;
  boolT testhorizon = True; /* needed if precise, e.g., rbox c D6 | qhull Q0 Tv */

  zinc_(Zfindbest);
  if (qh IStracing >= 3 || (qh TRACElevel && qh TRACEpoint >= 0 && qh TRACEpoint == qh_pointid(point))) {
    if (qh TRACElevel > qh IStracing)
      qh IStracing= qh TRACElevel;
    qh_fprintf(qh ferr, 8004, "qh_findbest: point p%d starting at f%d isnewfacets? %d, unless %d exit if > %2.2g\n",
             qh_pointid(point), startfacet->id, isnewfacets, bestoutside, qh MINoutside);
    qh_fprintf(qh ferr, 8005, "  testhorizon? %d noupper? %d", testhorizon, noupper);
    qh_fprintf(qh ferr, 8006, "  Last point added was p%d.", qh furthest_id);
    qh_fprintf(qh ferr, 8007, "  Last merge was #%d.  max_outside %2.2g\n", zzval_(Ztotmerge), qh max_outside);
  }
  if (isoutside)
    *isoutside= True;
  if (!startfacet->flipped) {  /* test startfacet */
    *numpart= 1;
    qh_distplane(point, startfacet, dist);  /* this code is duplicated below */
    if (!bestoutside && *dist >= qh MINoutside
    && (!startfacet->upperdelaunay || !noupper)) {
      bestfacet= startfacet;
      goto LABELreturn_best;
    }
    bestdist= *dist;
    if (!startfacet->upperdelaunay) {
      bestfacet= startfacet;
    }
  }else
    *numpart= 0;
  startfacet->visitid= visitid;
  facet= startfacet;
  while (facet) {
    trace4((qh ferr, 4001, "qh_findbest: neighbors of f%d, bestdist %2.2g f%d\n",
                facet->id, bestdist, getid_(bestfacet)));
    lastfacet= facet;
    FOREACHneighbor_(facet) {
      if (!neighbor->newfacet && isnewfacets)
        continue;
      if (neighbor->visitid == visitid)
        continue;
      neighbor->visitid= visitid;
      if (!neighbor->flipped) {  /* code duplicated above */
        (*numpart)++;
        qh_distplane(point, neighbor, dist);
        if (*dist > bestdist) {
          if (!bestoutside && *dist >= qh MINoutside
          && (!neighbor->upperdelaunay || !noupper)) {
            bestfacet= neighbor;
            goto LABELreturn_best;
          }
          if (!neighbor->upperdelaunay) {
            bestfacet= neighbor;
            bestdist= *dist;
            break; /* switch to neighbor */
          }else if (!bestfacet) {
            bestdist= *dist;
            break; /* switch to neighbor */
          }
        } /* end of *dist>bestdist */
      } /* end of !flipped */
    } /* end of FOREACHneighbor */
    facet= neighbor;  /* non-NULL only if *dist>bestdist */
  } /* end of while facet (directed search) */
  if (isnewfacets) {
    if (!bestfacet) {
      bestdist= -REALmax/2;
      bestfacet= qh_findbestnew(point, startfacet->next, &bestdist, bestoutside, isoutside, &numpartnew);
      testhorizon= False; /* qh_findbestnew calls qh_findbesthorizon */
    }else if (!qh findbest_notsharp && bestdist < - qh DISTround) {
      if (qh_sharpnewfacets()) {
        /* seldom used, qh_findbestnew will retest all facets */
        zinc_(Zfindnewsharp);
        bestfacet= qh_findbestnew(point, bestfacet, &bestdist, bestoutside, isoutside, &numpartnew);
        testhorizon= False; /* qh_findbestnew calls qh_findbesthorizon */
        qh findbestnew= True;
      }else
        qh findbest_notsharp= True;
    }
  }
  if (!bestfacet)
    bestfacet= qh_findbestlower(lastfacet, point, &bestdist, numpart);
  if (testhorizon)
    bestfacet= qh_findbesthorizon(!qh_IScheckmax, point, bestfacet, noupper, &bestdist, &numpartnew);
  *dist= bestdist;
  if (isoutside && bestdist < qh MINoutside)
    *isoutside= False;
LABELreturn_best:
  zadd_(Zfindbesttot, *numpart);
  zmax_(Zfindbestmax, *numpart);
  (*numpart) += numpartnew;
  qh IStracing= oldtrace;
  return bestfacet;
}  /* findbest */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="findbesthorizon">-</a>

  qh_findbesthorizon( qh_IScheckmax, point, startfacet, qh_NOupper, &bestdist, &numpart )
    search coplanar and better horizon facets from startfacet/bestdist
    ischeckmax turns off statistics and minsearch update
    all arguments must be initialized
  returns(ischeckmax):
    best facet
  returns(!ischeckmax):
    best facet that is not upperdelaunay
    allows upperdelaunay that is clearly outside
  returns:
    bestdist is distance to bestfacet
    numpart -- updates number of distance tests

  notes:
    no early out -- use qh_findbest() or qh_findbestnew()
    Searches coplanar or better horizon facets

  when called by qh_check_maxout() (qh_IScheckmax)
    startfacet must be closest to the point
      Otherwise, if point is beyond and below startfacet, startfacet may be a local minimum
      even though other facets are below the point.
    updates facet->maxoutside for good, visited facets
    may return NULL

    searchdist is qh.max_outside + 2 * DISTround
      + max( MINvisible('Vn'), MAXcoplanar('Un'));
    This setting is a guess.  It must be at least max_outside + 2*DISTround
    because a facet may have a geometric neighbor across a vertex

  design:
    for each horizon facet of coplanar best facets
      continue if clearly inside
      unless upperdelaunay or clearly outside
         update best facet
*/
facetT *qh_findbesthorizon(boolT ischeckmax, pointT* point, facetT *startfacet, boolT noupper, realT *bestdist, int *numpart) {
  facetT *bestfacet= startfacet;
  realT dist;
  facetT *neighbor, **neighborp, *facet;
  facetT *nextfacet= NULL; /* optimize last facet of coplanarfacetset */
  int numpartinit= *numpart, coplanarfacetset_size;
  unsigned int visitid= ++qh visit_id;
  boolT newbest= False; /* for tracing */
  realT minsearch, searchdist;  /* skip facets that are too far from point */

  if (!ischeckmax) {
    zinc_(Zfindhorizon);
  }else {
#if qh_MAXoutside
    if ((!qh ONLYgood || startfacet->good) && *bestdist > startfacet->maxoutside)
      startfacet->maxoutside= *bestdist;
#endif
  }
  searchdist= qh_SEARCHdist; /* multiple of qh.max_outside and precision constants */
  minsearch= *bestdist - searchdist;
  if (ischeckmax) {
    /* Always check coplanar facets.  Needed for RBOX 1000 s Z1 G1e-13 t996564279 | QHULL Tv */
    minimize_(minsearch, -searchdist);
  }
  coplanarfacetset_size= 0;
  facet= startfacet;
  while (True) {
    trace4((qh ferr, 4002, "qh_findbesthorizon: neighbors of f%d bestdist %2.2g f%d ischeckmax? %d noupper? %d minsearch %2.2g searchdist %2.2g\n",
                facet->id, *bestdist, getid_(bestfacet), ischeckmax, noupper,
                minsearch, searchdist));
    FOREACHneighbor_(facet) {
      if (neighbor->visitid == visitid)
        continue;
      neighbor->visitid= visitid;
      if (!neighbor->flipped) {
        qh_distplane(point, neighbor, &dist);
        (*numpart)++;
        if (dist > *bestdist) {
          if (!neighbor->upperdelaunay || ischeckmax || (!noupper && dist >= qh MINoutside)) {
            bestfacet= neighbor;
            *bestdist= dist;
            newbest= True;
            if (!ischeckmax) {
              minsearch= dist - searchdist;
              if (dist > *bestdist + searchdist) {
                zinc_(Zfindjump);  /* everything in qh.coplanarfacetset at least searchdist below */
                coplanarfacetset_size= 0;
              }
            }
          }
        }else if (dist < minsearch)
          continue;  /* if ischeckmax, dist can't be positive */
#if qh_MAXoutside
        if (ischeckmax && dist > neighbor->maxoutside)
          neighbor->maxoutside= dist;
#endif
      } /* end of !flipped */
      if (nextfacet) {
        if (!coplanarfacetset_size++) {
          SETfirst_(qh coplanarfacetset)= nextfacet;
          SETtruncate_(qh coplanarfacetset, 1);
        }else
          qh_setappend(&qh coplanarfacetset, nextfacet); /* Was needed for RBOX 1000 s W1e-13 P0 t996547055 | QHULL d Qbb Qc Tv
                                                 and RBOX 1000 s Z1 G1e-13 t996564279 | qhull Tv  */
      }
      nextfacet= neighbor;
    } /* end of EACHneighbor */
    facet= nextfacet;
    if (facet)
      nextfacet= NULL;
    else if (!coplanarfacetset_size)
      break;
    else if (!--coplanarfacetset_size) {
      facet= SETfirstt_(qh coplanarfacetset, facetT);
      SETtruncate_(qh coplanarfacetset, 0);
    }else
      facet= (facetT*)qh_setdellast(qh coplanarfacetset);
  } /* while True, for each facet in qh.coplanarfacetset */
  if (!ischeckmax) {
    zadd_(Zfindhorizontot, *numpart - numpartinit);
    zmax_(Zfindhorizonmax, *numpart - numpartinit);
    if (newbest)
      zinc_(Zparthorizon);
  }
  trace4((qh ferr, 4003, "qh_findbesthorizon: newbest? %d bestfacet f%d bestdist %2.2g\n", newbest, getid_(bestfacet), *bestdist));
  return bestfacet;
}  /* findbesthorizon */

/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="findbestnew">-</a>

  qh_findbestnew( point, startfacet, dist, isoutside, numpart )
    find best newfacet for point
    searches all of qh.newfacet_list starting at startfacet
    searches horizon facets of coplanar best newfacets
    searches all facets if startfacet == qh.facet_list
  returns:
    best new or horizon facet that is not upperdelaunay
    early out if isoutside and not 'Qf'
    dist is distance to facet
    isoutside is true if point is outside of facet
    numpart is number of distance tests

  notes:
    Always used for merged new facets (see qh_USEfindbestnew)
    Avoids upperdelaunay facet unless (isoutside and outside)

    Uses qh.visit_id, qh.coplanarfacetset.
    If share visit_id with qh_findbest, coplanarfacetset is incorrect.

    If merging (testhorizon), searches horizon facets of coplanar best facets because
    a point maybe coplanar to the bestfacet, below its horizon facet,
    and above a horizon facet of a coplanar newfacet.  For example,
      rbox 1000 s Z1 G1e-13 | qhull
      rbox 1000 s W1e-13 P0 t992110337 | QHULL d Qbb Qc

    qh_findbestnew() used if
       qh_sharpnewfacets -- newfacets contains a sharp angle
       if many merges, qh_premerge found a merge, or 'Qf' (qh.findbestnew)

  see also:
    qh_partitionall() and qh_findbest()

  design:
    for each new facet starting from startfacet
      test distance from point to facet
      return facet if clearly outside
      unless upperdelaunay and a lowerdelaunay exists
         update best facet
    test horizon facets
*/
facetT *qh_findbestnew(pointT *point, facetT *startfacet,
           realT *dist, boolT bestoutside, boolT *isoutside, int *numpart) {
  realT bestdist= -REALmax/2;
  facetT *bestfacet= NULL, *facet;
  int oldtrace= qh IStracing, i;
  unsigned int visitid= ++qh visit_id;
  realT distoutside= 0.0;
  boolT isdistoutside; /* True if distoutside is defined */
  boolT testhorizon = True; /* needed if precise, e.g., rbox c D6 | qhull Q0 Tv */

  if (!startfacet) {
    if (qh MERGING)
      qh_fprintf(qh ferr, 6001, "qhull precision error (qh_findbestnew): merging has formed and deleted a cone of new facets.  Can not continue.\n");
    else
      qh_fprintf(qh ferr, 6002, "qhull internal error (qh_findbestnew): no new facets for point p%d\n",
              qh furthest_id);
    qh_errexit(qh_ERRqhull, NULL, NULL);
  }
  zinc_(Zfindnew);
  if (qh BESToutside || bestoutside)
    isdistoutside= False;
  else {
    isdistoutside= True;
    distoutside= qh_DISToutside; /* multiple of qh.MINoutside & qh.max_outside, see user.h */
  }
  if (isoutside)
    *isoutside= True;
  *numpart= 0;
  if (qh IStracing >= 3 || (qh TRACElevel && qh TRACEpoint >= 0 && qh TRACEpoint == qh_pointid(point))) {
    if (qh TRACElevel > qh IStracing)
      qh IStracing= qh TRACElevel;
    qh_fprintf(qh ferr, 8008, "qh_findbestnew: point p%d facet f%d. Stop? %d if dist > %2.2g\n",
             qh_pointid(point), startfacet->id, isdistoutside, distoutside);
    qh_fprintf(qh ferr, 8009, "  Last point added p%d visitid %d.",  qh furthest_id, visitid);
    qh_fprintf(qh ferr, 8010, "  Last merge was #%d.\n", zzval_(Ztotmerge));
  }
  /* visit all new facets starting with startfacet, maybe qh facet_list */
  for (i=0, facet=startfacet; i < 2; i++, facet= qh newfacet_list) {
    FORALLfacet_(facet) {
      if (facet == startfacet && i)
        break;
      facet->visitid= visitid;
      if (!facet->flipped) {
        qh_distplane(point, facet, dist);
        (*numpart)++;
        if (*dist > bestdist) {
          if (!facet->upperdelaunay || *dist >= qh MINoutside) {
            bestfacet= facet;
            if (isdistoutside && *dist >= distoutside)
              goto LABELreturn_bestnew;
            bestdist= *dist;
          }
        }
      } /* end of !flipped */
    } /* FORALLfacet from startfacet or qh newfacet_list */
  }
  if (testhorizon || !bestfacet) /* testhorizon is always True.  Keep the same code as qh_findbest */
    bestfacet= qh_findbesthorizon(!qh_IScheckmax, point, bestfacet ? bestfacet : startfacet,
                                        !qh_NOupper, &bestdist, numpart);
  *dist= bestdist;
  if (isoutside && *dist < qh MINoutside)
    *isoutside= False;
LABELreturn_bestnew:
  zadd_(Zfindnewtot, *numpart);
  zmax_(Zfindnewmax, *numpart);
  trace4((qh ferr, 4004, "qh_findbestnew: bestfacet f%d bestdist %2.2g\n", getid_(bestfacet), *dist));
  qh IStracing= oldtrace;
  return bestfacet;
}  /* findbestnew */

/* ============ hyperplane functions -- keep code together [?] ============ */

/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="backnormal">-</a>

  qh_backnormal( rows, numrow, numcol, sign, normal, nearzero )
    given an upper-triangular rows array and a sign,
    solve for normal equation x using back substitution over rows U

  returns:
     normal= x

     if will not be able to divzero() when normalized(qh.MINdenom_2 and qh.MINdenom_1_2),
       if fails on last row
         this means that the hyperplane intersects [0,..,1]
         sets last coordinate of normal to sign
       otherwise
         sets tail of normal to [...,sign,0,...], i.e., solves for b= [0...0]
         sets nearzero

  notes:
     assumes numrow == numcol-1

     see Golub & van Loan, 1983, Eq. 4.4-9 for "Gaussian elimination with complete pivoting"

     solves Ux=b where Ax=b and PA=LU
     b= [0,...,0,sign or 0]  (sign is either -1 or +1)
     last row of A= [0,...,0,1]

     1) Ly=Pb == y=b since P only permutes the 0's of   b

  design:
    for each row from end
      perform back substitution
      if near zero
        use qh_divzero for division
        if zero divide and not last row
          set tail of normal to 0
*/
void qh_backnormal(realT **rows, int numrow, int numcol, boolT sign,
        coordT *normal, boolT *nearzero) {
  int i, j;
  coordT *normalp, *normal_tail, *ai, *ak;
  realT diagonal;
  boolT waszero;
  int zerocol= -1;

  normalp= normal + numcol - 1;
  *normalp--= (sign ? -1.0 : 1.0);
  for (i=numrow; i--; ) {
    *normalp= 0.0;
    ai= rows[i] + i + 1;
    ak= normalp+1;
    for (j=i+1; j < numcol; j++)
      *normalp -= *ai++ * *ak++;
    diagonal= (rows[i])[i];
    if (fabs_(diagonal) > qh MINdenom_2)
      *(normalp--) /= diagonal;
    else {
      waszero= False;
      *normalp= qh_divzero(*normalp, diagonal, qh MINdenom_1_2, &waszero);
      if (waszero) {
        zerocol= i;
        *(normalp--)= (sign ? -1.0 : 1.0);
        for (normal_tail= normalp+2; normal_tail < normal + numcol; normal_tail++)
          *normal_tail= 0.0;
      }else
        normalp--;
    }
  }
  if (zerocol != -1) {
    zzinc_(Zback0);
    *nearzero= True;
    trace4((qh ferr, 4005, "qh_backnormal: zero diagonal at column %d.\n", i));
    qh_precision("zero diagonal on back substitution");
  }
} /* backnormal */

/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="gausselim">-</a>

  qh_gausselim( rows, numrow, numcol, sign )
    Gaussian elimination with partial pivoting

  returns:
    rows is upper triangular (includes row exchanges)
    flips sign for each row exchange
    sets nearzero if pivot[k] < qh.NEARzero[k], else clears it

  notes:
    if nearzero, the determinant's sign may be incorrect.
    assumes numrow <= numcol

  design:
    for each row
      determine pivot and exchange rows if necessary
      test for near zero
      perform gaussian elimination step
*/
void qh_gausselim(realT **rows, int numrow, int numcol, boolT *sign, boolT *nearzero) {
  realT *ai, *ak, *rowp, *pivotrow;
  realT n, pivot, pivot_abs= 0.0, temp;
  int i, j, k, pivoti, flip=0;

  *nearzero= False;
  for (k=0; k < numrow; k++) {
    pivot_abs= fabs_((rows[k])[k]);
    pivoti= k;
    for (i=k+1; i < numrow; i++) {
      if ((temp= fabs_((rows[i])[k])) > pivot_abs) {
        pivot_abs= temp;
        pivoti= i;
      }
    }
    if (pivoti != k) {
      rowp= rows[pivoti];
      rows[pivoti]= rows[k];
      rows[k]= rowp;
      *sign ^= 1;
      flip ^= 1;
    }
    if (pivot_abs <= qh NEARzero[k]) {
      *nearzero= True;
      if (pivot_abs == 0.0) {   /* remainder of column == 0 */
        if (qh IStracing >= 4) {
          qh_fprintf(qh ferr, 8011, "qh_gausselim: 0 pivot at column %d. (%2.2g < %2.2g)\n", k, pivot_abs, qh DISTround);
          qh_printmatrix(qh ferr, "Matrix:", rows, numrow, numcol);
        }
        zzinc_(Zgauss0);
        qh_precision("zero pivot for Gaussian elimination");
        goto LABELnextcol;
      }
    }
    pivotrow= rows[k] + k;
    pivot= *pivotrow++;  /* signed value of pivot, and remainder of row */
    for (i=k+1; i < numrow; i++) {
      ai= rows[i] + k;
      ak= pivotrow;
      n= (*ai++)/pivot;   /* divzero() not needed since |pivot| >= |*ai| */
      for (j= numcol - (k+1); j--; )
        *ai++ -= n * *ak++;
    }
  LABELnextcol:
    ;
  }
  wmin_(Wmindenom, pivot_abs);  /* last pivot element */
  if (qh IStracing >= 5)
    qh_printmatrix(qh ferr, "qh_gausselem: result", rows, numrow, numcol);
} /* gausselim */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="getangle">-</a>

  qh_getangle( vect1, vect2 )
    returns the dot product of two vectors
    if qh.RANDOMdist, joggles result

  notes:
    the angle may be > 1.0 or < -1.0 because of roundoff errors

*/
realT qh_getangle(pointT *vect1, pointT *vect2) {
  realT angle= 0, randr;
  int k;

  for (k=qh hull_dim; k--; )
    angle += *vect1++ * *vect2++;
  if (qh RANDOMdist) {
    randr= qh_RANDOMint;
    angle += (2.0 * randr / qh_RANDOMmax - 1.0) *
      qh RANDOMfactor;
  }
  trace4((qh ferr, 4006, "qh_getangle: %2.2g\n", angle));
  return(angle);
} /* getangle */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="getcenter">-</a>

  qh_getcenter( vertices )
    returns arithmetic center of a set of vertices as a new point

  notes:
    allocates point array for center
*/
pointT *qh_getcenter(setT *vertices) {
  int k;
  pointT *center, *coord;
  vertexT *vertex, **vertexp;
  int count= qh_setsize(vertices);

  if (count < 2) {
    qh_fprintf(qh ferr, 6003, "qhull internal error (qh_getcenter): not defined for %d points\n", count);
    qh_errexit(qh_ERRqhull, NULL, NULL);
  }
  center= (pointT *)qh_memalloc(qh normal_size);
  for (k=0; k < qh hull_dim; k++) {
    coord= center+k;
    *coord= 0.0;
    FOREACHvertex_(vertices)
      *coord += vertex->point[k];
    *coord /= count;  /* count>=2 by QH6003 */
  }
  return(center);
} /* getcenter */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="getcentrum">-</a>

  qh_getcentrum( facet )
    returns the centrum for a facet as a new point

  notes:
    allocates the centrum
*/
pointT *qh_getcentrum(facetT *facet) {
  realT dist;
  pointT *centrum, *point;

  point= qh_getcenter(facet->vertices);
  zzinc_(Zcentrumtests);
  qh_distplane(point, facet, &dist);
  centrum= qh_projectpoint(point, facet, dist);
  qh_memfree(point, qh normal_size);
  trace4((qh ferr, 4007, "qh_getcentrum: for f%d, %d vertices dist= %2.2g\n",
          facet->id, qh_setsize(facet->vertices), dist));
  return centrum;
} /* getcentrum */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="getdistance">-</a>

  qh_getdistance( facet, neighbor, mindist, maxdist )
    returns the maxdist and mindist distance of any vertex from neighbor

  returns:
    the max absolute value

  design:
    for each vertex of facet that is not in neighbor
      test the distance from vertex to neighbor
*/
realT qh_getdistance(facetT *facet, facetT *neighbor, realT *mindist, realT *maxdist) {
  vertexT *vertex, **vertexp;
  realT dist, maxd, mind;

  FOREACHvertex_(facet->vertices)
    vertex->seen= False;
  FOREACHvertex_(neighbor->vertices)
    vertex->seen= True;
  mind= 0.0;
  maxd= 0.0;
  FOREACHvertex_(facet->vertices) {
    if (!vertex->seen) {
      zzinc_(Zbestdist);
      qh_distplane(vertex->point, neighbor, &dist);
      if (dist < mind)
        mind= dist;
      else if (dist > maxd)
        maxd= dist;
    }
  }
  *mindist= mind;
  *maxdist= maxd;
  mind= -mind;
  if (maxd > mind)
    return maxd;
  else
    return mind;
} /* getdistance */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="normalize">-</a>

  qh_normalize( normal, dim, toporient )
    normalize a vector and report if too small
    does not use min norm

  see:
    qh_normalize2
*/
void qh_normalize(coordT *normal, int dim, boolT toporient) {
  qh_normalize2( normal, dim, toporient, NULL, NULL);
} /* normalize */

/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="normalize2">-</a>

  qh_normalize2( normal, dim, toporient, minnorm, ismin )
    normalize a vector and report if too small
    qh.MINdenom/MINdenom1 are the upper limits for divide overflow

  returns:
    normalized vector
    flips sign if !toporient
    if minnorm non-NULL,
      sets ismin if normal < minnorm

  notes:
    if zero norm
       sets all elements to sqrt(1.0/dim)
    if divide by zero (divzero())
       sets largest element to   +/-1
       bumps Znearlysingular

  design:
    computes norm
    test for minnorm
    if not near zero
      normalizes normal
    else if zero norm
      sets normal to standard value
    else
      uses qh_divzero to normalize
      if nearzero
        sets norm to direction of maximum value
*/
void qh_normalize2(coordT *normal, int dim, boolT toporient,
            realT *minnorm, boolT *ismin) {
  int k;
  realT *colp, *maxp, norm= 0, temp, *norm1, *norm2, *norm3;
  boolT zerodiv;

  norm1= normal+1;
  norm2= normal+2;
  norm3= normal+3;
  if (dim == 2)
    norm= sqrt((*normal)*(*normal) + (*norm1)*(*norm1));
  else if (dim == 3)
    norm= sqrt((*normal)*(*normal) + (*norm1)*(*norm1) + (*norm2)*(*norm2));
  else if (dim == 4) {
    norm= sqrt((*normal)*(*normal) + (*norm1)*(*norm1) + (*norm2)*(*norm2)
               + (*norm3)*(*norm3));
  }else if (dim > 4) {
    norm= (*normal)*(*normal) + (*norm1)*(*norm1) + (*norm2)*(*norm2)
               + (*norm3)*(*norm3);
    for (k=dim-4, colp=normal+4; k--; colp++)
      norm += (*colp) * (*colp);
    norm= sqrt(norm);
  }
  if (minnorm) {
    if (norm < *minnorm)
      *ismin= True;
    else
      *ismin= False;
  }
  wmin_(Wmindenom, norm);
  if (norm > qh MINdenom) {
    if (!toporient)
      norm= -norm;
    *normal /= norm;
    *norm1 /= norm;
    if (dim == 2)
      ; /* all done */
    else if (dim == 3)
      *norm2 /= norm;
    else if (dim == 4) {
      *norm2 /= norm;
      *norm3 /= norm;
    }else if (dim >4) {
      *norm2 /= norm;
      *norm3 /= norm;
      for (k=dim-4, colp=normal+4; k--; )
        *colp++ /= norm;
    }
  }else if (norm == 0.0) {
    temp= sqrt(1.0/dim);
    for (k=dim, colp=normal; k--; )
      *colp++ = temp;
  }else {
    if (!toporient)
      norm= -norm;
    for (k=dim, colp=normal; k--; colp++) { /* k used below */
      temp= qh_divzero(*colp, norm, qh MINdenom_1, &zerodiv);
      if (!zerodiv)
        *colp= temp;
      else {
        maxp= qh_maxabsval(normal, dim);
        temp= ((*maxp * norm >= 0.0) ? 1.0 : -1.0);
        for (k=dim, colp=normal; k--; colp++)
          *colp= 0.0;
        *maxp= temp;
        zzinc_(Znearlysingular);
        trace0((qh ferr, 1, "qh_normalize: norm=%2.2g too small during p%d\n",
               norm, qh furthest_id));
        return;
      }
    }
  }
} /* normalize */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="projectpoint">-</a>

  qh_projectpoint( point, facet, dist )
    project point onto a facet by dist

  returns:
    returns a new point

  notes:
    if dist= distplane(point,facet)
      this projects point to hyperplane
    assumes qh_memfree_() is valid for normal_size
*/
pointT *qh_projectpoint(pointT *point, facetT *facet, realT dist) {
  pointT *newpoint, *np, *normal;
  int normsize= qh normal_size;
  int k;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */

  qh_memalloc_(normsize, freelistp, newpoint, pointT);
  np= newpoint;
  normal= facet->normal;
  for (k=qh hull_dim; k--; )
    *(np++)= *point++ - dist * *normal++;
  return(newpoint);
} /* projectpoint */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="setfacetplane">-</a>

  qh_setfacetplane( facet )
    sets the hyperplane for a facet
    if qh.RANDOMdist, joggles hyperplane

  notes:
    uses global buffers qh.gm_matrix and qh.gm_row
    overwrites facet->normal if already defined
    updates Wnewvertex if PRINTstatistics
    sets facet->upperdelaunay if upper envelope of Delaunay triangulation

  design:
    copy vertex coordinates to qh.gm_matrix/gm_row
    compute determinate
    if nearzero
      recompute determinate with gaussian elimination
      if nearzero
        force outside orientation by testing interior point
*/
void qh_setfacetplane(facetT *facet) {
  pointT *point;
  vertexT *vertex, **vertexp;
  int normsize= qh normal_size;
  int k,i, oldtrace= 0;
  realT dist;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */
  coordT *coord, *gmcoord;
  pointT *point0= SETfirstt_(facet->vertices, vertexT)->point;
  boolT nearzero= False;

  zzinc_(Zsetplane);
  if (!facet->normal)
    qh_memalloc_(normsize, freelistp, facet->normal, coordT);
  if (facet == qh tracefacet) {
    oldtrace= qh IStracing;
    qh IStracing= 5;
    qh_fprintf(qh ferr, 8012, "qh_setfacetplane: facet f%d created.\n", facet->id);
    qh_fprintf(qh ferr, 8013, "  Last point added to hull was p%d.", qh furthest_id);
    if (zzval_(Ztotmerge))
      qh_fprintf(qh ferr, 8014, "  Last merge was #%d.", zzval_(Ztotmerge));
    qh_fprintf(qh ferr, 8015, "\n\nCurrent summary is:\n");
      qh_printsummary(qh ferr);
  }
  if (qh hull_dim <= 4) {
    i= 0;
    if (qh RANDOMdist) {
      gmcoord= qh gm_matrix;
      FOREACHvertex_(facet->vertices) {
        qh gm_row[i++]= gmcoord;
        coord= vertex->point;
        for (k=qh hull_dim; k--; )
          *(gmcoord++)= *coord++ * qh_randomfactor(qh RANDOMa, qh RANDOMb);
      }
    }else {
      FOREACHvertex_(facet->vertices)
       qh gm_row[i++]= vertex->point;
    }
    qh_sethyperplane_det(qh hull_dim, qh gm_row, point0, facet->toporient,
                facet->normal, &facet->offset, &nearzero);
  }
  if (qh hull_dim > 4 || nearzero) {
    i= 0;
    gmcoord= qh gm_matrix;
    FOREACHvertex_(facet->vertices) {
      if (vertex->point != point0) {
        qh gm_row[i++]= gmcoord;
        coord= vertex->point;
        point= point0;
        for (k=qh hull_dim; k--; )
          *(gmcoord++)= *coord++ - *point++;
      }
    }
    qh gm_row[i]= gmcoord;  /* for areasimplex */
    if (qh RANDOMdist) {
      gmcoord= qh gm_matrix;
      for (i=qh hull_dim-1; i--; ) {
        for (k=qh hull_dim; k--; )
          *(gmcoord++) *= qh_randomfactor(qh RANDOMa, qh RANDOMb);
      }
    }
    qh_sethyperplane_gauss(qh hull_dim, qh gm_row, point0, facet->toporient,
                facet->normal, &facet->offset, &nearzero);
    if (nearzero) {
      if (qh_orientoutside(facet)) {
        trace0((qh ferr, 2, "qh_setfacetplane: flipped orientation after testing interior_point during p%d\n", qh furthest_id));
      /* this is part of using Gaussian Elimination.  For example in 5-d
           1 1 1 1 0
           1 1 1 1 1
           0 0 0 1 0
           0 1 0 0 0
           1 0 0 0 0
           norm= 0.38 0.38 -0.76 0.38 0
         has a determinate of 1, but g.e. after subtracting pt. 0 has
         0's in the diagonal, even with full pivoting.  It does work
         if you subtract pt. 4 instead. */
      }
    }
  }
  facet->upperdelaunay= False;
  if (qh DELAUNAY) {
    if (qh UPPERdelaunay) {     /* matches qh_triangulate_facet and qh.lower_threshold in qh_initbuild */
      if (facet->normal[qh hull_dim -1] >= qh ANGLEround * qh_ZEROdelaunay)
        facet->upperdelaunay= True;
    }else {
      if (facet->normal[qh hull_dim -1] > -qh ANGLEround * qh_ZEROdelaunay)
        facet->upperdelaunay= True;
    }
  }
  if (qh PRINTstatistics || qh IStracing || qh TRACElevel || qh JOGGLEmax < REALmax) {
    qh old_randomdist= qh RANDOMdist;
    qh RANDOMdist= False;
    FOREACHvertex_(facet->vertices) {
      if (vertex->point != point0) {
        boolT istrace= False;
        zinc_(Zdiststat);
        qh_distplane(vertex->point, facet, &dist);
        dist= fabs_(dist);
        zinc_(Znewvertex);
        wadd_(Wnewvertex, dist);
        if (dist > wwval_(Wnewvertexmax)) {
          wwval_(Wnewvertexmax)= dist;
          if (dist > qh max_outside) {
            qh max_outside= dist;  /* used by qh_maxouter() */
            if (dist > qh TRACEdist)
              istrace= True;
          }
        }else if (-dist > qh TRACEdist)
          istrace= True;
        if (istrace) {
          qh_fprintf(qh ferr, 8016, "qh_setfacetplane: ====== vertex p%d(v%d) increases max_outside to %2.2g for new facet f%d last p%d\n",
                qh_pointid(vertex->point), vertex->id, dist, facet->id, qh furthest_id);
          qh_errprint("DISTANT", facet, NULL, NULL, NULL);
        }
      }
    }
    qh RANDOMdist= qh old_randomdist;
  }
  if (qh IStracing >= 3) {
    qh_fprintf(qh ferr, 8017, "qh_setfacetplane: f%d offset %2.2g normal: ",
             facet->id, facet->offset);
    for (k=0; k < qh hull_dim; k++)
      qh_fprintf(qh ferr, 8018, "%2.2g ", facet->normal[k]);
    qh_fprintf(qh ferr, 8019, "\n");
  }
  if (facet == qh tracefacet)
    qh IStracing= oldtrace;
} /* setfacetplane */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="sethyperplane_det">-</a>

  qh_sethyperplane_det( dim, rows, point0, toporient, normal, offset, nearzero )
    given dim X dim array indexed by rows[], one row per point,
        toporient(flips all signs),
        and point0 (any row)
    set normalized hyperplane equation from oriented simplex

  returns:
    normal (normalized)
    offset (places point0 on the hyperplane)
    sets nearzero if hyperplane not through points

  notes:
    only defined for dim == 2..4
    rows[] is not modified
    solves det(P-V_0, V_n-V_0, ..., V_1-V_0)=0, i.e. every point is on hyperplane
    see Bower & Woodworth, A programmer's geometry, Butterworths 1983.

  derivation of 3-d minnorm
    Goal: all vertices V_i within qh.one_merge of hyperplane
    Plan: exactly translate the facet so that V_0 is the origin
          exactly rotate the facet so that V_1 is on the x-axis and y_2=0.
          exactly rotate the effective perturbation to only effect n_0
             this introduces a factor of sqrt(3)
    n_0 = ((y_2-y_0)*(z_1-z_0) - (z_2-z_0)*(y_1-y_0)) / norm
    Let M_d be the max coordinate difference
    Let M_a be the greater of M_d and the max abs. coordinate
    Let u be machine roundoff and distround be max error for distance computation
    The max error for n_0 is sqrt(3) u M_a M_d / norm.  n_1 is approx. 1 and n_2 is approx. 0
    The max error for distance of V_1 is sqrt(3) u M_a M_d M_d / norm.  Offset=0 at origin
    Then minnorm = 1.8 u M_a M_d M_d / qh.ONEmerge
    Note that qh.one_merge is approx. 45.5 u M_a and norm is usually about M_d M_d

  derivation of 4-d minnorm
    same as above except rotate the facet so that V_1 on x-axis and w_2, y_3, w_3=0
     [if two vertices fixed on x-axis, can rotate the other two in yzw.]
    n_0 = det3_(...) = y_2 det2_(z_1, w_1, z_3, w_3) = - y_2 w_1 z_3
     [all other terms contain at least two factors nearly zero.]
    The max error for n_0 is sqrt(4) u M_a M_d M_d / norm
    Then minnorm = 2 u M_a M_d M_d M_d / qh.ONEmerge
    Note that qh.one_merge is approx. 82 u M_a and norm is usually about M_d M_d M_d
*/
void qh_sethyperplane_det(int dim, coordT **rows, coordT *point0,
          boolT toporient, coordT *normal, realT *offset, boolT *nearzero) {
  realT maxround, dist;
  int i;
  pointT *point;


  if (dim == 2) {
    normal[0]= dY(1,0);
    normal[1]= dX(0,1);
    qh_normalize2(normal, dim, toporient, NULL, NULL);
    *offset= -(point0[0]*normal[0]+point0[1]*normal[1]);
    *nearzero= False;  /* since nearzero norm => incident points */
  }else if (dim == 3) {
    normal[0]= det2_(dY(2,0), dZ(2,0),
                     dY(1,0), dZ(1,0));
    normal[1]= det2_(dX(1,0), dZ(1,0),
                     dX(2,0), dZ(2,0));
    normal[2]= det2_(dX(2,0), dY(2,0),
                     dX(1,0), dY(1,0));
    qh_normalize2(normal, dim, toporient, NULL, NULL);
    *offset= -(point0[0]*normal[0] + point0[1]*normal[1]
               + point0[2]*normal[2]);
    maxround= qh DISTround;
    for (i=dim; i--; ) {
      point= rows[i];
      if (point != point0) {
        dist= *offset + (point[0]*normal[0] + point[1]*normal[1]
               + point[2]*normal[2]);
        if (dist > maxround || dist < -maxround) {
          *nearzero= True;
          break;
        }
      }
    }
  }else if (dim == 4) {
    normal[0]= - det3_(dY(2,0), dZ(2,0), dW(2,0),
                        dY(1,0), dZ(1,0), dW(1,0),
                        dY(3,0), dZ(3,0), dW(3,0));
    normal[1]=   det3_(dX(2,0), dZ(2,0), dW(2,0),
                        dX(1,0), dZ(1,0), dW(1,0),
                        dX(3,0), dZ(3,0), dW(3,0));
    normal[2]= - det3_(dX(2,0), dY(2,0), dW(2,0),
                        dX(1,0), dY(1,0), dW(1,0),
                        dX(3,0), dY(3,0), dW(3,0));
    normal[3]=   det3_(dX(2,0), dY(2,0), dZ(2,0),
                        dX(1,0), dY(1,0), dZ(1,0),
                        dX(3,0), dY(3,0), dZ(3,0));
    qh_normalize2(normal, dim, toporient, NULL, NULL);
    *offset= -(point0[0]*normal[0] + point0[1]*normal[1]
               + point0[2]*normal[2] + point0[3]*normal[3]);
    maxround= qh DISTround;
    for (i=dim; i--; ) {
      point= rows[i];
      if (point != point0) {
        dist= *offset + (point[0]*normal[0] + point[1]*normal[1]
               + point[2]*normal[2] + point[3]*normal[3]);
        if (dist > maxround || dist < -maxround) {
          *nearzero= True;
          break;
        }
      }
    }
  }
  if (*nearzero) {
    zzinc_(Zminnorm);
    trace0((qh ferr, 3, "qh_sethyperplane_det: degenerate norm during p%d.\n", qh furthest_id));
    zzinc_(Znearlysingular);
  }
} /* sethyperplane_det */


/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="sethyperplane_gauss">-</a>

  qh_sethyperplane_gauss( dim, rows, point0, toporient, normal, offset, nearzero )
    given(dim-1) X dim array of rows[i]= V_{i+1} - V_0 (point0)
    set normalized hyperplane equation from oriented simplex

  returns:
    normal (normalized)
    offset (places point0 on the hyperplane)

  notes:
    if nearzero
      orientation may be incorrect because of incorrect sign flips in gausselim
    solves [V_n-V_0,...,V_1-V_0, 0 .. 0 1] * N == [0 .. 0 1]
        or [V_n-V_0,...,V_1-V_0, 0 .. 0 1] * N == [0]
    i.e., N is normal to the hyperplane, and the unnormalized
        distance to [0 .. 1] is either 1 or   0

  design:
    perform gaussian elimination
    flip sign for negative values
    perform back substitution
    normalize result
    compute offset
*/
void qh_sethyperplane_gauss(int dim, coordT **rows, pointT *point0,
                boolT toporient, coordT *normal, coordT *offset, boolT *nearzero) {
  coordT *pointcoord, *normalcoef;
  int k;
  boolT sign= toporient, nearzero2= False;

  qh_gausselim(rows, dim-1, dim, &sign, nearzero);
  for (k=dim-1; k--; ) {
    if ((rows[k])[k] < 0)
      sign ^= 1;
  }
  if (*nearzero) {
    zzinc_(Znearlysingular);
    trace0((qh ferr, 4, "qh_sethyperplane_gauss: nearly singular or axis parallel hyperplane during p%d.\n", qh furthest_id));
    qh_backnormal(rows, dim-1, dim, sign, normal, &nearzero2);
  }else {
    qh_backnormal(rows, dim-1, dim, sign, normal, &nearzero2);
    if (nearzero2) {
      zzinc_(Znearlysingular);
      trace0((qh ferr, 5, "qh_sethyperplane_gauss: singular or axis parallel hyperplane at normalization during p%d.\n", qh furthest_id));
    }
  }
  if (nearzero2)
    *nearzero= True;
  qh_normalize2(normal, dim, True, NULL, NULL);
  pointcoord= point0;
  normalcoef= normal;
  *offset= -(*pointcoord++ * *normalcoef++);
  for (k=dim-1; k--; )
    *offset -= *pointcoord++ * *normalcoef++;
} /* sethyperplane_gauss */




/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly2_r.c
   implements polygons and simplicies

   see qh-poly_r.htm, poly_r.h and libqhull_r.h

   frequently used code is in poly_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/poly2_r.c#18 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*======== functions in alphabetical order ==========*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addfacetvertex">-</a>

  qh_addfacetvertex(qh, facet, newvertex )
    add newvertex to facet.vertices if not already there
    vertices are inverse sorted by vertex->id

  returns:
    True if new vertex for facet

  notes:
    see qh_replacefacetvertex
*/
boolT qh_addfacetvertex(qhT *qh, facetT *facet, vertexT *newvertex) {
  vertexT *vertex;
  int vertex_i= 0, vertex_n;
  boolT isnew= True;

  FOREACHvertex_i_(qh, facet->vertices) {
    if (vertex->id < newvertex->id) {
      break;
    }else if (vertex->id == newvertex->id) {
      isnew= False;
      break;
    }
  }
  if (isnew)
    qh_setaddnth(qh, &facet->vertices, vertex_i, newvertex);
  return isnew;
} /* addfacetvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="addhash">-</a>

  qh_addhash( newelem, hashtable, hashsize, hash )
    add newelem to linear hash table at hash if not already there
*/
void qh_addhash(void *newelem, setT *hashtable, int hashsize, int hash) {
  int scan;
  void *elem;

  for (scan= (int)hash; (elem= SETelem_(hashtable, scan));
       scan= (++scan >= hashsize ? 0 : scan)) {
    if (elem == newelem)
      break;
  }
  /* loop terminates because qh_HASHfactor >= 1.1 by qh_initbuffers */
  if (!elem)
    SETelem_(hashtable, scan)= newelem;
} /* addhash */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_bestdist">-</a>

  qh_check_bestdist(qh)
    check that all points are within max_outside of the nearest facet
    if qh.ONLYgood,
      ignores !good facets

  see:
    qh_check_maxout(), qh_outerinner()

  notes:
    only called from qh_check_points()
      seldom used since qh.MERGING is almost always set
    if notverified>0 at end of routine
      some points were well inside the hull.  If the hull contains
      a lens-shaped component, these points were not verified.  Use
      options 'Qi Tv' to verify all points.  (Exhaustive check also verifies)

  design:
    determine facet for each point (if any)
    for each point
      start with the assigned facet or with the first facet
      find the best facet for the point and check all coplanar facets
      error if point is outside of facet
*/
void qh_check_bestdist(qhT *qh) {
  boolT waserror= False, unassigned;
  facetT *facet, *bestfacet, *errfacet1= NULL, *errfacet2= NULL;
  facetT *facetlist;
  realT dist, maxoutside, maxdist= -REALmax;
  pointT *point;
  int numpart= 0, facet_i, facet_n, notgood= 0, notverified= 0;
  setT *facets;

  trace1((qh, qh->ferr, 1020, "qh_check_bestdist: check points below nearest facet.  Facet_list f%d\n",
      qh->facet_list->id));
  maxoutside= qh_maxouter(qh);
  maxoutside += qh->DISTround;
  /* one more qh.DISTround for check computation */
  trace1((qh, qh->ferr, 1021, "qh_check_bestdist: check that all points are within %2.2g of best facet\n", maxoutside));
  facets= qh_pointfacet(qh /* qh.facet_list */);
  if (!qh_QUICKhelp && qh->PRINTprecision)
    qh_fprintf(qh, qh->ferr, 8091, "\n\
qhull output completed.  Verifying that %d points are\n\
below %2.2g of the nearest %sfacet.\n",
             qh_setsize(qh, facets), maxoutside, (qh->ONLYgood ?  "good " : ""));
  FOREACHfacet_i_(qh, facets) {  /* for each point with facet assignment */
    if (facet)
      unassigned= False;
    else {
      unassigned= True;
      facet= qh->facet_list;
    }
    point= qh_point(qh, facet_i);
    if (point == qh->GOODpointp)
      continue;
    qh_distplane(qh, point, facet, &dist);
    numpart++;
    bestfacet= qh_findbesthorizon(qh, !qh_IScheckmax, point, facet, qh_NOupper, &dist, &numpart);
    /* occurs after statistics reported */
    maximize_(maxdist, dist);
    if (dist > maxoutside) {
      if (qh->ONLYgood && !bestfacet->good
      && !((bestfacet= qh_findgooddist(qh, point, bestfacet, &dist, &facetlist))
      && dist > maxoutside))
        notgood++;
      else {
        waserror= True;
        qh_fprintf(qh, qh->ferr, 6109, "qhull precision error (qh_check_bestdist): point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g\n",
                facet_i, bestfacet->id, dist, maxoutside);
        if (errfacet1 != bestfacet) {
          errfacet2= errfacet1;
          errfacet1= bestfacet;
        }
      }
    }else if (unassigned && dist < -qh->MAXcoplanar)
      notverified++;
  }
  qh_settempfree(qh, &facets);
  if (notverified && !qh->DELAUNAY && !qh_QUICKhelp && qh->PRINTprecision)
    qh_fprintf(qh, qh->ferr, 8092, "\n%d points were well inside the hull.  If the hull contains\n\
a lens-shaped component, these points were not verified.  Use\n\
options 'Qci Tv' to verify all points.\n", notverified);
  if (maxdist > qh->outside_err) {
    qh_fprintf(qh, qh->ferr, 6110, "qhull precision error (qh_check_bestdist): a coplanar point is %6.2g from convex hull.  The maximum value is qh.outside_err (%6.2g)\n",
              maxdist, qh->outside_err);
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
  }else if (waserror && qh->outside_err > REALmax/2)
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
  /* else if waserror, the error was logged to qh.ferr but does not effect the output */
  trace0((qh, qh->ferr, 20, "qh_check_bestdist: max distance outside %2.2g\n", maxdist));
} /* check_bestdist */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_maxout">-</a>

  qh_check_maxout(qh)
    updates qh.max_outside by checking all points against bestfacet
    if qh.ONLYgood, ignores !good facets

  returns:
    updates facet->maxoutside via qh_findbesthorizon()
    sets qh.maxoutdone
    if printing qh.min_vertex (qh_outerinner),
      it is updated to the current vertices
    removes inside/coplanar points from coplanarset as needed

  notes:
    defines coplanar as qh.min_vertex instead of qh.MAXcoplanar
    may not need to check near-inside points because of qh.MAXcoplanar
      and qh.KEEPnearinside (before it was -qh.DISTround)

  see also:
    qh_check_bestdist()

  design:
    if qh.min_vertex is needed
      for all neighbors of all vertices
        test distance from vertex to neighbor
    determine facet for each point (if any)
    for each point with an assigned facet
      find the best facet for the point and check all coplanar facets
        (updates outer planes)
    remove near-inside points from coplanar sets
*/
void qh_check_maxout(qhT *qh) {
  facetT *facet, *bestfacet, *neighbor, **neighborp, *facetlist, *maxbestfacet= NULL, *minfacet, *maxfacet, *maxpointfacet;
  realT dist, maxoutside, mindist, nearest;
  realT maxoutside_base, minvertex_base;
  pointT *point, *maxpoint= NULL;
  int numpart= 0, facet_i, facet_n, notgood= 0;
  setT *facets, *vertices;
  vertexT *vertex, *minvertex;

  trace1((qh, qh->ferr, 1022, "qh_check_maxout: check and update qh.min_vertex %2.2g and qh.max_outside %2.2g\n", qh->min_vertex, qh->max_outside));
  minvertex_base= fmin_(qh->min_vertex, -(qh->ONEmerge+qh->DISTround));
  maxoutside= mindist= 0.0;
  minvertex= qh->vertex_list;
  maxfacet= minfacet= maxpointfacet= qh->facet_list;
  if (qh->VERTEXneighbors
  && (qh->PRINTsummary || qh->KEEPinside || qh->KEEPcoplanar
        || qh->TRACElevel || qh->PRINTstatistics || qh->VERIFYoutput || qh->CHECKfrequently
        || qh->PRINTout[0] == qh_PRINTsummary || qh->PRINTout[0] == qh_PRINTnone)) {
    trace1((qh, qh->ferr, 1023, "qh_check_maxout: determine actual minvertex\n"));
    vertices= qh_pointvertex(qh /* qh.facet_list */);
    FORALLvertices {
      FOREACHneighbor_(vertex) {
        zinc_(Zdistvertex);  /* distance also computed by main loop below */
        qh_distplane(qh, vertex->point, neighbor, &dist);
        if (dist < mindist) {
          if (qh->min_vertex/minvertex_base > qh_WIDEmaxoutside && (qh->PRINTprecision || !qh->ALLOWwide)) {
            nearest= qh_vertex_bestdist(qh, neighbor->vertices);
            /* should be caught in qh_mergefacet */
            qh_fprintf(qh, qh->ferr, 7083, "Qhull precision warning: in post-processing (qh_check_maxout) p%d(v%d) is %2.2g below f%d nearest vertices %2.2g\n",
              qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, nearest);
          }
          mindist= dist;
          minvertex= vertex;
          minfacet= neighbor;
        }
#ifndef qh_NOtrace
        if (-dist > qh->TRACEdist || dist > qh->TRACEdist
        || neighbor == qh->tracefacet || vertex == qh->tracevertex) {
          nearest= qh_vertex_bestdist(qh, neighbor->vertices);
          qh_fprintf(qh, qh->ferr, 8093, "qh_check_maxout: p%d(v%d) is %.2g from f%d nearest vertices %2.2g\n",
                    qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, nearest);
        }
#endif
      }
    }
    if (qh->MERGING) {
      wmin_(Wminvertex, qh->min_vertex);
    }
    qh->min_vertex= mindist;
    qh_settempfree(qh, &vertices);
  }
  trace1((qh, qh->ferr, 1055, "qh_check_maxout: determine actual maxoutside\n"));
  maxoutside_base= fmax_(qh->max_outside, qh->ONEmerge+qh->DISTround);
  /* maxoutside_base is same as qh.MAXoutside without qh.MINoutside (qh_detmaxoutside) */
  facets= qh_pointfacet(qh /* qh.facet_list */);
  FOREACHfacet_i_(qh, facets) {     /* for each point with facet assignment */
    if (facet) {
      point= qh_point(qh, facet_i);
      if (point == qh->GOODpointp)
        continue;
      zzinc_(Ztotcheck);
      qh_distplane(qh, point, facet, &dist);
      numpart++;
      bestfacet= qh_findbesthorizon(qh, qh_IScheckmax, point, facet, !qh_NOupper, &dist, &numpart);
      if (bestfacet && dist >= maxoutside) { 
        if (qh->ONLYgood && !bestfacet->good
        && !((bestfacet= qh_findgooddist(qh, point, bestfacet, &dist, &facetlist))
        && dist > maxoutside)) {       
          notgood++;
        }else if (dist/maxoutside_base > qh_WIDEmaxoutside && (qh->PRINTprecision || !qh->ALLOWwide)) {
          nearest= qh_vertex_bestdist(qh, bestfacet->vertices);
          if (nearest < fmax_(qh->ONEmerge, qh->max_outside) * qh_RATIOcoplanaroutside * 2) {
            qh_fprintf(qh, qh->ferr, 7087, "Qhull precision warning: in post-processing (qh_check_maxout) p%d for f%d is %2.2g above twisted facet f%d nearest vertices %2.2g\n",
              qh_pointid(qh, point), facet->id, dist, bestfacet->id, nearest);
          }else {
            qh_fprintf(qh, qh->ferr, 7088, "Qhull precision warning: in post-processing (qh_check_maxout) p%d for f%d is %2.2g above hidden facet f%d nearest vertices %2.2g\n",
              qh_pointid(qh, point), facet->id, dist, bestfacet->id, nearest);
          }
          maxbestfacet= bestfacet;
        }
        maxoutside= dist;
        maxfacet= bestfacet;
        maxpoint= point;
        maxpointfacet= facet;
      }
      if (dist > qh->TRACEdist || (bestfacet && bestfacet == qh->tracefacet))
        qh_fprintf(qh, qh->ferr, 8094, "qh_check_maxout: p%d is %.2g above f%d\n",
              qh_pointid(qh, point), dist, (bestfacet ? bestfacet->id : UINT_MAX));
    }
  }
  zzadd_(Zcheckpart, numpart);
  qh_settempfree(qh, &facets);
  wval_(Wmaxout)= maxoutside - qh->max_outside;
  wmax_(Wmaxoutside, qh->max_outside);
  if (!qh->APPROXhull && maxoutside > qh->DISTround) { /* initial value for f.maxoutside */
    FORALLfacets {
      if (maxoutside < facet->maxoutside) {
        if (!qh->KEEPcoplanar) {
          maxoutside= facet->maxoutside;
        }else if (maxoutside + qh->DISTround < facet->maxoutside) { /* maxoutside is computed distance, e.g., rbox 100 s D3 t1547136913 | qhull R1e-3 Tcv Qc */
          qh_fprintf(qh, qh->ferr, 7082, "Qhull precision warning (qh_check_maxout): f%d.maxoutside (%4.4g) is greater than computed qh.max_outside (%2.2g) + qh.DISTround (%2.2g).  It should be less than or equal\n",
            facet->id, facet->maxoutside, maxoutside, qh->DISTround); 
        }
      }
    }
  }
  qh->max_outside= maxoutside; 
  qh_nearcoplanar(qh /* qh.facet_list */);
  qh->maxoutdone= True;
  trace1((qh, qh->ferr, 1024, "qh_check_maxout:  p%d(v%d) is qh.min_vertex %2.2g below facet f%d.  Point p%d for f%d is qh.max_outside %2.2g above f%d.  %d points are outside of not-good facets\n", 
    qh_pointid(qh, minvertex->point), minvertex->id, qh->min_vertex, minfacet->id, qh_pointid(qh, maxpoint), maxpointfacet->id, qh->max_outside, maxfacet->id, notgood));
  if(!qh->ALLOWwide) {
    if (maxoutside/maxoutside_base > qh_WIDEmaxoutside) {
      qh_fprintf(qh, qh->ferr, 6297, "Qhull precision error (qh_check_maxout): large increase in qh.max_outside during post-processing dist %2.2g (%.1fx).  See warning QH0032/QH0033.  Allow with 'Q12' (allow-wide) and 'Pp'\n",
        maxoutside, maxoutside/maxoutside_base);
      qh_errexit(qh, qh_ERRwide, maxbestfacet, NULL);
    }else if (!qh->APPROXhull && maxoutside_base > (qh->ONEmerge * qh_WIDEmaxoutside2)) {
      if (maxoutside > (qh->ONEmerge * qh_WIDEmaxoutside2)) {  /* wide facets may have been deleted */
        qh_fprintf(qh, qh->ferr, 6298, "Qhull precision error (qh_check_maxout): a facet merge, vertex merge, vertex, or coplanar point produced a wide facet %2.2g (%.1fx). Trace with option 'TWn' to identify the merge.   Allow with 'Q12' (allow-wide)\n",
          maxoutside_base, maxoutside_base/(qh->ONEmerge + qh->DISTround));
        qh_errexit(qh, qh_ERRwide, maxbestfacet, NULL);
      }
    }else if (qh->min_vertex/minvertex_base > qh_WIDEmaxoutside) {
      qh_fprintf(qh, qh->ferr, 6354, "Qhull precision error (qh_check_maxout): large increase in qh.min_vertex during post-processing dist %2.2g (%.1fx).  See warning QH7083.  Allow with 'Q12' (allow-wide) and 'Pp'\n",
        qh->min_vertex, qh->min_vertex/minvertex_base);
      qh_errexit(qh, qh_ERRwide, minfacet, NULL);
    }else if (minvertex_base < -(qh->ONEmerge * qh_WIDEmaxoutside2)) {
      if (qh->min_vertex < -(qh->ONEmerge * qh_WIDEmaxoutside2)) {  /* wide facets may have been deleted */
        qh_fprintf(qh, qh->ferr, 6380, "Qhull precision error (qh_check_maxout): a facet or vertex merge produced a wide facet: v%d below f%d distance %2.2g (%.1fx). Trace with option 'TWn' to identify the merge.  Allow with 'Q12' (allow-wide)\n",
          minvertex->id, minfacet->id, mindist, -qh->min_vertex/(qh->ONEmerge + qh->DISTround));
        qh_errexit(qh, qh_ERRwide, minfacet, NULL);
      }
    }
  }
} /* check_maxout */
#else /* qh_NOmerge */
void qh_check_maxout(qhT *qh) {
  QHULL_UNUSED(qh)
}
#endif

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_output">-</a>

  qh_check_output(qh)
    performs the checks at the end of qhull algorithm
    Maybe called after Voronoi output.  If so, it recomputes centrums since they are Voronoi centers instead.
*/
void qh_check_output(qhT *qh) {
  int i;

  if (qh->STOPcone)
    return;
  if (qh->VERIFYoutput || qh->IStracing || qh->CHECKfrequently) {
    qh_checkpolygon(qh, qh->facet_list);
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }else if (!qh->MERGING && qh_newstats(qh, qh->qhstat.precision, &i)) {
    qh_checkflipped_all(qh, qh->facet_list);
    qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }
} /* check_output */



/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_point">-</a>

  qh_check_point(qh, point, facet, maxoutside, maxdist, errfacet1, errfacet2, errcount )
    check that point is less than maxoutside from facet

  notes:
    only called from qh_checkpoints
    reports up to qh_MAXcheckpoint-1 errors per facet
*/
void qh_check_point(qhT *qh, pointT *point, facetT *facet, realT *maxoutside, realT *maxdist, facetT **errfacet1, facetT **errfacet2, int *errcount) {
  realT dist, nearest;

  /* occurs after statistics reported */
  qh_distplane(qh, point, facet, &dist);
  maximize_(*maxdist, dist);
  if (dist > *maxoutside) {
    (*errcount)++;
    if (*errfacet1 != facet) {
      *errfacet2= *errfacet1;
      *errfacet1= facet;
    }
    if (*errcount < qh_MAXcheckpoint) {
      nearest= qh_vertex_bestdist(qh, facet->vertices);
      qh_fprintf(qh, qh->ferr, 6111, "qhull precision error: point p%d is outside facet f%d, distance= %6.8g maxoutside= %6.8g nearest vertices %2.2g\n",
                qh_pointid(qh, point), facet->id, dist, *maxoutside, nearest);
    }
  }
} /* qh_check_point */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="check_points">-</a>

  qh_check_points(qh)
    checks that all points are inside all facets

  notes:
    if many points and qh_check_maxout not called (i.e., !qh.MERGING),
       calls qh_findbesthorizon via qh_check_bestdist (seldom done).
    ignores flipped facets
    maxoutside includes 2 qh.DISTrounds
      one qh.DISTround for the computed distances in qh_check_points
    qh_printafacet and qh_printsummary needs only one qh.DISTround
    the computation for qh.VERIFYdirect does not account for qh.other_points

  design:
    if many points
      use qh_check_bestdist()
    else
      for all facets
        for all points
          check that point is inside facet
*/
void qh_check_points(qhT *qh) {
  facetT *facet, *errfacet1= NULL, *errfacet2= NULL;
  realT total, maxoutside, maxdist= -REALmax;
  pointT *point, **pointp, *pointtemp;
  int errcount;
  boolT testouter;

  maxoutside= qh_maxouter(qh);
  maxoutside += qh->DISTround;
  /* one more qh.DISTround for check computation */
  trace1((qh, qh->ferr, 1025, "qh_check_points: check all points below %2.2g of all facet planes\n",
          maxoutside));
  if (qh->num_good)   /* miss counts other_points and !good facets */
     total= (float)qh->num_good * (float)qh->num_points;
  else
     total= (float)qh->num_facets * (float)qh->num_points;
  if (total >= qh_VERIFYdirect && !qh->maxoutdone) {
    if (!qh_QUICKhelp && qh->SKIPcheckmax && qh->MERGING)
      qh_fprintf(qh, qh->ferr, 7075, "qhull input warning: merging without checking outer planes('Q5' or 'Po').  Verify may report that a point is outside of a facet.\n");
    qh_check_bestdist(qh);
  }else {
    if (qh_MAXoutside && qh->maxoutdone)
      testouter= True;
    else
      testouter= False;
    if (!qh_QUICKhelp) {
      if (qh->MERGEexact)
        qh_fprintf(qh, qh->ferr, 7076, "qhull input warning: exact merge ('Qx').  Verify may report that a point is outside of a facet.  See qh-optq.htm#Qx\n");
      else if (qh->SKIPcheckmax || qh->NOnearinside)
        qh_fprintf(qh, qh->ferr, 7077, "qhull input warning: no outer plane check ('Q5') or no processing of near-inside points ('Q8').  Verify may report that a point is outside of a facet.\n");
    }
    if (qh->PRINTprecision) {
      if (testouter)
        qh_fprintf(qh, qh->ferr, 8098, "\n\
Output completed.  Verifying that all points are below outer planes of\n\
all %sfacets.  Will make %2.0f distance computations.\n",
              (qh->ONLYgood ?  "good " : ""), total);
      else
        qh_fprintf(qh, qh->ferr, 8099, "\n\
Output completed.  Verifying that all points are below %2.2g of\n\
all %sfacets.  Will make %2.0f distance computations.\n",
              maxoutside, (qh->ONLYgood ?  "good " : ""), total);
    }
    FORALLfacets {
      if (!facet->good && qh->ONLYgood)
        continue;
      if (facet->flipped)
        continue;
      if (!facet->normal) {
        qh_fprintf(qh, qh->ferr, 7061, "qhull warning (qh_check_points): missing normal for facet f%d\n", facet->id);
        if (!errfacet1)
          errfacet1= facet;
        continue;
      }
      if (testouter) {
#if qh_MAXoutside
        maxoutside= facet->maxoutside + 2 * qh->DISTround;
        /* one DISTround to actual point and another to computed point */
#endif
      }
      errcount= 0;
      FORALLpoints {
        if (point != qh->GOODpointp)
          qh_check_point(qh, point, facet, &maxoutside, &maxdist, &errfacet1, &errfacet2, &errcount);
      }
      FOREACHpoint_(qh->other_points) {
        if (point != qh->GOODpointp)
          qh_check_point(qh, point, facet, &maxoutside, &maxdist, &errfacet1, &errfacet2, &errcount);
      }
      if (errcount >= qh_MAXcheckpoint) {
        qh_fprintf(qh, qh->ferr, 6422, "qhull precision error (qh_check_points): %d additional points outside facet f%d, maxdist= %6.8g\n",
             errcount-qh_MAXcheckpoint+1, facet->id, maxdist);
      }
    }
    if (maxdist > qh->outside_err) {
      qh_fprintf(qh, qh->ferr, 6112, "qhull precision error (qh_check_points): a coplanar point is %6.2g from convex hull.  The maximum value(qh.outside_err) is %6.2g\n",
                maxdist, qh->outside_err );
      qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2 );
    }else if (errfacet1 && qh->outside_err > REALmax/2)
        qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2 );
    /* else if errfacet1, the error was logged to qh.ferr but does not effect the output */
    trace0((qh, qh->ferr, 21, "qh_check_points: max distance outside %2.2g\n", maxdist));
  }
} /* check_points */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkconvex">-</a>

  qh_checkconvex(qh, facetlist, fault )
    check that each ridge in facetlist is convex
    fault = qh_DATAfault if reporting errors from qh_initialhull with qh.ZEROcentrum
          = qh_ALGORITHMfault otherwise

  returns:
    counts Zconcaveridges and Zcoplanarridges
    errors if !qh.FORCEoutput ('Fo') and concaveridge or if merging a coplanar ridge
    overwrites Voronoi centers if set by qh_setvoronoi_all/qh_ASvoronoi

  notes:
    called by qh_initial_hull, qh_check_output, qh_all_merges ('Tc'), qh_build_withrestart ('QJ')
    does not test f.tricoplanar facets (qh_triangulate)
    must be no stronger than qh_test_appendmerge
    if not merging,
      tests vertices for neighboring simplicial facets < -qh.DISTround
    else if ZEROcentrum and simplicial facet,
      tests vertices for neighboring simplicial facets < 0.0
      tests centrums of neighboring nonsimplicial facets < 0.0
    else if ZEROcentrum 
      tests centrums of neighboring facets < 0.0
    else 
      tests centrums of neighboring facets < -qh.DISTround ('En' 'Rn')
    Does not test against -qh.centrum_radius since repeated computations may have different round-off errors (e.g., 'Rn')

  design:
    for all facets
      report flipped facets
      if ZEROcentrum and simplicial neighbors
        test vertices against neighbor
      else
        test centrum against neighbor
*/
void qh_checkconvex(qhT *qh, facetT *facetlist, int fault) {
  facetT *facet, *neighbor, **neighborp, *errfacet1=NULL, *errfacet2=NULL;
  vertexT *vertex;
  realT dist;
  pointT *centrum;
  boolT waserror= False, centrum_warning= False, tempcentrum= False, first_nonsimplicial= False, tested_simplicial, allsimplicial;
  int neighbor_i, neighbor_n;

  if (qh->ZEROcentrum) {
    trace1((qh, qh->ferr, 1064, "qh_checkconvex: check that facets are not-flipped and for qh.ZEROcentrum that simplicial vertices are below their neighbor (dist<0.0)\n"));
    first_nonsimplicial= True;
  }else if (!qh->MERGING) {
    trace1((qh, qh->ferr, 1026, "qh_checkconvex: check that facets are not-flipped and that simplicial vertices are convex by qh.DISTround ('En', 'Rn')\n"));
    first_nonsimplicial= True;
  }else
    trace1((qh, qh->ferr, 1062, "qh_checkconvex: check that facets are not-flipped and that their centrums are convex by qh.DISTround ('En', 'Rn') \n"));
  if (!qh->RERUN) {
    zzval_(Zconcaveridges)= 0;
    zzval_(Zcoplanarridges)= 0;
  }
  FORALLfacet_(facetlist) {
    if (facet->flipped) {
      qh_joggle_restart(qh, "flipped facet"); /* also tested by qh_checkflipped */
      qh_fprintf(qh, qh->ferr, 6113, "qhull precision error: f%d is flipped (interior point is outside)\n",
               facet->id);
      errfacet1= facet;
      waserror= True;
      continue;
    }
    if (facet->tricoplanar)
      continue;
    if (qh->MERGING && (!qh->ZEROcentrum || !facet->simplicial)) {
      allsimplicial= False;
      tested_simplicial= False;
    }else {
      allsimplicial= True;
      tested_simplicial= True;
      FOREACHneighbor_i_(qh, facet) {
        if (neighbor->tricoplanar)
          continue;
        if (!neighbor->simplicial) {
          allsimplicial= False;
          continue;
        }
        vertex= SETelemt_(facet->vertices, neighbor_i, vertexT);
        qh_distplane(qh, vertex->point, neighbor, &dist);
        if (dist >= -qh->DISTround) {
          if (fault == qh_DATAfault) {
            qh_joggle_restart(qh, "non-convex initial simplex");
            if (dist > qh->DISTround)
              qh_fprintf(qh, qh->ferr, 6114, "qhull precision error: initial simplex is not convex, since p%d(v%d) is %6.4g above opposite f%d\n", 
                  qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id);
            else
              qh_fprintf(qh, qh->ferr, 6379, "qhull precision error: initial simplex is not convex, since p%d(v%d) is within roundoff of opposite facet f%d (dist %6.4g)\n",
                  qh_pointid(qh, vertex->point), vertex->id, neighbor->id, dist);
            qh_errexit(qh, qh_ERRsingular, neighbor, NULL);
          }
          if (dist > qh->DISTround) {
            zzinc_(Zconcaveridges);
            qh_joggle_restart(qh, "concave ridge");
            qh_fprintf(qh, qh->ferr, 6115, "qhull precision error: f%d is concave to f%d, since p%d(v%d) is %6.4g above f%d\n",
              facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id);
            errfacet1= facet;
            errfacet2= neighbor;
            waserror= True;
          }else if (qh->ZEROcentrum) {
            if (dist > 0.0) {     /* qh_checkzero checked convex (dist < (- 2*qh->DISTround)), computation may differ e.g. 'Rn' */
              zzinc_(Zcoplanarridges);
              qh_joggle_restart(qh, "coplanar ridge");
              qh_fprintf(qh, qh->ferr, 6116, "qhull precision error: f%d is clearly not convex to f%d, since p%d(v%d) is %6.4g above or coplanar with f%d with qh.ZEROcentrum\n",
                facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id);
              errfacet1= facet;
              errfacet2= neighbor;
              waserror= True;
            }
          }else {
            zzinc_(Zcoplanarridges);
            qh_joggle_restart(qh, "coplanar ridge");
            trace0((qh, qh->ferr, 22, "qhull precision error: f%d is coplanar to f%d, since p%d(v%d) is within %6.4g of f%d, during p%d\n",
              facet->id, neighbor->id, qh_pointid(qh, vertex->point), vertex->id, dist, neighbor->id, qh->furthest_id));
          }
        }
      }
    }
    if (!allsimplicial) {
      if (first_nonsimplicial) {
        trace1((qh, qh->ferr, 1063, "qh_checkconvex: starting with f%d, also check that centrums of non-simplicial ridges are below their neighbors (dist<0.0)\n",
             facet->id));
        first_nonsimplicial= False;
      }
      if (qh->CENTERtype == qh_AScentrum) {
        if (!facet->center)
          facet->center= qh_getcentrum(qh, facet);
        centrum= facet->center;
      }else {
        if (!centrum_warning && !facet->simplicial) {  /* recomputed centrum correct for simplicial facets */
           centrum_warning= True;
           qh_fprintf(qh, qh->ferr, 7062, "qhull warning: recomputing centrums for convexity test.  This may lead to false, precision errors.\n");
        }
        centrum= qh_getcentrum(qh, facet);
        tempcentrum= True;
      }
      FOREACHneighbor_(facet) {
        if (neighbor->simplicial && tested_simplicial) /* tested above since f.simplicial */
          continue;
        if (neighbor->tricoplanar)
          continue;
        zzinc_(Zdistconvex);
        qh_distplane(qh, centrum, neighbor, &dist);
        if (dist > qh->DISTround) {
          zzinc_(Zconcaveridges);
          qh_joggle_restart(qh, "concave ridge");
          qh_fprintf(qh, qh->ferr, 6117, "qhull precision error: f%d is concave to f%d.  Centrum of f%d is %6.4g above f%d\n",
            facet->id, neighbor->id, facet->id, dist, neighbor->id);
          errfacet1= facet;
          errfacet2= neighbor;
          waserror= True;
        }else if (dist >= 0.0) {   /* if arithmetic always rounds the same,
                                     can test against centrum radius instead */
          zzinc_(Zcoplanarridges);
          qh_joggle_restart(qh, "coplanar ridge");
          qh_fprintf(qh, qh->ferr, 6118, "qhull precision error: f%d is coplanar or concave to f%d.  Centrum of f%d is %6.4g above f%d\n",
            facet->id, neighbor->id, facet->id, dist, neighbor->id);
          errfacet1= facet;
          errfacet2= neighbor;
          waserror= True;
        }
      }
      if (tempcentrum)
        qh_memfree(qh, centrum, qh->normal_size);
    }
  }
  if (waserror && !qh->FORCEoutput)
    qh_errexit2(qh, qh_ERRprec, errfacet1, errfacet2);
} /* checkconvex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkfacet">-</a>

  qh_checkfacet(qh, facet, newmerge, waserror )
    checks for consistency errors in facet
    newmerge set if from merge_r.c

  returns:
    sets waserror if any error occurs

  checks:
    vertex ids are inverse sorted
    unless newmerge, at least hull_dim neighbors and vertices (exactly if simplicial)
    if non-simplicial, at least as many ridges as neighbors
    neighbors are not duplicated
    ridges are not duplicated
    in 3-d, ridges=verticies
    (qh.hull_dim-1) ridge vertices
    neighbors are reciprocated
    ridge neighbors are facet neighbors and a ridge for every neighbor
    simplicial neighbors match facetintersect
    vertex intersection matches vertices of common ridges
    vertex neighbors and facet vertices agree
    all ridges have distinct vertex sets

  notes:
    called by qh_tracemerge and qh_checkpolygon
    uses neighbor->seen

  design:
    check sets
    check vertices
    check sizes of neighbors and vertices
    check for qh_MERGEridge and qh_DUPLICATEridge flags
    check neighbor set
    check ridge set
    check ridges, neighbors, and vertices
*/
void qh_checkfacet(qhT *qh, facetT *facet, boolT newmerge, boolT *waserrorp) {
  facetT *neighbor, **neighborp, *errother=NULL;
  ridgeT *ridge, **ridgep, *errridge= NULL, *ridge2;
  vertexT *vertex, **vertexp;
  unsigned int previousid= INT_MAX;
  int numneighbors, numvertices, numridges=0, numRvertices=0;
  boolT waserror= False;
  int skipA, skipB, ridge_i, ridge_n, i, last_v= qh->hull_dim-2;
  setT *intersection;

  trace4((qh, qh->ferr, 4088, "qh_checkfacet: check f%d newmerge? %d\n", facet->id, newmerge));
  if (facet->id >= qh->facet_id) {
    qh_fprintf(qh, qh->ferr, 6414, "qhull internal error (qh_checkfacet): unknown facet id f%d >= qh.facet_id (%d)\n", facet->id, qh->facet_id);
    waserror= True;
  }
  if (facet->visitid > qh->visit_id) {
    qh_fprintf(qh, qh->ferr, 6415, "qhull internal error (qh_checkfacet): expecting f%d.visitid <= qh.visit_id (%d).  Got visitid %d\n", facet->id, qh->visit_id, facet->visitid);
    waserror= True;
  }
  if (facet->visible && !qh->NEWtentative) {
    qh_fprintf(qh, qh->ferr, 6119, "qhull internal error (qh_checkfacet): facet f%d is on qh.visible_list\n",
      facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  if (facet->redundant && !facet->visible && qh_setsize(qh, qh->degen_mergeset)==0) {
    qh_fprintf(qh, qh->ferr, 6399, "qhull internal error (qh_checkfacet): redundant facet f%d not on qh.visible_list\n",
      facet->id);
    waserror= True;
  }
  if (facet->degenerate && !facet->visible && qh_setsize(qh, qh->degen_mergeset)==0) { 
    qh_fprintf(qh, qh->ferr, 6400, "qhull internal error (qh_checkfacet): degenerate facet f%d is not on qh.visible_list and qh.degen_mergeset is empty\n",
      facet->id);
    waserror= True;
  }
  if (!facet->normal) {
    qh_fprintf(qh, qh->ferr, 6120, "qhull internal error (qh_checkfacet): facet f%d does not have a normal\n",
      facet->id);
    waserror= True;
  }
  if (!facet->newfacet) {
    if (facet->dupridge) {
      qh_fprintf(qh, qh->ferr, 6349, "qhull internal error (qh_checkfacet): f%d is 'dupridge' but it is not a newfacet on qh.newfacet_list f%d\n",
        facet->id, getid_(qh->newfacet_list));
      waserror= True;
    }
    if (facet->newmerge) {
      qh_fprintf(qh, qh->ferr, 6383, "qhull internal error (qh_checkfacet): f%d is 'newmerge' but it is not a newfacet on qh.newfacet_list f%d.  Missing call to qh_reducevertices\n",  
        facet->id, getid_(qh->newfacet_list));
      waserror= True;
    }
  }
  qh_setcheck(qh, facet->vertices, "vertices for f", facet->id);
  qh_setcheck(qh, facet->ridges, "ridges for f", facet->id);
  qh_setcheck(qh, facet->outsideset, "outsideset for f", facet->id);
  qh_setcheck(qh, facet->coplanarset, "coplanarset for f", facet->id);
  qh_setcheck(qh, facet->neighbors, "neighbors for f", facet->id);
  FOREACHvertex_(facet->vertices) {
    if (vertex->deleted) {
      qh_fprintf(qh, qh->ferr, 6121, "qhull internal error (qh_checkfacet): deleted vertex v%d in f%d\n", vertex->id, facet->id);
      qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
      waserror= True;
    }
    if (vertex->id >= previousid) {
      qh_fprintf(qh, qh->ferr, 6122, "qhull internal error (qh_checkfacet): vertices of f%d are not in descending id order at v%d\n", facet->id, vertex->id);
      waserror= True;
      break;
    }
    previousid= vertex->id;
  }
  numneighbors= qh_setsize(qh, facet->neighbors);
  numvertices= qh_setsize(qh, facet->vertices);
  numridges= qh_setsize(qh, facet->ridges);
  if (facet->simplicial) {
    if (numvertices+numneighbors != 2*qh->hull_dim
    && !facet->degenerate && !facet->redundant) {
      qh_fprintf(qh, qh->ferr, 6123, "qhull internal error (qh_checkfacet): for simplicial facet f%d, #vertices %d + #neighbors %d != 2*qh->hull_dim\n",
                facet->id, numvertices, numneighbors);
      qh_setprint(qh, qh->ferr, "", facet->neighbors);
      waserror= True;
    }
  }else { /* non-simplicial */
    if (!newmerge
    &&(numvertices < qh->hull_dim || numneighbors < qh->hull_dim)
    && !facet->degenerate && !facet->redundant) {
      qh_fprintf(qh, qh->ferr, 6124, "qhull internal error (qh_checkfacet): for facet f%d, #vertices %d or #neighbors %d < qh->hull_dim\n",
         facet->id, numvertices, numneighbors);
       waserror= True;
    }
    /* in 3-d, can get a vertex twice in an edge list, e.g., RBOX 1000 s W1e-13 t995849315 D2 | QHULL d Tc Tv TP624 TW1e-13 T4 */
    if (numridges < numneighbors
    ||(qh->hull_dim == 3 && numvertices > numridges && !qh->NEWfacets)
    ||(qh->hull_dim == 2 && numridges + numvertices + numneighbors != 6)) {
      if (!facet->degenerate && !facet->redundant) {
        qh_fprintf(qh, qh->ferr, 6125, "qhull internal error (qh_checkfacet): for facet f%d, #ridges %d < #neighbors %d or(3-d) > #vertices %d or(2-d) not all 2\n",
            facet->id, numridges, numneighbors, numvertices);
        waserror= True;
      }
    }
  }
  FOREACHneighbor_(facet) {
    if (neighbor == qh_MERGEridge || neighbor == qh_DUPLICATEridge) {
      qh_fprintf(qh, qh->ferr, 6126, "qhull internal error (qh_checkfacet): facet f%d still has a MERGEridge or DUPLICATEridge neighbor\n", facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6401, "qhull internal error (qh_checkfacet): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      errother= neighbor;
      waserror= True;
    }
    neighbor->seen= True;
  }
  FOREACHneighbor_(facet) {
    if (!qh_setin(neighbor->neighbors, facet)) {
      qh_fprintf(qh, qh->ferr, 6127, "qhull internal error (qh_checkfacet): facet f%d has neighbor f%d, but f%d does not have neighbor f%d\n",
              facet->id, neighbor->id, neighbor->id, facet->id);
      errother= neighbor;
      waserror= True;
    }
    if (!neighbor->seen) {
      qh_fprintf(qh, qh->ferr, 6128, "qhull internal error (qh_checkfacet): facet f%d has a duplicate neighbor f%d\n",
              facet->id, neighbor->id);
      errother= neighbor;
      waserror= True;
    }
    neighbor->seen= False;
  }
  FOREACHridge_(facet->ridges) {
    qh_setcheck(qh, ridge->vertices, "vertices for r", ridge->id);
    ridge->seen= False;
  }
  FOREACHridge_(facet->ridges) {
    if (ridge->seen) {
      qh_fprintf(qh, qh->ferr, 6129, "qhull internal error (qh_checkfacet): facet f%d has a duplicate ridge r%d\n",
              facet->id, ridge->id);
      errridge= ridge;
      waserror= True;
    }
    ridge->seen= True;
    numRvertices= qh_setsize(qh, ridge->vertices);
    if (numRvertices != qh->hull_dim - 1) {
      qh_fprintf(qh, qh->ferr, 6130, "qhull internal error (qh_checkfacet): ridge between f%d and f%d has %d vertices\n",
                ridge->top->id, ridge->bottom->id, numRvertices);
      errridge= ridge;
      waserror= True;
    }
    neighbor= otherfacet_(ridge, facet);
    neighbor->seen= True;
    if (!qh_setin(facet->neighbors, neighbor)) {
      qh_fprintf(qh, qh->ferr, 6131, "qhull internal error (qh_checkfacet): for facet f%d, neighbor f%d of ridge r%d not in facet\n",
           facet->id, neighbor->id, ridge->id);
      errridge= ridge;
      waserror= True;
    }
    if (!facet->newfacet && !neighbor->newfacet) {
      if ((!ridge->tested) | ridge->nonconvex | ridge->mergevertex) {
        qh_fprintf(qh, qh->ferr, 6384, "qhull internal error (qh_checkfacet): ridge r%d is nonconvex (%d), mergevertex (%d) or not tested (%d) for facet f%d, neighbor f%d\n",
          ridge->id, ridge->nonconvex, ridge->mergevertex, ridge->tested, facet->id, neighbor->id);
        errridge= ridge;
        waserror= True;
      }
    }
  }
  if (!facet->simplicial) {
    FOREACHneighbor_(facet) {
      if (!neighbor->seen) {
        qh_fprintf(qh, qh->ferr, 6132, "qhull internal error (qh_checkfacet): facet f%d does not have a ridge for neighbor f%d\n",
              facet->id, neighbor->id);
        errother= neighbor;
        waserror= True;
      }
      intersection= qh_vertexintersect_new(qh, facet->vertices, neighbor->vertices);
      qh_settemppush(qh, intersection);
      FOREACHvertex_(facet->vertices) {
        vertex->seen= False;
        vertex->seen2= False;
      }
      FOREACHvertex_(intersection)
        vertex->seen= True;
      FOREACHridge_(facet->ridges) {
        if (neighbor != otherfacet_(ridge, facet))
            continue;
        FOREACHvertex_(ridge->vertices) {
          if (!vertex->seen) {
            qh_fprintf(qh, qh->ferr, 6133, "qhull internal error (qh_checkfacet): vertex v%d in r%d not in f%d intersect f%d\n",
                  vertex->id, ridge->id, facet->id, neighbor->id);
            qh_errexit(qh, qh_ERRqhull, facet, ridge);
          }
          vertex->seen2= True;
        }
      }
      if (!newmerge) {
        FOREACHvertex_(intersection) {
          if (!vertex->seen2) {
            if (!qh->MERGING) {
              qh_fprintf(qh, qh->ferr, 6420, "qhull topology error (qh_checkfacet): vertex v%d in f%d intersect f%d but not in a ridge.  Last point was p%d\n",
                     vertex->id, facet->id, neighbor->id, qh->furthest_id);
              if (!qh->FORCEoutput) {
                qh_errprint(qh, "ERRONEOUS", facet, neighbor, NULL, vertex);
                qh_errexit(qh, qh_ERRtopology, NULL, NULL);
              }
            }else {
              trace4((qh, qh->ferr, 4025, "qh_checkfacet: vertex v%d in f%d intersect f%d but not in a ridge.  Repaired by qh_remove_extravertices in qh_reducevertices\n",
                vertex->id, facet->id, neighbor->id));
            }
          }
        }
      }
      qh_settempfree(qh, &intersection);
    }
  }else { /* simplicial */
    FOREACHneighbor_(facet) {
      if (neighbor->simplicial && !facet->degenerate && !neighbor->degenerate) {
        skipA= SETindex_(facet->neighbors, neighbor);
        skipB= qh_setindex(neighbor->neighbors, facet);
        if (skipA<0 || skipB<0 || !qh_setequal_skip(facet->vertices, skipA, neighbor->vertices, skipB)) {
          qh_fprintf(qh, qh->ferr, 6135, "qhull internal error (qh_checkfacet): facet f%d skip %d and neighbor f%d skip %d do not match \n",
                   facet->id, skipA, neighbor->id, skipB);
          errother= neighbor;
          waserror= True;
        }
      }
    }
  }
  if (!newmerge && qh->CHECKduplicates && qh->hull_dim < 5 && (qh->IStracing > 2 || qh->CHECKfrequently)) {
    FOREACHridge_i_(qh, facet->ridges) {           /* expensive, if was merge and qh_maybe_duplicateridges hasn't been called yet */
      if (!ridge->mergevertex) {
        for (i=ridge_i+1; i < ridge_n; i++) {
          ridge2= SETelemt_(facet->ridges, i, ridgeT);
          if (SETelem_(ridge->vertices, last_v) == SETelem_(ridge2->vertices, last_v)) { /* SETfirst is likely to be the same */
            if (SETfirst_(ridge->vertices) == SETfirst_(ridge2->vertices)) {
              if (qh_setequal(ridge->vertices, ridge2->vertices)) {
                qh_fprintf(qh, qh->ferr, 6294, "qhull internal error (qh_checkfacet): ridges r%d and r%d (f%d) have the same vertices\n", /* same as duplicate ridge */
                    ridge->id, ridge2->id, facet->id);
                errridge= ridge;
                waserror= True;
              }
            }
          }
        }
      }
    }
  }
  if (waserror) {
    qh_errprint(qh, "ERRONEOUS", facet, errother, errridge, NULL);
    *waserrorp= True;
  }
} /* checkfacet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkflipped_all">-</a>

  qh_checkflipped_all(qh, facetlist )
    checks orientation of facets in list against interior point

  notes:
    called by qh_checkoutput
*/
void qh_checkflipped_all(qhT *qh, facetT *facetlist) {
  facetT *facet;
  boolT waserror= False;
  realT dist;

  if (facetlist == qh->facet_list)
    zzval_(Zflippedfacets)= 0;
  FORALLfacet_(facetlist) {
    if (facet->normal && !qh_checkflipped(qh, facet, &dist, !qh_ALL)) {
      qh_fprintf(qh, qh->ferr, 6136, "qhull precision error: facet f%d is flipped, distance= %6.12g\n",
              facet->id, dist);
      if (!qh->FORCEoutput) {
        qh_errprint(qh, "ERRONEOUS", facet, NULL, NULL, NULL);
        waserror= True;
      }
    }
  }
  if (waserror) {
    qh_fprintf(qh, qh->ferr, 8101, "\n\
A flipped facet occurs when its distance to the interior point is\n\
greater than or equal to %2.2g, the maximum roundoff error.\n", -qh->DISTround);
    qh_errexit(qh, qh_ERRprec, NULL, NULL);
  }
} /* checkflipped_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checklists">-</a>

  qh_checklists(qh, facetlist )
    Check and repair facetlist and qh.vertex_list for infinite loops or overwritten facets
    Checks that qh.newvertex_list is on qh.vertex_list
    if facetlist is qh.facet_list
      Checks that qh.visible_list and qh.newfacet_list are on qh.facet_list
    Updates qh.facetvisit and qh.vertexvisit

  returns:
    True if no errors found
    If false, repairs erroneous lists to prevent infinite loops by FORALL macros

  notes:
    called by qh_buildtracing, qh_checkpolygon, qh_collectstatistics, qh_printfacetlist, qh_printsummary
    not called by qh_printlists

  design:
    if facetlist
      check qh.facet_tail
      for each facet
        check for infinite loop or overwritten facet
        check previous facet
      if facetlist is qh.facet_list
        check qh.next_facet, qh.visible_list and qh.newfacet_list
    if vertexlist
      check qh.vertex_tail
      for each vertex
        check for infinite loop or overwritten vertex
        check previous vertex
      check qh.newvertex_list
*/
boolT qh_checklists(qhT *qh, facetT *facetlist) {
  facetT *facet, *errorfacet= NULL, *errorfacet2= NULL, *previousfacet;
  vertexT *vertex, *vertexlist, *previousvertex, *errorvertex= NULL;
  boolT waserror= False, newseen= False, nextseen= False, newvertexseen= False, visibleseen= False;

  if (facetlist == qh->newfacet_list || facetlist == qh->visible_list) {
    vertexlist= qh->vertex_list;
    previousvertex= NULL;
    trace2((qh, qh->ferr, 2110, "qh_checklists: check qh.%s_list f%d and qh.vertex_list v%d\n", 
        (facetlist == qh->newfacet_list ? "newfacet" : "visible"), facetlist->id, getid_(vertexlist)));
  }else {
    vertexlist= qh->vertex_list;
    previousvertex= NULL;
    trace2((qh, qh->ferr, 2111, "qh_checklists: check %slist f%d and qh.vertex_list v%d\n", 
        (facetlist == qh->facet_list ? "qh.facet_" : "facet"), getid_(facetlist), getid_(vertexlist)));
  }
  if (facetlist) {
    if (qh->facet_tail == NULL || qh->facet_tail->id != 0 || qh->facet_tail->next != NULL) {
      qh_fprintf(qh, qh->ferr, 6397, "qhull internal error (qh_checklists): either qh.facet_tail f%d is NULL, or its id is not 0, or its next is not NULL\n", 
          getid_(qh->facet_tail));
      qh_errexit(qh, qh_ERRqhull, qh->facet_tail, NULL);
    }
    previousfacet= (facetlist == qh->facet_list ? NULL : facetlist->previous);
    qh->visit_id++;
    FORALLfacet_(facetlist) {
      if (facet->visitid >= qh->visit_id || facet->id >= qh->facet_id) {
        waserror= True;
        errorfacet= facet;
        errorfacet2= previousfacet;
        if (facet->visitid == qh->visit_id)
          qh_fprintf(qh, qh->ferr, 6039, "qhull internal error (qh_checklists): f%d already in facetlist causing an infinite loop ... f%d > f%d ... > f%d > f%d.  Truncate facetlist at f%d\n", 
            facet->id, facet->id, facet->next->id, getid_(previousfacet), facet->id, getid_(previousfacet));
        else
          qh_fprintf(qh, qh->ferr, 6350, "qhull internal error (qh_checklists): unknown or overwritten facet f%d, either id >= qh.facet_id (%d) or f.visitid %u > qh.visit_id %u.  Facetlist terminated at previous facet f%d\n", 
              facet->id, qh->facet_id, facet->visitid, qh->visit_id, getid_(previousfacet));
        if (previousfacet)
          previousfacet->next= qh->facet_tail;
        else
          facetlist= qh->facet_tail;
        break;
      }
      facet->visitid= qh->visit_id;
      if (facet->previous != previousfacet) {
        qh_fprintf(qh, qh->ferr, 6416, "qhull internal error (qh_checklists): expecting f%d.previous == f%d.  Got f%d\n",
          facet->id, getid_(previousfacet), getid_(facet->previous));
        waserror= True;
        errorfacet= facet;
        errorfacet2= facet->previous;
      }
      previousfacet= facet;
      if (facetlist == qh->facet_list) {
        if (facet == qh->visible_list) {
          if(newseen){
            qh_fprintf(qh, qh->ferr, 6285, "qhull internal error (qh_checklists): qh.visible_list f%d is after qh.newfacet_list f%d.  It should be at, before, or NULL\n",
              facet->id, getid_(qh->newfacet_list));
            waserror= True;
            errorfacet= facet;
            errorfacet2= qh->newfacet_list;
          }
          visibleseen= True;
        }
        if (facet == qh->newfacet_list)
          newseen= True;
        if (facet == qh->facet_next)
          nextseen= True;
      }
    }
    if (facetlist == qh->facet_list) {
      if (!nextseen && qh->facet_next && qh->facet_next->next) {
        qh_fprintf(qh, qh->ferr, 6369, "qhull internal error (qh_checklists): qh.facet_next f%d for qh_addpoint is not on qh.facet_list f%d\n", 
          qh->facet_next->id, facetlist->id);
        waserror= True;
        errorfacet= qh->facet_next;
        errorfacet2= facetlist;
      }
      if (!newseen && qh->newfacet_list && qh->newfacet_list->next) {
        qh_fprintf(qh, qh->ferr, 6286, "qhull internal error (qh_checklists): qh.newfacet_list f%d is not on qh.facet_list f%d\n", 
          qh->newfacet_list->id, facetlist->id);
        waserror= True;
        errorfacet= qh->newfacet_list;
        errorfacet2= facetlist;
      }
      if (!visibleseen && qh->visible_list && qh->visible_list->next) {
        qh_fprintf(qh, qh->ferr, 6138, "qhull internal error (qh_checklists): qh.visible_list f%d is not on qh.facet_list f%d\n", 
          qh->visible_list->id, facetlist->id);
        waserror= True;
        errorfacet= qh->visible_list;
        errorfacet2= facetlist;
      }
    }
  }
  if (vertexlist) {
    if (qh->vertex_tail == NULL || qh->vertex_tail->id != 0 || qh->vertex_tail->next != NULL) {
      qh_fprintf(qh, qh->ferr, 6366, "qhull internal error (qh_checklists): either qh.vertex_tail v%d is NULL, or its id is not 0, or its next is not NULL\n", 
           getid_(qh->vertex_tail));
      qh_errprint(qh, "ERRONEOUS", errorfacet, errorfacet2, NULL, qh->vertex_tail);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    qh->vertex_visit++;
    FORALLvertex_(vertexlist) {
      if (vertex->visitid >= qh->vertex_visit || vertex->id >= qh->vertex_id) {
        waserror= True;
        errorvertex= vertex;
        if (vertex->visitid == qh->visit_id)
          qh_fprintf(qh, qh->ferr, 6367, "qhull internal error (qh_checklists): v%d already in vertexlist causing an infinite loop ... v%d > v%d ... > v%d > v%d.  Truncate vertexlist at v%d\n", 
            vertex->id, vertex->id, vertex->next->id, getid_(previousvertex), vertex->id, getid_(previousvertex));
        else
          qh_fprintf(qh, qh->ferr, 6368, "qhull internal error (qh_checklists): unknown or overwritten vertex v%d, either id >= qh.vertex_id (%d) or v.visitid %u > qh.visit_id %u.  vertexlist terminated at previous vertex v%d\n", 
            vertex->id, qh->vertex_id, vertex->visitid, qh->visit_id, getid_(previousvertex));
        if (previousvertex)
          previousvertex->next= qh->vertex_tail;
        else
          vertexlist= qh->vertex_tail;
        break;
      }
      vertex->visitid= qh->vertex_visit;
      if (vertex->previous != previousvertex) {
        qh_fprintf(qh, qh->ferr, 6427, "qhull internal error (qh_checklists): expecting v%d.previous == v%d.  Got v%d\n",
              vertex->id, previousvertex, getid_(vertex->previous));
        waserror= True;
        errorvertex= vertex;
      }
      previousvertex= vertex;
      if(vertex == qh->newvertex_list)
        newvertexseen= True;
    }
    if(!newvertexseen && qh->newvertex_list && qh->newvertex_list->next) {
      qh_fprintf(qh, qh->ferr, 6287, "qhull internal error (qh_checklists): new vertex list v%d is not on vertex list\n", qh->newvertex_list->id);
      waserror= True;
      errorvertex= qh->newvertex_list;
    }
  }
  if (waserror) {
    qh_errprint(qh, "ERRONEOUS", errorfacet, errorfacet2, NULL, errorvertex);
    return False;
  }
  return True;
} /* checklists */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkpolygon">-</a>

  qh_checkpolygon(qh, facetlist )
    checks the correctness of the structure

  notes:
    called by qh_addpoint, qh_all_vertexmerge, qh_check_output, qh_initialhull, qh_prepare_output, qh_triangulate
    call with qh.facet_list or qh.newfacet_list or another list
    checks num_facets and num_vertices if qh.facet_list

  design:
    check and repair lists for infinite loop
    for each facet
      check f.newfacet and f.visible
      check facet and outside set if qh.NEWtentative and not f.newfacet, or not f.visible
    initializes vertexlist for qh.facet_list or qh.newfacet_list
    for each vertex
      check vertex
      check v.newfacet
    for each facet
      count f.ridges
      check and count f.vertices
    if checking qh.facet_list
      check facet count
      if qh.VERTEXneighbors
        check and count v.neighbors for all vertices
        check v.neighbors count and report possible causes of mismatch
        check that facets are in their v.neighbors
      check vertex count
*/
void qh_checkpolygon(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  facetT *errorfacet= NULL, *errorfacet2= NULL;
  vertexT *vertex, **vertexp, *vertexlist;
  int numfacets= 0, numvertices= 0, numridges= 0;
  int totvneighbors= 0, totfacetvertices= 0;
  boolT waserror= False, newseen= False, newvertexseen= False, nextseen= False, visibleseen= False;
  boolT checkfacet;

  trace1((qh, qh->ferr, 1027, "qh_checkpolygon: check all facets from f%d, qh.NEWtentative? %d\n", facetlist->id, qh->NEWtentative));
  if (!qh_checklists(qh, facetlist)) {
    waserror= True;
    qh_fprintf(qh, qh->ferr, 6374, "qhull internal error: qh_checklists failed in qh_checkpolygon\n");
    if (qh->num_facets < 4000)
      qh_printlists(qh);
  }
  if (facetlist != qh->facet_list || qh->ONLYgood)
    nextseen= True; /* allow f.outsideset */
  FORALLfacet_(facetlist) {
    if (facet == qh->visible_list)
      visibleseen= True;
    if (facet == qh->newfacet_list)
      newseen= True;
    if (facet->newfacet && !newseen && !visibleseen) {
        qh_fprintf(qh, qh->ferr, 6289, "qhull internal error (qh_checkpolygon): f%d is 'newfacet' but it is not on qh.newfacet_list f%d or visible_list f%d\n",  facet->id, getid_(qh->newfacet_list), getid_(qh->visible_list));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (!facet->newfacet && newseen) {
        qh_fprintf(qh, qh->ferr, 6292, "qhull internal error (qh_checkpolygon): f%d is on qh.newfacet_list f%d but it is not 'newfacet'\n",  facet->id, getid_(qh->newfacet_list));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (facet->visible != (visibleseen & !newseen)) {
      if(facet->visible)
        qh_fprintf(qh, qh->ferr, 6290, "qhull internal error (qh_checkpolygon): f%d is 'visible' but it is not on qh.visible_list f%d\n", facet->id, getid_(qh->visible_list));
      else
        qh_fprintf(qh, qh->ferr, 6291, "qhull internal error (qh_checkpolygon): f%d is on qh.visible_list f%d but it is not 'visible'\n", facet->id, qh->newfacet_list->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (qh->NEWtentative) {
      checkfacet= !facet->newfacet;
    }else {
      checkfacet= !facet->visible;
    }
    if(checkfacet) {
      if (!nextseen) {
        if (facet == qh->facet_next)  /* previous facets do not have outsideset */
          nextseen= True;
        else if (qh_setsize(qh, facet->outsideset)) {
          if (!qh->NARROWhull
#if !qh_COMPUTEfurthest
          || facet->furthestdist >= qh->MINoutside
#endif
          ) {
            qh_fprintf(qh, qh->ferr, 6137, "qhull internal error (qh_checkpolygon): f%d has outside points before qh.facet_next f%d\n",
                     facet->id, getid_(qh->facet_next));
            qh_errexit2(qh, qh_ERRqhull, facet, qh->facet_next);
          }
        }
      }
      numfacets++;
      qh_checkfacet(qh, facet, False, &waserror);
    }else if (facet->visible && qh->NEWfacets) {
      if (!SETempty_(facet->neighbors) || !SETempty_(facet->ridges)) {
        qh_fprintf(qh, qh->ferr, 6376, "qhull internal error (qh_checkpolygon): expecting empty f.neighbors and f.ridges for visible facet f%d.  Got %d neighbors and %d ridges\n", 
          facet->id, qh_setsize(qh, facet->neighbors), qh_setsize(qh, facet->ridges));
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
    }
  }
  if (facetlist == qh->facet_list) {
    vertexlist= qh->vertex_list;
  }else if (facetlist == qh->newfacet_list) {
    vertexlist= qh->newvertex_list;
  }else {
    vertexlist= NULL;
  }
  FORALLvertex_(vertexlist) {
    qh_checkvertex(qh, vertex, !qh_ALL, &waserror);
    if(vertex == qh->newvertex_list)
      newvertexseen= True;
    vertex->seen= False;
    vertex->visitid= 0;
    if(vertex->newfacet && !newvertexseen && !vertex->deleted) {
      qh_fprintf(qh, qh->ferr, 6288, "qhull internal error (qh_checkpolygon): v%d is 'newfacet' but it is not on new vertex list v%d\n", vertex->id, getid_(qh->newvertex_list));
      qh_errexit(qh, qh_ERRqhull, qh->visible_list, NULL);
    }
  }
  FORALLfacet_(facetlist) {
    if (facet->visible)
      continue;
    if (facet->simplicial)
      numridges += qh->hull_dim;
    else
      numridges += qh_setsize(qh, facet->ridges);
    FOREACHvertex_(facet->vertices) {
      vertex->visitid++;
      if (!vertex->seen) {
        vertex->seen= True;
        numvertices++;
        if (qh_pointid(qh, vertex->point) == qh_IDunknown) {
          qh_fprintf(qh, qh->ferr, 6139, "qhull internal error (qh_checkpolygon): unknown point %p for vertex v%d first_point %p\n",
                   vertex->point, vertex->id, qh->first_point);
          waserror= True;
        }
      }
    }
  }
  qh->vertex_visit += (unsigned int)numfacets;
  if (facetlist == qh->facet_list) {
    if (numfacets != qh->num_facets - qh->num_visible) {
      qh_fprintf(qh, qh->ferr, 6140, "qhull internal error (qh_checkpolygon): actual number of facets is %d, cumulative facet count is %d - %d visible facets\n",
              numfacets, qh->num_facets, qh->num_visible);
      waserror= True;
    }
    qh->vertex_visit++;
    if (qh->VERTEXneighbors) {
      FORALLvertices {
        if (!vertex->neighbors) {
          qh_fprintf(qh, qh->ferr, 6407, "qhull internal error (qh_checkpolygon): missing vertex neighbors for v%d\n", vertex->id);
          waserror= True;
        }
        qh_setcheck(qh, vertex->neighbors, "neighbors for v", vertex->id);
        if (vertex->deleted)
          continue;
        totvneighbors += qh_setsize(qh, vertex->neighbors);
      }
      FORALLfacet_(facetlist) {
        if (!facet->visible)
          totfacetvertices += qh_setsize(qh, facet->vertices);
      }
      if (totvneighbors != totfacetvertices) {
        qh_fprintf(qh, qh->ferr, 6141, "qhull internal error (qh_checkpolygon): vertex neighbors inconsistent (tot_vneighbors %d != tot_facetvertices %d).  Maybe duplicate or missing vertex\n",
                totvneighbors, totfacetvertices);
        waserror= True;
        FORALLvertices {
          if (vertex->deleted)
            continue;
          qh->visit_id++;
          FOREACHneighbor_(vertex) {
            if (neighbor->visitid==qh->visit_id) {
              qh_fprintf(qh, qh->ferr, 6275, "qhull internal error (qh_checkpolygon): facet f%d occurs twice in neighbors of vertex v%d\n",
                  neighbor->id, vertex->id);
              errorfacet2= errorfacet;
              errorfacet= neighbor;
            }
            neighbor->visitid= qh->visit_id;
            if (!qh_setin(neighbor->vertices, vertex)) {
              qh_fprintf(qh, qh->ferr, 6276, "qhull internal error (qh_checkpolygon): facet f%d is a neighbor of vertex v%d but v%d is not a vertex of f%d\n",
                  neighbor->id, vertex->id, vertex->id, neighbor->id);
              errorfacet2= errorfacet;
              errorfacet= neighbor;
            }
          }
        }
        FORALLfacet_(facetlist){
          if (!facet->visible) {
            /* vertices are inverse sorted and are unlikely to be duplicated */
            FOREACHvertex_(facet->vertices){
              if (!qh_setin(vertex->neighbors, facet)) {
                qh_fprintf(qh, qh->ferr, 6277, "qhull internal error (qh_checkpolygon): v%d is a vertex of facet f%d but f%d is not a neighbor of v%d\n",
                  vertex->id, facet->id, facet->id, vertex->id);
                errorfacet2= errorfacet;
                errorfacet= facet;
              }
            }
          }
        }
      }
    }
    if (numvertices != qh->num_vertices - qh_setsize(qh, qh->del_vertices)) {
      qh_fprintf(qh, qh->ferr, 6142, "qhull internal error (qh_checkpolygon): actual number of vertices is %d, cumulative vertex count is %d\n",
              numvertices, qh->num_vertices - qh_setsize(qh, qh->del_vertices));
      waserror= True;
    }
    if (qh->hull_dim == 2 && numvertices != numfacets) {
      qh_fprintf(qh, qh->ferr, 6143, "qhull internal error (qh_checkpolygon): #vertices %d != #facets %d\n",
        numvertices, numfacets);
      waserror= True;
    }
    if (qh->hull_dim == 3 && numvertices + numfacets - numridges/2 != 2) {
      qh_fprintf(qh, qh->ferr, 7063, "qhull warning: #vertices %d + #facets %d - #edges %d != 2.  A vertex appears twice in a edge list.  May occur during merging.\n",
          numvertices, numfacets, numridges/2);
      /* occurs if lots of merging and a vertex ends up twice in an edge list.  e.g., RBOX 1000 s W1e-13 t995849315 D2 | QHULL d Tc Tv */
    }
  }
  if (waserror)
    qh_errexit2(qh, qh_ERRqhull, errorfacet, errorfacet2);
} /* checkpolygon */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkvertex">-</a>

  qh_checkvertex(qh, vertex, allchecks, &waserror )
    check vertex for consistency
    if allchecks, checks vertex->neighbors

  returns:
    sets waserror if any error occurs

  notes:
    called by qh_tracemerge and qh_checkpolygon
    neighbors checked efficiently in qh_checkpolygon
*/
void qh_checkvertex(qhT *qh, vertexT *vertex, boolT allchecks, boolT *waserrorp) {
  boolT waserror= False;
  facetT *neighbor, **neighborp, *errfacet=NULL;

  if (qh_pointid(qh, vertex->point) == qh_IDunknown) {
    qh_fprintf(qh, qh->ferr, 6144, "qhull internal error (qh_checkvertex): unknown point id %p\n", vertex->point);
    waserror= True;
  }
  if (vertex->id >= qh->vertex_id) {
    qh_fprintf(qh, qh->ferr, 6145, "qhull internal error (qh_checkvertex): unknown vertex id v%d >= qh.vertex_id (%d)\n", vertex->id, qh->vertex_id);
    waserror= True;
  }
  if (vertex->visitid > qh->vertex_visit) {
    qh_fprintf(qh, qh->ferr, 6413, "qhull internal error (qh_checkvertex): expecting v%d.visitid <= qh.vertex_visit (%d).  Got visitid %d\n", vertex->id, qh->vertex_visit, vertex->visitid);
    waserror= True;
  }
  if (allchecks && !waserror && !vertex->deleted) {
    if (qh_setsize(qh, vertex->neighbors)) {
      FOREACHneighbor_(vertex) {
        if (!qh_setin(neighbor->vertices, vertex)) {
          qh_fprintf(qh, qh->ferr, 6146, "qhull internal error (qh_checkvertex): neighbor f%d does not contain v%d\n", neighbor->id, vertex->id);
          errfacet= neighbor;
          waserror= True;
        }
      }
    }
  }
  if (waserror) {
    qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
    if (errfacet)
      qh_errexit(qh, qh_ERRqhull, errfacet, NULL);
    *waserrorp= True;
  }
} /* checkvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="clearcenters">-</a>

  qh_clearcenters(qh, type )
    clear old data from facet->center

  notes:
    sets new centertype
    nop if CENTERtype is the same
*/
void qh_clearcenters(qhT *qh, qh_CENTER type) {
  facetT *facet;

  if (qh->CENTERtype != type) {
    FORALLfacets {
      if (facet->tricoplanar && !facet->keepcentrum)
          facet->center= NULL;  /* center is owned by the ->keepcentrum facet */
      else if (qh->CENTERtype == qh_ASvoronoi){
        if (facet->center) {
          qh_memfree(qh, facet->center, qh->center_size);
          facet->center= NULL;
        }
      }else /* qh.CENTERtype == qh_AScentrum */ {
        if (facet->center) {
          qh_memfree(qh, facet->center, qh->normal_size);
          facet->center= NULL;
        }
      }
    }
    qh->CENTERtype= type;
  }
  trace2((qh, qh->ferr, 2043, "qh_clearcenters: switched to center type %d\n", type));
} /* clearcenters */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="createsimplex">-</a>

  qh_createsimplex(qh, vertices )
    creates a simplex from a set of vertices

  returns:
    initializes qh.facet_list to the simplex

  notes: 
    only called by qh_initialhull

  design:
    for each vertex
      create a new facet
    for each new facet
      create its neighbor set
*/
void qh_createsimplex(qhT *qh, setT *vertices /* qh.facet_list */) {
  facetT *facet= NULL, *newfacet;
  boolT toporient= True;
  int vertex_i, vertex_n, nth;
  setT *newfacets= qh_settemp(qh, qh->hull_dim+1);
  vertexT *vertex;

  FOREACHvertex_i_(qh, vertices) {
    newfacet= qh_newfacet(qh);
    newfacet->vertices= qh_setnew_delnthsorted(qh, vertices, vertex_n, vertex_i, 0);
    if (toporient)
      newfacet->toporient= True;
    qh_appendfacet(qh, newfacet);
    newfacet->newfacet= True;
    qh_appendvertex(qh, vertex);
    qh_setappend(qh, &newfacets, newfacet);
    toporient ^= True;
  }
  FORALLnew_facets {
    nth= 0;
    FORALLfacet_(qh->newfacet_list) {
      if (facet != newfacet)
        SETelem_(newfacet->neighbors, nth++)= facet;
    }
    qh_settruncate(qh, newfacet->neighbors, qh->hull_dim);
  }
  qh_settempfree(qh, &newfacets);
  trace1((qh, qh->ferr, 1028, "qh_createsimplex: created simplex\n"));
} /* createsimplex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delridge">-</a>

  qh_delridge(qh, ridge )
    delete a ridge's vertices and frees its memory

  notes:
    assumes r.top->ridges and r.bottom->ridges have been updated
*/
void qh_delridge(qhT *qh, ridgeT *ridge) {

  if (ridge == qh->traceridge)
    qh->traceridge= NULL;
  qh_setfree(qh, &(ridge->vertices));
  qh_memfree(qh, ridge, (int)sizeof(ridgeT));
} /* delridge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delvertex">-</a>

  qh_delvertex(qh, vertex )
    deletes a vertex and frees its memory

  notes:
    assumes vertex->adjacencies have been updated if needed
    unlinks from vertex_list
*/
void qh_delvertex(qhT *qh, vertexT *vertex) {

  if (vertex->deleted && !vertex->partitioned && !qh->NOerrexit) {
    qh_fprintf(qh, qh->ferr, 6395, "qhull internal error (qh_delvertex): vertex v%d was deleted but it was not partitioned as a coplanar point\n",
      vertex->id);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (vertex == qh->tracevertex)
    qh->tracevertex= NULL;
  qh_removevertex(qh, vertex);
  qh_setfree(qh, &vertex->neighbors);
  qh_memfree(qh, vertex, (int)sizeof(vertexT));
} /* delvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="facet3vertex">-</a>

  qh_facet3vertex(qh )
    return temporary set of 3-d vertices in qh_ORIENTclock order

  design:
    if simplicial facet
      build set from facet->vertices with facet->toporient
    else
      for each ridge in order
        build set from ridge's vertices
*/
setT *qh_facet3vertex(qhT *qh, facetT *facet) {
  ridgeT *ridge, *firstridge;
  vertexT *vertex;
  int cntvertices, cntprojected=0;
  setT *vertices;

  cntvertices= qh_setsize(qh, facet->vertices);
  vertices= qh_settemp(qh, cntvertices);
  if (facet->simplicial) {
    if (cntvertices != 3) {
      qh_fprintf(qh, qh->ferr, 6147, "qhull internal error (qh_facet3vertex): only %d vertices for simplicial facet f%d\n",
                  cntvertices, facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    qh_setappend(qh, &vertices, SETfirst_(facet->vertices));
    if (facet->toporient ^ qh_ORIENTclock)
      qh_setappend(qh, &vertices, SETsecond_(facet->vertices));
    else
      qh_setaddnth(qh, &vertices, 0, SETsecond_(facet->vertices));
    qh_setappend(qh, &vertices, SETelem_(facet->vertices, 2));
  }else {
    ridge= firstridge= SETfirstt_(facet->ridges, ridgeT);   /* no infinite */
    while ((ridge= qh_nextridge3d(ridge, facet, &vertex))) {
      qh_setappend(qh, &vertices, vertex);
      if (++cntprojected > cntvertices || ridge == firstridge)
        break;
    }
    if (!ridge || cntprojected != cntvertices) {
      qh_fprintf(qh, qh->ferr, 6148, "qhull internal error (qh_facet3vertex): ridges for facet %d don't match up.  got at least %d\n",
                  facet->id, cntprojected);
      qh_errexit(qh, qh_ERRqhull, facet, ridge);
    }
  }
  return vertices;
} /* facet3vertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findbestfacet">-</a>

  qh_findbestfacet(qh, point, bestoutside, bestdist, isoutside )
    find facet that is furthest below a point

    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    if bestoutside is set (e.g., qh_ALL)
      returns best facet that is not upperdelaunay
      if Delaunay and inside, point is outside circumsphere of bestfacet
    else
      returns first facet below point
      if point is inside, returns nearest, !upperdelaunay facet
    distance to facet
    isoutside set if outside of facet

  notes:
    Distance is measured by distance to the facet's hyperplane.  For
    Delaunay facets, this is not the same as the containing facet.  It may
    be an adjacent facet or a different tricoplanar facet.  See 
    <a href="../html/qh-code.htm#findfacet">locate a facet with qh_findbestfacet()</a>

    For tricoplanar facets, this finds one of the tricoplanar facets closest
    to the point.  

    If inside, qh_findbestfacet performs an exhaustive search
       this may be too conservative.  Sometimes it is clearly required.

    qh_findbestfacet is not used by qhull.
    uses qh.visit_id and qh.coplanarset

  see:
    <a href="geom_r.c#findbest">qh_findbest</a>
*/
facetT *qh_findbestfacet(qhT *qh, pointT *point, boolT bestoutside,
           realT *bestdist, boolT *isoutside) {
  facetT *bestfacet= NULL;
  int numpart, totpart= 0;

  bestfacet= qh_findbest(qh, point, qh->facet_list,
                            bestoutside, !qh_ISnewfacets, bestoutside /* qh_NOupper */,
                            bestdist, isoutside, &totpart);
  if (*bestdist < -qh->DISTround) {
    bestfacet= qh_findfacet_all(qh, point, !qh_NOupper, bestdist, isoutside, &numpart);
    totpart += numpart;
    if ((isoutside && *isoutside && bestoutside)
    || (isoutside && !*isoutside && bestfacet->upperdelaunay)) {
      bestfacet= qh_findbest(qh, point, bestfacet,
                            bestoutside, False, bestoutside,
                            bestdist, isoutside, &totpart);
      totpart += numpart;
    }
  }
  trace3((qh, qh->ferr, 3014, "qh_findbestfacet: f%d dist %2.2g isoutside %d totpart %d\n",
          bestfacet->id, *bestdist, (isoutside ? *isoutside : UINT_MAX), totpart));
  return bestfacet;
} /* findbestfacet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findbestlower">-</a>

  qh_findbestlower(qh, facet, point, bestdist, numpart )
    returns best non-upper, non-flipped neighbor of facet for point
    if needed, searches vertex neighbors

  returns:
    returns bestdist and updates numpart

  notes:
    called by qh_findbest() for points above an upperdelaunay facet
    if Delaunay and inside, point is outside of circumsphere of bestfacet

*/
facetT *qh_findbestlower(qhT *qh, facetT *upperfacet, pointT *point, realT *bestdistp, int *numpart) {
  facetT *neighbor, **neighborp, *bestfacet= NULL;
  realT bestdist= -REALmax/2 /* avoid underflow */;
  realT dist;
  vertexT *vertex;
  boolT isoutside= False;  /* not used */

  zinc_(Zbestlower);
  FOREACHneighbor_(upperfacet) {
    if (neighbor->upperdelaunay || neighbor->flipped)
      continue;
    (*numpart)++;
    qh_distplane(qh, point, neighbor, &dist);
    if (dist > bestdist) {
      bestfacet= neighbor;
      bestdist= dist;
    }
  }
  if (!bestfacet) {
    zinc_(Zbestlowerv);
    /* rarely called, numpart does not count nearvertex computations */
    vertex= qh_nearvertex(qh, upperfacet, point, &dist);
    qh_vertexneighbors(qh);
    FOREACHneighbor_(vertex) {
      if (neighbor->upperdelaunay || neighbor->flipped)
        continue;
      (*numpart)++;
      qh_distplane(qh, point, neighbor, &dist);
      if (dist > bestdist) {
        bestfacet= neighbor;
        bestdist= dist;
      }
    }
  }
  if (!bestfacet) {
    zinc_(Zbestlowerall);  /* invoked once per point in outsideset */
    zmax_(Zbestloweralln, qh->num_facets);
    /* [dec'15] Previously reported as QH6228 */
    trace3((qh, qh->ferr, 3025, "qh_findbestlower: all neighbors of facet %d are flipped or upper Delaunay.  Search all facets\n",
        upperfacet->id));
    /* rarely called */
    bestfacet= qh_findfacet_all(qh, point, qh_NOupper, &bestdist, &isoutside, numpart);
  }
  *bestdistp= bestdist;
  trace3((qh, qh->ferr, 3015, "qh_findbestlower: f%d dist %2.2g for f%d p%d\n",
          bestfacet->id, bestdist, upperfacet->id, qh_pointid(qh, point)));
  return bestfacet;
} /* findbestlower */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findfacet_all">-</a>

  qh_findfacet_all(qh, point, noupper, bestdist, isoutside, numpart )
    exhaustive search for facet below a point
    ignore flipped and visible facets, f.normal==NULL, and if noupper, f.upperdelaunay facets

    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    returns first facet below point
    if point is inside,
      returns nearest facet
    distance to facet
    isoutside if point is outside of the hull
    number of distance tests

  notes:
    called by qh_findbestlower if all neighbors are flipped or upper Delaunay (QH3025)
    primarily for library users (qh_findbestfacet), rarely used by Qhull
*/
facetT *qh_findfacet_all(qhT *qh, pointT *point, boolT noupper, realT *bestdist, boolT *isoutside,
                          int *numpart) {
  facetT *bestfacet= NULL, *facet;
  realT dist;
  int totpart= 0;

  *bestdist= -REALmax;
  *isoutside= False;
  FORALLfacets {
    if (facet->flipped || !facet->normal || facet->visible)
      continue;
    if (noupper && facet->upperdelaunay)
      continue;
    totpart++;
    qh_distplane(qh, point, facet, &dist);
    if (dist > *bestdist) {
      *bestdist= dist;
      bestfacet= facet;
      if (dist > qh->MINoutside) {
        *isoutside= True;
        break;
      }
    }
  }
  *numpart= totpart;
  trace3((qh, qh->ferr, 3016, "qh_findfacet_all: p%d, noupper? %d, f%d, dist %2.2g, isoutside %d, totpart %d\n",
      qh_pointid(qh, point), noupper, getid_(bestfacet), *bestdist, *isoutside, totpart));
  return bestfacet;
} /* findfacet_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findgood">-</a>

  qh_findgood(qh, facetlist, goodhorizon )
    identify good facets for qh.PRINTgood and qh_buildcone_onlygood
    goodhorizon is count of good, horizon facets from qh_find_horizon, otherwise 0 from qh_findgood_all
    if not qh.MERGING and qh.GOODvertex>0
      facet includes point as vertex
      if !match, returns goodhorizon
    if qh.GOODpoint
      facet is visible or coplanar (>0) or not visible (<0)
    if qh.GOODthreshold
      facet->normal matches threshold
    if !goodhorizon and !match,
      selects facet with closest angle to thresholds
      sets GOODclosest

  returns:
    number of new, good facets found
    determines facet->good
    may update qh.GOODclosest

  notes:
    called from qh_initbuild, qh_buildcone_onlygood, and qh_findgood_all
    qh_findgood_all (called from qh_prepare_output) further reduces the good region

  design:
    count good facets
    if not merging, clear good facets that fail qh.GOODvertex ('QVn', but not 'QV-n')
    clear good facets that fail qh.GOODpoint ('QGn' or 'QG-n')
    clear good facets that fail qh.GOODthreshold
    if !goodhorizon and !find f.good,
      sets GOODclosest to facet with closest angle to thresholds
*/
int qh_findgood(qhT *qh, facetT *facetlist, int goodhorizon) {
  facetT *facet, *bestfacet= NULL;
  realT angle, bestangle= REALmax, dist;
  int  numgood=0;

  FORALLfacet_(facetlist) {
    if (facet->good)
      numgood++;
  }
  if (qh->GOODvertex>0 && !qh->MERGING) {
    FORALLfacet_(facetlist) {
      if (facet->good && !qh_isvertex(qh->GOODvertexp, facet->vertices)) {
        facet->good= False;
        numgood--;
      }
    }
  }
  if (qh->GOODpoint && numgood) {
    FORALLfacet_(facetlist) {
      if (facet->good && facet->normal) {
        zinc_(Zdistgood);
        qh_distplane(qh, qh->GOODpointp, facet, &dist);
        if ((qh->GOODpoint > 0) ^ (dist > 0.0)) {
          facet->good= False;
          numgood--;
        }
      }
    }
  }
  if (qh->GOODthreshold && (numgood || goodhorizon || qh->GOODclosest)) {
    FORALLfacet_(facetlist) {
      if (facet->good && facet->normal) {
        if (!qh_inthresholds(qh, facet->normal, &angle)) {
          facet->good= False;
          numgood--;
          if (angle < bestangle) {
            bestangle= angle;
            bestfacet= facet;
          }
        }
      }
    }
    if (numgood == 0 && (goodhorizon == 0 || qh->GOODclosest)) {
      if (qh->GOODclosest) {
        if (qh->GOODclosest->visible)
          qh->GOODclosest= NULL;
        else {
          qh_inthresholds(qh, qh->GOODclosest->normal, &angle);
          if (angle < bestangle)
            bestfacet= qh->GOODclosest;
        }
      }
      if (bestfacet && bestfacet != qh->GOODclosest) {   /* numgood == 0 */
        if (qh->GOODclosest)
          qh->GOODclosest->good= False;
        qh->GOODclosest= bestfacet;
        bestfacet->good= True;
        numgood++;
        trace2((qh, qh->ferr, 2044, "qh_findgood: f%d is closest(%2.2g) to thresholds\n",
           bestfacet->id, bestangle));
        return numgood;
      }
    }else if (qh->GOODclosest) { /* numgood > 0 */
      qh->GOODclosest->good= False;
      qh->GOODclosest= NULL;
    }
  }
  zadd_(Zgoodfacet, numgood);
  trace2((qh, qh->ferr, 2045, "qh_findgood: found %d good facets with %d good horizon and qh.GOODclosest f%d\n",
               numgood, goodhorizon, getid_(qh->GOODclosest)));
  if (!numgood && qh->GOODvertex>0 && !qh->MERGING)
    return goodhorizon;
  return numgood;
} /* findgood */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="findgood_all">-</a>

  qh_findgood_all(qh, facetlist )
    apply other constraints for good facets (used by qh.PRINTgood)
    if qh.GOODvertex
      facet includes (>0) or doesn't include (<0) point as vertex
      if last good facet and ONLYgood, prints warning and continues
    if qh.SPLITthresholds (e.g., qh.DELAUNAY)
      facet->normal matches threshold, or if none, the closest one
    calls qh_findgood
    nop if good not used

  returns:
    clears facet->good if not good
    sets qh.num_good

  notes:
    called by qh_prepare_output and qh_printneighborhood
    unless qh.ONLYgood, calls qh_findgood first

  design:
    uses qh_findgood to mark good facets
    clear f.good for failed qh.GOODvertex
    clear f.good for failed qh.SPLITthreholds
       if no more good facets, select best of qh.SPLITthresholds
*/
void qh_findgood_all(qhT *qh, facetT *facetlist) {
  facetT *facet, *bestfacet=NULL;
  realT angle, bestangle= REALmax;
  int  numgood=0, startgood;

  if (!qh->GOODvertex && !qh->GOODthreshold && !qh->GOODpoint
  && !qh->SPLITthresholds)
    return;
  if (!qh->ONLYgood)
    qh_findgood(qh, qh->facet_list, 0);
  FORALLfacet_(facetlist) {
    if (facet->good)
      numgood++;
  }
  if (qh->GOODvertex <0 || (qh->GOODvertex > 0 && qh->MERGING)) {
    FORALLfacet_(facetlist) {
      if (facet->good && ((qh->GOODvertex > 0) ^ !!qh_isvertex(qh->GOODvertexp, facet->vertices))) { /* convert to bool */
        if (!--numgood) {
          if (qh->ONLYgood) {
            qh_fprintf(qh, qh->ferr, 7064, "qhull warning: good vertex p%d does not match last good facet f%d.  Ignored.\n",
               qh_pointid(qh, qh->GOODvertexp), facet->id);
            return;
          }else if (qh->GOODvertex > 0)
            qh_fprintf(qh, qh->ferr, 7065, "qhull warning: point p%d is not a vertex('QV%d').\n",
                qh->GOODvertex-1, qh->GOODvertex-1);
          else
            qh_fprintf(qh, qh->ferr, 7066, "qhull warning: point p%d is a vertex for every facet('QV-%d').\n",
                -qh->GOODvertex - 1, -qh->GOODvertex - 1);
        }
        facet->good= False;
      }
    }
  }
  startgood= numgood;
  if (qh->SPLITthresholds) {
    FORALLfacet_(facetlist) {
      if (facet->good) {
        if (!qh_inthresholds(qh, facet->normal, &angle)) {
          facet->good= False;
          numgood--;
          if (angle < bestangle) {
            bestangle= angle;
            bestfacet= facet;
          }
        }
      }
    }
    if (!numgood && bestfacet) {
      bestfacet->good= True;
      numgood++;
      trace0((qh, qh->ferr, 23, "qh_findgood_all: f%d is closest(%2.2g) to split thresholds\n",
           bestfacet->id, bestangle));
      return;
    }
  }
  if (numgood == 1 && !qh->PRINTgood && qh->GOODclosest && qh->GOODclosest->good) {
    trace2((qh, qh->ferr, 2109, "qh_findgood_all: undo selection of qh.GOODclosest f%d since it would fail qh_inthresholds in qh_skipfacet\n",
      qh->GOODclosest->id));
    qh->GOODclosest->good= False;
    numgood= 0;
  }
  qh->num_good= numgood;
  trace0((qh, qh->ferr, 24, "qh_findgood_all: %d good facets remain out of %d facets\n",
        numgood, startgood));
} /* findgood_all */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="furthestnext">-</a>

  qh_furthestnext()
    set qh.facet_next to facet with furthest of all furthest points
    searches all facets on qh.facet_list

  notes:
    this may help avoid precision problems
*/
void qh_furthestnext(qhT *qh /* qh.facet_list */) {
  facetT *facet, *bestfacet= NULL;
  realT dist, bestdist= -REALmax;

  FORALLfacets {
    if (facet->outsideset) {
#if qh_COMPUTEfurthest
      pointT *furthest;
      furthest= (pointT *)qh_setlast(facet->outsideset);
      zinc_(Zcomputefurthest);
      qh_distplane(qh, furthest, facet, &dist);
#else
      dist= facet->furthestdist;
#endif
      if (dist > bestdist) {
        bestfacet= facet;
        bestdist= dist;
      }
    }
  }
  if (bestfacet) {
    qh_removefacet(qh, bestfacet);
    qh_prependfacet(qh, bestfacet, &qh->facet_next);
    trace1((qh, qh->ferr, 1029, "qh_furthestnext: made f%d next facet(dist %.2g)\n",
            bestfacet->id, bestdist));
  }
} /* furthestnext */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="furthestout">-</a>

  qh_furthestout(qh, facet )
    make furthest outside point the last point of outsideset

  returns:
    updates facet->outsideset
    clears facet->notfurthest
    sets facet->furthestdist

  design:
    determine best point of outsideset
    make it the last point of outsideset
*/
void qh_furthestout(qhT *qh, facetT *facet) {
  pointT *point, **pointp, *bestpoint= NULL;
  realT dist, bestdist= -REALmax;

  FOREACHpoint_(facet->outsideset) {
    qh_distplane(qh, point, facet, &dist);
    zinc_(Zcomputefurthest);
    if (dist > bestdist) {
      bestpoint= point;
      bestdist= dist;
    }
  }
  if (bestpoint) {
    qh_setdel(facet->outsideset, point);
    qh_setappend(qh, &facet->outsideset, point);
#if !qh_COMPUTEfurthest
    facet->furthestdist= bestdist;
#endif
  }
  facet->notfurthest= False;
  trace3((qh, qh->ferr, 3017, "qh_furthestout: p%d is furthest outside point of f%d\n",
          qh_pointid(qh, point), facet->id));
} /* furthestout */


/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="infiniteloop">-</a>

  qh_infiniteloop(qh, facet )
    report infinite loop error due to facet
*/
void qh_infiniteloop(qhT *qh, facetT *facet) {

  qh_fprintf(qh, qh->ferr, 6149, "qhull internal error (qh_infiniteloop): potential infinite loop detected.  If visible, f.replace. If newfacet, f.samecycle\n");
  qh_errexit(qh, qh_ERRqhull, facet, NULL);
} /* qh_infiniteloop */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initbuild">-</a>

  qh_initbuild()
    initialize hull and outside sets with point array
    qh.FIRSTpoint/qh.NUMpoints is point array
    if qh.GOODpoint
      adds qh.GOODpoint to initial hull

  returns:
    qh_facetlist with initial hull
    points partioned into outside sets, coplanar sets, or inside
    initializes qh.GOODpointp, qh.GOODvertexp,

  design:
    initialize global variables used during qh_buildhull
    determine precision constants and points with max/min coordinate values
      if qh.SCALElast, scale last coordinate(for 'd')
    initialize qh.newfacet_list, qh.facet_tail
    initialize qh.vertex_list, qh.newvertex_list, qh.vertex_tail
    determine initial vertices
    build initial simplex
    partition input points into facets of initial simplex
    set up lists
    if qh.ONLYgood
      check consistency
      add qh.GOODvertex if defined
*/
void qh_initbuild(qhT *qh) {
  setT *maxpoints, *vertices;
  facetT *facet;
  int i, numpart;
  realT dist;
  boolT isoutside;

  if (qh->PRINTstatistics) {
    qh_fprintf(qh, qh->ferr, 9350, "qhull %s Statistics: %s | %s\n",
      qh_version, qh->rbox_command, qh->qhull_command);
    fflush(NULL);
  }
  qh->furthest_id= qh_IDunknown;
  qh->lastreport= 0;
  qh->lastfacets= 0;
  qh->lastmerges= 0;
  qh->lastplanes= 0;
  qh->lastdist= 0;
  qh->facet_id= qh->vertex_id= qh->ridge_id= 0;
  qh->visit_id= qh->vertex_visit= 0;
  qh->maxoutdone= False;

  if (qh->GOODpoint > 0)
    qh->GOODpointp= qh_point(qh, qh->GOODpoint-1);
  else if (qh->GOODpoint < 0)
    qh->GOODpointp= qh_point(qh, -qh->GOODpoint-1);
  if (qh->GOODvertex > 0)
    qh->GOODvertexp= qh_point(qh, qh->GOODvertex-1);
  else if (qh->GOODvertex < 0)
    qh->GOODvertexp= qh_point(qh, -qh->GOODvertex-1);
  if ((qh->GOODpoint
       && (qh->GOODpointp < qh->first_point  /* also catches !GOODpointp */
           || qh->GOODpointp > qh_point(qh, qh->num_points-1)))
  || (qh->GOODvertex
       && (qh->GOODvertexp < qh->first_point  /* also catches !GOODvertexp */
           || qh->GOODvertexp > qh_point(qh, qh->num_points-1)))) {
    qh_fprintf(qh, qh->ferr, 6150, "qhull input error: either QGn or QVn point is > p%d\n",
             qh->num_points-1);
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
  }
  maxpoints= qh_maxmin(qh, qh->first_point, qh->num_points, qh->hull_dim);
  if (qh->SCALElast)
    qh_scalelast(qh, qh->first_point, qh->num_points, qh->hull_dim, qh->MINlastcoord, qh->MAXlastcoord, qh->MAXabs_coord);
  qh_detroundoff(qh);
  if (qh->DELAUNAY && qh->upper_threshold[qh->hull_dim-1] > REALmax/2
                  && qh->lower_threshold[qh->hull_dim-1] < -REALmax/2) {
    for (i=qh_PRINTEND; i--; ) {
      if (qh->PRINTout[i] == qh_PRINTgeom && qh->DROPdim < 0
          && !qh->GOODthreshold && !qh->SPLITthresholds)
        break;  /* in this case, don't set upper_threshold */
    }
    if (i < 0) {
      if (qh->UPPERdelaunay) { /* matches qh.upperdelaunay in qh_setfacetplane */
        qh->lower_threshold[qh->hull_dim-1]= qh->ANGLEround * qh_ZEROdelaunay;
        qh->GOODthreshold= True;
      }else {
        qh->upper_threshold[qh->hull_dim-1]= -qh->ANGLEround * qh_ZEROdelaunay;
        if (!qh->GOODthreshold)
          qh->SPLITthresholds= True; /* build upper-convex hull even if Qg */
          /* qh_initqhull_globals errors if Qg without Pdk/etc. */
      }
    }
  }
  trace4((qh, qh->ferr, 4091, "qh_initbuild: create sentinels for qh.facet_tail and qh.vertex_tail\n"));
  qh->facet_list= qh->newfacet_list= qh->facet_tail= qh_newfacet(qh);
  qh->num_facets= qh->num_vertices= qh->num_visible= 0;
  qh->vertex_list= qh->newvertex_list= qh->vertex_tail= qh_newvertex(qh, NULL);
  vertices= qh_initialvertices(qh, qh->hull_dim, maxpoints, qh->first_point, qh->num_points);
  qh_initialhull(qh, vertices);  /* initial qh->facet_list */
  qh_partitionall(qh, vertices, qh->first_point, qh->num_points);
  if (qh->PRINToptions1st || qh->TRACElevel || qh->IStracing) {
    if (qh->TRACElevel || qh->IStracing)
      qh_fprintf(qh, qh->ferr, 8103, "\nTrace level T%d, IStracing %d, point TP%d, merge TM%d, dist TW%2.2g, qh.tracefacet_id %d, traceridge_id %d, tracevertex_id %d, last qh.RERUN %d, %s | %s\n",
         qh->TRACElevel, qh->IStracing, qh->TRACEpoint, qh->TRACEmerge, qh->TRACEdist, qh->tracefacet_id, qh->traceridge_id, qh->tracevertex_id, qh->TRACElastrun, qh->rbox_command, qh->qhull_command);
    qh_fprintf(qh, qh->ferr, 8104, "Options selected for Qhull %s:\n%s\n", qh_version, qh->qhull_options);
  }
  qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
  qh->facet_next= qh->facet_list;
  qh_furthestnext(qh /* qh.facet_list */);
  if (qh->PREmerge) {
    qh->cos_max= qh->premerge_cos;
    qh->centrum_radius= qh->premerge_centrum; /* overwritten by qh_premerge */
  }
  if (qh->ONLYgood) {
    if (qh->GOODvertex > 0 && qh->MERGING) {
      qh_fprintf(qh, qh->ferr, 6151, "qhull input error: 'Qg QVn' (only good vertex) does not work with merging.\nUse 'QJ' to joggle the input or 'Q0' to turn off merging.\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (!(qh->GOODthreshold || qh->GOODpoint
         || (!qh->MERGEexact && !qh->PREmerge && qh->GOODvertexp))) {
      qh_fprintf(qh, qh->ferr, 6152, "qhull input error: 'Qg' (ONLYgood) needs a good threshold('Pd0D0'), a good point(QGn or QG-n), or a good vertex with 'QJ' or 'Q0' (QVn).\n");
      qh_errexit(qh, qh_ERRinput, NULL, NULL);
    }
    if (qh->GOODvertex > 0  && !qh->MERGING  /* matches qh_partitionall */
    && !qh_isvertex(qh->GOODvertexp, vertices)) {
      facet= qh_findbestnew(qh, qh->GOODvertexp, qh->facet_list,
                          &dist, !qh_ALL, &isoutside, &numpart);
      zadd_(Zdistgood, numpart);
      if (!isoutside) {
        qh_fprintf(qh, qh->ferr, 6153, "qhull input error: point for QV%d is inside initial simplex.  It can not be made a vertex.\n",
               qh_pointid(qh, qh->GOODvertexp));
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      if (!qh_addpoint(qh, qh->GOODvertexp, facet, False)) {
        qh_settempfree(qh, &vertices);
        qh_settempfree(qh, &maxpoints);
        return;
      }
    }
    qh_findgood(qh, qh->facet_list, 0);
  }
  qh_settempfree(qh, &vertices);
  qh_settempfree(qh, &maxpoints);
  trace1((qh, qh->ferr, 1030, "qh_initbuild: initial hull created and points partitioned\n"));
} /* initbuild */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initialhull">-</a>

  qh_initialhull(qh, vertices )
    constructs the initial hull as a DIM3 simplex of vertices

  notes:
    only called by qh_initbuild

  design:
    creates a simplex (initializes lists)
    determines orientation of simplex
    sets hyperplanes for facets
    doubles checks orientation (in case of axis-parallel facets with Gaussian elimination)
    checks for flipped facets and qh.NARROWhull
    checks the result
*/
void qh_initialhull(qhT *qh, setT *vertices) {
  facetT *facet, *firstfacet, *neighbor, **neighborp;
  realT angle, minangle= REALmax, dist;

  qh_createsimplex(qh, vertices /* qh.facet_list */);
  qh_resetlists(qh, False, qh_RESETvisible);
  qh->facet_next= qh->facet_list;      /* advance facet when processed */
  qh->interior_point= qh_getcenter(qh, vertices);
  if (qh->IStracing) {
    qh_fprintf(qh, qh->ferr, 8105, "qh_initialhull: ");
    qh_printpoint(qh, qh->ferr, "qh.interior_point", qh->interior_point);
  }
  firstfacet= qh->facet_list;
  qh_setfacetplane(qh, firstfacet);   /* qh_joggle_restart if flipped */
  if (firstfacet->flipped) {
    trace1((qh, qh->ferr, 1065, "qh_initialhull: ignore f%d flipped.  Test qh.interior_point (p-2) for clearly flipped\n", firstfacet->id));
    firstfacet->flipped= False;
  }
  zzinc_(Zdistcheck);
  qh_distplane(qh, qh->interior_point, firstfacet, &dist);
  if (dist > qh->DISTround) {  /* clearly flipped */
    trace1((qh, qh->ferr, 1060, "qh_initialhull: initial orientation incorrect, qh.interior_point is %2.2g from f%d.  Reversing orientation of all facets\n",
          dist, firstfacet->id));
    FORALLfacets
      facet->toporient ^= (unsigned char)True;
    qh_setfacetplane(qh, firstfacet);
  }
  FORALLfacets {
    if (facet != firstfacet)
      qh_setfacetplane(qh, facet);    /* qh_joggle_restart if flipped */
  }
  FORALLfacets {
    if (facet->flipped) {
      trace1((qh, qh->ferr, 1066, "qh_initialhull: ignore f%d flipped.  Test qh.interior_point (p-2) for clearly flipped\n", facet->id));
      facet->flipped= False;
    }
    zzinc_(Zdistcheck);
    qh_distplane(qh, qh->interior_point, facet, &dist);  /* duplicates qh_setfacetplane */
    if (dist > qh->DISTround) {  /* clearly flipped, due to axis-parallel facet or coplanar firstfacet */
      trace1((qh, qh->ferr, 1031, "qh_initialhull: initial orientation incorrect, qh.interior_point is %2.2g from f%d.  Either axis-parallel facet or coplanar firstfacet f%d.  Force outside orientation of all facets\n"));
      FORALLfacets { /* reuse facet, then 'break' */
        facet->flipped= False;
        facet->toporient ^= (unsigned char)True;
        qh_orientoutside(qh, facet);  /* force outside orientation for f.normal */
      }
      break;
    }
  }
  FORALLfacets {
    if (!qh_checkflipped(qh, facet, NULL, qh_ALL)) {
      if (qh->DELAUNAY && ! qh->ATinfinity) {
        qh_joggle_restart(qh, "initial Delaunay cocircular or cospherical");
        if (qh->UPPERdelaunay)
          qh_fprintf(qh, qh->ferr, 6240, "Qhull precision error: initial Delaunay input sites are cocircular or cospherical.  Option 'Qs' searches all points.  Use option 'QJ' to joggle the input, otherwise cannot compute the upper Delaunay triangulation or upper Voronoi diagram of cocircular/cospherical points.\n");
        else
          qh_fprintf(qh, qh->ferr, 6239, "Qhull precision error: initial Delaunay input sites are cocircular or cospherical.  Use option 'Qz' for the Delaunay triangulation or Voronoi diagram of cocircular/cospherical points; it adds a point \"at infinity\".  Alternatively use option 'QJ' to joggle the input.  Use option 'Qs' to search all points for the initial simplex.\n");
        qh_printvertexlist(qh, qh->ferr, "\ninput sites with last coordinate projected to a paraboloid\n", qh->facet_list, NULL, qh_ALL);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }else {
        qh_joggle_restart(qh, "initial simplex is flat");
        qh_fprintf(qh, qh->ferr, 6154, "Qhull precision error: Initial simplex is flat (facet %d is coplanar with the interior point)\n",
                   facet->id);
        qh_errexit(qh, qh_ERRsingular, NULL, NULL);  /* calls qh_printhelp_singular */
      }
    }
    FOREACHneighbor_(facet) {
      angle= qh_getangle(qh, facet->normal, neighbor->normal);
      minimize_( minangle, angle);
    }
  }
  if (minangle < qh_MAXnarrow && !qh->NOnarrow) {
    realT diff= 1.0 + minangle;

    qh->NARROWhull= True;
    qh_option(qh, "_narrow-hull", NULL, &diff);
    if (minangle < qh_WARNnarrow && !qh->RERUN && qh->PRINTprecision)
      qh_printhelp_narrowhull(qh, qh->ferr, minangle);
  }
  zzval_(Zprocessed)= qh->hull_dim+1;
  qh_checkpolygon(qh, qh->facet_list);
  qh_checkconvex(qh, qh->facet_list, qh_DATAfault);
  if (qh->IStracing >= 1) {
    qh_fprintf(qh, qh->ferr, 8105, "qh_initialhull: simplex constructed\n");
  }
} /* initialhull */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="initialvertices">-</a>

  qh_initialvertices(qh, dim, maxpoints, points, numpoints )
    determines a non-singular set of initial vertices
    maxpoints may include duplicate points

  returns:
    temporary set of dim+1 vertices in descending order by vertex id
    if qh.RANDOMoutside && !qh.ALLpoints
      picks random points
    if dim >= qh_INITIALmax,
      uses min/max x and max points with non-zero determinants

  notes:
    unless qh.ALLpoints,
      uses maxpoints as long as determinate is non-zero
*/
setT *qh_initialvertices(qhT *qh, int dim, setT *maxpoints, pointT *points, int numpoints) {
  pointT *point, **pointp;
  setT *vertices, *simplex, *tested;
  realT randr;
  int idx, point_i, point_n, k;
  boolT nearzero= False;

  vertices= qh_settemp(qh, dim + 1);
  simplex= qh_settemp(qh, dim + 1);
  if (qh->ALLpoints)
    qh_maxsimplex(qh, dim, NULL, points, numpoints, &simplex);
  else if (qh->RANDOMoutside) {
    while (qh_setsize(qh, simplex) != dim+1) {
      randr= qh_RANDOMint;
      randr= randr/(qh_RANDOMmax+1);
      randr= floor(qh->num_points * randr);
      idx= (int)randr;
      while (qh_setin(simplex, qh_point(qh, idx))) {
        idx++; /* in case qh_RANDOMint always returns the same value */
        idx= idx < qh->num_points ? idx : 0;
      }
      qh_setappend(qh, &simplex, qh_point(qh, idx));
    }
  }else if (qh->hull_dim >= qh_INITIALmax) {
    tested= qh_settemp(qh, dim+1);
    qh_setappend(qh, &simplex, SETfirst_(maxpoints));   /* max and min X coord */
    qh_setappend(qh, &simplex, SETsecond_(maxpoints));
    qh_maxsimplex(qh, fmin_(qh_INITIALsearch, dim), maxpoints, points, numpoints, &simplex);
    k= qh_setsize(qh, simplex);
    FOREACHpoint_i_(qh, maxpoints) {
      if (k >= dim)  /* qh_maxsimplex for last point */
        break;
      if (point_i & 0x1) {     /* first try up to dim, max. coord. points */
        if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
          qh_detsimplex(qh, point, simplex, k, &nearzero);
          if (nearzero)
            qh_setappend(qh, &tested, point);
          else {
            qh_setappend(qh, &simplex, point);
            k++;
          }
        }
      }
    }
    FOREACHpoint_i_(qh, maxpoints) {
      if (k >= dim)  /* qh_maxsimplex for last point */
        break;
      if ((point_i & 0x1) == 0) {  /* then test min. coord points */
        if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
          qh_detsimplex(qh, point, simplex, k, &nearzero);
          if (nearzero)
            qh_setappend(qh, &tested, point);
          else {
            qh_setappend(qh, &simplex, point);
            k++;
          }
        }
      }
    }
    /* remove tested points from maxpoints */
    FOREACHpoint_i_(qh, maxpoints) {
      if (qh_setin(simplex, point) || qh_setin(tested, point))
        SETelem_(maxpoints, point_i)= NULL;
    }
    qh_setcompact(qh, maxpoints);
    idx= 0;
    while (k < dim && (point= qh_point(qh, idx++))) {
      if (!qh_setin(simplex, point) && !qh_setin(tested, point)){
        qh_detsimplex(qh, point, simplex, k, &nearzero);
        if (!nearzero){
          qh_setappend(qh, &simplex, point);
          k++;
        }
      }
    }
    qh_settempfree(qh, &tested);
    qh_maxsimplex(qh, dim, maxpoints, points, numpoints, &simplex);
  }else /* qh.hull_dim < qh_INITIALmax */
    qh_maxsimplex(qh, dim, maxpoints, points, numpoints, &simplex);
  FOREACHpoint_(simplex)
    qh_setaddnth(qh, &vertices, 0, qh_newvertex(qh, point)); /* descending order */
  qh_settempfree(qh, &simplex);
  return vertices;
} /* initialvertices */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="isvertex">-</a>

  qh_isvertex( point, vertices )
    returns vertex if point is in vertex set, else returns NULL

  notes:
    for qh.GOODvertex
*/
vertexT *qh_isvertex(pointT *point, setT *vertices) {
  vertexT *vertex, **vertexp;

  FOREACHvertex_(vertices) {
    if (vertex->point == point)
      return vertex;
  }
  return NULL;
} /* isvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewfacets">-</a>

  qh_makenewfacets(qh, point )
    make new facets from point and qh.visible_list

  returns:
    apex (point) of the new facets
    qh.newfacet_list= list of new facets with hyperplanes and ->newfacet
    qh.newvertex_list= list of vertices in new facets with ->newfacet set

    if (qh.NEWtentative)
      newfacets reference horizon facets, but not vice versa
      ridges reference non-simplicial horizon ridges, but not vice versa
      does not change existing facets
    else
      sets qh.NEWfacets
      new facets attached to horizon facets and ridges
      for visible facets,
        visible->r.replace is corresponding new facet

  see also:
    qh_makenewplanes() -- make hyperplanes for facets
    qh_attachnewfacets() -- attachnewfacets if not done here qh->NEWtentative
    qh_matchnewfacets() -- match up neighbors
    qh_update_vertexneighbors() -- update vertex neighbors and delvertices
    qh_deletevisible() -- delete visible facets
    qh_checkpolygon() --check the result
    qh_triangulate() -- triangulate a non-simplicial facet

  design:
    for each visible facet
      make new facets to its horizon facets
      update its f.replace
      clear its neighbor set
*/
vertexT *qh_makenewfacets(qhT *qh, pointT *point /* qh.visible_list */) {
  facetT *visible, *newfacet= NULL, *newfacet2= NULL, *neighbor, **neighborp;
  vertexT *apex;
  int numnew=0;

  if (qh->CHECKfrequently) {
    qh_checkdelridge(qh);
  }
  qh->newfacet_list= qh->facet_tail;
  qh->newvertex_list= qh->vertex_tail;
  apex= qh_newvertex(qh, point);
  qh_appendvertex(qh, apex);
  qh->visit_id++;
  FORALLvisible_facets {
    FOREACHneighbor_(visible)
      neighbor->seen= False;
    if (visible->ridges) {
      visible->visitid= qh->visit_id;
      newfacet2= qh_makenew_nonsimplicial(qh, visible, apex, &numnew);
    }
    if (visible->simplicial)
      newfacet= qh_makenew_simplicial(qh, visible, apex, &numnew);
    if (!qh->NEWtentative) {
      if (newfacet2)  /* newfacet is null if all ridges defined */
        newfacet= newfacet2;
      if (newfacet)
        visible->f.replace= newfacet;
      else
        zinc_(Zinsidevisible);
      if (visible->ridges)      /* ridges and neighbors are no longer valid for visible facet */
        SETfirst_(visible->ridges)= NULL;
      SETfirst_(visible->neighbors)= NULL;
    }
  }
  if (!qh->NEWtentative)
    qh->NEWfacets= True;
  trace1((qh, qh->ferr, 1032, "qh_makenewfacets: created %d new facets f%d..f%d from point p%d to horizon\n",
    numnew, qh->first_newfacet, qh->facet_id-1, qh_pointid(qh, point)));
  if (qh->IStracing >= 4)
    qh_printfacetlist(qh, qh->newfacet_list, NULL, qh_ALL);
  return apex;
} /* makenewfacets */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchdupridge">-</a>

  qh_matchdupridge(qh, atfacet, atskip, hashsize, hashcount )
    match duplicate ridges in qh.hash_table for atfacet@atskip
    duplicates marked with ->dupridge and qh_DUPLICATEridge

  returns:
    vertex-facet distance (>0.0) for qh_MERGEridge ridge
    updates hashcount
    set newfacet, facet, matchfacet's hyperplane (removes from mergecycle of coplanarhorizon facets)

  see also:
    qh_matchneighbor

  notes:
    only called by qh_matchnewfacets for qh_buildcone and qh_triangulate_facet
    assumes atfacet is simplicial
    assumes atfacet->neighbors @ atskip == qh_DUPLICATEridge
    usually keeps ridge with the widest merge
    both MRGdupridge and MRGflipped are required merges -- rbox 100 C1,2e-13 D4 t1 | qhull d Qbb
      can merge flipped f11842 skip 3 into f11862 skip 2 and vice versa (forced by goodmatch/goodmatch2)
         blocks -- cannot merge f11862 skip 2 and f11863 skip2 (the widest merge)
         must block -- can merge f11843 skip 3 into f11842 flipped skip 3, but not vice versa
      can merge f11843 skip 3 into f11863 skip 2, but not vice versa
    working/unused.h: [jan'19] Dropped qh_matchdupridge_coplanarhorizon, it was the same or slightly worse.  Complex addition, rarely occurs

  design:
    compute hash value for atfacet and atskip
    repeat twice -- once to make best matches, once to match the rest
      for each possible facet in qh.hash_table
        if it is a matching facet with the same orientation and pass 2
          make match
          unless tricoplanar, mark match for merging (qh_MERGEridge)
          [e.g., tricoplanar RBOX s 1000 t993602376 | QHULL C-1e-3 d Qbb FA Qt]
        if it is a matching facet with the same orientation and pass 1
          test if this is a better match
      if pass 1,
        make best match (it will not be merged)
        set newfacet, facet, matchfacet's hyperplane (removes from mergecycle of coplanarhorizon facets)

*/
coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  boolT same, ismatch, isduplicate= False;
  int hash, scan;
  facetT *facet, *newfacet, *nextfacet;
  facetT *maxmatch= NULL, *maxmatch2= NULL, *goodmatch= NULL, *goodmatch2= NULL;
  int skip, newskip, nextskip= 0, makematch;
  int maxskip= 0, maxskip2= 0, goodskip= 0, goodskip2= 0;
  coordT maxdist= -REALmax, maxdist2= 0.0, dupdist, dupdist2, low, high, maxgood, gooddist= 0.0;

  maxgood= qh_WIDEdupridge * (qh->ONEmerge + qh->DISTround); 
  hash= qh_gethash(qh, hashsize, atfacet->vertices, qh->hull_dim, 1,
                     SETelem_(atfacet->vertices, atskip));
  trace2((qh, qh->ferr, 2046, "qh_matchdupridge: find dupridge matches for f%d skip %d hash %d hashcount %d\n",
          atfacet->id, atskip, hash, *hashcount));
  for (makematch=0; makematch < 2; makematch++) { /* makematch is false on the first pass and 1 on the second */
    qh->visit_id++;
    for (newfacet=atfacet, newskip=atskip; newfacet; newfacet= nextfacet, newskip= nextskip) {
      zinc_(Zhashlookup);
      nextfacet= NULL; /* exit when ismatch found */
      newfacet->visitid= qh->visit_id;
      for (scan=hash; (facet= SETelemt_(qh->hash_table, scan, facetT));
           scan= (++scan >= hashsize ? 0 : scan)) {
        if (!facet->dupridge || facet->visitid == qh->visit_id)
          continue;
        zinc_(Zhashtests);
        if (qh_matchvertices(qh, 1, newfacet->vertices, newskip, facet->vertices, &skip, &same)) {
          if (SETelem_(newfacet->vertices, newskip) == SETelem_(facet->vertices, skip)) {
            trace3((qh, qh->ferr, 3053, "qh_matchdupridge: duplicate ridge due to duplicate facets (f%d skip %d and f%d skip %d) previously reported as QH7084.  Maximize dupdist to force vertex merge\n",
              newfacet->id, newskip, facet->id, skip));
            isduplicate= True;
          }
          ismatch= (same == (boolT)(newfacet->toporient ^ facet->toporient));
          if (SETelemt_(facet->neighbors, skip, facetT) != qh_DUPLICATEridge) {
            if (!makematch) {  /* occurs if many merges, e.g., rbox 100 W0 C2,1e-13 D6 t1546872462 | qhull C0 Qt Tcv */
              qh_fprintf(qh, qh->ferr, 6155, "qhull topology error (qh_matchdupridge): missing qh_DUPLICATEridge at f%d skip %d for new f%d skip %d hash %d ismatch %d.  Set by qh_matchneighbor\n",
                facet->id, skip, newfacet->id, newskip, hash, ismatch);
              qh_errexit2(qh, qh_ERRtopology, facet, newfacet);
            }
          }else if (!ismatch) {
            nextfacet= facet;
            nextskip= skip;
          }else if (SETelemt_(newfacet->neighbors, newskip, facetT) == qh_DUPLICATEridge) {
            if (makematch) {
              if (newfacet->tricoplanar) {
                SETelem_(facet->neighbors, skip)= newfacet;
                SETelem_(newfacet->neighbors, newskip)= facet;
                *hashcount -= 2; /* removed two unmatched facets */
                trace2((qh, qh->ferr, 2075, "qh_matchdupridge: allow tricoplanar dupridge for new f%d skip %d and f%d skip %d\n",
                    newfacet->id, newskip, facet->id, skip)); 
              }else if (goodmatch && goodmatch2) {
                SETelem_(goodmatch2->neighbors, goodskip2)= qh_MERGEridge;  /* undo selection of goodmatch */
                SETelem_(facet->neighbors, skip)= newfacet;
                SETelem_(newfacet->neighbors, newskip)= facet;
                *hashcount -= 2; /* removed two unmatched facets */
                trace2((qh, qh->ferr, 2105, "qh_matchdupridge: make good forced merge of dupridge f%d skip %d into f%d skip %d, keep new f%d skip %d and f%d skip %d, dist %4.4g\n",
                  goodmatch->id, goodskip, goodmatch2->id, goodskip2, newfacet->id, newskip, facet->id, skip, gooddist)); 
                goodmatch2= NULL;
              }else {
                SETelem_(facet->neighbors, skip)= newfacet;
                SETelem_(newfacet->neighbors, newskip)= qh_MERGEridge;  /* resolved by qh_mark_dupridges */
                *hashcount -= 2; /* removed two unmatched facets */
                trace3((qh, qh->ferr, 3073, "qh_matchdupridge: make forced merge of dupridge for new f%d skip %d and f%d skip %d, maxdist %4.4g in qh_forcedmerges\n",
                  newfacet->id, newskip, facet->id, skip, maxdist2));
              }
            }else { /* !makematch */
              if (!facet->normal)
                qh_setfacetplane(qh, facet); /* qh_mergecycle will ignore 'mergehorizon' facets with normals, too many cases otherwise */
              if (!newfacet->normal) 
                qh_setfacetplane(qh, newfacet);
              dupdist= qh_getdistance(qh, facet, newfacet, &low, &high); /* ignore low/high */
              dupdist2= qh_getdistance(qh, newfacet, facet, &low, &high);
              if (isduplicate) {
                goodmatch= NULL;
                minimize_(dupdist, dupdist2);
                maxdist= dupdist;
                maxdist2= REALmax/2;
                maxmatch= facet;
                maxskip= skip;
                maxmatch2= newfacet;
                maxskip2= newskip;
                break; /* force maxmatch */
              }else if (facet->flipped && !newfacet->flipped && dupdist < maxgood) {
                if (!goodmatch || !goodmatch->flipped || dupdist < gooddist) {
                  goodmatch= facet; 
                  goodskip= skip;
                  goodmatch2= newfacet;
                  goodskip2= newskip;
                  gooddist= dupdist;
                  trace3((qh, qh->ferr, 3070, "qh_matchdupridge: try good dupridge flipped f%d skip %d into new f%d skip %d at dist %2.2g otherdist %2.2g\n",
                    goodmatch->id, goodskip, goodmatch2->id, goodskip2, gooddist, dupdist2));
                }
              }else if (newfacet->flipped && !facet->flipped && dupdist2 < maxgood) {
                if (!goodmatch || !goodmatch->flipped || dupdist2 < gooddist) {
                  goodmatch= newfacet;  
                  goodskip= newskip;
                  goodmatch2= facet;
                  goodskip2= skip;
                  gooddist= dupdist2;
                  trace3((qh, qh->ferr, 3071, "qh_matchdupridge: try good dupridge flipped new f%d skip %d into f%d skip %d at dist %2.2g otherdist %2.2g\n",
                    goodmatch->id, goodskip, goodmatch2->id, goodskip2, gooddist, dupdist));
                }
              }else if (dupdist < maxgood && (!newfacet->flipped || facet->flipped)) { /* disallow not-flipped->flipped */
                if (!goodmatch || (!goodmatch->flipped && dupdist < gooddist)) {
                  goodmatch= facet;
                  goodskip= skip;
                  goodmatch2= newfacet;
                  goodskip2= newskip;
                  gooddist= dupdist;
                  trace3((qh, qh->ferr, 3072, "qh_matchdupridge: try good dupridge f%d skip %d into new f%d skip %d at dist %2.2g otherdist %2.2g\n",
                    goodmatch->id, goodskip, goodmatch2->id, goodskip2, gooddist, dupdist2));
                }
              }else if (dupdist2 < maxgood && (!facet->flipped || newfacet->flipped)) { /* disallow not-flipped->flipped */
                if (!goodmatch || (!goodmatch->flipped && dupdist2 < gooddist)) {
                  goodmatch= newfacet;  
                  goodskip= newskip;
                  goodmatch2= facet;
                  goodskip2= skip;
                  gooddist= dupdist2;
                  trace3((qh, qh->ferr, 3018, "qh_matchdupridge: try good dupridge new f%d skip %d into f%d skip %d at dist %2.2g otherdist %2.2g\n",
                    goodmatch->id, goodskip, goodmatch2->id, goodskip2, gooddist, dupdist));
                }
              }else if (!goodmatch) { /* otherwise match the furthest apart facets */
                if (!newfacet->flipped || facet->flipped) {
                  minimize_(dupdist, dupdist2);
                }
                if (dupdist > maxdist) { /* could keep !flipped->flipped, but probably lost anyway */
                  maxdist2= maxdist;
                  maxdist= dupdist;
                  maxmatch= facet;
                  maxskip= skip;
                  maxmatch2= newfacet;
                  maxskip2= newskip;
                  trace3((qh, qh->ferr, 3055, "qh_matchdupridge: try furthest dupridge f%d skip %d new f%d skip %d at dist %2.2g\n",
                    maxmatch->id, maxskip, maxmatch2->id, maxskip2, maxdist));
                }else if (dupdist > maxdist2)
                  maxdist2= dupdist;
              }
            }
          }
        }
      } /* end of foreach entry in qh.hash_table starting at 'hash' */
      if (makematch && SETelemt_(newfacet->neighbors, newskip, facetT) == qh_DUPLICATEridge) {
        qh_fprintf(qh, qh->ferr, 6156, "qhull internal error (qh_matchdupridge): no MERGEridge match for dupridge new f%d skip %d at hash %d..%d\n",
                    newfacet->id, newskip, hash, scan);
        qh_errexit(qh, qh_ERRqhull, newfacet, NULL);
      }
    } /* end of foreach newfacet at 'hash' */
    if (!makematch) {
      if (!maxmatch && !goodmatch) {
        qh_fprintf(qh, qh->ferr, 6157, "qhull internal error (qh_matchdupridge): no maximum or good match for dupridge new f%d skip %d at hash %d..%d\n",
          atfacet->id, atskip, hash, scan);
        qh_errexit(qh, qh_ERRqhull, atfacet, NULL);
      }
      if (goodmatch) {
        SETelem_(goodmatch->neighbors, goodskip)= goodmatch2;
        SETelem_(goodmatch2->neighbors, goodskip2)= goodmatch;
        *hashcount -= 2; /* removed two unmatched facets */
        if (goodmatch->flipped) {
          if (!goodmatch2->flipped) {
            zzinc_(Zflipridge);
          }else {
            zzinc_(Zflipridge2);
            /* qh_joggle_restart called by qh_matchneighbor if qh_DUPLICATEridge */
          }
        }
        /* previously traced */
      }else {
        SETelem_(maxmatch->neighbors, maxskip)= maxmatch2; /* maxmatch!=NULL by QH6157 */
        SETelem_(maxmatch2->neighbors, maxskip2)= maxmatch;
        *hashcount -= 2; /* removed two unmatched facets */
        zzinc_(Zmultiridge);
        /* qh_joggle_restart called by qh_matchneighbor if qh_DUPLICATEridge */
        trace0((qh, qh->ferr, 25, "qh_matchdupridge: keep dupridge f%d skip %d and f%d skip %d, dist %4.4g\n",
          maxmatch2->id, maxskip2, maxmatch->id, maxskip, maxdist));
      }
    }
  }
  if (goodmatch)
    return gooddist;
  return maxdist2;
} /* matchdupridge */

#else /* qh_NOmerge */
coordT qh_matchdupridge(qhT *qh, facetT *atfacet, int atskip, int hashsize, int *hashcount) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(atfacet)
  QHULL_UNUSED(atskip)
  QHULL_UNUSED(hashsize)
  QHULL_UNUSED(hashcount)

  return 0.0;
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nearcoplanar">-</a>

  qh_nearcoplanar()
    for all facets, remove near-inside points from facet->coplanarset</li>
    coplanar points defined by innerplane from qh_outerinner()

  returns:
    if qh->KEEPcoplanar && !qh->KEEPinside
      facet->coplanarset only contains coplanar points
    if qh.JOGGLEmax
      drops inner plane by another qh.JOGGLEmax diagonal since a
        vertex could shift out while a coplanar point shifts in

  notes:
    used for qh.PREmerge and qh.JOGGLEmax
    must agree with computation of qh.NEARcoplanar in qh_detroundoff

  design:
    if not keeping coplanar or inside points
      free all coplanar sets
    else if not keeping both coplanar and inside points
      remove !coplanar or !inside points from coplanar sets
*/
void qh_nearcoplanar(qhT *qh /* qh.facet_list */) {
  facetT *facet;
  pointT *point, **pointp;
  int numpart;
  realT dist, innerplane;

  if (!qh->KEEPcoplanar && !qh->KEEPinside) {
    FORALLfacets {
      if (facet->coplanarset)
        qh_setfree(qh, &facet->coplanarset);
    }
  }else if (!qh->KEEPcoplanar || !qh->KEEPinside) {
    qh_outerinner(qh, NULL, NULL, &innerplane);
    if (qh->JOGGLEmax < REALmax/2)
      innerplane -= qh->JOGGLEmax * sqrt((realT)qh->hull_dim);
    numpart= 0;
    FORALLfacets {
      if (facet->coplanarset) {
        FOREACHpoint_(facet->coplanarset) {
          numpart++;
          qh_distplane(qh, point, facet, &dist);
          if (dist < innerplane) {
            if (!qh->KEEPinside)
              SETref_(point)= NULL;
          }else if (!qh->KEEPcoplanar)
            SETref_(point)= NULL;
        }
        qh_setcompact(qh, facet->coplanarset);
      }
    }
    zzadd_(Zcheckpart, numpart);
  }
} /* nearcoplanar */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nearvertex">-</a>

  qh_nearvertex(qh, facet, point, bestdist )
    return nearest vertex in facet to point

  returns:
    vertex and its distance

  notes:
    if qh.DELAUNAY
      distance is measured in the input set
    searches neighboring tricoplanar facets (requires vertexneighbors)
      Slow implementation.  Recomputes vertex set for each point.
    The vertex set could be stored in the qh.keepcentrum facet.
*/
vertexT *qh_nearvertex(qhT *qh, facetT *facet, pointT *point, realT *bestdistp) {
  realT bestdist= REALmax, dist;
  vertexT *bestvertex= NULL, *vertex, **vertexp, *apex;
  coordT *center;
  facetT *neighbor, **neighborp;
  setT *vertices;
  int dim= qh->hull_dim;

  if (qh->DELAUNAY)
    dim--;
  if (facet->tricoplanar) {
    if (!qh->VERTEXneighbors || !facet->center) {
      qh_fprintf(qh, qh->ferr, 6158, "qhull internal error (qh_nearvertex): qh.VERTEXneighbors and facet->center required for tricoplanar facets\n");
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    vertices= qh_settemp(qh, qh->TEMPsize);
    apex= SETfirstt_(facet->vertices, vertexT);
    center= facet->center;
    FOREACHneighbor_(apex) {
      if (neighbor->center == center) {
        FOREACHvertex_(neighbor->vertices)
          qh_setappend(qh, &vertices, vertex);
      }
    }
  }else
    vertices= facet->vertices;
  FOREACHvertex_(vertices) {
    dist= qh_pointdist(vertex->point, point, -dim);
    if (dist < bestdist) {
      bestdist= dist;
      bestvertex= vertex;
    }
  }
  if (facet->tricoplanar)
    qh_settempfree(qh, &vertices);
  *bestdistp= sqrt(bestdist);
  if (!bestvertex) {
      qh_fprintf(qh, qh->ferr, 6261, "qhull internal error (qh_nearvertex): did not find bestvertex for f%d p%d\n", facet->id, qh_pointid(qh, point));
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  trace3((qh, qh->ferr, 3019, "qh_nearvertex: v%d dist %2.2g for f%d p%d\n",
        bestvertex->id, *bestdistp, facet->id, qh_pointid(qh, point))); /* bestvertex!=0 by QH2161 */
  return bestvertex;
} /* nearvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newhashtable">-</a>

  qh_newhashtable(qh, newsize )
    returns size of qh.hash_table of at least newsize slots

  notes:
    assumes qh.hash_table is NULL
    qh_HASHfactor determines the number of extra slots
    size is not divisible by 2, 3, or 5
*/
int qh_newhashtable(qhT *qh, int newsize) {
  int size;

  size= ((newsize+1)*qh_HASHfactor) | 0x1;  /* odd number */
  while (True) {
    if (newsize<0 || size<0) {
        qh_fprintf(qh, qh->qhmem.ferr, 6236, "qhull error (qh_newhashtable): negative request (%d) or size (%d).  Did int overflow due to high-D?\n", newsize, size); /* WARN64 */
        qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    }
    if ((size%3) && (size%5))
      break;
    size += 2;
    /* loop terminates because there is an infinite number of primes */
  }
  qh->hash_table= qh_setnew(qh, size);
  qh_setzero(qh, qh->hash_table, 0, size);
  return size;
} /* newhashtable */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newvertex">-</a>

  qh_newvertex(qh, point )
    returns a new vertex for point
*/
vertexT *qh_newvertex(qhT *qh, pointT *point) {
  vertexT *vertex;

  zinc_(Ztotvertices);
  vertex= (vertexT *)qh_memalloc(qh, (int)sizeof(vertexT));
  memset((char *) vertex, (size_t)0, sizeof(vertexT));
  if (qh->vertex_id == UINT_MAX) {
    qh_memfree(qh, vertex, (int)sizeof(vertexT));
    qh_fprintf(qh, qh->ferr, 6159, "qhull error: 2^32 or more vertices.  vertexT.id field overflows.  Vertices would not be sorted correctly.\n");
    qh_errexit(qh, qh_ERRother, NULL, NULL);
  }
  if (qh->vertex_id == qh->tracevertex_id)
    qh->tracevertex= vertex;
  vertex->id= qh->vertex_id++;
  vertex->point= point;
  trace4((qh, qh->ferr, 4060, "qh_newvertex: vertex p%d(v%d) created\n", qh_pointid(qh, vertex->point),
          vertex->id));
  return(vertex);
} /* newvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="nextridge3d">-</a>

  qh_nextridge3d( atridge, facet, vertex )
    return next ridge and vertex for a 3d facet
    returns NULL on error
    [for QhullFacet::nextRidge3d] Does not call qh_errexit nor access qhT.

  notes:
    in qh_ORIENTclock order
    this is a O(n^2) implementation to trace all ridges
    be sure to stop on any 2nd visit
    same as QhullRidge::nextRidge3d
    does not use qhT or qh_errexit [QhullFacet.cpp]

  design:
    for each ridge
      exit if it is the ridge after atridge
*/
ridgeT *qh_nextridge3d(ridgeT *atridge, facetT *facet, vertexT **vertexp) {
  vertexT *atvertex, *vertex, *othervertex;
  ridgeT *ridge, **ridgep;

  if ((atridge->top == facet) ^ qh_ORIENTclock)
    atvertex= SETsecondt_(atridge->vertices, vertexT);
  else
    atvertex= SETfirstt_(atridge->vertices, vertexT);
  FOREACHridge_(facet->ridges) {
    if (ridge == atridge)
      continue;
    if ((ridge->top == facet) ^ qh_ORIENTclock) {
      othervertex= SETsecondt_(ridge->vertices, vertexT);
      vertex= SETfirstt_(ridge->vertices, vertexT);
    }else {
      vertex= SETsecondt_(ridge->vertices, vertexT);
      othervertex= SETfirstt_(ridge->vertices, vertexT);
    }
    if (vertex == atvertex) {
      if (vertexp)
        *vertexp= othervertex;
      return ridge;
    }
  }
  return NULL;
} /* nextridge3d */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="opposite_vertex">-</a>

  qh_opposite_vertex(qh, facetA, neighbor )
    return the opposite vertex in facetA to neighbor

*/
vertexT *qh_opposite_vertex(qhT *qh, facetT *facetA,  facetT *neighbor) {
    vertexT *opposite= NULL;
    facetT *facet;
    int facet_i, facet_n;

    if (facetA->simplicial) {
      FOREACHfacet_i_(qh, facetA->neighbors) {
        if (facet == neighbor) {
          opposite= SETelemt_(facetA->vertices, facet_i, vertexT);
          break;
        }
      }
    }
    if (!opposite) {
      qh_fprintf(qh, qh->ferr, 6396, "qhull internal error (qh_opposite_vertex): opposite vertex in facet f%d to neighbor f%d is not defined.  Either is facet is not simplicial or neighbor not found\n",
        facetA->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facetA, neighbor);
    }
    return opposite;
} /* opposite_vertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="outcoplanar">-</a>

  qh_outcoplanar()
    move points from all facets' outsidesets to their coplanarsets

  notes:
    for post-processing under qh.NARROWhull

  design:
    for each facet
      for each outside point for facet
        partition point into coplanar set
*/
void qh_outcoplanar(qhT *qh /* facet_list */) {
  pointT *point, **pointp;
  facetT *facet;
  realT dist;

  trace1((qh, qh->ferr, 1033, "qh_outcoplanar: move outsideset to coplanarset for qh->NARROWhull\n"));
  FORALLfacets {
    FOREACHpoint_(facet->outsideset) {
      qh->num_outside--;
      if (qh->KEEPcoplanar || qh->KEEPnearinside) {
        qh_distplane(qh, point, facet, &dist);
        zinc_(Zpartition);
        qh_partitioncoplanar(qh, point, facet, &dist, qh->findbestnew);
      }
    }
    qh_setfree(qh, &facet->outsideset);
  }
} /* outcoplanar */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="point">-</a>

  qh_point(qh, id )
    return point for a point id, or NULL if unknown

  alternative code:
    return((pointT *)((unsigned long)qh.first_point
           + (unsigned long)((id)*qh.normal_size)));
*/
pointT *qh_point(qhT *qh, int id) {

  if (id < 0)
    return NULL;
  if (id < qh->num_points)
    return qh->first_point + id * qh->hull_dim;
  id -= qh->num_points;
  if (id < qh_setsize(qh, qh->other_points))
    return SETelemt_(qh->other_points, id, pointT);
  return NULL;
} /* point */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="point_add">-</a>

  qh_point_add(qh, set, point, elem )
    stores elem at set[point.id]

  returns:
    access function for qh_pointfacet and qh_pointvertex

  notes:
    checks point.id
*/
void qh_point_add(qhT *qh, setT *set, pointT *point, void *elem) {
  int id, size;

  SETreturnsize_(set, size);
  if ((id= qh_pointid(qh, point)) < 0)
    qh_fprintf(qh, qh->ferr, 7067, "qhull internal warning (point_add): unknown point %p id %d\n",
      point, id);
  else if (id >= size) {
    qh_fprintf(qh, qh->ferr, 6160, "qhull internal error (point_add): point p%d is out of bounds(%d)\n",
             id, size);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }else
    SETelem_(set, id)= elem;
} /* point_add */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointfacet">-</a>

  qh_pointfacet()
    return temporary set of facet for each point
    the set is indexed by point id
    at most one facet per point, arbitrary selection

  notes:
    each point is assigned to at most one of vertices, coplanarset, or outsideset
    unassigned points are interior points or 
    vertices assigned to one of its facets
    coplanarset assigned to the facet
    outside set assigned to the facet
    NULL if no facet for point (inside)
      includes qh.GOODpointp

  access:
    FOREACHfacet_i_(qh, facets) { ... }
    SETelem_(facets, i)

  design:
    for each facet
      add each vertex
      add each coplanar point
      add each outside point
*/
setT *qh_pointfacet(qhT *qh /* qh.facet_list */) {
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);
  setT *facets;
  facetT *facet;
  vertexT *vertex, **vertexp;
  pointT *point, **pointp;

  facets= qh_settemp(qh, numpoints);
  qh_setzero(qh, facets, 0, numpoints);
  qh->vertex_visit++;
  FORALLfacets {
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        vertex->visitid= qh->vertex_visit;
        qh_point_add(qh, facets, vertex->point, facet);
      }
    }
    FOREACHpoint_(facet->coplanarset)
      qh_point_add(qh, facets, point, facet);
    FOREACHpoint_(facet->outsideset)
      qh_point_add(qh, facets, point, facet);
  }
  return facets;
} /* pointfacet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointvertex">-</a>

  qh_pointvertex(qh )
    return temporary set of vertices indexed by point id
    entry is NULL if no vertex for a point
      this will include qh.GOODpointp

  access:
    FOREACHvertex_i_(qh, vertices) { ... }
    SETelem_(vertices, i)
*/
setT *qh_pointvertex(qhT *qh /* qh.facet_list */) {
  int numpoints= qh->num_points + qh_setsize(qh, qh->other_points);
  setT *vertices;
  vertexT *vertex;

  vertices= qh_settemp(qh, numpoints);
  qh_setzero(qh, vertices, 0, numpoints);
  FORALLvertices
    qh_point_add(qh, vertices, vertex->point, vertex);
  return vertices;
} /* pointvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="prependfacet">-</a>

  qh_prependfacet(qh, facet, facetlist )
    prepend facet to the start of a facetlist

  returns:
    increments qh.numfacets
    updates facetlist, qh.facet_list, facet_next

  notes:
    be careful of prepending since it can lose a pointer.
      e.g., can lose _next by deleting and then prepending before _next
*/
void qh_prependfacet(qhT *qh, facetT *facet, facetT **facetlist) {
  facetT *prevfacet, *list;

  trace4((qh, qh->ferr, 4061, "qh_prependfacet: prepend f%d before f%d\n",
          facet->id, getid_(*facetlist)));
  if (!*facetlist)
    (*facetlist)= qh->facet_tail;
  list= *facetlist;
  prevfacet= list->previous;
  facet->previous= prevfacet;
  if (prevfacet)
    prevfacet->next= facet;
  list->previous= facet;
  facet->next= *facetlist;
  if (qh->facet_list == list)  /* this may change *facetlist */
    qh->facet_list= facet;
  if (qh->facet_next == list)
    qh->facet_next= facet;
  *facetlist= facet;
  qh->num_facets++;
} /* prependfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="printhashtable">-</a>

  qh_printhashtable(qh, fp )
    print hash table to fp

  notes:
    not in I/O to avoid bringing io_r.c in

  design:
    for each hash entry
      if defined
        if unmatched or will merge (NULL, qh_MERGEridge, qh_DUPLICATEridge)
          print entry and neighbors
*/
void qh_printhashtable(qhT *qh, FILE *fp) {
  facetT *facet, *neighbor;
  int id, facet_i, facet_n, neighbor_i= 0, neighbor_n= 0;
  vertexT *vertex, **vertexp;

  FOREACHfacet_i_(qh, qh->hash_table) {
    if (facet) {
      FOREACHneighbor_i_(qh, facet) {
        if (!neighbor || neighbor == qh_MERGEridge || neighbor == qh_DUPLICATEridge)
          break;
      }
      if (neighbor_i == neighbor_n)
        continue;
      qh_fprintf(qh, fp, 9283, "hash %d f%d ", facet_i, facet->id);
      FOREACHvertex_(facet->vertices)
        qh_fprintf(qh, fp, 9284, "v%d ", vertex->id);
      qh_fprintf(qh, fp, 9285, "\n neighbors:");
      FOREACHneighbor_i_(qh, facet) {
        if (neighbor == qh_MERGEridge)
          id= -3;
        else if (neighbor == qh_DUPLICATEridge)
          id= -2;
        else
          id= getid_(neighbor);
        qh_fprintf(qh, fp, 9286, " %d", id);
      }
      qh_fprintf(qh, fp, 9287, "\n");
    }
  }
} /* printhashtable */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="printlists">-</a>

  qh_printlists(qh)
    print out facet and vertex lists for debugging (without 'f/v' tags)

  notes:
    not in I/O to avoid bringing io_r.c in
*/
void qh_printlists(qhT *qh) {
  facetT *facet;
  vertexT *vertex;
  int count= 0;

  qh_fprintf(qh, qh->ferr, 3062, "qh_printlists: max_outside %2.2g all facets:", qh->max_outside);
  FORALLfacets{
    if (++count % 100 == 0)
      qh_fprintf(qh, qh->ferr, 8109, "\n     ");
    qh_fprintf(qh, qh->ferr, 8110, " %d", facet->id);
  }
    qh_fprintf(qh, qh->ferr, 8111, "\n  qh.visible_list f%d, newfacet_list f%d, facet_next f%d for qh_addpoint\n  qh.newvertex_list v%d all vertices:",
      getid_(qh->visible_list), getid_(qh->newfacet_list), getid_(qh->facet_next), getid_(qh->newvertex_list));
  count= 0;
  FORALLvertices{
    if (++count % 100 == 0)
      qh_fprintf(qh, qh->ferr, 8112, "\n     ");
    qh_fprintf(qh, qh->ferr, 8113, " %d", vertex->id);
  }
  qh_fprintf(qh, qh->ferr, 8114, "\n");
} /* printlists */

/*-<a                             href="qh-poly.htm#TOC"
  >-------------------------------</a><a name="addfacetvertex">-</a>

  qh_replacefacetvertex(qh, facet, oldvertex, newvertex )
    replace oldvertex with newvertex in f.vertices
    vertices are inverse sorted by vertex->id

  returns:
    toporient is flipped if an odd parity, position change

  notes:
    for simplicial facets in qh_rename_adjacentvertex
    see qh_addfacetvertex
*/
void qh_replacefacetvertex(qhT *qh, facetT *facet, vertexT *oldvertex, vertexT *newvertex) {
  vertexT *vertex;
  facetT *neighbor;
  int vertex_i, vertex_n= 0;
  int old_i= -1, new_i= -1;

  trace3((qh, qh->ferr, 3038, "qh_replacefacetvertex: replace v%d with v%d in f%d\n", oldvertex->id, newvertex->id, facet->id));
  if (!facet->simplicial) {
    qh_fprintf(qh, qh->ferr, 6283, "qhull internal error (qh_replacefacetvertex): f%d is not simplicial\n", facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  FOREACHvertex_i_(qh, facet->vertices) {
    if (new_i == -1 && vertex->id < newvertex->id) {
      new_i= vertex_i;
    }else if (vertex->id == newvertex->id) {
      qh_fprintf(qh, qh->ferr, 6281, "qhull internal error (qh_replacefacetvertex): f%d already contains new v%d\n", facet->id, newvertex->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    if (vertex->id == oldvertex->id) {
      old_i= vertex_i;
    }
  }
  if (old_i == -1) {
    qh_fprintf(qh, qh->ferr, 6282, "qhull internal error (qh_replacefacetvertex): f%d does not contain old v%d\n", facet->id, oldvertex->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  if (new_i == -1) {
    new_i= vertex_n;
  }
  if (old_i < new_i)
    new_i--;
  if ((old_i & 0x1) != (new_i & 0x1))
    facet->toporient ^= 1;
  qh_setdelnthsorted(qh, facet->vertices, old_i);
  qh_setaddnth(qh, &facet->vertices, new_i, newvertex);
  neighbor= SETelemt_(facet->neighbors, old_i, facetT);
  qh_setdelnthsorted(qh, facet->neighbors, old_i);
  qh_setaddnth(qh, &facet->neighbors, new_i, neighbor);
} /* replacefacetvertex */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="resetlists">-</a>

  qh_resetlists(qh, stats, qh_RESETvisible )
    reset newvertex_list, newfacet_list, visible_list, NEWfacets, NEWtentative
    if stats,
      maintains statistics
    if resetVisible, 
      visible_list is restored to facet_list
      otherwise, f.visible/f.replace is retained

  returns:
    newvertex_list, newfacet_list, visible_list are NULL

  notes:
    To delete visible facets, call qh_deletevisible before qh_resetlists
*/
void qh_resetlists(qhT *qh, boolT stats, boolT resetVisible /* qh.newvertex_list newfacet_list visible_list */) {
  vertexT *vertex;
  facetT *newfacet, *visible;
  int totnew=0, totver=0;

  trace2((qh, qh->ferr, 2066, "qh_resetlists: reset newvertex_list v%d, newfacet_list f%d, visible_list f%d, facet_list f%d next f%d vertex_list v%d -- NEWfacets? %d, NEWtentative? %d, stats? %d\n",
    getid_(qh->newvertex_list), getid_(qh->newfacet_list), getid_(qh->visible_list), getid_(qh->facet_list), getid_(qh->facet_next), getid_(qh->vertex_list), qh->NEWfacets, qh->NEWtentative, stats));
  if (stats) {
    FORALLvertex_(qh->newvertex_list)
      totver++;
    FORALLnew_facets
      totnew++;
    zadd_(Zvisvertextot, totver);
    zmax_(Zvisvertexmax, totver);
    zadd_(Znewfacettot, totnew);
    zmax_(Znewfacetmax, totnew);
  }
  FORALLvertex_(qh->newvertex_list)
    vertex->newfacet= False;
  qh->newvertex_list= NULL;
  qh->first_newfacet= 0;
  FORALLnew_facets {
    newfacet->newfacet= False;
    newfacet->dupridge= False;
  }
  qh->newfacet_list= NULL;
  if (resetVisible) {
    FORALLvisible_facets {
      visible->f.replace= NULL;
      visible->visible= False;
    }
    qh->num_visible= 0;
  }
  qh->visible_list= NULL; 
  qh->NEWfacets= False;
  qh->NEWtentative= False;
} /* resetlists */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="setvoronoi_all">-</a>

  qh_setvoronoi_all(qh)
    compute Voronoi centers for all facets
    includes upperDelaunay facets if qh.UPPERdelaunay ('Qu')

  returns:
    facet->center is the Voronoi center

  notes:
    unused/untested code: please email bradb@shore.net if this works ok for you

  use:
    FORALLvertices {...} to locate the vertex for a point.
    FOREACHneighbor_(vertex) {...} to visit the Voronoi centers for a Voronoi cell.
*/
void qh_setvoronoi_all(qhT *qh) {
  facetT *facet;

  qh_clearcenters(qh, qh_ASvoronoi);
  qh_vertexneighbors(qh);

  FORALLfacets {
    if (!facet->normal || !facet->upperdelaunay || qh->UPPERdelaunay) {
      if (!facet->center)
        facet->center= qh_facetcenter(qh, facet->vertices);
    }
  }
} /* setvoronoi_all */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate">-</a>

  qh_triangulate()
    triangulate non-simplicial facets on qh.facet_list,
    if qh->VORONOI, sets Voronoi centers of non-simplicial facets
    nop if hasTriangulation

  returns:
    all facets simplicial
    each tricoplanar facet has ->f.triowner == owner of ->center,normal,etc.
    resets qh.newfacet_list and visible_list

  notes:
    called by qh_prepare_output and user_eg2_r.c
    call after qh_check_output since may switch to Voronoi centers, and qh_checkconvex skips f.tricoplanar facets
    Output may overwrite ->f.triowner with ->f.area
    while running, 'triangulated_facet_list' is a list of
       one non-simplicial facet followed by its 'f.tricoplanar' triangulated facets
    See qh_buildcone
*/
void qh_triangulate(qhT *qh /* qh.facet_list */) {
  facetT *facet, *nextfacet, *owner;
  facetT *neighbor, *visible= NULL, *facet1, *facet2, *triangulated_facet_list= NULL;
  facetT *orig_neighbor= NULL, *otherfacet;
  vertexT *triangulated_vertex_list= NULL;
  mergeT *merge;
  mergeType mergetype;
  int neighbor_i, neighbor_n;
  boolT onlygood= qh->ONLYgood;

  if (qh->hasTriangulation)
      return;
  trace1((qh, qh->ferr, 1034, "qh_triangulate: triangulate non-simplicial facets\n"));
  if (qh->hull_dim == 2)
    return;
  if (qh->VORONOI) {  /* otherwise lose Voronoi centers [could rebuild vertex set from tricoplanar] */
    qh_clearcenters(qh, qh_ASvoronoi);
    qh_vertexneighbors(qh);
  }
  qh->ONLYgood= False; /* for makenew_nonsimplicial */
  qh->visit_id++;
  qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);
  qh->newvertex_list= qh->vertex_tail;
  for (facet=qh->facet_list; facet && facet->next; facet= nextfacet) { /* non-simplicial facets moved to end */
    nextfacet= facet->next;
    if (facet->visible || facet->simplicial)
      continue;
    /* triangulate all non-simplicial facets, otherwise merging does not work, e.g., RBOX c P-0.1 P+0.1 P+0.1 D3 | QHULL d Qt Tv */
    if (!triangulated_facet_list)
      triangulated_facet_list= facet;  /* will be first triangulated facet */
    qh_triangulate_facet(qh, facet, &triangulated_vertex_list); /* qh_resetlists ! */
  }
  /* qh_checkpolygon invalid due to f.visible without qh.visible_list */
  trace2((qh, qh->ferr, 2047, "qh_triangulate: delete null facets from facetlist f%d.  A null facet has the same first (apex) and second vertices\n", getid_(triangulated_facet_list)));
  for (facet=triangulated_facet_list; facet && facet->next; facet= nextfacet) {
    nextfacet= facet->next;
    if (facet->visible)
      continue;
    if (facet->ridges) {
      if (qh_setsize(qh, facet->ridges) > 0) {
        qh_fprintf(qh, qh->ferr, 6161, "qhull internal error (qh_triangulate): ridges still defined for f%d\n", facet->id);
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
      qh_setfree(qh, &facet->ridges);
    }
    if (SETfirst_(facet->vertices) == SETsecond_(facet->vertices)) {
      zinc_(Ztrinull);
      qh_triangulate_null(qh, facet); /* will delete facet */
    }
  }
  trace2((qh, qh->ferr, 2048, "qh_triangulate: delete %d or more mirrored facets.  Mirrored facets have the same vertices due to a null facet\n", qh_setsize(qh, qh->degen_mergeset)));
  qh->visible_list= qh->facet_tail;
  while ((merge= (mergeT *)qh_setdellast(qh->degen_mergeset))) {
    facet1= merge->facet1;
    facet2= merge->facet2;
    mergetype= merge->mergetype;
    qh_memfree(qh, merge, (int)sizeof(mergeT));
    if (mergetype == MRGmirror) {
      zinc_(Ztrimirror);
      qh_triangulate_mirror(qh, facet1, facet2);  /* will delete both facets */
    }
  }
  qh_freemergesets(qh);
  trace2((qh, qh->ferr, 2049, "qh_triangulate: update neighbor lists for vertices from v%d\n", getid_(triangulated_vertex_list)));
  qh->newvertex_list= triangulated_vertex_list;  /* all vertices of triangulated facets */
  qh->visible_list= NULL;
  qh_update_vertexneighbors(qh /* qh.newvertex_list, empty newfacet_list and visible_list */);
  qh_resetlists(qh, False, !qh_RESETvisible /* qh.newvertex_list, empty newfacet_list and visible_list */);

  trace2((qh, qh->ferr, 2050, "qh_triangulate: identify degenerate tricoplanar facets from f%d\n", getid_(triangulated_facet_list)));
  trace2((qh, qh->ferr, 2051, "qh_triangulate: and replace facet->f.triowner with tricoplanar facets that own center, normal, etc.\n"));
  FORALLfacet_(triangulated_facet_list) {
    if (facet->tricoplanar && !facet->visible) {
      FOREACHneighbor_i_(qh, facet) {
        if (neighbor_i == 0) {  /* first iteration */
          if (neighbor->tricoplanar)
            orig_neighbor= neighbor->f.triowner;
          else
            orig_neighbor= neighbor;
        }else {
          if (neighbor->tricoplanar)
            otherfacet= neighbor->f.triowner;
          else
            otherfacet= neighbor;
          if (orig_neighbor == otherfacet) {
            zinc_(Ztridegen);
            facet->degenerate= True;
            break;
          }
        }
      }
    }
  }
  if (qh->IStracing >= 4)
    qh_printlists(qh);
  trace2((qh, qh->ferr, 2052, "qh_triangulate: delete visible facets -- non-simplicial, null, and mirrored facets\n"));
  owner= NULL;
  visible= NULL;
  for (facet=triangulated_facet_list; facet && facet->next; facet= nextfacet) { 
    /* deleting facets, triangulated_facet_list is no longer valid */
    nextfacet= facet->next;
    if (facet->visible) {
      if (facet->tricoplanar) { /* a null or mirrored facet */
        qh_delfacet(qh, facet);
        qh->num_visible--;
      }else {  /* a non-simplicial facet followed by its tricoplanars */
        if (visible && !owner) {
          /*  RBOX 200 s D5 t1001471447 | QHULL Qt C-0.01 Qx Qc Tv Qt -- f4483 had 6 vertices/neighbors and 8 ridges */
          trace2((qh, qh->ferr, 2053, "qh_triangulate: delete f%d.  All tricoplanar facets degenerate for non-simplicial facet\n",
                       visible->id));
          qh_delfacet(qh, visible);
          qh->num_visible--;
        }
        visible= facet;
        owner= NULL;
      }
    }else if (facet->tricoplanar) {
      if (facet->f.triowner != visible || visible==NULL) {
        qh_fprintf(qh, qh->ferr, 6162, "qhull internal error (qh_triangulate): tricoplanar facet f%d not owned by its visible, non-simplicial facet f%d\n", facet->id, getid_(visible));
        qh_errexit2(qh, qh_ERRqhull, facet, visible);
      }
      if (owner)
        facet->f.triowner= owner;
      else if (!facet->degenerate) {
        owner= facet;
        nextfacet= visible->next; /* rescan tricoplanar facets with owner, visible!=0 by QH6162 */
        facet->keepcentrum= True;  /* one facet owns ->normal, etc. */
        facet->coplanarset= visible->coplanarset;
        facet->outsideset= visible->outsideset;
        visible->coplanarset= NULL;
        visible->outsideset= NULL;
        if (!qh->TRInormals) { /* center and normal copied to tricoplanar facets */
          visible->center= NULL;
          visible->normal= NULL;
        }
        qh_delfacet(qh, visible);
        qh->num_visible--;
      }
    }
    facet->degenerate= False; /* reset f.degenerate set by qh_triangulate*/
  }
  if (visible && !owner) {
    trace2((qh, qh->ferr, 2054, "qh_triangulate: all tricoplanar facets degenerate for last non-simplicial facet f%d\n",
                 visible->id));
    qh_delfacet(qh, visible);
    qh->num_visible--;
  }
  qh->ONLYgood= onlygood; /* restore value */
  if (qh->CHECKfrequently)
    qh_checkpolygon(qh, qh->facet_list);
  qh->hasTriangulation= True;
} /* triangulate */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_facet">-</a>

  qh_triangulate_facet(qh, facetA, &firstVertex )
    triangulate a non-simplicial facet
      if qh.CENTERtype=qh_ASvoronoi, sets its Voronoi center
  returns:
    qh.newfacet_list == simplicial facets
      facet->tricoplanar set and ->keepcentrum false
      facet->degenerate set if duplicated apex
      facet->f.trivisible set to facetA
      facet->center copied from facetA (created if qh_ASvoronoi)
        qh_eachvoronoi, qh_detvridge, qh_detvridge3 assume centers copied
      facet->normal,offset,maxoutside copied from facetA

  notes:
      only called by qh_triangulate
      qh_makenew_nonsimplicial uses neighbor->seen for the same
      if qh.TRInormals, newfacet->normal will need qh_free
        if qh.TRInormals and qh_AScentrum, newfacet->center will need qh_free
        keepcentrum is also set on Zwidefacet in qh_mergefacet
        freed by qh_clearcenters

  see also:
      qh_addpoint() -- add a point
      qh_makenewfacets() -- construct a cone of facets for a new vertex

  design:
      if qh_ASvoronoi,
         compute Voronoi center (facet->center)
      select first vertex (highest ID to preserve ID ordering of ->vertices)
      triangulate from vertex to ridges
      copy facet->center, normal, offset
      update vertex neighbors
*/
void qh_triangulate_facet(qhT *qh, facetT *facetA, vertexT **first_vertex) {
  facetT *newfacet;
  facetT *neighbor, **neighborp;
  vertexT *apex;
  int numnew=0;

  trace3((qh, qh->ferr, 3020, "qh_triangulate_facet: triangulate facet f%d\n", facetA->id));

  qh->first_newfacet= qh->facet_id;
  if (qh->IStracing >= 4)
    qh_printfacet(qh, qh->ferr, facetA);
  FOREACHneighbor_(facetA) {
    neighbor->seen= False;
    neighbor->coplanarhorizon= False;
  }
  if (qh->CENTERtype == qh_ASvoronoi && !facetA->center  /* matches upperdelaunay in qh_setfacetplane() */
  && fabs_(facetA->normal[qh->hull_dim -1]) >= qh->ANGLEround * qh_ZEROdelaunay) {
    facetA->center= qh_facetcenter(qh, facetA->vertices);
  }
  qh->visible_list= qh->newfacet_list= qh->facet_tail;
  facetA->visitid= qh->visit_id;
  apex= SETfirstt_(facetA->vertices, vertexT);
  qh_makenew_nonsimplicial(qh, facetA, apex, &numnew);
  qh_willdelete(qh, facetA, NULL);
  FORALLnew_facets {
    newfacet->tricoplanar= True;
    newfacet->f.trivisible= facetA;
    newfacet->degenerate= False;
    newfacet->upperdelaunay= facetA->upperdelaunay;
    newfacet->good= facetA->good;
    if (qh->TRInormals) { /* 'Q11' triangulate duplicates ->normal and ->center */
      newfacet->keepcentrum= True;
      if(facetA->normal){
        newfacet->normal= (double *)qh_memalloc(qh, qh->normal_size);
        memcpy((char *)newfacet->normal, facetA->normal, (size_t)qh->normal_size);
      }
      if (qh->CENTERtype == qh_AScentrum)
        newfacet->center= qh_getcentrum(qh, newfacet);
      else if (qh->CENTERtype == qh_ASvoronoi && facetA->center){
        newfacet->center= (double *)qh_memalloc(qh, qh->center_size);
        memcpy((char *)newfacet->center, facetA->center, (size_t)qh->center_size);
      }
    }else {
      newfacet->keepcentrum= False;
      /* one facet will have keepcentrum=True at end of qh_triangulate */
      newfacet->normal= facetA->normal;
      newfacet->center= facetA->center;
    }
    newfacet->offset= facetA->offset;
#if qh_MAXoutside
    newfacet->maxoutside= facetA->maxoutside;
#endif
  }
  qh_matchnewfacets(qh /* qh.newfacet_list */); /* ignore returned value, maxdupdist */ 
  zinc_(Ztricoplanar);
  zadd_(Ztricoplanartot, numnew);
  zmax_(Ztricoplanarmax, numnew);
  if (!(*first_vertex))
    (*first_vertex)= qh->newvertex_list;
  qh->newvertex_list= NULL;
  qh->visible_list= NULL;
  /* only update v.neighbors for qh.newfacet_list.  qh.visible_list and qh.newvertex_list are NULL */
  qh_update_vertexneighbors(qh /* qh.newfacet_list */);
  qh_resetlists(qh, False, !qh_RESETvisible /* qh.newfacet_list */);
} /* triangulate_facet */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_link">-</a>

  qh_triangulate_link(qh, oldfacetA, facetA, oldfacetB, facetB)
    relink facetA to facetB via null oldfacetA or mirrored oldfacetA and oldfacetB
  returns:
    if neighbors are already linked, will merge as MRGmirror (qh.degen_mergeset, 4-d and up)
*/
void qh_triangulate_link(qhT *qh, facetT *oldfacetA, facetT *facetA, facetT *oldfacetB, facetT *facetB) {
  int errmirror= False;

  if (oldfacetA == oldfacetB) {
    trace3((qh, qh->ferr, 3052, "qh_triangulate_link: relink neighbors f%d and f%d of null facet f%d\n",
      facetA->id, facetB->id, oldfacetA->id));
  }else {
    trace3((qh, qh->ferr, 3021, "qh_triangulate_link: relink neighbors f%d and f%d of mirrored facets f%d and f%d\n",
      facetA->id, facetB->id, oldfacetA->id, oldfacetB->id));
  }
  if (qh_setin(facetA->neighbors, facetB)) {
    if (!qh_setin(facetB->neighbors, facetA))
      errmirror= True;
    else if (!facetA->redundant || !facetB->redundant || !qh_hasmerge(qh->degen_mergeset, MRGmirror, facetA, facetB))
      qh_appendmergeset(qh, facetA, facetB, MRGmirror, 0.0, 1.0);
  }else if (qh_setin(facetB->neighbors, facetA))
    errmirror= True;
  if (errmirror) {
    qh_fprintf(qh, qh->ferr, 6163, "qhull internal error (qh_triangulate_link): neighbors f%d and f%d do not match for null facet or mirrored facets f%d and f%d\n",
       facetA->id, facetB->id, oldfacetA->id, oldfacetB->id);
    qh_errexit2(qh, qh_ERRqhull, facetA, facetB);
  }
  qh_setreplace(qh, facetB->neighbors, oldfacetB, facetA);
  qh_setreplace(qh, facetA->neighbors, oldfacetA, facetB);
} /* triangulate_link */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_mirror">-</a>

  qh_triangulate_mirror(qh, facetA, facetB)
    delete two mirrored facets identified by qh_triangulate_null() and itself
      a mirrored facet shares the same vertices of a logical ridge
  design:
    since a null facet duplicates the first two vertices, the opposing neighbors absorb the null facet
    if they are already neighbors, the opposing neighbors become MRGmirror facets
*/
void qh_triangulate_mirror(qhT *qh, facetT *facetA, facetT *facetB) {
  facetT *neighbor, *neighborB;
  int neighbor_i, neighbor_n;

  trace3((qh, qh->ferr, 3022, "qh_triangulate_mirror: delete mirrored facets f%d and f%d and link their neighbors\n",
         facetA->id, facetB->id));
  FOREACHneighbor_i_(qh, facetA) {
    neighborB= SETelemt_(facetB->neighbors, neighbor_i, facetT);
    if (neighbor == facetB && neighborB == facetA)
      continue; /* occurs twice */
    else if (neighbor->redundant && neighborB->redundant) { /* also mirrored facets (D5+) */
      if (qh_hasmerge(qh->degen_mergeset, MRGmirror, neighbor, neighborB))
        continue;
    }
    if (neighbor->visible && neighborB->visible) /* previously deleted as mirrored facets */
      continue;
    qh_triangulate_link(qh, facetA, neighbor, facetB, neighborB);
  }
  qh_willdelete(qh, facetA, NULL);
  qh_willdelete(qh, facetB, NULL);
} /* triangulate_mirror */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="triangulate_null">-</a>

  qh_triangulate_null(qh, facetA)
    remove null facetA from qh_triangulate_facet()
      a null facet has vertex #1 (apex) == vertex #2
  returns:
    adds facetA to ->visible for deletion after qh_update_vertexneighbors
    qh->degen_mergeset contains mirror facets (4-d and up only)
  design:
    since a null facet duplicates the first two vertices, the opposing neighbors absorb the null facet
    if they are already neighbors, the opposing neighbors will be merged (MRGmirror)
*/
void qh_triangulate_null(qhT *qh, facetT *facetA) {
  facetT *neighbor, *otherfacet;

  trace3((qh, qh->ferr, 3023, "qh_triangulate_null: delete null facet f%d\n", facetA->id));
  neighbor= SETfirstt_(facetA->neighbors, facetT);
  otherfacet= SETsecondt_(facetA->neighbors, facetT);
  qh_triangulate_link(qh, facetA, neighbor, facetA, otherfacet);
  qh_willdelete(qh, facetA, NULL);
} /* triangulate_null */

#else /* qh_NOmerge */
void qh_triangulate(qhT *qh) {
  QHULL_UNUSED(qh)
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexintersect">-</a>

  qh_vertexintersect(qh, verticesA, verticesB )
    intersects two vertex sets (inverse id ordered)
    vertexsetA is a temporary set at the top of qh->qhmem.tempstack

  returns:
    replaces vertexsetA with the intersection

  notes:
    only called by qh_neighbor_intersections
    if !qh.QHULLfinished, non-simplicial facets may have f.vertices with extraneous vertices
      cleaned by qh_remove_extravertices in qh_reduce_vertices
    could optimize by overwriting vertexsetA
*/
void qh_vertexintersect(qhT *qh, setT **vertexsetA, setT *vertexsetB) {
  setT *intersection;

  intersection= qh_vertexintersect_new(qh, *vertexsetA, vertexsetB);
  qh_settempfree(qh, vertexsetA);
  *vertexsetA= intersection;
  qh_settemppush(qh, intersection);
} /* vertexintersect */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexintersect_new">-</a>

  qh_vertexintersect_new(qh, verticesA, verticesB )
    intersects two vertex sets (inverse id ordered)

  returns:
    a new set

  notes:
    called by qh_checkfacet, qh_vertexintersect, qh_rename_sharedvertex, qh_findbest_pinchedvertex, qh_neighbor_intersections
    if !qh.QHULLfinished, non-simplicial facets may have f.vertices with extraneous vertices
       cleaned by qh_remove_extravertices in qh_reduce_vertices
*/
setT *qh_vertexintersect_new(qhT *qh, setT *vertexsetA, setT *vertexsetB) {
  setT *intersection= qh_setnew(qh, qh->hull_dim - 1);
  vertexT **vertexA= SETaddr_(vertexsetA, vertexT);
  vertexT **vertexB= SETaddr_(vertexsetB, vertexT);

  while (*vertexA && *vertexB) {
    if (*vertexA  == *vertexB) {
      qh_setappend(qh, &intersection, *vertexA);
      vertexA++; vertexB++;
    }else {
      if ((*vertexA)->id > (*vertexB)->id)
        vertexA++;
      else
        vertexB++;
    }
  }
  return intersection;
} /* vertexintersect_new */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexneighbors">-</a>

  qh_vertexneighbors(qh)
    for each vertex in qh.facet_list,
      determine its neighboring facets

  returns:
    sets qh.VERTEXneighbors
      nop if qh.VERTEXneighbors already set
      qh_addpoint() will maintain them

  notes:
    assumes all vertex->neighbors are NULL

  design:
    for each facet
      for each vertex
        append facet to vertex->neighbors
*/
void qh_vertexneighbors(qhT *qh /* qh.facet_list */) {
  facetT *facet;
  vertexT *vertex, **vertexp;

  if (qh->VERTEXneighbors)
    return;
  trace1((qh, qh->ferr, 1035, "qh_vertexneighbors: determining neighboring facets for each vertex\n"));
  qh->vertex_visit++;
  FORALLfacets {
    if (facet->visible)
      continue;
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        vertex->visitid= qh->vertex_visit;
        vertex->neighbors= qh_setnew(qh, qh->hull_dim);
      }
      qh_setappend(qh, &vertex->neighbors, facet);
    }
  }
  qh->VERTEXneighbors= True;
} /* vertexneighbors */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="vertexsubset">-</a>

  qh_vertexsubset( vertexsetA, vertexsetB )
    returns True if vertexsetA is a subset of vertexsetB
    assumes vertexsets are sorted

  note:
    empty set is a subset of any other set
*/
boolT qh_vertexsubset(setT *vertexsetA, setT *vertexsetB) {
  vertexT **vertexA= (vertexT **) SETaddr_(vertexsetA, vertexT);
  vertexT **vertexB= (vertexT **) SETaddr_(vertexsetB, vertexT);

  while (True) {
    if (!*vertexA)
      return True;
    if (!*vertexB)
      return False;
    if ((*vertexA)->id > (*vertexB)->id)
      return False;
    if (*vertexA  == *vertexB)
      vertexA++;
    vertexB++;
  }
  return False; /* avoid warnings */
} /* vertexsubset */

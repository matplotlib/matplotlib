/*<html><pre>  -<a                             href="qh-qhull_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   libqhull_r.c
   Quickhull algorithm for convex hulls

   qhull() and top-level routines

   see qh-qhull_r.htm, libqhull_r.h, unix_r.c

   see qhull_ra.h for internal functions

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/libqhull_r.c#16 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*============= functions in alphabetic order after qhull() =======*/

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="qhull">-</a>

  qh_qhull(qh)
    compute DIM3 convex hull of qh.num_points starting at qh.first_point
    qh->contains all global options and variables

  returns:
    returns polyhedron
      qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices,

    returns global variables
      qh.hulltime, qh.max_outside, qh.interior_point, qh.max_vertex, qh.min_vertex

    returns precision constants
      qh.ANGLEround, centrum_radius, cos_max, DISTround, MAXabs_coord, ONEmerge

  notes:
    unless needed for output
      qh.max_vertex and qh.min_vertex are max/min due to merges

  see:
    to add individual points to either qh.num_points
      use qh_addpoint()

    if qh.GETarea
      qh_produceoutput() returns qh.totarea and qh.totvol via qh_getarea()

  design:
    record starting time
    initialize hull and partition points
    build convex hull
    unless early termination
      update facet->maxoutside for vertices, coplanar, and near-inside points
    error if temporary sets exist
    record end time
*/

void qh_qhull(qhT *qh) {
  int numoutside;

  qh->hulltime= qh_CPUclock;
  if (qh->RERUN || qh->JOGGLEmax < REALmax/2)
    qh_build_withrestart(qh);
  else {
    qh_initbuild(qh);
    qh_buildhull(qh);
  }
  if (!qh->STOPadd && !qh->STOPcone && !qh->STOPpoint) {
    if (qh->ZEROall_ok && !qh->TESTvneighbors && qh->MERGEexact)
      qh_checkzero(qh, qh_ALL);
    if (qh->ZEROall_ok && !qh->TESTvneighbors && !qh->WAScoplanar) {
      trace2((qh, qh->ferr, 2055, "qh_qhull: all facets are clearly convex and no coplanar points.  Post-merging and check of maxout not needed.\n"));
      qh->DOcheckmax= False;
    }else {
      qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);
      if (qh->MERGEexact || (qh->hull_dim > qh_DIMreduceBuild && qh->PREmerge))
        qh_postmerge(qh, "First post-merge", qh->premerge_centrum, qh->premerge_cos,
             (qh->POSTmerge ? False : qh->TESTvneighbors)); /* calls qh_reducevertices */
      else if (!qh->POSTmerge && qh->TESTvneighbors)
        qh_postmerge(qh, "For testing vertex neighbors", qh->premerge_centrum,
             qh->premerge_cos, True);                       /* calls qh_test_vneighbors */
      if (qh->POSTmerge)
        qh_postmerge(qh, "For post-merging", qh->postmerge_centrum,
             qh->postmerge_cos, qh->TESTvneighbors);
      if (qh->visible_list == qh->facet_list) {            /* qh_postmerge was called */
        qh->findbestnew= True;
        qh_partitionvisible(qh, !qh_ALL, &numoutside /* qh.visible_list */);
        qh->findbestnew= False;
        qh_deletevisible(qh /* qh.visible_list */);        /* stops at first !f.visible */
        qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
      }
      qh_all_vertexmerges(qh, -1, NULL, NULL);
      qh_freemergesets(qh);
    }
    if (qh->TRACEpoint == qh_IDunknown && qh->TRACElevel > qh->IStracing) {
      qh->IStracing= qh->TRACElevel;
      qh_fprintf(qh, qh->ferr, 2112, "qh_qhull: finished qh_buildhull and qh_postmerge, start tracing (TP-1)\n");
    }
    if (qh->DOcheckmax){
      if (qh->REPORTfreq) {
        qh_buildtracing(qh, NULL, NULL);
        qh_fprintf(qh, qh->ferr, 8115, "\nTesting all coplanar points.\n");
      }
      qh_check_maxout(qh);
    }
    if (qh->KEEPnearinside && !qh->maxoutdone)
      qh_nearcoplanar(qh);
  }
  if (qh_setsize(qh, qh->qhmem.tempstack) != 0) {
    qh_fprintf(qh, qh->ferr, 6164, "qhull internal error (qh_qhull): temporary sets not empty(%d) at end of Qhull\n",
             qh_setsize(qh, qh->qhmem.tempstack));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->hulltime= qh_CPUclock - qh->hulltime;
  qh->QHULLfinished= True;
  trace1((qh, qh->ferr, 1036, "Qhull: algorithm completed\n"));
} /* qhull */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="addpoint">-</a>

  qh_addpoint(qh, furthest, facet, checkdist )
    add point (usually furthest point) above facet to hull
    if checkdist,
      check that point is above facet.
      if point is not outside of the hull, uses qh_partitioncoplanar()
      assumes that facet is defined by qh_findbestfacet()
    else if facet specified,
      assumes that point is above facet (major damage if below)
    for Delaunay triangulations,
      Use qh_setdelaunay() to lift point to paraboloid and scale by 'Qbb' if needed
      Do not use options 'Qbk', 'QBk', or 'QbB' since they scale the coordinates.

  returns:
    returns False if user requested an early termination
      qh.visible_list, newfacet_list, delvertex_list, NEWfacets may be defined
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices
    clear qh.maxoutdone (will need to call qh_check_maxout() for facet->maxoutside)
    if unknown point, adds a pointer to qh.other_points
      do not deallocate the point's coordinates

  notes:
    called from qh_initbuild, qh_buildhull, and qh_addpoint
    tail recursive call if merged a pinchedvertex due to a duplicated ridge
      no more than qh.num_vertices calls (QH6296)
    assumes point is near its best facet and not at a local minimum of a lens
      distributions.  Use qh_findbestfacet to avoid this case.
    uses qh.visible_list, qh.newfacet_list, qh.delvertex_list, qh.NEWfacets
    if called from a user application after qh_qhull and 'QJ' (joggle),
      facet merging for precision problems is disabled by default

  design:
    exit if qh.STOPadd vertices 'TAn'
    add point to other_points if needed
    if checkdist
      if point not above facet
        partition coplanar point
        exit
    exit if pre STOPpoint requested
    find horizon and visible facets for point
    build cone of new facets to the horizon
    exit if build cone fails due to qh.ONLYgood
    tail recursive call if build cone fails due to pinched vertices
    exit if STOPcone requested
    merge non-convex new facets
    if merge found, many merges, or 'Qf'
       use qh_findbestnew() instead of qh_findbest()
    partition outside points from visible facets
    delete visible facets
    check polyhedron if requested
    exit if post STOPpoint requested
    reset working lists of facets and vertices
*/
boolT qh_addpoint(qhT *qh, pointT *furthest, facetT *facet, boolT checkdist) {
  realT dist, pbalance;
  facetT *replacefacet, *newfacet;
  vertexT *apex;
  boolT isoutside= False;
  int numpart, numpoints, goodvisible, goodhorizon, apexpointid;

  qh->maxoutdone= False;
  if (qh_pointid(qh, furthest) == qh_IDunknown)
    qh_setappend(qh, &qh->other_points, furthest);
  if (!facet) {
    qh_fprintf(qh, qh->ferr, 6213, "qhull internal error (qh_addpoint): NULL facet.  Need to call qh_findbestfacet first\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh_detmaxoutside(qh);
  if (checkdist) {
    facet= qh_findbest(qh, furthest, facet, !qh_ALL, !qh_ISnewfacets, !qh_NOupper,
                        &dist, &isoutside, &numpart);
    zzadd_(Zpartition, numpart);
    if (!isoutside) {
      zinc_(Znotmax);  /* last point of outsideset is no longer furthest. */
      facet->notfurthest= True;
      qh_partitioncoplanar(qh, furthest, facet, &dist, qh->findbestnew);
      return True;
    }
  }
  qh_buildtracing(qh, furthest, facet);
  if (qh->STOPpoint < 0 && qh->furthest_id == -qh->STOPpoint-1) {
    facet->notfurthest= True;
    return False;
  }
  qh_findhorizon(qh, furthest, facet, &goodvisible, &goodhorizon);
  if (qh->ONLYgood && !qh->GOODclosest && !(goodvisible+goodhorizon)) {
    zinc_(Znotgood);
    facet->notfurthest= True;
    /* last point of outsideset is no longer furthest.  This is ok
        since all points of the outside are likely to be bad */
    qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
    return True;
  }
  apex= qh_buildcone(qh, furthest, facet, goodhorizon, &replacefacet);
  /* qh.newfacet_list, visible_list, newvertex_list */
  if (!apex) {
    if (qh->ONLYgood)
      return True; /* ignore this furthest point, a good new facet was not found */
    if (replacefacet) {
      if (qh->retry_addpoint++ >= qh->num_vertices) {
        qh_fprintf(qh, qh->ferr, 6296, "qhull internal error (qh_addpoint): infinite loop (%d retries) of merging pinched vertices due to dupridge for point p%d, facet f%d, and %d vertices\n",
          qh->retry_addpoint, qh_pointid(qh, furthest), facet->id, qh->num_vertices);
        qh_errexit(qh, qh_ERRqhull, facet, NULL);
      }
      /* retry qh_addpoint after resolving a dupridge via qh_merge_pinchedvertices */
      return qh_addpoint(qh, furthest, replacefacet, True /* checkdisk */);
    }
    qh->retry_addpoint= 0;
    return True; /* ignore this furthest point, resolved a dupridge by making furthest a coplanar point */
  }
  if (qh->retry_addpoint) {
    zinc_(Zretryadd);
    zadd_(Zretryaddtot, qh->retry_addpoint);
    zmax_(Zretryaddmax, qh->retry_addpoint);
    qh->retry_addpoint= 0;
  }
  apexpointid= qh_pointid(qh, apex->point);
  zzinc_(Zprocessed);
  if (qh->STOPcone && qh->furthest_id == qh->STOPcone-1) {
    facet->notfurthest= True;
    return False;  /* visible_list etc. still defined */
  }
  qh->findbestnew= False;
  if (qh->PREmerge || qh->MERGEexact) {
    qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);
    qh_premerge(qh, apexpointid, qh->premerge_centrum, qh->premerge_cos /* qh.newfacet_list */);
    if (qh_USEfindbestnew)
      qh->findbestnew= True;
    else {
      FORALLnew_facets {
        if (!newfacet->simplicial) {
          qh->findbestnew= True;  /* use qh_findbestnew instead of qh_findbest*/
          break;
        }
      }
    }
  }else if (qh->BESToutside)
    qh->findbestnew= True;
  if (qh->IStracing >= 4)
    qh_checkpolygon(qh, qh->visible_list);
  qh_partitionvisible(qh, !qh_ALL, &numpoints /* qh.visible_list */);
  qh->findbestnew= False;
  qh->findbest_notsharp= False;
  zinc_(Zpbalance);
  pbalance= numpoints - (realT) qh->hull_dim /* assumes all points extreme */
                * (qh->num_points - qh->num_vertices)/qh->num_vertices;
  wadd_(Wpbalance, pbalance);
  wadd_(Wpbalance2, pbalance * pbalance);
  qh_deletevisible(qh /* qh.visible_list */);
  zmax_(Zmaxvertex, qh->num_vertices);
  qh->NEWfacets= False;
  if (qh->IStracing >= 4) {
    if (qh->num_facets < 200)
      qh_printlists(qh);
    qh_printfacetlist(qh, qh->newfacet_list, NULL, True);
    qh_checkpolygon(qh, qh->facet_list);
  }else if (qh->CHECKfrequently) {
    if (qh->num_facets < 1000)
      qh_checkpolygon(qh, qh->facet_list);
    else
      qh_checkpolygon(qh, qh->newfacet_list);
  }
  if (qh->STOPpoint > 0 && qh->furthest_id == qh->STOPpoint-1 && qh_setsize(qh, qh->vertex_mergeset) > 0)
    return False;
  qh_resetlists(qh, True, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
  if (qh->facet_mergeset) {
    /* vertex merges occur after facet merges (qh_premerge) and qh_resetlists */
    qh_all_vertexmerges(qh, apexpointid, NULL, NULL);
    qh_freemergesets(qh);
  }
  /* qh_triangulate(qh); to test qh.TRInormals */
  if (qh->STOPpoint > 0 && qh->furthest_id == qh->STOPpoint-1)
    return False;
  trace2((qh, qh->ferr, 2056, "qh_addpoint: added p%d to convex hull with point balance %2.2g\n",
    qh_pointid(qh, furthest), pbalance));
  return True;
} /* addpoint */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="build_withrestart">-</a>

  qh_build_withrestart(qh)
    allow restarts due to qh.JOGGLEmax while calling qh_buildhull()
       qh_errexit always undoes qh_build_withrestart()
    qh.FIRSTpoint/qh.NUMpoints is point array
       it may be moved by qh_joggleinput
*/
void qh_build_withrestart(qhT *qh) {
  int restart;
  vertexT *vertex, **vertexp;

  qh->ALLOWrestart= True;
  while (True) {
    restart= setjmp(qh->restartexit); /* simple statement for CRAY J916 */
    if (restart) {       /* only from qh_joggle_restart() */
      qh->last_errcode= qh_ERRnone;
      zzinc_(Zretry);
      wmax_(Wretrymax, qh->JOGGLEmax);
      /* QH7078 warns about using 'TCn' with 'QJn' */
      qh->STOPcone= qh_IDunknown; /* if break from joggle, prevents normal output */
      FOREACHvertex_(qh->del_vertices) {
        if (vertex->point && !vertex->partitioned)
          vertex->partitioned= True; /* avoid error in qh_freebuild -> qh_delvertex */
      }
    }
    if (!qh->RERUN && qh->JOGGLEmax < REALmax/2) {
      if (qh->build_cnt > qh_JOGGLEmaxretry) {
        qh_fprintf(qh, qh->ferr, 6229, "qhull input error: %d attempts to construct a convex hull with joggled input.  Increase joggle above 'QJ%2.2g' or modify qh_JOGGLE... parameters in user_r.h\n",
           qh->build_cnt, qh->JOGGLEmax);
        qh_errexit(qh, qh_ERRinput, NULL, NULL);
      }
      if (qh->build_cnt && !restart)
        break;
    }else if (qh->build_cnt && qh->build_cnt >= qh->RERUN)
      break;
    qh->STOPcone= 0;
    qh_freebuild(qh, True);  /* first call is a nop */
    qh->build_cnt++;
    if (!qh->qhull_optionsiz)
      qh->qhull_optionsiz= (int)strlen(qh->qhull_options);   /* WARN64 */
    else {
      qh->qhull_options[qh->qhull_optionsiz]= '\0';
      qh->qhull_optionlen= qh_OPTIONline;  /* starts a new line */
    }
    qh_option(qh, "_run", &qh->build_cnt, NULL);
    if (qh->build_cnt == qh->RERUN) {
      qh->IStracing= qh->TRACElastrun;  /* duplicated from qh_initqhull_globals */
      if (qh->TRACEpoint != qh_IDnone || qh->TRACEdist < REALmax/2 || qh->TRACEmerge) {
        qh->TRACElevel= (qh->IStracing? qh->IStracing : 3);
        qh->IStracing= 0;
      }
      qh->qhmem.IStracing= qh->IStracing;
    }
    if (qh->JOGGLEmax < REALmax/2)
      qh_joggleinput(qh);
    qh_initbuild(qh);
    qh_buildhull(qh);
    if (qh->JOGGLEmax < REALmax/2 && !qh->MERGING)
      qh_checkconvex(qh, qh->facet_list, qh_ALGORITHMfault);
  }
  qh->ALLOWrestart= False;
} /* qh_build_withrestart */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildcone">-</a>

  qh_buildcone(qh, furthest, facet, goodhorizon, &replacefacet )
    build cone of new facets from furthest to the horizon
    goodhorizon is count of good, horizon facets from qh_find_horizon

  returns:
    returns apex of cone with qh.newfacet_list and qh.first_newfacet (f.id)
    returns NULL if qh.ONLYgood and no good facets
    returns NULL and retryfacet if merging pinched vertices will resolve a dupridge
      a horizon vertex was nearly adjacent to another vertex
      will retry qh_addpoint
    returns NULL if resolve a dupridge by making furthest a coplanar point
      furthest was nearly adjacent to an existing vertex
    updates qh.degen_mergeset (MRGridge) if resolve a dupridge by merging facets
    updates qh.newfacet_list, visible_list, newvertex_list
    updates qh.facet_list, vertex_list, num_facets, num_vertices

  notes:
    called by qh_addpoint
    see qh_triangulate, it triangulates non-simplicial facets in post-processing

  design:
    make new facets for point to horizon
    compute balance statistics
    make hyperplanes for point
    exit if qh.ONLYgood and not good (qh_buildcone_onlygood)
    match neighboring new facets
    if dupridges
      exit if !qh.IGNOREpinched and dupridge resolved by coplanar furthest
      retry qh_buildcone if !qh.IGNOREpinched and dupridge resolved by qh_buildcone_mergepinched
      otherwise dupridges resolved by merging facets
    update vertex neighbors and delete interior vertices
*/
vertexT *qh_buildcone(qhT *qh, pointT *furthest, facetT *facet, int goodhorizon, facetT **retryfacet) {
  vertexT *apex;
  realT newbalance;
  int numnew;

  *retryfacet= NULL;
  qh->first_newfacet= qh->facet_id;
  qh->NEWtentative= (qh->MERGEpinched || qh->ONLYgood); /* cleared by qh_attachnewfacets or qh_resetlists */
  apex= qh_makenewfacets(qh, furthest /* qh.newfacet_list visible_list, attaches new facets if !qh.NEWtentative */);
  numnew= (int)(qh->facet_id - qh->first_newfacet);
  newbalance= numnew - (realT)(qh->num_facets - qh->num_visible) * qh->hull_dim / qh->num_vertices;
  /* newbalance statistics updated below if the new facets are accepted */
  if (qh->ONLYgood) { /* qh.MERGEpinched is false by QH6362 */
    if (!qh_buildcone_onlygood(qh, apex, goodhorizon /* qh.newfacet_list */)) {
      facet->notfurthest= True;
      return NULL;
    }
  }else if(qh->MERGEpinched) {
#ifndef qh_NOmerge
    if (qh_buildcone_mergepinched(qh, apex, facet, retryfacet /* qh.newfacet_list */))
      return NULL;
#else
    qh_fprintf(qh, qh->ferr, 6375, "qhull option error (qh_buildcone): option 'Q14' (qh.MERGEpinched) is not available due to qh_NOmerge\n");
    qh_errexit(qh, qh_ERRinput, NULL, NULL);
#endif
  }else {
    /* qh_makenewfacets attached new facets to the horizon */
    qh_matchnewfacets(qh); /* ignore returned value.  qh_forcedmerges will merge dupridges if any */
    qh_makenewplanes(qh /* qh.newfacet_list */);
    qh_update_vertexneighbors_cone(qh);
  }
  wadd_(Wnewbalance, newbalance);
  wadd_(Wnewbalance2, newbalance * newbalance);
  trace2((qh, qh->ferr, 2067, "qh_buildcone: created %d newfacets for p%d(v%d) new facet balance %2.2g\n",
    numnew, qh_pointid(qh, furthest), apex->id, newbalance));
  return apex;
} /* buildcone */

#ifndef qh_NOmerge
/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildcone_mergepinched">-</a>

  qh_buildcone_mergepinched(qh, apex, facet, maxdupdist, &retryfacet )
    build cone of new facets from furthest to the horizon
    maxdupdist>0.0 for merging dupridges (qh_matchdupridge)

  returns:
    returns True if merged a pinched vertex and deleted the cone of new facets
      if retryfacet is set
        a dupridge was resolved by qh_merge_pinchedvertices
        retry qh_addpoint
      otherwise
        apex/furthest was partitioned as a coplanar point
        ignore this furthest point
    returns False if no dupridges or if dupridges will be resolved by MRGridge
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices

  notes:
    only called from qh_buildcone with qh.MERGEpinched

  design:
    match neighboring new facets
    if matching detected dupridges with a wide merge (qh_RATIOtrypinched)
      if pinched vertices (i.e., nearly adjacent)
        delete the cone of new facets
        delete the apex and reset the facet lists
        if coplanar, pinched apex
          partition the apex as a coplanar point
        else
           repeatedly merge the nearest pair of pinched vertices and subsequent facet merges
        return True
      otherwise
        MRGridge are better than vertex merge, but may report an error
    attach new facets
    make hyperplanes for point
    update vertex neighbors and delete interior vertices
*/
boolT qh_buildcone_mergepinched(qhT *qh, vertexT *apex, facetT *facet, facetT **retryfacet) {
  facetT *newfacet, *nextfacet;
  pointT *apexpoint;
  coordT maxdupdist;
  int apexpointid;
  boolT iscoplanar;

  *retryfacet= NULL;
  maxdupdist= qh_matchnewfacets(qh);
  if (maxdupdist > qh_RATIOtrypinched * qh->ONEmerge) { /* one or more dupridges with a wide merge */
    if (qh->IStracing >= 4 && qh->num_facets < 1000)
      qh_printlists(qh);
    qh_initmergesets(qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */);
    if (qh_getpinchedmerges(qh, apex, maxdupdist, &iscoplanar /* qh.newfacet_list, qh.vertex_mergeset */)) {
      for (newfacet=qh->newfacet_list; newfacet && newfacet->next; newfacet= nextfacet) {
        nextfacet= newfacet->next;
        qh_delfacet(qh, newfacet);
      }
      apexpoint= apex->point;
      apexpointid= qh_pointid(qh, apexpoint);
      qh_delvertex(qh, apex);
      qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
      if (iscoplanar) {
        zinc_(Zpinchedapex);
        facet->notfurthest= True;
        qh_partitioncoplanar(qh, apexpoint, facet, NULL, qh->findbestnew);
      }else {
        qh_all_vertexmerges(qh, apexpointid, facet, retryfacet);
      }
      qh_freemergesets(qh); /* errors if not empty */
      return True;
    }
    /* MRGridge are better than vertex merge, but may report an error */
    qh_freemergesets(qh);
  }
  qh_attachnewfacets(qh /* qh.visible_list */);
  qh_makenewplanes(qh /* qh.newfacet_list */);
  qh_update_vertexneighbors_cone(qh);
  return False;
} /* buildcone_mergepinched */
#endif /* !qh_NOmerge */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildcone_onlygood">-</a>

  qh_buildcone_onlygood(qh, apex, goodhorizon )
    build cone of good, new facets from apex and its qh.newfacet_list to the horizon
    goodhorizon is count of good, horizon facets from qh_find_horizon

  returns:
    False if a f.good facet or a qh.GOODclosest facet is not found
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices

  notes:
    called from qh_buildcone
    QH11030 FIX: Review effect of qh.GOODclosest on qh_buildcone_onlygood ('Qg').  qh_findgood preserves old value if didn't find a good facet.  See qh_findgood_all for disabling

  design:
    make hyperplanes for point
    if qh_findgood fails to find a f.good facet or a qh.GOODclosest facet
      delete cone of new facets
      return NULL (ignores apex)
    else
      attach cone to horizon
      match neighboring new facets
*/
boolT qh_buildcone_onlygood(qhT *qh, vertexT *apex, int goodhorizon) {
  facetT *newfacet, *nextfacet;

  qh_makenewplanes(qh /* qh.newfacet_list */);
  if(qh_findgood(qh, qh->newfacet_list, goodhorizon) == 0) {
    if (!qh->GOODclosest) {
      for (newfacet=qh->newfacet_list; newfacet && newfacet->next; newfacet= nextfacet) {
        nextfacet= newfacet->next;
        qh_delfacet(qh, newfacet);
      }
      qh_delvertex(qh, apex);
      qh_resetlists(qh, False /*no stats*/, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
      zinc_(Znotgoodnew);
      /* !good outside points dropped from hull */
      return False;
    }
  }
  qh_attachnewfacets(qh /* qh.visible_list */);
  qh_matchnewfacets(qh); /* ignore returned value.  qh_forcedmerges will merge dupridges if any */
  qh_update_vertexneighbors_cone(qh);
  return True;
} /* buildcone_onlygood */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildhull">-</a>

  qh_buildhull(qh)
    construct a convex hull by adding outside points one at a time

  returns:

  notes:
    may be called multiple times
    checks facet and vertex lists for incorrect flags
    to recover from STOPcone, call qh_deletevisible and qh_resetlists

  design:
    check visible facet and newfacet flags
    check newfacet vertex flags and qh.STOPcone/STOPpoint
    for each facet with a furthest outside point
      add point to facet
      exit if qh.STOPcone or qh.STOPpoint requested
    if qh.NARROWhull for initial simplex
      partition remaining outside points to coplanar sets
*/
void qh_buildhull(qhT *qh) {
  facetT *facet;
  pointT *furthest;
  vertexT *vertex;
  int id;

  trace1((qh, qh->ferr, 1037, "qh_buildhull: start build hull\n"));
  FORALLfacets {
    if (facet->visible || facet->newfacet) {
      qh_fprintf(qh, qh->ferr, 6165, "qhull internal error (qh_buildhull): visible or new facet f%d in facet list\n",
                   facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
  }
  FORALLvertices {
    if (vertex->newfacet) {
      qh_fprintf(qh, qh->ferr, 6166, "qhull internal error (qh_buildhull): new vertex f%d in vertex list\n",
                   vertex->id);
      qh_errprint(qh, "ERRONEOUS", NULL, NULL, NULL, vertex);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    id= qh_pointid(qh, vertex->point);
    if ((qh->STOPpoint>0 && id == qh->STOPpoint-1) ||
        (qh->STOPpoint<0 && id == -qh->STOPpoint-1) ||
        (qh->STOPcone>0 && id == qh->STOPcone-1)) {
      trace1((qh, qh->ferr, 1038,"qh_buildhull: stop point or cone P%d in initial hull\n", id));
      return;
    }
  }
  qh->facet_next= qh->facet_list;      /* advance facet when processed */
  while ((furthest= qh_nextfurthest(qh, &facet))) {
    qh->num_outside--;  /* if ONLYmax, furthest may not be outside */
    if (qh->STOPadd>0 && (qh->num_vertices - qh->hull_dim - 1 >= qh->STOPadd - 1)) {
      trace1((qh, qh->ferr, 1059, "qh_buildhull: stop after adding %d vertices\n", qh->STOPadd-1));
      return;
    }
    if (!qh_addpoint(qh, furthest, facet, qh->ONLYmax))
      break;
  }
  if (qh->NARROWhull) /* move points from outsideset to coplanarset */
    qh_outcoplanar(qh /* facet_list */ );
  if (qh->num_outside && !furthest) {
    qh_fprintf(qh, qh->ferr, 6167, "qhull internal error (qh_buildhull): %d outside points were never processed.\n", qh->num_outside);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  trace1((qh, qh->ferr, 1039, "qh_buildhull: completed the hull construction\n"));
} /* buildhull */


/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="buildtracing">-</a>

  qh_buildtracing(qh, furthest, facet )
    trace an iteration of qh_buildhull() for furthest point and facet
    if !furthest, prints progress message

  returns:
    tracks progress with qh.lastreport, lastcpu, lastfacets, lastmerges, lastplanes, lastdist
    updates qh.furthest_id (-3 if furthest is NULL)
    also resets visit_id, vertext_visit on wrap around

  see:
    qh_tracemerging()

  design:
    if !furthest
      print progress message
      exit
    if 'TFn' iteration
      print progress message
    else if tracing
      trace furthest point and facet
    reset qh.visit_id and qh.vertex_visit if overflow may occur
    set qh.furthest_id for tracing
*/
void qh_buildtracing(qhT *qh, pointT *furthest, facetT *facet) {
  realT dist= 0;
  double cpu;
  int total, furthestid;
  time_t timedata;
  struct tm *tp;
  vertexT *vertex;

  qh->old_randomdist= qh->RANDOMdist;
  qh->RANDOMdist= False;
  if (!furthest) {
    time(&timedata);
    tp= localtime(&timedata);
    cpu= (double)qh_CPUclock - (double)qh->hulltime;
    cpu /= (double)qh_SECticks;
    total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);
    qh_fprintf(qh, qh->ferr, 8118, "\n\
At %02d:%02d:%02d & %2.5g CPU secs, qhull has created %d facets and merged %d.\n\
 The current hull contains %d facets and %d vertices.  Last point was p%d\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu, qh->facet_id -1,
      total, qh->num_facets, qh->num_vertices, qh->furthest_id);
    return;
  }
  furthestid= qh_pointid(qh, furthest);
#ifndef qh_NOtrace
  if (qh->TRACEpoint == furthestid) {
    trace1((qh, qh->ferr, 1053, "qh_buildtracing: start trace T%d for point TP%d above facet f%d\n", qh->TRACElevel, furthestid, facet->id));
    qh->IStracing= qh->TRACElevel;
    qh->qhmem.IStracing= qh->TRACElevel;
  }else if (qh->TRACEpoint != qh_IDnone && qh->TRACEdist < REALmax/2) {
    qh->IStracing= 0;
    qh->qhmem.IStracing= 0;
  }
#endif
  if (qh->REPORTfreq && (qh->facet_id-1 > qh->lastreport + (unsigned int)qh->REPORTfreq)) {
    qh->lastreport= qh->facet_id-1;
    time(&timedata);
    tp= localtime(&timedata);
    cpu= (double)qh_CPUclock - (double)qh->hulltime;
    cpu /= (double)qh_SECticks;
    total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);
    zinc_(Zdistio);
    qh_distplane(qh, furthest, facet, &dist);
    qh_fprintf(qh, qh->ferr, 8119, "\n\
At %02d:%02d:%02d & %2.5g CPU secs, qhull has created %d facets and merged %d.\n\
 The current hull contains %d facets and %d vertices.  There are %d\n\
 outside points.  Next is point p%d(v%d), %2.2g above f%d.\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu, qh->facet_id -1,
      total, qh->num_facets, qh->num_vertices, qh->num_outside+1,
      furthestid, qh->vertex_id, dist, getid_(facet));
  }else if (qh->IStracing >=1) {
    cpu= (double)qh_CPUclock - (double)qh->hulltime;
    cpu /= (double)qh_SECticks;
    qh_distplane(qh, furthest, facet, &dist);
    qh_fprintf(qh, qh->ferr, 1049, "qh_addpoint: add p%d(v%d) %2.2g above f%d to hull of %d facets, %d merges, %d outside at %4.4g CPU secs.  Previous p%d(v%d) delta %4.4g CPU, %d facets, %d merges, %d hyperplanes, %d distplanes, %d retries\n",
      furthestid, qh->vertex_id, dist, getid_(facet), qh->num_facets, zzval_(Ztotmerge), qh->num_outside+1, cpu, qh->furthest_id, qh->vertex_id - 1,
      cpu - qh->lastcpu, qh->num_facets - qh->lastfacets,  zzval_(Ztotmerge) - qh->lastmerges, zzval_(Zsetplane) - qh->lastplanes, zzval_(Zdistplane) - qh->lastdist, qh->retry_addpoint);
    qh->lastcpu= cpu;
    qh->lastfacets= qh->num_facets;
    qh->lastmerges= zzval_(Ztotmerge);
    qh->lastplanes= zzval_(Zsetplane);
    qh->lastdist= zzval_(Zdistplane);
  }
  zmax_(Zvisit2max, (int)qh->visit_id/2);
  if (qh->visit_id > (unsigned int) INT_MAX) { /* 31 bits */
    zinc_(Zvisit);
    if (!qh_checklists(qh, qh->facet_list)) {
      qh_fprintf(qh, qh->ferr, 6370, "qhull internal error: qh_checklists failed on reset of qh.visit_id %u\n", qh->visit_id);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    qh->visit_id= 0;
    FORALLfacets
      facet->visitid= 0;
  }
  zmax_(Zvvisit2max, (int)qh->vertex_visit/2);
  if (qh->vertex_visit > (unsigned int) INT_MAX) { /* 31 bits */
    zinc_(Zvvisit);
    if (qh->visit_id && !qh_checklists(qh, qh->facet_list)) {
      qh_fprintf(qh, qh->ferr, 6371, "qhull internal error: qh_checklists failed on reset of qh.vertex_visit %u\n", qh->vertex_visit);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    qh->vertex_visit= 0;
    FORALLvertices
      vertex->visitid= 0;
  }
  qh->furthest_id= furthestid;
  qh->RANDOMdist= qh->old_randomdist;
} /* buildtracing */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="errexit2">-</a>

  qh_errexit2(qh, exitcode, facet, otherfacet )
    return exitcode to system after an error
    report two facets

  returns:
    assumes exitcode non-zero

  see:
    normally use qh_errexit() in user_r.c(reports a facet and a ridge)
*/
void qh_errexit2(qhT *qh, int exitcode, facetT *facet, facetT *otherfacet) {
  qh->tracefacet= NULL;  /* avoid infinite recursion through qh_fprintf */
  qh->traceridge= NULL;
  qh->tracevertex= NULL;
  qh_errprint(qh, "ERRONEOUS", facet, otherfacet, NULL, NULL);
  qh_errexit(qh, exitcode, NULL, NULL);
} /* errexit2 */


/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="findhorizon">-</a>

  qh_findhorizon(qh, point, facet, goodvisible, goodhorizon )
    given a visible facet, find the point's horizon and visible facets
    for all facets, !facet-visible

  returns:
    returns qh.visible_list/num_visible with all visible facets
      marks visible facets with ->visible
    updates count of good visible and good horizon facets
    updates qh.max_outside, qh.max_vertex, facet->maxoutside

  see:
    similar to qh_delpoint()

  design:
    move facet to qh.visible_list at end of qh.facet_list
    for all visible facets
     for each unvisited neighbor of a visible facet
       compute distance of point to neighbor
       if point above neighbor
         move neighbor to end of qh.visible_list
       else if point is coplanar with neighbor
         update qh.max_outside, qh.max_vertex, neighbor->maxoutside
         mark neighbor coplanar (will create a samecycle later)
         update horizon statistics
*/
void qh_findhorizon(qhT *qh, pointT *point, facetT *facet, int *goodvisible, int *goodhorizon) {
  facetT *neighbor, **neighborp, *visible;
  int numhorizon= 0, coplanar= 0;
  realT dist;

  trace1((qh, qh->ferr, 1040, "qh_findhorizon: find horizon for point p%d facet f%d\n",qh_pointid(qh, point),facet->id));
  *goodvisible= *goodhorizon= 0;
  zinc_(Ztotvisible);
  qh_removefacet(qh, facet);  /* visible_list at end of qh->facet_list */
  qh_appendfacet(qh, facet);
  qh->num_visible= 1;
  if (facet->good)
    (*goodvisible)++;
  qh->visible_list= facet;
  facet->visible= True;
  facet->f.replace= NULL;
  if (qh->IStracing >=4)
    qh_errprint(qh, "visible", facet, NULL, NULL, NULL);
  qh->visit_id++;
  FORALLvisible_facets {
    if (visible->tricoplanar && !qh->TRInormals) {
      qh_fprintf(qh, qh->ferr, 6230, "qhull internal error (qh_findhorizon): does not work for tricoplanar facets.  Use option 'Q11'\n");
      qh_errexit(qh, qh_ERRqhull, visible, NULL);
    }
    if (qh_setsize(qh, visible->neighbors) == 0) {
      qh_fprintf(qh, qh->ferr, 6295, "qhull internal error (qh_findhorizon): visible facet f%d does not have neighbors\n", visible->id);
      qh_errexit(qh, qh_ERRqhull, visible, NULL);
    }
    visible->visitid= qh->visit_id;
    FOREACHneighbor_(visible) {
      if (neighbor->visitid == qh->visit_id)
        continue;
      neighbor->visitid= qh->visit_id;
      zzinc_(Znumvisibility);
      qh_distplane(qh, point, neighbor, &dist);
      if (dist > qh->MINvisible) {
        zinc_(Ztotvisible);
        qh_removefacet(qh, neighbor);  /* append to end of qh->visible_list */
        qh_appendfacet(qh, neighbor);
        neighbor->visible= True;
        neighbor->f.replace= NULL;
        qh->num_visible++;
        if (neighbor->good)
          (*goodvisible)++;
        if (qh->IStracing >=4)
          qh_errprint(qh, "visible", neighbor, NULL, NULL, NULL);
      }else {
        if (dist >= -qh->MAXcoplanar) {
          neighbor->coplanarhorizon= True;
          zzinc_(Zcoplanarhorizon);
          qh_joggle_restart(qh, "coplanar horizon");
          coplanar++;
          if (qh->MERGING) {
            if (dist > 0) {
              maximize_(qh->max_outside, dist);
              maximize_(qh->max_vertex, dist);
#if qh_MAXoutside
              maximize_(neighbor->maxoutside, dist);
#endif
            }else
              minimize_(qh->min_vertex, dist);  /* due to merge later */
          }
          trace2((qh, qh->ferr, 2057, "qh_findhorizon: point p%d is coplanar to horizon f%d, dist=%2.7g < qh->MINvisible(%2.7g)\n",
              qh_pointid(qh, point), neighbor->id, dist, qh->MINvisible));
        }else
          neighbor->coplanarhorizon= False;
        zinc_(Ztothorizon);
        numhorizon++;
        if (neighbor->good)
          (*goodhorizon)++;
        if (qh->IStracing >=4)
          qh_errprint(qh, "horizon", neighbor, NULL, NULL, NULL);
      }
    }
  }
  if (!numhorizon) {
    qh_joggle_restart(qh, "empty horizon");
    qh_fprintf(qh, qh->ferr, 6168, "qhull topology error (qh_findhorizon): empty horizon for p%d.  It was above all facets.\n", qh_pointid(qh, point));
    if (qh->num_facets < 100) {
      qh_printfacetlist(qh, qh->facet_list, NULL, True);
    }
    qh_errexit(qh, qh_ERRtopology, NULL, NULL);
  }
  trace1((qh, qh->ferr, 1041, "qh_findhorizon: %d horizon facets(good %d), %d visible(good %d), %d coplanar\n",
       numhorizon, *goodhorizon, qh->num_visible, *goodvisible, coplanar));
  if (qh->IStracing >= 4 && qh->num_facets < 100)
    qh_printlists(qh);
} /* findhorizon */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="joggle_restart">-</a>

  qh_joggle_restart(qh, reason )
    if joggle ('QJn') and not merging, restart on precision and topology errors
*/
void qh_joggle_restart(qhT *qh, const char *reason) {

  if (qh->JOGGLEmax < REALmax/2) {
    if (qh->ALLOWrestart && !qh->PREmerge && !qh->MERGEexact) {
      trace0((qh, qh->ferr, 26, "qh_joggle_restart: qhull restart because of %s\n", reason));
      /* May be called repeatedly if qh->ALLOWrestart */
      longjmp(qh->restartexit, qh_ERRprec);
    }
  }
} /* qh_joggle_restart */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="nextfurthest">-</a>

  qh_nextfurthest(qh, visible )
    returns next furthest point and visible facet for qh_addpoint()
    starts search at qh.facet_next

  returns:
    removes furthest point from outside set
    NULL if none available
    advances qh.facet_next over facets with empty outside sets

  design:
    for each facet from qh.facet_next
      if empty outside set
        advance qh.facet_next
      else if qh.NARROWhull
        determine furthest outside point
        if furthest point is not outside
          advance qh.facet_next(point will be coplanar)
    remove furthest point from outside set
*/
pointT *qh_nextfurthest(qhT *qh, facetT **visible) {
  facetT *facet;
  int size, idx, loopcount= 0;
  realT randr, dist;
  pointT *furthest;

  while ((facet= qh->facet_next) != qh->facet_tail) {
    if (!facet || loopcount++ > qh->num_facets) {
      qh_fprintf(qh, qh->ferr, 6406, "qhull internal error (qh_nextfurthest): null facet or infinite loop detected for qh.facet_next f%d facet_tail f%d\n",
        getid_(facet), getid_(qh->facet_tail));
      qh_printlists(qh);
      qh_errexit2(qh, qh_ERRqhull, facet, qh->facet_tail);
    }
    if (!facet->outsideset) {
      qh->facet_next= facet->next;
      continue;
    }
    SETreturnsize_(facet->outsideset, size);
    if (!size) {
      qh_setfree(qh, &facet->outsideset);
      qh->facet_next= facet->next;
      continue;
    }
    if (qh->NARROWhull) {
      if (facet->notfurthest)
        qh_furthestout(qh, facet);
      furthest= (pointT *)qh_setlast(facet->outsideset);
#if qh_COMPUTEfurthest
      qh_distplane(qh, furthest, facet, &dist);
      zinc_(Zcomputefurthest);
#else
      dist= facet->furthestdist;
#endif
      if (dist < qh->MINoutside) { /* remainder of outside set is coplanar for qh_outcoplanar */
        qh->facet_next= facet->next;
        continue;
      }
    }
    if (!qh->RANDOMoutside && !qh->VIRTUALmemory) {
      if (qh->PICKfurthest) {
        qh_furthestnext(qh /* qh.facet_list */);
        facet= qh->facet_next;
      }
      *visible= facet;
      return ((pointT *)qh_setdellast(facet->outsideset));
    }
    if (qh->RANDOMoutside) {
      int outcoplanar= 0;
      if (qh->NARROWhull) {
        FORALLfacets {
          if (facet == qh->facet_next)
            break;
          if (facet->outsideset)
            outcoplanar += qh_setsize(qh, facet->outsideset);
        }
      }
      randr= qh_RANDOMint;
      randr= randr/(qh_RANDOMmax+1);
      randr= floor((qh->num_outside - outcoplanar) * randr);
      idx= (int)randr;
      FORALLfacet_(qh->facet_next) {
        if (facet->outsideset) {
          SETreturnsize_(facet->outsideset, size);
          if (!size)
            qh_setfree(qh, &facet->outsideset);
          else if (size > idx) {
            *visible= facet;
            return ((pointT *)qh_setdelnth(qh, facet->outsideset, idx));
          }else
            idx -= size;
        }
      }
      qh_fprintf(qh, qh->ferr, 6169, "qhull internal error (qh_nextfurthest): num_outside %d is too low\nby at least %d, or a random real %g >= 1.0\n",
              qh->num_outside, idx+1, randr);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }else { /* VIRTUALmemory */
      facet= qh->facet_tail->previous;
      if (!(furthest= (pointT *)qh_setdellast(facet->outsideset))) {
        if (facet->outsideset)
          qh_setfree(qh, &facet->outsideset);
        qh_removefacet(qh, facet);
        qh_prependfacet(qh, facet, &qh->facet_list);
        continue;
      }
      *visible= facet;
      return furthest;
    }
  }
  return NULL;
} /* nextfurthest */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="partitionall">-</a>

  qh_partitionall(qh, vertices, points, numpoints )
    partitions all points in points/numpoints to the outsidesets of facets
    vertices= vertices in qh.facet_list(!partitioned)

  returns:
    builds facet->outsideset
    does not partition qh.GOODpoint
    if qh.ONLYgood && !qh.MERGING,
      does not partition qh.GOODvertex

  notes:
    faster if qh.facet_list sorted by anticipated size of outside set

  design:
    initialize pointset with all points
    remove vertices from pointset
    remove qh.GOODpointp from pointset (unless it's qh.STOPcone or qh.STOPpoint)
    for all facets
      for all remaining points in pointset
        compute distance from point to facet
        if point is outside facet
          remove point from pointset (by not reappending)
          update bestpoint
          append point or old bestpoint to facet's outside set
      append bestpoint to facet's outside set (furthest)
    for all points remaining in pointset
      partition point into facets' outside sets and coplanar sets
*/
void qh_partitionall(qhT *qh, setT *vertices, pointT *points, int numpoints){
  setT *pointset;
  vertexT *vertex, **vertexp;
  pointT *point, **pointp, *bestpoint;
  int size, point_i, point_n, point_end, remaining, i, id;
  facetT *facet;
  realT bestdist= -REALmax, dist, distoutside;

  trace1((qh, qh->ferr, 1042, "qh_partitionall: partition all points into outside sets\n"));
  pointset= qh_settemp(qh, numpoints);
  qh->num_outside= 0;
  pointp= SETaddr_(pointset, pointT);
  for (i=numpoints, point= points; i--; point += qh->hull_dim)
    *(pointp++)= point;
  qh_settruncate(qh, pointset, numpoints);
  FOREACHvertex_(vertices) {
    if ((id= qh_pointid(qh, vertex->point)) >= 0)
      SETelem_(pointset, id)= NULL;
  }
  id= qh_pointid(qh, qh->GOODpointp);
  if (id >=0 && qh->STOPcone-1 != id && -qh->STOPpoint-1 != id)
    SETelem_(pointset, id)= NULL;
  if (qh->GOODvertexp && qh->ONLYgood && !qh->MERGING) { /* matches qhull()*/
    if ((id= qh_pointid(qh, qh->GOODvertexp)) >= 0)
      SETelem_(pointset, id)= NULL;
  }
  if (!qh->BESToutside) {  /* matches conditional for qh_partitionpoint below */
    distoutside= qh_DISToutside; /* multiple of qh.MINoutside & qh.max_outside, see user_r.h */
    zval_(Ztotpartition)= qh->num_points - qh->hull_dim - 1; /*misses GOOD... */
    remaining= qh->num_facets;
    point_end= numpoints;
    FORALLfacets {
      size= point_end/(remaining--) + 100;
      facet->outsideset= qh_setnew(qh, size);
      bestpoint= NULL;
      point_end= 0;
      FOREACHpoint_i_(qh, pointset) {
        if (point) {
          zzinc_(Zpartitionall);
          qh_distplane(qh, point, facet, &dist);
          if (dist < distoutside)
            SETelem_(pointset, point_end++)= point;
          else {
            qh->num_outside++;
            if (!bestpoint) {
              bestpoint= point;
              bestdist= dist;
            }else if (dist > bestdist) {
              qh_setappend(qh, &facet->outsideset, bestpoint);
              bestpoint= point;
              bestdist= dist;
            }else
              qh_setappend(qh, &facet->outsideset, point);
          }
        }
      }
      if (bestpoint) {
        qh_setappend(qh, &facet->outsideset, bestpoint);
#if !qh_COMPUTEfurthest
        facet->furthestdist= bestdist;
#endif
      }else
        qh_setfree(qh, &facet->outsideset);
      qh_settruncate(qh, pointset, point_end);
    }
  }
  /* if !qh->BESToutside, pointset contains points not assigned to outsideset */
  if (qh->BESToutside || qh->MERGING || qh->KEEPcoplanar || qh->KEEPinside || qh->KEEPnearinside) {
    qh->findbestnew= True;
    FOREACHpoint_i_(qh, pointset) {
      if (point)
        qh_partitionpoint(qh, point, qh->facet_list);
    }
    qh->findbestnew= False;
  }
  zzadd_(Zpartitionall, zzval_(Zpartition));
  zzval_(Zpartition)= 0;
  qh_settempfree(qh, &pointset);
  if (qh->IStracing >= 4)
    qh_printfacetlist(qh, qh->facet_list, NULL, True);
} /* partitionall */


/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="partitioncoplanar">-</a>

  qh_partitioncoplanar(qh, point, facet, dist, allnew )
    partition coplanar point to a facet
    dist is distance from point to facet
    if dist NULL,
      searches for bestfacet and does nothing if inside
    if allnew (qh.findbestnew)
      searches new facets instead of using qh_findbest()

  returns:
    qh.max_ouside updated
    if qh.KEEPcoplanar or qh.KEEPinside
      point assigned to best coplanarset
    qh.repart_facetid==0 (for detecting infinite recursion via qh_partitionpoint)

  notes:
    facet->maxoutside is updated at end by qh_check_maxout

  design:
    if dist undefined
      find best facet for point
      if point sufficiently below facet (depends on qh.NEARinside and qh.KEEPinside)
        exit
    if keeping coplanar/nearinside/inside points
      if point is above furthest coplanar point
        append point to coplanar set (it is the new furthest)
        update qh.max_outside
      else
        append point one before end of coplanar set
    else if point is clearly outside of qh.max_outside and bestfacet->coplanarset
    and bestfacet is more than perpendicular to facet
      repartition the point using qh_findbest() -- it may be put on an outsideset
    else
      update qh.max_outside
*/
void qh_partitioncoplanar(qhT *qh, pointT *point, facetT *facet, realT *dist, boolT allnew) {
  facetT *bestfacet;
  pointT *oldfurthest;
  realT bestdist, angle, nearest, dist2= 0.0;
  int numpart= 0;
  boolT isoutside, oldfindbest, repartition= False;

  trace4((qh, qh->ferr, 4090, "qh_partitioncoplanar: partition coplanar point p%d starting with f%d dist? %2.2g, allnew? %d, gh.repart_facetid f%d\n",
    qh_pointid(qh, point), facet->id, (dist ? *dist : 0.0), allnew, qh->repart_facetid));
  qh->WAScoplanar= True;
  if (!dist) {
    if (allnew)
      bestfacet= qh_findbestnew(qh, point, facet, &bestdist, qh_ALL, &isoutside, &numpart);
    else
      bestfacet= qh_findbest(qh, point, facet, qh_ALL, !qh_ISnewfacets, qh->DELAUNAY,
                          &bestdist, &isoutside, &numpart);
    zinc_(Ztotpartcoplanar);
    zzadd_(Zpartcoplanar, numpart);
    if (!qh->DELAUNAY && !qh->KEEPinside) { /*  for 'd', bestdist skips upperDelaunay facets */
      if (qh->KEEPnearinside) {
        if (bestdist < -qh->NEARinside) {
          zinc_(Zcoplanarinside);
          trace4((qh, qh->ferr, 4062, "qh_partitioncoplanar: point p%d is more than near-inside facet f%d dist %2.2g allnew? %d\n",
                  qh_pointid(qh, point), bestfacet->id, bestdist, allnew));
          qh->repart_facetid= 0;
          return;
        }
      }else if (bestdist < -qh->MAXcoplanar) {
          trace4((qh, qh->ferr, 4063, "qh_partitioncoplanar: point p%d is inside facet f%d dist %2.2g allnew? %d\n",
                  qh_pointid(qh, point), bestfacet->id, bestdist, allnew));
        zinc_(Zcoplanarinside);
        qh->repart_facetid= 0;
        return;
      }
    }
  }else {
    bestfacet= facet;
    bestdist= *dist;
  }
  if(bestfacet->visible){
    qh_fprintf(qh, qh->ferr, 6405, "qhull internal error (qh_partitioncoplanar): cannot partition coplanar p%d of f%d into visible facet f%d\n",
        qh_pointid(qh, point), facet->id, bestfacet->id);
    qh_errexit2(qh, qh_ERRqhull, facet, bestfacet);
  }
  if (bestdist > qh->max_outside) {
    if (!dist && facet != bestfacet) { /* can't be recursive from qh_partitionpoint since facet != bestfacet */
      zinc_(Zpartangle);
      angle= qh_getangle(qh, facet->normal, bestfacet->normal);
      if (angle < 0) {
        nearest= qh_vertex_bestdist(qh, bestfacet->vertices);
        /* typically due to deleted vertex and coplanar facets, e.g.,
        RBOX 1000 s Z1 G1e-13 t1001185205 | QHULL Tv */
        zinc_(Zpartcorner);
        trace2((qh, qh->ferr, 2058, "qh_partitioncoplanar: repartition coplanar point p%d from f%d as an outside point above corner facet f%d dist %2.2g with angle %2.2g\n",
          qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, angle));
        repartition= True;
      }
    }
    if (!repartition) {
      if (bestdist > qh->MAXoutside * qh_RATIOcoplanaroutside) {
        nearest= qh_vertex_bestdist(qh, bestfacet->vertices);
        if (facet->id == bestfacet->id) {
          if (facet->id == qh->repart_facetid) {
            qh_fprintf(qh, qh->ferr, 6404, "Qhull internal error (qh_partitioncoplanar): infinite loop due to recursive call to qh_partitionpoint.  Repartition point p%d from f%d as a outside point dist %2.2g nearest vertices %2.2g\n",
              qh_pointid(qh, point), facet->id, bestdist, nearest);
            qh_errexit(qh, qh_ERRqhull, facet, NULL);
          }
          qh->repart_facetid= facet->id; /* reset after call to qh_partitionpoint */
        }
        if (point == qh->coplanar_apex) {
          /* otherwise may loop indefinitely, the point is well above a facet, yet near a vertex */
          qh_fprintf(qh, qh->ferr, 6425, "Qhull topology error (qh_partitioncoplanar): can not repartition coplanar point p%d from f%d as outside point above f%d.  It previously failed to form a cone of facets, dist %2.2g, nearest vertices %2.2g\n",
            qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, nearest);
          qh_errexit(qh, qh_ERRtopology, facet, NULL);
        }
        if (nearest < 2 * qh->MAXoutside * qh_RATIOcoplanaroutside) {
          zinc_(Zparttwisted);
          qh_fprintf(qh, qh->ferr, 7085, "Qhull precision warning: repartition coplanar point p%d from f%d as an outside point above twisted facet f%d dist %2.2g nearest vertices %2.2g\n",
            qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, nearest);
        }else {
          zinc_(Zparthidden);
          qh_fprintf(qh, qh->ferr, 7086, "Qhull precision warning: repartition coplanar point p%d from f%d as an outside point above hidden facet f%d dist %2.2g nearest vertices %2.2g\n",
            qh_pointid(qh, point), facet->id, bestfacet->id, bestdist, nearest);
        }
        repartition= True;
      }
    }
    if (repartition) {
      oldfindbest= qh->findbestnew;
      qh->findbestnew= False;
      qh_partitionpoint(qh, point, bestfacet);
      qh->findbestnew= oldfindbest;
      qh->repart_facetid= 0;
      return;
    }
    qh->repart_facetid= 0;
    qh->max_outside= bestdist;
    if (bestdist > qh->TRACEdist || qh->IStracing >= 3) {
      qh_fprintf(qh, qh->ferr, 3041, "qh_partitioncoplanar: == p%d from f%d increases qh.max_outside to %2.2g of f%d last p%d\n",
                     qh_pointid(qh, point), facet->id, bestdist, bestfacet->id, qh->furthest_id);
      qh_errprint(qh, "DISTANT", facet, bestfacet, NULL, NULL);
    }
  }
  if (qh->KEEPcoplanar + qh->KEEPinside + qh->KEEPnearinside) {
    oldfurthest= (pointT *)qh_setlast(bestfacet->coplanarset);
    if (oldfurthest) {
      zinc_(Zcomputefurthest);
      qh_distplane(qh, oldfurthest, bestfacet, &dist2);
    }
    if (!oldfurthest || dist2 < bestdist)
      qh_setappend(qh, &bestfacet->coplanarset, point);
    else
      qh_setappend2ndlast(qh, &bestfacet->coplanarset, point);
  }
  trace4((qh, qh->ferr, 4064, "qh_partitioncoplanar: point p%d is coplanar with facet f%d (or inside) dist %2.2g\n",
          qh_pointid(qh, point), bestfacet->id, bestdist));
} /* partitioncoplanar */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="partitionpoint">-</a>

  qh_partitionpoint(qh, point, facet )
    assigns point to an outside set, coplanar set, or inside set (i.e., dropt)
    if qh.findbestnew
      uses qh_findbestnew() to search all new facets
    else
      uses qh_findbest()

  notes:
    after qh_distplane(), this and qh_findbest() are most expensive in 3-d

  design:
    find best facet for point
      (either exhaustive search of new facets or directed search from facet)
    if qh.NARROWhull
      retain coplanar and nearinside points as outside points
    if point is outside bestfacet
      if point above furthest point for bestfacet
        append point to outside set (it becomes the new furthest)
        if outside set was empty
          move bestfacet to end of qh.facet_list (i.e., after qh.facet_next)
        update bestfacet->furthestdist
      else
        append point one before end of outside set
    else if point is coplanar to bestfacet
      if keeping coplanar points or need to update qh.max_outside
        partition coplanar point into bestfacet
    else if near-inside point
      partition as coplanar point into bestfacet
    else is an inside point
      if keeping inside points
        partition as coplanar point into bestfacet
*/
void qh_partitionpoint(qhT *qh, pointT *point, facetT *facet) {
  realT bestdist, previousdist;
  boolT isoutside, isnewoutside= False;
  facetT *bestfacet;
  int numpart;

  if (qh->findbestnew)
    bestfacet= qh_findbestnew(qh, point, facet, &bestdist, qh->BESToutside, &isoutside, &numpart);
  else
    bestfacet= qh_findbest(qh, point, facet, qh->BESToutside, qh_ISnewfacets, !qh_NOupper,
                          &bestdist, &isoutside, &numpart);
  zinc_(Ztotpartition);
  zzadd_(Zpartition, numpart);
  if(bestfacet->visible){
    qh_fprintf(qh, qh->ferr, 6293, "qhull internal error (qh_partitionpoint): cannot partition p%d of f%d into visible facet f%d\n",
      qh_pointid(qh, point), facet->id, bestfacet->id);
    qh_errexit2(qh, qh_ERRqhull, facet, bestfacet);
  }
  if (qh->NARROWhull) {
    if (qh->DELAUNAY && !isoutside && bestdist >= -qh->MAXcoplanar)
      qh_joggle_restart(qh, "nearly incident point (narrow hull)");
    if (qh->KEEPnearinside) {
      if (bestdist >= -qh->NEARinside)
        isoutside= True;
    }else if (bestdist >= -qh->MAXcoplanar)
      isoutside= True;
  }

  if (isoutside) {
    if (!bestfacet->outsideset
    || !qh_setlast(bestfacet->outsideset)) { /* empty outside set */
      qh_setappend(qh, &(bestfacet->outsideset), point);
      if (!qh->NARROWhull || bestdist > qh->MINoutside)
        isnewoutside= True;
#if !qh_COMPUTEfurthest
      bestfacet->furthestdist= bestdist;
#endif
    }else {
#if qh_COMPUTEfurthest
      zinc_(Zcomputefurthest);
      qh_distplane(qh, oldfurthest, bestfacet, &previousdist);
      if (previousdist < bestdist)
        qh_setappend(qh, &(bestfacet->outsideset), point);
      else
        qh_setappend2ndlast(qh, &(bestfacet->outsideset), point);
#else
      previousdist= bestfacet->furthestdist;
      if (previousdist < bestdist) {
        qh_setappend(qh, &(bestfacet->outsideset), point);
        bestfacet->furthestdist= bestdist;
        if (qh->NARROWhull && previousdist < qh->MINoutside && bestdist >= qh->MINoutside)
          isnewoutside= True;
      }else
        qh_setappend2ndlast(qh, &(bestfacet->outsideset), point);
#endif
    }
    if (isnewoutside && qh->facet_next != bestfacet) {
      if (bestfacet->newfacet) {
        if (qh->facet_next->newfacet)
          qh->facet_next= qh->newfacet_list; /* make sure it's after qh.facet_next */
      }else {
        qh_removefacet(qh, bestfacet);  /* make sure it's after qh.facet_next */
        qh_appendfacet(qh, bestfacet);
        if(qh->newfacet_list){
          bestfacet->newfacet= True;
        }
      }
    }
    qh->num_outside++;
    trace4((qh, qh->ferr, 4065, "qh_partitionpoint: point p%d is outside facet f%d newfacet? %d, newoutside? %d (or narrowhull)\n",
          qh_pointid(qh, point), bestfacet->id, bestfacet->newfacet, isnewoutside));
  }else if (qh->DELAUNAY || bestdist >= -qh->MAXcoplanar) { /* for 'd', bestdist skips upperDelaunay facets */
    if (qh->DELAUNAY)
      qh_joggle_restart(qh, "nearly incident point");
    /* allow coplanar points with joggle, may be interior */
    zzinc_(Zcoplanarpart);
    if ((qh->KEEPcoplanar + qh->KEEPnearinside) || bestdist > qh->max_outside)
      qh_partitioncoplanar(qh, point, bestfacet, &bestdist, qh->findbestnew);
    else {
      trace4((qh, qh->ferr, 4066, "qh_partitionpoint: point p%d is coplanar to facet f%d (dropped)\n",
          qh_pointid(qh, point), bestfacet->id));
    }
  }else if (qh->KEEPnearinside && bestdist >= -qh->NEARinside) {
    zinc_(Zpartnear);
    qh_partitioncoplanar(qh, point, bestfacet, &bestdist, qh->findbestnew);
  }else {
    zinc_(Zpartinside);
    trace4((qh, qh->ferr, 4067, "qh_partitionpoint: point p%d is inside all facets, closest to f%d dist %2.2g\n",
          qh_pointid(qh, point), bestfacet->id, bestdist));
    if (qh->KEEPinside)
      qh_partitioncoplanar(qh, point, bestfacet, &bestdist, qh->findbestnew);
  }
} /* partitionpoint */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="partitionvisible">-</a>

  qh_partitionvisible(qh, allpoints, numoutside )
    partitions outside points in visible facets (qh.visible_list) to qh.newfacet_list
    if keeping coplanar/near-inside/inside points
      partitions coplanar points; repartitions if 'allpoints' (not used)
    1st neighbor (if any) of visible facets points to a horizon facet or a new facet

  returns:
    updates outside sets and coplanar sets of qh.newfacet_list
    updates qh.num_outside (count of outside points)
    does not truncate f.outsideset, f.coplanarset, or qh.del_vertices (see qh_deletevisible)

  notes:
    called by qh_qhull, qh_addpoint, and qh_all_vertexmerges
    qh.findbest_notsharp should be clear (extra work if set)

  design:
    for all visible facets with outside set or coplanar set
      select a newfacet for visible facet
      if outside set
        partition outside set into new facets
      if coplanar set and keeping coplanar/near-inside/inside points
        if allpoints
          partition coplanar set into new facets, may be assigned outside
        else
          partition coplanar set into coplanar sets of new facets
    for each deleted vertex
      if allpoints
        partition vertex into new facets, may be assigned outside
      else
        partition vertex into coplanar sets of new facets
*/
void qh_partitionvisible(qhT *qh, boolT allpoints, int *numoutside /* qh.visible_list */) {
  facetT *visible, *newfacet;
  pointT *point, **pointp;
  int delsize, coplanar=0, size;
  vertexT *vertex, **vertexp;

  trace3((qh, qh->ferr, 3042, "qh_partitionvisible: partition outside and coplanar points of visible and merged facets f%d into new facets f%d\n",
    qh->visible_list->id, qh->newfacet_list->id));
  if (qh->ONLYmax)
    maximize_(qh->MINoutside, qh->max_vertex);
  *numoutside= 0;
  FORALLvisible_facets {
    if (!visible->outsideset && !visible->coplanarset)
      continue;
    newfacet= qh_getreplacement(qh, visible);
    if (!newfacet)
      newfacet= qh->newfacet_list;
    if (!newfacet->next) {
      qh_fprintf(qh, qh->ferr, 6170, "qhull topology error (qh_partitionvisible): all new facets deleted as\n       degenerate facets. Can not continue.\n");
      qh_errexit(qh, qh_ERRtopology, NULL, NULL);
    }
    if (visible->outsideset) {
      size= qh_setsize(qh, visible->outsideset);
      *numoutside += size;
      qh->num_outside -= size;
      FOREACHpoint_(visible->outsideset)
        qh_partitionpoint(qh, point, newfacet);
    }
    if (visible->coplanarset && (qh->KEEPcoplanar + qh->KEEPinside + qh->KEEPnearinside)) {
      size= qh_setsize(qh, visible->coplanarset);
      coplanar += size;
      FOREACHpoint_(visible->coplanarset) {
        if (allpoints) /* not used */
          qh_partitionpoint(qh, point, newfacet);
        else
          qh_partitioncoplanar(qh, point, newfacet, NULL, qh->findbestnew);
      }
    }
  }
  delsize= qh_setsize(qh, qh->del_vertices);
  if (delsize > 0) {
    trace3((qh, qh->ferr, 3049, "qh_partitionvisible: partition %d deleted vertices as coplanar? %d points into new facets f%d\n",
      delsize, !allpoints, qh->newfacet_list->id));
    FOREACHvertex_(qh->del_vertices) {
      if (vertex->point && !vertex->partitioned) {
        if (!qh->newfacet_list || qh->newfacet_list == qh->facet_tail) {
          qh_fprintf(qh, qh->ferr, 6284, "qhull internal error (qh_partitionvisible): all new facets deleted or none defined.  Can not partition deleted v%d.\n", vertex->id);
          qh_errexit(qh, qh_ERRqhull, NULL, NULL);
        }
        if (allpoints) /* not used */
          /* [apr'2019] infinite loop if vertex recreates the same facets from the same horizon
             e.g., qh_partitionpoint if qh.DELAUNAY with qh.MERGEindependent for all mergetype, ../eg/qtest.sh t427764 '1000 s W1e-13 D3' 'd' */
          qh_partitionpoint(qh, vertex->point, qh->newfacet_list);
        else
          qh_partitioncoplanar(qh, vertex->point, qh->newfacet_list, NULL, qh_ALL); /* search all new facets */
        vertex->partitioned= True;
      }
    }
  }
  trace1((qh, qh->ferr, 1043,"qh_partitionvisible: partitioned %d points from outsidesets, %d points from coplanarsets, and %d deleted vertices\n", *numoutside, coplanar, delsize));
} /* partitionvisible */

/*-<a                             href="qh-qhull_r.htm#TOC"
  >-------------------------------</a><a name="printsummary">-</a>

  qh_printsummary(qh, fp )
    prints summary to fp

  notes:
    not in io_r.c so that user_eg.c can prevent io_r.c from loading
    qh_printsummary and qh_countfacets must match counts
    updates qh.facet_visit to detect infinite loop

  design:
    determine number of points, vertices, and coplanar points
    print summary
*/
void qh_printsummary(qhT *qh, FILE *fp) {
  realT ratio, outerplane, innerplane;
  double cpu;
  int size, id, nummerged, numpinched, numvertices, numcoplanars= 0, nonsimplicial=0, numdelaunay= 0;
  facetT *facet;
  const char *s;
  int numdel= zzval_(Zdelvertextot);
  int numtricoplanars= 0;
  boolT goodused;

  size= qh->num_points + qh_setsize(qh, qh->other_points);
  numvertices= qh->num_vertices - qh_setsize(qh, qh->del_vertices);
  id= qh_pointid(qh, qh->GOODpointp);
  if (!qh_checklists(qh, qh->facet_list) && !qh->ERREXITcalled) {
    qh_fprintf(qh, fp, 6372, "qhull internal error: qh_checklists failed at qh_printsummary\n");
    if (qh->num_facets < 4000)
      qh_printlists(qh);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (qh->DELAUNAY && qh->ERREXITcalled) {
    /* update f.good and determine qh.num_good as in qh_findgood_all */
    FORALLfacets {
      if (facet->visible)
        facet->good= False; /* will be deleted */
      else if (facet->good) {
        if (facet->normal && !qh_inthresholds(qh, facet->normal, NULL))
          facet->good= False;
        else
          numdelaunay++;
      }
    }
    qh->num_good= numdelaunay;
  }
  FORALLfacets {
    if (facet->coplanarset)
      numcoplanars += qh_setsize(qh, facet->coplanarset);
    if (facet->good) {
      if (facet->simplicial) {
        if (facet->keepcentrum && facet->tricoplanar)
          numtricoplanars++;
      }else if (qh_setsize(qh, facet->vertices) != qh->hull_dim)
        nonsimplicial++;
    }
  }
  if (id >=0 && qh->STOPcone-1 != id && -qh->STOPpoint-1 != id)
    size--;
  if (qh->STOPadd || qh->STOPcone || qh->STOPpoint)
    qh_fprintf(qh, fp, 9288, "\nEarly exit due to 'TAn', 'TVn', 'TCn', 'TRn', or precision error with 'QJn'.");
  goodused= False;
  if (qh->ERREXITcalled)
    ; /* qh_findgood_all not called */
  else if (qh->UPPERdelaunay) {
    if (qh->GOODvertex || qh->GOODpoint || qh->SPLITthresholds)
      goodused= True;
  }else if (qh->DELAUNAY) {
    if (qh->GOODvertex || qh->GOODpoint || qh->GOODthreshold)
      goodused= True;
  }else if (qh->num_good > 0 || qh->GOODthreshold)
    goodused= True;
  nummerged= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);
  if (qh->VORONOI) {
    if (qh->UPPERdelaunay)
      qh_fprintf(qh, fp, 9289, "\n\
Furthest-site Voronoi vertices by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    else
      qh_fprintf(qh, fp, 9290, "\n\
Voronoi diagram by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    qh_fprintf(qh, fp, 9291, "  Number of Voronoi regions%s: %d\n",
              qh->ATinfinity ? " and at-infinity" : "", numvertices);
    if (numdel)
      qh_fprintf(qh, fp, 9292, "  Total number of deleted points due to merging: %d\n", numdel);
    if (numcoplanars - numdel > 0)
      qh_fprintf(qh, fp, 9293, "  Number of nearly incident points: %d\n", numcoplanars - numdel);
    else if (size - numvertices - numdel > 0)
      qh_fprintf(qh, fp, 9294, "  Total number of nearly incident points: %d\n", size - numvertices - numdel);
    qh_fprintf(qh, fp, 9295, "  Number of%s Voronoi vertices: %d\n",
              goodused ? " 'good'" : "", qh->num_good);
    if (nonsimplicial)
      qh_fprintf(qh, fp, 9296, "  Number of%s non-simplicial Voronoi vertices: %d\n",
              goodused ? " 'good'" : "", nonsimplicial);
  }else if (qh->DELAUNAY) {
    if (qh->UPPERdelaunay)
      qh_fprintf(qh, fp, 9297, "\n\
Furthest-site Delaunay triangulation by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    else
      qh_fprintf(qh, fp, 9298, "\n\
Delaunay triangulation by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    qh_fprintf(qh, fp, 9299, "  Number of input sites%s: %d\n",
              qh->ATinfinity ? " and at-infinity" : "", numvertices);
    if (numdel)
      qh_fprintf(qh, fp, 9300, "  Total number of deleted points due to merging: %d\n", numdel);
    if (numcoplanars - numdel > 0)
      qh_fprintf(qh, fp, 9301, "  Number of nearly incident points: %d\n", numcoplanars - numdel);
    else if (size - numvertices - numdel > 0)
      qh_fprintf(qh, fp, 9302, "  Total number of nearly incident points: %d\n", size - numvertices - numdel);
    qh_fprintf(qh, fp, 9303, "  Number of%s Delaunay regions: %d\n",
              goodused ? " 'good'" : "", qh->num_good);
    if (nonsimplicial)
      qh_fprintf(qh, fp, 9304, "  Number of%s non-simplicial Delaunay regions: %d\n",
              goodused ? " 'good'" : "", nonsimplicial);
  }else if (qh->HALFspace) {
    qh_fprintf(qh, fp, 9305, "\n\
Halfspace intersection by the convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    qh_fprintf(qh, fp, 9306, "  Number of halfspaces: %d\n", size);
    qh_fprintf(qh, fp, 9307, "  Number of non-redundant halfspaces: %d\n", numvertices);
    if (numcoplanars) {
      if (qh->KEEPinside && qh->KEEPcoplanar)
        s= "similar and redundant";
      else if (qh->KEEPinside)
        s= "redundant";
      else
        s= "similar";
      qh_fprintf(qh, fp, 9308, "  Number of %s halfspaces: %d\n", s, numcoplanars);
    }
    qh_fprintf(qh, fp, 9309, "  Number of intersection points: %d\n", qh->num_facets - qh->num_visible);
    if (goodused)
      qh_fprintf(qh, fp, 9310, "  Number of 'good' intersection points: %d\n", qh->num_good);
    if (nonsimplicial)
      qh_fprintf(qh, fp, 9311, "  Number of%s non-simplicial intersection points: %d\n",
              goodused ? " 'good'" : "", nonsimplicial);
  }else {
    qh_fprintf(qh, fp, 9312, "\n\
Convex hull of %d points in %d-d:\n\n", size, qh->hull_dim);
    qh_fprintf(qh, fp, 9313, "  Number of vertices: %d\n", numvertices);
    if (numcoplanars) {
      if (qh->KEEPinside && qh->KEEPcoplanar)
        s= "coplanar and interior";
      else if (qh->KEEPinside)
        s= "interior";
      else
        s= "coplanar";
      qh_fprintf(qh, fp, 9314, "  Number of %s points: %d\n", s, numcoplanars);
    }
    qh_fprintf(qh, fp, 9315, "  Number of facets: %d\n", qh->num_facets - qh->num_visible);
    if (goodused)
      qh_fprintf(qh, fp, 9316, "  Number of 'good' facets: %d\n", qh->num_good);
    if (nonsimplicial)
      qh_fprintf(qh, fp, 9317, "  Number of%s non-simplicial facets: %d\n",
              goodused ? " 'good'" : "", nonsimplicial);
  }
  if (numtricoplanars)
      qh_fprintf(qh, fp, 9318, "  Number of triangulated facets: %d\n", numtricoplanars);
  qh_fprintf(qh, fp, 9319, "\nStatistics for: %s | %s",
                      qh->rbox_command, qh->qhull_command);
  if (qh->ROTATErandom != INT_MIN)
    qh_fprintf(qh, fp, 9320, " QR%d\n\n", qh->ROTATErandom);
  else
    qh_fprintf(qh, fp, 9321, "\n\n");
  qh_fprintf(qh, fp, 9322, "  Number of points processed: %d\n", zzval_(Zprocessed));
  qh_fprintf(qh, fp, 9323, "  Number of hyperplanes created: %d\n", zzval_(Zsetplane));
  if (qh->DELAUNAY)
    qh_fprintf(qh, fp, 9324, "  Number of facets in hull: %d\n", qh->num_facets - qh->num_visible);
  qh_fprintf(qh, fp, 9325, "  Number of distance tests for qhull: %d\n", zzval_(Zpartition)+
      zzval_(Zpartitionall)+zzval_(Znumvisibility)+zzval_(Zpartcoplanar));
#if 0  /* NOTE: must print before printstatistics() */
  {realT stddev, ave;
  qh_fprintf(qh, fp, 9326, "  average new facet balance: %2.2g\n",
          wval_(Wnewbalance)/zval_(Zprocessed));
  stddev= qh_stddev(zval_(Zprocessed), wval_(Wnewbalance),
                                 wval_(Wnewbalance2), &ave);
  qh_fprintf(qh, fp, 9327, "  new facet standard deviation: %2.2g\n", stddev);
  qh_fprintf(qh, fp, 9328, "  average partition balance: %2.2g\n",
          wval_(Wpbalance)/zval_(Zpbalance));
  stddev= qh_stddev(zval_(Zpbalance), wval_(Wpbalance),
                                 wval_(Wpbalance2), &ave);
  qh_fprintf(qh, fp, 9329, "  partition standard deviation: %2.2g\n", stddev);
  }
#endif
  if (nummerged) {
    qh_fprintf(qh, fp, 9330,"  Number of distance tests for merging: %d\n",zzval_(Zbestdist)+
          zzval_(Zcentrumtests)+zzval_(Zvertextests)+zzval_(Zdistcheck)+zzval_(Zdistzero));
    qh_fprintf(qh, fp, 9331,"  Number of distance tests for checking: %d\n",zzval_(Zcheckpart)+zzval_(Zdistconvex));
    qh_fprintf(qh, fp, 9332,"  Number of merged facets: %d\n", nummerged);
  }
  numpinched= zzval_(Zpinchduplicate) + zzval_(Zpinchedvertex);
  if (numpinched)
    qh_fprintf(qh, fp, 9375,"  Number of merged pinched vertices: %d\n", numpinched);
  if (!qh->RANDOMoutside && qh->QHULLfinished) {
    cpu= (double)qh->hulltime;
    cpu /= (double)qh_SECticks;
    wval_(Wcpu)= cpu;
    qh_fprintf(qh, fp, 9333, "  CPU seconds to compute hull (after input): %2.4g\n", cpu);
  }
  if (qh->RERUN) {
    if (!qh->PREmerge && !qh->MERGEexact)
      qh_fprintf(qh, fp, 9334, "  Percentage of runs with precision errors: %4.1f\n",
           zzval_(Zretry)*100.0/qh->build_cnt);  /* careful of order */
  }else if (qh->JOGGLEmax < REALmax/2) {
    if (zzval_(Zretry))
      qh_fprintf(qh, fp, 9335, "  After %d retries, input joggled by: %2.2g\n",
         zzval_(Zretry), qh->JOGGLEmax);
    else
      qh_fprintf(qh, fp, 9336, "  Input joggled by: %2.2g\n", qh->JOGGLEmax);
  }
  if (qh->totarea != 0.0)
    qh_fprintf(qh, fp, 9337, "  %s facet area:   %2.8g\n",
            zzval_(Ztotmerge) ? "Approximate" : "Total", qh->totarea);
  if (qh->totvol != 0.0)
    qh_fprintf(qh, fp, 9338, "  %s volume:       %2.8g\n",
            zzval_(Ztotmerge) ? "Approximate" : "Total", qh->totvol);
  if (qh->MERGING) {
    qh_outerinner(qh, NULL, &outerplane, &innerplane);
    if (outerplane > 2 * qh->DISTround) {
      qh_fprintf(qh, fp, 9339, "  Maximum distance of point above facet: %2.2g", outerplane);
      ratio= outerplane/(qh->ONEmerge + qh->DISTround);
      /* don't report ratio if MINoutside is large */
      if (ratio > 0.05 && 2* qh->ONEmerge > qh->MINoutside && qh->JOGGLEmax > REALmax/2)
        qh_fprintf(qh, fp, 9340, " (%.1fx)\n", ratio);
      else
        qh_fprintf(qh, fp, 9341, "\n");
    }
    if (innerplane < -2 * qh->DISTround) {
      qh_fprintf(qh, fp, 9342, "  Maximum distance of vertex below facet: %2.2g", innerplane);
      ratio= -innerplane/(qh->ONEmerge+qh->DISTround);
      if (ratio > 0.05 && qh->JOGGLEmax > REALmax/2)
        qh_fprintf(qh, fp, 9343, " (%.1fx)\n", ratio);
      else
        qh_fprintf(qh, fp, 9344, "\n");
    }
  }
  qh_fprintf(qh, fp, 9345, "\n");
} /* printsummary */



/*<html><pre>  -<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="TOP">-</a>

   merge.c
   merges non-convex facets

   see qh-merge.htm and merge.h

   other modules call qh_premerge() and qh_postmerge()

   the user may call qh_postmerge() to perform additional merges.

   To remove deleted facets and vertices (qhull() in libqhull.c):
     qh_partitionvisible(!qh_ALL, &numoutside);  // visible_list, newfacet_list
     qh_deletevisible();         // qh.visible_list
     qh_resetlists(False, qh_RESETvisible);       // qh.visible_list newvertex_list newfacet_list

   assumes qh.CENTERtype= centrum

   merges occur in qh_mergefacet and in qh_mergecycle
   vertex->neighbors not set until the first merge occurs

   Copyright (c) 1993-2012 C.B. Barber.
   $Id: //main/2011/qhull/src/libqhull/merge.c#4 $$Change: 1490 $
   $DateTime: 2012/02/19 20:27:01 $$Author: bbarber $
*/

#include "qhull_a.h"

#ifndef qh_NOmerge

/*===== functions(alphabetical after premerge and postmerge) ======*/

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="premerge">-</a>

  qh_premerge( apex, maxcentrum )
    pre-merge nonconvex facets in qh.newfacet_list for apex
    maxcentrum defines coplanar and concave (qh_test_appendmerge)

  returns:
    deleted facets added to qh.visible_list with facet->visible set

  notes:
    uses globals, qh.MERGEexact, qh.PREmerge

  design:
    mark duplicate ridges in qh.newfacet_list
    merge facet cycles in qh.newfacet_list
    merge duplicate ridges and concave facets in qh.newfacet_list
    check merged facet cycles for degenerate and redundant facets
    merge degenerate and redundant facets
    collect coplanar and concave facets
    merge concave, coplanar, degenerate, and redundant facets
*/
void qh_premerge(vertexT *apex, realT maxcentrum, realT maxangle) {
  boolT othermerge= False;
  facetT *newfacet;

  if (qh ZEROcentrum && qh_checkzero(!qh_ALL))
    return;
  trace2((qh ferr, 2008, "qh_premerge: premerge centrum %2.2g angle %2.2g for apex v%d facetlist f%d\n",
            maxcentrum, maxangle, apex->id, getid_(qh newfacet_list)));
  if (qh IStracing >= 4 && qh num_facets < 50)
    qh_printlists();
  qh centrum_radius= maxcentrum;
  qh cos_max= maxangle;
  qh degen_mergeset= qh_settemp(qh TEMPsize);
  qh facet_mergeset= qh_settemp(qh TEMPsize);
  if (qh hull_dim >=3) {
    qh_mark_dupridges(qh newfacet_list); /* facet_mergeset */
    qh_mergecycle_all(qh newfacet_list, &othermerge);
    qh_forcedmerges(&othermerge /* qh facet_mergeset */);
    FORALLnew_facets {  /* test samecycle merges */
      if (!newfacet->simplicial && !newfacet->mergeridge)
        qh_degen_redundant_neighbors(newfacet, NULL);
    }
    if (qh_merge_degenredundant())
      othermerge= True;
  }else /* qh hull_dim == 2 */
    qh_mergecycle_all(qh newfacet_list, &othermerge);
  qh_flippedmerges(qh newfacet_list, &othermerge);
  if (!qh MERGEexact || zzval_(Ztotmerge)) {
    zinc_(Zpremergetot);
    qh POSTmerging= False;
    qh_getmergeset_initial(qh newfacet_list);
    qh_all_merges(othermerge, False);
  }
  qh_settempfree(&qh facet_mergeset);
  qh_settempfree(&qh degen_mergeset);
} /* premerge */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="postmerge">-</a>

  qh_postmerge( reason, maxcentrum, maxangle, vneighbors )
    post-merge nonconvex facets as defined by maxcentrum and maxangle
    'reason' is for reporting progress
    if vneighbors,
      calls qh_test_vneighbors at end of qh_all_merge
    if firstmerge,
      calls qh_reducevertices before qh_getmergeset

  returns:
    if first call (qh.visible_list != qh.facet_list),
      builds qh.facet_newlist, qh.newvertex_list
    deleted facets added to qh.visible_list with facet->visible
    qh.visible_list == qh.facet_list

  notes:


  design:
    if first call
      set qh.visible_list and qh.newfacet_list to qh.facet_list
      add all facets to qh.newfacet_list
      mark non-simplicial facets, facet->newmerge
      set qh.newvertext_list to qh.vertex_list
      add all vertices to qh.newvertex_list
      if a pre-merge occured
        set vertex->delridge {will retest the ridge}
        if qh.MERGEexact
          call qh_reducevertices()
      if no pre-merging
        merge flipped facets
    determine non-convex facets
    merge all non-convex facets
*/
void qh_postmerge(const char *reason, realT maxcentrum, realT maxangle,
                      boolT vneighbors) {
  facetT *newfacet;
  boolT othermerges= False;
  vertexT *vertex;

  if (qh REPORTfreq || qh IStracing) {
    qh_buildtracing(NULL, NULL);
    qh_printsummary(qh ferr);
    if (qh PRINTstatistics)
      qh_printallstatistics(qh ferr, "reason");
    qh_fprintf(qh ferr, 8062, "\n%s with 'C%.2g' and 'A%.2g'\n",
        reason, maxcentrum, maxangle);
  }
  trace2((qh ferr, 2009, "qh_postmerge: postmerge.  test vneighbors? %d\n",
            vneighbors));
  qh centrum_radius= maxcentrum;
  qh cos_max= maxangle;
  qh POSTmerging= True;
  qh degen_mergeset= qh_settemp(qh TEMPsize);
  qh facet_mergeset= qh_settemp(qh TEMPsize);
  if (qh visible_list != qh facet_list) {  /* first call */
    qh NEWfacets= True;
    qh visible_list= qh newfacet_list= qh facet_list;
    FORALLnew_facets {
      newfacet->newfacet= True;
       if (!newfacet->simplicial)
        newfacet->newmerge= True;
     zinc_(Zpostfacets);
    }
    qh newvertex_list= qh vertex_list;
    FORALLvertices
      vertex->newlist= True;
    if (qh VERTEXneighbors) { /* a merge has occurred */
      FORALLvertices
        vertex->delridge= True; /* test for redundant, needed? */
      if (qh MERGEexact) {
        if (qh hull_dim <= qh_DIMreduceBuild)
          qh_reducevertices(); /* was skipped during pre-merging */
      }
    }
    if (!qh PREmerge && !qh MERGEexact)
      qh_flippedmerges(qh newfacet_list, &othermerges);
  }
  qh_getmergeset_initial(qh newfacet_list);
  qh_all_merges(False, vneighbors);
  qh_settempfree(&qh facet_mergeset);
  qh_settempfree(&qh degen_mergeset);
} /* post_merge */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="all_merges">-</a>

  qh_all_merges( othermerge, vneighbors )
    merge all non-convex facets

    set othermerge if already merged facets (for qh_reducevertices)
    if vneighbors
      tests vertex neighbors for convexity at end
    qh.facet_mergeset lists the non-convex ridges in qh_newfacet_list
    qh.degen_mergeset is defined
    if qh.MERGEexact && !qh.POSTmerging,
      does not merge coplanar facets

  returns:
    deleted facets added to qh.visible_list with facet->visible
    deleted vertices added qh.delvertex_list with vertex->delvertex

  notes:
    unless !qh.MERGEindependent,
      merges facets in independent sets
    uses qh.newfacet_list as argument since merges call qh_removefacet()

  design:
    while merges occur
      for each merge in qh.facet_mergeset
        unless one of the facets was already merged in this pass
          merge the facets
        test merged facets for additional merges
        add merges to qh.facet_mergeset
      if vertices record neighboring facets
        rename redundant vertices
          update qh.facet_mergeset
    if vneighbors ??
      tests vertex neighbors for convexity at end
*/
void qh_all_merges(boolT othermerge, boolT vneighbors) {
  facetT *facet1, *facet2;
  mergeT *merge;
  boolT wasmerge= True, isreduce;
  void **freelistp;  /* used !qh_NOmem */
  vertexT *vertex;
  mergeType mergetype;
  int numcoplanar=0, numconcave=0, numdegenredun= 0, numnewmerges= 0;

  trace2((qh ferr, 2010, "qh_all_merges: starting to merge facets beginning from f%d\n",
            getid_(qh newfacet_list)));
  while (True) {
    wasmerge= False;
    while (qh_setsize(qh facet_mergeset)) {
      while ((merge= (mergeT*)qh_setdellast(qh facet_mergeset))) {
        facet1= merge->facet1;
        facet2= merge->facet2;
        mergetype= merge->type;
        qh_memfree_(merge, (int)sizeof(mergeT), freelistp);
        if (facet1->visible || facet2->visible) /*deleted facet*/
          continue;
        if ((facet1->newfacet && !facet1->tested)
                || (facet2->newfacet && !facet2->tested)) {
          if (qh MERGEindependent && mergetype <= MRGanglecoplanar)
            continue;      /* perform independent sets of merges */
        }
        qh_merge_nonconvex(facet1, facet2, mergetype);
        numdegenredun += qh_merge_degenredundant();
        numnewmerges++;
        wasmerge= True;
        if (mergetype == MRGconcave)
          numconcave++;
        else /* MRGcoplanar or MRGanglecoplanar */
          numcoplanar++;
      } /* while setdellast */
      if (qh POSTmerging && qh hull_dim <= qh_DIMreduceBuild
      && numnewmerges > qh_MAXnewmerges) {
        numnewmerges= 0;
        qh_reducevertices();  /* otherwise large post merges too slow */
      }
      qh_getmergeset(qh newfacet_list); /* facet_mergeset */
    } /* while mergeset */
    if (qh VERTEXneighbors) {
      isreduce= False;
      if (qh hull_dim >=4 && qh POSTmerging) {
        FORALLvertices
          vertex->delridge= True;
        isreduce= True;
      }
      if ((wasmerge || othermerge) && (!qh MERGEexact || qh POSTmerging)
          && qh hull_dim <= qh_DIMreduceBuild) {
        othermerge= False;
        isreduce= True;
      }
      if (isreduce) {
        if (qh_reducevertices()) {
          qh_getmergeset(qh newfacet_list); /* facet_mergeset */
          continue;
        }
      }
    }
    if (vneighbors && qh_test_vneighbors(/* qh newfacet_list */))
      continue;
    break;
  } /* while (True) */
  if (qh CHECKfrequently && !qh MERGEexact) {
    qh old_randomdist= qh RANDOMdist;
    qh RANDOMdist= False;
    qh_checkconvex(qh newfacet_list, qh_ALGORITHMfault);
    /* qh_checkconnect(); [this is slow and it changes the facet order] */
    qh RANDOMdist= qh old_randomdist;
  }
  trace1((qh ferr, 1009, "qh_all_merges: merged %d coplanar facets %d concave facets and %d degen or redundant facets.\n",
    numcoplanar, numconcave, numdegenredun));
  if (qh IStracing >= 4 && qh num_facets < 50)
    qh_printlists();
} /* all_merges */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="appendmergeset">-</a>

  qh_appendmergeset( facet, neighbor, mergetype, angle )
    appends an entry to qh.facet_mergeset or qh.degen_mergeset

    angle ignored if NULL or !qh.ANGLEmerge

  returns:
    merge appended to facet_mergeset or degen_mergeset
      sets ->degenerate or ->redundant if degen_mergeset

  see:
    qh_test_appendmerge()

  design:
    allocate merge entry
    if regular merge
      append to qh.facet_mergeset
    else if degenerate merge and qh.facet_mergeset is all degenerate
      append to qh.degen_mergeset
    else if degenerate merge
      prepend to qh.degen_mergeset
    else if redundant merge
      append to qh.degen_mergeset
*/
void qh_appendmergeset(facetT *facet, facetT *neighbor, mergeType mergetype, realT *angle) {
  mergeT *merge, *lastmerge;
  void **freelistp; /* used !qh_NOmem */

  if (facet->redundant)
    return;
  if (facet->degenerate && mergetype == MRGdegen)
    return;
  qh_memalloc_((int)sizeof(mergeT), freelistp, merge, mergeT);
  merge->facet1= facet;
  merge->facet2= neighbor;
  merge->type= mergetype;
  if (angle && qh ANGLEmerge)
    merge->angle= *angle;
  if (mergetype < MRGdegen)
    qh_setappend(&(qh facet_mergeset), merge);
  else if (mergetype == MRGdegen) {
    facet->degenerate= True;
    if (!(lastmerge= (mergeT*)qh_setlast(qh degen_mergeset))
    || lastmerge->type == MRGdegen)
      qh_setappend(&(qh degen_mergeset), merge);
    else
      qh_setaddnth(&(qh degen_mergeset), 0, merge);
  }else if (mergetype == MRGredundant) {
    facet->redundant= True;
    qh_setappend(&(qh degen_mergeset), merge);
  }else /* mergetype == MRGmirror */ {
    if (facet->redundant || neighbor->redundant) {
      qh_fprintf(qh ferr, 6092, "qhull error (qh_appendmergeset): facet f%d or f%d is already a mirrored facet\n",
           facet->id, neighbor->id);
      qh_errexit2 (qh_ERRqhull, facet, neighbor);
    }
    if (!qh_setequal(facet->vertices, neighbor->vertices)) {
      qh_fprintf(qh ferr, 6093, "qhull error (qh_appendmergeset): mirrored facets f%d and f%d do not have the same vertices\n",
           facet->id, neighbor->id);
      qh_errexit2 (qh_ERRqhull, facet, neighbor);
    }
    facet->redundant= True;
    neighbor->redundant= True;
    qh_setappend(&(qh degen_mergeset), merge);
  }
} /* appendmergeset */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="basevertices">-</a>

  qh_basevertices( samecycle )
    return temporary set of base vertices for samecycle
    samecycle is first facet in the cycle
    assumes apex is SETfirst_( samecycle->vertices )

  returns:
    vertices(settemp)
    all ->seen are cleared

  notes:
    uses qh_vertex_visit;

  design:
    for each facet in samecycle
      for each unseen vertex in facet->vertices
        append to result
*/
setT *qh_basevertices(facetT *samecycle) {
  facetT *same;
  vertexT *apex, *vertex, **vertexp;
  setT *vertices= qh_settemp(qh TEMPsize);

  apex= SETfirstt_(samecycle->vertices, vertexT);
  apex->visitid= ++qh vertex_visit;
  FORALLsame_cycle_(samecycle) {
    if (same->mergeridge)
      continue;
    FOREACHvertex_(same->vertices) {
      if (vertex->visitid != qh vertex_visit) {
        qh_setappend(&vertices, vertex);
        vertex->visitid= qh vertex_visit;
        vertex->seen= False;
      }
    }
  }
  trace4((qh ferr, 4019, "qh_basevertices: found %d vertices\n",
         qh_setsize(vertices)));
  return vertices;
} /* basevertices */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="checkconnect">-</a>

  qh_checkconnect()
    check that new facets are connected
    new facets are on qh.newfacet_list

  notes:
    this is slow and it changes the order of the facets
    uses qh.visit_id

  design:
    move first new facet to end of qh.facet_list
    for all newly appended facets
      append unvisited neighbors to end of qh.facet_list
    for all new facets
      report error if unvisited
*/
void qh_checkconnect(void /* qh newfacet_list */) {
  facetT *facet, *newfacet, *errfacet= NULL, *neighbor, **neighborp;

  facet= qh newfacet_list;
  qh_removefacet(facet);
  qh_appendfacet(facet);
  facet->visitid= ++qh visit_id;
  FORALLfacet_(facet) {
    FOREACHneighbor_(facet) {
      if (neighbor->visitid != qh visit_id) {
        qh_removefacet(neighbor);
        qh_appendfacet(neighbor);
        neighbor->visitid= qh visit_id;
      }
    }
  }
  FORALLnew_facets {
    if (newfacet->visitid == qh visit_id)
      break;
    qh_fprintf(qh ferr, 6094, "qhull error: f%d is not attached to the new facets\n",
         newfacet->id);
    errfacet= newfacet;
  }
  if (errfacet)
    qh_errexit(qh_ERRqhull, errfacet, NULL);
} /* checkconnect */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="checkzero">-</a>

  qh_checkzero( testall )
    check that facets are clearly convex for qh.DISTround with qh.MERGEexact

    if testall,
      test all facets for qh.MERGEexact post-merging
    else
      test qh.newfacet_list

    if qh.MERGEexact,
      allows coplanar ridges
      skips convexity test while qh.ZEROall_ok

  returns:
    True if all facets !flipped, !dupridge, normal
         if all horizon facets are simplicial
         if all vertices are clearly below neighbor
         if all opposite vertices of horizon are below
    clears qh.ZEROall_ok if any problems or coplanar facets

  notes:
    uses qh.vertex_visit
    horizon facets may define multiple new facets

  design:
    for all facets in qh.newfacet_list or qh.facet_list
      check for flagged faults (flipped, etc.)
    for all facets in qh.newfacet_list or qh.facet_list
      for each neighbor of facet
        skip horizon facets for qh.newfacet_list
        test the opposite vertex
      if qh.newfacet_list
        test the other vertices in the facet's horizon facet
*/
boolT qh_checkzero(boolT testall) {
  facetT *facet, *neighbor, **neighborp;
  facetT *horizon, *facetlist;
  int neighbor_i;
  vertexT *vertex, **vertexp;
  realT dist;

  if (testall)
    facetlist= qh facet_list;
  else {
    facetlist= qh newfacet_list;
    FORALLfacet_(facetlist) {
      horizon= SETfirstt_(facet->neighbors, facetT);
      if (!horizon->simplicial)
        goto LABELproblem;
      if (facet->flipped || facet->dupridge || !facet->normal)
        goto LABELproblem;
    }
    if (qh MERGEexact && qh ZEROall_ok) {
      trace2((qh ferr, 2011, "qh_checkzero: skip convexity check until first pre-merge\n"));
      return True;
    }
  }
  FORALLfacet_(facetlist) {
    qh vertex_visit++;
    neighbor_i= 0;
    horizon= NULL;
    FOREACHneighbor_(facet) {
      if (!neighbor_i && !testall) {
        horizon= neighbor;
        neighbor_i++;
        continue; /* horizon facet tested in qh_findhorizon */
      }
      vertex= SETelemt_(facet->vertices, neighbor_i++, vertexT);
      vertex->visitid= qh vertex_visit;
      zzinc_(Zdistzero);
      qh_distplane(vertex->point, neighbor, &dist);
      if (dist >= -qh DISTround) {
        qh ZEROall_ok= False;
        if (!qh MERGEexact || testall || dist > qh DISTround)
          goto LABELnonconvex;
      }
    }
    if (!testall) {
      FOREACHvertex_(horizon->vertices) {
        if (vertex->visitid != qh vertex_visit) {
          zzinc_(Zdistzero);
          qh_distplane(vertex->point, facet, &dist);
          if (dist >= -qh DISTround) {
            qh ZEROall_ok= False;
            if (!qh MERGEexact || dist > qh DISTround)
              goto LABELnonconvex;
          }
          break;
        }
      }
    }
  }
  trace2((qh ferr, 2012, "qh_checkzero: testall %d, facets are %s\n", testall,
        (qh MERGEexact && !testall) ?
           "not concave, flipped, or duplicate ridged" : "clearly convex"));
  return True;

 LABELproblem:
  qh ZEROall_ok= False;
  trace2((qh ferr, 2013, "qh_checkzero: facet f%d needs pre-merging\n",
       facet->id));
  return False;

 LABELnonconvex:
  trace2((qh ferr, 2014, "qh_checkzero: facet f%d and f%d are not clearly convex.  v%d dist %.2g\n",
         facet->id, neighbor->id, vertex->id, dist));
  return False;
} /* checkzero */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="compareangle">-</a>

  qh_compareangle( angle1, angle2 )
    used by qsort() to order merges by angle
*/
int qh_compareangle(const void *p1, const void *p2) {
  const mergeT *a= *((mergeT *const*)p1), *b= *((mergeT *const*)p2);

  return((a->angle > b->angle) ? 1 : -1);
} /* compareangle */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="comparemerge">-</a>

  qh_comparemerge( merge1, merge2 )
    used by qsort() to order merges
*/
int qh_comparemerge(const void *p1, const void *p2) {
  const mergeT *a= *((mergeT *const*)p1), *b= *((mergeT *const*)p2);

  return(a->type - b->type);
} /* comparemerge */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="comparevisit">-</a>

  qh_comparevisit( vertex1, vertex2 )
    used by qsort() to order vertices by their visitid
*/
int qh_comparevisit(const void *p1, const void *p2) {
  const vertexT *a= *((vertexT *const*)p1), *b= *((vertexT *const*)p2);

  return(a->visitid - b->visitid);
} /* comparevisit */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="copynonconvex">-</a>

  qh_copynonconvex( atridge )
    set non-convex flag on other ridges (if any) between same neighbors

  notes:
    may be faster if use smaller ridge set

  design:
    for each ridge of atridge's top facet
      if ridge shares the same neighbor
        set nonconvex flag
*/
void qh_copynonconvex(ridgeT *atridge) {
  facetT *facet, *otherfacet;
  ridgeT *ridge, **ridgep;

  facet= atridge->top;
  otherfacet= atridge->bottom;
  FOREACHridge_(facet->ridges) {
    if (otherfacet == otherfacet_(ridge, facet) && ridge != atridge) {
      ridge->nonconvex= True;
      trace4((qh ferr, 4020, "qh_copynonconvex: moved nonconvex flag from r%d to r%d\n",
              atridge->id, ridge->id));
      break;
    }
  }
} /* copynonconvex */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="degen_redundant_facet">-</a>

  qh_degen_redundant_facet( facet )
    check facet for degen. or redundancy

  notes:
    bumps vertex_visit
    called if a facet was redundant but no longer is (qh_merge_degenredundant)
    qh_appendmergeset() only appends first reference to facet (i.e., redundant)

  see:
    qh_degen_redundant_neighbors()

  design:
    test for redundant neighbor
    test for degenerate facet
*/
void qh_degen_redundant_facet(facetT *facet) {
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;

  trace4((qh ferr, 4021, "qh_degen_redundant_facet: test facet f%d for degen/redundant\n",
          facet->id));
  FOREACHneighbor_(facet) {
    qh vertex_visit++;
    FOREACHvertex_(neighbor->vertices)
      vertex->visitid= qh vertex_visit;
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh vertex_visit)
        break;
    }
    if (!vertex) {
      qh_appendmergeset(facet, neighbor, MRGredundant, NULL);
      trace2((qh ferr, 2015, "qh_degen_redundant_facet: f%d is contained in f%d.  merge\n", facet->id, neighbor->id));
      return;
    }
  }
  if (qh_setsize(facet->neighbors) < qh hull_dim) {
    qh_appendmergeset(facet, facet, MRGdegen, NULL);
    trace2((qh ferr, 2016, "qh_degen_redundant_neighbors: f%d is degenerate.\n", facet->id));
  }
} /* degen_redundant_facet */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="degen_redundant_neighbors">-</a>

  qh_degen_redundant_neighbors( facet, delfacet,  )
    append degenerate and redundant neighbors to facet_mergeset
    if delfacet,
      only checks neighbors of both delfacet and facet
    also checks current facet for degeneracy

  notes:
    bumps vertex_visit
    called for each qh_mergefacet() and qh_mergecycle()
    merge and statistics occur in merge_nonconvex
    qh_appendmergeset() only appends first reference to facet (i.e., redundant)
      it appends redundant facets after degenerate ones

    a degenerate facet has fewer than hull_dim neighbors
    a redundant facet's vertices is a subset of its neighbor's vertices
    tests for redundant merges first (appendmergeset is nop for others)
    in a merge, only needs to test neighbors of merged facet

  see:
    qh_merge_degenredundant() and qh_degen_redundant_facet()

  design:
    test for degenerate facet
    test for redundant neighbor
    test for degenerate neighbor
*/
void qh_degen_redundant_neighbors(facetT *facet, facetT *delfacet) {
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;
  int size;

  trace4((qh ferr, 4022, "qh_degen_redundant_neighbors: test neighbors of f%d with delfacet f%d\n",
          facet->id, getid_(delfacet)));
  if ((size= qh_setsize(facet->neighbors)) < qh hull_dim) {
    qh_appendmergeset(facet, facet, MRGdegen, NULL);
    trace2((qh ferr, 2017, "qh_degen_redundant_neighbors: f%d is degenerate with %d neighbors.\n", facet->id, size));
  }
  if (!delfacet)
    delfacet= facet;
  qh vertex_visit++;
  FOREACHvertex_(facet->vertices)
    vertex->visitid= qh vertex_visit;
  FOREACHneighbor_(delfacet) {
    /* uses early out instead of checking vertex count */
    if (neighbor == facet)
      continue;
    FOREACHvertex_(neighbor->vertices) {
      if (vertex->visitid != qh vertex_visit)
        break;
    }
    if (!vertex) {
      qh_appendmergeset(neighbor, facet, MRGredundant, NULL);
      trace2((qh ferr, 2018, "qh_degen_redundant_neighbors: f%d is contained in f%d.  merge\n", neighbor->id, facet->id));
    }
  }
  FOREACHneighbor_(delfacet) {   /* redundant merges occur first */
    if (neighbor == facet)
      continue;
    if ((size= qh_setsize(neighbor->neighbors)) < qh hull_dim) {
      qh_appendmergeset(neighbor, neighbor, MRGdegen, NULL);
      trace2((qh ferr, 2019, "qh_degen_redundant_neighbors: f%d is degenerate with %d neighbors.  Neighbor of f%d.\n", neighbor->id, size, facet->id));
    }
  }
} /* degen_redundant_neighbors */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="find_newvertex">-</a>

  qh_find_newvertex( oldvertex, vertices, ridges )
    locate new vertex for renaming old vertex
    vertices is a set of possible new vertices
      vertices sorted by number of deleted ridges

  returns:
    newvertex or NULL
      each ridge includes both vertex and oldvertex
    vertices sorted by number of deleted ridges

  notes:
    modifies vertex->visitid
    new vertex is in one of the ridges
    renaming will not cause a duplicate ridge
    renaming will minimize the number of deleted ridges
    newvertex may not be adjacent in the dual (though unlikely)

  design:
    for each vertex in vertices
      set vertex->visitid to number of references in ridges
    remove unvisited vertices
    set qh.vertex_visit above all possible values
    sort vertices by number of references in ridges
    add each ridge to qh.hash_table
    for each vertex in vertices
      look for a vertex that would not cause a duplicate ridge after a rename
*/
vertexT *qh_find_newvertex(vertexT *oldvertex, setT *vertices, setT *ridges) {
  vertexT *vertex, **vertexp;
  setT *newridges;
  ridgeT *ridge, **ridgep;
  int size, hashsize;
  int hash;

#ifndef qh_NOtrace
  if (qh IStracing >= 4) {
    qh_fprintf(qh ferr, 8063, "qh_find_newvertex: find new vertex for v%d from ",
             oldvertex->id);
    FOREACHvertex_(vertices)
      qh_fprintf(qh ferr, 8064, "v%d ", vertex->id);
    FOREACHridge_(ridges)
      qh_fprintf(qh ferr, 8065, "r%d ", ridge->id);
    qh_fprintf(qh ferr, 8066, "\n");
  }
#endif
  FOREACHvertex_(vertices)
    vertex->visitid= 0;
  FOREACHridge_(ridges) {
    FOREACHvertex_(ridge->vertices)
      vertex->visitid++;
  }
  FOREACHvertex_(vertices) {
    if (!vertex->visitid) {
      qh_setdelnth(vertices, SETindex_(vertices,vertex));
      vertexp--; /* repeat since deleted this vertex */
    }
  }
  qh vertex_visit += (unsigned int)qh_setsize(ridges);
  if (!qh_setsize(vertices)) {
    trace4((qh ferr, 4023, "qh_find_newvertex: vertices not in ridges for v%d\n",
            oldvertex->id));
    return NULL;
  }
  qsort(SETaddr_(vertices, vertexT), (size_t)qh_setsize(vertices),
                sizeof(vertexT *), qh_comparevisit);
  /* can now use qh vertex_visit */
  if (qh PRINTstatistics) {
    size= qh_setsize(vertices);
    zinc_(Zintersect);
    zadd_(Zintersecttot, size);
    zmax_(Zintersectmax, size);
  }
  hashsize= qh_newhashtable(qh_setsize(ridges));
  FOREACHridge_(ridges)
    qh_hashridge(qh hash_table, hashsize, ridge, oldvertex);
  FOREACHvertex_(vertices) {
    newridges= qh_vertexridges(vertex);
    FOREACHridge_(newridges) {
      if (qh_hashridge_find(qh hash_table, hashsize, ridge, vertex, oldvertex, &hash)) {
        zinc_(Zdupridge);
        break;
      }
    }
    qh_settempfree(&newridges);
    if (!ridge)
      break;  /* found a rename */
  }
  if (vertex) {
    /* counted in qh_renamevertex */
    trace2((qh ferr, 2020, "qh_find_newvertex: found v%d for old v%d from %d vertices and %d ridges.\n",
      vertex->id, oldvertex->id, qh_setsize(vertices), qh_setsize(ridges)));
  }else {
    zinc_(Zfindfail);
    trace0((qh ferr, 14, "qh_find_newvertex: no vertex for renaming v%d(all duplicated ridges) during p%d\n",
      oldvertex->id, qh furthest_id));
  }
  qh_setfree(&qh hash_table);
  return vertex;
} /* find_newvertex */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="findbest_test">-</a>

  qh_findbest_test( testcentrum, facet, neighbor, bestfacet, dist, mindist, maxdist )
    test neighbor of facet for qh_findbestneighbor()
    if testcentrum,
      tests centrum (assumes it is defined)
    else
      tests vertices

  returns:
    if a better facet (i.e., vertices/centrum of facet closer to neighbor)
      updates bestfacet, dist, mindist, and maxdist
*/
void qh_findbest_test(boolT testcentrum, facetT *facet, facetT *neighbor,
      facetT **bestfacet, realT *distp, realT *mindistp, realT *maxdistp) {
  realT dist, mindist, maxdist;

  if (testcentrum) {
    zzinc_(Zbestdist);
    qh_distplane(facet->center, neighbor, &dist);
    dist *= qh hull_dim; /* estimate furthest vertex */
    if (dist < 0) {
      maxdist= 0;
      mindist= dist;
      dist= -dist;
    }else {
      mindist= 0;
      maxdist= dist;
    }
  }else
    dist= qh_getdistance(facet, neighbor, &mindist, &maxdist);
  if (dist < *distp) {
    *bestfacet= neighbor;
    *mindistp= mindist;
    *maxdistp= maxdist;
    *distp= dist;
  }
} /* findbest_test */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="findbestneighbor">-</a>

  qh_findbestneighbor( facet, dist, mindist, maxdist )
    finds best neighbor (least dist) of a facet for merging

  returns:
    returns min and max distances and their max absolute value

  notes:
    avoids merging old into new
    assumes ridge->nonconvex only set on one ridge between a pair of facets
    could use an early out predicate but not worth it

  design:
    if a large facet
      will test centrum
    else
      will test vertices
    if a large facet
      test nonconvex neighbors for best merge
    else
      test all neighbors for the best merge
    if testing centrum
      get distance information
*/
facetT *qh_findbestneighbor(facetT *facet, realT *distp, realT *mindistp, realT *maxdistp) {
  facetT *neighbor, **neighborp, *bestfacet= NULL;
  ridgeT *ridge, **ridgep;
  boolT nonconvex= True, testcentrum= False;
  int size= qh_setsize(facet->vertices);

  *distp= REALmax;
  if (size > qh_BESTcentrum2 * qh hull_dim + qh_BESTcentrum) {
    testcentrum= True;
    zinc_(Zbestcentrum);
    if (!facet->center)
       facet->center= qh_getcentrum(facet);
  }
  if (size > qh hull_dim + qh_BESTnonconvex) {
    FOREACHridge_(facet->ridges) {
      if (ridge->nonconvex) {
        neighbor= otherfacet_(ridge, facet);
        qh_findbest_test(testcentrum, facet, neighbor,
                          &bestfacet, distp, mindistp, maxdistp);
      }
    }
  }
  if (!bestfacet) {
    nonconvex= False;
    FOREACHneighbor_(facet)
      qh_findbest_test(testcentrum, facet, neighbor,
                        &bestfacet, distp, mindistp, maxdistp);
  }
  if (!bestfacet) {
    qh_fprintf(qh ferr, 6095, "qhull internal error (qh_findbestneighbor): no neighbors for f%d\n", facet->id);

    qh_errexit(qh_ERRqhull, facet, NULL);
  }
  if (testcentrum)
    qh_getdistance(facet, bestfacet, mindistp, maxdistp);
  trace3((qh ferr, 3002, "qh_findbestneighbor: f%d is best neighbor for f%d testcentrum? %d nonconvex? %d dist %2.2g min %2.2g max %2.2g\n",
     bestfacet->id, facet->id, testcentrum, nonconvex, *distp, *mindistp, *maxdistp));
  return(bestfacet);
} /* findbestneighbor */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="flippedmerges">-</a>

  qh_flippedmerges( facetlist, wasmerge )
    merge flipped facets into best neighbor
    assumes qh.facet_mergeset at top of temporary stack

  returns:
    no flipped facets on facetlist
    sets wasmerge if merge occurred
    degen/redundant merges passed through

  notes:
    othermerges not needed since qh.facet_mergeset is empty before & after
      keep it in case of change

  design:
    append flipped facets to qh.facetmergeset
    for each flipped merge
      find best neighbor
      merge facet into neighbor
      merge degenerate and redundant facets
    remove flipped merges from qh.facet_mergeset
*/
void qh_flippedmerges(facetT *facetlist, boolT *wasmerge) {
  facetT *facet, *neighbor, *facet1;
  realT dist, mindist, maxdist;
  mergeT *merge, **mergep;
  setT *othermerges;
  int nummerge=0;

  trace4((qh ferr, 4024, "qh_flippedmerges: begin\n"));
  FORALLfacet_(facetlist) {
    if (facet->flipped && !facet->visible)
      qh_appendmergeset(facet, facet, MRGflip, NULL);
  }
  othermerges= qh_settemppop(); /* was facet_mergeset */
  qh facet_mergeset= qh_settemp(qh TEMPsize);
  qh_settemppush(othermerges);
  FOREACHmerge_(othermerges) {
    facet1= merge->facet1;
    if (merge->type != MRGflip || facet1->visible)
      continue;
    if (qh TRACEmerge-1 == zzval_(Ztotmerge))
      qhmem.IStracing= qh IStracing= qh TRACElevel;
    neighbor= qh_findbestneighbor(facet1, &dist, &mindist, &maxdist);
    trace0((qh ferr, 15, "qh_flippedmerges: merge flipped f%d into f%d dist %2.2g during p%d\n",
      facet1->id, neighbor->id, dist, qh furthest_id));
    qh_mergefacet(facet1, neighbor, &mindist, &maxdist, !qh_MERGEapex);
    nummerge++;
    if (qh PRINTstatistics) {
      zinc_(Zflipped);
      wadd_(Wflippedtot, dist);
      wmax_(Wflippedmax, dist);
    }
    qh_merge_degenredundant();
  }
  FOREACHmerge_(othermerges) {
    if (merge->facet1->visible || merge->facet2->visible)
      qh_memfree(merge, (int)sizeof(mergeT));
    else
      qh_setappend(&qh facet_mergeset, merge);
  }
  qh_settempfree(&othermerges);
  if (nummerge)
    *wasmerge= True;
  trace1((qh ferr, 1010, "qh_flippedmerges: merged %d flipped facets into a good neighbor\n", nummerge));
} /* flippedmerges */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="forcedmerges">-</a>

  qh_forcedmerges( wasmerge )
    merge duplicated ridges

  returns:
    removes all duplicate ridges on facet_mergeset
    wasmerge set if merge
    qh.facet_mergeset may include non-forced merges(none for now)
    qh.degen_mergeset includes degen/redun merges

  notes:
    duplicate ridges occur when the horizon is pinched,
        i.e. a subridge occurs in more than two horizon ridges.
     could rename vertices that pinch the horizon
    assumes qh_merge_degenredundant() has not be called
    othermerges isn't needed since facet_mergeset is empty afterwards
      keep it in case of change

  design:
    for each duplicate ridge
      find current facets by chasing f.replace links
      determine best direction for facet
      merge one facet into the other
      remove duplicate ridges from qh.facet_mergeset
*/
void qh_forcedmerges(boolT *wasmerge) {
  facetT *facet1, *facet2;
  mergeT *merge, **mergep;
  realT dist1, dist2, mindist1, mindist2, maxdist1, maxdist2;
  setT *othermerges;
  int nummerge=0, numflip=0;

  if (qh TRACEmerge-1 == zzval_(Ztotmerge))
    qhmem.IStracing= qh IStracing= qh TRACElevel;
  trace4((qh ferr, 4025, "qh_forcedmerges: begin\n"));
  othermerges= qh_settemppop(); /* was facet_mergeset */
  qh facet_mergeset= qh_settemp(qh TEMPsize);
  qh_settemppush(othermerges);
  FOREACHmerge_(othermerges) {
    if (merge->type != MRGridge)
        continue;
    facet1= merge->facet1;
    facet2= merge->facet2;
    while (facet1->visible)      /* must exist, no qh_merge_degenredunant */
      facet1= facet1->f.replace; /* previously merged facet */
    while (facet2->visible)
      facet2= facet2->f.replace; /* previously merged facet */
    if (facet1 == facet2)
      continue;
    if (!qh_setin(facet2->neighbors, facet1)) {
      qh_fprintf(qh ferr, 6096, "qhull internal error (qh_forcedmerges): f%d and f%d had a duplicate ridge but as f%d and f%d they are no longer neighbors\n",
               merge->facet1->id, merge->facet2->id, facet1->id, facet2->id);
      qh_errexit2 (qh_ERRqhull, facet1, facet2);
    }
    if (qh TRACEmerge-1 == zzval_(Ztotmerge))
      qhmem.IStracing= qh IStracing= qh TRACElevel;
    dist1= qh_getdistance(facet1, facet2, &mindist1, &maxdist1);
    dist2= qh_getdistance(facet2, facet1, &mindist2, &maxdist2);
    trace0((qh ferr, 16, "qh_forcedmerges: duplicate ridge between f%d and f%d, dist %2.2g and reverse dist %2.2g during p%d\n",
            facet1->id, facet2->id, dist1, dist2, qh furthest_id));
    if (dist1 < dist2)
      qh_mergefacet(facet1, facet2, &mindist1, &maxdist1, !qh_MERGEapex);
    else {
      qh_mergefacet(facet2, facet1, &mindist2, &maxdist2, !qh_MERGEapex);
      dist1= dist2;
      facet1= facet2;
    }
    if (facet1->flipped) {
      zinc_(Zmergeflipdup);
      numflip++;
    }else
      nummerge++;
    if (qh PRINTstatistics) {
      zinc_(Zduplicate);
      wadd_(Wduplicatetot, dist1);
      wmax_(Wduplicatemax, dist1);
    }
  }
  FOREACHmerge_(othermerges) {
    if (merge->type == MRGridge)
      qh_memfree(merge, (int)sizeof(mergeT));
    else
      qh_setappend(&qh facet_mergeset, merge);
  }
  qh_settempfree(&othermerges);
  if (nummerge)
    *wasmerge= True;
  trace1((qh ferr, 1011, "qh_forcedmerges: merged %d facets and %d flipped facets across duplicated ridges\n",
                nummerge, numflip));
} /* forcedmerges */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="getmergeset">-</a>

  qh_getmergeset( facetlist )
    determines nonconvex facets on facetlist
    tests !tested ridges and nonconvex ridges of !tested facets

  returns:
    returns sorted qh.facet_mergeset of facet-neighbor pairs to be merged
    all ridges tested

  notes:
    assumes no nonconvex ridges with both facets tested
    uses facet->tested/ridge->tested to prevent duplicate tests
    can not limit tests to modified ridges since the centrum changed
    uses qh.visit_id

  see:
    qh_getmergeset_initial()

  design:
    for each facet on facetlist
      for each ridge of facet
        if untested ridge
          test ridge for convexity
          if non-convex
            append ridge to qh.facet_mergeset
    sort qh.facet_mergeset by angle
*/
void qh_getmergeset(facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int nummerges;

  nummerges= qh_setsize(qh facet_mergeset);
  trace4((qh ferr, 4026, "qh_getmergeset: started.\n"));
  qh visit_id++;
  FORALLfacet_(facetlist) {
    if (facet->tested)
      continue;
    facet->visitid= qh visit_id;
    facet->tested= True;  /* must be non-simplicial due to merge */
    FOREACHneighbor_(facet)
      neighbor->seen= False;
    FOREACHridge_(facet->ridges) {
      if (ridge->tested && !ridge->nonconvex)
        continue;
      /* if tested & nonconvex, need to append merge */
      neighbor= otherfacet_(ridge, facet);
      if (neighbor->seen) {
        ridge->tested= True;
        ridge->nonconvex= False;
      }else if (neighbor->visitid != qh visit_id) {
        ridge->tested= True;
        ridge->nonconvex= False;
        neighbor->seen= True;      /* only one ridge is marked nonconvex */
        if (qh_test_appendmerge(facet, neighbor))
          ridge->nonconvex= True;
      }
    }
  }
  nummerges= qh_setsize(qh facet_mergeset);
  if (qh ANGLEmerge)
    qsort(SETaddr_(qh facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compareangle);
  else
    qsort(SETaddr_(qh facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_comparemerge);
  if (qh POSTmerging) {
    zadd_(Zmergesettot2, nummerges);
  }else {
    zadd_(Zmergesettot, nummerges);
    zmax_(Zmergesetmax, nummerges);
  }
  trace2((qh ferr, 2021, "qh_getmergeset: %d merges found\n", nummerges));
} /* getmergeset */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="getmergeset_initial">-</a>

  qh_getmergeset_initial( facetlist )
    determine initial qh.facet_mergeset for facets
    tests all facet/neighbor pairs on facetlist

  returns:
    sorted qh.facet_mergeset with nonconvex ridges
    sets facet->tested, ridge->tested, and ridge->nonconvex

  notes:
    uses visit_id, assumes ridge->nonconvex is False

  see:
    qh_getmergeset()

  design:
    for each facet on facetlist
      for each untested neighbor of facet
        test facet and neighbor for convexity
        if non-convex
          append merge to qh.facet_mergeset
          mark one of the ridges as nonconvex
    sort qh.facet_mergeset by angle
*/
void qh_getmergeset_initial(facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int nummerges;

  qh visit_id++;
  FORALLfacet_(facetlist) {
    facet->visitid= qh visit_id;
    facet->tested= True;
    FOREACHneighbor_(facet) {
      if (neighbor->visitid != qh visit_id) {
        if (qh_test_appendmerge(facet, neighbor)) {
          FOREACHridge_(neighbor->ridges) {
            if (facet == otherfacet_(ridge, neighbor)) {
              ridge->nonconvex= True;
              break;    /* only one ridge is marked nonconvex */
            }
          }
        }
      }
    }
    FOREACHridge_(facet->ridges)
      ridge->tested= True;
  }
  nummerges= qh_setsize(qh facet_mergeset);
  if (qh ANGLEmerge)
    qsort(SETaddr_(qh facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compareangle);
  else
    qsort(SETaddr_(qh facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_comparemerge);
  if (qh POSTmerging) {
    zadd_(Zmergeinittot2, nummerges);
  }else {
    zadd_(Zmergeinittot, nummerges);
    zmax_(Zmergeinitmax, nummerges);
  }
  trace2((qh ferr, 2022, "qh_getmergeset_initial: %d merges found\n", nummerges));
} /* getmergeset_initial */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="hashridge">-</a>

  qh_hashridge( hashtable, hashsize, ridge, oldvertex )
    add ridge to hashtable without oldvertex

  notes:
    assumes hashtable is large enough

  design:
    determine hash value for ridge without oldvertex
    find next empty slot for ridge
*/
void qh_hashridge(setT *hashtable, int hashsize, ridgeT *ridge, vertexT *oldvertex) {
  int hash;
  ridgeT *ridgeA;

  hash= qh_gethash(hashsize, ridge->vertices, qh hull_dim-1, 0, oldvertex);
  while (True) {
    if (!(ridgeA= SETelemt_(hashtable, hash, ridgeT))) {
      SETelem_(hashtable, hash)= ridge;
      break;
    }else if (ridgeA == ridge)
      break;
    if (++hash == hashsize)
      hash= 0;
  }
} /* hashridge */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="hashridge_find">-</a>

  qh_hashridge_find( hashtable, hashsize, ridge, vertex, oldvertex, hashslot )
    returns matching ridge without oldvertex in hashtable
      for ridge without vertex
    if oldvertex is NULL
      matches with any one skip

  returns:
    matching ridge or NULL
    if no match,
      if ridge already in   table
        hashslot= -1
      else
        hashslot= next NULL index

  notes:
    assumes hashtable is large enough
    can't match ridge to itself

  design:
    get hash value for ridge without vertex
    for each hashslot
      return match if ridge matches ridgeA without oldvertex
*/
ridgeT *qh_hashridge_find(setT *hashtable, int hashsize, ridgeT *ridge,
              vertexT *vertex, vertexT *oldvertex, int *hashslot) {
  int hash;
  ridgeT *ridgeA;

  *hashslot= 0;
  zinc_(Zhashridge);
  hash= qh_gethash(hashsize, ridge->vertices, qh hull_dim-1, 0, vertex);
  while ((ridgeA= SETelemt_(hashtable, hash, ridgeT))) {
    if (ridgeA == ridge)
      *hashslot= -1;
    else {
      zinc_(Zhashridgetest);
      if (qh_setequal_except(ridge->vertices, vertex, ridgeA->vertices, oldvertex))
        return ridgeA;
    }
    if (++hash == hashsize)
      hash= 0;
  }
  if (!*hashslot)
    *hashslot= hash;
  return NULL;
} /* hashridge_find */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="makeridges">-</a>

  qh_makeridges( facet )
    creates explicit ridges between simplicial facets

  returns:
    facet with ridges and without qh_MERGEridge
    ->simplicial is False

  notes:
    allows qh_MERGEridge flag
    uses existing ridges
    duplicate neighbors ok if ridges already exist (qh_mergecycle_ridges)

  see:
    qh_mergecycle_ridges()

  design:
    look for qh_MERGEridge neighbors
    mark neighbors that already have ridges
    for each unprocessed neighbor of facet
      create a ridge for neighbor and facet
    if any qh_MERGEridge neighbors
      delete qh_MERGEridge flags (already handled by qh_mark_dupridges)
*/
void qh_makeridges(facetT *facet) {
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int neighbor_i, neighbor_n;
  boolT toporient, mergeridge= False;

  if (!facet->simplicial)
    return;
  trace4((qh ferr, 4027, "qh_makeridges: make ridges for f%d\n", facet->id));
  facet->simplicial= False;
  FOREACHneighbor_(facet) {
    if (neighbor == qh_MERGEridge)
      mergeridge= True;
    else
      neighbor->seen= False;
  }
  FOREACHridge_(facet->ridges)
    otherfacet_(ridge, facet)->seen= True;
  FOREACHneighbor_i_(facet) {
    if (neighbor == qh_MERGEridge)
      continue;  /* fixed by qh_mark_dupridges */
    else if (!neighbor->seen) {  /* no current ridges */
      ridge= qh_newridge();
      ridge->vertices= qh_setnew_delnthsorted(facet->vertices, qh hull_dim,
                                                          neighbor_i, 0);
      toporient= facet->toporient ^ (neighbor_i & 0x1);
      if (toporient) {
        ridge->top= facet;
        ridge->bottom= neighbor;
      }else {
        ridge->top= neighbor;
        ridge->bottom= facet;
      }
#if 0 /* this also works */
      flip= (facet->toporient ^ neighbor->toporient)^(skip1 & 0x1) ^ (skip2 & 0x1);
      if (facet->toporient ^ (skip1 & 0x1) ^ flip) {
        ridge->top= neighbor;
        ridge->bottom= facet;
      }else {
        ridge->top= facet;
        ridge->bottom= neighbor;
      }
#endif
      qh_setappend(&(facet->ridges), ridge);
      qh_setappend(&(neighbor->ridges), ridge);
    }
  }
  if (mergeridge) {
    while (qh_setdel(facet->neighbors, qh_MERGEridge))
      ; /* delete each one */
  }
} /* makeridges */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mark_dupridges">-</a>

  qh_mark_dupridges( facetlist )
    add duplicated ridges to qh.facet_mergeset
    facet->dupridge is true

  returns:
    duplicate ridges on qh.facet_mergeset
    ->mergeridge/->mergeridge2 set
    duplicate ridges marked by qh_MERGEridge and both sides facet->dupridge
    no MERGEridges in neighbor sets

  notes:
    duplicate ridges occur when the horizon is pinched,
        i.e. a subridge occurs in more than two horizon ridges.
    could rename vertices that pinch the horizon
    uses qh.visit_id

  design:
    for all facets on facetlist
      if facet contains a duplicate ridge
        for each neighbor of facet
          if neighbor marked qh_MERGEridge (one side of the merge)
            set facet->mergeridge
          else
            if neighbor contains a duplicate ridge
            and the back link is qh_MERGEridge
              append duplicate ridge to qh.facet_mergeset
   for each duplicate ridge
     make ridge sets in preparation for merging
     remove qh_MERGEridge from neighbor set
   for each duplicate ridge
     restore the missing neighbor from the neighbor set that was qh_MERGEridge
     add the missing ridge for this neighbor
*/
void qh_mark_dupridges(facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  int nummerge=0;
  mergeT *merge, **mergep;


  trace4((qh ferr, 4028, "qh_mark_dupridges: identify duplicate ridges\n"));
  FORALLfacet_(facetlist) {
    if (facet->dupridge) {
      FOREACHneighbor_(facet) {
        if (neighbor == qh_MERGEridge) {
          facet->mergeridge= True;
          continue;
        }
        if (neighbor->dupridge
        && !qh_setin(neighbor->neighbors, facet)) { /* qh_MERGEridge */
          qh_appendmergeset(facet, neighbor, MRGridge, NULL);
          facet->mergeridge2= True;
          facet->mergeridge= True;
          nummerge++;
        }
      }
    }
  }
  if (!nummerge)
    return;
  FORALLfacet_(facetlist) {            /* gets rid of qh_MERGEridge */
    if (facet->mergeridge && !facet->mergeridge2)
      qh_makeridges(facet);
  }
  FOREACHmerge_(qh facet_mergeset) {   /* restore the missing neighbors */
    if (merge->type == MRGridge) {
      qh_setappend(&merge->facet2->neighbors, merge->facet1);
      qh_makeridges(merge->facet1);   /* and the missing ridges */
    }
  }
  trace1((qh ferr, 1012, "qh_mark_dupridges: found %d duplicated ridges\n",
                nummerge));
} /* mark_dupridges */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="maydropneighbor">-</a>

  qh_maydropneighbor( facet )
    drop neighbor relationship if no ridge between facet and neighbor

  returns:
    neighbor sets updated
    appends degenerate facets to qh.facet_mergeset

  notes:
    won't cause redundant facets since vertex inclusion is the same
    may drop vertex and neighbor if no ridge
    uses qh.visit_id

  design:
    visit all neighbors with ridges
    for each unvisited neighbor of facet
      delete neighbor and facet from the neighbor sets
      if neighbor becomes degenerate
        append neighbor to qh.degen_mergeset
    if facet is degenerate
      append facet to qh.degen_mergeset
*/
void qh_maydropneighbor(facetT *facet) {
  ridgeT *ridge, **ridgep;
  realT angledegen= qh_ANGLEdegen;
  facetT *neighbor, **neighborp;

  qh visit_id++;
  trace4((qh ferr, 4029, "qh_maydropneighbor: test f%d for no ridges to a neighbor\n",
          facet->id));
  FOREACHridge_(facet->ridges) {
    ridge->top->visitid= qh visit_id;
    ridge->bottom->visitid= qh visit_id;
  }
  FOREACHneighbor_(facet) {
    if (neighbor->visitid != qh visit_id) {
      trace0((qh ferr, 17, "qh_maydropneighbor: facets f%d and f%d are no longer neighbors during p%d\n",
            facet->id, neighbor->id, qh furthest_id));
      zinc_(Zdropneighbor);
      qh_setdel(facet->neighbors, neighbor);
      neighborp--;  /* repeat, deleted a neighbor */
      qh_setdel(neighbor->neighbors, facet);
      if (qh_setsize(neighbor->neighbors) < qh hull_dim) {
        zinc_(Zdropdegen);
        qh_appendmergeset(neighbor, neighbor, MRGdegen, &angledegen);
        trace2((qh ferr, 2023, "qh_maydropneighbors: f%d is degenerate.\n", neighbor->id));
      }
    }
  }
  if (qh_setsize(facet->neighbors) < qh hull_dim) {
    zinc_(Zdropdegen);
    qh_appendmergeset(facet, facet, MRGdegen, &angledegen);
    trace2((qh ferr, 2024, "qh_maydropneighbors: f%d is degenerate.\n", facet->id));
  }
} /* maydropneighbor */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="merge_degenredundant">-</a>

  qh_merge_degenredundant()
    merge all degenerate and redundant facets
    qh.degen_mergeset contains merges from qh_degen_redundant_neighbors()

  returns:
    number of merges performed
    resets facet->degenerate/redundant
    if deleted (visible) facet has no neighbors
      sets ->f.replace to NULL

  notes:
    redundant merges happen before degenerate ones
    merging and renaming vertices can result in degen/redundant facets

  design:
    for each merge on qh.degen_mergeset
      if redundant merge
        if non-redundant facet merged into redundant facet
          recheck facet for redundancy
        else
          merge redundant facet into other facet
*/
int qh_merge_degenredundant(void) {
  int size;
  mergeT *merge;
  facetT *bestneighbor, *facet1, *facet2;
  realT dist, mindist, maxdist;
  vertexT *vertex, **vertexp;
  int nummerges= 0;
  mergeType mergetype;

  while ((merge= (mergeT*)qh_setdellast(qh degen_mergeset))) {
    facet1= merge->facet1;
    facet2= merge->facet2;
    mergetype= merge->type;
    qh_memfree(merge, (int)sizeof(mergeT));
    if (facet1->visible)
      continue;
    facet1->degenerate= False;
    facet1->redundant= False;
    if (qh TRACEmerge-1 == zzval_(Ztotmerge))
      qhmem.IStracing= qh IStracing= qh TRACElevel;
    if (mergetype == MRGredundant) {
      zinc_(Zneighbor);
      while (facet2->visible) {
        if (!facet2->f.replace) {
          qh_fprintf(qh ferr, 6097, "qhull internal error (qh_merge_degenredunant): f%d redundant but f%d has no replacement\n",
               facet1->id, facet2->id);
          qh_errexit2 (qh_ERRqhull, facet1, facet2);
        }
        facet2= facet2->f.replace;
      }
      if (facet1 == facet2) {
        qh_degen_redundant_facet(facet1); /* in case of others */
        continue;
      }
      trace2((qh ferr, 2025, "qh_merge_degenredundant: facet f%d is contained in f%d, will merge\n",
            facet1->id, facet2->id));
      qh_mergefacet(facet1, facet2, NULL, NULL, !qh_MERGEapex);
      /* merge distance is already accounted for */
      nummerges++;
    }else {  /* mergetype == MRGdegen, other merges may have fixed */
      if (!(size= qh_setsize(facet1->neighbors))) {
        zinc_(Zdelfacetdup);
        trace2((qh ferr, 2026, "qh_merge_degenredundant: facet f%d has no neighbors.  Deleted\n", facet1->id));
        qh_willdelete(facet1, NULL);
        FOREACHvertex_(facet1->vertices) {
          qh_setdel(vertex->neighbors, facet1);
          if (!SETfirst_(vertex->neighbors)) {
            zinc_(Zdegenvertex);
            trace2((qh ferr, 2027, "qh_merge_degenredundant: deleted v%d because f%d has no neighbors\n",
                 vertex->id, facet1->id));
            vertex->deleted= True;
            qh_setappend(&qh del_vertices, vertex);
          }
        }
        nummerges++;
      }else if (size < qh hull_dim) {
        bestneighbor= qh_findbestneighbor(facet1, &dist, &mindist, &maxdist);
        trace2((qh ferr, 2028, "qh_merge_degenredundant: facet f%d has %d neighbors, merge into f%d dist %2.2g\n",
              facet1->id, size, bestneighbor->id, dist));
        qh_mergefacet(facet1, bestneighbor, &mindist, &maxdist, !qh_MERGEapex);
        nummerges++;
        if (qh PRINTstatistics) {
          zinc_(Zdegen);
          wadd_(Wdegentot, dist);
          wmax_(Wdegenmax, dist);
        }
      } /* else, another merge fixed the degeneracy and redundancy tested */
    }
  }
  return nummerges;
} /* merge_degenredundant */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="merge_nonconvex">-</a>

  qh_merge_nonconvex( facet1, facet2, mergetype )
    remove non-convex ridge between facet1 into facet2
    mergetype gives why the facet's are non-convex

  returns:
    merges one of the facets into the best neighbor

  design:
    if one of the facets is a new facet
      prefer merging new facet into old facet
    find best neighbors for both facets
    merge the nearest facet into its best neighbor
    update the statistics
*/
void qh_merge_nonconvex(facetT *facet1, facetT *facet2, mergeType mergetype) {
  facetT *bestfacet, *bestneighbor, *neighbor;
  realT dist, dist2, mindist, mindist2, maxdist, maxdist2;

  if (qh TRACEmerge-1 == zzval_(Ztotmerge))
    qhmem.IStracing= qh IStracing= qh TRACElevel;
  trace3((qh ferr, 3003, "qh_merge_nonconvex: merge #%d for f%d and f%d type %d\n",
      zzval_(Ztotmerge) + 1, facet1->id, facet2->id, mergetype));
  /* concave or coplanar */
  if (!facet1->newfacet) {
    bestfacet= facet2;   /* avoid merging old facet if new is ok */
    facet2= facet1;
    facet1= bestfacet;
  }else
    bestfacet= facet1;
  bestneighbor= qh_findbestneighbor(bestfacet, &dist, &mindist, &maxdist);
  neighbor= qh_findbestneighbor(facet2, &dist2, &mindist2, &maxdist2);
  if (dist < dist2) {
    qh_mergefacet(bestfacet, bestneighbor, &mindist, &maxdist, !qh_MERGEapex);
  }else if (qh AVOIDold && !facet2->newfacet
  && ((mindist >= -qh MAXcoplanar && maxdist <= qh max_outside)
       || dist * 1.5 < dist2)) {
    zinc_(Zavoidold);
    wadd_(Wavoidoldtot, dist);
    wmax_(Wavoidoldmax, dist);
    trace2((qh ferr, 2029, "qh_merge_nonconvex: avoid merging old facet f%d dist %2.2g.  Use f%d dist %2.2g instead\n",
           facet2->id, dist2, facet1->id, dist2));
    qh_mergefacet(bestfacet, bestneighbor, &mindist, &maxdist, !qh_MERGEapex);
  }else {
    qh_mergefacet(facet2, neighbor, &mindist2, &maxdist2, !qh_MERGEapex);
    dist= dist2;
  }
  if (qh PRINTstatistics) {
    if (mergetype == MRGanglecoplanar) {
      zinc_(Zacoplanar);
      wadd_(Wacoplanartot, dist);
      wmax_(Wacoplanarmax, dist);
    }else if (mergetype == MRGconcave) {
      zinc_(Zconcave);
      wadd_(Wconcavetot, dist);
      wmax_(Wconcavemax, dist);
    }else { /* MRGcoplanar */
      zinc_(Zcoplanar);
      wadd_(Wcoplanartot, dist);
      wmax_(Wcoplanarmax, dist);
    }
  }
} /* merge_nonconvex */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergecycle">-</a>

  qh_mergecycle( samecycle, newfacet )
    merge a cycle of facets starting at samecycle into a newfacet
    newfacet is a horizon facet with ->normal
    samecycle facets are simplicial from an apex

  returns:
    initializes vertex neighbors on first merge
    samecycle deleted (placed on qh.visible_list)
    newfacet at end of qh.facet_list
    deleted vertices on qh.del_vertices

  see:
    qh_mergefacet()
    called by qh_mergecycle_all() for multiple, same cycle facets

  design:
    make vertex neighbors if necessary
    make ridges for newfacet
    merge neighbor sets of samecycle into newfacet
    merge ridges of samecycle into newfacet
    merge vertex neighbors of samecycle into newfacet
    make apex of samecycle the apex of newfacet
    if newfacet wasn't a new facet
      add its vertices to qh.newvertex_list
    delete samecycle facets a make newfacet a newfacet
*/
void qh_mergecycle(facetT *samecycle, facetT *newfacet) {
  int traceonce= False, tracerestore= 0;
  vertexT *apex;
#ifndef qh_NOtrace
  facetT *same;
#endif

  if (newfacet->tricoplanar) {
    if (!qh TRInormals) {
      qh_fprintf(qh ferr, 6224, "Qhull internal error (qh_mergecycle): does not work for tricoplanar facets.  Use option 'Q11'\n");
      qh_errexit(qh_ERRqhull, newfacet, NULL);
    }
    newfacet->tricoplanar= False;
    newfacet->keepcentrum= False;
  }
  if (!qh VERTEXneighbors)
    qh_vertexneighbors();
  zzinc_(Ztotmerge);
  if (qh REPORTfreq2 && qh POSTmerging) {
    if (zzval_(Ztotmerge) > qh mergereport + qh REPORTfreq2)
      qh_tracemerging();
  }
#ifndef qh_NOtrace
  if (qh TRACEmerge == zzval_(Ztotmerge))
    qhmem.IStracing= qh IStracing= qh TRACElevel;
  trace2((qh ferr, 2030, "qh_mergecycle: merge #%d for facets from cycle f%d into coplanar horizon f%d\n",
        zzval_(Ztotmerge), samecycle->id, newfacet->id));
  if (newfacet == qh tracefacet) {
    tracerestore= qh IStracing;
    qh IStracing= 4;
    qh_fprintf(qh ferr, 8068, "qh_mergecycle: ========= trace merge %d of samecycle %d into trace f%d, furthest is p%d\n",
               zzval_(Ztotmerge), samecycle->id, newfacet->id,  qh furthest_id);
    traceonce= True;
  }
  if (qh IStracing >=4) {
    qh_fprintf(qh ferr, 8069, "  same cycle:");
    FORALLsame_cycle_(samecycle)
      qh_fprintf(qh ferr, 8070, " f%d", same->id);
    qh_fprintf(qh ferr, 8071, "\n");
  }
  if (qh IStracing >=4)
    qh_errprint("MERGING CYCLE", samecycle, newfacet, NULL, NULL);
#endif /* !qh_NOtrace */
  apex= SETfirstt_(samecycle->vertices, vertexT);
  qh_makeridges(newfacet);
  qh_mergecycle_neighbors(samecycle, newfacet);
  qh_mergecycle_ridges(samecycle, newfacet);
  qh_mergecycle_vneighbors(samecycle, newfacet);
  if (SETfirstt_(newfacet->vertices, vertexT) != apex)
    qh_setaddnth(&newfacet->vertices, 0, apex);  /* apex has last id */
  if (!newfacet->newfacet)
    qh_newvertices(newfacet->vertices);
  qh_mergecycle_facets(samecycle, newfacet);
  qh_tracemerge(samecycle, newfacet);
  /* check for degen_redundant_neighbors after qh_forcedmerges() */
  if (traceonce) {
    qh_fprintf(qh ferr, 8072, "qh_mergecycle: end of trace facet\n");
    qh IStracing= tracerestore;
  }
} /* mergecycle */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergecycle_all">-</a>

  qh_mergecycle_all( facetlist, wasmerge )
    merge all samecycles of coplanar facets into horizon
    don't merge facets with ->mergeridge (these already have ->normal)
    all facets are simplicial from apex
    all facet->cycledone == False

  returns:
    all newfacets merged into coplanar horizon facets
    deleted vertices on  qh.del_vertices
    sets wasmerge if any merge

  see:
    calls qh_mergecycle for multiple, same cycle facets

  design:
    for each facet on facetlist
      skip facets with duplicate ridges and normals
      check that facet is in a samecycle (->mergehorizon)
      if facet only member of samecycle
        sets vertex->delridge for all vertices except apex
        merge facet into horizon
      else
        mark all facets in samecycle
        remove facets with duplicate ridges from samecycle
        merge samecycle into horizon (deletes facets from facetlist)
*/
void qh_mergecycle_all(facetT *facetlist, boolT *wasmerge) {
  facetT *facet, *same, *prev, *horizon;
  facetT *samecycle= NULL, *nextfacet, *nextsame;
  vertexT *apex, *vertex, **vertexp;
  int cycles=0, total=0, facets, nummerge;

  trace2((qh ferr, 2031, "qh_mergecycle_all: begin\n"));
  for (facet= facetlist; facet && (nextfacet= facet->next); facet= nextfacet) {
    if (facet->normal)
      continue;
    if (!facet->mergehorizon) {
      qh_fprintf(qh ferr, 6225, "Qhull internal error (qh_mergecycle_all): f%d without normal\n", facet->id);
      qh_errexit(qh_ERRqhull, facet, NULL);
    }
    horizon= SETfirstt_(facet->neighbors, facetT);
    if (facet->f.samecycle == facet) {
      zinc_(Zonehorizon);
      /* merge distance done in qh_findhorizon */
      apex= SETfirstt_(facet->vertices, vertexT);
      FOREACHvertex_(facet->vertices) {
        if (vertex != apex)
          vertex->delridge= True;
      }
      horizon->f.newcycle= NULL;
      qh_mergefacet(facet, horizon, NULL, NULL, qh_MERGEapex);
    }else {
      samecycle= facet;
      facets= 0;
      prev= facet;
      for (same= facet->f.samecycle; same;  /* FORALLsame_cycle_(facet) */
           same= (same == facet ? NULL :nextsame)) { /* ends at facet */
        nextsame= same->f.samecycle;
        if (same->cycledone || same->visible)
          qh_infiniteloop(same);
        same->cycledone= True;
        if (same->normal) {
          prev->f.samecycle= same->f.samecycle; /* unlink ->mergeridge */
          same->f.samecycle= NULL;
        }else {
          prev= same;
          facets++;
        }
      }
      while (nextfacet && nextfacet->cycledone)  /* will delete samecycle */
        nextfacet= nextfacet->next;
      horizon->f.newcycle= NULL;
      qh_mergecycle(samecycle, horizon);
      nummerge= horizon->nummerge + facets;
      if (nummerge > qh_MAXnummerge)
        horizon->nummerge= qh_MAXnummerge;
      else
        horizon->nummerge= (short unsigned int)nummerge;
      zzinc_(Zcyclehorizon);
      total += facets;
      zzadd_(Zcyclefacettot, facets);
      zmax_(Zcyclefacetmax, facets);
    }
    cycles++;
  }
  if (cycles)
    *wasmerge= True;
  trace1((qh ferr, 1013, "qh_mergecycle_all: merged %d same cycles or facets into coplanar horizons\n", cycles));
} /* mergecycle_all */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergecycle_facets">-</a>

  qh_mergecycle_facets( samecycle, newfacet )
    finish merge of samecycle into newfacet

  returns:
    samecycle prepended to visible_list for later deletion and partitioning
      each facet->f.replace == newfacet

    newfacet moved to end of qh.facet_list
      makes newfacet a newfacet (get's facet1->id if it was old)
      sets newfacet->newmerge
      clears newfacet->center (unless merging into a large facet)
      clears newfacet->tested and ridge->tested for facet1

    adds neighboring facets to facet_mergeset if redundant or degenerate

  design:
    make newfacet a new facet and set its flags
    move samecycle facets to qh.visible_list for later deletion
    unless newfacet is large
      remove its centrum
*/
void qh_mergecycle_facets(facetT *samecycle, facetT *newfacet) {
  facetT *same, *next;

  trace4((qh ferr, 4030, "qh_mergecycle_facets: make newfacet new and samecycle deleted\n"));
  qh_removefacet(newfacet);  /* append as a newfacet to end of qh facet_list */
  qh_appendfacet(newfacet);
  newfacet->newfacet= True;
  newfacet->simplicial= False;
  newfacet->newmerge= True;

  for (same= samecycle->f.samecycle; same; same= (same == samecycle ?  NULL : next)) {
    next= same->f.samecycle;  /* reused by willdelete */
    qh_willdelete(same, newfacet);
  }
  if (newfacet->center
      && qh_setsize(newfacet->vertices) <= qh hull_dim + qh_MAXnewcentrum) {
    qh_memfree(newfacet->center, qh normal_size);
    newfacet->center= NULL;
  }
  trace3((qh ferr, 3004, "qh_mergecycle_facets: merged facets from cycle f%d into f%d\n",
             samecycle->id, newfacet->id));
} /* mergecycle_facets */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergecycle_neighbors">-</a>

  qh_mergecycle_neighbors( samecycle, newfacet )
    add neighbors for samecycle facets to newfacet

  returns:
    newfacet with updated neighbors and vice-versa
    newfacet has ridges
    all neighbors of newfacet marked with qh.visit_id
    samecycle facets marked with qh.visit_id-1
    ridges updated for simplicial neighbors of samecycle with a ridge

  notes:
    assumes newfacet not in samecycle
    usually, samecycle facets are new, simplicial facets without internal ridges
      not so if horizon facet is coplanar to two different samecycles

  see:
    qh_mergeneighbors()

  design:
    check samecycle
    delete neighbors from newfacet that are also in samecycle
    for each neighbor of a facet in samecycle
      if neighbor is simplicial
        if first visit
          move the neighbor relation to newfacet
          update facet links for its ridges
        else
          make ridges for neighbor
          remove samecycle reference
      else
        update neighbor sets
*/
void qh_mergecycle_neighbors(facetT *samecycle, facetT *newfacet) {
  facetT *same, *neighbor, **neighborp;
  int delneighbors= 0, newneighbors= 0;
  unsigned int samevisitid;
  ridgeT *ridge, **ridgep;

  samevisitid= ++qh visit_id;
  FORALLsame_cycle_(samecycle) {
    if (same->visitid == samevisitid || same->visible)
      qh_infiniteloop(samecycle);
    same->visitid= samevisitid;
  }
  newfacet->visitid= ++qh visit_id;
  trace4((qh ferr, 4031, "qh_mergecycle_neighbors: delete shared neighbors from newfacet\n"));
  FOREACHneighbor_(newfacet) {
    if (neighbor->visitid == samevisitid) {
      SETref_(neighbor)= NULL;  /* samecycle neighbors deleted */
      delneighbors++;
    }else
      neighbor->visitid= qh visit_id;
  }
  qh_setcompact(newfacet->neighbors);

  trace4((qh ferr, 4032, "qh_mergecycle_neighbors: update neighbors\n"));
  FORALLsame_cycle_(samecycle) {
    FOREACHneighbor_(same) {
      if (neighbor->visitid == samevisitid)
        continue;
      if (neighbor->simplicial) {
        if (neighbor->visitid != qh visit_id) {
          qh_setappend(&newfacet->neighbors, neighbor);
          qh_setreplace(neighbor->neighbors, same, newfacet);
          newneighbors++;
          neighbor->visitid= qh visit_id;
          FOREACHridge_(neighbor->ridges) { /* update ridge in case of qh_makeridges */
            if (ridge->top == same) {
              ridge->top= newfacet;
              break;
            }else if (ridge->bottom == same) {
              ridge->bottom= newfacet;
              break;
            }
          }
        }else {
          qh_makeridges(neighbor);
          qh_setdel(neighbor->neighbors, same);
          /* same can't be horizon facet for neighbor */
        }
      }else { /* non-simplicial neighbor */
        qh_setdel(neighbor->neighbors, same);
        if (neighbor->visitid != qh visit_id) {
          qh_setappend(&neighbor->neighbors, newfacet);
          qh_setappend(&newfacet->neighbors, neighbor);
          neighbor->visitid= qh visit_id;
          newneighbors++;
        }
      }
    }
  }
  trace2((qh ferr, 2032, "qh_mergecycle_neighbors: deleted %d neighbors and added %d\n",
             delneighbors, newneighbors));
} /* mergecycle_neighbors */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergecycle_ridges">-</a>

  qh_mergecycle_ridges( samecycle, newfacet )
    add ridges/neighbors for facets in samecycle to newfacet
    all new/old neighbors of newfacet marked with qh.visit_id
    facets in samecycle marked with qh.visit_id-1
    newfacet marked with qh.visit_id

  returns:
    newfacet has merged ridges

  notes:
    ridge already updated for simplicial neighbors of samecycle with a ridge

  see:
    qh_mergeridges()
    qh_makeridges()

  design:
    remove ridges between newfacet and samecycle
    for each facet in samecycle
      for each ridge in facet
        update facet pointers in ridge
        skip ridges processed in qh_mergecycle_neighors
        free ridges between newfacet and samecycle
        free ridges between facets of samecycle (on 2nd visit)
        append remaining ridges to newfacet
      if simpilicial facet
        for each neighbor of facet
          if simplicial facet
          and not samecycle facet or newfacet
            make ridge between neighbor and newfacet
*/
void qh_mergecycle_ridges(facetT *samecycle, facetT *newfacet) {
  facetT *same, *neighbor= NULL;
  int numold=0, numnew=0;
  int neighbor_i, neighbor_n;
  unsigned int samevisitid;
  ridgeT *ridge, **ridgep;
  boolT toporient;
  void **freelistp; /* used !qh_NOmem */

  trace4((qh ferr, 4033, "qh_mergecycle_ridges: delete shared ridges from newfacet\n"));
  samevisitid= qh visit_id -1;
  FOREACHridge_(newfacet->ridges) {
    neighbor= otherfacet_(ridge, newfacet);
    if (neighbor->visitid == samevisitid)
      SETref_(ridge)= NULL; /* ridge free'd below */
  }
  qh_setcompact(newfacet->ridges);

  trace4((qh ferr, 4034, "qh_mergecycle_ridges: add ridges to newfacet\n"));
  FORALLsame_cycle_(samecycle) {
    FOREACHridge_(same->ridges) {
      if (ridge->top == same) {
        ridge->top= newfacet;
        neighbor= ridge->bottom;
      }else if (ridge->bottom == same) {
        ridge->bottom= newfacet;
        neighbor= ridge->top;
      }else if (ridge->top == newfacet || ridge->bottom == newfacet) {
        qh_setappend(&newfacet->ridges, ridge);
        numold++;  /* already set by qh_mergecycle_neighbors */
        continue;
      }else {
        qh_fprintf(qh ferr, 6098, "qhull internal error (qh_mergecycle_ridges): bad ridge r%d\n", ridge->id);
        qh_errexit(qh_ERRqhull, NULL, ridge);
      }
      if (neighbor == newfacet) {
        qh_setfree(&(ridge->vertices));
        qh_memfree_(ridge, (int)sizeof(ridgeT), freelistp);
        numold++;
      }else if (neighbor->visitid == samevisitid) {
        qh_setdel(neighbor->ridges, ridge);
        qh_setfree(&(ridge->vertices));
        qh_memfree_(ridge, (int)sizeof(ridgeT), freelistp);
        numold++;
      }else {
        qh_setappend(&newfacet->ridges, ridge);
        numold++;
      }
    }
    if (same->ridges)
      qh_settruncate(same->ridges, 0);
    if (!same->simplicial)
      continue;
    FOREACHneighbor_i_(same) {       /* note: !newfact->simplicial */
      if (neighbor->visitid != samevisitid && neighbor->simplicial) {
        ridge= qh_newridge();
        ridge->vertices= qh_setnew_delnthsorted(same->vertices, qh hull_dim,
                                                          neighbor_i, 0);
        toporient= same->toporient ^ (neighbor_i & 0x1);
        if (toporient) {
          ridge->top= newfacet;
          ridge->bottom= neighbor;
        }else {
          ridge->top= neighbor;
          ridge->bottom= newfacet;
        }
        qh_setappend(&(newfacet->ridges), ridge);
        qh_setappend(&(neighbor->ridges), ridge);
        numnew++;
      }
    }
  }

  trace2((qh ferr, 2033, "qh_mergecycle_ridges: found %d old ridges and %d new ones\n",
             numold, numnew));
} /* mergecycle_ridges */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergecycle_vneighbors">-</a>

  qh_mergecycle_vneighbors( samecycle, newfacet )
    create vertex neighbors for newfacet from vertices of facets in samecycle
    samecycle marked with visitid == qh.visit_id - 1

  returns:
    newfacet vertices with updated neighbors
    marks newfacet with qh.visit_id-1
    deletes vertices that are merged away
    sets delridge on all vertices (faster here than in mergecycle_ridges)

  see:
    qh_mergevertex_neighbors()

  design:
    for each vertex of samecycle facet
      set vertex->delridge
      delete samecycle facets from vertex neighbors
      append newfacet to vertex neighbors
      if vertex only in newfacet
        delete it from newfacet
        add it to qh.del_vertices for later deletion
*/
void qh_mergecycle_vneighbors(facetT *samecycle, facetT *newfacet) {
  facetT *neighbor, **neighborp;
  unsigned int mergeid;
  vertexT *vertex, **vertexp, *apex;
  setT *vertices;

  trace4((qh ferr, 4035, "qh_mergecycle_vneighbors: update vertex neighbors for newfacet\n"));
  mergeid= qh visit_id - 1;
  newfacet->visitid= mergeid;
  vertices= qh_basevertices(samecycle); /* temp */
  apex= SETfirstt_(samecycle->vertices, vertexT);
  qh_setappend(&vertices, apex);
  FOREACHvertex_(vertices) {
    vertex->delridge= True;
    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == mergeid)
        SETref_(neighbor)= NULL;
    }
    qh_setcompact(vertex->neighbors);
    qh_setappend(&vertex->neighbors, newfacet);
    if (!SETsecond_(vertex->neighbors)) {
      zinc_(Zcyclevertex);
      trace2((qh ferr, 2034, "qh_mergecycle_vneighbors: deleted v%d when merging cycle f%d into f%d\n",
        vertex->id, samecycle->id, newfacet->id));
      qh_setdelsorted(newfacet->vertices, vertex);
      vertex->deleted= True;
      qh_setappend(&qh del_vertices, vertex);
    }
  }
  qh_settempfree(&vertices);
  trace3((qh ferr, 3005, "qh_mergecycle_vneighbors: merged vertices from cycle f%d into f%d\n",
             samecycle->id, newfacet->id));
} /* mergecycle_vneighbors */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergefacet">-</a>

  qh_mergefacet( facet1, facet2, mindist, maxdist, mergeapex )
    merges facet1 into facet2
    mergeapex==qh_MERGEapex if merging new facet into coplanar horizon

  returns:
    qh.max_outside and qh.min_vertex updated
    initializes vertex neighbors on first merge

  returns:
    facet2 contains facet1's vertices, neighbors, and ridges
      facet2 moved to end of qh.facet_list
      makes facet2 a newfacet
      sets facet2->newmerge set
      clears facet2->center (unless merging into a large facet)
      clears facet2->tested and ridge->tested for facet1

    facet1 prepended to visible_list for later deletion and partitioning
      facet1->f.replace == facet2

    adds neighboring facets to facet_mergeset if redundant or degenerate

  notes:
    mindist/maxdist may be NULL (only if both NULL)
    traces merge if fmax_(maxdist,-mindist) > TRACEdist

  see:
    qh_mergecycle()

  design:
    trace merge and check for degenerate simplex
    make ridges for both facets
    update qh.max_outside, qh.max_vertex, qh.min_vertex
    update facet2->maxoutside and keepcentrum
    update facet2->nummerge
    update tested flags for facet2
    if facet1 is simplicial
      merge facet1 into facet2
    else
      merge facet1's neighbors into facet2
      merge facet1's ridges into facet2
      merge facet1's vertices into facet2
      merge facet1's vertex neighbors into facet2
      add facet2's vertices to qh.new_vertexlist
      unless qh_MERGEapex
        test facet2 for degenerate or redundant neighbors
      move facet1 to qh.visible_list for later deletion
      move facet2 to end of qh.newfacet_list
*/
void qh_mergefacet(facetT *facet1, facetT *facet2, realT *mindist, realT *maxdist, boolT mergeapex) {
  boolT traceonce= False;
  vertexT *vertex, **vertexp;
  int tracerestore=0, nummerge;

  if (facet1->tricoplanar || facet2->tricoplanar) {
    if (!qh TRInormals) {
      qh_fprintf(qh ferr, 6226, "Qhull internal error (qh_mergefacet): does not work for tricoplanar facets.  Use option 'Q11'\n");
      qh_errexit2 (qh_ERRqhull, facet1, facet2);
    }
    if (facet2->tricoplanar) {
      facet2->tricoplanar= False;
      facet2->keepcentrum= False;
    }
  }
  zzinc_(Ztotmerge);
  if (qh REPORTfreq2 && qh POSTmerging) {
    if (zzval_(Ztotmerge) > qh mergereport + qh REPORTfreq2)
      qh_tracemerging();
  }
#ifndef qh_NOtrace
  if (qh build_cnt >= qh RERUN) {
    if (mindist && (-*mindist > qh TRACEdist || *maxdist > qh TRACEdist)) {
      tracerestore= 0;
      qh IStracing= qh TRACElevel;
      traceonce= True;
      qh_fprintf(qh ferr, 8075, "qh_mergefacet: ========= trace wide merge #%d(%2.2g) for f%d into f%d, last point was p%d\n", zzval_(Ztotmerge),
             fmax_(-*mindist, *maxdist), facet1->id, facet2->id, qh furthest_id);
    }else if (facet1 == qh tracefacet || facet2 == qh tracefacet) {
      tracerestore= qh IStracing;
      qh IStracing= 4;
      traceonce= True;
      qh_fprintf(qh ferr, 8076, "qh_mergefacet: ========= trace merge #%d involving f%d, furthest is p%d\n",
                 zzval_(Ztotmerge), qh tracefacet_id,  qh furthest_id);
    }
  }
  if (qh IStracing >= 2) {
    realT mergemin= -2;
    realT mergemax= -2;

    if (mindist) {
      mergemin= *mindist;
      mergemax= *maxdist;
    }
    qh_fprintf(qh ferr, 8077, "qh_mergefacet: #%d merge f%d into f%d, mindist= %2.2g, maxdist= %2.2g\n",
    zzval_(Ztotmerge), facet1->id, facet2->id, mergemin, mergemax);
  }
#endif /* !qh_NOtrace */
  if (facet1 == facet2 || facet1->visible || facet2->visible) {
    qh_fprintf(qh ferr, 6099, "qhull internal error (qh_mergefacet): either f%d and f%d are the same or one is a visible facet\n",
             facet1->id, facet2->id);
    qh_errexit2 (qh_ERRqhull, facet1, facet2);
  }
  if (qh num_facets - qh num_visible <= qh hull_dim + 1) {
    qh_fprintf(qh ferr, 6227, "\n\
qhull precision error: Only %d facets remain.  Can not merge another\n\
pair.  The input is too degenerate or the convexity constraints are\n\
too strong.\n", qh hull_dim+1);
    if (qh hull_dim >= 5 && !qh MERGEexact)
      qh_fprintf(qh ferr, 8079, "Option 'Qx' may avoid this problem.\n");
    qh_errexit(qh_ERRprec, NULL, NULL);
  }
  if (!qh VERTEXneighbors)
    qh_vertexneighbors();
  qh_makeridges(facet1);
  qh_makeridges(facet2);
  if (qh IStracing >=4)
    qh_errprint("MERGING", facet1, facet2, NULL, NULL);
  if (mindist) {
    maximize_(qh max_outside, *maxdist);
    maximize_(qh max_vertex, *maxdist);
#if qh_MAXoutside
    maximize_(facet2->maxoutside, *maxdist);
#endif
    minimize_(qh min_vertex, *mindist);
    if (!facet2->keepcentrum
    && (*maxdist > qh WIDEfacet || *mindist < -qh WIDEfacet)) {
      facet2->keepcentrum= True;
      zinc_(Zwidefacet);
    }
  }
  nummerge= facet1->nummerge + facet2->nummerge + 1;
  if (nummerge >= qh_MAXnummerge)
    facet2->nummerge= qh_MAXnummerge;
  else
    facet2->nummerge= (short unsigned int)nummerge;
  facet2->newmerge= True;
  facet2->dupridge= False;
  qh_updatetested  (facet1, facet2);
  if (qh hull_dim > 2 && qh_setsize(facet1->vertices) == qh hull_dim)
    qh_mergesimplex(facet1, facet2, mergeapex);
  else {
    qh vertex_visit++;
    FOREACHvertex_(facet2->vertices)
      vertex->visitid= qh vertex_visit;
    if (qh hull_dim == 2)
      qh_mergefacet2d(facet1, facet2);
    else {
      qh_mergeneighbors(facet1, facet2);
      qh_mergevertices(facet1->vertices, &facet2->vertices);
    }
    qh_mergeridges(facet1, facet2);
    qh_mergevertex_neighbors(facet1, facet2);
    if (!facet2->newfacet)
      qh_newvertices(facet2->vertices);
  }
  if (!mergeapex)
    qh_degen_redundant_neighbors(facet2, facet1);
  if (facet2->coplanar || !facet2->newfacet) {
    zinc_(Zmergeintohorizon);
  }else if (!facet1->newfacet && facet2->newfacet) {
    zinc_(Zmergehorizon);
  }else {
    zinc_(Zmergenew);
  }
  qh_willdelete(facet1, facet2);
  qh_removefacet(facet2);  /* append as a newfacet to end of qh facet_list */
  qh_appendfacet(facet2);
  facet2->newfacet= True;
  facet2->tested= False;
  qh_tracemerge(facet1, facet2);
  if (traceonce) {
    qh_fprintf(qh ferr, 8080, "qh_mergefacet: end of wide tracing\n");
    qh IStracing= tracerestore;
  }
} /* mergefacet */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergefacet2d">-</a>

  qh_mergefacet2d( facet1, facet2 )
    in 2d, merges neighbors and vertices of facet1 into facet2

  returns:
    build ridges for neighbors if necessary
    facet2 looks like a simplicial facet except for centrum, ridges
      neighbors are opposite the corresponding vertex
      maintains orientation of facet2

  notes:
    qh_mergefacet() retains non-simplicial structures
      they are not needed in 2d, but later routines may use them
    preserves qh.vertex_visit for qh_mergevertex_neighbors()

  design:
    get vertices and neighbors
    determine new vertices and neighbors
    set new vertices and neighbors and adjust orientation
    make ridges for new neighbor if needed
*/
void qh_mergefacet2d(facetT *facet1, facetT *facet2) {
  vertexT *vertex1A, *vertex1B, *vertex2A, *vertex2B, *vertexA, *vertexB;
  facetT *neighbor1A, *neighbor1B, *neighbor2A, *neighbor2B, *neighborA, *neighborB;

  vertex1A= SETfirstt_(facet1->vertices, vertexT);
  vertex1B= SETsecondt_(facet1->vertices, vertexT);
  vertex2A= SETfirstt_(facet2->vertices, vertexT);
  vertex2B= SETsecondt_(facet2->vertices, vertexT);
  neighbor1A= SETfirstt_(facet1->neighbors, facetT);
  neighbor1B= SETsecondt_(facet1->neighbors, facetT);
  neighbor2A= SETfirstt_(facet2->neighbors, facetT);
  neighbor2B= SETsecondt_(facet2->neighbors, facetT);
  if (vertex1A == vertex2A) {
    vertexA= vertex1B;
    vertexB= vertex2B;
    neighborA= neighbor2A;
    neighborB= neighbor1A;
  }else if (vertex1A == vertex2B) {
    vertexA= vertex1B;
    vertexB= vertex2A;
    neighborA= neighbor2B;
    neighborB= neighbor1A;
  }else if (vertex1B == vertex2A) {
    vertexA= vertex1A;
    vertexB= vertex2B;
    neighborA= neighbor2A;
    neighborB= neighbor1B;
  }else { /* 1B == 2B */
    vertexA= vertex1A;
    vertexB= vertex2A;
    neighborA= neighbor2B;
    neighborB= neighbor1B;
  }
  /* vertexB always from facet2, neighborB always from facet1 */
  if (vertexA->id > vertexB->id) {
    SETfirst_(facet2->vertices)= vertexA;
    SETsecond_(facet2->vertices)= vertexB;
    if (vertexB == vertex2A)
      facet2->toporient= !facet2->toporient;
    SETfirst_(facet2->neighbors)= neighborA;
    SETsecond_(facet2->neighbors)= neighborB;
  }else {
    SETfirst_(facet2->vertices)= vertexB;
    SETsecond_(facet2->vertices)= vertexA;
    if (vertexB == vertex2B)
      facet2->toporient= !facet2->toporient;
    SETfirst_(facet2->neighbors)= neighborB;
    SETsecond_(facet2->neighbors)= neighborA;
  }
  qh_makeridges(neighborB);
  qh_setreplace(neighborB->neighbors, facet1, facet2);
  trace4((qh ferr, 4036, "qh_mergefacet2d: merged v%d and neighbor f%d of f%d into f%d\n",
       vertexA->id, neighborB->id, facet1->id, facet2->id));
} /* mergefacet2d */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergeneighbors">-</a>

  qh_mergeneighbors( facet1, facet2 )
    merges the neighbors of facet1 into facet2

  see:
    qh_mergecycle_neighbors()

  design:
    for each neighbor of facet1
      if neighbor is also a neighbor of facet2
        if neighbor is simpilicial
          make ridges for later deletion as a degenerate facet
        update its neighbor set
      else
        move the neighbor relation to facet2
    remove the neighbor relation for facet1 and facet2
*/
void qh_mergeneighbors(facetT *facet1, facetT *facet2) {
  facetT *neighbor, **neighborp;

  trace4((qh ferr, 4037, "qh_mergeneighbors: merge neighbors of f%d and f%d\n",
          facet1->id, facet2->id));
  qh visit_id++;
  FOREACHneighbor_(facet2) {
    neighbor->visitid= qh visit_id;
  }
  FOREACHneighbor_(facet1) {
    if (neighbor->visitid == qh visit_id) {
      if (neighbor->simplicial)    /* is degen, needs ridges */
        qh_makeridges(neighbor);
      if (SETfirstt_(neighbor->neighbors, facetT) != facet1) /*keep newfacet->horizon*/
        qh_setdel(neighbor->neighbors, facet1);
      else {
        qh_setdel(neighbor->neighbors, facet2);
        qh_setreplace(neighbor->neighbors, facet1, facet2);
      }
    }else if (neighbor != facet2) {
      qh_setappend(&(facet2->neighbors), neighbor);
      qh_setreplace(neighbor->neighbors, facet1, facet2);
    }
  }
  qh_setdel(facet1->neighbors, facet2);  /* here for makeridges */
  qh_setdel(facet2->neighbors, facet1);
} /* mergeneighbors */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergeridges">-</a>

  qh_mergeridges( facet1, facet2 )
    merges the ridge set of facet1 into facet2

  returns:
    may delete all ridges for a vertex
    sets vertex->delridge on deleted ridges

  see:
    qh_mergecycle_ridges()

  design:
    delete ridges between facet1 and facet2
      mark (delridge) vertices on these ridges for later testing
    for each remaining ridge
      rename facet1 to facet2
*/
void qh_mergeridges(facetT *facet1, facetT *facet2) {
  ridgeT *ridge, **ridgep;
  vertexT *vertex, **vertexp;

  trace4((qh ferr, 4038, "qh_mergeridges: merge ridges of f%d and f%d\n",
          facet1->id, facet2->id));
  FOREACHridge_(facet2->ridges) {
    if ((ridge->top == facet1) || (ridge->bottom == facet1)) {
      FOREACHvertex_(ridge->vertices)
        vertex->delridge= True;
      qh_delridge(ridge);  /* expensive in high-d, could rebuild */
      ridgep--; /*repeat*/
    }
  }
  FOREACHridge_(facet1->ridges) {
    if (ridge->top == facet1)
      ridge->top= facet2;
    else
      ridge->bottom= facet2;
    qh_setappend(&(facet2->ridges), ridge);
  }
} /* mergeridges */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergesimplex">-</a>

  qh_mergesimplex( facet1, facet2, mergeapex )
    merge simplicial facet1 into facet2
    mergeapex==qh_MERGEapex if merging samecycle into horizon facet
      vertex id is latest (most recently created)
    facet1 may be contained in facet2
    ridges exist for both facets

  returns:
    facet2 with updated vertices, ridges, neighbors
    updated neighbors for facet1's vertices
    facet1 not deleted
    sets vertex->delridge on deleted ridges

  notes:
    special case code since this is the most common merge
    called from qh_mergefacet()

  design:
    if qh_MERGEapex
      add vertices of facet2 to qh.new_vertexlist if necessary
      add apex to facet2
    else
      for each ridge between facet1 and facet2
        set vertex->delridge
      determine the apex for facet1 (i.e., vertex to be merged)
      unless apex already in facet2
        insert apex into vertices for facet2
      add vertices of facet2 to qh.new_vertexlist if necessary
      add apex to qh.new_vertexlist if necessary
      for each vertex of facet1
        if apex
          rename facet1 to facet2 in its vertex neighbors
        else
          delete facet1 from vertex neighors
          if only in facet2
            add vertex to qh.del_vertices for later deletion
      for each ridge of facet1
        delete ridges between facet1 and facet2
        append other ridges to facet2 after renaming facet to facet2
*/
void qh_mergesimplex(facetT *facet1, facetT *facet2, boolT mergeapex) {
  vertexT *vertex, **vertexp, *apex;
  ridgeT *ridge, **ridgep;
  boolT issubset= False;
  int vertex_i= -1, vertex_n;
  facetT *neighbor, **neighborp, *otherfacet;

  if (mergeapex) {
    if (!facet2->newfacet)
      qh_newvertices(facet2->vertices);  /* apex is new */
    apex= SETfirstt_(facet1->vertices, vertexT);
    if (SETfirstt_(facet2->vertices, vertexT) != apex)
      qh_setaddnth(&facet2->vertices, 0, apex);  /* apex has last id */
    else
      issubset= True;
  }else {
    zinc_(Zmergesimplex);
    FOREACHvertex_(facet1->vertices)
      vertex->seen= False;
    FOREACHridge_(facet1->ridges) {
      if (otherfacet_(ridge, facet1) == facet2) {
        FOREACHvertex_(ridge->vertices) {
          vertex->seen= True;
          vertex->delridge= True;
        }
        break;
      }
    }
    FOREACHvertex_(facet1->vertices) {
      if (!vertex->seen)
        break;  /* must occur */
    }
    apex= vertex;
    trace4((qh ferr, 4039, "qh_mergesimplex: merge apex v%d of f%d into facet f%d\n",
          apex->id, facet1->id, facet2->id));
    FOREACHvertex_i_(facet2->vertices) {
      if (vertex->id < apex->id) {
        break;
      }else if (vertex->id == apex->id) {
        issubset= True;
        break;
      }
    }
    if (!issubset)
      qh_setaddnth(&facet2->vertices, vertex_i, apex);
    if (!facet2->newfacet)
      qh_newvertices(facet2->vertices);
    else if (!apex->newlist) {
      qh_removevertex(apex);
      qh_appendvertex(apex);
    }
  }
  trace4((qh ferr, 4040, "qh_mergesimplex: update vertex neighbors of f%d\n",
          facet1->id));
  FOREACHvertex_(facet1->vertices) {
    if (vertex == apex && !issubset)
      qh_setreplace(vertex->neighbors, facet1, facet2);
    else {
      qh_setdel(vertex->neighbors, facet1);
      if (!SETsecond_(vertex->neighbors))
        qh_mergevertex_del(vertex, facet1, facet2);
    }
  }
  trace4((qh ferr, 4041, "qh_mergesimplex: merge ridges and neighbors of f%d into f%d\n",
          facet1->id, facet2->id));
  qh visit_id++;
  FOREACHneighbor_(facet2)
    neighbor->visitid= qh visit_id;
  FOREACHridge_(facet1->ridges) {
    otherfacet= otherfacet_(ridge, facet1);
    if (otherfacet == facet2) {
      qh_setdel(facet2->ridges, ridge);
      qh_setfree(&(ridge->vertices));
      qh_memfree(ridge, (int)sizeof(ridgeT));
      qh_setdel(facet2->neighbors, facet1);
    }else {
      qh_setappend(&facet2->ridges, ridge);
      if (otherfacet->visitid != qh visit_id) {
        qh_setappend(&facet2->neighbors, otherfacet);
        qh_setreplace(otherfacet->neighbors, facet1, facet2);
        otherfacet->visitid= qh visit_id;
      }else {
        if (otherfacet->simplicial)    /* is degen, needs ridges */
          qh_makeridges(otherfacet);
        if (SETfirstt_(otherfacet->neighbors, facetT) != facet1)
          qh_setdel(otherfacet->neighbors, facet1);
        else {   /*keep newfacet->neighbors->horizon*/
          qh_setdel(otherfacet->neighbors, facet2);
          qh_setreplace(otherfacet->neighbors, facet1, facet2);
        }
      }
      if (ridge->top == facet1) /* wait until after qh_makeridges */
        ridge->top= facet2;
      else
        ridge->bottom= facet2;
    }
  }
  SETfirst_(facet1->ridges)= NULL; /* it will be deleted */
  trace3((qh ferr, 3006, "qh_mergesimplex: merged simplex f%d apex v%d into facet f%d\n",
          facet1->id, getid_(apex), facet2->id));
} /* mergesimplex */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergevertex_del">-</a>

  qh_mergevertex_del( vertex, facet1, facet2 )
    delete a vertex because of merging facet1 into facet2

  returns:
    deletes vertex from facet2
    adds vertex to qh.del_vertices for later deletion
*/
void qh_mergevertex_del(vertexT *vertex, facetT *facet1, facetT *facet2) {

  zinc_(Zmergevertex);
  trace2((qh ferr, 2035, "qh_mergevertex_del: deleted v%d when merging f%d into f%d\n",
          vertex->id, facet1->id, facet2->id));
  qh_setdelsorted(facet2->vertices, vertex);
  vertex->deleted= True;
  qh_setappend(&qh del_vertices, vertex);
} /* mergevertex_del */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergevertex_neighbors">-</a>

  qh_mergevertex_neighbors( facet1, facet2 )
    merge the vertex neighbors of facet1 to facet2

  returns:
    if vertex is current qh.vertex_visit
      deletes facet1 from vertex->neighbors
    else
      renames facet1 to facet2 in vertex->neighbors
    deletes vertices if only one neighbor

  notes:
    assumes vertex neighbor sets are good
*/
void qh_mergevertex_neighbors(facetT *facet1, facetT *facet2) {
  vertexT *vertex, **vertexp;

  trace4((qh ferr, 4042, "qh_mergevertex_neighbors: merge vertex neighbors of f%d and f%d\n",
          facet1->id, facet2->id));
  if (qh tracevertex) {
    qh_fprintf(qh ferr, 8081, "qh_mergevertex_neighbors: of f%d and f%d at furthest p%d f0= %p\n",
             facet1->id, facet2->id, qh furthest_id, qh tracevertex->neighbors->e[0].p);
    qh_errprint("TRACE", NULL, NULL, NULL, qh tracevertex);
  }
  FOREACHvertex_(facet1->vertices) {
    if (vertex->visitid != qh vertex_visit)
      qh_setreplace(vertex->neighbors, facet1, facet2);
    else {
      qh_setdel(vertex->neighbors, facet1);
      if (!SETsecond_(vertex->neighbors))
        qh_mergevertex_del(vertex, facet1, facet2);
    }
  }
  if (qh tracevertex)
    qh_errprint("TRACE", NULL, NULL, NULL, qh tracevertex);
} /* mergevertex_neighbors */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="mergevertices">-</a>

  qh_mergevertices( vertices1, vertices2 )
    merges the vertex set of facet1 into facet2

  returns:
    replaces vertices2 with merged set
    preserves vertex_visit for qh_mergevertex_neighbors
    updates qh.newvertex_list

  design:
    create a merged set of both vertices (in inverse id order)
*/
void qh_mergevertices(setT *vertices1, setT **vertices2) {
  int newsize= qh_setsize(vertices1)+qh_setsize(*vertices2) - qh hull_dim + 1;
  setT *mergedvertices;
  vertexT *vertex, **vertexp, **vertex2= SETaddr_(*vertices2, vertexT);

  mergedvertices= qh_settemp(newsize);
  FOREACHvertex_(vertices1) {
    if (!*vertex2 || vertex->id > (*vertex2)->id)
      qh_setappend(&mergedvertices, vertex);
    else {
      while (*vertex2 && (*vertex2)->id > vertex->id)
        qh_setappend(&mergedvertices, *vertex2++);
      if (!*vertex2 || (*vertex2)->id < vertex->id)
        qh_setappend(&mergedvertices, vertex);
      else
        qh_setappend(&mergedvertices, *vertex2++);
    }
  }
  while (*vertex2)
    qh_setappend(&mergedvertices, *vertex2++);
  if (newsize < qh_setsize(mergedvertices)) {
    qh_fprintf(qh ferr, 6100, "qhull internal error (qh_mergevertices): facets did not share a ridge\n");
    qh_errexit(qh_ERRqhull, NULL, NULL);
  }
  qh_setfree(vertices2);
  *vertices2= mergedvertices;
  qh_settemppop();
} /* mergevertices */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="neighbor_intersections">-</a>

  qh_neighbor_intersections( vertex )
    return intersection of all vertices in vertex->neighbors except for vertex

  returns:
    returns temporary set of vertices
    does not include vertex
    NULL if a neighbor is simplicial
    NULL if empty set

  notes:
    used for renaming vertices

  design:
    initialize the intersection set with vertices of the first two neighbors
    delete vertex from the intersection
    for each remaining neighbor
      intersect its vertex set with the intersection set
      return NULL if empty
    return the intersection set
*/
setT *qh_neighbor_intersections(vertexT *vertex) {
  facetT *neighbor, **neighborp, *neighborA, *neighborB;
  setT *intersect;
  int neighbor_i, neighbor_n;

  FOREACHneighbor_(vertex) {
    if (neighbor->simplicial)
      return NULL;
  }
  neighborA= SETfirstt_(vertex->neighbors, facetT);
  neighborB= SETsecondt_(vertex->neighbors, facetT);
  zinc_(Zintersectnum);
  if (!neighborA)
    return NULL;
  if (!neighborB)
    intersect= qh_setcopy(neighborA->vertices, 0);
  else
    intersect= qh_vertexintersect_new(neighborA->vertices, neighborB->vertices);
  qh_settemppush(intersect);
  qh_setdelsorted(intersect, vertex);
  FOREACHneighbor_i_(vertex) {
    if (neighbor_i >= 2) {
      zinc_(Zintersectnum);
      qh_vertexintersect(&intersect, neighbor->vertices);
      if (!SETfirst_(intersect)) {
        zinc_(Zintersectfail);
        qh_settempfree(&intersect);
        return NULL;
      }
    }
  }
  trace3((qh ferr, 3007, "qh_neighbor_intersections: %d vertices in neighbor intersection of v%d\n",
          qh_setsize(intersect), vertex->id));
  return intersect;
} /* neighbor_intersections */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="newvertices">-</a>

  qh_newvertices( vertices )
    add vertices to end of qh.vertex_list (marks as new vertices)

  returns:
    vertices on qh.newvertex_list
    vertex->newlist set
*/
void qh_newvertices(setT *vertices) {
  vertexT *vertex, **vertexp;

  FOREACHvertex_(vertices) {
    if (!vertex->newlist) {
      qh_removevertex(vertex);
      qh_appendvertex(vertex);
    }
  }
} /* newvertices */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="reducevertices">-</a>

  qh_reducevertices()
    reduce extra vertices, shared vertices, and redundant vertices
    facet->newmerge is set if merged since last call
    if !qh.MERGEvertices, only removes extra vertices

  returns:
    True if also merged degen_redundant facets
    vertices are renamed if possible
    clears facet->newmerge and vertex->delridge

  notes:
    ignored if 2-d

  design:
    merge any degenerate or redundant facets
    for each newly merged facet
      remove extra vertices
    if qh.MERGEvertices
      for each newly merged facet
        for each vertex
          if vertex was on a deleted ridge
            rename vertex if it is shared
      remove delridge flag from new vertices
*/
boolT qh_reducevertices(void) {
  int numshare=0, numrename= 0;
  boolT degenredun= False;
  facetT *newfacet;
  vertexT *vertex, **vertexp;

  if (qh hull_dim == 2)
    return False;
  if (qh_merge_degenredundant())
    degenredun= True;
 LABELrestart:
  FORALLnew_facets {
    if (newfacet->newmerge) {
      if (!qh MERGEvertices)
        newfacet->newmerge= False;
      qh_remove_extravertices(newfacet);
    }
  }
  if (!qh MERGEvertices)
    return False;
  FORALLnew_facets {
    if (newfacet->newmerge) {
      newfacet->newmerge= False;
      FOREACHvertex_(newfacet->vertices) {
        if (vertex->delridge) {
          if (qh_rename_sharedvertex(vertex, newfacet)) {
            numshare++;
            vertexp--; /* repeat since deleted vertex */
          }
        }
      }
    }
  }
  FORALLvertex_(qh newvertex_list) {
    if (vertex->delridge && !vertex->deleted) {
      vertex->delridge= False;
      if (qh hull_dim >= 4 && qh_redundant_vertex(vertex)) {
        numrename++;
        if (qh_merge_degenredundant()) {
          degenredun= True;
          goto LABELrestart;
        }
      }
    }
  }
  trace1((qh ferr, 1014, "qh_reducevertices: renamed %d shared vertices and %d redundant vertices. Degen? %d\n",
          numshare, numrename, degenredun));
  return degenredun;
} /* reducevertices */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="redundant_vertex">-</a>

  qh_redundant_vertex( vertex )
    detect and rename a redundant vertex
    vertices have full vertex->neighbors

  returns:
    returns true if find a redundant vertex
      deletes vertex(vertex->deleted)

  notes:
    only needed if vertex->delridge and hull_dim >= 4
    may add degenerate facets to qh.facet_mergeset
    doesn't change vertex->neighbors or create redundant facets

  design:
    intersect vertices of all facet neighbors of vertex
    determine ridges for these vertices
    if find a new vertex for vertex amoung these ridges and vertices
      rename vertex to the new vertex
*/
vertexT *qh_redundant_vertex(vertexT *vertex) {
  vertexT *newvertex= NULL;
  setT *vertices, *ridges;

  trace3((qh ferr, 3008, "qh_redundant_vertex: check if v%d can be renamed\n", vertex->id));
  if ((vertices= qh_neighbor_intersections(vertex))) {
    ridges= qh_vertexridges(vertex);
    if ((newvertex= qh_find_newvertex(vertex, vertices, ridges)))
      qh_renamevertex(vertex, newvertex, ridges, NULL, NULL);
    qh_settempfree(&ridges);
    qh_settempfree(&vertices);
  }
  return newvertex;
} /* redundant_vertex */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="remove_extravertices">-</a>

  qh_remove_extravertices( facet )
    remove extra vertices from non-simplicial facets

  returns:
    returns True if it finds them

  design:
    for each vertex in facet
      if vertex not in a ridge (i.e., no longer used)
        delete vertex from facet
        delete facet from vertice's neighbors
        unless vertex in another facet
          add vertex to qh.del_vertices for later deletion
*/
boolT qh_remove_extravertices(facetT *facet) {
  ridgeT *ridge, **ridgep;
  vertexT *vertex, **vertexp;
  boolT foundrem= False;

  trace4((qh ferr, 4043, "qh_remove_extravertices: test f%d for extra vertices\n",
          facet->id));
  FOREACHvertex_(facet->vertices)
    vertex->seen= False;
  FOREACHridge_(facet->ridges) {
    FOREACHvertex_(ridge->vertices)
      vertex->seen= True;
  }
  FOREACHvertex_(facet->vertices) {
    if (!vertex->seen) {
      foundrem= True;
      zinc_(Zremvertex);
      qh_setdelsorted(facet->vertices, vertex);
      qh_setdel(vertex->neighbors, facet);
      if (!qh_setsize(vertex->neighbors)) {
        vertex->deleted= True;
        qh_setappend(&qh del_vertices, vertex);
        zinc_(Zremvertexdel);
        trace2((qh ferr, 2036, "qh_remove_extravertices: v%d deleted because it's lost all ridges\n", vertex->id));
      }else
        trace3((qh ferr, 3009, "qh_remove_extravertices: v%d removed from f%d because it's lost all ridges\n", vertex->id, facet->id));
      vertexp--; /*repeat*/
    }
  }
  return foundrem;
} /* remove_extravertices */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="rename_sharedvertex">-</a>

  qh_rename_sharedvertex( vertex, facet )
    detect and rename if shared vertex in facet
    vertices have full ->neighbors

  returns:
    newvertex or NULL
    the vertex may still exist in other facets (i.e., a neighbor was pinched)
    does not change facet->neighbors
    updates vertex->neighbors

  notes:
    a shared vertex for a facet is only in ridges to one neighbor
    this may undo a pinched facet

    it does not catch pinches involving multiple facets.  These appear
      to be difficult to detect, since an exhaustive search is too expensive.

  design:
    if vertex only has two neighbors
      determine the ridges that contain the vertex
      determine the vertices shared by both neighbors
      if can find a new vertex in this set
        rename the vertex to the new vertex
*/
vertexT *qh_rename_sharedvertex(vertexT *vertex, facetT *facet) {
  facetT *neighbor, **neighborp, *neighborA= NULL;
  setT *vertices, *ridges;
  vertexT *newvertex;

  if (qh_setsize(vertex->neighbors) == 2) {
    neighborA= SETfirstt_(vertex->neighbors, facetT);
    if (neighborA == facet)
      neighborA= SETsecondt_(vertex->neighbors, facetT);
  }else if (qh hull_dim == 3)
    return NULL;
  else {
    qh visit_id++;
    FOREACHneighbor_(facet)
      neighbor->visitid= qh visit_id;
    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == qh visit_id) {
        if (neighborA)
          return NULL;
        neighborA= neighbor;
      }
    }
    if (!neighborA) {
      qh_fprintf(qh ferr, 6101, "qhull internal error (qh_rename_sharedvertex): v%d's neighbors not in f%d\n",
        vertex->id, facet->id);
      qh_errprint("ERRONEOUS", facet, NULL, NULL, vertex);
      qh_errexit(qh_ERRqhull, NULL, NULL);
    }
  }
  /* the vertex is shared by facet and neighborA */
  ridges= qh_settemp(qh TEMPsize);
  neighborA->visitid= ++qh visit_id;
  qh_vertexridges_facet(vertex, facet, &ridges);
  trace2((qh ferr, 2037, "qh_rename_sharedvertex: p%d(v%d) is shared by f%d(%d ridges) and f%d\n",
    qh_pointid(vertex->point), vertex->id, facet->id, qh_setsize(ridges), neighborA->id));
  zinc_(Zintersectnum);
  vertices= qh_vertexintersect_new(facet->vertices, neighborA->vertices);
  qh_setdel(vertices, vertex);
  qh_settemppush(vertices);
  if ((newvertex= qh_find_newvertex(vertex, vertices, ridges)))
    qh_renamevertex(vertex, newvertex, ridges, facet, neighborA);
  qh_settempfree(&vertices);
  qh_settempfree(&ridges);
  return newvertex;
} /* rename_sharedvertex */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="renameridgevertex">-</a>

  qh_renameridgevertex( ridge, oldvertex, newvertex )
    renames oldvertex as newvertex in ridge

  returns:

  design:
    delete oldvertex from ridge
    if newvertex already in ridge
      copy ridge->noconvex to another ridge if possible
      delete the ridge
    else
      insert newvertex into the ridge
      adjust the ridge's orientation
*/
void qh_renameridgevertex(ridgeT *ridge, vertexT *oldvertex, vertexT *newvertex) {
  int nth= 0, oldnth;
  facetT *temp;
  vertexT *vertex, **vertexp;

  oldnth= qh_setindex(ridge->vertices, oldvertex);
  qh_setdelnthsorted(ridge->vertices, oldnth);
  FOREACHvertex_(ridge->vertices) {
    if (vertex == newvertex) {
      zinc_(Zdelridge);
      if (ridge->nonconvex) /* only one ridge has nonconvex set */
        qh_copynonconvex(ridge);
      qh_delridge(ridge);
      trace2((qh ferr, 2038, "qh_renameridgevertex: ridge r%d deleted.  It contained both v%d and v%d\n",
        ridge->id, oldvertex->id, newvertex->id));
      return;
    }
    if (vertex->id < newvertex->id)
      break;
    nth++;
  }
  qh_setaddnth(&ridge->vertices, nth, newvertex);
  if (abs(oldnth - nth)%2) {
    trace3((qh ferr, 3010, "qh_renameridgevertex: swapped the top and bottom of ridge r%d\n",
            ridge->id));
    temp= ridge->top;
    ridge->top= ridge->bottom;
    ridge->bottom= temp;
  }
} /* renameridgevertex */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="renamevertex">-</a>

  qh_renamevertex( oldvertex, newvertex, ridges, oldfacet, neighborA )
    renames oldvertex as newvertex in ridges
    gives oldfacet/neighborA if oldvertex is shared between two facets

  returns:
    oldvertex may still exist afterwards


  notes:
    can not change neighbors of newvertex (since it's a subset)

  design:
    for each ridge in ridges
      rename oldvertex to newvertex and delete degenerate ridges
    if oldfacet not defined
      for each neighbor of oldvertex
        delete oldvertex from neighbor's vertices
        remove extra vertices from neighbor
      add oldvertex to qh.del_vertices
    else if oldvertex only between oldfacet and neighborA
      delete oldvertex from oldfacet and neighborA
      add oldvertex to qh.del_vertices
    else oldvertex is in oldfacet and neighborA and other facets (i.e., pinched)
      delete oldvertex from oldfacet
      delete oldfacet from oldvertice's neighbors
      remove extra vertices (e.g., oldvertex) from neighborA
*/
void qh_renamevertex(vertexT *oldvertex, vertexT *newvertex, setT *ridges, facetT *oldfacet, facetT *neighborA) {
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  boolT istrace= False;

  if (qh IStracing >= 2 || oldvertex->id == qh tracevertex_id ||
        newvertex->id == qh tracevertex_id)
    istrace= True;
  FOREACHridge_(ridges)
    qh_renameridgevertex(ridge, oldvertex, newvertex);
  if (!oldfacet) {
    zinc_(Zrenameall);
    if (istrace)
      qh_fprintf(qh ferr, 8082, "qh_renamevertex: renamed v%d to v%d in several facets\n",
               oldvertex->id, newvertex->id);
    FOREACHneighbor_(oldvertex) {
      qh_maydropneighbor(neighbor);
      qh_setdelsorted(neighbor->vertices, oldvertex);
      if (qh_remove_extravertices(neighbor))
        neighborp--; /* neighbor may be deleted */
    }
    if (!oldvertex->deleted) {
      oldvertex->deleted= True;
      qh_setappend(&qh del_vertices, oldvertex);
    }
  }else if (qh_setsize(oldvertex->neighbors) == 2) {
    zinc_(Zrenameshare);
    if (istrace)
      qh_fprintf(qh ferr, 8083, "qh_renamevertex: renamed v%d to v%d in oldfacet f%d\n",
               oldvertex->id, newvertex->id, oldfacet->id);
    FOREACHneighbor_(oldvertex)
      qh_setdelsorted(neighbor->vertices, oldvertex);
    oldvertex->deleted= True;
    qh_setappend(&qh del_vertices, oldvertex);
  }else {
    zinc_(Zrenamepinch);
    if (istrace || qh IStracing)
      qh_fprintf(qh ferr, 8084, "qh_renamevertex: renamed pinched v%d to v%d between f%d and f%d\n",
               oldvertex->id, newvertex->id, oldfacet->id, neighborA->id);
    qh_setdelsorted(oldfacet->vertices, oldvertex);
    qh_setdel(oldvertex->neighbors, oldfacet);
    qh_remove_extravertices(neighborA);
  }
} /* renamevertex */


/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="test_appendmerge">-</a>

  qh_test_appendmerge( facet, neighbor )
    tests facet/neighbor for convexity
    appends to mergeset if non-convex
    if pre-merging,
      nop if qh.SKIPconvex, or qh.MERGEexact and coplanar

  returns:
    true if appends facet/neighbor to mergeset
    sets facet->center as needed
    does not change facet->seen

  design:
    if qh.cos_max is defined
      if the angle between facet normals is too shallow
        append an angle-coplanar merge to qh.mergeset
        return True
    make facet's centrum if needed
    if facet's centrum is above the neighbor
      set isconcave
    else
      if facet's centrum is not below the neighbor
        set iscoplanar
      make neighbor's centrum if needed
      if neighbor's centrum is above the facet
        set isconcave
      else if neighbor's centrum is not below the facet
        set iscoplanar
   if isconcave or iscoplanar
     get angle if needed
     append concave or coplanar merge to qh.mergeset
*/
boolT qh_test_appendmerge(facetT *facet, facetT *neighbor) {
  realT dist, dist2= -REALmax, angle= -REALmax;
  boolT isconcave= False, iscoplanar= False, okangle= False;

  if (qh SKIPconvex && !qh POSTmerging)
    return False;
  if ((!qh MERGEexact || qh POSTmerging) && qh cos_max < REALmax/2) {
    angle= qh_getangle(facet->normal, neighbor->normal);
    zinc_(Zangletests);
    if (angle > qh cos_max) {
      zinc_(Zcoplanarangle);
      qh_appendmergeset(facet, neighbor, MRGanglecoplanar, &angle);
      trace2((qh ferr, 2039, "qh_test_appendmerge: coplanar angle %4.4g between f%d and f%d\n",
         angle, facet->id, neighbor->id));
      return True;
    }else
      okangle= True;
  }
  if (!facet->center)
    facet->center= qh_getcentrum(facet);
  zzinc_(Zcentrumtests);
  qh_distplane(facet->center, neighbor, &dist);
  if (dist > qh centrum_radius)
    isconcave= True;
  else {
    if (dist > -qh centrum_radius)
      iscoplanar= True;
    if (!neighbor->center)
      neighbor->center= qh_getcentrum(neighbor);
    zzinc_(Zcentrumtests);
    qh_distplane(neighbor->center, facet, &dist2);
    if (dist2 > qh centrum_radius)
      isconcave= True;
    else if (!iscoplanar && dist2 > -qh centrum_radius)
      iscoplanar= True;
  }
  if (!isconcave && (!iscoplanar || (qh MERGEexact && !qh POSTmerging)))
    return False;
  if (!okangle && qh ANGLEmerge) {
    angle= qh_getangle(facet->normal, neighbor->normal);
    zinc_(Zangletests);
  }
  if (isconcave) {
    zinc_(Zconcaveridge);
    if (qh ANGLEmerge)
      angle += qh_ANGLEconcave + 0.5;
    qh_appendmergeset(facet, neighbor, MRGconcave, &angle);
    trace0((qh ferr, 18, "qh_test_appendmerge: concave f%d to f%d dist %4.4g and reverse dist %4.4g angle %4.4g during p%d\n",
           facet->id, neighbor->id, dist, dist2, angle, qh furthest_id));
  }else /* iscoplanar */ {
    zinc_(Zcoplanarcentrum);
    qh_appendmergeset(facet, neighbor, MRGcoplanar, &angle);
    trace2((qh ferr, 2040, "qh_test_appendmerge: coplanar f%d to f%d dist %4.4g, reverse dist %4.4g angle %4.4g\n",
              facet->id, neighbor->id, dist, dist2, angle));
  }
  return True;
} /* test_appendmerge */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="test_vneighbors">-</a>

  qh_test_vneighbors()
    test vertex neighbors for convexity
    tests all facets on qh.newfacet_list

  returns:
    true if non-convex vneighbors appended to qh.facet_mergeset
    initializes vertex neighbors if needed

  notes:
    assumes all facet neighbors have been tested
    this can be expensive
    this does not guarantee that a centrum is below all facets
      but it is unlikely
    uses qh.visit_id

  design:
    build vertex neighbors if necessary
    for all new facets
      for all vertices
        for each unvisited facet neighbor of the vertex
          test new facet and neighbor for convexity
*/
boolT qh_test_vneighbors(void /* qh newfacet_list */) {
  facetT *newfacet, *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  int nummerges= 0;

  trace1((qh ferr, 1015, "qh_test_vneighbors: testing vertex neighbors for convexity\n"));
  if (!qh VERTEXneighbors)
    qh_vertexneighbors();
  FORALLnew_facets
    newfacet->seen= False;
  FORALLnew_facets {
    newfacet->seen= True;
    newfacet->visitid= qh visit_id++;
    FOREACHneighbor_(newfacet)
      newfacet->visitid= qh visit_id;
    FOREACHvertex_(newfacet->vertices) {
      FOREACHneighbor_(vertex) {
        if (neighbor->seen || neighbor->visitid == qh visit_id)
          continue;
        if (qh_test_appendmerge(newfacet, neighbor))
          nummerges++;
      }
    }
  }
  zadd_(Ztestvneighbor, nummerges);
  trace1((qh ferr, 1016, "qh_test_vneighbors: found %d non-convex, vertex neighbors\n",
           nummerges));
  return (nummerges > 0);
} /* test_vneighbors */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="tracemerge">-</a>

  qh_tracemerge( facet1, facet2 )
    print trace message after merge
*/
void qh_tracemerge(facetT *facet1, facetT *facet2) {
  boolT waserror= False;

#ifndef qh_NOtrace
  if (qh IStracing >= 4)
    qh_errprint("MERGED", facet2, NULL, NULL, NULL);
  if (facet2 == qh tracefacet || (qh tracevertex && qh tracevertex->newlist)) {
    qh_fprintf(qh ferr, 8085, "qh_tracemerge: trace facet and vertex after merge of f%d and f%d, furthest p%d\n", facet1->id, facet2->id, qh furthest_id);
    if (facet2 != qh tracefacet)
      qh_errprint("TRACE", qh tracefacet,
        (qh tracevertex && qh tracevertex->neighbors) ?
           SETfirstt_(qh tracevertex->neighbors, facetT) : NULL,
        NULL, qh tracevertex);
  }
  if (qh tracevertex) {
    if (qh tracevertex->deleted)
      qh_fprintf(qh ferr, 8086, "qh_tracemerge: trace vertex deleted at furthest p%d\n",
            qh furthest_id);
    else
      qh_checkvertex(qh tracevertex);
  }
  if (qh tracefacet) {
    qh_checkfacet(qh tracefacet, True, &waserror);
    if (waserror)
      qh_errexit(qh_ERRqhull, qh tracefacet, NULL);
  }
#endif /* !qh_NOtrace */
  if (qh CHECKfrequently || qh IStracing >= 4) { /* can't check polygon here */
    qh_checkfacet(facet2, True, &waserror);
    if (waserror)
      qh_errexit(qh_ERRqhull, NULL, NULL);
  }
} /* tracemerge */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="tracemerging">-</a>

  qh_tracemerging()
    print trace message during POSTmerging

  returns:
    updates qh.mergereport

  notes:
    called from qh_mergecycle() and qh_mergefacet()

  see:
    qh_buildtracing()
*/
void qh_tracemerging(void) {
  realT cpu;
  int total;
  time_t timedata;
  struct tm *tp;

  qh mergereport= zzval_(Ztotmerge);
  time(&timedata);
  tp= localtime(&timedata);
  cpu= qh_CPUclock;
  cpu /= qh_SECticks;
  total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);
  qh_fprintf(qh ferr, 8087, "\n\
At %d:%d:%d & %2.5g CPU secs, qhull has merged %d facets.  The hull\n\
  contains %d facets and %d vertices.\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu,
      total, qh num_facets - qh num_visible,
      qh num_vertices-qh_setsize(qh del_vertices));
} /* tracemerging */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="updatetested">-</a>

  qh_updatetested( facet1, facet2 )
    clear facet2->tested and facet1->ridge->tested for merge

  returns:
    deletes facet2->center unless it's already large
      if so, clears facet2->ridge->tested

  design:
    clear facet2->tested
    clear ridge->tested for facet1's ridges
    if facet2 has a centrum
      if facet2 is large
        set facet2->keepcentrum
      else if facet2 has 3 vertices due to many merges, or not large and post merging
        clear facet2->keepcentrum
      unless facet2->keepcentrum
        clear facet2->center to recompute centrum later
        clear ridge->tested for facet2's ridges
*/
void qh_updatetested(facetT *facet1, facetT *facet2) {
  ridgeT *ridge, **ridgep;
  int size;

  facet2->tested= False;
  FOREACHridge_(facet1->ridges)
    ridge->tested= False;
  if (!facet2->center)
    return;
  size= qh_setsize(facet2->vertices);
  if (!facet2->keepcentrum) {
    if (size > qh hull_dim + qh_MAXnewcentrum) {
      facet2->keepcentrum= True;
      zinc_(Zwidevertices);
    }
  }else if (size <= qh hull_dim + qh_MAXnewcentrum) {
    /* center and keepcentrum was set */
    if (size == qh hull_dim || qh POSTmerging)
      facet2->keepcentrum= False; /* if many merges need to recompute centrum */
  }
  if (!facet2->keepcentrum) {
    qh_memfree(facet2->center, qh normal_size);
    facet2->center= NULL;
    FOREACHridge_(facet2->ridges)
      ridge->tested= False;
  }
} /* updatetested */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="vertexridges">-</a>

  qh_vertexridges( vertex )
    return temporary set of ridges adjacent to a vertex
    vertex->neighbors defined

  ntoes:
    uses qh.visit_id
    does not include implicit ridges for simplicial facets

  design:
    for each neighbor of vertex
      add ridges that include the vertex to ridges
*/
setT *qh_vertexridges(vertexT *vertex) {
  facetT *neighbor, **neighborp;
  setT *ridges= qh_settemp(qh TEMPsize);
  int size;

  qh visit_id++;
  FOREACHneighbor_(vertex)
    neighbor->visitid= qh visit_id;
  FOREACHneighbor_(vertex) {
    if (*neighborp)   /* no new ridges in last neighbor */
      qh_vertexridges_facet(vertex, neighbor, &ridges);
  }
  if (qh PRINTstatistics || qh IStracing) {
    size= qh_setsize(ridges);
    zinc_(Zvertexridge);
    zadd_(Zvertexridgetot, size);
    zmax_(Zvertexridgemax, size);
    trace3((qh ferr, 3011, "qh_vertexridges: found %d ridges for v%d\n",
             size, vertex->id));
  }
  return ridges;
} /* vertexridges */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="vertexridges_facet">-</a>

  qh_vertexridges_facet( vertex, facet, ridges )
    add adjacent ridges for vertex in facet
    neighbor->visitid==qh.visit_id if it hasn't been visited

  returns:
    ridges updated
    sets facet->visitid to qh.visit_id-1

  design:
    for each ridge of facet
      if ridge of visited neighbor (i.e., unprocessed)
        if vertex in ridge
          append ridge to vertex
    mark facet processed
*/
void qh_vertexridges_facet(vertexT *vertex, facetT *facet, setT **ridges) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor;

  FOREACHridge_(facet->ridges) {
    neighbor= otherfacet_(ridge, facet);
    if (neighbor->visitid == qh visit_id
    && qh_setin(ridge->vertices, vertex))
      qh_setappend(ridges, ridge);
  }
  facet->visitid= qh visit_id-1;
} /* vertexridges_facet */

/*-<a                             href="qh-merge.htm#TOC"
  >-------------------------------</a><a name="willdelete">-</a>

  qh_willdelete( facet, replace )
    moves facet to visible list
    sets facet->f.replace to replace (may be NULL)

  returns:
    bumps qh.num_visible
*/
void qh_willdelete(facetT *facet, facetT *replace) {

  qh_removefacet(facet);
  qh_prependfacet(facet, &qh visible_list);
  qh num_visible++;
  facet->visible= True;
  facet->f.replace= replace;
} /* willdelete */

#else /* qh_NOmerge */
void qh_premerge(vertexT *apex, realT maxcentrum, realT maxangle) {
}
void qh_postmerge(const char *reason, realT maxcentrum, realT maxangle,
                      boolT vneighbors) {
}
boolT qh_checkzero(boolT testall) {
   }
#endif /* qh_NOmerge */

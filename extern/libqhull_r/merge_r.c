/*<html><pre>  -<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="TOP">-</a>

   merge_r.c
   merges non-convex facets

   see qh-merge_r.htm and merge_r.h

   other modules call qh_premerge() and qh_postmerge()

   the user may call qh_postmerge() to perform additional merges.

   To remove deleted facets and vertices (qhull() in libqhull_r.c):
     qh_partitionvisible(qh, !qh_ALL, &numoutside);  // visible_list, newfacet_list
     qh_deletevisible();         // qh.visible_list
     qh_resetlists(qh, False, qh_RESETvisible);       // qh.visible_list newvertex_list newfacet_list

   assumes qh.CENTERtype= centrum

   merges occur in qh_mergefacet and in qh_mergecycle
   vertex->neighbors not set until the first merge occurs

   Copyright (c) 1993-2019 C.B. Barber.
   $Id: //main/2019/qhull/src/libqhull_r/merge_r.c#12 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

#ifndef qh_NOmerge

/* MRGnone, etc. */
const char *mergetypes[]= {
  "none",
  "coplanar",
  "anglecoplanar",
  "concave",
  "concavecoplanar",
  "twisted",
  "flip",
  "dupridge",
  "subridge",
  "vertices",
  "degen",
  "redundant",
  "mirror",
  "coplanarhorizon",
};

/*===== functions(alphabetical after premerge and postmerge) ======*/

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="premerge">-</a>

  qh_premerge(qh, apexpointid, maxcentrum )
    pre-merge nonconvex facets in qh.newfacet_list for apexpointid
    maxcentrum defines coplanar and concave (qh_test_appendmerge)

  returns:
    deleted facets added to qh.visible_list with facet->visible set

  notes:
    only called by qh_addpoint
    uses globals, qh.MERGEexact, qh.PREmerge

  design:
    mark dupridges in qh.newfacet_list
    merge facet cycles in qh.newfacet_list
    merge dupridges and concave facets in qh.newfacet_list
    check merged facet cycles for degenerate and redundant facets
    merge degenerate and redundant facets
    collect coplanar and concave facets
    merge concave, coplanar, degenerate, and redundant facets
*/
void qh_premerge(qhT *qh, int apexpointid, realT maxcentrum, realT maxangle /* qh.newfacet_list */) {
  boolT othermerge= False;

  if (qh->ZEROcentrum && qh_checkzero(qh, !qh_ALL))
    return;
  trace2((qh, qh->ferr, 2008, "qh_premerge: premerge centrum %2.2g angle %4.4g for apex p%d newfacet_list f%d\n",
            maxcentrum, maxangle, apexpointid, getid_(qh->newfacet_list)));
  if (qh->IStracing >= 4 && qh->num_facets < 100)
    qh_printlists(qh);
  qh->centrum_radius= maxcentrum;
  qh->cos_max= maxangle;
  if (qh->hull_dim >=3) {
    qh_mark_dupridges(qh, qh->newfacet_list, qh_ALL); /* facet_mergeset */
    qh_mergecycle_all(qh, qh->newfacet_list, &othermerge);
    qh_forcedmerges(qh, &othermerge /* qh.facet_mergeset */);
  }else /* qh.hull_dim == 2 */
    qh_mergecycle_all(qh, qh->newfacet_list, &othermerge);
  qh_flippedmerges(qh, qh->newfacet_list, &othermerge);
  if (!qh->MERGEexact || zzval_(Ztotmerge)) {
    zinc_(Zpremergetot);
    qh->POSTmerging= False;
    qh_getmergeset_initial(qh, qh->newfacet_list);
    qh_all_merges(qh, othermerge, False);
  }
} /* premerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="postmerge">-</a>

  qh_postmerge(qh, reason, maxcentrum, maxangle, vneighbors )
    post-merge nonconvex facets as defined by maxcentrum and maxangle
    'reason' is for reporting progress
    if vneighbors ('Qv'),
      calls qh_test_vneighbors at end of qh_all_merge from qh_postmerge

  returns:
    if first call (qh.visible_list != qh.facet_list),
      builds qh.facet_newlist, qh.newvertex_list
    deleted facets added to qh.visible_list with facet->visible
    qh.visible_list == qh.facet_list

  notes:
    called by qh_qhull after qh_buildhull
    called if a merge may be needed due to
      qh.MERGEexact ('Qx'), qh_DIMreduceBuild, POSTmerge (e.g., 'Cn'), or TESTvneighbors ('Qv')
    if firstmerge,
      calls qh_reducevertices before qh_getmergeset

  design:
    if first call
      set qh.visible_list and qh.newfacet_list to qh.facet_list
      add all facets to qh.newfacet_list
      mark non-simplicial facets, facet->newmerge
      set qh.newvertext_list to qh.vertex_list
      add all vertices to qh.newvertex_list
      if a pre-merge occurred
        set vertex->delridge {will retest the ridge}
        if qh.MERGEexact
          call qh_reducevertices()
      if no pre-merging
        merge flipped facets
    determine non-convex facets
    merge all non-convex facets
*/
void qh_postmerge(qhT *qh, const char *reason, realT maxcentrum, realT maxangle,
                      boolT vneighbors) {
  facetT *newfacet;
  boolT othermerges= False;
  vertexT *vertex;

  if (qh->REPORTfreq || qh->IStracing) {
    qh_buildtracing(qh, NULL, NULL);
    qh_printsummary(qh, qh->ferr);
    if (qh->PRINTstatistics)
      qh_printallstatistics(qh, qh->ferr, "reason");
    qh_fprintf(qh, qh->ferr, 8062, "\n%s with 'C%.2g' and 'A%.2g'\n",
        reason, maxcentrum, maxangle);
  }
  trace2((qh, qh->ferr, 2009, "qh_postmerge: postmerge.  test vneighbors? %d\n",
            vneighbors));
  qh->centrum_radius= maxcentrum;
  qh->cos_max= maxangle;
  qh->POSTmerging= True;
  if (qh->visible_list != qh->facet_list) {  /* first call due to qh_buildhull, multiple calls if qh.POSTmerge */
    qh->NEWfacets= True;
    qh->visible_list= qh->newfacet_list= qh->facet_list;
    FORALLnew_facets {              /* all facets are new facets for qh_postmerge */
      newfacet->newfacet= True;
       if (!newfacet->simplicial)
        newfacet->newmerge= True;   /* test f.vertices for 'delridge'.  'newmerge' was cleared at end of qh_all_merges */
     zinc_(Zpostfacets);
    }
    qh->newvertex_list= qh->vertex_list;
    FORALLvertices
      vertex->newfacet= True;
    if (qh->VERTEXneighbors) {  /* a merge has occurred */
      if (qh->MERGEexact && qh->hull_dim <= qh_DIMreduceBuild)
        qh_reducevertices(qh);  /* qh_all_merges did not call qh_reducevertices for v.delridge */
    }
    if (!qh->PREmerge && !qh->MERGEexact)
      qh_flippedmerges(qh, qh->newfacet_list, &othermerges);
  }
  qh_getmergeset_initial(qh, qh->newfacet_list);
  qh_all_merges(qh, False, vneighbors); /* calls qh_reducevertices before exiting */
  FORALLnew_facets
    newfacet->newmerge= False;   /* Was True if no vertex in f.vertices was 'delridge' */
} /* post_merge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="all_merges">-</a>

  qh_all_merges(qh, othermerge, vneighbors )
    merge all non-convex facets

    set othermerge if already merged facets (calls qh_reducevertices)
    if vneighbors ('Qv' at qh.POSTmerge)
      tests vertex neighbors for convexity at end (qh_test_vneighbors)
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
    uses qh.newfacet_list as implicit argument since merges call qh_removefacet()
    [apr'19] restored qh_setdellast in place of qh_next_facetmerge.  Much faster for post-merge

  design:
    while merges occur
      for each merge in qh.facet_mergeset
        unless one of the facets was already merged in this pass
          merge the facets
        test merged facets for additional merges
        add merges to qh.facet_mergeset
        if qh.POSTmerging
          periodically call qh_reducevertices to reduce extra vertices and redundant vertices
      after each pass, if qh.VERTEXneighbors
        if qh.POSTmerging or was a merge with qh.hull_dim<=5
          call qh_reducevertices
          update qh.facet_mergeset if degenredundant merges
      if 'Qv' and qh.POSTmerging
        test vertex neighbors for convexity
*/
void qh_all_merges(qhT *qh, boolT othermerge, boolT vneighbors) {
  facetT *facet1, *facet2, *newfacet;
  mergeT *merge;
  boolT wasmerge= False, isreduce;
  void **freelistp;  /* used if !qh_NOmem by qh_memfree_() */
  vertexT *vertex;
  realT angle, distance;
  mergeType mergetype;
  int numcoplanar=0, numconcave=0, numconcavecoplanar= 0, numdegenredun= 0, numnewmerges= 0, numtwisted= 0;

  trace2((qh, qh->ferr, 2010, "qh_all_merges: starting to merge %d facet and %d degenerate merges for new facets f%d, othermerge? %d\n",
            qh_setsize(qh, qh->facet_mergeset), qh_setsize(qh, qh->degen_mergeset), getid_(qh->newfacet_list), othermerge));

  while (True) {
    wasmerge= False;
    while (qh_setsize(qh, qh->facet_mergeset) > 0 || qh_setsize(qh, qh->degen_mergeset) > 0) {
      if (qh_setsize(qh, qh->degen_mergeset) > 0) {
        numdegenredun += qh_merge_degenredundant(qh);
        wasmerge= True;
      }
      while ((merge= (mergeT *)qh_setdellast(qh->facet_mergeset))) {
        facet1= merge->facet1;
        facet2= merge->facet2;
        vertex= merge->vertex1;  /* not used for qh.facet_mergeset*/
        mergetype= merge->mergetype;
        angle= merge->angle;
        distance= merge->distance;
        qh_memfree_(qh, merge, (int)sizeof(mergeT), freelistp);   /* 'merge' is invalid */
        if (facet1->visible || facet2->visible) {
          trace3((qh, qh->ferr, 3045, "qh_all_merges: drop merge of f%d (del? %d) into f%d (del? %d) mergetype %d, dist %4.4g, angle %4.4g.  One or both facets is deleted\n",
            facet1->id, facet1->visible, facet2->id, facet2->visible, mergetype, distance, angle));
          continue;
        }else if (mergetype == MRGcoplanar || mergetype == MRGanglecoplanar) {
          if (qh->MERGEindependent) {
            if ((!facet1->tested && facet1->newfacet)
            || (!facet2->tested && facet2->newfacet)) {
              trace3((qh, qh->ferr, 3064, "qh_all_merges: drop merge of f%d (tested? %d) into f%d (tested? %d) mergetype %d, dist %2.2g, angle %4.4g.  Merge independent sets of coplanar merges\n",
                facet1->id, facet1->visible, facet2->id, facet2->visible, mergetype, distance, angle));
              continue;
            }
          }
        }
        trace3((qh, qh->ferr, 3047, "qh_all_merges: merge f%d and f%d type %d dist %2.2g angle %4.4g\n",
          facet1->id, facet2->id, mergetype, distance, angle));
        if (mergetype == MRGtwisted)
          qh_merge_twisted(qh, facet1, facet2);
        else
          qh_merge_nonconvex(qh, facet1, facet2, mergetype);
        numnewmerges++;
        numdegenredun += qh_merge_degenredundant(qh);
        wasmerge= True;
        if (mergetype == MRGconcave)
          numconcave++;
        else if (mergetype == MRGconcavecoplanar)
          numconcavecoplanar++;
        else if (mergetype == MRGtwisted)
          numtwisted++;
        else if (mergetype == MRGcoplanar || mergetype == MRGanglecoplanar)
          numcoplanar++;
        else {
          qh_fprintf(qh, qh->ferr, 6394, "qhull internal error (qh_all_merges): expecting concave, coplanar, or twisted merge.  Got merge f%d f%d v%d mergetype %d\n",
            getid_(facet1), getid_(facet2), getid_(vertex), mergetype);
          qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
        }
      } /* while qh_setdellast */
      if (qh->POSTmerging && qh->hull_dim <= qh_DIMreduceBuild
      && numnewmerges > qh_MAXnewmerges) {
        numnewmerges= 0;
        wasmerge= othermerge= False;
        qh_reducevertices(qh);  /* otherwise large post merges too slow */
      }
      qh_getmergeset(qh, qh->newfacet_list); /* qh.facet_mergeset */
    } /* while facet_mergeset or degen_mergeset */
    if (qh->VERTEXneighbors) {  /* at least one merge */
      isreduce= False;
      if (qh->POSTmerging && qh->hull_dim >= 4) {
        isreduce= True;
      }else if (qh->POSTmerging || !qh->MERGEexact) {
        if ((wasmerge || othermerge) && qh->hull_dim > 2 && qh->hull_dim <= qh_DIMreduceBuild)
          isreduce= True;
      }
      if (isreduce) {
        wasmerge= othermerge= False;
        if (qh_reducevertices(qh)) {
          qh_getmergeset(qh, qh->newfacet_list); /* facet_mergeset */
          continue;
        }
      }
    }
    if (vneighbors && qh_test_vneighbors(qh /* qh.newfacet_list */))
      continue;
    break;
  } /* while (True) */
  if (wasmerge || othermerge) {
    trace3((qh, qh->ferr, 3033, "qh_all_merges: skip qh_reducevertices due to post-merging, no qh.VERTEXneighbors (%d), or hull_dim %d ==2 or >%d\n", qh->VERTEXneighbors, qh->hull_dim, qh_DIMreduceBuild))
    FORALLnew_facets {
      newfacet->newmerge= False;
    }
  }
  if (qh->CHECKfrequently && !qh->MERGEexact) {
    qh->old_randomdist= qh->RANDOMdist;
    qh->RANDOMdist= False;
    qh_checkconvex(qh, qh->newfacet_list, qh_ALGORITHMfault);
    /* qh_checkconnect(qh); [this is slow and it changes the facet order] */
    qh->RANDOMdist= qh->old_randomdist;
  }
  trace1((qh, qh->ferr, 1009, "qh_all_merges: merged %d coplanar %d concave %d concavecoplanar %d twisted facets and %d degen or redundant facets.\n",
    numcoplanar, numconcave, numconcavecoplanar, numtwisted, numdegenredun));
  if (qh->IStracing >= 4 && qh->num_facets < 500)
    qh_printlists(qh);
} /* all_merges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="all_vertexmerges">-</a>

  qh_all_vertexmerges(qh, apexpointid, facet, &retryfacet )
    merge vertices in qh.vertex_mergeset and subsequent merges

  returns:
    returns retryfacet for facet (if defined)
    updates qh.facet_list, qh.num_facets, qh.vertex_list, qh.num_vertices
    mergesets are empty
    if merges, resets facet lists

  notes:
    called from qh_qhull, qh_addpoint, and qh_buildcone_mergepinched
    vertex merges occur after facet merges and qh_resetlists

  design:
    while merges in vertex_mergeset (MRGvertices)
      merge a pair of pinched vertices
      update vertex neighbors
      merge non-convex and degenerate facets and check for ridges with duplicate vertices
      partition outside points of deleted, "visible" facets
*/
void qh_all_vertexmerges(qhT *qh, int apexpointid, facetT *facet, facetT **retryfacet) {
  int numpoints; /* ignore count of partitioned points.  Used by qh_addpoint for Zpbalance */

  if (retryfacet)
    *retryfacet= facet;
  while (qh_setsize(qh, qh->vertex_mergeset) > 0) {
    trace1((qh, qh->ferr, 1057, "qh_all_vertexmerges: starting to merge %d vertex merges for apex p%d facet f%d\n",
            qh_setsize(qh, qh->vertex_mergeset), apexpointid, getid_(facet)));
    if (qh->IStracing >= 4  && qh->num_facets < 1000)
      qh_printlists(qh);
    qh_merge_pinchedvertices(qh, apexpointid /* qh.vertex_mergeset, visible_list, newvertex_list, newfacet_list */);
    qh_update_vertexneighbors(qh); /* update neighbors of qh.newvertex_list from qh_newvertices for deleted facets on qh.visible_list */
                           /* test ridges and merge non-convex facets */
    qh_getmergeset(qh, qh->newfacet_list);
    qh_all_merges(qh, True, False); /* calls qh_reducevertices */
    if (qh->CHECKfrequently)
      qh_checkpolygon(qh, qh->facet_list);
    qh_partitionvisible(qh, !qh_ALL, &numpoints /* qh.visible_list qh.del_vertices*/);
    if (retryfacet)
      *retryfacet= qh_getreplacement(qh, *retryfacet);
    qh_deletevisible(qh /* qh.visible_list  qh.del_vertices*/);
    qh_resetlists(qh, False, qh_RESETvisible /* qh.visible_list newvertex_list qh.newfacet_list */);
    if (qh->IStracing >= 4  && qh->num_facets < 1000) {
      qh_printlists(qh);
      qh_checkpolygon(qh, qh->facet_list);
    }
  }
} /* all_vertexmerges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="appendmergeset">-</a>

  qh_appendmergeset(qh, facet, vertex, neighbor, mergetype, dist, angle )
    appends an entry to qh.facet_mergeset or qh.degen_mergeset
    if 'dist' is unknown, set it to 0.0
        if 'angle' is unknown, set it to 1.0 (coplanar)

  returns:
    merge appended to facet_mergeset or degen_mergeset
      sets ->degenerate or ->redundant if degen_mergeset

  notes:
    caller collects statistics and/or caller of qh_mergefacet
    see: qh_test_appendmerge()

  design:
    allocate merge entry
    if regular merge
      append to qh.facet_mergeset
    else if degenerate merge and qh.facet_mergeset is all degenerate
      append to qh.degen_mergeset
    else if degenerate merge
      prepend to qh.degen_mergeset (merged last)
    else if redundant merge
      append to qh.degen_mergeset
*/
void qh_appendmergeset(qhT *qh, facetT *facet, facetT *neighbor, mergeType mergetype, coordT dist, realT angle) {
  mergeT *merge, *lastmerge;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */
  const char *mergename;

  if ((facet->redundant && mergetype != MRGmirror) || neighbor->redundant) {
    trace3((qh, qh->ferr, 3051, "qh_appendmergeset: f%d is already redundant (%d) or f%d is already redundant (%d).  Ignore merge f%d and f%d type %d\n",
      facet->id, facet->redundant, neighbor->id, neighbor->redundant, facet->id, neighbor->id, mergetype));
    return;
  }
  if (facet->degenerate && mergetype == MRGdegen) {
    trace3((qh, qh->ferr, 3077, "qh_appendmergeset: f%d is already degenerate.  Ignore merge f%d type %d (MRGdegen)\n",
      facet->id, facet->id, mergetype));
    return;
  }
  if (!qh->facet_mergeset || !qh->degen_mergeset) {
    qh_fprintf(qh, qh->ferr, 6403, "qhull internal error (qh_appendmergeset): expecting temp set defined for qh.facet_mergeset (0x%x) and qh.degen_mergeset (0x%x).  Got NULL\n",
      qh->facet_mergeset, qh->degen_mergeset);
    /* otherwise qh_setappend creates a new set that is not freed by qh_freebuild() */
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (neighbor->flipped && !facet->flipped) {
    if (mergetype != MRGdupridge) {
      qh_fprintf(qh, qh->ferr, 6355, "qhull internal error (qh_appendmergeset): except for MRGdupridge, cannot merge a non-flipped facet f%d into flipped f%d, mergetype %d, dist %4.4g\n",
        facet->id, neighbor->id, mergetype, dist);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }else {
      trace2((qh, qh->ferr, 2106, "qh_appendmergeset: dupridge will merge a non-flipped facet f%d into flipped f%d, dist %4.4g\n",
        facet->id, neighbor->id, dist));
    }
  }
  qh_memalloc_(qh, (int)sizeof(mergeT), freelistp, merge, mergeT);
  merge->angle= angle;
  merge->distance= dist;
  merge->facet1= facet;
  merge->facet2= neighbor;
  merge->vertex1= NULL;
  merge->vertex2= NULL;
  merge->ridge1= NULL;
  merge->ridge2= NULL;
  merge->mergetype= mergetype;
  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  else
    mergename= mergetypes[MRGnone];
  if (mergetype < MRGdegen)
    qh_setappend(qh, &(qh->facet_mergeset), merge);
  else if (mergetype == MRGdegen) {
    facet->degenerate= True;
    if (!(lastmerge= (mergeT *)qh_setlast(qh->degen_mergeset))
    || lastmerge->mergetype == MRGdegen)
      qh_setappend(qh, &(qh->degen_mergeset), merge);
    else
      qh_setaddnth(qh, &(qh->degen_mergeset), 0, merge);    /* merged last */
  }else if (mergetype == MRGredundant) {
    facet->redundant= True;
    qh_setappend(qh, &(qh->degen_mergeset), merge);
  }else /* mergetype == MRGmirror */ {
    if (facet->redundant || neighbor->redundant) {
      qh_fprintf(qh, qh->ferr, 6092, "qhull internal error (qh_appendmergeset): facet f%d or f%d is already a mirrored facet (i.e., 'redundant')\n",
           facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    if (!qh_setequal(facet->vertices, neighbor->vertices)) {
      qh_fprintf(qh, qh->ferr, 6093, "qhull internal error (qh_appendmergeset): mirrored facets f%d and f%d do not have the same vertices\n",
           facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    facet->redundant= True;
    neighbor->redundant= True;
    qh_setappend(qh, &(qh->degen_mergeset), merge);
  }
  if (merge->mergetype >= MRGdegen) {
    trace3((qh, qh->ferr, 3044, "qh_appendmergeset: append merge f%d and f%d type %d (%s) to qh.degen_mergeset (size %d)\n",
      merge->facet1->id, merge->facet2->id, merge->mergetype, mergename, qh_setsize(qh, qh->degen_mergeset)));
  }else {
    trace3((qh, qh->ferr, 3027, "qh_appendmergeset: append merge f%d and f%d type %d (%s) dist %2.2g angle %4.4g to qh.facet_mergeset (size %d)\n",
      merge->facet1->id, merge->facet2->id, merge->mergetype, mergename, merge->distance, merge->angle, qh_setsize(qh, qh->facet_mergeset)));
  }
} /* appendmergeset */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="appendvertexmerge">-</a>

  qh_appendvertexmerge(qh, vertex, vertex2, mergetype, distance, ridge1, ridge2 )
    appends a vertex merge to qh.vertex_mergeset
    MRGsubridge includes two ridges (from MRGdupridge)
    MRGvertices includes two ridges

  notes:
    called by qh_getpinchedmerges for MRGsubridge
    called by qh_maybe_duplicateridge and qh_maybe_duplicateridges for MRGvertices
    only way to add a vertex merge to qh.vertex_mergeset
    checked by qh_next_vertexmerge
*/
void qh_appendvertexmerge(qhT *qh, vertexT *vertex, vertexT *destination, mergeType mergetype, realT distance, ridgeT *ridge1, ridgeT *ridge2) {
  mergeT *merge;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */
  const char *mergename;

  if (!qh->vertex_mergeset) {
    qh_fprintf(qh, qh->ferr, 6387, "qhull internal error (qh_appendvertexmerge): expecting temp set defined for qh.vertex_mergeset (0x%x).  Got NULL\n",
      qh->vertex_mergeset);
    /* otherwise qh_setappend creates a new set that is not freed by qh_freebuild() */
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh_memalloc_(qh, (int)sizeof(mergeT), freelistp, merge, mergeT);
  merge->angle= qh_ANGLEnone;
  merge->distance= distance;
  merge->facet1= NULL;
  merge->facet2= NULL;
  merge->vertex1= vertex;
  merge->vertex2= destination;
  merge->ridge1= ridge1;
  merge->ridge2= ridge2;
  merge->mergetype= mergetype;
  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  else
    mergename= mergetypes[MRGnone];
  if (mergetype == MRGvertices) {
    if (!ridge1 || !ridge2 || ridge1 == ridge2) {
      qh_fprintf(qh, qh->ferr, 6106, "qhull internal error (qh_appendvertexmerge): expecting two distinct ridges for MRGvertices.  Got r%d r%d\n",
        getid_(ridge1), getid_(ridge2));
      qh_errexit(qh, qh_ERRqhull, NULL, ridge1);
    }
  }
  qh_setappend(qh, &(qh->vertex_mergeset), merge);
  trace3((qh, qh->ferr, 3034, "qh_appendvertexmerge: append merge v%d into v%d r%d r%d dist %2.2g type %d (%s)\n",
    vertex->id, destination->id, getid_(ridge1), getid_(ridge2), distance, merge->mergetype, mergename));
} /* appendvertexmerge */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="basevertices">-</a>

  qh_basevertices(qh, samecycle )
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
setT *qh_basevertices(qhT *qh, facetT *samecycle) {
  facetT *same;
  vertexT *apex, *vertex, **vertexp;
  setT *vertices= qh_settemp(qh, qh->TEMPsize);

  apex= SETfirstt_(samecycle->vertices, vertexT);
  apex->visitid= ++qh->vertex_visit;
  FORALLsame_cycle_(samecycle) {
    if (same->mergeridge)
      continue;
    FOREACHvertex_(same->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        qh_setappend(qh, &vertices, vertex);
        vertex->visitid= qh->vertex_visit;
        vertex->seen= False;
      }
    }
  }
  trace4((qh, qh->ferr, 4019, "qh_basevertices: found %d vertices\n",
         qh_setsize(qh, vertices)));
  return vertices;
} /* basevertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="check_dupridge">-</a>

  qh_check_dupridge(qh, facet1, dist1, facet2, dist2 )
    Check dupridge between facet1 and facet2 for wide merge
    dist1 is the maximum distance of facet1's vertices to facet2
    dist2 is the maximum distance of facet2's vertices to facet1

  returns
    Level 1 log of the dupridge with the minimum distance between vertices
    Throws error if the merge will increase the maximum facet width by qh_WIDEduplicate (100x)

  notes:
    only called from qh_forcedmerges
*/
void qh_check_dupridge(qhT *qh, facetT *facet1, realT dist1, facetT *facet2, realT dist2) {
  vertexT *vertex, **vertexp, *vertexA, **vertexAp;
  realT dist, innerplane, mergedist, outerplane, prevdist, ratio, vertexratio;
  realT minvertex= REALmax;

  mergedist= fmin_(dist1, dist2);
  qh_outerinner(qh, NULL, &outerplane, &innerplane);  /* ratio from qh_printsummary */
  FOREACHvertex_(facet1->vertices) {     /* The dupridge is between facet1 and facet2, so either facet can be tested */
    FOREACHvertexA_(facet1->vertices) {
      if (vertex > vertexA){   /* Test each pair once */
        dist= qh_pointdist(vertex->point, vertexA->point, qh->hull_dim);
        minimize_(minvertex, dist);
        /* Not quite correct.  A facet may have a dupridge and another pair of nearly adjacent vertices. */
      }
    }
  }
  prevdist= fmax_(outerplane, innerplane);
  maximize_(prevdist, qh->ONEmerge + qh->DISTround);
  maximize_(prevdist, qh->MINoutside + qh->DISTround);
  ratio= mergedist/prevdist;
  vertexratio= minvertex/prevdist;
  trace0((qh, qh->ferr, 16, "qh_check_dupridge: dupridge between f%d and f%d (vertex dist %2.2g), dist %2.2g, reverse dist %2.2g, ratio %2.2g while processing p%d\n",
        facet1->id, facet2->id, minvertex, dist1, dist2, ratio, qh->furthest_id));
  if (ratio > qh_WIDEduplicate) {
    qh_fprintf(qh, qh->ferr, 6271, "qhull topology error (qh_check_dupridge): wide merge (%.1fx wider) due to dupridge between f%d and f%d (vertex dist %2.2g), merge dist %2.2g, while processing p%d\n- Allow error with option 'Q12'\n",
      ratio, facet1->id, facet2->id, minvertex, mergedist, qh->furthest_id);
    if (vertexratio < qh_WIDEpinched)
      qh_fprintf(qh, qh->ferr, 8145, "- Experimental option merge-pinched-vertices ('Q14') may avoid this error.  It merges nearly adjacent vertices.\n");
    if (qh->DELAUNAY)
      qh_fprintf(qh, qh->ferr, 8145, "- A bounding box for the input sites may alleviate this error.\n");
    if (!qh->ALLOWwide)
      qh_errexit2(qh, qh_ERRwide, facet1, facet2);
  }
} /* check_dupridge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkconnect">-</a>

  qh_checkconnect(qh)
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
void qh_checkconnect(qhT *qh /* qh.newfacet_list */) {
  facetT *facet, *newfacet, *errfacet= NULL, *neighbor, **neighborp;

  facet= qh->newfacet_list;
  qh_removefacet(qh, facet);
  qh_appendfacet(qh, facet);
  facet->visitid= ++qh->visit_id;
  FORALLfacet_(facet) {
    FOREACHneighbor_(facet) {
      if (neighbor->visitid != qh->visit_id) {
        qh_removefacet(qh, neighbor);
        qh_appendfacet(qh, neighbor);
        neighbor->visitid= qh->visit_id;
      }
    }
  }
  FORALLnew_facets {
    if (newfacet->visitid == qh->visit_id)
      break;
    qh_fprintf(qh, qh->ferr, 6094, "qhull internal error (qh_checkconnect): f%d is not attached to the new facets\n",
         newfacet->id);
    errfacet= newfacet;
  }
  if (errfacet)
    qh_errexit(qh, qh_ERRqhull, errfacet, NULL);
} /* checkconnect */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkdelfacet">-</a>

  qh_checkdelfacet(qh, facet, mergeset )
    check that mergeset does not reference facet

*/
void qh_checkdelfacet(qhT *qh, facetT *facet, setT *mergeset) {
  mergeT *merge, **mergep;

  FOREACHmerge_(mergeset) {
    if (merge->facet1 == facet || merge->facet2 == facet) {
      qh_fprintf(qh, qh->ferr, 6390, "qhull internal error (qh_checkdelfacet): cannot delete f%d.  It is referenced by merge f%d f%d mergetype %d\n",
        facet->id, merge->facet1->id, getid_(merge->facet2), merge->mergetype);
      qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
    }
  }
} /* checkdelfacet */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkdelridge">-</a>

  qh_checkdelridge(qh)
    check that qh_delridge_merge is not needed for deleted ridges

    notes:
      called from qh_mergecycle, qh_makenewfacets, qh_attachnewfacets
      errors if qh.vertex_mergeset is non-empty
      errors if any visible or new facet has a ridge with r.nonconvex set
      assumes that vertex.delfacet is not needed
*/
void qh_checkdelridge(qhT *qh /* qh.visible_facets, vertex_mergeset */) {
  facetT *newfacet, *visible;
  ridgeT *ridge, **ridgep;

  if (!SETempty_(qh->vertex_mergeset)) {
    qh_fprintf(qh, qh->ferr, 6382, "qhull internal error (qh_checkdelridge): expecting empty qh.vertex_mergeset in order to avoid calling qh_delridge_merge.  Got %d merges\n", qh_setsize(qh, qh->vertex_mergeset));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }

  FORALLnew_facets {
    FOREACHridge_(newfacet->ridges) {
      if (ridge->nonconvex) {
        qh_fprintf(qh, qh->ferr, 6313, "qhull internal error (qh_checkdelridge): unexpected 'nonconvex' flag for ridge r%d in newfacet f%d.  Otherwise need to call qh_delridge_merge\n",
           ridge->id, newfacet->id);
        qh_errexit(qh, qh_ERRqhull, newfacet, ridge);
      }
    }
  }

  FORALLvisible_facets {
    FOREACHridge_(visible->ridges) {
      if (ridge->nonconvex) {
        qh_fprintf(qh, qh->ferr, 6385, "qhull internal error (qh_checkdelridge): unexpected 'nonconvex' flag for ridge r%d in visible facet f%d.  Otherwise need to call qh_delridge_merge\n",
          ridge->id, visible->id);
        qh_errexit(qh, qh_ERRqhull, visible, ridge);
      }
    }
  }
} /* checkdelridge */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="checkzero">-</a>

  qh_checkzero(qh, testall )
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
    called by qh_premerge (qh.CHECKzero, 'C-0') and qh_qhull ('Qx')
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
boolT qh_checkzero(qhT *qh, boolT testall) {
  facetT *facet, *neighbor;
  facetT *horizon, *facetlist;
  int neighbor_i, neighbor_n;
  vertexT *vertex, **vertexp;
  realT dist;

  if (testall)
    facetlist= qh->facet_list;
  else {
    facetlist= qh->newfacet_list;
    FORALLfacet_(facetlist) {
      horizon= SETfirstt_(facet->neighbors, facetT);
      if (!horizon->simplicial)
        goto LABELproblem;
      if (facet->flipped || facet->dupridge || !facet->normal)
        goto LABELproblem;
    }
    if (qh->MERGEexact && qh->ZEROall_ok) {
      trace2((qh, qh->ferr, 2011, "qh_checkzero: skip convexity check until first pre-merge\n"));
      return True;
    }
  }
  FORALLfacet_(facetlist) {
    qh->vertex_visit++;
    horizon= NULL;
    FOREACHneighbor_i_(qh, facet) {
      if (!neighbor_i && !testall) {
        horizon= neighbor;
        continue; /* horizon facet tested in qh_findhorizon */
      }
      vertex= SETelemt_(facet->vertices, neighbor_i, vertexT);
      vertex->visitid= qh->vertex_visit;
      zzinc_(Zdistzero);
      qh_distplane(qh, vertex->point, neighbor, &dist);
      if (dist >= -2 * qh->DISTround) {  /* need 2x for qh_distround and 'Rn' for qh_checkconvex, same as qh.premerge_centrum */
        qh->ZEROall_ok= False;
        if (!qh->MERGEexact || testall || dist > qh->DISTround)
          goto LABELnonconvex;
      }
    }
    if (!testall && horizon) {
      FOREACHvertex_(horizon->vertices) {
        if (vertex->visitid != qh->vertex_visit) {
          zzinc_(Zdistzero);
          qh_distplane(qh, vertex->point, facet, &dist);
          if (dist >= -2 * qh->DISTround) {
            qh->ZEROall_ok= False;
            if (!qh->MERGEexact || dist > qh->DISTround)
              goto LABELnonconvexhorizon;
          }
          break;
        }
      }
    }
  }
  trace2((qh, qh->ferr, 2012, "qh_checkzero: testall %d, facets are %s\n", testall,
        (qh->MERGEexact && !testall) ?
           "not concave, flipped, or dupridge" : "clearly convex"));
  return True;

 LABELproblem:
  qh->ZEROall_ok= False;
  trace2((qh, qh->ferr, 2013, "qh_checkzero: qh_premerge is needed.  New facet f%d or its horizon f%d is non-simplicial, flipped, dupridge, or mergehorizon\n",
       facet->id, horizon->id));
  return False;

 LABELnonconvex:
  trace2((qh, qh->ferr, 2014, "qh_checkzero: facet f%d and f%d are not clearly convex.  v%d dist %.2g\n",
         facet->id, neighbor->id, vertex->id, dist));
  return False;

 LABELnonconvexhorizon:
  trace2((qh, qh->ferr, 2060, "qh_checkzero: facet f%d and horizon f%d are not clearly convex.  v%d dist %.2g\n",
      facet->id, horizon->id, vertex->id, dist));
  return False;
} /* checkzero */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="compare_anglemerge">-</a>

  qh_compare_anglemerge( mergeA, mergeB )
    used by qsort() to order qh.facet_mergeset by mergetype and angle (qh.ANGLEmerge, 'Q1')
    lower numbered mergetypes done first (MRGcoplanar before MRGconcave)

  notes:
    qh_all_merges processes qh.facet_mergeset by qh_setdellast
    [mar'19] evaluated various options with eg/q_benchmark and merging of pinched vertices (Q14)
*/
int qh_compare_anglemerge(const void *p1, const void *p2) {
  const mergeT *a= *((mergeT *const*)p1), *b= *((mergeT *const*)p2);

  if (a->mergetype != b->mergetype)
    return (a->mergetype < b->mergetype ? 1 : -1); /* select MRGcoplanar (1) before MRGconcave (3) */
  else
    return (a->angle > b->angle ? 1 : -1);         /* select coplanar merge (1.0) before sharp merge (-0.5) */
} /* compare_anglemerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="compare_facetmerge">-</a>

  qh_compare_facetmerge( mergeA, mergeB )
    used by qsort() to order merges by mergetype, first merge, first
    lower numbered mergetypes done first (MRGcoplanar before MRGconcave)
    if same merge type, flat merges are first

  notes:
    qh_all_merges processes qh.facet_mergeset by qh_setdellast
    [mar'19] evaluated various options with eg/q_benchmark and merging of pinched vertices (Q14)
*/
int qh_compare_facetmerge(const void *p1, const void *p2) {
  const mergeT *a= *((mergeT *const*)p1), *b= *((mergeT *const*)p2);

  if (a->mergetype != b->mergetype)
    return (a->mergetype < b->mergetype ? 1 : -1); /* select MRGcoplanar (1) before MRGconcave (3) */
  else if (a->mergetype == MRGanglecoplanar)
    return (a->angle > b->angle ? 1 : -1);         /* if MRGanglecoplanar, select coplanar merge (1.0) before sharp merge (-0.5) */
  else
    return (a->distance < b->distance ? 1 : -1);   /* select flat (0.0) merge before wide (1e-10) merge */
} /* compare_facetmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="comparevisit">-</a>

  qh_comparevisit( vertexA, vertexB )
    used by qsort() to order vertices by their visitid

  notes:
    only called by qh_find_newvertex
*/
int qh_comparevisit(const void *p1, const void *p2) {
  const vertexT *a= *((vertexT *const*)p1), *b= *((vertexT *const*)p2);

  if (a->visitid > b->visitid)
    return 1;
  return -1;
} /* comparevisit */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="copynonconvex">-</a>

  qh_copynonconvex(qh, atridge )
    set non-convex flag on other ridges (if any) between same neighbors

  notes:
    may be faster if use smaller ridge set

  design:
    for each ridge of atridge's top facet
      if ridge shares the same neighbor
        set nonconvex flag
*/
void qh_copynonconvex(qhT *qh, ridgeT *atridge) {
  facetT *facet, *otherfacet;
  ridgeT *ridge, **ridgep;

  facet= atridge->top;
  otherfacet= atridge->bottom;
  atridge->nonconvex= False;
  FOREACHridge_(facet->ridges) {
    if (otherfacet == ridge->top || otherfacet == ridge->bottom) {
      if (ridge != atridge) {
        ridge->nonconvex= True;
        trace4((qh, qh->ferr, 4020, "qh_copynonconvex: moved nonconvex flag from r%d to r%d between f%d and f%d\n",
                atridge->id, ridge->id, facet->id, otherfacet->id));
        break;
      }
    }
  }
} /* copynonconvex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="degen_redundant_facet">-</a>

  qh_degen_redundant_facet(qh, facet )
    check for a degenerate (too few neighbors) or redundant (subset of vertices) facet

  notes:
    called at end of qh_mergefacet, qh_renamevertex, and qh_reducevertices
    bumps vertex_visit
    called if a facet was redundant but no longer is (qh_merge_degenredundant)
    qh_appendmergeset() only appends first reference to facet (i.e., redundant)
    see: qh_test_redundant_neighbors, qh_maydropneighbor

  design:
    test for redundant neighbor
    test for degenerate facet
*/
void qh_degen_redundant_facet(qhT *qh, facetT *facet) {
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;

  trace3((qh, qh->ferr, 3028, "qh_degen_redundant_facet: test facet f%d for degen/redundant\n",
          facet->id));
  if (facet->flipped) {
    trace2((qh, qh->ferr, 3074, "qh_degen_redundant_facet: f%d is flipped, will merge later\n", facet->id));
    return;
  }
  FOREACHneighbor_(facet) {
    if (neighbor->flipped) /* disallow merge of non-flipped into flipped, neighbor will be merged later */
      continue;
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6357, "qhull internal error (qh_degen_redundant_facet): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    qh->vertex_visit++;
    FOREACHvertex_(neighbor->vertices)
      vertex->visitid= qh->vertex_visit;
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit)
        break;
    }
    if (!vertex) {
      trace2((qh, qh->ferr, 2015, "qh_degen_redundant_facet: f%d is contained in f%d.  merge\n", facet->id, neighbor->id));
      qh_appendmergeset(qh, facet, neighbor, MRGredundant, 0.0, 1.0);
      return;
    }
  }
  if (qh_setsize(qh, facet->neighbors) < qh->hull_dim) {
    qh_appendmergeset(qh, facet, facet, MRGdegen, 0.0, 1.0);
    trace2((qh, qh->ferr, 2016, "qh_degen_redundant_facet: f%d is degenerate.\n", facet->id));
  }
} /* degen_redundant_facet */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="delridge_merge">-</a>

  qh_delridge_merge(qh, ridge )
    delete ridge due to a merge

  notes:
    only called by merge_r.c (qh_mergeridges, qh_renameridgevertex)
    ridges also freed in qh_freeqhull and qh_mergecycle_ridges

  design:
    if needed, moves ridge.nonconvex to another ridge
    sets vertex.delridge for qh_reducevertices
    deletes ridge from qh.vertex_mergeset
    deletes ridge from its neighboring facets
    frees up its memory
*/
void qh_delridge_merge(qhT *qh, ridgeT *ridge) {
  vertexT *vertex, **vertexp;
  mergeT *merge;
  int merge_i, merge_n;

  trace3((qh, qh->ferr, 3036, "qh_delridge_merge: delete ridge r%d between f%d and f%d\n",
    ridge->id, ridge->top->id, ridge->bottom->id));
  if (ridge->nonconvex)
    qh_copynonconvex(qh, ridge);
  FOREACHvertex_(ridge->vertices)
    vertex->delridge= True;
  FOREACHmerge_i_(qh, qh->vertex_mergeset) {
    if (merge->ridge1 == ridge || merge->ridge2 == ridge) {
      trace3((qh, qh->ferr, 3029, "qh_delridge_merge: drop merge of v%d into v%d (dist %2.2g r%d r%d) due to deleted, duplicated ridge r%d\n",
        merge->vertex1->id, merge->vertex2->id, merge->distance, merge->ridge1->id, merge->ridge2->id, ridge->id));
      if (merge->ridge1 == ridge)
        merge->ridge2->mergevertex= False;
      else
        merge->ridge1->mergevertex= False;
      qh_setdelnth(qh, qh->vertex_mergeset, merge_i);
      merge_i--; merge_n--; /* next merge after deleted */
    }
  }
  qh_setdel(ridge->top->ridges, ridge);
  qh_setdel(ridge->bottom->ridges, ridge);
  qh_delridge(qh, ridge);
} /* delridge_merge */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="drop_mergevertex">-</a>

  qh_drop_mergevertex(qh, merge )

  clear mergevertex flags for ridges of a vertex merge
*/
void qh_drop_mergevertex(qhT *qh, mergeT *merge)
{
  if (merge->mergetype == MRGvertices) {
    merge->ridge1->mergevertex= False;
    merge->ridge1->mergevertex2= True;
    merge->ridge2->mergevertex= False;
    merge->ridge2->mergevertex2= True;
    trace3((qh, qh->ferr, 3032, "qh_drop_mergevertex: unset mergevertex for r%d and r%d due to dropped vertex merge v%d to v%d.  Sets mergevertex2\n",
      merge->ridge1->id, merge->ridge2->id, merge->vertex1->id, merge->vertex2->id));
  }
} /* drop_mergevertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="find_newvertex">-</a>

  qh_find_newvertex(qh, oldvertex, vertices, ridges )
    locate new vertex for renaming old vertex
    vertices is a set of possible new vertices
      vertices sorted by number of deleted ridges

  returns:
    newvertex or NULL
      each ridge includes both newvertex and oldvertex
    vertices without oldvertex sorted by number of deleted ridges
    qh.vertex_visit updated
    sets v.seen

  notes:
    called by qh_redundant_vertex due to vertex->delridge and qh_rename_sharedvertex
    sets vertex->visitid to 0..setsize() for vertices
    new vertex is in one of the ridges
    renaming will not cause a duplicate ridge
    renaming will minimize the number of deleted ridges
    newvertex may not be adjacent in the dual (though unlikely)

  design:
    for each vertex in vertices
      set vertex->visitid to number of ridges
    remove unvisited vertices
    set qh.vertex_visit above all possible values
    sort vertices by number of ridges (minimize ridges that need renaming
    add each ridge to qh.hash_table
    for each vertex in vertices
      find the first vertex that would not cause a duplicate ridge after a rename
*/
vertexT *qh_find_newvertex(qhT *qh, vertexT *oldvertex, setT *vertices, setT *ridges) {
  vertexT *vertex, **vertexp;
  setT *newridges;
  ridgeT *ridge, **ridgep;
  int size, hashsize;
  int hash;
  unsigned int maxvisit;

#ifndef qh_NOtrace
  if (qh->IStracing >= 4) {
    qh_fprintf(qh, qh->ferr, 8063, "qh_find_newvertex: find new vertex for v%d from ",
             oldvertex->id);
    FOREACHvertex_(vertices)
      qh_fprintf(qh, qh->ferr, 8064, "v%d ", vertex->id);
    FOREACHridge_(ridges)
      qh_fprintf(qh, qh->ferr, 8065, "r%d ", ridge->id);
    qh_fprintf(qh, qh->ferr, 8066, "\n");
  }
#endif
  FOREACHridge_(ridges) {
    FOREACHvertex_(ridge->vertices)
      vertex->seen= False;
  }
  FOREACHvertex_(vertices) {
    vertex->visitid= 0;  /* v.visitid will be number of ridges */
    vertex->seen= True;
  }
  FOREACHridge_(ridges) {
    FOREACHvertex_(ridge->vertices) {
      if (vertex->seen)
        vertex->visitid++;
    }
  }
  FOREACHvertex_(vertices) {
    if (!vertex->visitid) {
      qh_setdelnth(qh, vertices, SETindex_(vertices,vertex));
      vertexp--; /* repeat since deleted this vertex */
    }
  }
  maxvisit= (unsigned int)qh_setsize(qh, ridges);
  maximize_(qh->vertex_visit, maxvisit);
  if (!qh_setsize(qh, vertices)) {
    trace4((qh, qh->ferr, 4023, "qh_find_newvertex: vertices not in ridges for v%d\n",
            oldvertex->id));
    return NULL;
  }
  qsort(SETaddr_(vertices, vertexT), (size_t)qh_setsize(qh, vertices),
                sizeof(vertexT *), qh_comparevisit);
  /* can now use qh->vertex_visit */
  if (qh->PRINTstatistics) {
    size= qh_setsize(qh, vertices);
    zinc_(Zintersect);
    zadd_(Zintersecttot, size);
    zmax_(Zintersectmax, size);
  }
  hashsize= qh_newhashtable(qh, qh_setsize(qh, ridges));
  FOREACHridge_(ridges)
    qh_hashridge(qh, qh->hash_table, hashsize, ridge, oldvertex);
  FOREACHvertex_(vertices) {
    newridges= qh_vertexridges(qh, vertex, !qh_ALL);
    FOREACHridge_(newridges) {
      if (qh_hashridge_find(qh, qh->hash_table, hashsize, ridge, vertex, oldvertex, &hash)) {
        zinc_(Zvertexridge);
        break;
      }
    }
    qh_settempfree(qh, &newridges);
    if (!ridge)
      break;  /* found a rename */
  }
  if (vertex) {
    /* counted in qh_renamevertex */
    trace2((qh, qh->ferr, 2020, "qh_find_newvertex: found v%d for old v%d from %d vertices and %d ridges.\n",
      vertex->id, oldvertex->id, qh_setsize(qh, vertices), qh_setsize(qh, ridges)));
  }else {
    zinc_(Zfindfail);
    trace0((qh, qh->ferr, 14, "qh_find_newvertex: no vertex for renaming v%d (all duplicated ridges) during p%d\n",
      oldvertex->id, qh->furthest_id));
  }
  qh_setfree(qh, &qh->hash_table);
  return vertex;
} /* find_newvertex */

/*-<a                             href="qh-geom2_r.htm#TOC"
  >-------------------------------</a><a name="findbest_pinchedvertex">-</a>

  qh_findbest_pinchedvertex(qh, merge, apex, nearestp, distp )
    Determine the best pinched vertex to rename as its nearest neighboring vertex
    Renaming will remove a duplicate MRGdupridge in newfacet_list

  returns:
    pinched vertex (either apex or subridge), nearest vertex (subridge or neighbor vertex), and the distance between them

  notes:
    only called by qh_getpinchedmerges
    assumes qh.VERTEXneighbors
    see qh_findbest_ridgevertex

  design:
    if the facets have the same vertices
      return the nearest vertex pair
    else
      the subridge is the intersection of the two new facets minus the apex
      the subridge consists of qh.hull_dim-2 horizon vertices
      the subridge is also a matched ridge for the new facets (its duplicate)
      determine the nearest vertex to the apex
      determine the nearest pair of subridge vertices
      for each vertex in the subridge
        determine the nearest neighbor vertex (not in the subridge)
*/
vertexT *qh_findbest_pinchedvertex(qhT *qh, mergeT *merge, vertexT *apex, vertexT **nearestp, coordT *distp /* qh.newfacet_list */) {
  vertexT *vertex, **vertexp, *vertexA, **vertexAp;
  vertexT *bestvertex= NULL, *bestpinched= NULL;
  setT *subridge, *maybepinched;
  coordT dist, bestdist= REALmax;
  coordT pincheddist= (qh->ONEmerge+qh->DISTround)*qh_RATIOpinchedsubridge;

  if (!merge->facet1->simplicial || !merge->facet2->simplicial) {
    qh_fprintf(qh, qh->ferr, 6351, "qhull internal error (qh_findbest_pinchedvertex): expecting merge of adjacent, simplicial new facets.  f%d or f%d is not simplicial\n",
      merge->facet1->id, merge->facet2->id);
    qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
  }
  subridge= qh_vertexintersect_new(qh, merge->facet1->vertices, merge->facet2->vertices); /* new setT.  No error_exit() */
  if (qh_setsize(qh, subridge) == qh->hull_dim) { /* duplicate vertices */
    bestdist= qh_vertex_bestdist2(qh, subridge, &bestvertex, &bestpinched);
    if(bestvertex == apex) {
      bestvertex= bestpinched;
      bestpinched= apex;
    }
  }else {
    qh_setdel(subridge, apex);
    if (qh_setsize(qh, subridge) != qh->hull_dim - 2) {
      qh_fprintf(qh, qh->ferr, 6409, "qhull internal error (qh_findbest_pinchedvertex): expecting subridge of qh.hull_dim-2 vertices for the intersection of new facets f%d and f%d minus their apex.  Got %d vertices\n",
          merge->facet1->id, merge->facet2->id, qh_setsize(qh, subridge));
      qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
    }
    FOREACHvertex_(subridge) {
      dist= qh_pointdist(vertex->point, apex->point, qh->hull_dim);
      if (dist < bestdist) {
        bestpinched= apex;
        bestvertex= vertex;
        bestdist= dist;
      }
    }
    if (bestdist > pincheddist) {
      FOREACHvertex_(subridge) {
        FOREACHvertexA_(subridge) {
          if (vertexA->id > vertex->id) { /* once per vertex pair, do not compare addresses */
            dist= qh_pointdist(vertexA->point, vertex->point, qh->hull_dim);
            if (dist < bestdist) {
              bestpinched= vertexA;
              bestvertex= vertex;
              bestdist= dist;
            }
          }
        }
      }
    }
    if (bestdist > pincheddist) {
      FOREACHvertexA_(subridge) {
        maybepinched= qh_neighbor_vertices(qh, vertexA, subridge); /* subridge and apex tested above */
        FOREACHvertex_(maybepinched) {
          dist= qh_pointdist(vertex->point, vertexA->point, qh->hull_dim);
          if (dist < bestdist) {
            bestvertex= vertex;
            bestpinched= vertexA;
            bestdist= dist;
          }
        }
        qh_settempfree(qh, &maybepinched);
      }
    }
  }
  *distp= bestdist;
  qh_setfree(qh, &subridge); /* qh_err_exit not called since allocated */
  if (!bestvertex) {  /* should never happen if qh.hull_dim > 2 */
    qh_fprintf(qh, qh->ferr, 6274, "qhull internal error (qh_findbest_pinchedvertex): did not find best vertex for subridge of dupridge between f%d and f%d, while processing p%d\n", merge->facet1->id, merge->facet2->id, qh->furthest_id);
    qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
  }
  *nearestp= bestvertex;
  trace2((qh, qh->ferr, 2061, "qh_findbest_pinchedvertex: best pinched p%d(v%d) and vertex p%d(v%d) are closest (%2.2g) for duplicate subridge between f%d and f%d\n",
      qh_pointid(qh, bestpinched->point), bestpinched->id, qh_pointid(qh, bestvertex->point), bestvertex->id, bestdist, merge->facet1->id, merge->facet2->id));
  return bestpinched;
} /* findbest_pinchedvertex */

/*-<a                             href="qh-geom2_r.htm#TOC"
  >-------------------------------</a><a name="findbest_ridgevertex">-</a>

  qh_findbest_ridgevertex(qh, ridge, pinchedp, distp )
    Determine the best vertex/pinched-vertex to merge for ridges with the same vertices

  returns:
    vertex, pinched vertex, and the distance between them

  notes:
    assumes qh.hull_dim>=3
    see qh_findbest_pinchedvertex

*/
vertexT *qh_findbest_ridgevertex(qhT *qh, ridgeT *ridge, vertexT **pinchedp, coordT *distp) {
  vertexT *bestvertex;

  *distp= qh_vertex_bestdist2(qh, ridge->vertices, &bestvertex, pinchedp);
  trace4((qh, qh->ferr, 4069, "qh_findbest_ridgevertex: best pinched p%d(v%d) and vertex p%d(v%d) are closest (%2.2g) for duplicated ridge r%d (same vertices) between f%d and f%d\n",
      qh_pointid(qh, (*pinchedp)->point), (*pinchedp)->id, qh_pointid(qh, bestvertex->point), bestvertex->id, *distp, ridge->id, ridge->top->id, ridge->bottom->id));
  return bestvertex;
} /* findbest_ridgevertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="findbest_test">-</a>

  qh_findbest_test(qh, testcentrum, facet, neighbor, &bestfacet, &dist, &mindist, &maxdist )
    test neighbor of facet for qh_findbestneighbor()
    if testcentrum,
      tests centrum (assumes it is defined)
    else
      tests vertices
    initially *bestfacet==NULL and *dist==REALmax

  returns:
    if a better facet (i.e., vertices/centrum of facet closer to neighbor)
      updates bestfacet, dist, mindist, and maxdist

  notes:
    called by qh_findbestneighbor
    ignores pairs of flipped facets, unless that's all there is
*/
void qh_findbest_test(qhT *qh, boolT testcentrum, facetT *facet, facetT *neighbor,
      facetT **bestfacet, realT *distp, realT *mindistp, realT *maxdistp) {
  realT dist, mindist, maxdist;

  if (facet->flipped && neighbor->flipped && *bestfacet && !(*bestfacet)->flipped)
    return; /* do not merge flipped into flipped facets */
  if (testcentrum) {
    zzinc_(Zbestdist);
    qh_distplane(qh, facet->center, neighbor, &dist);
    dist *= qh->hull_dim; /* estimate furthest vertex */
    if (dist < 0) {
      maxdist= 0;
      mindist= dist;
      dist= -dist;
    }else {
      mindist= 0;
      maxdist= dist;
    }
  }else
    dist= qh_getdistance(qh, facet, neighbor, &mindist, &maxdist);
  if (dist < *distp) {
    *bestfacet= neighbor;
    *mindistp= mindist;
    *maxdistp= maxdist;
    *distp= dist;
  }
} /* findbest_test */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="findbestneighbor">-</a>

  qh_findbestneighbor(qh, facet, dist, mindist, maxdist )
    finds best neighbor (least dist) of a facet for merging

  returns:
    returns min and max distances and their max absolute value

  notes:
    error if qh_ASvoronoi
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
facetT *qh_findbestneighbor(qhT *qh, facetT *facet, realT *distp, realT *mindistp, realT *maxdistp) {
  facetT *neighbor, **neighborp, *bestfacet= NULL;
  ridgeT *ridge, **ridgep;
  boolT nonconvex= True, testcentrum= False;
  int size= qh_setsize(qh, facet->vertices);

  if(qh->CENTERtype==qh_ASvoronoi){
    qh_fprintf(qh, qh->ferr, 6272, "qhull internal error: cannot call qh_findbestneighor for f%d while qh.CENTERtype is qh_ASvoronoi\n", facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  *distp= REALmax;
  if (size > qh_BESTcentrum2 * qh->hull_dim + qh_BESTcentrum) {
    testcentrum= True;
    zinc_(Zbestcentrum);
    if (!facet->center)
       facet->center= qh_getcentrum(qh, facet);
  }
  if (size > qh->hull_dim + qh_BESTnonconvex) {
    FOREACHridge_(facet->ridges) {
      if (ridge->nonconvex) {
        neighbor= otherfacet_(ridge, facet);
        qh_findbest_test(qh, testcentrum, facet, neighbor,
                          &bestfacet, distp, mindistp, maxdistp);
      }
    }
  }
  if (!bestfacet) {
    nonconvex= False;
    FOREACHneighbor_(facet)
      qh_findbest_test(qh, testcentrum, facet, neighbor,
                        &bestfacet, distp, mindistp, maxdistp);
  }
  if (!bestfacet) {
    qh_fprintf(qh, qh->ferr, 6095, "qhull internal error (qh_findbestneighbor): no neighbors for f%d\n", facet->id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  if (testcentrum)
    qh_getdistance(qh, facet, bestfacet, mindistp, maxdistp);
  trace3((qh, qh->ferr, 3002, "qh_findbestneighbor: f%d is best neighbor for f%d testcentrum? %d nonconvex? %d dist %2.2g min %2.2g max %2.2g\n",
     bestfacet->id, facet->id, testcentrum, nonconvex, *distp, *mindistp, *maxdistp));
  return(bestfacet);
} /* findbestneighbor */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="flippedmerges">-</a>

  qh_flippedmerges(qh, facetlist, wasmerge )
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
void qh_flippedmerges(qhT *qh, facetT *facetlist, boolT *wasmerge) {
  facetT *facet, *neighbor, *facet1;
  realT dist, mindist, maxdist;
  mergeT *merge, **mergep;
  setT *othermerges;
  int nummerge= 0, numdegen= 0;

  trace4((qh, qh->ferr, 4024, "qh_flippedmerges: begin\n"));
  FORALLfacet_(facetlist) {
    if (facet->flipped && !facet->visible)
      qh_appendmergeset(qh, facet, facet, MRGflip, 0.0, 1.0);
  }
  othermerges= qh_settemppop(qh);
  if(othermerges != qh->facet_mergeset) {
    qh_fprintf(qh, qh->ferr, 6392, "qhull internal error (qh_flippedmerges): facet_mergeset (%d merges) not at top of tempstack (%d merges)\n",
        qh_setsize(qh, qh->facet_mergeset), qh_setsize(qh, othermerges));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->facet_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh_settemppush(qh, othermerges);
  FOREACHmerge_(othermerges) {
    facet1= merge->facet1;
    if (merge->mergetype != MRGflip || facet1->visible)
      continue;
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    neighbor= qh_findbestneighbor(qh, facet1, &dist, &mindist, &maxdist);
    trace0((qh, qh->ferr, 15, "qh_flippedmerges: merge flipped f%d into f%d dist %2.2g during p%d\n",
      facet1->id, neighbor->id, dist, qh->furthest_id));
    qh_mergefacet(qh, facet1, neighbor, merge->mergetype, &mindist, &maxdist, !qh_MERGEapex);
    nummerge++;
    if (qh->PRINTstatistics) {
      zinc_(Zflipped);
      wadd_(Wflippedtot, dist);
      wmax_(Wflippedmax, dist);
    }
  }
  FOREACHmerge_(othermerges) {
    if (merge->facet1->visible || merge->facet2->visible)
      qh_memfree(qh, merge, (int)sizeof(mergeT)); /* invalidates merge and othermerges */
    else
      qh_setappend(qh, &qh->facet_mergeset, merge);
  }
  qh_settempfree(qh, &othermerges);
  numdegen += qh_merge_degenredundant(qh); /* somewhat better here than after each flipped merge -- qtest.sh 10 '500 C1,2e-13 D4' 'd Qbb' */
  if (nummerge)
    *wasmerge= True;
  trace1((qh, qh->ferr, 1010, "qh_flippedmerges: merged %d flipped and %d degenredundant facets into a good neighbor\n",
    nummerge, numdegen));
} /* flippedmerges */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="forcedmerges">-</a>

  qh_forcedmerges(qh, wasmerge )
    merge dupridges
    calls qh_check_dupridge to report an error on wide merges
    assumes qh_settemppop is qh.facet_mergeset

  returns:
    removes all dupridges on facet_mergeset
    wasmerge set if merge
    qh.facet_mergeset may include non-forced merges(none for now)
    qh.degen_mergeset includes degen/redun merges

  notes:
    called by qh_premerge
    dupridges occur when the horizon is pinched,
        i.e. a subridge occurs in more than two horizon ridges.
     could rename vertices that pinch the horizon
    assumes qh_merge_degenredundant() has not be called
    othermerges isn't needed since facet_mergeset is empty afterwards
      keep it in case of change

  design:
    for each dupridge
      find current facets by chasing f.replace links
      check for wide merge due to dupridge
      determine best direction for facet
      merge one facet into the other
      remove dupridges from qh.facet_mergeset
*/
void qh_forcedmerges(qhT *qh, boolT *wasmerge) {
  facetT *facet1, *facet2, *merging, *merged, *newfacet;
  mergeT *merge, **mergep;
  realT dist, mindist, maxdist, dist2, mindist2, maxdist2;
  setT *othermerges;
  int nummerge=0, numflip=0, numdegen= 0;
  boolT wasdupridge= False;

  if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
  trace3((qh, qh->ferr, 3054, "qh_forcedmerges: merge dupridges\n"));
  othermerges= qh_settemppop(qh); /* was facet_mergeset */
  if (qh->facet_mergeset != othermerges ) {
      qh_fprintf(qh, qh->ferr, 6279, "qhull internal error (qh_forcedmerges): qh_settemppop (size %d) is not qh->facet_mergeset (size %d)\n",
          qh_setsize(qh, othermerges), qh_setsize(qh, qh->facet_mergeset));
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->facet_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh_settemppush(qh, othermerges);
  FOREACHmerge_(othermerges) {
    if (merge->mergetype != MRGdupridge)
        continue;
    wasdupridge= True;
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
        qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    facet1= qh_getreplacement(qh, merge->facet1);  /* must exist, no qh_merge_degenredunant */
    facet2= qh_getreplacement(qh, merge->facet2);  /* previously merged facet, if any */
    if (facet1 == facet2)
      continue;
    if (!qh_setin(facet2->neighbors, facet1)) {
      qh_fprintf(qh, qh->ferr, 6096, "qhull internal error (qh_forcedmerges): f%d and f%d had a dupridge but as f%d and f%d they are no longer neighbors\n",
               merge->facet1->id, merge->facet2->id, facet1->id, facet2->id);
      qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
    }
    dist= qh_getdistance(qh, facet1, facet2, &mindist, &maxdist);
    dist2= qh_getdistance(qh, facet2, facet1, &mindist2, &maxdist2);
    qh_check_dupridge(qh, facet1, dist, facet2, dist2);
    if (dist < dist2) {
      if (facet2->flipped && !facet1->flipped && dist2 < qh_WIDEdupridge*(qh->ONEmerge+qh->DISTround)) { /* prefer merge of flipped facet */
        merging= facet2;
        merged= facet1;
        dist= dist2;
        mindist= mindist2;
        maxdist= maxdist2;
      }else {
        merging= facet1;
        merged= facet2;
      }
    }else {
      if (facet1->flipped && !facet2->flipped && dist < qh_WIDEdupridge*(qh->ONEmerge+qh->DISTround)) { /* prefer merge of flipped facet */
        merging= facet1;
        merged= facet2;
      }else {
        merging= facet2;
        merged= facet1;
        dist= dist2;
        mindist= mindist2;
        maxdist= maxdist2;
      }
    }
    qh_mergefacet(qh, merging, merged, merge->mergetype, &mindist, &maxdist, !qh_MERGEapex);
    numdegen += qh_merge_degenredundant(qh); /* better here than at end -- qtest.sh 10 '500 C1,2e-13 D4' 'd Qbb' */
    if (facet1->flipped) {
      zinc_(Zmergeflipdup);
      numflip++;
    }else
      nummerge++;
    if (qh->PRINTstatistics) {
      zinc_(Zduplicate);
      wadd_(Wduplicatetot, dist);
      wmax_(Wduplicatemax, dist);
    }
  }
  FOREACHmerge_(othermerges) {
    if (merge->mergetype == MRGdupridge)
      qh_memfree(qh, merge, (int)sizeof(mergeT)); /* invalidates merge and othermerges */
    else
      qh_setappend(qh, &qh->facet_mergeset, merge);
  }
  qh_settempfree(qh, &othermerges);
  if (wasdupridge) {
    FORALLnew_facets {
      if (newfacet->dupridge) {
        newfacet->dupridge= False;
        newfacet->mergeridge= False;
        newfacet->mergeridge2= False;
        if (qh_setsize(qh, newfacet->neighbors) < qh->hull_dim) { /* not tested for MRGdupridge */
          qh_appendmergeset(qh, newfacet, newfacet, MRGdegen, 0.0, 1.0);
          trace2((qh, qh->ferr, 2107, "qh_forcedmerges: dupridge f%d is degenerate with fewer than %d neighbors\n",
                      newfacet->id, qh->hull_dim));
        }
      }
    }
    numdegen += qh_merge_degenredundant(qh);
  }
  if (nummerge || numflip) {
    *wasmerge= True;
    trace1((qh, qh->ferr, 1011, "qh_forcedmerges: merged %d facets, %d flipped facets, and %d degenredundant facets across dupridges\n",
                  nummerge, numflip, numdegen));
  }
} /* forcedmerges */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="freemergesets">-</a>

  qh_freemergesets(qh )
    free the merge sets

  notes:
    matches qh_initmergesets
*/
void qh_freemergesets(qhT *qh) {

  if (!qh->facet_mergeset || !qh->degen_mergeset || !qh->vertex_mergeset) {
    qh_fprintf(qh, qh->ferr, 6388, "qhull internal error (qh_freemergesets): expecting mergesets.  Got a NULL mergeset, qh.facet_mergeset (0x%x), qh.degen_mergeset (0x%x), qh.vertex_mergeset (0x%x)\n",
      qh->facet_mergeset, qh->degen_mergeset, qh->vertex_mergeset);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (!SETempty_(qh->facet_mergeset) || !SETempty_(qh->degen_mergeset) || !SETempty_(qh->vertex_mergeset)) {
    qh_fprintf(qh, qh->ferr, 6389, "qhull internal error (qh_freemergesets): expecting empty mergesets.  Got qh.facet_mergeset (%d merges), qh.degen_mergeset (%d merges), qh.vertex_mergeset (%d merges)\n",
      qh_setsize(qh, qh->facet_mergeset), qh_setsize(qh, qh->degen_mergeset), qh_setsize(qh, qh->vertex_mergeset));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh_settempfree(qh, &qh->facet_mergeset);
  qh_settempfree(qh, &qh->vertex_mergeset);
  qh_settempfree(qh, &qh->degen_mergeset);
} /* freemergesets */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="getmergeset">-</a>

  qh_getmergeset(qh, facetlist )
    determines nonconvex facets on facetlist
    tests !tested ridges and nonconvex ridges of !tested facets

  returns:
    returns sorted qh.facet_mergeset of facet-neighbor pairs to be merged
    all ridges tested

  notes:
    facetlist is qh.facet_newlist, use qh_getmergeset_initial for all facets
    assumes no nonconvex ridges with both facets tested
    uses facet->tested/ridge->tested to prevent duplicate tests
    can not limit tests to modified ridges since the centrum changed
    uses qh.visit_id

  design:
    for each facet on facetlist
      for each ridge of facet
        if untested ridge
          test ridge for convexity
          if non-convex
            append ridge to qh.facet_mergeset
    sort qh.facet_mergeset by mergetype and angle or distance
*/
void qh_getmergeset(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int nummerges;
  boolT simplicial;

  nummerges= qh_setsize(qh, qh->facet_mergeset);
  trace4((qh, qh->ferr, 4026, "qh_getmergeset: started.\n"));
  qh->visit_id++;
  FORALLfacet_(facetlist) {
    if (facet->tested)
      continue;
    facet->visitid= qh->visit_id;
    FOREACHneighbor_(facet)
      neighbor->seen= False;
    /* facet must be non-simplicial due to merge to qh.facet_newlist */
    FOREACHridge_(facet->ridges) {
      if (ridge->tested && !ridge->nonconvex)
        continue;
      /* if r.tested & r.nonconvex, need to retest and append merge */
      neighbor= otherfacet_(ridge, facet);
      if (neighbor->seen) { /* another ridge for this facet-neighbor pair was already tested in this loop */
        ridge->tested= True;
        ridge->nonconvex= False;   /* only one ridge is marked nonconvex per facet-neighbor pair */
      }else if (neighbor->visitid != qh->visit_id) {
        neighbor->seen= True;
        ridge->nonconvex= False;
        simplicial= False;
        if (ridge->simplicialbot && ridge->simplicialtop)
          simplicial= True;
        if (qh_test_appendmerge(qh, facet, neighbor, simplicial))
          ridge->nonconvex= True;
        ridge->tested= True;
      }
    }
    facet->tested= True;
  }
  nummerges= qh_setsize(qh, qh->facet_mergeset);
  if (qh->ANGLEmerge)
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_anglemerge);
  else
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_facetmerge);
  nummerges += qh_setsize(qh, qh->degen_mergeset);
  if (qh->POSTmerging) {
    zadd_(Zmergesettot2, nummerges);
  }else {
    zadd_(Zmergesettot, nummerges);
    zmax_(Zmergesetmax, nummerges);
  }
  trace2((qh, qh->ferr, 2021, "qh_getmergeset: %d merges found\n", nummerges));
} /* getmergeset */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="getmergeset_initial">-</a>

  qh_getmergeset_initial(qh, facetlist )
    determine initial qh.facet_mergeset for facets
    tests all facet/neighbor pairs on facetlist

  returns:
    sorted qh.facet_mergeset with nonconvex ridges
    sets facet->tested, ridge->tested, and ridge->nonconvex

  notes:
    uses visit_id, assumes ridge->nonconvex is False
    see qh_getmergeset

  design:
    for each facet on facetlist
      for each untested neighbor of facet
        test facet and neighbor for convexity
        if non-convex
          append merge to qh.facet_mergeset
          mark one of the ridges as nonconvex
    sort qh.facet_mergeset by mergetype and angle or distance
*/
void qh_getmergeset_initial(qhT *qh, facetT *facetlist) {
  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int nummerges;
  boolT simplicial;

  qh->visit_id++;
  FORALLfacet_(facetlist) {
    facet->visitid= qh->visit_id;
    FOREACHneighbor_(facet) {
      if (neighbor->visitid != qh->visit_id) {
        simplicial= False; /* ignores r.simplicialtop/simplicialbot.  Need to test horizon facets */
        if (facet->simplicial && neighbor->simplicial)
          simplicial= True;
        if (qh_test_appendmerge(qh, facet, neighbor, simplicial)) {
          FOREACHridge_(neighbor->ridges) {
            if (facet == otherfacet_(ridge, neighbor)) {
              ridge->nonconvex= True;
              break;    /* only one ridge is marked nonconvex */
            }
          }
        }
      }
    }
    facet->tested= True;
    FOREACHridge_(facet->ridges)
      ridge->tested= True;
  }
  nummerges= qh_setsize(qh, qh->facet_mergeset);
  if (qh->ANGLEmerge)
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_anglemerge);
  else
    qsort(SETaddr_(qh->facet_mergeset, mergeT), (size_t)nummerges, sizeof(mergeT *), qh_compare_facetmerge);
  nummerges += qh_setsize(qh, qh->degen_mergeset);
  if (qh->POSTmerging) {
    zadd_(Zmergeinittot2, nummerges);
  }else {
    zadd_(Zmergeinittot, nummerges);
    zmax_(Zmergeinitmax, nummerges);
  }
  trace2((qh, qh->ferr, 2022, "qh_getmergeset_initial: %d merges found\n", nummerges));
} /* getmergeset_initial */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="getpinchedmerges">-</a>

  qh_getpinchedmerges(qh, apex, maxdist, iscoplanar )
    get pinched merges for dupridges in qh.facet_mergeset
    qh.NEWtentative==True
      qh.newfacet_list with apex
      qh.horizon_list is attached to qh.visible_list instead of qh.newfacet_list
      maxdist for vertex-facet of a dupridge
    qh.facet_mergeset is empty
    qh.vertex_mergeset is a temporary set

  returns:
    False if nearest vertex would increase facet width by more than maxdist or qh_WIDEpinched
    True and iscoplanar, if the pinched vertex is the apex (i.e., make the apex a coplanar point)
    True and !iscoplanar, if should merge a pinched vertex of a dupridge
      qh.vertex_mergeset contains one or more MRGsubridge with a pinched vertex and a nearby, neighboring vertex
    qh.facet_mergeset is empty

  notes:
    called by qh_buildcone_mergepinched
    hull_dim >= 3
    a pinched vertex is in a dupridge and the horizon
    selects the pinched vertex that is closest to its neighbor

  design:
    for each dupridge
        determine the best pinched vertex to be merged into a neighboring vertex
        if merging the pinched vertex would produce a wide merge (qh_WIDEpinched)
           ignore pinched vertex with a warning, and use qh_merge_degenredundant instead
        else
           append the pinched vertex to vertex_mergeset for merging
*/
boolT qh_getpinchedmerges(qhT *qh, vertexT *apex, coordT maxdupdist, boolT *iscoplanar /* qh.newfacet_list, qh.vertex_mergeset */) {
  mergeT *merge, **mergep, *bestmerge= NULL;
  vertexT *nearest, *pinched, *bestvertex= NULL, *bestpinched= NULL;
  boolT result;
  coordT dist, prevdist, bestdist= REALmax/(qh_RATIOcoplanarapex+1.0); /* allow *3.0 */
  realT ratio;

  trace2((qh, qh->ferr, 2062, "qh_getpinchedmerges: try to merge pinched vertices for dupridges in new facets with apex p%d(v%d) max dupdist %2.2g\n",
      qh_pointid(qh, apex->point), apex->id, maxdupdist));
  *iscoplanar= False;
  prevdist= fmax_(qh->ONEmerge + qh->DISTround, qh->MINoutside + qh->DISTround);
  maximize_(prevdist, qh->max_outside);
  maximize_(prevdist, -qh->min_vertex);
  qh_mark_dupridges(qh, qh->newfacet_list, !qh_ALL); /* qh.facet_mergeset, creates ridges */
  /* qh_mark_dupridges is called a second time in qh_premerge */
  FOREACHmerge_(qh->facet_mergeset) {  /* read-only */
    if (merge->mergetype != MRGdupridge) {
      qh_fprintf(qh, qh->ferr, 6393, "qhull internal error (qh_getpinchedmerges): expecting MRGdupridge from qh_mark_dupridges.  Got merge f%d f%d type %d\n",
        getid_(merge->facet1), getid_(merge->facet2), merge->mergetype);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
    }
    /* dist is distance between vertices */
    pinched= qh_findbest_pinchedvertex(qh, merge, apex, &nearest, &dist /* qh.newfacet_list */);
    if (pinched == apex && dist < qh_RATIOcoplanarapex*bestdist) { /* prefer coplanar apex since it always works */
      bestdist= dist/qh_RATIOcoplanarapex;
      bestmerge= merge;
      bestpinched= pinched;
      bestvertex= nearest;
    }else if (dist < bestdist) {
      bestdist= dist;
      bestmerge= merge;
      bestpinched= pinched;
      bestvertex= nearest;
    }
  }
  result= False;
  if (bestmerge && bestdist < maxdupdist) {
    ratio= bestdist / prevdist;
    if (ratio > qh_WIDEpinched) {
      if (bestmerge->facet1->mergehorizon || bestmerge->facet2->mergehorizon) { /* e.g., rbox 175 C3,2e-13 t1539182828 | qhull d */
        trace1((qh, qh->ferr, 1051, "qh_getpinchedmerges: dupridge (MRGdupridge) of coplanar horizon would produce a wide merge (%.0fx) due to pinched vertices v%d and v%d (dist %2.2g) for f%d and f%d.  qh_mergecycle_all will merge one or both facets\n",
          ratio, bestpinched->id, bestvertex->id, bestdist, bestmerge->facet1->id, bestmerge->facet2->id));
      }else {
        qh_fprintf(qh, qh->ferr, 7081, "qhull precision warning (qh_getpinchedmerges): pinched vertices v%d and v%d (dist %2.2g, %.0fx) would produce a wide merge for f%d and f%d.  Will merge dupridge instead\n",
          bestpinched->id, bestvertex->id, bestdist, ratio, bestmerge->facet1->id, bestmerge->facet2->id);
      }
    }else {
      if (bestpinched == apex) {
        trace2((qh, qh->ferr, 2063, "qh_getpinchedmerges: will make the apex a coplanar point.  apex p%d(v%d) is the nearest vertex to v%d on dupridge.  Dist %2.2g\n",
          qh_pointid(qh, apex->point), apex->id, bestvertex->id, bestdist*qh_RATIOcoplanarapex));
        qh->coplanar_apex= apex->point;
        *iscoplanar= True;
        result= True;
      }else if (qh_setin(bestmerge->facet1->vertices, bestpinched) != qh_setin(bestmerge->facet2->vertices, bestpinched)) { /* pinched in one facet but not the other facet */
        trace2((qh, qh->ferr, 2064, "qh_getpinchedmerges: will merge new facets to resolve dupridge between f%d and f%d with pinched v%d and v%d\n",
          bestmerge->facet1->id, bestmerge->facet2->id, bestpinched->id, bestvertex->id));
        qh_appendvertexmerge(qh, bestpinched, bestvertex, MRGsubridge, bestdist, NULL, NULL);
        result= True;
      }else {
        trace2((qh, qh->ferr, 2065, "qh_getpinchedmerges: will merge pinched v%d into v%d to resolve dupridge between f%d and f%d\n",
          bestpinched->id, bestvertex->id, bestmerge->facet1->id, bestmerge->facet2->id));
        qh_appendvertexmerge(qh, bestpinched, bestvertex, MRGsubridge, bestdist, NULL, NULL);
        result= True;
      }
    }
  }
  /* delete MRGdupridge, qh_mark_dupridges is called a second time in qh_premerge */
  while ((merge= (mergeT *)qh_setdellast(qh->facet_mergeset)))
    qh_memfree(qh, merge, (int)sizeof(mergeT));
  return result;
}/* getpinchedmerges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="hasmerge">-</a>

  qh_hasmerge( mergeset, mergetype, facetA, facetB )
    True if mergeset has mergetype for facetA and facetB
*/
boolT   qh_hasmerge(setT *mergeset, mergeType type, facetT *facetA, facetT *facetB) {
  mergeT *merge, **mergep;

  FOREACHmerge_(mergeset) {
    if (merge->mergetype == type) {
      if (merge->facet1 == facetA && merge->facet2 == facetB)
        return True;
      if (merge->facet1 == facetB && merge->facet2 == facetA)
        return True;
    }
  }
  return False;
}/* hasmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="hashridge">-</a>

  qh_hashridge(qh, hashtable, hashsize, ridge, oldvertex )
    add ridge to hashtable without oldvertex

  notes:
    assumes hashtable is large enough

  design:
    determine hash value for ridge without oldvertex
    find next empty slot for ridge
*/
void qh_hashridge(qhT *qh, setT *hashtable, int hashsize, ridgeT *ridge, vertexT *oldvertex) {
  int hash;
  ridgeT *ridgeA;

  hash= qh_gethash(qh, hashsize, ridge->vertices, qh->hull_dim-1, 0, oldvertex);
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


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="hashridge_find">-</a>

  qh_hashridge_find(qh, hashtable, hashsize, ridge, vertex, oldvertex, hashslot )
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
ridgeT *qh_hashridge_find(qhT *qh, setT *hashtable, int hashsize, ridgeT *ridge,
              vertexT *vertex, vertexT *oldvertex, int *hashslot) {
  int hash;
  ridgeT *ridgeA;

  *hashslot= 0;
  zinc_(Zhashridge);
  hash= qh_gethash(qh, hashsize, ridge->vertices, qh->hull_dim-1, 0, vertex);
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


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="initmergesets">-</a>

  qh_initmergesets(qh )
    initialize the merge sets
    if 'all', include qh.degen_mergeset

  notes:
    matches qh_freemergesets
*/
void qh_initmergesets(qhT *qh /* qh.facet_mergeset,degen_mergeset,vertex_mergeset */) {

  if (qh->facet_mergeset || qh->degen_mergeset || qh->vertex_mergeset) {
    qh_fprintf(qh, qh->ferr, 6386, "qhull internal error (qh_initmergesets): expecting NULL mergesets.  Got qh.facet_mergeset (0x%x), qh.degen_mergeset (0x%x), qh.vertex_mergeset (0x%x)\n",
      qh->facet_mergeset, qh->degen_mergeset, qh->vertex_mergeset);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->degen_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh->vertex_mergeset= qh_settemp(qh, qh->TEMPsize);
  qh->facet_mergeset= qh_settemp(qh, qh->TEMPsize); /* last temporary set for qh_forcedmerges */
} /* initmergesets */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="makeridges">-</a>

  qh_makeridges(qh, facet )
    creates explicit ridges between simplicial facets

  returns:
    facet with ridges and without qh_MERGEridge
    ->simplicial is False
    if facet was tested, new ridges are tested

  notes:
    allows qh_MERGEridge flag
    uses existing ridges
    duplicate neighbors ok if ridges already exist (qh_mergecycle_ridges)

  see:
    qh_mergecycle_ridges()
    qh_rename_adjacentvertex for qh_merge_pinchedvertices

  design:
    look for qh_MERGEridge neighbors
    mark neighbors that already have ridges
    for each unprocessed neighbor of facet
      create a ridge for neighbor and facet
    if any qh_MERGEridge neighbors
      delete qh_MERGEridge flags (previously processed by qh_mark_dupridges)
*/
void qh_makeridges(qhT *qh, facetT *facet) {
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int neighbor_i, neighbor_n;
  boolT toporient, mergeridge= False;

  if (!facet->simplicial)
    return;
  trace4((qh, qh->ferr, 4027, "qh_makeridges: make ridges for f%d\n", facet->id));
  facet->simplicial= False;
  FOREACHneighbor_(facet) {
    if (neighbor == qh_MERGEridge)
      mergeridge= True;
    else
      neighbor->seen= False;
  }
  FOREACHridge_(facet->ridges)
    otherfacet_(ridge, facet)->seen= True;
  FOREACHneighbor_i_(qh, facet) {
    if (neighbor == qh_MERGEridge)
      continue;  /* fixed by qh_mark_dupridges */
    else if (!neighbor->seen) {  /* no current ridges */
      ridge= qh_newridge(qh);
      ridge->vertices= qh_setnew_delnthsorted(qh, facet->vertices, qh->hull_dim,
                                                          neighbor_i, 0);
      toporient= (boolT)(facet->toporient ^ (neighbor_i & 0x1));
      if (toporient) {
        ridge->top= facet;
        ridge->bottom= neighbor;
        ridge->simplicialtop= True;
        ridge->simplicialbot= neighbor->simplicial;
      }else {
        ridge->top= neighbor;
        ridge->bottom= facet;
        ridge->simplicialtop= neighbor->simplicial;
        ridge->simplicialbot= True;
      }
      if (facet->tested && !mergeridge)
        ridge->tested= True;
#if 0 /* this also works */
      flip= (facet->toporient ^ neighbor->toporient)^(skip1 & 0x1) ^ (skip2 & 0x1);
      if (facet->toporient ^ (skip1 & 0x1) ^ flip) {
        ridge->top= neighbor;
        ridge->bottom= facet;
        ridge->simplicialtop= True;
        ridge->simplicialbot= neighbor->simplicial;
      }else {
        ridge->top= facet;
        ridge->bottom= neighbor;
        ridge->simplicialtop= neighbor->simplicial;
        ridge->simplicialbot= True;
      }
#endif
      qh_setappend(qh, &(facet->ridges), ridge);
      trace5((qh, qh->ferr, 5005, "makeridges: appended r%d to ridges for f%d.  Next is ridges for neighbor f%d\n",
            ridge->id, facet->id, neighbor->id));
      qh_setappend(qh, &(neighbor->ridges), ridge);
      if (qh->ridge_id == qh->traceridge_id)
        qh->traceridge= ridge;
    }
  }
  if (mergeridge) {
    while (qh_setdel(facet->neighbors, qh_MERGEridge))
      ; /* delete each one */
  }
} /* makeridges */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mark_dupridges">-</a>

  qh_mark_dupridges(qh, facetlist, allmerges )
    add duplicated ridges to qh.facet_mergeset
    facet-dupridge is true if it contains a subridge shared by more than one new facet
    for each such facet, one has a neighbor marked qh_MERGEridge
    allmerges is true if merging dupridges
    allmerges is false if merging pinched vertices followed by retry addpoint
      qh_mark_dupridges will be called again if pinched vertices not found

  returns:
    dupridges on qh.facet_mergeset (MRGdupridge)
    f.mergeridge and f.mergeridge2 set for facet
    f.mergeridge set for neighbor
    if allmerges is true
      make ridges for facets with dupridges as marked by qh_MERGEridge and both sides facet->dupridge
      removes qh_MERGEridge from neighbor sets

  notes:
    called by qh_premerge and qh_getpinchedmerges
    dupridges are due to duplicate subridges
        i.e. a subridge occurs in more than two horizon ridges.
        i.e., a ridge has more than two neighboring facets
    dupridges occur in at least two cases
    1) a pinched horizon with nearly adjacent vertices -> merge the vertices (qh_getpinchedmerges)
    2) more than one newfacet for a horizon face -> merge coplanar facets (qh_premerge)
    qh_matchdupridge previously identified the furthest apart pair of facets to retain
       they must have a matching subridge and the same orientation
    only way to set facet->mergeridge and mergeridge2
    uses qh.visit_id

  design:
    for all facets on facetlist
      if facet contains a dupridge
        for each neighbor of facet
          if neighbor marked qh_MERGEridge (one side of the merge)
            set facet->mergeridge
          else
            if neighbor contains a dupridge
            and the back link is qh_MERGEridge
              append dupridge to qh.facet_mergeset
   exit if !allmerges for repeating qh_mark_dupridges later
   for each dupridge
     make ridge sets in preparation for merging
     remove qh_MERGEridge from neighbor set
   for each dupridge
     restore the missing neighbor from the neighbor set that was qh_MERGEridge
     add the missing ridge for this neighbor
*/
void qh_mark_dupridges(qhT *qh, facetT *facetlist, boolT allmerges) {
  facetT *facet, *neighbor, **neighborp;
  int nummerge=0;
  mergeT *merge, **mergep;

  trace4((qh, qh->ferr, 4028, "qh_mark_dupridges: identify dupridges in facetlist f%d, allmerges? %d\n",
    facetlist->id, allmerges));
  FORALLfacet_(facetlist) {  /* not necessary for first call */
    facet->mergeridge2= False;
    facet->mergeridge= False;
  }
  FORALLfacet_(facetlist) {
    if (facet->dupridge) {
      FOREACHneighbor_(facet) {
        if (neighbor == qh_MERGEridge) {
          facet->mergeridge= True;
          continue;
        }
        if (neighbor->dupridge) {
          if (!qh_setin(neighbor->neighbors, facet)) { /* i.e., it is qh_MERGEridge, neighbors are distinct */
            qh_appendmergeset(qh, facet, neighbor, MRGdupridge, 0.0, 1.0);
            facet->mergeridge2= True;
            facet->mergeridge= True;
            nummerge++;
          }else if (qh_setequal(facet->vertices, neighbor->vertices)) { /* neighbors are the same except for horizon and qh_MERGEridge, see QH7085 */
            trace3((qh, qh->ferr, 3043, "qh_mark_dupridges): dupridge due to duplicate vertices for subridges f%d and f%d\n",
                 facet->id, neighbor->id));
            qh_appendmergeset(qh, facet, neighbor, MRGdupridge, 0.0, 1.0);
            facet->mergeridge2= True;
            facet->mergeridge= True;
            nummerge++;
            break; /* same for all neighbors */
          }
        }
      }
    }
  }
  if (!nummerge)
    return;
  if (!allmerges) {
    trace1((qh, qh->ferr, 1012, "qh_mark_dupridges: found %d duplicated ridges (MRGdupridge) for qh_getpinchedmerges\n", nummerge));
    return;
  }
  trace1((qh, qh->ferr, 1048, "qh_mark_dupridges: found %d duplicated ridges (MRGdupridge) for qh_premerge.  Prepare facets for merging\n", nummerge));
  /* make ridges in preparation for merging */
  FORALLfacet_(facetlist) {
    if (facet->mergeridge && !facet->mergeridge2)
      qh_makeridges(qh, facet);
  }
  trace3((qh, qh->ferr, 3075, "qh_mark_dupridges: restore missing neighbors and ridges due to qh_MERGEridge\n"));
  FOREACHmerge_(qh->facet_mergeset) {   /* restore the missing neighbors */
    if (merge->mergetype == MRGdupridge) { /* only between simplicial facets */
      if (merge->facet2->mergeridge2 && qh_setin(merge->facet2->neighbors, merge->facet1)) {
        /* Due to duplicate or multiple subridges, e.g., ../eg/qtest.sh t712682 '200 s W1e-13  C1,1e-13 D5' 'd'
            merge->facet1:    - neighboring facets: f27779 f59186 f59186 f59186 MERGEridge f59186
            merge->facet2:    - neighboring facets: f27779 f59100 f59100 f59100 f59100 f59100
           or, ../eg/qtest.sh 100 '500 s W1e-13 C1,1e-13 D4' 'd'
           both facets will be degenerate after merge, consider for special case handling
        */
        qh_fprintf(qh, qh->ferr, 6361, "qhull topological error (qh_mark_dupridges): multiple dupridges for f%d and f%d, including reverse\n",
          merge->facet1->id, merge->facet2->id);
        qh_errexit2(qh, qh_ERRtopology, merge->facet1, merge->facet2);
      }else
        qh_setappend(qh, &merge->facet2->neighbors, merge->facet1);
      qh_makeridges(qh, merge->facet1);   /* and the missing ridges */
    }
  }
} /* mark_dupridges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="maybe_duplicateridge">-</a>

  qh_maybe_duplicateridge(qh, ridge )
    add MRGvertices if neighboring facet has another ridge with the same vertices

  returns:
    adds rename requests to qh.vertex_mergeset

  notes:
    called by qh_renamevertex
    nop if 2-D
    expensive test
    Duplicate ridges may lead to new facets with same vertex set (QH7084), will try merging vertices
    same as qh_maybe_duplicateridges

  design:
    for the two neighbors
      if non-simplicial
        for each ridge with the same first and last vertices (max id and min id)
          if the remaining vertices are the same
            get the closest pair of vertices
            add to vertex_mergeset for merging
*/
void qh_maybe_duplicateridge(qhT *qh, ridgeT *ridgeA) {
  ridgeT *ridge, **ridgep;
  vertexT *vertex, *pinched;
  facetT *neighbor;
  coordT dist;
  int i, k, last= qh->hull_dim-2;

  if (qh->hull_dim < 3 )
    return;

  for (neighbor= ridgeA->top, i=0; i<2; neighbor= ridgeA->bottom, i++) {
    if (!neighbor->simplicial && neighbor->nummerge > 0) { /* skip degenerate neighbors with both new and old vertices that will be merged */
      FOREACHridge_(neighbor->ridges) {
        if (ridge != ridgeA && SETfirst_(ridge->vertices) == SETfirst_(ridgeA->vertices)) {
          if (SETelem_(ridge->vertices, last) == SETelem_(ridgeA->vertices, last)) {
            for (k=1; k<last; k++) {
              if (SETelem_(ridge->vertices, k) != SETelem_(ridgeA->vertices, k))
                break;
            }
            if (k == last) {
              vertex= qh_findbest_ridgevertex(qh, ridge, &pinched, &dist);
              trace2((qh, qh->ferr, 2069, "qh_maybe_duplicateridge: will merge v%d into v%d (dist %2.2g) due to duplicate ridges r%d/r%d with the same vertices.  mergevertex set\n",
                pinched->id, vertex->id, dist, ridgeA->id, ridge->id, ridgeA->top->id, ridgeA->bottom->id, ridge->top->id, ridge->bottom->id));
              qh_appendvertexmerge(qh, pinched, vertex, MRGvertices, dist, ridgeA, ridge);
              ridge->mergevertex= True; /* disables check for duplicate vertices in qh_checkfacet */
              ridgeA->mergevertex= True;
            }
          }
        }
      }
    }
  }
} /* maybe_duplicateridge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="maybe_duplicateridges">-</a>

  qh_maybe_duplicateridges(qh, facet )
    if Q15, add MRGvertices if facet has ridges with the same vertices

  returns:
    adds rename requests to qh.vertex_mergeset

  notes:
    called at end of qh_mergefacet and qh_mergecycle_all
    only enabled if qh.CHECKduplicates ('Q15') and 3-D or more
    expensive test, not worth it
    same as qh_maybe_duplicateridge

  design:
    for all ridge pairs in facet
        if the same first and last vertices (max id and min id)
          if the remaining vertices are the same
            get the closest pair of vertices
            add to vertex_mergeset for merging
*/
void qh_maybe_duplicateridges(qhT *qh, facetT *facet) {
  facetT *otherfacet;
  ridgeT *ridge, *ridge2;
  vertexT *vertex, *pinched;
  coordT dist;
  int ridge_i, ridge_n, i, k, last_v= qh->hull_dim-2;

  if (qh->hull_dim < 3 || !qh->CHECKduplicates)
    return;

  FOREACHridge_i_(qh, facet->ridges) {
    otherfacet= otherfacet_(ridge, facet);
    if (otherfacet->degenerate || otherfacet->redundant || otherfacet->dupridge || otherfacet->flipped) /* will merge */
      continue;
    for (i=ridge_i+1; i < ridge_n; i++) {
      ridge2= SETelemt_(facet->ridges, i, ridgeT);
      otherfacet= otherfacet_(ridge2, facet);
      if (otherfacet->degenerate || otherfacet->redundant || otherfacet->dupridge || otherfacet->flipped) /* will merge */
        continue;
      /* optimize qh_setequal(ridge->vertices, ridge2->vertices) */
      if (SETelem_(ridge->vertices, last_v) == SETelem_(ridge2->vertices, last_v)) { /* SETfirst is likely to be the same */
        if (SETfirst_(ridge->vertices) == SETfirst_(ridge2->vertices)) {
          for (k=1; k<last_v; k++) {
            if (SETelem_(ridge->vertices, k) != SETelem_(ridge2->vertices, k))
              break;
          }
          if (k == last_v) {
            vertex= qh_findbest_ridgevertex(qh, ridge, &pinched, &dist);
            if (ridge->top == ridge2->bottom && ridge->bottom == ridge2->top) {
              /* proof that ridges may have opposite orientation */
              trace2((qh, qh->ferr, 2088, "qh_maybe_duplicateridges: will merge v%d into v%d (dist %2.2g) due to opposite oriented ridges r%d/r%d for f%d and f%d\n",
                pinched->id, vertex->id, dist, ridge->id, ridge2->id, ridge->top->id, ridge->bottom->id));
            }else {
              trace2((qh, qh->ferr, 2083, "qh_maybe_duplicateridges: will merge v%d into v%d (dist %2.2g) due to duplicate ridges with the same vertices r%d/r%d in merged facet f%d\n",
                pinched->id, vertex->id, dist, ridge->id, ridge2->id, facet->id));
            }
            qh_appendvertexmerge(qh, pinched, vertex, MRGvertices, dist, ridge, ridge2);
            ridge->mergevertex= True; /* disables check for duplicate vertices in qh_checkfacet */
            ridge2->mergevertex= True;
          }
        }
      }
    }
  }
} /* maybe_duplicateridges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="maydropneighbor">-</a>

  qh_maydropneighbor(qh, facet )
    drop neighbor relationship if ridge was deleted between a non-simplicial facet and its neighbors

  returns:
    for deleted ridges
      ridges made for simplicial neighbors
      neighbor sets updated
      appends degenerate facets to qh.facet_mergeset

  notes:
    called by qh_renamevertex
    assumes neighbors do not include qh_MERGEridge (qh_makeridges)
    won't cause redundant facets since vertex inclusion is the same
    may drop vertex and neighbor if no ridge
    uses qh.visit_id

  design:
    visit all neighbors with ridges
    for each unvisited neighbor of facet
      delete neighbor and facet from the non-simplicial neighbor sets
      if neighbor becomes degenerate
        append neighbor to qh.degen_mergeset
    if facet is degenerate
      append facet to qh.degen_mergeset
*/
void qh_maydropneighbor(qhT *qh, facetT *facet) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor, **neighborp;

  qh->visit_id++;
  trace4((qh, qh->ferr, 4029, "qh_maydropneighbor: test f%d for no ridges to a neighbor\n",
          facet->id));
  if (facet->simplicial) {
    qh_fprintf(qh, qh->ferr, 6278, "qhull internal error (qh_maydropneighbor): not valid for simplicial f%d while adding furthest p%d\n",
      facet->id, qh->furthest_id);
    qh_errexit(qh, qh_ERRqhull, facet, NULL);
  }
  FOREACHridge_(facet->ridges) {
    ridge->top->visitid= qh->visit_id;
    ridge->bottom->visitid= qh->visit_id;
  }
  FOREACHneighbor_(facet) {
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6358, "qhull internal error (qh_maydropneighbor): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
            facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    if (neighbor->visitid != qh->visit_id) {
      trace2((qh, qh->ferr, 2104, "qh_maydropneighbor: facets f%d and f%d are no longer neighbors while adding furthest p%d\n",
            facet->id, neighbor->id, qh->furthest_id));
      if (neighbor->simplicial) {
        qh_fprintf(qh, qh->ferr, 6280, "qhull internal error (qh_maydropneighbor): not valid for simplicial neighbor f%d of f%d while adding furthest p%d\n",
            neighbor->id, facet->id, qh->furthest_id);
        qh_errexit2(qh, qh_ERRqhull, neighbor, facet);
      }
      zinc_(Zdropneighbor);
      qh_setdel(neighbor->neighbors, facet);
      if (qh_setsize(qh, neighbor->neighbors) < qh->hull_dim) {
        zinc_(Zdropdegen);
        qh_appendmergeset(qh, neighbor, neighbor, MRGdegen, 0.0, qh_ANGLEnone);
        trace2((qh, qh->ferr, 2023, "qh_maydropneighbors: f%d is degenerate.\n", neighbor->id));
      }
      qh_setdel(facet->neighbors, neighbor);
      neighborp--;  /* repeat, deleted a neighbor */
    }
  }
  if (qh_setsize(qh, facet->neighbors) < qh->hull_dim) {
    zinc_(Zdropdegen);
    qh_appendmergeset(qh, facet, facet, MRGdegen, 0.0, qh_ANGLEnone);
    trace2((qh, qh->ferr, 2024, "qh_maydropneighbors: f%d is degenerate.\n", facet->id));
  }
} /* maydropneighbor */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_degenredundant">-</a>

  qh_merge_degenredundant(qh)
    merge all degenerate and redundant facets
    qh.degen_mergeset contains merges from  qh_test_degen_neighbors, qh_test_redundant_neighbors, and qh_degen_redundant_facet

  returns:
    number of merges performed
    resets facet->degenerate/redundant
    if deleted (visible) facet has no neighbors
      sets ->f.replace to NULL

  notes:
    redundant merges happen before degenerate ones
    merging and renaming vertices can result in degen/redundant facets
    check for coplanar and convex neighbors afterwards

  design:
    for each merge on qh.degen_mergeset
      if redundant merge
        if non-redundant facet merged into redundant facet
          recheck facet for redundancy
        else
          merge redundant facet into other facet
*/
int qh_merge_degenredundant(qhT *qh) {
  int size;
  mergeT *merge;
  facetT *bestneighbor, *facet1, *facet2, *facet3;
  realT dist, mindist, maxdist;
  vertexT *vertex, **vertexp;
  int nummerges= 0;
  mergeType mergetype;
  setT *mergedfacets;

  trace2((qh, qh->ferr, 2095, "qh_merge_degenredundant: merge %d degenerate, redundant, and mirror facets\n",
    qh_setsize(qh, qh->degen_mergeset)));
  mergedfacets= qh_settemp(qh, qh->TEMPsize);
  while ((merge= (mergeT *)qh_setdellast(qh->degen_mergeset))) {
    facet1= merge->facet1;
    facet2= merge->facet2;
    mergetype= merge->mergetype;
    qh_memfree(qh, merge, (int)sizeof(mergeT)); /* 'merge' is invalidated */
    if (facet1->visible)
      continue;
    facet1->degenerate= False;
    facet1->redundant= False;
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    if (mergetype == MRGredundant) {
      zinc_(Zredundant);
      facet3= qh_getreplacement(qh, facet2); /* the same facet if !facet2.visible */
      if (!facet3) {
          qh_fprintf(qh, qh->ferr, 6097, "qhull internal error (qh_merge_degenredunant): f%d is redundant but visible f%d has no replacement\n",
               facet1->id, getid_(facet2));
          qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
      }
      qh_setunique(qh, &mergedfacets, facet3);
      if (facet1 == facet3) {
        continue;
      }
      trace2((qh, qh->ferr, 2025, "qh_merge_degenredundant: merge redundant f%d into f%d (arg f%d)\n",
            facet1->id, facet3->id, facet2->id));
      qh_mergefacet(qh, facet1, facet3, mergetype, NULL, NULL, !qh_MERGEapex);
      /* merge distance is already accounted for */
      nummerges++;
    }else {  /* mergetype == MRGdegen or MRGmirror, other merges may have fixed */
      if (!(size= qh_setsize(qh, facet1->neighbors))) {
        zinc_(Zdelfacetdup);
        trace2((qh, qh->ferr, 2026, "qh_merge_degenredundant: facet f%d has no neighbors.  Deleted\n", facet1->id));
        qh_willdelete(qh, facet1, NULL);
        FOREACHvertex_(facet1->vertices) {
          qh_setdel(vertex->neighbors, facet1);
          if (!SETfirst_(vertex->neighbors)) {
            zinc_(Zdegenvertex);
            trace2((qh, qh->ferr, 2027, "qh_merge_degenredundant: deleted v%d because f%d has no neighbors\n",
                 vertex->id, facet1->id));
            vertex->deleted= True;
            qh_setappend(qh, &qh->del_vertices, vertex);
          }
        }
        nummerges++;
      }else if (size < qh->hull_dim) {
        bestneighbor= qh_findbestneighbor(qh, facet1, &dist, &mindist, &maxdist);
        trace2((qh, qh->ferr, 2028, "qh_merge_degenredundant: facet f%d has %d neighbors, merge into f%d dist %2.2g\n",
              facet1->id, size, bestneighbor->id, dist));
        qh_mergefacet(qh, facet1, bestneighbor, mergetype, &mindist, &maxdist, !qh_MERGEapex);
        nummerges++;
        if (qh->PRINTstatistics) {
          zinc_(Zdegen);
          wadd_(Wdegentot, dist);
          wmax_(Wdegenmax, dist);
        }
      } /* else, another merge fixed the degeneracy and redundancy tested */
    }
  }
  qh_settempfree(qh, &mergedfacets);
  return nummerges;
} /* merge_degenredundant */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_nonconvex">-</a>

  qh_merge_nonconvex(qh, facet1, facet2, mergetype )
    remove non-convex ridge between facet1 into facet2
    mergetype gives why the facet's are non-convex

  returns:
    merges one of the facets into the best neighbor

  notes:
    mergetype is MRGcoplanar..MRGconvex

  design:
    if one of the facets is a new facet
      prefer merging new facet into old facet
    find best neighbors for both facets
    merge the nearest facet into its best neighbor
    update the statistics
*/
void qh_merge_nonconvex(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype) {
  facetT *bestfacet, *bestneighbor, *neighbor, *merging, *merged;
  realT dist, dist2, mindist, mindist2, maxdist, maxdist2;

  if (mergetype < MRGcoplanar || mergetype > MRGconcavecoplanar) {
    qh_fprintf(qh, qh->ferr, 6398, "qhull internal error (qh_merge_nonconvex): expecting mergetype MRGcoplanar..MRGconcavecoplanar.  Got merge f%d and f%d type %d\n",
      facet1->id, facet2->id, mergetype);
    qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
  }
  if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
  trace3((qh, qh->ferr, 3003, "qh_merge_nonconvex: merge #%d for f%d and f%d type %d\n",
      zzval_(Ztotmerge) + 1, facet1->id, facet2->id, mergetype));
  /* concave or coplanar */
  if (!facet1->newfacet) {
    bestfacet= facet2;   /* avoid merging old facet if new is ok */
    facet2= facet1;
    facet1= bestfacet;
  }else
    bestfacet= facet1;
  bestneighbor= qh_findbestneighbor(qh, bestfacet, &dist, &mindist, &maxdist);
  neighbor= qh_findbestneighbor(qh, facet2, &dist2, &mindist2, &maxdist2);
  if (dist < dist2) {
    merging= bestfacet;
    merged= bestneighbor;
  }else if (qh->AVOIDold && !facet2->newfacet
  && ((mindist >= -qh->MAXcoplanar && maxdist <= qh->max_outside)
       || dist * 1.5 < dist2)) {
    zinc_(Zavoidold);
    wadd_(Wavoidoldtot, dist);
    wmax_(Wavoidoldmax, dist);
    trace2((qh, qh->ferr, 2029, "qh_merge_nonconvex: avoid merging old facet f%d dist %2.2g.  Use f%d dist %2.2g instead\n",
           facet2->id, dist2, facet1->id, dist2));
    merging= bestfacet;
    merged= bestneighbor;
  }else {
    merging= facet2;
    merged= neighbor;
    dist= dist2;
    mindist= mindist2;
    maxdist= maxdist2;
  }
  qh_mergefacet(qh, merging, merged, mergetype, &mindist, &maxdist, !qh_MERGEapex);
  /* caller merges qh_degenredundant */
  if (qh->PRINTstatistics) {
    if (mergetype == MRGanglecoplanar) {
      zinc_(Zacoplanar);
      wadd_(Wacoplanartot, dist);
      wmax_(Wacoplanarmax, dist);
    }else if (mergetype == MRGconcave) {
      zinc_(Zconcave);
      wadd_(Wconcavetot, dist);
      wmax_(Wconcavemax, dist);
    }else if (mergetype == MRGconcavecoplanar) {
      zinc_(Zconcavecoplanar);
      wadd_(Wconcavecoplanartot, dist);
      wmax_(Wconcavecoplanarmax, dist);
    }else { /* MRGcoplanar */
      zinc_(Zcoplanar);
      wadd_(Wcoplanartot, dist);
      wmax_(Wcoplanarmax, dist);
    }
  }
} /* merge_nonconvex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_pinchedvertices">-</a>

  qh_merge_pinchedvertices(qh, apex )
    merge pinched vertices in qh.vertex_mergeset to avoid qh_forcedmerges of dupridges

  notes:
    only called by qh_all_vertexmerges
    hull_dim >= 3

  design:
    make vertex neighbors if necessary
    for each pinched vertex
      determine the ridges for the pinched vertex (make ridges as needed)
      merge the pinched vertex into the horizon vertex
      merge the degenerate and redundant facets that result
    check and resolve new dupridges
*/
void qh_merge_pinchedvertices(qhT *qh, int apexpointid /* qh.newfacet_list */) {
  mergeT *merge, *mergeA, **mergeAp;
  vertexT *vertex, *vertex2;
  realT dist;
  boolT firstmerge= True;

  qh_vertexneighbors(qh);
  if (qh->visible_list || qh->newfacet_list || qh->newvertex_list) {
    qh_fprintf(qh, qh->ferr, 6402, "qhull internal error (qh_merge_pinchedvertices): qh.visible_list (f%d), newfacet_list (f%d), or newvertex_list (v%d) not empty\n",
      getid_(qh->visible_list), getid_(qh->newfacet_list), getid_(qh->newvertex_list));
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->visible_list= qh->newfacet_list= qh->facet_tail;
  qh->newvertex_list= qh->vertex_tail;
  qh->isRenameVertex= True; /* disable duplicate ridge vertices check in qh_checkfacet */
  while ((merge= qh_next_vertexmerge(qh /* qh.vertex_mergeset */))) { /* only one at a time from qh_getpinchedmerges */
    if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
      qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
    if (merge->mergetype == MRGsubridge) {
      zzinc_(Zpinchedvertex);
      trace1((qh, qh->ferr, 1050, "qh_merge_pinchedvertices: merge one of %d pinched vertices before adding apex p%d.  Try to resolve duplicate ridges in newfacets\n",
        qh_setsize(qh, qh->vertex_mergeset)+1, apexpointid));
      qh_remove_mergetype(qh, qh->vertex_mergeset, MRGsubridge);
    }else {
      zzinc_(Zpinchduplicate);
      if (firstmerge)
        trace1((qh, qh->ferr, 1056, "qh_merge_pinchedvertices: merge %d pinched vertices from dupridges in merged facets, apex p%d\n",
           qh_setsize(qh, qh->vertex_mergeset)+1, apexpointid));
      firstmerge= False;
    }
    vertex= merge->vertex1;
    vertex2= merge->vertex2;
    dist= merge->distance;
    qh_memfree(qh, merge, (int)sizeof(mergeT)); /* merge is invalidated */
    qh_rename_adjacentvertex(qh, vertex, vertex2, dist);
#ifndef qh_NOtrace
    if (qh->IStracing >= 2) {
      FOREACHmergeA_(qh->degen_mergeset) {
        if (mergeA->mergetype== MRGdegen) {
          qh_fprintf(qh, qh->ferr, 2072, "qh_merge_pinchedvertices: merge degenerate f%d into an adjacent facet\n", mergeA->facet1->id);
        }else {
          qh_fprintf(qh, qh->ferr, 2084, "qh_merge_pinchedvertices: merge f%d into f%d mergeType %d\n", mergeA->facet1->id, mergeA->facet2->id, mergeA->mergetype);
        }
      }
    }
#endif
    qh_merge_degenredundant(qh); /* simplicial facets with both old and new vertices */
  }
  qh->isRenameVertex= False;
}/* merge_pinchedvertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="merge_twisted">-</a>

  qh_merge_twisted(qh, facet1, facet2 )
    remove twisted ridge between facet1 into facet2 or report error

  returns:
    merges one of the facets into the best neighbor

  notes:
    a twisted ridge has opposite vertices that are convex and concave

  design:
    find best neighbors for both facets
    error if wide merge
    merge the nearest facet into its best neighbor
    update statistics
*/
void qh_merge_twisted(qhT *qh, facetT *facet1, facetT *facet2) {
  facetT *neighbor2, *neighbor, *merging, *merged;
  vertexT *bestvertex, *bestpinched;
  realT dist, dist2, mindist, mindist2, maxdist, maxdist2, mintwisted, bestdist;

  if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
  trace3((qh, qh->ferr, 3050, "qh_merge_twisted: merge #%d for twisted f%d and f%d\n",
      zzval_(Ztotmerge) + 1, facet1->id, facet2->id));
  /* twisted */
  neighbor= qh_findbestneighbor(qh, facet1, &dist, &mindist, &maxdist);
  neighbor2= qh_findbestneighbor(qh, facet2, &dist2, &mindist2, &maxdist2);
  mintwisted= qh_RATIOtwisted * qh->ONEmerge;
  maximize_(mintwisted, facet1->maxoutside);
  maximize_(mintwisted, facet2->maxoutside);
  if (dist > mintwisted && dist2 > mintwisted) {
    bestdist= qh_vertex_bestdist2(qh, facet1->vertices, &bestvertex, &bestpinched);
    if (bestdist > mintwisted) {
      qh_fprintf(qh, qh->ferr, 6417, "qhull precision error (qh_merge_twisted): twisted facet f%d does not contain pinched vertices.  Too wide to merge into neighbor.  mindist %2.2g maxdist %2.2g vertexdist %2.2g maxpinched %2.2g neighbor f%d mindist %2.2g maxdist %2.2g\n",
        facet1->id, mindist, maxdist, bestdist, mintwisted, facet2->id, mindist2, maxdist2);
    }else {
      qh_fprintf(qh, qh->ferr, 6418, "qhull precision error (qh_merge_twisted): twisted facet f%d with pinched vertices.  Could merge vertices, but too wide to merge into neighbor.   mindist %2.2g maxdist %2.2g vertexdist %2.2g neighbor f%d mindist %2.2g maxdist %2.2g\n",
        facet1->id, mindist, maxdist, bestdist, facet2->id, mindist2, maxdist2);
    }
    qh_errexit2(qh, qh_ERRwide, facet1, facet2);
  }
  if (dist < dist2) {
    merging= facet1;
    merged= neighbor;
  }else {
    /* ignores qh.AVOIDold ('Q4') */
    merging= facet2;
    merged= neighbor2;
    dist= dist2;
    mindist= mindist2;
    maxdist= maxdist2;
  }
  qh_mergefacet(qh, merging, merged, MRGtwisted, &mindist, &maxdist, !qh_MERGEapex);
  /* caller merges qh_degenredundant */
  zinc_(Ztwisted);
  wadd_(Wtwistedtot, dist);
  wmax_(Wtwistedmax, dist);
} /* merge_twisted */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle">-</a>

  qh_mergecycle(qh, samecycle, newfacet )
    merge a cycle of facets starting at samecycle into a newfacet
    newfacet is a horizon facet with ->normal
    samecycle facets are simplicial from an apex

  returns:
    initializes vertex neighbors on first merge
    samecycle deleted (placed on qh.visible_list)
    newfacet at end of qh.facet_list
    deleted vertices on qh.del_vertices

  notes:
    only called by qh_mergecycle_all for multiple, same cycle facets
    see qh_mergefacet

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
void qh_mergecycle(qhT *qh, facetT *samecycle, facetT *newfacet) {
  int traceonce= False, tracerestore= 0;
  vertexT *apex;
#ifndef qh_NOtrace
  facetT *same;
#endif

  zzinc_(Ztotmerge);
  if (qh->REPORTfreq2 && qh->POSTmerging) {
    if (zzval_(Ztotmerge) > qh->mergereport + qh->REPORTfreq2)
      qh_tracemerging(qh);
  }
#ifndef qh_NOtrace
  if (qh->TRACEmerge == zzval_(Ztotmerge))
    qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
  trace2((qh, qh->ferr, 2030, "qh_mergecycle: merge #%d for facets from cycle f%d into coplanar horizon f%d\n",
        zzval_(Ztotmerge), samecycle->id, newfacet->id));
  if (newfacet == qh->tracefacet) {
    tracerestore= qh->IStracing;
    qh->IStracing= 4;
    qh_fprintf(qh, qh->ferr, 8068, "qh_mergecycle: ========= trace merge %d of samecycle %d into trace f%d, furthest is p%d\n",
               zzval_(Ztotmerge), samecycle->id, newfacet->id,  qh->furthest_id);
    traceonce= True;
  }
  if (qh->IStracing >=4) {
    qh_fprintf(qh, qh->ferr, 8069, "  same cycle:");
    FORALLsame_cycle_(samecycle)
      qh_fprintf(qh, qh->ferr, 8070, " f%d", same->id);
    qh_fprintf(qh, qh->ferr, 8071, "\n");
  }
  if (qh->IStracing >=4)
    qh_errprint(qh, "MERGING CYCLE", samecycle, newfacet, NULL, NULL);
#endif /* !qh_NOtrace */
  if (newfacet->tricoplanar) {
    if (!qh->TRInormals) {
      qh_fprintf(qh, qh->ferr, 6224, "qhull internal error (qh_mergecycle): does not work for tricoplanar facets.  Use option 'Q11'\n");
      qh_errexit(qh, qh_ERRqhull, newfacet, NULL);
    }
    newfacet->tricoplanar= False;
    newfacet->keepcentrum= False;
  }
  if (qh->CHECKfrequently)
    qh_checkdelridge(qh);
  if (!qh->VERTEXneighbors)
    qh_vertexneighbors(qh);
  apex= SETfirstt_(samecycle->vertices, vertexT);
  qh_makeridges(qh, newfacet);
  qh_mergecycle_neighbors(qh, samecycle, newfacet);
  qh_mergecycle_ridges(qh, samecycle, newfacet);
  qh_mergecycle_vneighbors(qh, samecycle, newfacet);
  if (SETfirstt_(newfacet->vertices, vertexT) != apex)
    qh_setaddnth(qh, &newfacet->vertices, 0, apex);  /* apex has last id */
  if (!newfacet->newfacet)
    qh_newvertices(qh, newfacet->vertices);
  qh_mergecycle_facets(qh, samecycle, newfacet);
  qh_tracemerge(qh, samecycle, newfacet, MRGcoplanarhorizon);
  /* check for degen_redundant_neighbors after qh_forcedmerges() */
  if (traceonce) {
    qh_fprintf(qh, qh->ferr, 8072, "qh_mergecycle: end of trace facet\n");
    qh->IStracing= tracerestore;
  }
} /* mergecycle */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_all">-</a>

  qh_mergecycle_all(qh, facetlist, wasmerge )
    merge all samecycles of coplanar facets into horizon
    don't merge facets with ->mergeridge (these already have ->normal)
    all facets are simplicial from apex
    all facet->cycledone == False

  returns:
    all newfacets merged into coplanar horizon facets
    deleted vertices on  qh.del_vertices
    sets wasmerge if any merge

  notes:
    called by qh_premerge
    calls qh_mergecycle for multiple, same cycle facets

  design:
    for each facet on facetlist
      skip facets with dupridges and normals
      check that facet is in a samecycle (->mergehorizon)
      if facet only member of samecycle
        sets vertex->delridge for all vertices except apex
        merge facet into horizon
      else
        mark all facets in samecycle
        remove facets with dupridges from samecycle
        merge samecycle into horizon (deletes facets from facetlist)
*/
void qh_mergecycle_all(qhT *qh, facetT *facetlist, boolT *wasmerge) {
  facetT *facet, *same, *prev, *horizon, *newfacet;
  facetT *samecycle= NULL, *nextfacet, *nextsame;
  vertexT *apex, *vertex, **vertexp;
  int cycles=0, total=0, facets, nummerge, numdegen= 0;

  trace2((qh, qh->ferr, 2031, "qh_mergecycle_all: merge new facets into coplanar horizon facets.  Bulk merge a cycle of facets with the same horizon facet\n"));
  for (facet=facetlist; facet && (nextfacet= facet->next); facet= nextfacet) {
    if (facet->normal)
      continue;
    if (!facet->mergehorizon) {
      qh_fprintf(qh, qh->ferr, 6225, "qhull internal error (qh_mergecycle_all): f%d without normal\n", facet->id);
      qh_errexit(qh, qh_ERRqhull, facet, NULL);
    }
    horizon= SETfirstt_(facet->neighbors, facetT);
    if (facet->f.samecycle == facet) {
      if (qh->TRACEmerge-1 == zzval_(Ztotmerge))
        qh->qhmem.IStracing= qh->IStracing= qh->TRACElevel;
      zinc_(Zonehorizon);
      /* merge distance done in qh_findhorizon */
      apex= SETfirstt_(facet->vertices, vertexT);
      FOREACHvertex_(facet->vertices) {
        if (vertex != apex)
          vertex->delridge= True;
      }
      horizon->f.newcycle= NULL;
      qh_mergefacet(qh, facet, horizon, MRGcoplanarhorizon, NULL, NULL, qh_MERGEapex);
    }else {
      samecycle= facet;
      facets= 0;
      prev= facet;
      for (same= facet->f.samecycle; same;  /* FORALLsame_cycle_(facet) */
           same= (same == facet ? NULL :nextsame)) { /* ends at facet */
        nextsame= same->f.samecycle;
        if (same->cycledone || same->visible)
          qh_infiniteloop(qh, same);
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
      qh_mergecycle(qh, samecycle, horizon);
      nummerge= horizon->nummerge + facets;
      if (nummerge > qh_MAXnummerge)
        horizon->nummerge= qh_MAXnummerge;
      else
        horizon->nummerge= (short unsigned int)nummerge; /* limited to 9 bits by qh_MAXnummerge, -Wconversion */
      zzinc_(Zcyclehorizon);
      total += facets;
      zzadd_(Zcyclefacettot, facets);
      zmax_(Zcyclefacetmax, facets);
    }
    cycles++;
  }
  if (cycles) {
    FORALLnew_facets {
      /* qh_maybe_duplicateridges postponed since qh_mergecycle_ridges deletes ridges without calling qh_delridge_merge */
      if (newfacet->coplanarhorizon) {
        qh_test_redundant_neighbors(qh, newfacet);
        qh_maybe_duplicateridges(qh, newfacet);
        newfacet->coplanarhorizon= False;
      }
    }
    numdegen += qh_merge_degenredundant(qh);
    *wasmerge= True;
    trace1((qh, qh->ferr, 1013, "qh_mergecycle_all: merged %d same cycles or facets into coplanar horizons and %d degenredundant facets\n",
      cycles, numdegen));
  }
} /* mergecycle_all */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_facets">-</a>

  qh_mergecycle_facets(qh, samecycle, newfacet )
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
void qh_mergecycle_facets(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *same, *next;

  trace4((qh, qh->ferr, 4030, "qh_mergecycle_facets: make newfacet new and samecycle deleted\n"));
  qh_removefacet(qh, newfacet);  /* append as a newfacet to end of qh->facet_list */
  qh_appendfacet(qh, newfacet);
  newfacet->newfacet= True;
  newfacet->simplicial= False;
  newfacet->newmerge= True;

  for (same= samecycle->f.samecycle; same; same= (same == samecycle ?  NULL : next)) {
    next= same->f.samecycle;  /* reused by willdelete */
    qh_willdelete(qh, same, newfacet);
  }
  if (newfacet->center
      && qh_setsize(qh, newfacet->vertices) <= qh->hull_dim + qh_MAXnewcentrum) {
    qh_memfree(qh, newfacet->center, qh->normal_size);
    newfacet->center= NULL;
  }
  trace3((qh, qh->ferr, 3004, "qh_mergecycle_facets: merged facets from cycle f%d into f%d\n",
             samecycle->id, newfacet->id));
} /* mergecycle_facets */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_neighbors">-</a>

  qh_mergecycle_neighbors(qh, samecycle, newfacet )
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
void qh_mergecycle_neighbors(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *same, *neighbor, **neighborp;
  int delneighbors= 0, newneighbors= 0;
  unsigned int samevisitid;
  ridgeT *ridge, **ridgep;

  samevisitid= ++qh->visit_id;
  FORALLsame_cycle_(samecycle) {
    if (same->visitid == samevisitid || same->visible)
      qh_infiniteloop(qh, samecycle);
    same->visitid= samevisitid;
  }
  newfacet->visitid= ++qh->visit_id;
  trace4((qh, qh->ferr, 4031, "qh_mergecycle_neighbors: delete shared neighbors from newfacet\n"));
  FOREACHneighbor_(newfacet) {
    if (neighbor->visitid == samevisitid) {
      SETref_(neighbor)= NULL;  /* samecycle neighbors deleted */
      delneighbors++;
    }else
      neighbor->visitid= qh->visit_id;
  }
  qh_setcompact(qh, newfacet->neighbors);

  trace4((qh, qh->ferr, 4032, "qh_mergecycle_neighbors: update neighbors\n"));
  FORALLsame_cycle_(samecycle) {
    FOREACHneighbor_(same) {
      if (neighbor->visitid == samevisitid)
        continue;
      if (neighbor->simplicial) {
        if (neighbor->visitid != qh->visit_id) {
          qh_setappend(qh, &newfacet->neighbors, neighbor);
          qh_setreplace(qh, neighbor->neighbors, same, newfacet);
          newneighbors++;
          neighbor->visitid= qh->visit_id;
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
          qh_makeridges(qh, neighbor);
          qh_setdel(neighbor->neighbors, same);
          /* same can't be horizon facet for neighbor */
        }
      }else { /* non-simplicial neighbor */
        qh_setdel(neighbor->neighbors, same);
        if (neighbor->visitid != qh->visit_id) {
          qh_setappend(qh, &neighbor->neighbors, newfacet);
          qh_setappend(qh, &newfacet->neighbors, neighbor);
          neighbor->visitid= qh->visit_id;
          newneighbors++;
        }
      }
    }
  }
  trace2((qh, qh->ferr, 2032, "qh_mergecycle_neighbors: deleted %d neighbors and added %d\n",
             delneighbors, newneighbors));
} /* mergecycle_neighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_ridges">-</a>

  qh_mergecycle_ridges(qh, samecycle, newfacet )
    add ridges/neighbors for facets in samecycle to newfacet
    all new/old neighbors of newfacet marked with qh.visit_id
    facets in samecycle marked with qh.visit_id-1
    newfacet marked with qh.visit_id

  returns:
    newfacet has merged ridges

  notes:
    ridge already updated for simplicial neighbors of samecycle with a ridge
    qh_checkdelridge called by qh_mergecycle

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
      if simplicial facet
        for each neighbor of facet
          if simplicial facet
          and not samecycle facet or newfacet
            make ridge between neighbor and newfacet
*/
void qh_mergecycle_ridges(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *same, *neighbor= NULL;
  int numold=0, numnew=0;
  int neighbor_i, neighbor_n;
  unsigned int samevisitid;
  ridgeT *ridge, **ridgep;
  boolT toporient;
  void **freelistp; /* used if !qh_NOmem by qh_memfree_() */

  trace4((qh, qh->ferr, 4033, "qh_mergecycle_ridges: delete shared ridges from newfacet\n"));
  samevisitid= qh->visit_id -1;
  FOREACHridge_(newfacet->ridges) {
    neighbor= otherfacet_(ridge, newfacet);
    if (neighbor->visitid == samevisitid)
      SETref_(ridge)= NULL; /* ridge free'd below */
  }
  qh_setcompact(qh, newfacet->ridges);

  trace4((qh, qh->ferr, 4034, "qh_mergecycle_ridges: add ridges to newfacet\n"));
  FORALLsame_cycle_(samecycle) {
    FOREACHridge_(same->ridges) {
      if (ridge->top == same) {
        ridge->top= newfacet;
        neighbor= ridge->bottom;
      }else if (ridge->bottom == same) {
        ridge->bottom= newfacet;
        neighbor= ridge->top;
      }else if (ridge->top == newfacet || ridge->bottom == newfacet) {
        qh_setappend(qh, &newfacet->ridges, ridge);
        numold++;  /* already set by qh_mergecycle_neighbors */
        continue;
      }else {
        qh_fprintf(qh, qh->ferr, 6098, "qhull internal error (qh_mergecycle_ridges): bad ridge r%d\n", ridge->id);
        qh_errexit(qh, qh_ERRqhull, NULL, ridge);
      }
      if (neighbor == newfacet) {
        if (qh->traceridge == ridge)
          qh->traceridge= NULL;
        qh_setfree(qh, &(ridge->vertices));
        qh_memfree_(qh, ridge, (int)sizeof(ridgeT), freelistp);
        numold++;
      }else if (neighbor->visitid == samevisitid) {
        qh_setdel(neighbor->ridges, ridge);
        if (qh->traceridge == ridge)
          qh->traceridge= NULL;
        qh_setfree(qh, &(ridge->vertices));
        qh_memfree_(qh, ridge, (int)sizeof(ridgeT), freelistp);
        numold++;
      }else {
        qh_setappend(qh, &newfacet->ridges, ridge);
        numold++;
      }
    }
    if (same->ridges)
      qh_settruncate(qh, same->ridges, 0);
    if (!same->simplicial)
      continue;
    FOREACHneighbor_i_(qh, same) {       /* note: !newfact->simplicial */
      if (neighbor->visitid != samevisitid && neighbor->simplicial) {
        ridge= qh_newridge(qh);
        ridge->vertices= qh_setnew_delnthsorted(qh, same->vertices, qh->hull_dim,
                                                          neighbor_i, 0);
        toporient= (boolT)(same->toporient ^ (neighbor_i & 0x1));
        if (toporient) {
          ridge->top= newfacet;
          ridge->bottom= neighbor;
          ridge->simplicialbot= True;
        }else {
          ridge->top= neighbor;
          ridge->bottom= newfacet;
          ridge->simplicialtop= True;
        }
        qh_setappend(qh, &(newfacet->ridges), ridge);
        qh_setappend(qh, &(neighbor->ridges), ridge);
        if (qh->ridge_id == qh->traceridge_id)
          qh->traceridge= ridge;
        numnew++;
      }
    }
  }

  trace2((qh, qh->ferr, 2033, "qh_mergecycle_ridges: found %d old ridges and %d new ones\n",
             numold, numnew));
} /* mergecycle_ridges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergecycle_vneighbors">-</a>

  qh_mergecycle_vneighbors(qh, samecycle, newfacet )
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
void qh_mergecycle_vneighbors(qhT *qh, facetT *samecycle, facetT *newfacet) {
  facetT *neighbor, **neighborp;
  unsigned int mergeid;
  vertexT *vertex, **vertexp, *apex;
  setT *vertices;

  trace4((qh, qh->ferr, 4035, "qh_mergecycle_vneighbors: update vertex neighbors for newfacet\n"));
  mergeid= qh->visit_id - 1;
  newfacet->visitid= mergeid;
  vertices= qh_basevertices(qh, samecycle); /* temp */
  apex= SETfirstt_(samecycle->vertices, vertexT);
  qh_setappend(qh, &vertices, apex);
  FOREACHvertex_(vertices) {
    vertex->delridge= True;
    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == mergeid)
        SETref_(neighbor)= NULL;
    }
    qh_setcompact(qh, vertex->neighbors);
    qh_setappend(qh, &vertex->neighbors, newfacet);
    if (!SETsecond_(vertex->neighbors)) {
      zinc_(Zcyclevertex);
      trace2((qh, qh->ferr, 2034, "qh_mergecycle_vneighbors: deleted v%d when merging cycle f%d into f%d\n",
        vertex->id, samecycle->id, newfacet->id));
      qh_setdelsorted(newfacet->vertices, vertex);
      vertex->deleted= True;
      qh_setappend(qh, &qh->del_vertices, vertex);
    }
  }
  qh_settempfree(qh, &vertices);
  trace3((qh, qh->ferr, 3005, "qh_mergecycle_vneighbors: merged vertices from cycle f%d into f%d\n",
             samecycle->id, newfacet->id));
} /* mergecycle_vneighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergefacet">-</a>

  qh_mergefacet(qh, facet1, facet2, mergetype, mindist, maxdist, mergeapex )
    merges facet1 into facet2
    mergeapex==qh_MERGEapex if merging new facet into coplanar horizon (optimizes qh_mergesimplex)

  returns:
    qh.max_outside and qh.min_vertex updated
    initializes vertex neighbors on first merge

  note:
    mergetype only used for logging and error reporting

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
    when done, tests facet1 and facet2 for degenerate or redundant neighbors and dupridges
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
    move facet2 to end of qh.newfacet_list
    unless MRGcoplanarhorizon
      test facet2 for redundant neighbors
      test facet1 for degenerate neighbors
      test for redundant facet2
      maybe test for duplicate ridges ('Q15')
    move facet1 to qh.visible_list for later deletion
*/
void qh_mergefacet(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype, realT *mindist, realT *maxdist, boolT mergeapex) {
  boolT traceonce= False;
  vertexT *vertex, **vertexp;
  realT mintwisted, vertexdist;
  realT onemerge;
  int tracerestore=0, nummerge;
  const char *mergename;

  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  else
    mergename= mergetypes[MRGnone];
  if (facet1->tricoplanar || facet2->tricoplanar) {
    if (!qh->TRInormals) {
      qh_fprintf(qh, qh->ferr, 6226, "qhull internal error (qh_mergefacet): merge f%d into f%d for mergetype %d (%s) does not work for tricoplanar facets.  Use option 'Q11'\n",
        facet1->id, facet2->id, mergetype, mergename);
      qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
    }
    if (facet2->tricoplanar) {
      facet2->tricoplanar= False;
      facet2->keepcentrum= False;
    }
  }
  zzinc_(Ztotmerge);
  if (qh->REPORTfreq2 && qh->POSTmerging) {
    if (zzval_(Ztotmerge) > qh->mergereport + qh->REPORTfreq2)
      qh_tracemerging(qh);
  }
#ifndef qh_NOtrace
  if (qh->build_cnt >= qh->RERUN) {
    if (mindist && (-*mindist > qh->TRACEdist || *maxdist > qh->TRACEdist)) {
      tracerestore= 0;
      qh->IStracing= qh->TRACElevel;
      traceonce= True;
      qh_fprintf(qh, qh->ferr, 8075, "qh_mergefacet: ========= trace wide merge #%d(%2.2g) for f%d into f%d for mergetype %d (%s), last point was p%d\n",
          zzval_(Ztotmerge), fmax_(-*mindist, *maxdist), facet1->id, facet2->id, mergetype, mergename, qh->furthest_id);
    }else if (facet1 == qh->tracefacet || facet2 == qh->tracefacet) {
      tracerestore= qh->IStracing;
      qh->IStracing= 4;
      traceonce= True;
      qh_fprintf(qh, qh->ferr, 8076, "qh_mergefacet: ========= trace merge #%d for f%d into f%d for mergetype %d (%s), furthest is p%d\n",
                 zzval_(Ztotmerge), facet1->id, facet2->id, mergetype, mergename, qh->furthest_id);
    }
  }
  if (qh->IStracing >= 2) {
    realT mergemin= -2;
    realT mergemax= -2;

    if (mindist) {
      mergemin= *mindist;
      mergemax= *maxdist;
    }
    qh_fprintf(qh, qh->ferr, 2081, "qh_mergefacet: #%d merge f%d into f%d for merge for mergetype %d (%s), mindist= %2.2g, maxdist= %2.2g, max_outside %2.2g\n",
    zzval_(Ztotmerge), facet1->id, facet2->id, mergetype, mergename, mergemin, mergemax, qh->max_outside);
  }
#endif /* !qh_NOtrace */
  if(!qh->ALLOWwide && mindist) {
    mintwisted= qh_WIDEmaxoutside * qh->ONEmerge;  /* same as qh_merge_twisted and qh_check_maxout (poly2) */
    maximize_(mintwisted, facet1->maxoutside);
    maximize_(mintwisted, facet2->maxoutside);
    if (*maxdist > mintwisted || -*mindist > mintwisted) {
      vertexdist= qh_vertex_bestdist(qh, facet1->vertices);
      onemerge= qh->ONEmerge + qh->DISTround;
      if (vertexdist > mintwisted) {
        qh_fprintf(qh, qh->ferr, 6347, "qhull precision error (qh_mergefacet): wide merge for facet f%d into f%d for mergetype %d (%s).  maxdist %2.2g (%.1fx) mindist %2.2g (%.1fx) vertexdist %2.2g  Allow with 'Q12' (allow-wide)\n",
          facet1->id, facet2->id, mergetype, mergename, *maxdist, *maxdist/onemerge, *mindist, -*mindist/onemerge, vertexdist);
      }else {
        qh_fprintf(qh, qh->ferr, 6348, "qhull precision error (qh_mergefacet): wide merge for pinched facet f%d into f%d for mergetype %d (%s).  maxdist %2.2g (%.fx) mindist %2.2g (%.1fx) vertexdist %2.2g  Allow with 'Q12' (allow-wide)\n",
          facet1->id, facet2->id, mergetype, mergename, *maxdist, *maxdist/onemerge, *mindist, -*mindist/onemerge, vertexdist);
      }
      qh_errexit2(qh, qh_ERRwide, facet1, facet2);
    }
  }
  if (facet1 == facet2 || facet1->visible || facet2->visible) {
    qh_fprintf(qh, qh->ferr, 6099, "qhull internal error (qh_mergefacet): either f%d and f%d are the same or one is a visible facet, mergetype %d (%s)\n",
             facet1->id, facet2->id, mergetype, mergename);
    qh_errexit2(qh, qh_ERRqhull, facet1, facet2);
  }
  if (qh->num_facets - qh->num_visible <= qh->hull_dim + 1) {
    qh_fprintf(qh, qh->ferr, 6227, "qhull topology error: Only %d facets remain.  The input is too degenerate or the convexity constraints are too strong.\n", 
          qh->hull_dim+1);
    if (qh->hull_dim >= 5 && !qh->MERGEexact)
      qh_fprintf(qh, qh->ferr, 8079, "    Option 'Qx' may avoid this problem.\n");
    qh_errexit(qh, qh_ERRtopology, NULL, NULL);
  }
  if (!qh->VERTEXneighbors)
    qh_vertexneighbors(qh);
  qh_makeridges(qh, facet1);
  qh_makeridges(qh, facet2);
  if (qh->IStracing >=4)
    qh_errprint(qh, "MERGING", facet1, facet2, NULL, NULL);
  if (mindist) {
    maximize_(qh->max_outside, *maxdist);
    maximize_(qh->max_vertex, *maxdist);
#if qh_MAXoutside
    maximize_(facet2->maxoutside, *maxdist);
#endif
    minimize_(qh->min_vertex, *mindist);
    if (!facet2->keepcentrum
    && (*maxdist > qh->WIDEfacet || *mindist < -qh->WIDEfacet)) {
      facet2->keepcentrum= True;
      zinc_(Zwidefacet);
    }
  }
  nummerge= facet1->nummerge + facet2->nummerge + 1;
  if (nummerge >= qh_MAXnummerge)
    facet2->nummerge= qh_MAXnummerge;
  else
    facet2->nummerge= (short unsigned int)nummerge; /* limited to 9 bits by qh_MAXnummerge, -Wconversion */
  facet2->newmerge= True;
  facet2->dupridge= False;
  qh_updatetested(qh, facet1, facet2);
  if (qh->hull_dim > 2 && qh_setsize(qh, facet1->vertices) == qh->hull_dim)
    qh_mergesimplex(qh, facet1, facet2, mergeapex);
  else {
    qh->vertex_visit++;
    FOREACHvertex_(facet2->vertices)
      vertex->visitid= qh->vertex_visit;
    if (qh->hull_dim == 2)
      qh_mergefacet2d(qh, facet1, facet2);
    else {
      qh_mergeneighbors(qh, facet1, facet2);
      qh_mergevertices(qh, facet1->vertices, &facet2->vertices);
    }
    qh_mergeridges(qh, facet1, facet2);
    qh_mergevertex_neighbors(qh, facet1, facet2);
    if (!facet2->newfacet)
      qh_newvertices(qh, facet2->vertices);
  }
  if (facet2->coplanarhorizon) {
    zinc_(Zmergeintocoplanar);
  }else if (!facet2->newfacet) {
    zinc_(Zmergeintohorizon);
  }else if (!facet1->newfacet && facet2->newfacet) {
    zinc_(Zmergehorizon);
  }else {
    zinc_(Zmergenew);
  }
  qh_removefacet(qh, facet2);  /* append as a newfacet to end of qh->facet_list */
  qh_appendfacet(qh, facet2);
  facet2->newfacet= True;
  facet2->tested= False;
  qh_tracemerge(qh, facet1, facet2, mergetype);
  if (traceonce) {
    qh_fprintf(qh, qh->ferr, 8080, "qh_mergefacet: end of wide tracing\n");
    qh->IStracing= tracerestore;
  }
  if (mergetype != MRGcoplanarhorizon) {
    trace3((qh, qh->ferr, 3076, "qh_mergefacet: check f%d and f%d for redundant and degenerate neighbors\n",
        facet1->id, facet2->id));
    qh_test_redundant_neighbors(qh, facet2);
    qh_test_degen_neighbors(qh, facet1);  /* after qh_test_redundant_neighbors since MRGdegen more difficult than MRGredundant
                                             and before qh_willdelete which clears facet1.neighbors */
    qh_degen_redundant_facet(qh, facet2); /* may occur in qh_merge_pinchedvertices, e.g., rbox 175 C3,2e-13 D4 t1545228104 | qhull d */
    qh_maybe_duplicateridges(qh, facet2);
  }
  qh_willdelete(qh, facet1, facet2);
} /* mergefacet */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergefacet2d">-</a>

  qh_mergefacet2d(qh, facet1, facet2 )
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
void qh_mergefacet2d(qhT *qh, facetT *facet1, facetT *facet2) {
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
  /* qh_makeridges not needed since neighborB is not degenerate */
  qh_setreplace(qh, neighborB->neighbors, facet1, facet2);
  trace4((qh, qh->ferr, 4036, "qh_mergefacet2d: merged v%d and neighbor f%d of f%d into f%d\n",
       vertexA->id, neighborB->id, facet1->id, facet2->id));
} /* mergefacet2d */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergeneighbors">-</a>

  qh_mergeneighbors(qh, facet1, facet2 )
    merges the neighbors of facet1 into facet2

  notes:
    only called by qh_mergefacet
    qh.hull_dim >= 3
    see qh_mergecycle_neighbors

  design:
    for each neighbor of facet1
      if neighbor is also a neighbor of facet2
        if neighbor is simplicial
          make ridges for later deletion as a degenerate facet
        update its neighbor set
      else
        move the neighbor relation to facet2
    remove the neighbor relation for facet1 and facet2
*/
void qh_mergeneighbors(qhT *qh, facetT *facet1, facetT *facet2) {
  facetT *neighbor, **neighborp;

  trace4((qh, qh->ferr, 4037, "qh_mergeneighbors: merge neighbors of f%d and f%d\n",
          facet1->id, facet2->id));
  qh->visit_id++;
  FOREACHneighbor_(facet2) {
    neighbor->visitid= qh->visit_id;
  }
  FOREACHneighbor_(facet1) {
    if (neighbor->visitid == qh->visit_id) {
      if (neighbor->simplicial)    /* is degen, needs ridges */
        qh_makeridges(qh, neighbor);
      if (SETfirstt_(neighbor->neighbors, facetT) != facet1) /*keep newfacet->horizon*/
        qh_setdel(neighbor->neighbors, facet1);
      else {
        qh_setdel(neighbor->neighbors, facet2);
        qh_setreplace(qh, neighbor->neighbors, facet1, facet2);
      }
    }else if (neighbor != facet2) {
      qh_setappend(qh, &(facet2->neighbors), neighbor);
      qh_setreplace(qh, neighbor->neighbors, facet1, facet2);
    }
  }
  qh_setdel(facet1->neighbors, facet2);  /* here for makeridges */
  qh_setdel(facet2->neighbors, facet1);
} /* mergeneighbors */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergeridges">-</a>

  qh_mergeridges(qh, facet1, facet2 )
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
void qh_mergeridges(qhT *qh, facetT *facet1, facetT *facet2) {
  ridgeT *ridge, **ridgep;

  trace4((qh, qh->ferr, 4038, "qh_mergeridges: merge ridges of f%d into f%d\n",
          facet1->id, facet2->id));
  FOREACHridge_(facet2->ridges) {
    if ((ridge->top == facet1) || (ridge->bottom == facet1)) {
      /* ridge.nonconvex is irrelevant due to merge */
      qh_delridge_merge(qh, ridge);  /* expensive in high-d, could rebuild */
      ridgep--; /* deleted this ridge, repeat with next ridge*/
    }
  }
  FOREACHridge_(facet1->ridges) {
    if (ridge->top == facet1) {
      ridge->top= facet2;
      ridge->simplicialtop= False;
    }else { /* ridge.bottom is facet1 */
      ridge->bottom= facet2;
      ridge->simplicialbot= False;
    }
    qh_setappend(qh, &(facet2->ridges), ridge);
  }
} /* mergeridges */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergesimplex">-</a>

  qh_mergesimplex(qh, facet1, facet2, mergeapex )
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
          delete facet1 from vertex neighbors
          if only in facet2
            add vertex to qh.del_vertices for later deletion
      for each ridge of facet1
        delete ridges between facet1 and facet2
        append other ridges to facet2 after renaming facet to facet2
*/
void qh_mergesimplex(qhT *qh, facetT *facet1, facetT *facet2, boolT mergeapex) {
  vertexT *vertex, **vertexp, *opposite;
  ridgeT *ridge, **ridgep;
  boolT isnew= False;
  facetT *neighbor, **neighborp, *otherfacet;

  if (mergeapex) {
    opposite= SETfirstt_(facet1->vertices, vertexT); /* apex is opposite facet2.  It has the last vertex id */
    trace4((qh, qh->ferr, 4086, "qh_mergesimplex: merge apex v%d of f%d into facet f%d\n",
      opposite->id, facet1->id, facet2->id));
    if (!facet2->newfacet)
      qh_newvertices(qh, facet2->vertices);  /* apex, the first vertex, is already new */
    if (SETfirstt_(facet2->vertices, vertexT) != opposite) {
      qh_setaddnth(qh, &facet2->vertices, 0, opposite);
      isnew= True;
    }
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
    opposite= vertex;
    trace4((qh, qh->ferr, 4039, "qh_mergesimplex: merge opposite v%d of f%d into facet f%d\n",
          opposite->id, facet1->id, facet2->id));
    isnew= qh_addfacetvertex(qh, facet2, opposite);
    if (!facet2->newfacet)
      qh_newvertices(qh, facet2->vertices);
    else if (!opposite->newfacet) {
      qh_removevertex(qh, opposite);
      qh_appendvertex(qh, opposite);
    }
  }
  trace4((qh, qh->ferr, 4040, "qh_mergesimplex: update vertex neighbors of f%d\n",
          facet1->id));
  FOREACHvertex_(facet1->vertices) {
    if (vertex == opposite && isnew)
      qh_setreplace(qh, vertex->neighbors, facet1, facet2);
    else {
      qh_setdel(vertex->neighbors, facet1);
      if (!SETsecond_(vertex->neighbors))
        qh_mergevertex_del(qh, vertex, facet1, facet2);
    }
  }
  trace4((qh, qh->ferr, 4041, "qh_mergesimplex: merge ridges and neighbors of f%d into f%d\n",
          facet1->id, facet2->id));
  qh->visit_id++;
  FOREACHneighbor_(facet2)
    neighbor->visitid= qh->visit_id;
  FOREACHridge_(facet1->ridges) {
    otherfacet= otherfacet_(ridge, facet1);
    if (otherfacet == facet2) {
      /* ridge.nonconvex is irrelevant due to merge */
      qh_delridge_merge(qh, ridge);  /* expensive in high-d, could rebuild */
      ridgep--; /* deleted this ridge, repeat with next ridge*/
      qh_setdel(facet2->neighbors, facet1); /* a simplicial facet may have duplicate neighbors, need to delete each one */
    }else if (otherfacet->dupridge && !qh_setin(otherfacet->neighbors, facet1)) {
      qh_fprintf(qh, qh->ferr, 6356, "qhull topology error (qh_mergesimplex): f%d is a dupridge of f%d, cannot merge f%d into f%d\n",
        facet1->id, otherfacet->id, facet1->id, facet2->id);
      qh_errexit2(qh, qh_ERRqhull, facet1, otherfacet);
    }else {
      trace4((qh, qh->ferr, 4059, "qh_mergesimplex: move r%d with f%d to f%d, new neighbor? %d, maybe horizon? %d\n",
        ridge->id, otherfacet->id, facet2->id, (otherfacet->visitid != qh->visit_id), (SETfirstt_(otherfacet->neighbors, facetT) == facet1)));
      qh_setappend(qh, &facet2->ridges, ridge);
      if (otherfacet->visitid != qh->visit_id) {
        qh_setappend(qh, &facet2->neighbors, otherfacet);
        qh_setreplace(qh, otherfacet->neighbors, facet1, facet2);
        otherfacet->visitid= qh->visit_id;
      }else {
        if (otherfacet->simplicial)    /* is degen, needs ridges */
          qh_makeridges(qh, otherfacet);
        if (SETfirstt_(otherfacet->neighbors, facetT) == facet1) {
          /* keep new, otherfacet->neighbors->horizon */
          qh_setdel(otherfacet->neighbors, facet2);
          qh_setreplace(qh, otherfacet->neighbors, facet1, facet2);
        }else {
          /* facet2 is already a neighbor of otherfacet, by f.visitid */
          qh_setdel(otherfacet->neighbors, facet1);
        }
      }
      if (ridge->top == facet1) { /* wait until after qh_makeridges */
        ridge->top= facet2;
        ridge->simplicialtop= False;
      }else {
        ridge->bottom= facet2;
        ridge->simplicialbot= False;
      }
    }
  }
  trace3((qh, qh->ferr, 3006, "qh_mergesimplex: merged simplex f%d v%d into facet f%d\n",
          facet1->id, opposite->id, facet2->id));
} /* mergesimplex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergevertex_del">-</a>

  qh_mergevertex_del(qh, vertex, facet1, facet2 )
    delete a vertex because of merging facet1 into facet2

  returns:
    deletes vertex from facet2
    adds vertex to qh.del_vertices for later deletion
*/
void qh_mergevertex_del(qhT *qh, vertexT *vertex, facetT *facet1, facetT *facet2) {

  zinc_(Zmergevertex);
  trace2((qh, qh->ferr, 2035, "qh_mergevertex_del: deleted v%d when merging f%d into f%d\n",
          vertex->id, facet1->id, facet2->id));
  qh_setdelsorted(facet2->vertices, vertex);
  vertex->deleted= True;
  qh_setappend(qh, &qh->del_vertices, vertex);
} /* mergevertex_del */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergevertex_neighbors">-</a>

  qh_mergevertex_neighbors(qh, facet1, facet2 )
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
void qh_mergevertex_neighbors(qhT *qh, facetT *facet1, facetT *facet2) {
  vertexT *vertex, **vertexp;

  trace4((qh, qh->ferr, 4042, "qh_mergevertex_neighbors: merge vertex neighborset for f%d into f%d\n",
          facet1->id, facet2->id));
  if (qh->tracevertex) {
    qh_fprintf(qh, qh->ferr, 8081, "qh_mergevertex_neighbors: of f%d into f%d at furthest p%d f0= %p\n",
             facet1->id, facet2->id, qh->furthest_id, qh->tracevertex->neighbors->e[0].p);
    qh_errprint(qh, "TRACE", NULL, NULL, NULL, qh->tracevertex);
  }
  FOREACHvertex_(facet1->vertices) {
    if (vertex->visitid != qh->vertex_visit)
      qh_setreplace(qh, vertex->neighbors, facet1, facet2);
    else {
      qh_setdel(vertex->neighbors, facet1);
      if (!SETsecond_(vertex->neighbors))
        qh_mergevertex_del(qh, vertex, facet1, facet2);
    }
  }
  if (qh->tracevertex)
    qh_errprint(qh, "TRACE", NULL, NULL, NULL, qh->tracevertex);
} /* mergevertex_neighbors */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="mergevertices">-</a>

  qh_mergevertices(qh, vertices1, vertices2 )
    merges the vertex set of facet1 into facet2

  returns:
    replaces vertices2 with merged set
    preserves vertex_visit for qh_mergevertex_neighbors
    updates qh.newvertex_list

  design:
    create a merged set of both vertices (in inverse id order)
*/
void qh_mergevertices(qhT *qh, setT *vertices1, setT **vertices2) {
  int newsize= qh_setsize(qh, vertices1)+qh_setsize(qh, *vertices2) - qh->hull_dim + 1;
  setT *mergedvertices;
  vertexT *vertex, **vertexp, **vertex2= SETaddr_(*vertices2, vertexT);

  mergedvertices= qh_settemp(qh, newsize);
  FOREACHvertex_(vertices1) {
    if (!*vertex2 || vertex->id > (*vertex2)->id)
      qh_setappend(qh, &mergedvertices, vertex);
    else {
      while (*vertex2 && (*vertex2)->id > vertex->id)
        qh_setappend(qh, &mergedvertices, *vertex2++);
      if (!*vertex2 || (*vertex2)->id < vertex->id)
        qh_setappend(qh, &mergedvertices, vertex);
      else
        qh_setappend(qh, &mergedvertices, *vertex2++);
    }
  }
  while (*vertex2)
    qh_setappend(qh, &mergedvertices, *vertex2++);
  if (newsize < qh_setsize(qh, mergedvertices)) {
    qh_fprintf(qh, qh->ferr, 6100, "qhull internal error (qh_mergevertices): facets did not share a ridge\n");
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh_setfree(qh, vertices2);
  *vertices2= mergedvertices;
  qh_settemppop(qh);
} /* mergevertices */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="neighbor_intersections">-</a>

  qh_neighbor_intersections(qh, vertex )
    return intersection of all vertices in vertex->neighbors except for vertex

  returns:
    returns temporary set of vertices
    does not include vertex
    NULL if a neighbor is simplicial
    NULL if empty set

  notes:
    only called by qh_redundant_vertex for qh_reducevertices
      so f.vertices does not contain extraneous vertices that are not in f.ridges
    used for renaming vertices

  design:
    initialize the intersection set with vertices of the first two neighbors
    delete vertex from the intersection
    for each remaining neighbor
      intersect its vertex set with the intersection set
      return NULL if empty
    return the intersection set
*/
setT *qh_neighbor_intersections(qhT *qh, vertexT *vertex) {
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
    intersect= qh_setcopy(qh, neighborA->vertices, 0);
  else
    intersect= qh_vertexintersect_new(qh, neighborA->vertices, neighborB->vertices);
  qh_settemppush(qh, intersect);
  qh_setdelsorted(intersect, vertex);
  FOREACHneighbor_i_(qh, vertex) {
    if (neighbor_i >= 2) {
      zinc_(Zintersectnum);
      qh_vertexintersect(qh, &intersect, neighbor->vertices);
      if (!SETfirst_(intersect)) {
        zinc_(Zintersectfail);
        qh_settempfree(qh, &intersect);
        return NULL;
      }
    }
  }
  trace3((qh, qh->ferr, 3007, "qh_neighbor_intersections: %d vertices in neighbor intersection of v%d\n",
          qh_setsize(qh, intersect), vertex->id));
  return intersect;
} /* neighbor_intersections */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="neighbor_vertices">-</a>

  qh_neighbor_vertices(qh, vertex )
    return neighboring vertices for a vertex (not in subridge)
    assumes vertices have full vertex->neighbors

  returns:
    temporary set of vertices

  notes:
    updates qh.visit_id and qh.vertex_visit
    similar to qh_vertexridges

*/
setT *qh_neighbor_vertices(qhT *qh, vertexT *vertexA, setT *subridge) {
  facetT *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  setT *vertices= qh_settemp(qh, qh->TEMPsize);

  qh->visit_id++;
  FOREACHneighbor_(vertexA)
    neighbor->visitid= qh->visit_id;
  qh->vertex_visit++;
  vertexA->visitid= qh->vertex_visit;
  FOREACHvertex_(subridge) {
    vertex->visitid= qh->vertex_visit;
  }
  FOREACHneighbor_(vertexA) {
    if (*neighborp)   /* no new ridges in last neighbor */
      qh_neighbor_vertices_facet(qh, vertexA, neighbor, &vertices);
  }
  trace3((qh, qh->ferr, 3035, "qh_neighbor_vertices: %d non-subridge, vertex neighbors for v%d\n",
    qh_setsize(qh, vertices), vertexA->id));
  return vertices;
} /* neighbor_vertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="neighbor_vertices_facet">-</a>

  qh_neighbor_vertices_facet(qh, vertex, facet, vertices )
    add neighboring vertices on ridges for vertex in facet
    neighbor->visitid==qh.visit_id if it hasn't been visited
    v.visitid==qh.vertex_visit if it is already in vertices

  returns:
    vertices updated
    sets facet->visitid to qh.visit_id-1

  notes:
    only called by qh_neighbor_vertices
    similar to qh_vertexridges_facet

  design:
    for each ridge of facet
      if ridge of visited neighbor (i.e., unprocessed)
        if vertex in ridge
          append unprocessed vertices of ridge
    mark facet processed
*/
void qh_neighbor_vertices_facet(qhT *qh, vertexT *vertexA, facetT *facet, setT **vertices) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor;
  vertexT *second, *last, *vertex, **vertexp;
  int last_i= qh->hull_dim-2, count= 0;
  boolT isridge;

  if (facet->simplicial) {
    FOREACHvertex_(facet->vertices) {
      if (vertex->visitid != qh->vertex_visit) {
        vertex->visitid= qh->vertex_visit;
        qh_setappend(qh, vertices, vertex);
        count++;
      }
    }
  }else {
    FOREACHridge_(facet->ridges) {
      neighbor= otherfacet_(ridge, facet);
      if (neighbor->visitid == qh->visit_id) {
        isridge= False;
        if (SETfirst_(ridge->vertices) == vertexA) {
          isridge= True;
        }else if (last_i > 2) {
          second= SETsecondt_(ridge->vertices, vertexT);
          last= SETelemt_(ridge->vertices, last_i, vertexT);
          if (second->id >= vertexA->id && last->id <= vertexA->id) { /* vertices inverse sorted by id */
            if (second == vertexA || last == vertexA)
              isridge= True;
            else if (qh_setin(ridge->vertices, vertexA))
              isridge= True;
          }
        }else if (SETelem_(ridge->vertices, last_i) == vertexA) {
          isridge= True;
        }else if (last_i > 1 && SETsecond_(ridge->vertices) == vertexA) {
          isridge= True;
        }
        if (isridge) {
          FOREACHvertex_(ridge->vertices) {
            if (vertex->visitid != qh->vertex_visit) {
              vertex->visitid= qh->vertex_visit;
              qh_setappend(qh, vertices, vertex);
              count++;
            }
          }
        }
      }
    }
  }
  facet->visitid= qh->visit_id-1;
  if (count) {
    trace4((qh, qh->ferr, 4079, "qh_neighbor_vertices_facet: found %d vertex neighbors for v%d in f%d (simplicial? %d)\n",
      count, vertexA->id, facet->id, facet->simplicial));
  }
} /* neighbor_vertices_facet */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="newvertices">-</a>

  qh_newvertices(qh, vertices )
    add vertices to end of qh.vertex_list (marks as new vertices)

  returns:
    vertices on qh.newvertex_list
    vertex->newfacet set
*/
void qh_newvertices(qhT *qh, setT *vertices) {
  vertexT *vertex, **vertexp;

  FOREACHvertex_(vertices) {
    if (!vertex->newfacet) {
      qh_removevertex(qh, vertex);
      qh_appendvertex(qh, vertex);
    }
  }
} /* newvertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="next_vertexmerge">-</a>

  qh_next_vertexmerge(qh )
    return next vertex merge from qh.vertex_mergeset

  returns:
    vertex merge either MRGvertices or MRGsubridge
    drops merges of deleted vertices

  notes:
    called from qh_merge_pinchedvertices
*/
mergeT *qh_next_vertexmerge(qhT *qh /* qh.vertex_mergeset */) {
  mergeT *merge;
  int merge_i, merge_n, best_i= -1;
  realT bestdist= REALmax;

  FOREACHmerge_i_(qh, qh->vertex_mergeset) {
    if (!merge->vertex1 || !merge->vertex2) {
      qh_fprintf(qh, qh->ferr, 6299, "qhull internal error (qh_next_vertexmerge): expecting two vertices for vertex merge.  Got v%d v%d and optional f%d\n",
        getid_(merge->vertex1), getid_(merge->vertex2), getid_(merge->facet1));
      qh_errexit(qh, qh_ERRqhull, merge->facet1, NULL);
    }
    if (merge->vertex1->deleted || merge->vertex2->deleted) {
      trace3((qh, qh->ferr, 3030, "qh_next_vertexmerge: drop merge of v%d (del? %d) into v%d (del? %d) due to deleted vertex of r%d and r%d\n",
        merge->vertex1->id, merge->vertex1->deleted, merge->vertex2->id, merge->vertex2->deleted, getid_(merge->ridge1), getid_(merge->ridge2)));
      qh_drop_mergevertex(qh, merge);
      qh_setdelnth(qh, qh->vertex_mergeset, merge_i);
      merge_i--; merge_n--;
      qh_memfree(qh, merge, (int)sizeof(mergeT));
    }else if (merge->distance < bestdist) {
      bestdist= merge->distance;
      best_i= merge_i;
    }
  }
  merge= NULL;
  if (best_i >= 0) {
    merge= SETelemt_(qh->vertex_mergeset, best_i, mergeT);
    if (bestdist/qh->ONEmerge > qh_WIDEpinched) {
      if (merge->mergetype==MRGvertices) {
        if (merge->ridge1->top == merge->ridge2->bottom && merge->ridge1->bottom == merge->ridge2->top)
          qh_fprintf(qh, qh->ferr, 6391, "qhull topology error (qh_next_vertexmerge): no nearly adjacent vertices to resolve opposite oriented ridges r%d and r%d in f%d and f%d.  Nearest v%d and v%d dist %2.2g (%.1fx)\n",
            merge->ridge1->id, merge->ridge2->id, merge->ridge1->top->id, merge->ridge1->bottom->id, merge->vertex1->id, merge->vertex2->id, bestdist, bestdist/qh->ONEmerge);
        else
          qh_fprintf(qh, qh->ferr, 6381, "qhull topology error (qh_next_vertexmerge): no nearly adjacent vertices to resolve duplicate ridges r%d and r%d.  Nearest v%d and v%d dist %2.2g (%.1fx)\n",
            merge->ridge1->id, merge->ridge2->id, merge->vertex1->id, merge->vertex2->id, bestdist, bestdist/qh->ONEmerge);
      }else {
        qh_fprintf(qh, qh->ferr, 6208, "qhull topology error (qh_next_vertexmerge): no nearly adjacent vertices to resolve dupridge.  Nearest v%d and v%d dist %2.2g (%.1fx)\n",
          merge->vertex1->id, merge->vertex2->id, bestdist, bestdist/qh->ONEmerge);
      }
      /* it may be possible to find a different vertex, after other vertex merges have occurred */
      qh_errexit(qh, qh_ERRtopology, NULL, merge->ridge1);
    }
    qh_setdelnth(qh, qh->vertex_mergeset, best_i);
  }
  return merge;
} /* next_vertexmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="opposite_horizonfacet">-</a>

  qh_opposite_horizonfacet(qh, merge, opposite )
    return horizon facet for one of the merge facets, and its opposite vertex across the ridge
    assumes either facet1 or facet2 of merge is 'mergehorizon'
    assumes both facets are simplicial facets on qh.new_facetlist

  returns:
    horizon facet and opposite vertex

  notes:
    called by qh_getpinchedmerges
*/
facetT *qh_opposite_horizonfacet(qhT *qh, mergeT *merge, vertexT **opposite) {
  facetT *facet, *horizon, *otherfacet;
  int neighbor_i;

  if (!merge->facet1->simplicial || !merge->facet2->simplicial || (!merge->facet1->mergehorizon && !merge->facet2->mergehorizon)) {
    qh_fprintf(qh, qh->ferr, 6273, "qhull internal error (qh_opposite_horizonfacet): expecting merge of simplicial facets, at least one of which is mergehorizon.  Either simplicial or mergehorizon is wrong\n");
    qh_errexit2(qh, qh_ERRqhull, merge->facet1, merge->facet2);
  }
  if (merge->facet1->mergehorizon) {
    facet= merge->facet1;
    otherfacet= merge->facet2;
  }else {
    facet= merge->facet2;
    otherfacet= merge->facet1;
  }
  horizon= SETfirstt_(facet->neighbors, facetT);
  neighbor_i= qh_setindex(otherfacet->neighbors, facet);
  if (neighbor_i==-1)
    neighbor_i= qh_setindex(otherfacet->neighbors, qh_MERGEridge);
  if (neighbor_i==-1) {
    qh_fprintf(qh, qh->ferr, 6238, "qhull internal error (qh_opposite_horizonfacet): merge facet f%d not connected to mergehorizon f%d\n",
      otherfacet->id, facet->id);
    qh_errexit2(qh, qh_ERRqhull, otherfacet, facet);
  }
  *opposite= SETelemt_(otherfacet->vertices, neighbor_i, vertexT);
  return horizon;
} /* opposite_horizonfacet */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="reducevertices">-</a>

  qh_reducevertices(qh)
    reduce extra vertices, shared vertices, and redundant vertices
    facet->newmerge is set if merged since last call
    vertex->delridge is set if vertex was on a deleted ridge
    if !qh.MERGEvertices, only removes extra vertices

  returns:
    True if also merged degen_redundant facets
    vertices are renamed if possible
    clears facet->newmerge and vertex->delridge

  notes:
    called by qh_all_merges and qh_postmerge
    ignored if 2-d

  design:
    merge any degenerate or redundant facets
    repeat until no more degenerate or redundant facets
      for each newly merged facet
        remove extra vertices
      if qh.MERGEvertices
        for each newly merged facet
          for each vertex
            if vertex was on a deleted ridge
              rename vertex if it is shared
        for each new, undeleted vertex
          remove delridge flag
          if vertex is redundant
            merge degenerate or redundant facets
*/
boolT qh_reducevertices(qhT *qh) {
  int numshare=0, numrename= 0;
  boolT degenredun= False;
  facetT *newfacet;
  vertexT *vertex, **vertexp;

  if (qh->hull_dim == 2)
    return False;
  trace2((qh, qh->ferr, 2101, "qh_reducevertices: reduce extra vertices, shared vertices, and redundant vertices\n"));
  if (qh_merge_degenredundant(qh))
    degenredun= True;
LABELrestart:
  FORALLnew_facets {
    if (newfacet->newmerge) {
      if (!qh->MERGEvertices)
        newfacet->newmerge= False;
      if (qh_remove_extravertices(qh, newfacet)) {
        qh_degen_redundant_facet(qh, newfacet);
        if (qh_merge_degenredundant(qh)) {
          degenredun= True;
          goto LABELrestart;
        }
      }
    }
  }
  if (!qh->MERGEvertices)
    return False;
  FORALLnew_facets {
    if (newfacet->newmerge) {
      newfacet->newmerge= False;
      FOREACHvertex_(newfacet->vertices) {
        if (vertex->delridge) {
          if (qh_rename_sharedvertex(qh, vertex, newfacet)) {
            numshare++;
            if (qh_merge_degenredundant(qh)) {
              degenredun= True;
              goto LABELrestart;
            }
            vertexp--; /* repeat since deleted vertex */
          }
        }
      }
    }
  }
  FORALLvertex_(qh->newvertex_list) {
    if (vertex->delridge && !vertex->deleted) {
      vertex->delridge= False;
      if (qh->hull_dim >= 4 && qh_redundant_vertex(qh, vertex)) {
        numrename++;
        if (qh_merge_degenredundant(qh)) {
          degenredun= True;
          goto LABELrestart;
        }
      }
    }
  }
  trace1((qh, qh->ferr, 1014, "qh_reducevertices: renamed %d shared vertices and %d redundant vertices. Degen? %d\n",
          numshare, numrename, degenredun));
  return degenredun;
} /* reducevertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="redundant_vertex">-</a>

  qh_redundant_vertex(qh, vertex )
    rename a redundant vertex if qh_find_newvertex succeeds
    assumes vertices have full vertex->neighbors

  returns:
    if find a replacement vertex
      returns new vertex
      qh_renamevertex sets vertex->deleted for redundant vertex

  notes:
    only called by qh_reducevertices for vertex->delridge and hull_dim >= 4
    may add degenerate facets to qh.facet_mergeset
    doesn't change vertex->neighbors or create redundant facets

  design:
    intersect vertices of all facet neighbors of vertex
    determine ridges for these vertices
    if find a new vertex for vertex among these ridges and vertices
      rename vertex to the new vertex
*/
vertexT *qh_redundant_vertex(qhT *qh, vertexT *vertex) {
  vertexT *newvertex= NULL;
  setT *vertices, *ridges;

  trace3((qh, qh->ferr, 3008, "qh_redundant_vertex: check if v%d from a deleted ridge can be renamed\n", vertex->id));
  if ((vertices= qh_neighbor_intersections(qh, vertex))) {
    ridges= qh_vertexridges(qh, vertex, !qh_ALL);
    if ((newvertex= qh_find_newvertex(qh, vertex, vertices, ridges))) {
      zinc_(Zrenameall);
      qh_renamevertex(qh, vertex, newvertex, ridges, NULL, NULL); /* ridges invalidated */
    }
    qh_settempfree(qh, &ridges);
    qh_settempfree(qh, &vertices);
  }
  return newvertex;
} /* redundant_vertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="remove_extravertices">-</a>

  qh_remove_extravertices(qh, facet )
    remove extra vertices from non-simplicial facets

  returns:
    returns True if it finds them
      deletes facet from vertex neighbors
      facet may be redundant (test with qh_degen_redundant)

  notes:
    called by qh_renamevertex and qh_reducevertices
    a merge (qh_reducevertices) or qh_renamevertex may drop all ridges for a vertex in a facet

  design:
    for each vertex in facet
      if vertex not in a ridge (i.e., no longer used)
        delete vertex from facet
        delete facet from vertex's neighbors
        unless vertex in another facet
          add vertex to qh.del_vertices for later deletion
*/
boolT qh_remove_extravertices(qhT *qh, facetT *facet) {
  ridgeT *ridge, **ridgep;
  vertexT *vertex, **vertexp;
  boolT foundrem= False;

  if (facet->simplicial) {
    return False;
  }
  trace4((qh, qh->ferr, 4043, "qh_remove_extravertices: test non-simplicial f%d for extra vertices\n",
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
      if (!qh_setsize(qh, vertex->neighbors)) {
        vertex->deleted= True;
        qh_setappend(qh, &qh->del_vertices, vertex);
        zinc_(Zremvertexdel);
        trace2((qh, qh->ferr, 2036, "qh_remove_extravertices: v%d deleted because it's lost all ridges\n", vertex->id));
      }else
        trace3((qh, qh->ferr, 3009, "qh_remove_extravertices: v%d removed from f%d because it's lost all ridges\n", vertex->id, facet->id));
      vertexp--; /*repeat*/
    }
  }
  return foundrem;
} /* remove_extravertices */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="remove_mergetype">-</a>

  qh_remove_mergetype(qh, mergeset, mergetype )
    Remove mergetype merges from mergeset

  notes:
    Does not preserve order
*/
void qh_remove_mergetype(qhT *qh, setT *mergeset, mergeType type) {
  mergeT *merge;
  int merge_i, merge_n;

  FOREACHmerge_i_(qh, mergeset) {
    if (merge->mergetype == type) {
        trace3((qh, qh->ferr, 3037, "qh_remove_mergetype: remove merge f%d f%d v%d v%d r%d r%d dist %2.2g type %d",
            getid_(merge->facet1), getid_(merge->facet2), getid_(merge->vertex1), getid_(merge->vertex2), getid_(merge->ridge1), getid_(merge->ridge2), merge->distance, type));
        qh_setdelnth(qh, mergeset, merge_i);
        merge_i--; merge_n--;  /* repeat with next merge */
    }
  }
} /* remove_mergetype */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="rename_adjacentvertex">-</a>

  qh_rename_adjacentvertex(qh, oldvertex, newvertex )
    renames oldvertex as newvertex.  Must be adjacent (i.e., in the same subridge)
    no-op if either vertex is deleted

  notes:
    called from qh_merge_pinchedvertices

  design:
    for all neighbors of oldvertex
      if simplicial, rename oldvertex to newvertex and drop if degenerate
      if needed, add oldvertex neighbor to newvertex
    determine ridges for oldvertex
    rename oldvertex as newvertex in ridges (qh_renamevertex)
*/
void qh_rename_adjacentvertex(qhT *qh, vertexT *oldvertex, vertexT *newvertex, realT dist) {
  setT *ridges;
  facetT *neighbor, **neighborp, *maxfacet= NULL;
  ridgeT *ridge, **ridgep;
  boolT istrace= False;
  int oldsize= qh_setsize(qh, oldvertex->neighbors);
  int newsize= qh_setsize(qh, newvertex->neighbors);
  coordT maxdist2= -REALmax, dist2;

  if (qh->IStracing >= 4 || oldvertex->id == qh->tracevertex_id || newvertex->id == qh->tracevertex_id) {
    istrace= True;
  }
  zzinc_(Ztotmerge);
  if (istrace) {
    qh_fprintf(qh, qh->ferr, 2071, "qh_rename_adjacentvertex: merge #%d rename v%d (%d neighbors) to v%d (%d neighbors) dist %2.2g\n",
      zzval_(Ztotmerge), oldvertex->id, oldsize, newvertex->id, newsize, dist);
  }
  if (oldvertex->deleted || newvertex->deleted) {
    if (istrace || qh->IStracing >= 2) {
      qh_fprintf(qh, qh->ferr, 2072, "qh_rename_adjacentvertex: ignore rename.  Either v%d (%d) or v%d (%d) is deleted\n",
        oldvertex->id, oldvertex->deleted, newvertex->id, newvertex->deleted);
    }
    return;
  }
  if (oldsize == 0 || newsize == 0) {
    qh_fprintf(qh, qh->ferr, 2072, "qhull internal error (qh_rename_adjacentvertex): expecting neighbor facets for v%d and v%d.  Got %d and %d neighbors resp.\n",
      oldvertex->id, newvertex->id, oldsize, newsize);
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  FOREACHneighbor_(oldvertex) {
    if (neighbor->simplicial) {
      if (qh_setin(neighbor->vertices, newvertex)) {
        if (istrace || qh->IStracing >= 2) {
          qh_fprintf(qh, qh->ferr, 2070, "qh_rename_adjacentvertex: simplicial f%d contains old v%d and new v%d.  Will be marked degenerate by qh_renamevertex\n",
            neighbor->id, oldvertex->id, newvertex->id);
        }
        qh_makeridges(qh, neighbor); /* no longer simplicial, nummerge==0, skipped by qh_maybe_duplicateridge */
      }else {
        qh_replacefacetvertex(qh, neighbor, oldvertex, newvertex);
        qh_setunique(qh, &newvertex->neighbors, neighbor);
        qh_newvertices(qh, neighbor->vertices);  /* for qh_update_vertexneighbors of vertex neighbors */
      }
    }
  }
  ridges= qh_vertexridges(qh, oldvertex, qh_ALL);
  if (istrace) {
    FOREACHridge_(ridges) {
      qh_printridge(qh, qh->ferr, ridge);
    }
  }
  FOREACHneighbor_(oldvertex) {
    if (!neighbor->simplicial){
      qh_addfacetvertex(qh, neighbor, newvertex);
      qh_setunique(qh, &newvertex->neighbors, neighbor);
      qh_newvertices(qh, neighbor->vertices);  /* for qh_update_vertexneighbors of vertex neighbors */
      if (qh->newfacet_list == qh->facet_tail) {
        qh_removefacet(qh, neighbor);  /* add a neighbor to newfacet_list so that qh_partitionvisible has a newfacet */
        qh_appendfacet(qh, neighbor);
        neighbor->newfacet= True;
      }
    }
  }
  qh_renamevertex(qh, oldvertex, newvertex, ridges, NULL, NULL);  /* ridges invalidated */
  if (oldvertex->deleted && !oldvertex->partitioned) {
    FOREACHneighbor_(newvertex) {
      if (!neighbor->visible) {
        qh_distplane(qh, oldvertex->point, neighbor, &dist2);
        if (dist2>maxdist2) {
          maxdist2= dist2;
          maxfacet= neighbor;
        }
      }
    }
    trace2((qh, qh->ferr, 2096, "qh_rename_adjacentvertex: partition old p%d(v%d) as a coplanar point for furthest f%d dist %2.2g.  Maybe repartition later (QH0031)\n",
      qh_pointid(qh, oldvertex->point), oldvertex->id, maxfacet->id, maxdist2))
    qh_partitioncoplanar(qh, oldvertex->point, maxfacet, NULL, !qh_ALL);  /* faster with maxdist2, otherwise duplicates distance tests from maxdist2/dist2 */
    oldvertex->partitioned= True;
  }
  qh_settempfree(qh, &ridges);
} /* rename_adjacentvertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="rename_sharedvertex">-</a>

  qh_rename_sharedvertex(qh, vertex, facet )
    detect and rename if shared vertex in facet
    vertices have full ->neighbors

  returns:
    newvertex or NULL
    the vertex may still exist in other facets (i.e., a neighbor was pinched)
    does not change facet->neighbors
    updates vertex->neighbors

  notes:
    only called by qh_reducevertices after qh_remove_extravertices
       so f.vertices does not contain extraneous vertices
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
vertexT *qh_rename_sharedvertex(qhT *qh, vertexT *vertex, facetT *facet) {
  facetT *neighbor, **neighborp, *neighborA= NULL;
  setT *vertices, *ridges;
  vertexT *newvertex= NULL;

  if (qh_setsize(qh, vertex->neighbors) == 2) {
    neighborA= SETfirstt_(vertex->neighbors, facetT);
    if (neighborA == facet)
      neighborA= SETsecondt_(vertex->neighbors, facetT);
  }else if (qh->hull_dim == 3)
    return NULL;
  else {
    qh->visit_id++;
    FOREACHneighbor_(facet)
      neighbor->visitid= qh->visit_id;
    FOREACHneighbor_(vertex) {
      if (neighbor->visitid == qh->visit_id) {
        if (neighborA)
          return NULL;
        neighborA= neighbor;
      }
    }
  }
  if (!neighborA) {
    qh_fprintf(qh, qh->ferr, 6101, "qhull internal error (qh_rename_sharedvertex): v%d's neighbors not in f%d\n",
        vertex->id, facet->id);
    qh_errprint(qh, "ERRONEOUS", facet, NULL, NULL, vertex);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  if (neighborA) { /* avoid warning */
    /* the vertex is shared by facet and neighborA */
    ridges= qh_settemp(qh, qh->TEMPsize);
    neighborA->visitid= ++qh->visit_id;
    qh_vertexridges_facet(qh, vertex, facet, &ridges);
    trace2((qh, qh->ferr, 2037, "qh_rename_sharedvertex: p%d(v%d) is shared by f%d(%d ridges) and f%d\n",
      qh_pointid(qh, vertex->point), vertex->id, facet->id, qh_setsize(qh, ridges), neighborA->id));
    zinc_(Zintersectnum);
    vertices= qh_vertexintersect_new(qh, facet->vertices, neighborA->vertices);
    qh_setdel(vertices, vertex);
    qh_settemppush(qh, vertices);
    if ((newvertex= qh_find_newvertex(qh, vertex, vertices, ridges)))
      qh_renamevertex(qh, vertex, newvertex, ridges, facet, neighborA);  /* ridges invalidated */
    qh_settempfree(qh, &vertices);
    qh_settempfree(qh, &ridges);
  }
  return newvertex;
} /* rename_sharedvertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="renameridgevertex">-</a>

  qh_renameridgevertex(qh, ridge, oldvertex, newvertex )
    renames oldvertex as newvertex in ridge

  returns:
    True if renames oldvertex
    False if deleted the ridge

  notes:
    called by qh_renamevertex
    caller sets newvertex->delridge for deleted ridges (qh_reducevertices)

  design:
    delete oldvertex from ridge
    if newvertex already in ridge
      copy ridge->noconvex to another ridge if possible
      delete the ridge
    else
      insert newvertex into the ridge
      adjust the ridge's orientation
*/
boolT qh_renameridgevertex(qhT *qh, ridgeT *ridge, vertexT *oldvertex, vertexT *newvertex) {
  int nth= 0, oldnth;
  facetT *temp;
  vertexT *vertex, **vertexp;

  oldnth= qh_setindex(ridge->vertices, oldvertex);
  if (oldnth < 0) {
    qh_fprintf(qh, qh->ferr, 6424, "qhull internal error (qh_renameridgevertex): oldvertex v%d not found in r%d.  Cannot rename to v%d\n",
        oldvertex->id, ridge->id, newvertex->id);
    qh_errexit(qh, qh_ERRqhull, NULL, ridge);
  }
  qh_setdelnthsorted(qh, ridge->vertices, oldnth);
  FOREACHvertex_(ridge->vertices) {
    if (vertex == newvertex) {
      zinc_(Zdelridge);
      if (ridge->nonconvex) /* only one ridge has nonconvex set */
        qh_copynonconvex(qh, ridge);
      trace2((qh, qh->ferr, 2038, "qh_renameridgevertex: ridge r%d deleted.  It contained both v%d and v%d\n",
        ridge->id, oldvertex->id, newvertex->id));
      qh_delridge_merge(qh, ridge); /* ridge.vertices deleted */
      return False;
    }
    if (vertex->id < newvertex->id)
      break;
    nth++;
  }
  qh_setaddnth(qh, &ridge->vertices, nth, newvertex);
  ridge->simplicialtop= False;
  ridge->simplicialbot= False;
  if (abs(oldnth - nth)%2) {
    trace3((qh, qh->ferr, 3010, "qh_renameridgevertex: swapped the top and bottom of ridge r%d\n",
            ridge->id));
    temp= ridge->top;
    ridge->top= ridge->bottom;
    ridge->bottom= temp;
  }
  return True;
} /* renameridgevertex */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="renamevertex">-</a>

  qh_renamevertex(qh, oldvertex, newvertex, ridges, oldfacet, neighborA )
    renames oldvertex as newvertex in ridges of non-simplicial neighbors
    set oldfacet/neighborA if oldvertex is shared between two facets (qh_rename_sharedvertex)
    otherwise qh_redundant_vertex or qh_rename_adjacentvertex

  returns:
    if oldfacet and multiple neighbors, oldvertex may still exist afterwards
    otherwise sets oldvertex->deleted for later deletion
    one or more ridges maybe deleted
    ridges is invalidated
    merges may be added to degen_mergeset via qh_maydropneighbor or qh_degen_redundant_facet

  notes:
    qh_rename_sharedvertex can not change neighbors of newvertex (since it's a subset)
    qh_redundant_vertex due to vertex->delridge for qh_reducevertices
    qh_rename_adjacentvertex for complete renames

  design:
    for each ridge in ridges
      rename oldvertex to newvertex and delete degenerate ridges
    if oldfacet not defined
      for each non-simplicial neighbor of oldvertex
        delete oldvertex from neighbor's vertices
        remove extra vertices from neighbor
      add oldvertex to qh.del_vertices
    else if oldvertex only between oldfacet and neighborA
      delete oldvertex from oldfacet and neighborA
      add oldvertex to qh.del_vertices
    else oldvertex is in oldfacet and neighborA and other facets (i.e., pinched)
      delete oldvertex from oldfacet
      delete oldfacet from old vertex's neighbors
      remove extra vertices (e.g., oldvertex) from neighborA
*/
void qh_renamevertex(qhT *qh, vertexT *oldvertex, vertexT *newvertex, setT *ridges, facetT *oldfacet, facetT *neighborA) {
  facetT *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
  int topsize, bottomsize;
  boolT istrace= False;

#ifndef qh_NOtrace
  if (qh->IStracing >= 2 || oldvertex->id == qh->tracevertex_id ||
        newvertex->id == qh->tracevertex_id) {
    istrace= True;
    qh_fprintf(qh, qh->ferr, 2086, "qh_renamevertex: rename v%d to v%d in %d ridges with old f%d and neighbor f%d\n",
      oldvertex->id, newvertex->id, qh_setsize(qh, ridges), getid_(oldfacet), getid_(neighborA));
  }
#endif
  FOREACHridge_(ridges) {
    if (qh_renameridgevertex(qh, ridge, oldvertex, newvertex)) { /* ridge is deleted if False, invalidating ridges */
      topsize= qh_setsize(qh, ridge->top->vertices);
      bottomsize= qh_setsize(qh, ridge->bottom->vertices);
      if (topsize < qh->hull_dim || (topsize == qh->hull_dim && !ridge->top->simplicial && qh_setin(ridge->top->vertices, newvertex))) {
        trace4((qh, qh->ferr, 4070, "qh_renamevertex: ignore duplicate check for r%d.  top f%d (size %d) will be degenerate after rename v%d to v%d\n",
          ridge->id, ridge->top->id, topsize, oldvertex->id, newvertex->id));
      }else if (bottomsize < qh->hull_dim || (bottomsize == qh->hull_dim && !ridge->bottom->simplicial && qh_setin(ridge->bottom->vertices, newvertex))) {
        trace4((qh, qh->ferr, 4071, "qh_renamevertex: ignore duplicate check for r%d.  bottom f%d (size %d) will be degenerate after rename v%d to v%d\n",
          ridge->id, ridge->bottom->id, bottomsize, oldvertex->id, newvertex->id));
      }else
        qh_maybe_duplicateridge(qh, ridge);
    }
  }
  if (!oldfacet) {
    /* stat Zrenameall or Zpinchduplicate */
    if (istrace)
      qh_fprintf(qh, qh->ferr, 2087, "qh_renamevertex: renaming v%d to v%d in several facets for qh_redundant_vertex or MRGsubridge\n",
               oldvertex->id, newvertex->id);
    FOREACHneighbor_(oldvertex) {
      if (neighbor->simplicial) {
        qh_degen_redundant_facet(qh, neighbor); /* e.g., rbox 175 C3,2e-13 D4 t1545235541 | qhull d */
      }else {
        if (istrace)
          qh_fprintf(qh, qh->ferr, 4080, "qh_renamevertex: rename vertices in non-simplicial neighbor f%d of v%d\n", neighbor->id, oldvertex->id);
        qh_maydropneighbor(qh, neighbor);
        qh_setdelsorted(neighbor->vertices, oldvertex); /* if degenerate, qh_degen_redundant_facet will add to mergeset */
        if (qh_remove_extravertices(qh, neighbor))
          neighborp--; /* neighbor deleted from oldvertex neighborset */
        qh_degen_redundant_facet(qh, neighbor); /* either direction may be redundant, faster if combine? */
        qh_test_redundant_neighbors(qh, neighbor);
        qh_test_degen_neighbors(qh, neighbor);
      }
    }
    if (!oldvertex->deleted) {
      oldvertex->deleted= True;
      qh_setappend(qh, &qh->del_vertices, oldvertex);
    }
  }else if (qh_setsize(qh, oldvertex->neighbors) == 2) {
    zinc_(Zrenameshare);
    if (istrace)
      qh_fprintf(qh, qh->ferr, 3039, "qh_renamevertex: renaming v%d to v%d in oldfacet f%d for qh_rename_sharedvertex\n",
               oldvertex->id, newvertex->id, oldfacet->id);
    FOREACHneighbor_(oldvertex) {
      qh_setdelsorted(neighbor->vertices, oldvertex);
      qh_degen_redundant_facet(qh, neighbor);
    }
    oldvertex->deleted= True;
    qh_setappend(qh, &qh->del_vertices, oldvertex);
  }else {
    zinc_(Zrenamepinch);
    if (istrace || qh->IStracing >= 1)
      qh_fprintf(qh, qh->ferr, 3040, "qh_renamevertex: renaming pinched v%d to v%d between f%d and f%d\n",
               oldvertex->id, newvertex->id, oldfacet->id, neighborA->id);
    qh_setdelsorted(oldfacet->vertices, oldvertex);
    qh_setdel(oldvertex->neighbors, oldfacet);
    if (qh_remove_extravertices(qh, neighborA))
      qh_degen_redundant_facet(qh, neighborA);
  }
  if (oldfacet)
    qh_degen_redundant_facet(qh, oldfacet);
} /* renamevertex */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_appendmerge">-</a>

  qh_test_appendmerge(qh, facet, neighbor, simplicial )
    test convexity and append to qh.facet_mergeset if non-convex
    if pre-merging,
      no-op if qh.SKIPconvex, or qh.MERGEexact and coplanar
    if simplicial, assumes centrum test is valid (e.g., adjacent, simplicial new facets)

  returns:
    true if appends facet/neighbor to qh.facet_mergeset
    sets facet->center as needed
    does not change facet->seen

  notes:
    called from qh_getmergeset_initial, qh_getmergeset, and qh_test_vneighbors
    must be at least as strong as qh_checkconvex (poly2_r.c)
    assumes !f.flipped

  design:
    exit if qh.SKIPconvex ('Q0') and !qh.POSTmerging
    if qh.cos_max ('An') is defined and merging coplanars
      if the angle between facet normals is too shallow
        append an angle-coplanar merge to qh.mergeset
        return True
    test convexity of facet and neighbor
*/
boolT qh_test_appendmerge(qhT *qh, facetT *facet, facetT *neighbor, boolT simplicial) {
  realT angle= -REALmax;
  boolT okangle= False;

  if (qh->SKIPconvex && !qh->POSTmerging)
    return False;
  if (qh->cos_max < REALmax/2 && (!qh->MERGEexact || qh->POSTmerging)) {
    angle= qh_getangle(qh, facet->normal, neighbor->normal);
    okangle= True;
    zinc_(Zangletests);
    if (angle > qh->cos_max) {
      zinc_(Zcoplanarangle);
      qh_appendmergeset(qh, facet, neighbor, MRGanglecoplanar, 0.0, angle);
      trace2((qh, qh->ferr, 2039, "qh_test_appendmerge: coplanar angle %4.4g between f%d and f%d\n",
         angle, facet->id, neighbor->id));
      return True;
    }
  }
  if (simplicial || qh->hull_dim <= 3)
    return qh_test_centrum_merge(qh, facet, neighbor, angle, okangle);
  else
    return qh_test_nonsimplicial_merge(qh, facet, neighbor, angle, okangle);
} /* test_appendmerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_centrum_merge">-</a>

  qh_test_centrum_merge(qh, facet, neighbor, angle, okangle )
    test centrum convexity and append non-convex facets to qh.facet_mergeset
    'angle' is angle between facets if okangle is true, otherwise use 0.0

  returns:
    true if append facet/neighbor to qh.facet_mergeset
    sets facet->center as needed
    does not change facet->seen

  notes:
    called from test_appendmerge if adjacent simplicial facets or 2-d/3-d
    at least as strict as qh_checkconvex, including qh.DISTround ('En' and 'Rn')

  design:
    make facet's centrum if needed
    if facet's centrum is above the neighbor (qh.centrum_radius)
      set isconcave

    if facet's centrum is not below the neighbor (-qh.centrum_radius)
      set iscoplanar
    make neighbor's centrum if needed
    if neighbor's centrum is above the facet
      set isconcave
    else if neighbor's centrum is not below the facet
      set iscoplanar
    if isconcave or iscoplanar and merging coplanars
      get angle if needed (qh.ANGLEmerge 'An')
      append concave-coplanar, concave ,or coplanar merge to qh.mergeset
*/
boolT qh_test_centrum_merge(qhT *qh, facetT *facet, facetT *neighbor, realT angle, boolT okangle) {
  coordT dist, dist2, mergedist;
  boolT isconcave= False, iscoplanar= False;

  if (!facet->center)
    facet->center= qh_getcentrum(qh, facet);
  zzinc_(Zcentrumtests);
  qh_distplane(qh, facet->center, neighbor, &dist);
  if (dist > qh->centrum_radius)
    isconcave= True;
  else if (dist >= -qh->centrum_radius)
    iscoplanar= True;
  if (!neighbor->center)
    neighbor->center= qh_getcentrum(qh, neighbor);
  zzinc_(Zcentrumtests);
  qh_distplane(qh, neighbor->center, facet, &dist2);
  if (dist2 > qh->centrum_radius)
    isconcave= True;
  else if (!iscoplanar && dist2 >= -qh->centrum_radius)
    iscoplanar= True;
  if (!isconcave && (!iscoplanar || (qh->MERGEexact && !qh->POSTmerging)))
    return False;
  if (!okangle && qh->ANGLEmerge) {
    angle= qh_getangle(qh, facet->normal, neighbor->normal);
    zinc_(Zangletests);
  }
  if (isconcave && iscoplanar) {
    zinc_(Zconcavecoplanarridge);
    if (dist > dist2)
      qh_appendmergeset(qh, facet, neighbor, MRGconcavecoplanar, dist, angle);
    else
      qh_appendmergeset(qh, neighbor, facet, MRGconcavecoplanar, dist2, angle);
    trace0((qh, qh->ferr, 36, "qh_test_centrum_merge: concave f%d to coplanar f%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
           facet->id, neighbor->id, dist, dist2, angle, qh->furthest_id));
  }else if (isconcave) {
    mergedist= fmax_(dist, dist2);
    zinc_(Zconcaveridge);
    qh_appendmergeset(qh, facet, neighbor, MRGconcave, mergedist, angle);
    trace0((qh, qh->ferr, 37, "qh_test_centrum_merge: concave f%d to f%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, neighbor->id, dist, dist2, angle, qh->furthest_id));
  }else /* iscoplanar */ {
    mergedist= fmin_(fabs_(dist), fabs_(dist2));
    zinc_(Zcoplanarcentrum);
    qh_appendmergeset(qh, facet, neighbor, MRGcoplanar, mergedist, angle);
    trace2((qh, qh->ferr, 2097, "qh_test_centrum_merge: coplanar f%d to f%d dist %4.4g, reverse dist %4.4g angle %4.4g\n",
              facet->id, neighbor->id, dist, dist2, angle));
  }
  return True;
} /* test_centrum_merge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_degen_neighbors">-</a>

  qh_test_degen_neighbors(qh, facet )
    append degenerate neighbors to qh.degen_mergeset

  notes:
  called at end of qh_mergefacet() and qh_renamevertex()
  call after test_redundant_facet() since MRGredundant is less expensive then MRGdegen
    a degenerate facet has fewer than hull_dim neighbors
    see: qh_merge_degenredundant()

*/
void qh_test_degen_neighbors(qhT *qh, facetT *facet) {
  facetT *neighbor, **neighborp;
  int size;

  trace4((qh, qh->ferr, 4073, "qh_test_degen_neighbors: test for degenerate neighbors of f%d\n", facet->id));
  FOREACHneighbor_(facet) {
    if (neighbor->visible) {
      qh_fprintf(qh, qh->ferr, 6359, "qhull internal error (qh_test_degen_neighbors): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
        facet->id, neighbor->id);
      qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
    }
    if (neighbor->degenerate || neighbor->redundant || neighbor->dupridge) /* will merge or delete */
      continue;
    /* merge flipped-degenerate facet before flipped facets */
    if ((size= qh_setsize(qh, neighbor->neighbors)) < qh->hull_dim) {
      qh_appendmergeset(qh, neighbor, neighbor, MRGdegen, 0.0, 1.0);
      trace2((qh, qh->ferr, 2019, "qh_test_degen_neighbors: f%d is degenerate with %d neighbors.  Neighbor of f%d.\n", neighbor->id, size, facet->id));
    }
  }
} /* test_degen_neighbors */


/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_nonsimplicial_merge">-</a>

  qh_test_nonsimplicial_merge(qh, facet, neighbor, angle, okangle )
    test centrum and vertex convexity and append non-convex or redundant facets to qh.facet_mergeset
    'angle' is angle between facets if okangle is true, otherwise use 0.0
    skips coplanar merges if pre-merging with qh.MERGEexact ('Qx')

  returns:
    true if appends facet/neighbor to qh.facet_mergeset
    sets facet->center as needed
    does not change facet->seen

  notes:
    only called from test_appendmerge if a non-simplicial facet and at least 4-d
    at least as strict as qh_checkconvex, including qh.DISTround ('En' and 'Rn')
      centrums must be < -qh.centrum_radius
    tests vertices as well as centrums since a facet may be twisted relative to its neighbor

  design:
    set precision constants for maxoutside, clearlyconcave, minvertex, and coplanarcentrum
      use maxoutside for coplanarcentrum if premerging with 'Qx' and qh_MAXcoplanarcentrum merges
      otherwise use qh.centrum_radious for coplanarcentrum
    make facet and neighbor centrums if needed
    isconcave if a centrum is above neighbor (coplanarcentrum)
    iscoplanar if a centrum is not below neighbor (-qh.centrum_radius)
    maybeconvex if a centrum is clearly below neighbor (-clearyconvex)
    return False if both centrums clearly below neighbor (-clearyconvex)
    return MRGconcave if isconcave

    facets are neither clearly convex nor clearly concave
    test vertices as well as centrums
    if maybeconvex
      determine mindist and maxdist for vertices of the other facet
      maybe MRGredundant
    otherwise
      determine mindist and maxdist for vertices of either facet
      maybe MRGredundant
      maybeconvex if a vertex is clearly below neighbor (-clearconvex)

    vertices are concave if dist > clearlyconcave
    vertices are twisted if dist > maxoutside (isconcave and maybeconvex)
    return False if not concave and pre-merge of 'Qx' (qh.MERGEexact)
    vertices are coplanar if dist in -minvertex..maxoutside
    if !isconcave, vertices are coplanar if dist >= -qh.MAXcoplanar (n*qh.premerge_centrum)

    return False if neither concave nor coplanar
    return MRGtwisted if isconcave and maybeconvex
    return MRGconcavecoplanar if isconcave and isconvex
    return MRGconcave if isconcave
    return MRGcoplanar if iscoplanar
*/
boolT qh_test_nonsimplicial_merge(qhT *qh, facetT *facet, facetT *neighbor, realT angle, boolT okangle) {
  coordT dist, mindist, maxdist, mindist2, maxdist2, dist2, maxoutside, clearlyconcave, minvertex, clearlyconvex, mergedist, coplanarcentrum;
  boolT isconcave= False, iscoplanar= False, maybeconvex= False, isredundant= False;
  vertexT *maxvertex= NULL, *maxvertex2= NULL;

  maxoutside= fmax_(neighbor->maxoutside, qh->ONEmerge + qh->DISTround);
  maxoutside= fmax_(maxoutside, facet->maxoutside);
  clearlyconcave= qh_RATIOconcavehorizon * maxoutside;
  minvertex= fmax_(-qh->min_vertex, qh->MAXcoplanar); /* non-negative, not available per facet, not used for iscoplanar */
  clearlyconvex= qh_RATIOconvexmerge * minvertex; /* must be convex for MRGtwisted */
  if (qh->MERGEexact && !qh->POSTmerging && (facet->nummerge > qh_MAXcoplanarcentrum || neighbor->nummerge > qh_MAXcoplanarcentrum))
    coplanarcentrum= maxoutside;
  else
    coplanarcentrum= qh->centrum_radius;

  if (!facet->center)
    facet->center= qh_getcentrum(qh, facet);
  zzinc_(Zcentrumtests);
  qh_distplane(qh, facet->center, neighbor, &dist);
  if (dist > coplanarcentrum)
    isconcave= True;
  else if (dist >= -qh->centrum_radius)
    iscoplanar= True;
  else if (dist < -clearlyconvex)
    maybeconvex= True;
  if (!neighbor->center)
    neighbor->center= qh_getcentrum(qh, neighbor);
  zzinc_(Zcentrumtests);
  qh_distplane(qh, neighbor->center, facet, &dist2);
  if (dist2 > coplanarcentrum)
    isconcave= True;
  else if (dist2 >= -qh->centrum_radius)
    iscoplanar= True;
  else if (dist2 < -clearlyconvex) {
    if (maybeconvex)
      return False; /* both centrums clearly convex */
    maybeconvex= True;
  }
  if (isconcave) {
    if (!okangle && qh->ANGLEmerge) {
      angle= qh_getangle(qh, facet->normal, neighbor->normal);
      zinc_(Zangletests);
    }
    mergedist= fmax_(dist, dist2);
    zinc_(Zconcaveridge);
    qh_appendmergeset(qh, facet, neighbor, MRGconcave, mergedist, angle);
    trace0((qh, qh->ferr, 18, "qh_test_nonsimplicial_merge: concave centrum for f%d or f%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, neighbor->id, dist, dist2, angle, qh->furthest_id));
    return True;
  }
  /* neither clearly convex nor clearly concave, test vertices as well as centrums */
  if (maybeconvex) {
    if (dist < -clearlyconvex) {
      maxdist= dist;  /* facet centrum clearly convex, no need to test its vertex distance */
      mindist= dist;
      maxvertex2= qh_furthestvertex(qh, neighbor, facet, &maxdist2, &mindist2);
      if (!maxvertex2) {
        qh_appendmergeset(qh, neighbor, facet, MRGredundant, maxdist2, qh_ANGLEnone);
        isredundant= True;
      }
    }else { /* dist2 < -clearlyconvex */
      maxdist2= dist2;   /* neighbor centrum clearly convex, no need to test its vertex distance */
      mindist2= dist2;
      maxvertex= qh_furthestvertex(qh, facet, neighbor, &maxdist, &mindist);
      if (!maxvertex) {
        qh_appendmergeset(qh, facet, neighbor, MRGredundant, maxdist, qh_ANGLEnone);
        isredundant= True;
      }
    }
  }else {
    maxvertex= qh_furthestvertex(qh, facet, neighbor, &maxdist, &mindist);
    if (maxvertex) {
      maxvertex2= qh_furthestvertex(qh, neighbor, facet, &maxdist2, &mindist2);
      if (!maxvertex2) {
        qh_appendmergeset(qh, neighbor, facet, MRGredundant, maxdist2, qh_ANGLEnone);
        isredundant= True;
      }else if (mindist < -clearlyconvex || mindist2 < -clearlyconvex)
        maybeconvex= True;
    }else { /* !maxvertex */
      qh_appendmergeset(qh, facet, neighbor, MRGredundant, maxdist, qh_ANGLEnone);
      isredundant= True;
    }
  }
  if (isredundant) {
    zinc_(Zredundantmerge);
    return True;
  }

  if (maxdist > clearlyconcave || maxdist2 > clearlyconcave)
    isconcave= True;
  else if (maybeconvex) {
    if (maxdist > maxoutside || maxdist2 > maxoutside)
      isconcave= True;  /* MRGtwisted */
  }
  if (!isconcave && qh->MERGEexact && !qh->POSTmerging)
    return False;
  if (isconcave && !iscoplanar) {
    if (maxdist < maxoutside && (-qh->MAXcoplanar || (maxdist2 < maxoutside && mindist2 >= -qh->MAXcoplanar)))
      iscoplanar= True; /* MRGconcavecoplanar */
  }else if (!iscoplanar) {
    if (mindist >= -qh->MAXcoplanar || mindist2 >= -qh->MAXcoplanar)
      iscoplanar= True;  /* MRGcoplanar */
  }
  if (!isconcave && !iscoplanar)
    return False;
  if (!okangle && qh->ANGLEmerge) {
    angle= qh_getangle(qh, facet->normal, neighbor->normal);
    zinc_(Zangletests);
  }
  if (isconcave && maybeconvex) {
    zinc_(Ztwistedridge);
    if (maxdist > maxdist2)
      qh_appendmergeset(qh, facet, neighbor, MRGtwisted, maxdist, angle);
    else
      qh_appendmergeset(qh, neighbor, facet, MRGtwisted, maxdist2, angle);
    trace0((qh, qh->ferr, 27, "qh_test_nonsimplicial_merge: twisted concave f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
           facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }else if (isconcave && iscoplanar) {
    zinc_(Zconcavecoplanarridge);
    if (maxdist > maxdist2)
      qh_appendmergeset(qh, facet, neighbor, MRGconcavecoplanar, maxdist, angle);
    else
      qh_appendmergeset(qh, neighbor, facet, MRGconcavecoplanar, maxdist2, angle);
    trace0((qh, qh->ferr, 28, "qh_test_nonsimplicial_merge: concave coplanar f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }else if (isconcave) {
    mergedist= fmax_(maxdist, maxdist2);
    zinc_(Zconcaveridge);
    qh_appendmergeset(qh, facet, neighbor, MRGconcave, mergedist, angle);
    trace0((qh, qh->ferr, 29, "qh_test_nonsimplicial_merge: concave f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }else /* iscoplanar */ {
    mergedist= fmax_(fmax_(maxdist, maxdist2), fmax_(-mindist, -mindist2));
    zinc_(Zcoplanarcentrum);
    qh_appendmergeset(qh, facet, neighbor, MRGcoplanar, mergedist, angle);
    trace2((qh, qh->ferr, 2099, "qh_test_nonsimplicial_merge: coplanar f%d v%d to f%d v%d, dist %4.4g and reverse dist %4.4g, angle %4.4g during p%d\n",
      facet->id, getid_(maxvertex), neighbor->id, getid_(maxvertex2), maxdist, maxdist2, angle, qh->furthest_id));
  }
  return True;
} /* test_nonsimplicial_merge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_redundant_neighbors">-</a>

  qh_test_redundant_neighbors(qh, facet )
    append degenerate facet or its redundant neighbors to qh.degen_mergeset

  returns:
    bumps vertex_visit

  notes:
    called at end of qh_mergefacet(), qh_mergecycle_all(), and qh_renamevertex
    call before qh_test_degen_neighbors (MRGdegen are more likely to cause problems)
    a redundant neighbor's vertices is a subset of the facet's vertices
    with pinched and flipped facets, a redundant neighbor may have a wildly different normal

    see qh_merge_degenredundant() and qh_-_facet()

  design:
    if facet is degenerate
       appends facet to degen_mergeset
    else
       appends redundant neighbors of facet to degen_mergeset
*/
void qh_test_redundant_neighbors(qhT *qh, facetT *facet) {
  vertexT *vertex, **vertexp;
  facetT *neighbor, **neighborp;
  int size;

  trace4((qh, qh->ferr, 4022, "qh_test_redundant_neighbors: test neighbors of f%d vertex_visit %d\n",
          facet->id, qh->vertex_visit+1));
  if ((size= qh_setsize(qh, facet->neighbors)) < qh->hull_dim) {
    qh_appendmergeset(qh, facet, facet, MRGdegen, 0.0, 1.0);
    trace2((qh, qh->ferr, 2017, "qh_test_redundant_neighbors: f%d is degenerate with %d neighbors.\n", facet->id, size));
  }else {
    qh->vertex_visit++;
    FOREACHvertex_(facet->vertices)
      vertex->visitid= qh->vertex_visit;
    FOREACHneighbor_(facet) {
      if (neighbor->visible) {
        qh_fprintf(qh, qh->ferr, 6360, "qhull internal error (qh_test_redundant_neighbors): facet f%d has deleted neighbor f%d (qh.visible_list)\n",
          facet->id, neighbor->id);
        qh_errexit2(qh, qh_ERRqhull, facet, neighbor);
      }
      if (neighbor->degenerate || neighbor->redundant || neighbor->dupridge) /* will merge or delete */
        continue;
      if (facet->flipped && !neighbor->flipped) /* do not merge non-flipped into flipped */
        continue;
      /* merge redundant-flipped facet first */
      /* uses early out instead of checking vertex count */
      FOREACHvertex_(neighbor->vertices) {
        if (vertex->visitid != qh->vertex_visit)
          break;
      }
      if (!vertex) {
        qh_appendmergeset(qh, neighbor, facet, MRGredundant, 0.0, 1.0);
        trace2((qh, qh->ferr, 2018, "qh_test_redundant_neighbors: f%d is contained in f%d.  merge\n", neighbor->id, facet->id));
      }
    }
  }
} /* test_redundant_neighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="test_vneighbors">-</a>

  qh_test_vneighbors(qh)
    test vertex neighbors for convexity
    tests all facets on qh.newfacet_list

  returns:
    true if non-convex vneighbors appended to qh.facet_mergeset
    initializes vertex neighbors if needed

  notes:
    called by qh_all_merges from qh_postmerge if qh.TESTvneighbors ('Qv')
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
boolT qh_test_vneighbors(qhT *qh /* qh.newfacet_list */) {
  facetT *newfacet, *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  int nummerges= 0;

  trace1((qh, qh->ferr, 1015, "qh_test_vneighbors: testing vertex neighbors for convexity\n"));
  if (!qh->VERTEXneighbors)
    qh_vertexneighbors(qh);
  FORALLnew_facets
    newfacet->seen= False;
  FORALLnew_facets {
    newfacet->seen= True;
    newfacet->visitid= qh->visit_id++;
    FOREACHneighbor_(newfacet)
      newfacet->visitid= qh->visit_id;
    FOREACHvertex_(newfacet->vertices) {
      FOREACHneighbor_(vertex) {
        if (neighbor->seen || neighbor->visitid == qh->visit_id)
          continue;
        if (qh_test_appendmerge(qh, newfacet, neighbor, False)) /* ignores optimization for simplicial ridges */
          nummerges++;
      }
    }
  }
  zadd_(Ztestvneighbor, nummerges);
  trace1((qh, qh->ferr, 1016, "qh_test_vneighbors: found %d non-convex, vertex neighbors\n",
           nummerges));
  return (nummerges > 0);
} /* test_vneighbors */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="tracemerge">-</a>

  qh_tracemerge(qh, facet1, facet2 )
    print trace message after merge
*/
void qh_tracemerge(qhT *qh, facetT *facet1, facetT *facet2, mergeType mergetype) {
  boolT waserror= False;
  const char *mergename;

#ifndef qh_NOtrace
  if(mergetype > 0 && mergetype <= sizeof(mergetypes))
    mergename= mergetypes[mergetype];
  else
    mergename= mergetypes[MRGnone];
  if (qh->IStracing >= 4)
    qh_errprint(qh, "MERGED", facet2, NULL, NULL, NULL);
  if (facet2 == qh->tracefacet || (qh->tracevertex && qh->tracevertex->newfacet)) {
    qh_fprintf(qh, qh->ferr, 8085, "qh_tracemerge: trace facet and vertex after merge of f%d into f%d type %d (%s), furthest p%d\n",
      facet1->id, facet2->id, mergetype, mergename, qh->furthest_id);
    if (facet2 != qh->tracefacet)
      qh_errprint(qh, "TRACE", qh->tracefacet,
        (qh->tracevertex && qh->tracevertex->neighbors) ?
           SETfirstt_(qh->tracevertex->neighbors, facetT) : NULL,
        NULL, qh->tracevertex);
  }
  if (qh->tracevertex) {
    if (qh->tracevertex->deleted)
      qh_fprintf(qh, qh->ferr, 8086, "qh_tracemerge: trace vertex deleted at furthest p%d\n",
            qh->furthest_id);
    else
      qh_checkvertex(qh, qh->tracevertex, qh_ALL, &waserror);
  }
  if (qh->tracefacet && qh->tracefacet->normal && !qh->tracefacet->visible)
    qh_checkfacet(qh, qh->tracefacet, True /* newmerge */, &waserror);
#endif /* !qh_NOtrace */
  if (qh->CHECKfrequently || qh->IStracing >= 4) { /* can't check polygon here */
    if (qh->IStracing >= 4 && qh->num_facets < 500) {
      qh_printlists(qh);
    }
    qh_checkfacet(qh, facet2, True /* newmerge */, &waserror);
  }
  if (waserror)
    qh_errexit(qh, qh_ERRqhull, NULL, NULL); /* erroneous facet logged by qh_checkfacet */
} /* tracemerge */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="tracemerging">-</a>

  qh_tracemerging(qh)
    print trace message during POSTmerging

  returns:
    updates qh.mergereport

  notes:
    called from qh_mergecycle() and qh_mergefacet()

  see:
    qh_buildtracing()
*/
void qh_tracemerging(qhT *qh) {
  realT cpu;
  int total;
  time_t timedata;
  struct tm *tp;

  qh->mergereport= zzval_(Ztotmerge);
  time(&timedata);
  tp= localtime(&timedata);
  cpu= qh_CPUclock;
  cpu /= qh_SECticks;
  total= zzval_(Ztotmerge) - zzval_(Zcyclehorizon) + zzval_(Zcyclefacettot);
  qh_fprintf(qh, qh->ferr, 8087, "\n\
At %d:%d:%d & %2.5g CPU secs, qhull has merged %d facets with max_outside %2.2g, min_vertex %2.2g.\n\
  The hull contains %d facets and %d vertices.\n",
      tp->tm_hour, tp->tm_min, tp->tm_sec, cpu, total, qh->max_outside, qh->min_vertex,
      qh->num_facets - qh->num_visible,
      qh->num_vertices-qh_setsize(qh, qh->del_vertices));
} /* tracemerging */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="updatetested">-</a>

  qh_updatetested(qh, facet1, facet2 )
    clear facet2->tested and facet1->ridge->tested for merge

  returns:
    deletes facet2->center unless it's already large
      if so, clears facet2->ridge->tested

  notes:
    only called by qh_mergefacet

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
void qh_updatetested(qhT *qh, facetT *facet1, facetT *facet2) {
  ridgeT *ridge, **ridgep;
  int size;

  facet2->tested= False;
  FOREACHridge_(facet1->ridges)
    ridge->tested= False;
  if (!facet2->center)
    return;
  size= qh_setsize(qh, facet2->vertices);
  if (!facet2->keepcentrum) {
    if (size > qh->hull_dim + qh_MAXnewcentrum) {
      facet2->keepcentrum= True;
      zinc_(Zwidevertices);
    }
  }else if (size <= qh->hull_dim + qh_MAXnewcentrum) {
    /* center and keepcentrum was set */
    if (size == qh->hull_dim || qh->POSTmerging)
      facet2->keepcentrum= False; /* if many merges need to recompute centrum */
  }
  if (!facet2->keepcentrum) {
    qh_memfree(qh, facet2->center, qh->normal_size);
    facet2->center= NULL;
    FOREACHridge_(facet2->ridges)
      ridge->tested= False;
  }
} /* updatetested */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="vertexridges">-</a>

  qh_vertexridges(qh, vertex, allneighbors )
    return temporary set of ridges adjacent to a vertex
    vertex->neighbors defined (qh_vertexneighbors)

  notes:
    uses qh.visit_id
    does not include implicit ridges for simplicial facets
    skips last neighbor, unless allneighbors.  For new facets, the last neighbor shares ridges with adjacent neighbors
    if the last neighbor is not simplicial, it will have ridges for its simplicial neighbors
    Use allneighbors when a new cone is attached to an existing convex hull
    similar to qh_neighbor_vertices

  design:
    for each neighbor of vertex
      add ridges that include the vertex to ridges
*/
setT *qh_vertexridges(qhT *qh, vertexT *vertex, boolT allneighbors) {
  facetT *neighbor, **neighborp;
  setT *ridges= qh_settemp(qh, qh->TEMPsize);
  int size;

  qh->visit_id += 2;  /* visit_id for vertex neighbors, visit_id-1 for facets of visited ridges */
  FOREACHneighbor_(vertex)
    neighbor->visitid= qh->visit_id;
  FOREACHneighbor_(vertex) {
    if (*neighborp || allneighbors)   /* no new ridges in last neighbor */
      qh_vertexridges_facet(qh, vertex, neighbor, &ridges);
  }
  if (qh->PRINTstatistics || qh->IStracing) {
    size= qh_setsize(qh, ridges);
    zinc_(Zvertexridge);
    zadd_(Zvertexridgetot, size);
    zmax_(Zvertexridgemax, size);
    trace3((qh, qh->ferr, 3011, "qh_vertexridges: found %d ridges for v%d\n",
             size, vertex->id));
  }
  return ridges;
} /* vertexridges */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="vertexridges_facet">-</a>

  qh_vertexridges_facet(qh, vertex, facet, ridges )
    add adjacent ridges for vertex in facet
    neighbor->visitid==qh.visit_id if it hasn't been visited

  returns:
    ridges updated
    sets facet->visitid to qh.visit_id-1

  design:
    for each ridge of facet
      if ridge of visited neighbor (i.e., unprocessed)
        if vertex in ridge
          append ridge
    mark facet processed
*/
void qh_vertexridges_facet(qhT *qh, vertexT *vertex, facetT *facet, setT **ridges) {
  ridgeT *ridge, **ridgep;
  facetT *neighbor;
  int last_i= qh->hull_dim-2;
  vertexT *second, *last;

  FOREACHridge_(facet->ridges) {
    neighbor= otherfacet_(ridge, facet);
    if (neighbor->visitid == qh->visit_id) {
      if (SETfirst_(ridge->vertices) == vertex) {
        qh_setappend(qh, ridges, ridge);
      }else if (last_i > 2) {
        second= SETsecondt_(ridge->vertices, vertexT);
        last= SETelemt_(ridge->vertices, last_i, vertexT);
        if (second->id >= vertex->id && last->id <= vertex->id) { /* vertices inverse sorted by id */
          if (second == vertex || last == vertex)
            qh_setappend(qh, ridges, ridge);
          else if (qh_setin(ridge->vertices, vertex))
            qh_setappend(qh, ridges, ridge);
        }
      }else if (SETelem_(ridge->vertices, last_i) == vertex
          || (last_i > 1 && SETsecond_(ridge->vertices) == vertex)) {
        qh_setappend(qh, ridges, ridge);
      }
    }
  }
  facet->visitid= qh->visit_id-1;
} /* vertexridges_facet */

/*-<a                             href="qh-merge_r.htm#TOC"
  >-------------------------------</a><a name="willdelete">-</a>

  qh_willdelete(qh, facet, replace )
    moves facet to visible list for qh_deletevisible
    sets facet->f.replace to replace (may be NULL)
    clears f.ridges and f.neighbors -- no longer valid

  returns:
    bumps qh.num_visible
*/
void qh_willdelete(qhT *qh, facetT *facet, facetT *replace) {

  trace4((qh, qh->ferr, 4081, "qh_willdelete: move f%d to visible list, set its replacement as f%d, and clear f.neighbors and f.ridges\n", facet->id, getid_(replace)));
  if (!qh->visible_list && qh->newfacet_list) {
    qh_fprintf(qh, qh->ferr, 6378, "qhull internal error (qh_willdelete): expecting qh.visible_list at before qh.newfacet_list f%d.   Got NULL\n",
        qh->newfacet_list->id);
    qh_errexit2(qh, qh_ERRqhull, NULL, NULL);
  }
  qh_removefacet(qh, facet);
  qh_prependfacet(qh, facet, &qh->visible_list);
  qh->num_visible++;
  facet->visible= True;
  facet->f.replace= replace;
  if (facet->ridges)
    SETfirst_(facet->ridges)= NULL;
  if (facet->neighbors)
    SETfirst_(facet->neighbors)= NULL;
} /* willdelete */

#else /* qh_NOmerge */

void qh_all_vertexmerges(qhT *qh, int apexpointid, facetT *facet, facetT **retryfacet) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(apexpointid)
  QHULL_UNUSED(facet)
  QHULL_UNUSED(retryfacet)
}
void qh_premerge(qhT *qh, int apexpointid, realT maxcentrum, realT maxangle) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(apexpointid)
  QHULL_UNUSED(maxcentrum)
  QHULL_UNUSED(maxangle)
}
void qh_postmerge(qhT *qh, const char *reason, realT maxcentrum, realT maxangle,
                      boolT vneighbors) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(reason)
  QHULL_UNUSED(maxcentrum)
  QHULL_UNUSED(maxangle)
  QHULL_UNUSED(vneighbors)
}
void qh_checkdelfacet(qhT *qh, facetT *facet, setT *mergeset) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(facet)
  QHULL_UNUSED(mergeset)
}
void qh_checkdelridge(qhT *qh /* qh.visible_facets, vertex_mergeset */) {
  QHULL_UNUSED(qh)
}
boolT qh_checkzero(qhT *qh, boolT testall) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(testall)
    
  return True;
}
void qh_freemergesets(qhT *qh) {
  QHULL_UNUSED(qh)
}
void qh_initmergesets(qhT *qh) {
  QHULL_UNUSED(qh)
}
void qh_merge_pinchedvertices(qhT *qh, int apexpointid /* qh.newfacet_list */) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(apexpointid)
}
#endif /* qh_NOmerge */


/*<html><pre>  -<a                             href="qh-poly_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   poly_r.c
   implements polygons and simplices

   see qh-poly_r.htm, poly_r.h and libqhull_r.h

   infrequent code is in poly2_r.c
   (all but top 50 and their callers 12/3/95)

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/poly_r.c#7 $$Change: 2705 $
   $DateTime: 2019/06/26 16:34:45 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*======== functions in alphabetical order ==========*/

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="appendfacet">-</a>

  qh_appendfacet(qh, facet )
    appends facet to end of qh.facet_list,

  returns:
    updates qh.newfacet_list, facet_next, facet_list
    increments qh.numfacets

  notes:
    assumes qh.facet_list/facet_tail is defined (createsimplex)

  see:
    qh_removefacet()

*/
void qh_appendfacet(qhT *qh, facetT *facet) {
  facetT *tail= qh->facet_tail;

  if (tail == qh->newfacet_list) {
    qh->newfacet_list= facet;
    if (tail == qh->visible_list) /* visible_list is at or before newfacet_list */
      qh->visible_list= facet;
  }
  if (tail == qh->facet_next)
    qh->facet_next= facet;
  facet->previous= tail->previous;
  facet->next= tail;
  if (tail->previous)
    tail->previous->next= facet;
  else
    qh->facet_list= facet;
  tail->previous= facet;
  qh->num_facets++;
  trace4((qh, qh->ferr, 4044, "qh_appendfacet: append f%d to facet_list\n", facet->id));
} /* appendfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="appendvertex">-</a>

  qh_appendvertex(qh, vertex )
    appends vertex to end of qh.vertex_list,

  returns:
    sets vertex->newfacet
    updates qh.vertex_list, newvertex_list
    increments qh.num_vertices

  notes:
    assumes qh.vertex_list/vertex_tail is defined (createsimplex)

*/
void qh_appendvertex(qhT *qh, vertexT *vertex) {
  vertexT *tail= qh->vertex_tail;

  if (tail == qh->newvertex_list)
    qh->newvertex_list= vertex;
  vertex->newfacet= True;
  vertex->previous= tail->previous;
  vertex->next= tail;
  if (tail->previous)
    tail->previous->next= vertex;
  else
    qh->vertex_list= vertex;
  tail->previous= vertex;
  qh->num_vertices++;
  trace4((qh, qh->ferr, 4045, "qh_appendvertex: append v%d to qh.newvertex_list and set v.newfacet\n", vertex->id));
} /* appendvertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="attachnewfacets">-</a>

  qh_attachnewfacets(qh)
    attach horizon facets to new facets in qh.newfacet_list
    newfacets have neighbor and ridge links to horizon but not vice versa

  returns:
    clears qh.NEWtentative
    set qh.NEWfacets
    horizon facets linked to new facets
      ridges changed from visible facets to new facets
      simplicial ridges deleted
    qh.visible_list, no ridges valid
    facet->f.replace is a newfacet (if any)

  notes:
    used for qh.NEWtentative, otherwise see qh_makenew_nonsimplicial and qh_makenew_simplicial
    qh_delridge_merge not needed (as tested by qh_checkdelridge)

  design:
    delete interior ridges and neighbor sets by
      for each visible, non-simplicial facet
        for each ridge
          if last visit or if neighbor is simplicial
            if horizon neighbor
              delete ridge for horizon's ridge set
            delete ridge
        erase neighbor set
    attach horizon facets and new facets by
      for all new facets
        if corresponding horizon facet is simplicial
          locate corresponding visible facet {may be more than one}
          link visible facet to new facet
          replace visible facet with new facet in horizon
        else it is non-simplicial
          for all visible neighbors of the horizon facet
            link visible neighbor to new facet
            delete visible neighbor from horizon facet
          append new facet to horizon's neighbors
          the first ridge of the new facet is the horizon ridge
          link the new facet into the horizon ridge
*/
void qh_attachnewfacets(qhT *qh /* qh.visible_list, qh.newfacet_list */) {
  facetT *newfacet= NULL, *neighbor, **neighborp, *horizon, *visible;
  ridgeT *ridge, **ridgep;

  trace3((qh, qh->ferr, 3012, "qh_attachnewfacets: delete interior ridges\n"));
  if (qh->CHECKfrequently) {
    qh_checkdelridge(qh);
  }
  qh->visit_id++;
  FORALLvisible_facets {
    visible->visitid= qh->visit_id;
    if (visible->ridges) {
      FOREACHridge_(visible->ridges) {
        neighbor= otherfacet_(ridge, visible);
        if (neighbor->visitid == qh->visit_id
            || (!neighbor->visible && neighbor->simplicial)) {
          if (!neighbor->visible)  /* delete ridge for simplicial horizon */
            qh_setdel(neighbor->ridges, ridge);
          qh_delridge(qh, ridge); /* delete on second visit */
        }
      }
    }
  }
  trace1((qh, qh->ferr, 1017, "qh_attachnewfacets: attach horizon facets to new facets\n"));
  FORALLnew_facets {
    horizon= SETfirstt_(newfacet->neighbors, facetT);
    if (horizon->simplicial) {
      visible= NULL;
      FOREACHneighbor_(horizon) {   /* may have more than one horizon ridge */
        if (neighbor->visible) {
          if (visible) {
            if (qh_setequal_skip(newfacet->vertices, 0, horizon->vertices,
                                  SETindex_(horizon->neighbors, neighbor))) {
              visible= neighbor;
              break;
            }
          }else
            visible= neighbor;
        }
      }
      if (visible) {
        visible->f.replace= newfacet;
        qh_setreplace(qh, horizon->neighbors, visible, newfacet);
      }else {
        qh_fprintf(qh, qh->ferr, 6102, "qhull internal error (qh_attachnewfacets): could not find visible facet for horizon f%d of newfacet f%d\n",
                 horizon->id, newfacet->id);
        qh_errexit2(qh, qh_ERRqhull, horizon, newfacet);
      }
    }else { /* non-simplicial, with a ridge for newfacet */
      FOREACHneighbor_(horizon) {    /* may hold for many new facets */
        if (neighbor->visible) {
          neighbor->f.replace= newfacet;
          qh_setdelnth(qh, horizon->neighbors, SETindex_(horizon->neighbors, neighbor));
          neighborp--; /* repeat */
        }
      }
      qh_setappend(qh, &horizon->neighbors, newfacet);
      ridge= SETfirstt_(newfacet->ridges, ridgeT);
      if (ridge->top == horizon) {
        ridge->bottom= newfacet;
        ridge->simplicialbot= True;
      }else {
        ridge->top= newfacet;
        ridge->simplicialtop= True;
      }
    }
  } /* newfacets */
  trace4((qh, qh->ferr, 4094, "qh_attachnewfacets: clear f.ridges and f.neighbors for visible facets, may become invalid before qh_deletevisible\n"));
  FORALLvisible_facets {
    if (visible->ridges)
      SETfirst_(visible->ridges)= NULL; 
    SETfirst_(visible->neighbors)= NULL;
  }
  qh->NEWtentative= False;
  qh->NEWfacets= True;
  if (qh->PRINTstatistics) {
    FORALLvisible_facets {
      if (!visible->f.replace)
        zinc_(Zinsidevisible);
    }
  }
} /* attachnewfacets */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="checkflipped">-</a>

  qh_checkflipped(qh, facet, dist, allerror )
    checks facet orientation to interior point

    if allerror set,
      tests against -qh.DISTround
    else
      tests against 0.0 since tested against -qh.DISTround before

  returns:
    False if it flipped orientation (sets facet->flipped)
    distance if non-NULL

  notes:
    called by qh_setfacetplane, qh_initialhull, and qh_checkflipped_all
*/
boolT qh_checkflipped(qhT *qh, facetT *facet, realT *distp, boolT allerror) {
  realT dist;

  if (facet->flipped && !distp)
    return False;
  zzinc_(Zdistcheck);
  qh_distplane(qh, qh->interior_point, facet, &dist);
  if (distp)
    *distp= dist;
  if ((allerror && dist >= -qh->DISTround) || (!allerror && dist > 0.0)) {
    facet->flipped= True;
    trace0((qh, qh->ferr, 19, "qh_checkflipped: facet f%d flipped, allerror? %d, distance= %6.12g during p%d\n",
              facet->id, allerror, dist, qh->furthest_id));
    if (qh->num_facets > qh->hull_dim+1) { /* qh_initialhull reverses orientation if !qh_checkflipped */
      zzinc_(Zflippedfacets);
      qh_joggle_restart(qh, "flipped facet");
    }
    return False;
  }
  return True;
} /* checkflipped */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="delfacet">-</a>

  qh_delfacet(qh, facet )
    removes facet from facet_list and frees up its memory

  notes:
    assumes vertices and ridges already freed or referenced elsewhere
*/
void qh_delfacet(qhT *qh, facetT *facet) {
  void **freelistp; /* used if !qh_NOmem by qh_memfree_() */

  trace3((qh, qh->ferr, 3057, "qh_delfacet: delete f%d\n", facet->id));
  if (qh->CHECKfrequently || qh->VERIFYoutput) { 
    if (!qh->NOerrexit) {
      qh_checkdelfacet(qh, facet, qh->facet_mergeset);
      qh_checkdelfacet(qh, facet, qh->degen_mergeset);
      qh_checkdelfacet(qh, facet, qh->vertex_mergeset);
    }
  }
  if (facet == qh->tracefacet)
    qh->tracefacet= NULL;
  if (facet == qh->GOODclosest)
    qh->GOODclosest= NULL;
  qh_removefacet(qh, facet);
  if (!facet->tricoplanar || facet->keepcentrum) {
    qh_memfree_(qh, facet->normal, qh->normal_size, freelistp);
    if (qh->CENTERtype == qh_ASvoronoi) {   /* braces for macro calls */
      qh_memfree_(qh, facet->center, qh->center_size, freelistp);
    }else /* AScentrum */ {
      qh_memfree_(qh, facet->center, qh->normal_size, freelistp);
    }
  }
  qh_setfree(qh, &(facet->neighbors));
  if (facet->ridges)
    qh_setfree(qh, &(facet->ridges));
  qh_setfree(qh, &(facet->vertices));
  if (facet->outsideset)
    qh_setfree(qh, &(facet->outsideset));
  if (facet->coplanarset)
    qh_setfree(qh, &(facet->coplanarset));
  qh_memfree_(qh, facet, (int)sizeof(facetT), freelistp);
} /* delfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="deletevisible">-</a>

  qh_deletevisible()
    delete visible facets and vertices

  returns:
    deletes each facet and removes from facetlist
    deletes vertices on qh.del_vertices and ridges in qh.del_ridges
    at exit, qh.visible_list empty (== qh.newfacet_list)

  notes:
    called by qh_all_vertexmerges, qh_addpoint, and qh_qhull
    ridges already deleted or moved elsewhere
    deleted vertices on qh.del_vertices
    horizon facets do not reference facets on qh.visible_list
    new facets in qh.newfacet_list
    uses   qh.visit_id;
*/
void qh_deletevisible(qhT *qh /* qh.visible_list */) {
  facetT *visible, *nextfacet;
  vertexT *vertex, **vertexp;
  int numvisible= 0, numdel= qh_setsize(qh, qh->del_vertices);

  trace1((qh, qh->ferr, 1018, "qh_deletevisible: delete %d visible facets and %d vertices\n",
         qh->num_visible, numdel));
  for (visible=qh->visible_list; visible && visible->visible;
                visible= nextfacet) { /* deleting current */
    nextfacet= visible->next;
    numvisible++;
    qh_delfacet(qh, visible);  /* f.ridges deleted or moved elsewhere, deleted f.vertices on qh.del_vertices */
  }
  if (numvisible != qh->num_visible) {
    qh_fprintf(qh, qh->ferr, 6103, "qhull internal error (qh_deletevisible): qh->num_visible %d is not number of visible facets %d\n",
             qh->num_visible, numvisible);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  qh->num_visible= 0;
  zadd_(Zvisfacettot, numvisible);
  zmax_(Zvisfacetmax, numvisible);
  zzadd_(Zdelvertextot, numdel);
  zmax_(Zdelvertexmax, numdel);
  FOREACHvertex_(qh->del_vertices)
    qh_delvertex(qh, vertex);
  qh_settruncate(qh, qh->del_vertices, 0);
} /* deletevisible */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="facetintersect">-</a>

  qh_facetintersect(qh, facetA, facetB, skipa, skipB, prepend )
    return vertices for intersection of two simplicial facets
    may include 1 prepended entry (if more, need to settemppush)

  returns:
    returns set of qh.hull_dim-1 + prepend vertices
    returns skipped index for each test and checks for exactly one

  notes:
    does not need settemp since set in quick memory

  see also:
    qh_vertexintersect and qh_vertexintersect_new
    use qh_setnew_delnthsorted to get nth ridge (no skip information)

  design:
    locate skipped vertex by scanning facet A's neighbors
    locate skipped vertex by scanning facet B's neighbors
    intersect the vertex sets
*/
setT *qh_facetintersect(qhT *qh, facetT *facetA, facetT *facetB,
                         int *skipA,int *skipB, int prepend) {
  setT *intersect;
  int dim= qh->hull_dim, i, j;
  facetT **neighborsA, **neighborsB;

  neighborsA= SETaddr_(facetA->neighbors, facetT);
  neighborsB= SETaddr_(facetB->neighbors, facetT);
  i= j= 0;
  if (facetB == *neighborsA++)
    *skipA= 0;
  else if (facetB == *neighborsA++)
    *skipA= 1;
  else if (facetB == *neighborsA++)
    *skipA= 2;
  else {
    for (i=3; i < dim; i++) {
      if (facetB == *neighborsA++) {
        *skipA= i;
        break;
      }
    }
  }
  if (facetA == *neighborsB++)
    *skipB= 0;
  else if (facetA == *neighborsB++)
    *skipB= 1;
  else if (facetA == *neighborsB++)
    *skipB= 2;
  else {
    for (j=3; j < dim; j++) {
      if (facetA == *neighborsB++) {
        *skipB= j;
        break;
      }
    }
  }
  if (i >= dim || j >= dim) {
    qh_fprintf(qh, qh->ferr, 6104, "qhull internal error (qh_facetintersect): f%d or f%d not in other's neighbors\n",
            facetA->id, facetB->id);
    qh_errexit2(qh, qh_ERRqhull, facetA, facetB);
  }
  intersect= qh_setnew_delnthsorted(qh, facetA->vertices, qh->hull_dim, *skipA, prepend);
  trace4((qh, qh->ferr, 4047, "qh_facetintersect: f%d skip %d matches f%d skip %d\n",
          facetA->id, *skipA, facetB->id, *skipB));
  return(intersect);
} /* facetintersect */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="gethash">-</a>

  qh_gethash(qh, hashsize, set, size, firstindex, skipelem )
    return hashvalue for a set with firstindex and skipelem

  notes:
    returned hash is in [0,hashsize)
    assumes at least firstindex+1 elements
    assumes skipelem is NULL, in set, or part of hash

    hashes memory addresses which may change over different runs of the same data
    using sum for hash does badly in high d
*/
int qh_gethash(qhT *qh, int hashsize, setT *set, int size, int firstindex, void *skipelem) {
  void **elemp= SETelemaddr_(set, firstindex, void);
  ptr_intT hash= 0, elem;
  unsigned int uresult;
  int i;
#ifdef _MSC_VER                   /* Microsoft Visual C++ -- warn about 64-bit issues */
#pragma warning( push)            /* WARN64 -- ptr_intT holds a 64-bit pointer */
#pragma warning( disable : 4311)  /* 'type cast': pointer truncation from 'void*' to 'ptr_intT' */
#endif

  switch (size-firstindex) {
  case 1:
    hash= (ptr_intT)(*elemp) - (ptr_intT) skipelem;
    break;
  case 2:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] - (ptr_intT) skipelem;
    break;
  case 3:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      - (ptr_intT) skipelem;
    break;
  case 4:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      + (ptr_intT)elemp[3] - (ptr_intT) skipelem;
    break;
  case 5:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      + (ptr_intT)elemp[3] + (ptr_intT)elemp[4] - (ptr_intT) skipelem;
    break;
  case 6:
    hash= (ptr_intT)(*elemp) + (ptr_intT)elemp[1] + (ptr_intT)elemp[2]
      + (ptr_intT)elemp[3] + (ptr_intT)elemp[4]+ (ptr_intT)elemp[5]
      - (ptr_intT) skipelem;
    break;
  default:
    hash= 0;
    i= 3;
    do {     /* this is about 10% in 10-d */
      if ((elem= (ptr_intT)*elemp++) != (ptr_intT)skipelem) {
        hash ^= (elem << i) + (elem >> (32-i));
        i += 3;
        if (i >= 32)
          i -= 32;
      }
    }while (*elemp);
    break;
  }
  if (hashsize<0) {
    qh_fprintf(qh, qh->ferr, 6202, "qhull internal error: negative hashsize %d passed to qh_gethash [poly_r.c]\n", hashsize);
    qh_errexit2(qh, qh_ERRqhull, NULL, NULL);
  }
  uresult= (unsigned int)hash;
  uresult %= (unsigned int)hashsize;
  /* result= 0; for debugging */
  return (int)uresult;
#ifdef _MSC_VER
#pragma warning( pop)
#endif
} /* gethash */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="getreplacement">-</a>

  qh_getreplacement(qh, visible )
    get replacement for visible facet

  returns:
    valid facet from visible.replace (may be chained)
*/
facetT *qh_getreplacement(qhT *qh, facetT *visible) {
  unsigned int count= 0;

  facetT *result= visible;
  while (result && result->visible) {
    result= result->f.replace;
    if (count++ > qh->facet_id)
      qh_infiniteloop(qh, visible);
  }
  return result;
}

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewfacet">-</a>

  qh_makenewfacet(qh, vertices, toporient, horizon )
    creates a toporient? facet from vertices

  returns:
    returns newfacet
      adds newfacet to qh.facet_list
      newfacet->vertices= vertices
      if horizon
        newfacet->neighbor= horizon, but not vice versa
    newvertex_list updated with vertices
*/
facetT *qh_makenewfacet(qhT *qh, setT *vertices, boolT toporient, facetT *horizon) {
  facetT *newfacet;
  vertexT *vertex, **vertexp;

  FOREACHvertex_(vertices) {
    if (!vertex->newfacet) {
      qh_removevertex(qh, vertex);
      qh_appendvertex(qh, vertex);
    }
  }
  newfacet= qh_newfacet(qh);
  newfacet->vertices= vertices;
  if (toporient)
    newfacet->toporient= True;
  if (horizon)
    qh_setappend(qh, &(newfacet->neighbors), horizon);
  qh_appendfacet(qh, newfacet);
  return(newfacet);
} /* makenewfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenewplanes">-</a>

  qh_makenewplanes()
    make new hyperplanes for facets on qh.newfacet_list

  returns:
    all facets have hyperplanes or are marked for   merging
    doesn't create hyperplane if horizon is coplanar (will merge)
    updates qh.min_vertex if qh.JOGGLEmax

  notes:
    facet->f.samecycle is defined for facet->mergehorizon facets
*/
void qh_makenewplanes(qhT *qh /* qh.newfacet_list */) {
  facetT *newfacet;

  trace4((qh, qh->ferr, 4074, "qh_makenewplanes: make new hyperplanes for facets on qh.newfacet_list f%d\n",
    qh->newfacet_list->id));
  FORALLnew_facets {
    if (!newfacet->mergehorizon)
      qh_setfacetplane(qh, newfacet); /* updates Wnewvertexmax */
  }
  if (qh->JOGGLEmax < REALmax/2)
    minimize_(qh->min_vertex, -wwval_(Wnewvertexmax));
} /* makenewplanes */

#ifndef qh_NOmerge
/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenew_nonsimplicial">-</a>

  qh_makenew_nonsimplicial(qh, visible, apex, numnew )
    make new facets for ridges of a visible facet

  returns:
    first newfacet, bumps numnew as needed
    attaches new facets if !qh->NEWtentative
    marks ridge neighbors for simplicial visible
    if (qh.NEWtentative)
      ridges on newfacet, horizon, and visible
    else
      ridge and neighbors between newfacet and horizon
      visible facet's ridges are deleted
      visible facet's f.neighbors is empty

  notes:
    called by qh_makenewfacets and qh_triangulatefacet
    qh.visit_id if visible has already been processed
    sets neighbor->seen for building f.samecycle
      assumes all 'seen' flags initially false
    qh_delridge_merge not needed (as tested by qh_checkdelridge in qh_makenewfacets)

  design:
    for each ridge of visible facet
      get neighbor of visible facet
      if neighbor was already processed
        delete the ridge (will delete all visible facets later)
      if neighbor is a horizon facet
        create a new facet
        if neighbor coplanar
          adds newfacet to f.samecycle for later merging
        else
          updates neighbor's neighbor set
          (checks for non-simplicial facet with multiple ridges to visible facet)
        updates neighbor's ridge set
        (checks for simplicial neighbor to non-simplicial visible facet)
        (deletes ridge if neighbor is simplicial)

*/
facetT *qh_makenew_nonsimplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew) {
  void **freelistp; /* used if !qh_NOmem by qh_memfree_() */
  ridgeT *ridge, **ridgep;
  facetT *neighbor, *newfacet= NULL, *samecycle;
  setT *vertices;
  boolT toporient;
  unsigned int ridgeid;

  FOREACHridge_(visible->ridges) {
    ridgeid= ridge->id;
    neighbor= otherfacet_(ridge, visible);
    if (neighbor->visible) {
      if (!qh->NEWtentative) {
        if (neighbor->visitid == qh->visit_id) {
          if (qh->traceridge == ridge)
            qh->traceridge= NULL;
          qh_setfree(qh, &(ridge->vertices));  /* delete on 2nd visit */
          qh_memfree_(qh, ridge, (int)sizeof(ridgeT), freelistp);
        }
      }
    }else {  /* neighbor is an horizon facet */
      toporient= (ridge->top == visible);
      vertices= qh_setnew(qh, qh->hull_dim); /* makes sure this is quick */
      qh_setappend(qh, &vertices, apex);
      qh_setappend_set(qh, &vertices, ridge->vertices);
      newfacet= qh_makenewfacet(qh, vertices, toporient, neighbor);
      (*numnew)++;
      if (neighbor->coplanarhorizon) {
        newfacet->mergehorizon= True;
        if (!neighbor->seen) {
          newfacet->f.samecycle= newfacet;
          neighbor->f.newcycle= newfacet;
        }else {
          samecycle= neighbor->f.newcycle;
          newfacet->f.samecycle= samecycle->f.samecycle;
          samecycle->f.samecycle= newfacet;
        }
      }
      if (qh->NEWtentative) {
        if (!neighbor->simplicial)
          qh_setappend(qh, &(newfacet->ridges), ridge);
      }else {  /* qh_attachnewfacets */
        if (neighbor->seen) {
          if (neighbor->simplicial) {
            qh_fprintf(qh, qh->ferr, 6105, "qhull internal error (qh_makenew_nonsimplicial): simplicial f%d sharing two ridges with f%d\n",
                   neighbor->id, visible->id);
            qh_errexit2(qh, qh_ERRqhull, neighbor, visible);
          }
          qh_setappend(qh, &(neighbor->neighbors), newfacet);
        }else
          qh_setreplace(qh, neighbor->neighbors, visible, newfacet);
        if (neighbor->simplicial) {
          qh_setdel(neighbor->ridges, ridge);
          qh_delridge(qh, ridge);
        }else {
          qh_setappend(qh, &(newfacet->ridges), ridge);
          if (toporient) {
            ridge->top= newfacet;
            ridge->simplicialtop= True;
          }else {
            ridge->bottom= newfacet;
            ridge->simplicialbot= True;
          }
        }
      }
      trace4((qh, qh->ferr, 4048, "qh_makenew_nonsimplicial: created facet f%d from v%d and r%d of horizon f%d\n",
          newfacet->id, apex->id, ridgeid, neighbor->id));
    }
    neighbor->seen= True;
  } /* for each ridge */
  return newfacet;
} /* makenew_nonsimplicial */

#else /* qh_NOmerge */
facetT *qh_makenew_nonsimplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew) {
  QHULL_UNUSED(qh)
  QHULL_UNUSED(visible)
  QHULL_UNUSED(apex)
  QHULL_UNUSED(numnew)

  return NULL;
}
#endif /* qh_NOmerge */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="makenew_simplicial">-</a>

  qh_makenew_simplicial(qh, visible, apex, numnew )
    make new facets for simplicial visible facet and apex

  returns:
    attaches new facets if !qh.NEWtentative
      neighbors between newfacet and horizon

  notes:
    nop if neighbor->seen or neighbor->visible(see qh_makenew_nonsimplicial)

  design:
    locate neighboring horizon facet for visible facet
    determine vertices and orientation
    create new facet
    if coplanar,
      add new facet to f.samecycle
    update horizon facet's neighbor list
*/
facetT *qh_makenew_simplicial(qhT *qh, facetT *visible, vertexT *apex, int *numnew) {
  facetT *neighbor, **neighborp, *newfacet= NULL;
  setT *vertices;
  boolT flip, toporient;
  int horizonskip= 0, visibleskip= 0;

  FOREACHneighbor_(visible) {
    if (!neighbor->seen && !neighbor->visible) {
      vertices= qh_facetintersect(qh, neighbor,visible, &horizonskip, &visibleskip, 1);
      SETfirst_(vertices)= apex;
      flip= ((horizonskip & 0x1) ^ (visibleskip & 0x1));
      if (neighbor->toporient)
        toporient= horizonskip & 0x1;
      else
        toporient= (horizonskip & 0x1) ^ 0x1;
      newfacet= qh_makenewfacet(qh, vertices, toporient, neighbor);
      (*numnew)++;
      if (neighbor->coplanarhorizon && (qh->PREmerge || qh->MERGEexact)) {
#ifndef qh_NOmerge
        newfacet->f.samecycle= newfacet;
        newfacet->mergehorizon= True;
#endif
      }
      if (!qh->NEWtentative)
        SETelem_(neighbor->neighbors, horizonskip)= newfacet;
      trace4((qh, qh->ferr, 4049, "qh_makenew_simplicial: create facet f%d top %d from v%d and horizon f%d skip %d top %d and visible f%d skip %d, flip? %d\n",
            newfacet->id, toporient, apex->id, neighbor->id, horizonskip,
              neighbor->toporient, visible->id, visibleskip, flip));
    }
  }
  return newfacet;
} /* makenew_simplicial */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchneighbor">-</a>

  qh_matchneighbor(qh, newfacet, newskip, hashsize, hashcount )
    either match subridge of newfacet with neighbor or add to hash_table

  returns:
    matched ridges of newfacet, except for duplicate ridges
    duplicate ridges marked by qh_DUPLICATEridge for qh_matchdupridge

  notes:
    called by qh_matchnewfacets
    assumes newfacet is simplicial
    ridge is newfacet->vertices w/o newskip vertex
    do not allocate memory (need to free hash_table cleanly)
    uses linear hash chains
    see qh_matchdupridge (poly2_r.c)

  design:
    for each possible matching facet in qh.hash_table
      if vertices match
        set ismatch, if facets have opposite orientation
        if ismatch and matching facet doesn't have a match
          match the facets by updating their neighbor sets
        else
          note: dupridge detected when a match 'f&d skip %d' has already been seen 
                need to mark all of the dupridges for qh_matchdupridge
          indicate a duplicate ridge by qh_DUPLICATEridge and f.dupridge
          add facet to hashtable
          unless the other facet was already a duplicate ridge
            mark both facets with a duplicate ridge
            add other facet (if defined) to hash table

  state at "indicate a duplicate ridge":
    newfacet@newskip= the argument
    facet= the hashed facet@skip that has the same vertices as newfacet@newskip
    same= true if matched vertices have the same orientation
    matchfacet= neighbor at facet@skip
    matchfacet=qh_DUPLICATEridge, matchfacet was previously detected as a dupridge of facet@skip
    ismatch if 'vertex orientation (same) matches facet/newfacet orientation (toporient)
    unknown facet will match later

  details at "indicate a duplicate ridge":
    if !ismatch and matchfacet,
      dupridge is between hashed facet@skip/matchfacet@matchskip and arg newfacet@newskip/unknown 
      set newfacet@newskip, facet@skip, and matchfacet@matchskip to qh_DUPLICATEridge
      add newfacet and matchfacet to hash_table
      if ismatch and matchfacet, 
        same as !ismatch and matchfacet -- it matches facet instead of matchfacet
      if !ismatch and !matchfacet
        dupridge between hashed facet@skip/unknown and arg newfacet@newskip/unknown 
        set newfacet@newskip and facet@skip to qh_DUPLICATEridge
        add newfacet to hash_table
      if ismatch and matchfacet==qh_DUPLICATEridge
        dupridge with already duplicated hashed facet@skip and arg newfacet@newskip/unknown
        set newfacet@newskip to qh_DUPLICATEridge
        add newfacet to hash_table
        facet's hyperplane already set
*/
void qh_matchneighbor(qhT *qh, facetT *newfacet, int newskip, int hashsize, int *hashcount) {
  boolT newfound= False;   /* True, if new facet is already in hash chain */
  boolT same, ismatch;
  int hash, scan;
  facetT *facet, *matchfacet;
  int skip, matchskip;

  hash= qh_gethash(qh, hashsize, newfacet->vertices, qh->hull_dim, 1,
                     SETelem_(newfacet->vertices, newskip));
  trace4((qh, qh->ferr, 4050, "qh_matchneighbor: newfacet f%d skip %d hash %d hashcount %d\n",
          newfacet->id, newskip, hash, *hashcount));
  zinc_(Zhashlookup);
  for (scan=hash; (facet= SETelemt_(qh->hash_table, scan, facetT));
       scan= (++scan >= hashsize ? 0 : scan)) {
    if (facet == newfacet) {
      newfound= True;
      continue;
    }
    zinc_(Zhashtests);
    if (qh_matchvertices(qh, 1, newfacet->vertices, newskip, facet->vertices, &skip, &same)) {
      if (SETelem_(newfacet->vertices, newskip) == SETelem_(facet->vertices, skip)) {
        qh_joggle_restart(qh, "two new facets with the same vertices");
        /* duplicated for multiple skips, not easily avoided */
        qh_fprintf(qh, qh->ferr, 7084, "qhull topology warning (qh_matchneighbor): will merge vertices to undo new facets -- f%d and f%d have the same vertices (skip %d, skip %d) and same horizon ridges to f%d and f%d\n",
          facet->id, newfacet->id, skip, newskip, SETfirstt_(facet->neighbors, facetT)->id, SETfirstt_(newfacet->neighbors, facetT)->id);
        /* will rename a vertex (QH3053).  The fault was duplicate ridges (same vertices) in different facets due to a previous rename.  Expensive to detect beforehand */
      }
      ismatch= (same == (boolT)((newfacet->toporient ^ facet->toporient)));
      matchfacet= SETelemt_(facet->neighbors, skip, facetT);
      if (ismatch && !matchfacet) {
        SETelem_(facet->neighbors, skip)= newfacet;
        SETelem_(newfacet->neighbors, newskip)= facet;
        (*hashcount)--;
        trace4((qh, qh->ferr, 4051, "qh_matchneighbor: f%d skip %d matched with new f%d skip %d\n",
           facet->id, skip, newfacet->id, newskip));
        return;
      }
      if (!qh->PREmerge && !qh->MERGEexact) {
        qh_joggle_restart(qh, "a ridge with more than two neighbors");
        qh_fprintf(qh, qh->ferr, 6107, "qhull topology error: facets f%d, f%d and f%d meet at a ridge with more than 2 neighbors.  Can not continue due to no qh.PREmerge and no 'Qx' (MERGEexact)\n",
                 facet->id, newfacet->id, getid_(matchfacet));
        qh_errexit2(qh, qh_ERRtopology, facet, newfacet);
      }
      SETelem_(newfacet->neighbors, newskip)= qh_DUPLICATEridge;
      newfacet->dupridge= True;
      qh_addhash(newfacet, qh->hash_table, hashsize, hash);
      (*hashcount)++;
      if (matchfacet != qh_DUPLICATEridge) {
        SETelem_(facet->neighbors, skip)= qh_DUPLICATEridge;
        facet->dupridge= True;
        if (matchfacet) {
          matchskip= qh_setindex(matchfacet->neighbors, facet);
          if (matchskip<0) {
              qh_fprintf(qh, qh->ferr, 6260, "qhull topology error (qh_matchneighbor): matchfacet f%d is in f%d neighbors but not vice versa.  Can not continue.\n",
                  matchfacet->id, facet->id);
              qh_errexit2(qh, qh_ERRtopology, matchfacet, facet);
          }
          SETelem_(matchfacet->neighbors, matchskip)= qh_DUPLICATEridge; /* matchskip>=0 by QH6260 */
          matchfacet->dupridge= True;
          qh_addhash(matchfacet, qh->hash_table, hashsize, hash);
          *hashcount += 2;
        }
      }
      trace4((qh, qh->ferr, 4052, "qh_matchneighbor: new f%d skip %d duplicates ridge for f%d skip %d matching f%d ismatch %d at hash %d\n",
           newfacet->id, newskip, facet->id, skip,
           (matchfacet == qh_DUPLICATEridge ? -2 : getid_(matchfacet)),
           ismatch, hash));
      return; /* end of duplicate ridge */
    }
  }
  if (!newfound)
    SETelem_(qh->hash_table, scan)= newfacet;  /* same as qh_addhash */
  (*hashcount)++;
  trace4((qh, qh->ferr, 4053, "qh_matchneighbor: no match for f%d skip %d at hash %d\n",
           newfacet->id, newskip, hash));
} /* matchneighbor */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchnewfacets">-</a>

  qh_matchnewfacets(qh )
    match new facets in qh.newfacet_list to their newfacet neighbors
    all facets are simplicial

  returns:
    if dupridges and merging 
      returns maxdupdist (>=0.0) from vertex to opposite facet
      sets facet->dupridge
      missing neighbor links identify dupridges to be merged (qh_DUPLICATEridge)
    else  
      qh.newfacet_list with full neighbor sets
        vertices for the nth neighbor match all but the nth vertex
    if not merging and qh.FORCEoutput
      for facets with normals (i.e., with dupridges)
      sets facet->flippped for flipped normals, also prevents point partitioning

  notes:
    called by qh_buildcone* and qh_triangulate_facet
    neighbor[0] of new facets is the horizon facet
    if NEWtentative, new facets not attached to the horizon
    assumes qh.hash_table is NULL
    vertex->neighbors has not been updated yet
    do not allocate memory after qh.hash_table (need to free it cleanly)
    
  design:
    truncate neighbor sets to horizon facet for all new facets
    initialize a hash table
    for all new facets
      match facet with neighbors
    if unmatched facets (due to duplicate ridges)
      for each new facet with a duplicate ridge
        try to match facets with the same coplanar horizon
    if not all matched
      for each new facet with a duplicate ridge
        match it with a coplanar facet, or identify a pinched vertex
    if not merging and qh.FORCEoutput
      check for flipped facets
*/
coordT qh_matchnewfacets(qhT *qh /* qh.newfacet_list */) {
  int numnew=0, hashcount=0, newskip;
  facetT *newfacet, *neighbor;
  coordT maxdupdist= 0.0, maxdist2;
  int dim= qh->hull_dim, hashsize, neighbor_i, neighbor_n;
  setT *neighbors;
#ifndef qh_NOtrace
  int facet_i, facet_n, numunused= 0;
  facetT *facet;
#endif

  trace1((qh, qh->ferr, 1019, "qh_matchnewfacets: match neighbors for new facets.\n"));
  FORALLnew_facets {
    numnew++;
    {  /* inline qh_setzero(qh, newfacet->neighbors, 1, qh->hull_dim); */
      neighbors= newfacet->neighbors;
      neighbors->e[neighbors->maxsize].i= dim+1; /*may be overwritten*/
      memset((char *)SETelemaddr_(neighbors, 1, void), 0, (size_t)(dim * SETelemsize));
    }
  }

  qh_newhashtable(qh, numnew*(qh->hull_dim-1)); /* twice what is normally needed,
                                     but every ridge could be DUPLICATEridge */
  hashsize= qh_setsize(qh, qh->hash_table);
  FORALLnew_facets {
    if (!newfacet->simplicial) {
      qh_fprintf(qh, qh->ferr, 6377, "qhull internal error (qh_matchnewfacets): expecting simplicial facets on qh.newfacet_list f%d for qh_matchneighbors, qh_matchneighbor, and qh_matchdupridge.  Got non-simplicial f%d\n",
        qh->newfacet_list->id, newfacet->id);
      qh_errexit2(qh, qh_ERRqhull, newfacet, qh->newfacet_list);
    }
    for (newskip=1; newskip<qh->hull_dim; newskip++) /* furthest/horizon already matched */
      /* hashsize>0 because hull_dim>1 and numnew>0 */
      qh_matchneighbor(qh, newfacet, newskip, hashsize, &hashcount);
#if 0   /* use the following to trap hashcount errors */
    {
      int count= 0, k;
      facetT *facet, *neighbor;

      count= 0;
      FORALLfacet_(qh->newfacet_list) {  /* newfacet already in use */
        for (k=1; k < qh->hull_dim; k++) {
          neighbor= SETelemt_(facet->neighbors, k, facetT);
          if (!neighbor || neighbor == qh_DUPLICATEridge)
            count++;
        }
        if (facet == newfacet)
          break;
      }
      if (count != hashcount) {
        qh_fprintf(qh, qh->ferr, 6266, "qhull error (qh_matchnewfacets): after adding facet %d, hashcount %d != count %d\n",
                 newfacet->id, hashcount, count);
        qh_errexit(qh, qh_ERRdebug, newfacet, NULL);
      }
    }
#endif  /* end of trap code */
  } /* end FORALLnew_facets */
  if (hashcount) { /* all neighbors matched, except for qh_DUPLICATEridge neighbors */
    qh_joggle_restart(qh, "ridge with multiple neighbors");
    if (hashcount) {
      FORALLnew_facets {
        if (newfacet->dupridge) {
          FOREACHneighbor_i_(qh, newfacet) {
            if (neighbor == qh_DUPLICATEridge) {
              maxdist2= qh_matchdupridge(qh, newfacet, neighbor_i, hashsize, &hashcount);
              maximize_(maxdupdist, maxdist2);
            }
          }
        }
      }
    }
  }
  if (hashcount) {
    qh_fprintf(qh, qh->ferr, 6108, "qhull internal error (qh_matchnewfacets): %d neighbors did not match up\n",
        hashcount);
    qh_printhashtable(qh, qh->ferr);
    qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
#ifndef qh_NOtrace
  if (qh->IStracing >= 3) {
    FOREACHfacet_i_(qh, qh->hash_table) {
      if (!facet)
        numunused++;
    }
    qh_fprintf(qh, qh->ferr, 3063, "qh_matchnewfacets: maxdupdist %2.2g, new facets %d, unused hash entries %d, hashsize %d\n",
             maxdupdist, numnew, numunused, qh_setsize(qh, qh->hash_table));
  }
#endif /* !qh_NOtrace */
  qh_setfree(qh, &qh->hash_table);
  if (qh->PREmerge || qh->MERGEexact) {
    if (qh->IStracing >= 4)
      qh_printfacetlist(qh, qh->newfacet_list, NULL, qh_ALL);
  }
  return maxdupdist;
} /* matchnewfacets */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="matchvertices">-</a>

  qh_matchvertices(qh, firstindex, verticesA, skipA, verticesB, skipB, same )
    tests whether vertices match with a single skip
    starts match at firstindex since all new facets have a common vertex

  returns:
    true if matched vertices
    skip index for skipB
    sets same iff vertices have the same orientation

  notes:
    called by qh_matchneighbor and qh_matchdupridge
    assumes skipA is in A and both sets are the same size

  design:
    set up pointers
    scan both sets checking for a match
    test orientation
*/
boolT qh_matchvertices(qhT *qh, int firstindex, setT *verticesA, int skipA,
       setT *verticesB, int *skipB, boolT *same) {
  vertexT **elemAp, **elemBp, **skipBp=NULL, **skipAp;

  elemAp= SETelemaddr_(verticesA, firstindex, vertexT);
  elemBp= SETelemaddr_(verticesB, firstindex, vertexT);
  skipAp= SETelemaddr_(verticesA, skipA, vertexT);
  do if (elemAp != skipAp) {
    while (*elemAp != *elemBp++) {
      if (skipBp)
        return False;
      skipBp= elemBp;  /* one extra like FOREACH */
    }
  }while (*(++elemAp));
  if (!skipBp)
    skipBp= ++elemBp;
  *skipB= SETindex_(verticesB, skipB); /* i.e., skipBp - verticesB
                                       verticesA and verticesB are the same size, otherwise trace4 may segfault */
  *same= !((skipA & 0x1) ^ (*skipB & 0x1)); /* result is 0 or 1 */
  trace4((qh, qh->ferr, 4054, "qh_matchvertices: matched by skip %d(v%d) and skip %d(v%d) same? %d\n",
          skipA, (*skipAp)->id, *skipB, (*(skipBp-1))->id, *same));
  return(True);
} /* matchvertices */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newfacet">-</a>

  qh_newfacet(qh)
    return a new facet

  returns:
    all fields initialized or cleared   (NULL)
    preallocates neighbors set
*/
facetT *qh_newfacet(qhT *qh) {
  facetT *facet;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */

  qh_memalloc_(qh, (int)sizeof(facetT), freelistp, facet, facetT);
  memset((char *)facet, (size_t)0, sizeof(facetT));
  if (qh->facet_id == qh->tracefacet_id)
    qh->tracefacet= facet;
  facet->id= qh->facet_id++;
  facet->neighbors= qh_setnew(qh, qh->hull_dim);
#if !qh_COMPUTEfurthest
  facet->furthestdist= 0.0;
#endif
#if qh_MAXoutside
  if (qh->FORCEoutput && qh->APPROXhull)
    facet->maxoutside= qh->MINoutside;
  else
    facet->maxoutside= qh->DISTround; /* same value as test for QH7082 */
#endif
  facet->simplicial= True;
  facet->good= True;
  facet->newfacet= True;
  trace4((qh, qh->ferr, 4055, "qh_newfacet: created facet f%d\n", facet->id));
  return(facet);
} /* newfacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="newridge">-</a>

  qh_newridge()
    return a new ridge
  notes:
    caller sets qh.traceridge
*/
ridgeT *qh_newridge(qhT *qh) {
  ridgeT *ridge;
  void **freelistp;   /* used if !qh_NOmem by qh_memalloc_() */

  qh_memalloc_(qh, (int)sizeof(ridgeT), freelistp, ridge, ridgeT);
  memset((char *)ridge, (size_t)0, sizeof(ridgeT));
  zinc_(Ztotridges);
  if (qh->ridge_id == UINT_MAX) {
    qh_fprintf(qh, qh->ferr, 7074, "qhull warning: more than 2^32 ridges.  Qhull results are OK.  Since the ridge ID wraps around to 0, two ridges may have the same identifier.\n");
  }
  ridge->id= qh->ridge_id++;
  trace4((qh, qh->ferr, 4056, "qh_newridge: created ridge r%d\n", ridge->id));
  return(ridge);
} /* newridge */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="pointid">-</a>

  qh_pointid(qh, point )
    return id for a point,
    returns qh_IDnone(-3) if null, qh_IDinterior(-2) if interior, or qh_IDunknown(-1) if not known

  alternative code if point is in qh.first_point...
    unsigned long id;
    id= ((unsigned long)point - (unsigned long)qh.first_point)/qh.normal_size;

  notes:
    Valid points are non-negative
    WARN64 -- id truncated to 32-bits, at most 2G points
    NOerrors returned (QhullPoint::id)
    if point not in point array
      the code does a comparison of unrelated pointers.
*/
int qh_pointid(qhT *qh, pointT *point) {
  ptr_intT offset, id;

  if (!point || !qh)
    return qh_IDnone;
  else if (point == qh->interior_point)
    return qh_IDinterior;
  else if (point >= qh->first_point
  && point < qh->first_point + qh->num_points * qh->hull_dim) {
    offset= (ptr_intT)(point - qh->first_point);
    id= offset / qh->hull_dim;
  }else if ((id= qh_setindex(qh->other_points, point)) != -1)
    id += qh->num_points;
  else
    return qh_IDunknown;
  return (int)id;
} /* pointid */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="removefacet">-</a>

  qh_removefacet(qh, facet )
    unlinks facet from qh.facet_list,

  returns:
    updates qh.facet_list .newfacet_list .facet_next visible_list
    decrements qh.num_facets

  see:
    qh_appendfacet
*/
void qh_removefacet(qhT *qh, facetT *facet) {
  facetT *next= facet->next, *previous= facet->previous; /* next is always defined */

  if (facet == qh->newfacet_list)
    qh->newfacet_list= next;
  if (facet == qh->facet_next)
    qh->facet_next= next;
  if (facet == qh->visible_list)
    qh->visible_list= next;
  if (previous) {
    previous->next= next;
    next->previous= previous;
  }else {  /* 1st facet in qh->facet_list */
    qh->facet_list= next;
    qh->facet_list->previous= NULL;
  }
  qh->num_facets--;
  trace4((qh, qh->ferr, 4057, "qh_removefacet: removed f%d from facet_list, newfacet_list, and visible_list\n", facet->id));
} /* removefacet */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="removevertex">-</a>

  qh_removevertex(qh, vertex )
    unlinks vertex from qh.vertex_list,

  returns:
    updates qh.vertex_list .newvertex_list
    decrements qh.num_vertices
*/
void qh_removevertex(qhT *qh, vertexT *vertex) {
  vertexT *next= vertex->next, *previous= vertex->previous; /* next is always defined */

  trace4((qh, qh->ferr, 4058, "qh_removevertex: remove v%d from qh.vertex_list\n", vertex->id));
  if (vertex == qh->newvertex_list)
    qh->newvertex_list= next;
  if (previous) {
    previous->next= next;
    next->previous= previous;
  }else {  /* 1st vertex in qh->vertex_list */
    qh->vertex_list= next;
    qh->vertex_list->previous= NULL;
  }
  qh->num_vertices--;
} /* removevertex */


/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="update_vertexneighbors">-</a>

  qh_update_vertexneighbors(qh )
    update vertex neighbors and delete interior vertices

  returns:
    if qh.VERTEXneighbors, 
      if qh.newvertex_list,
         removes visible neighbors from vertex neighbors
      if qh.newfacet_list
         adds new facets to vertex neighbors
      if qh.visible_list
         interior vertices added to qh.del_vertices for later partitioning as coplanar points
    if not qh.VERTEXneighbors (not merging)
      interior vertices of visible facets added to qh.del_vertices for later partitioning as coplanar points
  
  notes
    [jan'19] split off qh_update_vertexneighbors_cone.  Optimize the remaining cases in a future release
    called by qh_triangulate_facet after triangulating a non-simplicial facet, followed by reset_lists
    called by qh_triangulate after triangulating null and mirror facets
    called by qh_all_vertexmerges after calling qh_merge_pinchedvertices

  design:
    if qh.VERTEXneighbors
      for each vertex on newvertex_list (i.e., new vertices and vertices of new facets)
        delete visible facets from vertex neighbors
      for each new facet on newfacet_list
        for each vertex of facet
          append facet to vertex neighbors
      for each visible facet on qh.visible_list
        for each vertex of facet
          if the vertex is not on a new facet and not itself deleted
            if the vertex has a not-visible neighbor (due to merging)
               remove the visible facet from the vertex's neighbors
            otherwise
               add the vertex to qh.del_vertices for later deletion

    if not qh.VERTEXneighbors (not merging)
      for each vertex of a visible facet
        if the vertex is not on a new facet and not itself deleted
           add the vertex to qh.del_vertices for later deletion
*/
void qh_update_vertexneighbors(qhT *qh /* qh.newvertex_list, newfacet_list, visible_list */) {
  facetT *newfacet= NULL, *neighbor, **neighborp, *visible;
  vertexT *vertex, **vertexp;
  int neighborcount= 0;

  if (qh->VERTEXneighbors) {
    trace3((qh, qh->ferr, 3013, "qh_update_vertexneighbors: update v.neighbors for qh.newvertex_list (v%d) and qh.newfacet_list (f%d)\n",
         getid_(qh->newvertex_list), getid_(qh->newfacet_list)));
    FORALLvertex_(qh->newvertex_list) {
      neighborcount= 0;
      FOREACHneighbor_(vertex) {
        if (neighbor->visible) {
          neighborcount++;
          SETref_(neighbor)= NULL;
        }
      }
      if (neighborcount) {
        trace4((qh, qh->ferr, 4046, "qh_update_vertexneighbors: delete %d of %d vertex neighbors for v%d.  Removes to-be-deleted, visible facets\n",
          neighborcount, qh_setsize(qh, vertex->neighbors), vertex->id));
        qh_setcompact(qh, vertex->neighbors);
      }
    }
    FORALLnew_facets {
      if (qh->first_newfacet && newfacet->id >= qh->first_newfacet) {
        FOREACHvertex_(newfacet->vertices)
          qh_setappend(qh, &vertex->neighbors, newfacet);
      }else {  /* called after qh_merge_pinchedvertices.  In 7-D, many more neighbors than new facets.  qh_setin is expensive */
        FOREACHvertex_(newfacet->vertices)
          qh_setunique(qh, &vertex->neighbors, newfacet); 
      }
    }
    trace3((qh, qh->ferr, 3058, "qh_update_vertexneighbors: delete interior vertices for qh.visible_list (f%d)\n",
        getid_(qh->visible_list)));
    FORALLvisible_facets {
      FOREACHvertex_(visible->vertices) {
        if (!vertex->newfacet && !vertex->deleted) {
          FOREACHneighbor_(vertex) { /* this can happen under merging */
            if (!neighbor->visible)
              break;
          }
          if (neighbor)
            qh_setdel(vertex->neighbors, visible);
          else {
            vertex->deleted= True;
            qh_setappend(qh, &qh->del_vertices, vertex);
            trace2((qh, qh->ferr, 2041, "qh_update_vertexneighbors: delete interior vertex p%d(v%d) of visible f%d\n",
                  qh_pointid(qh, vertex->point), vertex->id, visible->id));
          }
        }
      }
    }
  }else {  /* !VERTEXneighbors */
    trace3((qh, qh->ferr, 3058, "qh_update_vertexneighbors: delete old vertices for qh.visible_list (f%d)\n",
      getid_(qh->visible_list)));
    FORALLvisible_facets {
      FOREACHvertex_(visible->vertices) {
        if (!vertex->newfacet && !vertex->deleted) {
          vertex->deleted= True;
          qh_setappend(qh, &qh->del_vertices, vertex);
          trace2((qh, qh->ferr, 2042, "qh_update_vertexneighbors: will delete interior vertex p%d(v%d) of visible f%d\n",
                  qh_pointid(qh, vertex->point), vertex->id, visible->id));
        }
      }
    }
  }
} /* update_vertexneighbors */

/*-<a                             href="qh-poly_r.htm#TOC"
  >-------------------------------</a><a name="update_vertexneighbors_cone">-</a>

  qh_update_vertexneighbors_cone(qh )
    update vertex neighbors for a cone of new facets and delete interior vertices

  returns:
    if qh.VERTEXneighbors, 
      if qh.newvertex_list,
         removes visible neighbors from vertex neighbors
      if qh.newfacet_list
         adds new facets to vertex neighbors
      if qh.visible_list
         interior vertices added to qh.del_vertices for later partitioning as coplanar points
    if not qh.VERTEXneighbors (not merging)
      interior vertices of visible facets added to qh.del_vertices for later partitioning as coplanar points
  
  notes
    called by qh_addpoint after create cone and before premerge

  design:
    if qh.VERTEXneighbors
      for each vertex on newvertex_list (i.e., new vertices and vertices of new facets)
        delete visible facets from vertex neighbors
      for each new facet on newfacet_list
        for each vertex of facet
          append facet to vertex neighbors
      for each visible facet on qh.visible_list
        for each vertex of facet
          if the vertex is not on a new facet and not itself deleted
            if the vertex has a not-visible neighbor (due to merging)
               remove the visible facet from the vertex's neighbors
            otherwise
               add the vertex to qh.del_vertices for later deletion

    if not qh.VERTEXneighbors (not merging)
      for each vertex of a visible facet
        if the vertex is not on a new facet and not itself deleted
           add the vertex to qh.del_vertices for later deletion

*/
void qh_update_vertexneighbors_cone(qhT *qh /* qh.newvertex_list, newfacet_list, visible_list */) {
  facetT *newfacet= NULL, *neighbor, **neighborp, *visible;
  vertexT *vertex, **vertexp;
  int delcount= 0;

  if (qh->VERTEXneighbors) {
    trace3((qh, qh->ferr, 3059, "qh_update_vertexneighbors_cone: update v.neighbors for qh.newvertex_list (v%d) and qh.newfacet_list (f%d)\n",
         getid_(qh->newvertex_list), getid_(qh->newfacet_list)));
    FORALLvertex_(qh->newvertex_list) {
      delcount= 0;
      FOREACHneighbor_(vertex) {
        if (neighbor->visible) { /* alternative design is a loop over visible facets, but needs qh_setdel() */
          delcount++;
          qh_setdelnth(qh, vertex->neighbors, SETindex_(vertex->neighbors, neighbor));
          neighborp--; /* repeat */
        }
      }
      if (delcount) {
        trace4((qh, qh->ferr, 4021, "qh_update_vertexneighbors_cone: deleted %d visible vertexneighbors of v%d\n",
          delcount, vertex->id));
      }
    }
    FORALLnew_facets {
      FOREACHvertex_(newfacet->vertices)
        qh_setappend(qh, &vertex->neighbors, newfacet);
    }
    trace3((qh, qh->ferr, 3065, "qh_update_vertexneighbors_cone: delete interior vertices, if any, for qh.visible_list (f%d)\n",
        getid_(qh->visible_list)));
    FORALLvisible_facets {
      FOREACHvertex_(visible->vertices) {
        if (!vertex->newfacet && !vertex->deleted) {
          FOREACHneighbor_(vertex) { /* this can happen under merging, qh_checkfacet QH4025 */
            if (!neighbor->visible)
              break;
          }
          if (neighbor)
            qh_setdel(vertex->neighbors, visible);
          else {
            vertex->deleted= True;
            qh_setappend(qh, &qh->del_vertices, vertex);
            trace2((qh, qh->ferr, 2102, "qh_update_vertexneighbors_cone: will delete interior vertex p%d(v%d) of visible f%d\n",
              qh_pointid(qh, vertex->point), vertex->id, visible->id));
          }
        }
      }
    }
  }else {  /* !VERTEXneighbors */
    trace3((qh, qh->ferr, 3066, "qh_update_vertexneighbors_cone: delete interior vertices for qh.visible_list (f%d)\n",
      getid_(qh->visible_list)));
    FORALLvisible_facets {
      FOREACHvertex_(visible->vertices) {
        if (!vertex->newfacet && !vertex->deleted) {
          vertex->deleted= True;
          qh_setappend(qh, &qh->del_vertices, vertex);
          trace2((qh, qh->ferr, 2059, "qh_update_vertexneighbors_cone: will delete interior vertex p%d(v%d) of visible f%d\n",
                  qh_pointid(qh, vertex->point), vertex->id, visible->id));
        }
      }
    }
  }
} /* update_vertexneighbors_cone */


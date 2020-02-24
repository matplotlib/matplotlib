/*<html><pre>  -<a                             href="qh-stat_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   stat_r.c
   contains all statistics that are collected for qhull

   see qh-stat_r.htm and stat_r.h

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/stat_r.c#7 $$Change: 2712 $
   $DateTime: 2019/06/28 12:57:00 $$Author: bbarber $
*/

#include "qhull_ra.h"

/*========== functions in alphabetic order ================*/

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="allstatA">-</a>

  qh_allstatA()
    define statistics in groups of 20

  notes:
    (otherwise, 'gcc -O2' uses too much memory)
    uses qhstat.next
*/
void qh_allstatA(qhT *qh) {

   /* zdef_(type,name,doc,average) */
  zzdef_(zdoc, Zdoc2, "precision statistics", -1);
  zdef_(zinc, Znewvertex, NULL, -1);
  zdef_(wadd, Wnewvertex, "ave. distance of a new vertex to a facet", Znewvertex);
  zzdef_(wmax, Wnewvertexmax, "max. distance of a new vertex to a facet", -1);
  zdef_(wmax, Wvertexmax, "max. distance of an output vertex to a facet", -1);
  zdef_(wmin, Wvertexmin, "min. distance of an output vertex to a facet", -1);
  zdef_(wmin, Wmindenom, "min. denominator in hyperplane computation", -1);

  qh->qhstat.precision= qh->qhstat.next;  /* usually call qh_joggle_restart, printed if Q0 or QJn */
  zzdef_(zdoc, Zdoc3, "precision problems (corrected unless 'Q0' or an error)", -1);
  zzdef_(zinc, Zcoplanarridges, "coplanar half ridges in output", -1);
  zzdef_(zinc, Zconcaveridges, "concave half ridges in output", -1);
  zzdef_(zinc, Zflippedfacets, "flipped facets", -1);
  zzdef_(zinc, Zcoplanarhorizon, "coplanar horizon facets for new vertices", -1);
  zzdef_(zinc, Zcoplanarpart, "coplanar points during partitioning", -1);
  zzdef_(zinc, Zminnorm, "degenerate hyperplanes recomputed with gaussian elimination", -1);
  zzdef_(zinc, Znearlysingular, "nearly singular or axis-parallel hyperplanes", -1);
  zzdef_(zinc, Zback0, "zero divisors during back substitute", -1);
  zzdef_(zinc, Zgauss0, "zero divisors during gaussian elimination", -1);
  zzdef_(zinc, Zmultiridge, "dupridges with multiple neighbors", -1);
  zzdef_(zinc, Zflipridge, "dupridges with flip facet into good neighbor", -1);
  zzdef_(zinc, Zflipridge2, "dupridges with flip facet into good flip neighbor", -1);
}
void qh_allstatB(qhT *qh) {
  zzdef_(zdoc, Zdoc1, "summary information", -1);
  zdef_(zinc, Zvertices, "number of vertices in output", -1);
  zdef_(zinc, Znumfacets, "number of facets in output", -1);
  zdef_(zinc, Znonsimplicial, "number of non-simplicial facets in output", -1);
  zdef_(zinc, Znowsimplicial, "simplicial facets that were non-simplicial", -1);
  zdef_(zinc, Znumridges, "number of ridges in output", -1);
  zdef_(zadd, Znumridges, "average number of ridges per facet", Znumfacets);
  zdef_(zmax, Zmaxridges, "maximum number of ridges", -1);
  zdef_(zadd, Znumneighbors, "average number of neighbors per facet", Znumfacets);
  zdef_(zmax, Zmaxneighbors, "maximum number of neighbors", -1);
  zdef_(zadd, Znumvertices, "average number of vertices per facet", Znumfacets);
  zdef_(zmax, Zmaxvertices, "maximum number of vertices", -1);
  zdef_(zadd, Znumvneighbors, "average number of neighbors per vertex", Zvertices);
  zdef_(zmax, Zmaxvneighbors, "maximum number of neighbors", -1);
  zdef_(wadd, Wcpu, "cpu seconds for qhull after input", -1);
  zdef_(zinc, Ztotvertices, "vertices created altogether", -1);
  zzdef_(zinc, Zsetplane, "facets created altogether", -1);
  zdef_(zinc, Ztotridges, "ridges created altogether", -1);
  zdef_(zinc, Zpostfacets, "facets before post merge", -1);
  zdef_(zadd, Znummergetot, "average merges per facet (at most 511)", Znumfacets);
  zdef_(zmax, Znummergemax, "  maximum merges for a facet (at most 511)", -1);
  zdef_(zinc, Zangle, NULL, -1);
  zdef_(wadd, Wangle, "average cosine (angle) of facet normals for all ridges", Zangle);
  zdef_(wmax, Wanglemax, "  maximum cosine of facet normals (flatest) across a ridge", -1);
  zdef_(wmin, Wanglemin, "  minimum cosine of facet normals (sharpest) across a ridge", -1);
  zdef_(wadd, Wareatot, "total area of facets", -1);
  zdef_(wmax, Wareamax, "  maximum facet area", -1);
  zdef_(wmin, Wareamin, "  minimum facet area", -1);
}
void qh_allstatC(qhT *qh) {
  zdef_(zdoc, Zdoc9, "build hull statistics", -1);
  zzdef_(zinc, Zprocessed, "points processed", -1);
  zzdef_(zinc, Zretry, "retries due to precision problems", -1);
  zdef_(wmax, Wretrymax, "  max. random joggle", -1);
  zdef_(zmax, Zmaxvertex, "max. vertices at any one time", -1);
  zdef_(zinc, Ztotvisible, "ave. visible facets per iteration", Zprocessed);
  zdef_(zinc, Zinsidevisible, "  ave. visible facets without an horizon neighbor", Zprocessed);
  zdef_(zadd, Zvisfacettot,  "  ave. facets deleted per iteration", Zprocessed);
  zdef_(zmax, Zvisfacetmax,  "    maximum", -1);
  zdef_(zadd, Zvisvertextot, "ave. visible vertices per iteration", Zprocessed);
  zdef_(zmax, Zvisvertexmax, "    maximum", -1);
  zdef_(zinc, Ztothorizon, "ave. horizon facets per iteration", Zprocessed);
  zdef_(zadd, Znewfacettot,  "ave. new or merged facets per iteration", Zprocessed);
  zdef_(zmax, Znewfacetmax,  "    maximum (includes initial simplex)", -1);
  zdef_(wadd, Wnewbalance, "average new facet balance", Zprocessed);
  zdef_(wadd, Wnewbalance2, "  standard deviation", -1);
  zdef_(wadd, Wpbalance, "average partition balance", Zpbalance);
  zdef_(wadd, Wpbalance2, "  standard deviation", -1);
  zdef_(zinc, Zpbalance,  "  count", -1);
  zdef_(zinc, Zsearchpoints, "searches of all points for initial simplex", -1);
  zdef_(zinc, Zdetfacetarea, "determinants for facet area", -1);
  zdef_(zinc, Znoarea, "  determinants not computed because vertex too low", -1);
  zdef_(zinc, Zdetsimplex, "determinants for initial hull or voronoi vertices", -1);
  zdef_(zinc, Znotmax, "points ignored (!above max_outside)", -1);
  zdef_(zinc, Zpinchedapex, "points ignored (pinched apex)", -1);
  zdef_(zinc, Znotgood, "points ignored (!above a good facet)", -1);
  zdef_(zinc, Znotgoodnew, "points ignored (didn't create a good new facet)", -1);
  zdef_(zinc, Zgoodfacet, "good facets found", -1);
  zzdef_(zinc, Znumvisibility, "distance tests for facet visibility", -1);
  zdef_(zinc, Zdistvertex, "distance tests to report minimum vertex", -1);
  zzdef_(zinc, Ztotcheck, "points checked for facets' outer planes", -1);
  zzdef_(zinc, Zcheckpart, "  ave. distance tests per check", Ztotcheck);
}
void qh_allstatD(qhT *qh) {
  zdef_(zinc, Zvisit, "resets of visit_id", -1);
  zdef_(zinc, Zvvisit, "  resets of vertex_visit", -1);
  zdef_(zmax, Zvisit2max, "  max visit_id/2", -1);
  zdef_(zmax, Zvvisit2max, "  max vertex_visit/2", -1);

  zdef_(zdoc, Zdoc4, "partitioning statistics (see previous for outer planes)", -1);
  zzdef_(zadd, Zdelvertextot, "total vertices deleted", -1);
  zdef_(zmax, Zdelvertexmax, "    maximum vertices deleted per iteration", -1);
  zdef_(zinc, Zfindbest, "calls to findbest", -1);
  zdef_(zadd, Zfindbesttot, " ave. facets tested", Zfindbest);
  zdef_(zmax, Zfindbestmax, " max. facets tested", -1);
  zdef_(zadd, Zfindcoplanar, " ave. coplanar search", Zfindbest);
  zdef_(zinc, Zfindnew, "calls to findbestnew", -1);
  zdef_(zadd, Zfindnewtot, " ave. facets tested", Zfindnew);
  zdef_(zmax, Zfindnewmax, " max. facets tested", -1);
  zdef_(zinc, Zfindnewjump, " ave. clearly better", Zfindnew);
  zdef_(zinc, Zfindnewsharp, " calls due to qh_sharpnewfacets", -1);
  zdef_(zinc, Zfindhorizon, "calls to findhorizon", -1);
  zdef_(zadd, Zfindhorizontot, " ave. facets tested", Zfindhorizon);
  zdef_(zmax, Zfindhorizonmax, " max. facets tested", -1);
  zdef_(zinc, Zfindjump,       " ave. clearly better", Zfindhorizon);
  zdef_(zinc, Znewbesthorizon, " new bestfacets during qh_findbesthorizon", -1);
  zdef_(zinc, Zpartangle, "angle tests for repartitioned coplanar points", -1);
  zdef_(zinc, Zpartcorner, "  repartitioned coplanar points above a corner facet", -1);
  zdef_(zinc, Zparthidden, "  repartitioned coplanar points above a hidden facet", -1);
  zdef_(zinc, Zparttwisted, "  repartitioned coplanar points above a twisted facet", -1);
}
void qh_allstatE(qhT *qh) {
  zdef_(zinc, Zpartinside, "inside points", -1);
  zdef_(zinc, Zpartnear, "  near inside points kept with a facet", -1);
  zdef_(zinc, Zcoplanarinside, "  inside points that were coplanar with a facet", -1);
  zdef_(zinc, Zbestlower, "calls to findbestlower", -1);
  zdef_(zinc, Zbestlowerv, "  with search of vertex neighbors", -1);
  zdef_(zinc, Zbestlowerall, "  with rare search of all facets", -1);
  zdef_(zmax, Zbestloweralln, "  facets per search of all facets", -1);
  zdef_(wadd, Wmaxout, "difference in max_outside at final check", -1);
  zzdef_(zinc, Zpartitionall, "distance tests for initial partition", -1);
  zdef_(zinc, Ztotpartition, "partitions of a point", -1);
  zzdef_(zinc, Zpartition, "distance tests for partitioning", -1);
  zzdef_(zinc, Zdistcheck, "distance tests for checking flipped facets", -1);
  zzdef_(zinc, Zdistconvex, "distance tests for checking convexity", -1);
  zdef_(zinc, Zdistgood, "distance tests for checking good point", -1);
  zdef_(zinc, Zdistio, "distance tests for output", -1);
  zdef_(zinc, Zdiststat, "distance tests for statistics", -1);
  zzdef_(zinc, Zdistplane, "total number of distance tests", -1);
  zdef_(zinc, Ztotpartcoplanar, "partitions of coplanar points or deleted vertices", -1);
  zzdef_(zinc, Zpartcoplanar, "   distance tests for these partitions", -1);
  zdef_(zinc, Zcomputefurthest, "distance tests for computing furthest", -1);
}
void qh_allstatE2(qhT *qh) {
  zdef_(zdoc, Zdoc5, "statistics for matching ridges", -1);
  zdef_(zinc, Zhashlookup, "total lookups for matching ridges of new facets", -1);
  zdef_(zinc, Zhashtests, "average number of tests to match a ridge", Zhashlookup);
  zdef_(zinc, Zhashridge, "total lookups of subridges (duplicates and boundary)", -1);
  zdef_(zinc, Zhashridgetest, "average number of tests per subridge", Zhashridge);
  zdef_(zinc, Zdupsame, "duplicated ridges in same merge cycle", -1);
  zdef_(zinc, Zdupflip, "duplicated ridges with flipped facets", -1);

  zdef_(zdoc, Zdoc6, "statistics for determining merges", -1);
  zdef_(zinc, Zangletests, "angles computed for ridge convexity", -1);
  zdef_(zinc, Zbestcentrum, "best merges used centrum instead of vertices",-1);
  zzdef_(zinc, Zbestdist, "distance tests for best merge", -1);
  zzdef_(zinc, Zcentrumtests, "distance tests for centrum convexity", -1);
  zzdef_(zinc, Zvertextests, "distance tests for vertex convexity", -1);
  zzdef_(zinc, Zdistzero, "distance tests for checking simplicial convexity", -1);
  zdef_(zinc, Zcoplanarangle, "coplanar angles in getmergeset", -1);
  zdef_(zinc, Zcoplanarcentrum, "coplanar centrums or vertices in getmergeset", -1);
  zdef_(zinc, Zconcaveridge, "concave ridges in getmergeset", -1);
  zdef_(zinc, Zconcavecoplanarridge, "concave-coplanar ridges in getmergeset", -1);
  zdef_(zinc, Ztwistedridge, "twisted ridges in getmergeset", -1);
}
void qh_allstatF(qhT *qh) {
  zdef_(zdoc, Zdoc7, "statistics for merging", -1);
  zdef_(zinc, Zpremergetot, "merge iterations", -1);
  zdef_(zadd, Zmergeinittot, "ave. initial non-convex ridges per iteration", Zpremergetot);
  zdef_(zadd, Zmergeinitmax, "  maximum", -1);
  zdef_(zadd, Zmergesettot, "  ave. additional non-convex ridges per iteration", Zpremergetot);
  zdef_(zadd, Zmergesetmax, "  maximum additional in one pass", -1);
  zdef_(zadd, Zmergeinittot2, "initial non-convex ridges for post merging", -1);
  zdef_(zadd, Zmergesettot2, "  additional non-convex ridges", -1);
  zdef_(wmax, Wmaxoutside, "max distance of vertex or coplanar point above facet (w/roundoff)", -1);
  zdef_(wmin, Wminvertex, "max distance of vertex below facet (or roundoff)", -1);
  zdef_(zinc, Zwidefacet, "centrums frozen due to a wide merge", -1);
  zdef_(zinc, Zwidevertices, "centrums frozen due to extra vertices", -1);
  zzdef_(zinc, Ztotmerge, "total number of facets or cycles of facets merged", -1);
  zdef_(zinc, Zmergesimplex, "merged a simplex", -1);
  zdef_(zinc, Zonehorizon, "simplices merged into coplanar horizon", -1);
  zzdef_(zinc, Zcyclehorizon, "cycles of facets merged into coplanar horizon", -1);
  zzdef_(zadd, Zcyclefacettot, "  ave. facets per cycle", Zcyclehorizon);
  zdef_(zmax, Zcyclefacetmax, "  max. facets", -1);
  zdef_(zinc, Zmergeintocoplanar, "new facets merged into coplanar horizon", -1);
  zdef_(zinc, Zmergeintohorizon, "new facets merged into horizon", -1);
  zdef_(zinc, Zmergenew, "new facets merged", -1);
  zdef_(zinc, Zmergehorizon, "horizon facets merged into new facets", -1);
  zdef_(zinc, Zmergevertex, "vertices deleted by merging", -1);
  zdef_(zinc, Zcyclevertex, "vertices deleted by merging into coplanar horizon", -1);
  zdef_(zinc, Zdegenvertex, "vertices deleted by degenerate facet", -1);
  zdef_(zinc, Zmergeflipdup, "merges due to flipped facets in duplicated ridge", -1);
  zdef_(zinc, Zredundant, "merges due to redundant neighbors", -1);
  zdef_(zinc, Zredundantmerge, "  detected by qh_test_nonsimplicial_merge instead of qh_test_redundant_neighbors", -1);
  zdef_(zadd, Ztestvneighbor, "non-convex vertex neighbors", -1);
}
void qh_allstatG(qhT *qh) {
  zdef_(zinc, Zacoplanar, "merges due to angle coplanar facets", -1);
  zdef_(wadd, Wacoplanartot, "  average merge distance", Zacoplanar);
  zdef_(wmax, Wacoplanarmax, "  maximum merge distance", -1);
  zdef_(zinc, Zcoplanar, "merges due to coplanar facets", -1);
  zdef_(wadd, Wcoplanartot, "  average merge distance", Zcoplanar);
  zdef_(wmax, Wcoplanarmax, "  maximum merge distance", -1);
  zdef_(zinc, Zconcave, "merges due to concave facets", -1);
  zdef_(wadd, Wconcavetot, "  average merge distance", Zconcave);
  zdef_(wmax, Wconcavemax, "  maximum merge distance", -1);
  zdef_(zinc, Zconcavecoplanar, "merges due to concave-coplanar facets", -1);
  zdef_(wadd, Wconcavecoplanartot, "  average merge distance", Zconcavecoplanar);
  zdef_(wmax, Wconcavecoplanarmax, "  maximum merge distance", -1);
  zdef_(zinc, Zavoidold, "coplanar/concave merges due to avoiding old merge", -1);
  zdef_(wadd, Wavoidoldtot, "  average merge distance", Zavoidold);
  zdef_(wmax, Wavoidoldmax, "  maximum merge distance", -1);
  zdef_(zinc, Zdegen, "merges due to degenerate facets", -1);
  zdef_(wadd, Wdegentot, "  average merge distance", Zdegen);
  zdef_(wmax, Wdegenmax, "  maximum merge distance", -1);
  zdef_(zinc, Zflipped, "merges due to removing flipped facets", -1);
  zdef_(wadd, Wflippedtot, "  average merge distance", Zflipped);
  zdef_(wmax, Wflippedmax, "  maximum merge distance", -1);
  zdef_(zinc, Zduplicate, "merges due to dupridges", -1);
  zdef_(wadd, Wduplicatetot, "  average merge distance", Zduplicate);
  zdef_(wmax, Wduplicatemax, "  maximum merge distance", -1);
  zdef_(zinc, Ztwisted, "merges due to twisted facets", -1);
  zdef_(wadd, Wtwistedtot, "  average merge distance", Ztwisted);
  zdef_(wmax, Wtwistedmax, "  maximum merge distance", -1);
}
void qh_allstatH(qhT *qh) {
  zdef_(zdoc, Zdoc8, "statistics for vertex merges", -1);
  zzdef_(zinc, Zpinchduplicate, "merge pinched vertices for a duplicate ridge", -1);
  zzdef_(zinc, Zpinchedvertex, "merge pinched vertices for a dupridge", -1);
  zdef_(zinc, Zrenameshare, "renamed vertices shared by two facets", -1);
  zdef_(zinc, Zrenamepinch, "renamed vertices in a pinched facet", -1);
  zdef_(zinc, Zrenameall, "renamed vertices shared by multiple facets", -1);
  zdef_(zinc, Zfindfail, "rename failures due to duplicated ridges", -1);
  zdef_(zinc, Znewvertexridge, "  found new vertex in ridge", -1);
  zdef_(zinc, Zdelridge, "deleted ridges due to renamed vertices", -1);
  zdef_(zinc, Zdropneighbor, "dropped neighbors due to renamed vertices", -1);
  zdef_(zinc, Zdropdegen, "merge degenerate facets due to dropped neighbors", -1);
  zdef_(zinc, Zdelfacetdup, "  facets deleted because of no neighbors", -1);
  zdef_(zinc, Zremvertex, "vertices removed from facets due to no ridges", -1);
  zdef_(zinc, Zremvertexdel, "  deleted", -1);
  zdef_(zinc, Zretryadd, "retry qh_addpoint after merge pinched vertex", -1);
  zdef_(zadd, Zretryaddtot, "  tot. merge pinched vertex due to dupridge", -1);
  zdef_(zmax, Zretryaddmax, "  max. merge pinched vertex for a qh_addpoint", -1);
  zdef_(zinc, Zintersectnum, "vertex intersections for locating redundant vertices", -1);
  zdef_(zinc, Zintersectfail, "intersections failed to find a redundant vertex", -1);
  zdef_(zinc, Zintersect, "intersections found redundant vertices", -1);
  zdef_(zadd, Zintersecttot, "   ave. number found per vertex", Zintersect);
  zdef_(zmax, Zintersectmax, "   max. found for a vertex", -1);
  zdef_(zinc, Zvertexridge, NULL, -1);
  zdef_(zadd, Zvertexridgetot, "  ave. number of ridges per tested vertex", Zvertexridge);
  zdef_(zmax, Zvertexridgemax, "  max. number of ridges per tested vertex", -1);

  zdef_(zdoc, Zdoc10, "memory usage statistics (in bytes)", -1);
  zdef_(zadd, Zmemfacets, "for facets and their normals, neighbor and vertex sets", -1);
  zdef_(zadd, Zmemvertices, "for vertices and their neighbor sets", -1);
  zdef_(zadd, Zmempoints, "for input points, outside and coplanar sets, and qhT",-1);
  zdef_(zadd, Zmemridges, "for ridges and their vertex sets", -1);
} /* allstat */

void qh_allstatI(qhT *qh) {
  qh->qhstat.vridges= qh->qhstat.next; /* printed in qh_produce_output2 if non-zero Zridge or Zridgemid */
  zzdef_(zdoc, Zdoc11, "Voronoi ridge statistics", -1);
  zzdef_(zinc, Zridge, "non-simplicial Voronoi vertices for all ridges", -1);
  zzdef_(wadd, Wridge, "  ave. distance to ridge", Zridge);
  zzdef_(wmax, Wridgemax, "  max. distance to ridge", -1);
  zzdef_(zinc, Zridgemid, "bounded ridges", -1);
  zzdef_(wadd, Wridgemid, "  ave. distance of midpoint to ridge", Zridgemid);
  zzdef_(wmax, Wridgemidmax, "  max. distance of midpoint to ridge", -1);
  zzdef_(zinc, Zridgeok, "bounded ridges with ok normal", -1);
  zzdef_(wadd, Wridgeok, "  ave. angle to ridge", Zridgeok);
  zzdef_(wmax, Wridgeokmax, "  max. angle to ridge", -1);
  zzdef_(zinc, Zridge0, "bounded ridges with near-zero normal", -1);
  zzdef_(wadd, Wridge0, "  ave. angle to ridge", Zridge0);
  zzdef_(wmax, Wridge0max, "  max. angle to ridge", -1);

  zdef_(zdoc, Zdoc12, "Triangulation statistics ('Qt')", -1);
  zdef_(zinc, Ztricoplanar, "non-simplicial facets triangulated", -1);
  zdef_(zadd, Ztricoplanartot, "  ave. new facets created (may be deleted)", Ztricoplanar);
  zdef_(zmax, Ztricoplanarmax, "  max. new facets created", -1);
  zdef_(zinc, Ztrinull, "null new facets deleted (duplicated vertex)", -1);
  zdef_(zinc, Ztrimirror, "mirrored pairs of new facets deleted (same vertices)", -1);
  zdef_(zinc, Ztridegen, "degenerate new facets in output (same ridge)", -1);
} /* allstat */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="allstatistics">-</a>

  qh_allstatistics()
    reset printed flag for all statistics
*/
void qh_allstatistics(qhT *qh) {
  int i;

  for(i=ZEND; i--; )
    qh->qhstat.printed[i]= False;
} /* allstatistics */

#if qh_KEEPstatistics
/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="collectstatistics">-</a>

  qh_collectstatistics()
    collect statistics for qh.facet_list

*/
void qh_collectstatistics(qhT *qh) {
  facetT *facet, *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  realT dotproduct, dist;
  int sizneighbors, sizridges, sizvertices, i;

  qh->old_randomdist= qh->RANDOMdist;
  qh->RANDOMdist= False;
  zval_(Zmempoints)= qh->num_points * qh->normal_size + (int)sizeof(qhT);
  zval_(Zmemfacets)= 0;
  zval_(Zmemridges)= 0;
  zval_(Zmemvertices)= 0;
  zval_(Zangle)= 0;
  wval_(Wangle)= 0.0;
  zval_(Znumridges)= 0;
  zval_(Znumfacets)= 0;
  zval_(Znumneighbors)= 0;
  zval_(Znumvertices)= 0;
  zval_(Znumvneighbors)= 0;
  zval_(Znummergetot)= 0;
  zval_(Znummergemax)= 0;
  zval_(Zvertices)= qh->num_vertices - qh_setsize(qh, qh->del_vertices);
  if (qh->MERGING || qh->APPROXhull || qh->JOGGLEmax < REALmax/2)
    wmax_(Wmaxoutside, qh->max_outside);
  if (qh->MERGING)
    wmin_(Wminvertex, qh->min_vertex);
  if (!qh_checklists(qh, qh->facet_list)) {
    qh_fprintf(qh, qh->ferr, 6373, "qhull internal error: qh_checklists failed on qh_collectstatistics\n");
    if (!qh->ERREXITcalled)
      qh_errexit(qh, qh_ERRqhull, NULL, NULL);
  }
  FORALLfacets
    facet->seen= False;
  if (qh->DELAUNAY) {
    FORALLfacets {
      if (facet->upperdelaunay != qh->UPPERdelaunay)
        facet->seen= True; /* remove from angle statistics */
    }
  }
  FORALLfacets {
    if (facet->visible && qh->NEWfacets)
      continue;
    sizvertices= qh_setsize(qh, facet->vertices);
    sizneighbors= qh_setsize(qh, facet->neighbors);
    sizridges= qh_setsize(qh, facet->ridges);
    zinc_(Znumfacets);
    zadd_(Znumvertices, sizvertices);
    zmax_(Zmaxvertices, sizvertices);
    zadd_(Znumneighbors, sizneighbors);
    zmax_(Zmaxneighbors, sizneighbors);
    zadd_(Znummergetot, facet->nummerge);
    i= facet->nummerge; /* avoid warnings */
    zmax_(Znummergemax, i);
    if (!facet->simplicial) {
      if (sizvertices == qh->hull_dim) {
        zinc_(Znowsimplicial);
      }else {
        zinc_(Znonsimplicial);
      }
    }
    if (sizridges) {
      zadd_(Znumridges, sizridges);
      zmax_(Zmaxridges, sizridges);
    }
    zadd_(Zmemfacets, (int)sizeof(facetT) + qh->normal_size + 2*(int)sizeof(setT)
       + SETelemsize * (sizneighbors + sizvertices));
    if (facet->ridges) {
      zadd_(Zmemridges,
        (int)sizeof(setT) + SETelemsize * sizridges + sizridges *
         ((int)sizeof(ridgeT) + (int)sizeof(setT) + SETelemsize * (qh->hull_dim-1))/2);
    }
    if (facet->outsideset)
      zadd_(Zmempoints, (int)sizeof(setT) + SETelemsize * qh_setsize(qh, facet->outsideset));
    if (facet->coplanarset)
      zadd_(Zmempoints, (int)sizeof(setT) + SETelemsize * qh_setsize(qh, facet->coplanarset));
    if (facet->seen) /* Delaunay upper envelope */
      continue;
    facet->seen= True;
    FOREACHneighbor_(facet) {
      if (neighbor == qh_DUPLICATEridge || neighbor == qh_MERGEridge
          || neighbor->seen || !facet->normal || !neighbor->normal)
        continue;
      dotproduct= qh_getangle(qh, facet->normal, neighbor->normal);
      zinc_(Zangle);
      wadd_(Wangle, dotproduct);
      wmax_(Wanglemax, dotproduct)
      wmin_(Wanglemin, dotproduct)
    }
    if (facet->normal) {
      FOREACHvertex_(facet->vertices) {
        zinc_(Zdiststat);
        qh_distplane(qh, vertex->point, facet, &dist);
        wmax_(Wvertexmax, dist);
        wmin_(Wvertexmin, dist);
      }
    }
  }
  FORALLvertices {
    if (vertex->deleted)
      continue;
    zadd_(Zmemvertices, (int)sizeof(vertexT));
    if (vertex->neighbors) {
      sizneighbors= qh_setsize(qh, vertex->neighbors);
      zadd_(Znumvneighbors, sizneighbors);
      zmax_(Zmaxvneighbors, sizneighbors);
      zadd_(Zmemvertices, (int)sizeof(vertexT) + SETelemsize * sizneighbors);
    }
  }
  qh->RANDOMdist= qh->old_randomdist;
} /* collectstatistics */
#endif /* qh_KEEPstatistics */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="initstatistics">-</a>

  qh_initstatistics(qh)
    initialize statistics

  notes:
    NOerrors -- qh_initstatistics can not use qh_errexit(), qh_fprintf, or qh.ferr
    On first call, only qhmem.ferr is defined.  qh_memalloc is not setup.
    Also invoked by QhullQh().
*/
void qh_initstatistics(qhT *qh) {
  int i;
  realT realx;
  int intx;

  qh_allstatistics(qh);
  qh->qhstat.next= 0;
  qh_allstatA(qh);
  qh_allstatB(qh);
  qh_allstatC(qh);
  qh_allstatD(qh);
  qh_allstatE(qh);
  qh_allstatE2(qh);
  qh_allstatF(qh);
  qh_allstatG(qh);
  qh_allstatH(qh);
  qh_allstatI(qh);
  if (qh->qhstat.next > (int)sizeof(qh->qhstat.id)) {
    qh_fprintf_stderr(6184, "qhull internal error (qh_initstatistics): increase size of qhstat.id[].  qhstat.next %d should be <= sizeof(qh->qhstat.id) %d\n", 
          qh->qhstat.next, (int)sizeof(qh->qhstat.id));
#if 0 /* for locating error, Znumridges should be duplicated */
    for(i=0; i < ZEND; i++) {
      int j;
      for(j=i+1; j < ZEND; j++) {
        if (qh->qhstat.id[i] == qh->qhstat.id[j]) {
          qh_fprintf_stderr(6185, "qhull error (qh_initstatistics): duplicated statistic %d at indices %d and %d\n",
              qh->qhstat.id[i], i, j);
        }
      }
    }
#endif
    qh_exit(qh_ERRqhull);  /* can not use qh_errexit() */
  }
  qh->qhstat.init[zinc].i= 0;
  qh->qhstat.init[zadd].i= 0;
  qh->qhstat.init[zmin].i= INT_MAX;
  qh->qhstat.init[zmax].i= INT_MIN;
  qh->qhstat.init[wadd].r= 0;
  qh->qhstat.init[wmin].r= REALmax;
  qh->qhstat.init[wmax].r= -REALmax;
  for(i=0; i < ZEND; i++) {
    if (qh->qhstat.type[i] > ZTYPEreal) {
      realx= qh->qhstat.init[(unsigned char)(qh->qhstat.type[i])].r;
      qh->qhstat.stats[i].r= realx;
    }else if (qh->qhstat.type[i] != zdoc) {
      intx= qh->qhstat.init[(unsigned char)(qh->qhstat.type[i])].i;
      qh->qhstat.stats[i].i= intx;
    }
  }
} /* initstatistics */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="newstats">-</a>

  qh_newstats(qh )
    returns True if statistics for zdoc

  returns:
    next zdoc
*/
boolT qh_newstats(qhT *qh, int idx, int *nextindex) {
  boolT isnew= False;
  int start, i;

  if (qh->qhstat.type[qh->qhstat.id[idx]] == zdoc)
    start= idx+1;
  else
    start= idx;
  for(i= start; i < qh->qhstat.next && qh->qhstat.type[qh->qhstat.id[i]] != zdoc; i++) {
    if (!qh_nostatistic(qh, qh->qhstat.id[i]) && !qh->qhstat.printed[qh->qhstat.id[i]])
        isnew= True;
  }
  *nextindex= i;
  return isnew;
} /* newstats */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="nostatistic">-</a>

  qh_nostatistic(qh, index )
    true if no statistic to print
*/
boolT qh_nostatistic(qhT *qh, int i) {

  if ((qh->qhstat.type[i] > ZTYPEreal
       &&qh->qhstat.stats[i].r == qh->qhstat.init[(unsigned char)(qh->qhstat.type[i])].r)
      || (qh->qhstat.type[i] < ZTYPEreal
          &&qh->qhstat.stats[i].i == qh->qhstat.init[(unsigned char)(qh->qhstat.type[i])].i))
    return True;
  return False;
} /* nostatistic */

#if qh_KEEPstatistics
/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printallstatistics">-</a>

  qh_printallstatistics(qh, fp, string )
    print all statistics with header 'string'
*/
void qh_printallstatistics(qhT *qh, FILE *fp, const char *string) {

  qh_allstatistics(qh);
  qh_collectstatistics(qh);
  qh_printstatistics(qh, fp, string);
  qh_memstatistics(qh, fp);
}


/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printstatistics">-</a>

  qh_printstatistics(qh, fp, string )
    print statistics to a file with header 'string'
    skips statistics with qhstat.printed[] (reset with qh_allstatistics)

  see:
    qh_printallstatistics()
*/
void qh_printstatistics(qhT *qh, FILE *fp, const char *string) {
  int i, k;
  realT ave; /* ignored */

  if (qh->num_points != qh->num_vertices || zval_(Zpbalance) == 0) {
    wval_(Wpbalance)= 0.0;
    wval_(Wpbalance2)= 0.0;
  }else
    wval_(Wpbalance2)= qh_stddev(qh, zval_(Zpbalance), wval_(Wpbalance),
                                 wval_(Wpbalance2), &ave);
  if (zval_(Zprocessed) == 0)
    wval_(Wnewbalance2)= 0.0;
  else
    wval_(Wnewbalance2)= qh_stddev(qh, zval_(Zprocessed), wval_(Wnewbalance),
                                 wval_(Wnewbalance2), &ave);
  qh_fprintf(qh, fp, 9350, "\n\
%s\n\
qhull invoked by: %s | %s\n  %s with options:\n%s\n", 
    string, qh->rbox_command, qh->qhull_command, qh_version, qh->qhull_options);

  qh_fprintf(qh, fp, 9351, "\nprecision constants:\n\
 %6.2g max. abs. coordinate in the (transformed) input ('Qbd:n')\n\
 %6.2g max. roundoff error for distance computation ('En')\n\
 %6.2g max. roundoff error for angle computations\n\
 %6.2g min. distance for outside points ('Wn')\n\
 %6.2g min. distance for visible facets ('Vn')\n\
 %6.2g max. distance for coplanar facets ('Un')\n\
 %6.2g max. facet width for recomputing centrum and area\n\
",
  qh->MAXabs_coord, qh->DISTround, qh->ANGLEround, qh->MINoutside,
        qh->MINvisible, qh->MAXcoplanar, qh->WIDEfacet);
  if (qh->KEEPnearinside)
    qh_fprintf(qh, fp, 9352, "\
 %6.2g max. distance for near-inside points\n", qh->NEARinside);
  if (qh->premerge_cos < REALmax/2) qh_fprintf(qh, fp, 9353, "\
 %6.2g max. cosine for pre-merge angle\n", qh->premerge_cos);
  if (qh->PREmerge) qh_fprintf(qh, fp, 9354, "\
 %6.2g radius of pre-merge centrum\n", qh->premerge_centrum);
  if (qh->postmerge_cos < REALmax/2) qh_fprintf(qh, fp, 9355, "\
 %6.2g max. cosine for post-merge angle\n", qh->postmerge_cos);
  if (qh->POSTmerge) qh_fprintf(qh, fp, 9356, "\
 %6.2g radius of post-merge centrum\n", qh->postmerge_centrum);
  qh_fprintf(qh, fp, 9357, "\
 %6.2g max. distance for merging two simplicial facets\n\
 %6.2g max. roundoff error for arithmetic operations\n\
 %6.2g min. denominator for division\n\
  zero diagonal for Gauss: ", qh->ONEmerge, REALepsilon, qh->MINdenom);
  for(k=0; k < qh->hull_dim; k++)
    qh_fprintf(qh, fp, 9358, "%6.2e ", qh->NEARzero[k]);
  qh_fprintf(qh, fp, 9359, "\n\n");
  for(i=0 ; i < qh->qhstat.next; )
    qh_printstats(qh, fp, i, &i);
} /* printstatistics */
#endif /* qh_KEEPstatistics */

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printstatlevel">-</a>

  qh_printstatlevel(qh, fp, id )
    print level information for a statistic

  notes:
    nop if id >= ZEND, printed, or same as initial value
*/
void qh_printstatlevel(qhT *qh, FILE *fp, int id) {

  if (id >= ZEND || qh->qhstat.printed[id])
    return;
  if (qh->qhstat.type[id] == zdoc) {
    qh_fprintf(qh, fp, 9360, "%s\n", qh->qhstat.doc[id]);
    return;
  }
  if (qh_nostatistic(qh, id) || !qh->qhstat.doc[id])
    return;
  qh->qhstat.printed[id]= True;
  if (qh->qhstat.count[id] != -1
      && qh->qhstat.stats[(unsigned char)(qh->qhstat.count[id])].i == 0)
    qh_fprintf(qh, fp, 9361, " *0 cnt*");
  else if (qh->qhstat.type[id] >= ZTYPEreal && qh->qhstat.count[id] == -1)
    qh_fprintf(qh, fp, 9362, "%7.2g", qh->qhstat.stats[id].r);
  else if (qh->qhstat.type[id] >= ZTYPEreal && qh->qhstat.count[id] != -1)
    qh_fprintf(qh, fp, 9363, "%7.2g", qh->qhstat.stats[id].r/ qh->qhstat.stats[(unsigned char)(qh->qhstat.count[id])].i);
  else if (qh->qhstat.type[id] < ZTYPEreal && qh->qhstat.count[id] == -1)
    qh_fprintf(qh, fp, 9364, "%7d", qh->qhstat.stats[id].i);
  else if (qh->qhstat.type[id] < ZTYPEreal && qh->qhstat.count[id] != -1)
    qh_fprintf(qh, fp, 9365, "%7.3g", (realT) qh->qhstat.stats[id].i / qh->qhstat.stats[(unsigned char)(qh->qhstat.count[id])].i);
  qh_fprintf(qh, fp, 9366, " %s\n", qh->qhstat.doc[id]);
} /* printstatlevel */


/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="printstats">-</a>

  qh_printstats(qh, fp, index, nextindex )
    print statistics for a zdoc group

  returns:
    next zdoc if non-null
*/
void qh_printstats(qhT *qh, FILE *fp, int idx, int *nextindex) {
  int j, nexti;

  if (qh_newstats(qh, idx, &nexti)) {
    qh_fprintf(qh, fp, 9367, "\n");
    for (j=idx; j<nexti; j++)
      qh_printstatlevel(qh, fp, qh->qhstat.id[j]);
  }
  if (nextindex)
    *nextindex= nexti;
} /* printstats */

#if qh_KEEPstatistics

/*-<a                             href="qh-stat_r.htm#TOC"
  >-------------------------------</a><a name="stddev">-</a>

  qh_stddev(qh, num, tot, tot2, ave )
    compute the standard deviation and average from statistics

    tot2 is the sum of the squares
  notes:
    computes r.m.s.:
      (x-ave)^2
      == x^2 - 2x tot/num +   (tot/num)^2
      == tot2 - 2 tot tot/num + tot tot/num
      == tot2 - tot ave
*/
realT qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave) {
  realT stddev;

  if (num <= 0) {
    qh_fprintf(qh, qh->ferr, 7101, "qhull warning (qh_stddev): expecting num > 0.  Got num %d, tot %4.4g, tot2 %4.4g.  Returning 0.0\n",
      num, tot, tot2);
    return 0.0;
  }
  *ave= tot/num;
  stddev= sqrt(fabs(tot2/num - *ave * *ave));
  return stddev;
} /* stddev */
#else
realT qh_stddev(qhT *qh, int num, realT tot, realT tot2, realT *ave) { /* for qhull_r-exports.def */
  QHULL_UNUSED(qh)
  QHULL_UNUSED(num)
  QHULL_UNUSED(tot)
  QHULL_UNUSED(tot2)
  QHULL_UNUSED(ave)
    
  return 0.0;
}
#endif /* qh_KEEPstatistics */

#if !qh_KEEPstatistics
void    qh_collectstatistics(qhT *qh) {}
void    qh_printallstatistics(qhT *qh, FILE *fp, const char *string) {}
void    qh_printstatistics(qhT *qh, FILE *fp, const char *string) {}
#endif


/*
  GCNTR.C
  General purpose contour tracer for quadrilateral meshes.
  Handles single level contours, or region between a pair of levels.

  $Id$
 */

/*#include "gist.h"*/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern long GcInit1(long iMax, long jMax, double *x, double *y, int *reg, short *triangle, int\
		    region, const double *zz, double lev, long *nparts);
extern long GcInit2(long iMax, long jMax, double *x, double *y, int *reg, short *triangle, int\
		    region, const double *zz, double levs[2], long nchunk, long *nparts);
extern long GcTrace(long *n, double *px, double *py);
extern int GaFreeScratch(void);



typedef double GpReal;



/* note: The static functions in this file that actually do the work
 * are independent of gist.h, except for the data type GpReal.
 * However, they do make use of the region and triangulation arrays
 * explained in gist.h in the definition of the GaQuadMesh struct.  */

/* What is a contour?
 *
 * Given a quadrilateral mesh (x,y), and values of a z at the points
 * of that mesh, we seek a set of polylines connecting points at a
 * particular value of z.  Each point on such a contour curve lies
 * on an edge of the mesh, at a point linearly interpolated to the
 * contour level z0 between the given values of z at the endpoints
 * of the edge.
 *
 * Identifying these points is easy.  Figuring out how to connect them
 * into a curve -- or possibly a set of disjoint curves -- is difficult.
 * Each disjoint curve may be either a closed circuit, or it may begin
 * and end on a mesh boundary.
 *
 * One of the problems with a quadrilateral mesh is that when the z
 * values at one pair of diagonally opposite points lie below z0, and
 * the values at the other diagonal pair of the same zone lie above z0,
 * all four edges of the zone are cut, and there is an ambiguity in
 * how we should connect the points.  I call this a saddle zone.
 * The problem is that two disjoint curves cut through a saddle zone
 * (I reject the alternative of connecting the opposite points to make
 * a single self-intersecting curve, since those make ugly contour plots
 * -- I've tried it).  The real problem with saddle zones is that you
 * need to communicate the connectivity decision you make back to the
 * calling routine, since for the next contour level, we need to tell
 * the contour tracer to make the same decision as on the previous
 * level.  The input/output triangulation array is the solution to this
 * nasty problem.
 *
 * Another complicating factor is that there may be logical holes in
 * the mesh -- zones which do not exist.  We want our contours to stop
 * if they hit the edge of such a zone, just as if they'd hit the edge
 * of the whole mesh.  The input region array addresses this issue.
 *
 * Yet another complication: We may want a list of closed polygons which
 * outline the region between two contour levels z0 and z1.  These may
 * include sections of the mesh boundary (including edges of logical
 * holes defined by the region array), in addition to sections of the
 * contour curves at one or both levels.  This introduces a huge
 * topological problem -- if one of the closed contours (possibly
 * including an interior logical hole in the mesh, but not any part of
 * the boundary of the whole mesh) encloses a region which is not
 * between z0 and z1, that curve must be connected by a slit (or "branch
 * cut") to the enclosing curve, so that the list of disjoint polygons
 * we return is each simply connected.
 *
 * Okay, one final stunning difficulty: For the two level case, no
 * individual polygon should have more than a few thousand sides, since
 * huge filled polygons place an inordinate load on rendering software,
 * which needs an amount of scratch space proportional to the number
 * of sides it needs to fill.  So in the two level case, we want to
 * chunk the mesh into rectangular pieces of no more than, say, 30x30
 * zones, which keeps each returned polygon to less than a few thousand
 * sides (the worst case is very very bad -- you can easily write down
 * a function and two level values which produce a polygon that cuts
 * every edge of the mesh twice).
 */

/*
 * Here is the numbering scheme for points, edges, and zones in
 * the mesh -- note that each ij corresponds to one point, one zone,
 * one i-edge (i=constant edge) and one j-edge (j=constant edge):
 *
 *             (ij-1)-------(ij)-------(ij)
 *                |                     |
 *                |                     |
 *                |                     |
 *             (ij-1)       (ij)       (ij)
 *                |                     |
 *                |                     |
 *                |                     |
 *           (ij-iX-1)----(ij-iX)----(ij-iX)
 *
 * At each point, the function value is either 0, 1, or 2, depending
 * on whether it is below z0, between z0 and z1, or above z1.
 * Each zone either exists (1) or not (0).
 * From these three bits of data, all of the curve connectivity follows.
 *
 * The tracing algorithm is naturally edge-based: Either you are at a
 * point where a level cuts an edge, ready to step across a zone to
 * another edge, or you are drawing the edge itself, if it happens to
 * be a boundary with at least one section between z0 and z1.
 *
 * In either case, the edge is a directed edge -- either the zone
 * you are advancing into is to its left or right, or you are actually
 * drawing it.  I always trace curves keeping the region between z0 and
 * z1 to the left of the curve.  If I'm tracing a boundary, I'm always
 * moving CCW (counter clockwise) around the zone that exists.  And if
 * I'm about to cross a zone, I'll make the direction of the edge I'm
 * sitting on be such that the zone I'm crossing is to its left.
 *
 * I start tracing each curve near its lower left corner (mesh oriented
 * as above), which is the first point I encounter scanning through the
 * mesh in order.  When I figure the 012 z values and zonal existence,
 * I also mark the potential starting points: Each edge may harbor a
 * potential starting point corresponding to either direction, so there
 * are four start possibilities at each ij point.  Only the following
 * possibilities need to be marked as potential starting edges:
 *
 *     +-+-+-+
 *     | | | |
 *     A-0-C-+    One or both levels cut E and have z=1 above them, and
 *     | EZ| |    0A is cut and either 0C is cut or CD is cut.
 *     +-B-D-+    Or, one or both levels cut E and E is a boundary edge.
 *     | | | |    (and Z exists)
 *     +-+-+-+
 *
 *     +-+-+-+
 *     | | | |
 *     +-A-0-C    One or both levels cut E and have z=1 below them, and
 *     | |ZE |    0A is cut and either 0C is cut or CD is cut.
 *     +-+-B-D    Or, one or both levels cut E and E is a boundary edge.
 *     | | | |    (and Z exists)
 *     +-+-+-+
 *
 *     +-+-+-+
 *     | | | |
 *     +-+-+-+    E is a boundary edge, Z exists, at some point on E
 *     | |Z| |    lies between the levels.
 *     +-+E+-+
 *     | | | |
 *     +-+-+-+
 *
 *     +-+-+-+
 *     | | | |
 *     +-+E+-+    E is a boundary edge, Z exists, at some point on E
 *     | |Z| |    lies between the levels.
 *     +-+-+-+
 *     | | | |
 *     +-+-+-+
 *
 * During the first tracing pass, the start mark is erased whenever
 * any non-starting edge is encountered, reducing the number of points
 * that need to be considered for the second pass.  The first pass
 * makes the basic connectivity decisions.  It figures out how many
 * disjoint curves there will be, and identifies slits for the two level
 * case or open contours for the single level case, and removes all but
 * the actual start markers.  A second tracing pass can perform the
 * actual final trace.
 */

/* ------------------------------------------------------------------------ */

/* the data about edges, zones, and points -- boundary or not, exists
 * or not, z value 0, 1, or 2 -- is kept in a mesh sized data array */
typedef short Cdata;

/* here is the minimum structure required to tell where we are in the
 * mesh sized data array */
typedef struct Csite Csite;
struct Csite {
  long edge;  /* ij of current edge */
  long left;  /* +-1 or +-imax as the zone is to right, left, below,
	       * or above the edge */
  long imax;  /* imax for the mesh */
  long jmax;  /* jmax for the mesh */
  long n;     /* number of points marked on this curve so far */
  long count; /* count of start markers visited */
  GpReal zlevel[2];    /* contour levels, zlevel[1]<=zlevel[0]
			* signals single level case */
  short *triangle;     /* triangulation array for the mesh */
  int *reg;            /* region array for the mesh */

  long edge0, left0;   /* starting site on this curve for closure */
  int level0;          /* starting level for closure */
  long edge00;         /* site needing START_ROW mark */

  /* making the actual marks requires a bunch of other stuff */
  const GpReal *x, *y, *z;   /* mesh coordinates and function values */
  GpReal *xcp, *ycp;         /* output contour points */
};

/* the Cdata array consists of the following bits:
 * Z_VALUE     (2 bits) 0, 1, or 2 function value at point
 * ZONE_EX     1 zone exists, 0 zone doesn't exist
 * I_BNDY      this i-edge (i=constant edge) is a mesh boundary
 * J_BNDY      this j-edge (i=constant edge) is a mesh boundary
 * I0_START    this i-edge is a start point into zone to left
 * I1_START    this i-edge is a start point into zone to right
 * J0_START    this j-edge is a start point into zone below
 * J1_START    this j-edge is a start point into zone above
 * START_ROW   next start point is in current row (accelerates 2nd pass)
 * SLIT_UP     marks this i-edge as the beginning of a slit upstroke
 * SLIT_DN     marks this i-edge as the beginning of a slit downstroke
 * OPEN_END    marks an i-edge start point whose other endpoint is
 *             on a boundary for the single level case
 * ALL_DONE    marks final start point
 */
#define Z_VALUE   0x0003
#define ZONE_EX   0x0004
#define I_BNDY    0x0008
#define J_BNDY    0x0010
#define I0_START  0x0020
#define I1_START  0x0040
#define J0_START  0x0080
#define J1_START  0x0100
#define START_ROW 0x0200
#define SLIT_UP   0x0400
#define SLIT_DN   0x0800
#define OPEN_END  0x1000
#define ALL_DONE  0x2000

/* some helpful macros to find points relative to a given directed
 * edge -- points are designated 0, 1, 2, 3 CCW around zone with 0 and
 * 1 the endpoints of the current edge */
#define FORWARD(left,ix) ((left)>0?((left)>1?1:-(ix)):((left)<-1?-1:(ix)))
#define POINT0(edge,fwd) ((edge)-((fwd)>0?fwd:0))
#define POINT1(edge,fwd) ((edge)+((fwd)<0?fwd:0))
#define IS_JEDGE(edge,left) ((left)>0?((left)>1?1:0):((left)<-1?1:0))
#define ANY_START (I0_START|I1_START|J0_START|J1_START)
#define START_MARK(left) \
  ((left)>0?((left)>1?J1_START:I1_START):((left)<-1?J0_START:I0_START))

/* ------------------------------------------------------------------------ */

/* these actually mark points */
static int zone_crosser(Csite *site, Cdata *data, int level, int pass2);
static int edge_walker(Csite *site, Cdata *data, int pass2);
static int slit_cutter(Csite *site, Cdata *data, int up, int pass2);

/* this calls the first three to trace the next disjoint curve
 * -- return value is number of points on this curve, or
 *    0 if there are no more curves this pass
 *    -(number of points) on first pass if:
 *      this is two level case, and the curve closed on a hole
 *      this is single level case, curve is open, and will start from
 *      a different point on the second pass
 *      -- in both cases, this curve will be combined with another
 *         on the second pass */
static long curve_tracer(Csite *site, Cdata *data, int pass2);

/* this initializes the data array for curve_tracer */
static void data_init(Csite *site, Cdata *data, int region, long nchunk);

/* ------------------------------------------------------------------------ */

/* zone_crosser assumes you are sitting at a cut edge about to cross
 * the current zone.  It always marks the initial point, crosses at
 * least one zone, and marks the final point.  On non-boundary i-edges,
 * it is responsible for removing start markers on the first pass.  */
static int zone_crosser(Csite *site, Cdata *data, int level, int pass2)
{
  long edge= site->edge;
  long left= site->left;
  long n= site->n;
  long fwd= FORWARD(left,site->imax);
  long p0, p1;
  int jedge= IS_JEDGE(edge,left);
  long edge0= site->edge0;
  long left0= site->left0;
  int level0= site->level0==level;
  int two_levels= site->zlevel[1]>site->zlevel[0];
  short *triangle= site->triangle;

  const GpReal *x= pass2? site->x : 0;
  const GpReal *y= pass2? site->y : 0;
  const GpReal *z= pass2? site->z : 0;
  GpReal zlevel= pass2? site->zlevel[level] : 0.0;
  GpReal *xcp= pass2? site->xcp : 0;
  GpReal *ycp= pass2? site->ycp : 0;

  int z0, z1, z2, z3;
  int keep_left= 0;  /* flag to try to minimize curvature in saddles */
  int done= 0;

  if (level) level= 2;

  for (;;) {
    /* set edge endpoints */
    p0= POINT0(edge,fwd);
    p1= POINT1(edge,fwd);

    /* always mark cut on current edge */
    if (pass2) {
      /* second pass actually computes and stores the point */
      GpReal zcp= (zlevel-z[p0])/(z[p1]-z[p0]);
      xcp[n]= zcp*(x[p1]-x[p0]) + x[p0];
      ycp[n]= zcp*(y[p1]-y[p0]) + y[p0];
    }
    if (!done && !jedge) {
      if (n) {
	/* if this is not the first point on the curve, and we're
	 * not done, and this is an i-edge, check several things */
	if (!two_levels && !pass2 && (data[edge]&OPEN_END)) {
	  /* reached an OPEN_END mark, skip the n++ */
	  done= 4;   /* same return value 4 used below */
	  break;
	}

	/* check for curve closure -- if not, erase any start mark */
	if (edge==edge0 && left==left0) {
	  /* may signal closure on a downstroke */
	  if (level0) done= (!pass2 && two_levels && left<0)? 5 : 3;
	} else if (!pass2) {
	  Cdata start= data[edge]&(fwd>0?I0_START:I1_START);
	  if (start) { data[edge]&=~start; site->count--; }
	  if (!two_levels) {
	    start= data[edge]&(fwd>0?I1_START:I0_START);
	    if (start) { data[edge]&=~start; site->count--; }
	  }
	}
      }
    }
    n++;
    if (done) break;

    /* cross current zone to another cut edge */
    z0= (data[p0]&Z_VALUE) != level;     /* 1 if fill toward p0 */
    z1= !z0;                             /* know level cuts edge */
    z2= (data[p1+left]&Z_VALUE) != level;
    z3= (data[p0+left]&Z_VALUE) != level;
    if (z0==z2) {
      if (z1==z3) {
	/* this is a saddle zone, need triangle to decide
	 * -- set triangle if not already decided for this zone */
	long zone= edge + (left>0? left : 0);
	if (triangle) {
	  if (!triangle[zone]) {
	    if (keep_left) triangle[zone]= jedge? -1 : 1;
	    else triangle[zone]= jedge? 1 : -1;
	  }
	  if (triangle[zone]>0? !jedge : jedge) goto bkwd;
	} else {
	  if (keep_left) goto bkwd;
	}
      }
      /* bend forward (right along curve) */
      keep_left= 1;
      jedge= !jedge;
      edge= p1 + (left>0? left : 0);
      { long tmp=fwd; fwd=-left; left=tmp; }
    } else if (z1==z3) {
    bkwd:
      /* bend backward (left along curve) */
      keep_left= 0;
      jedge= !jedge;
      edge= p0 + (left>0? left : 0);
      { long tmp=fwd; fwd=left; left=-tmp; }
    } else {
      /* straight across to opposite edge */
      edge+= left;
    }
    /* after crossing zone, edge/left/fwd is oriented CCW relative to
     * the next zone, assuming we will step there */

    /* now that we've taken a step, check for the downstroke
     * of a slit on the second pass (upstroke checked above)
     * -- taking step first avoids a race condition */
    if (pass2 && two_levels && !jedge) {
      if (left>0) {
	if (data[edge]&SLIT_UP) done= 6;
      } else {
	if (data[edge]&SLIT_DN) done= 5;
      }
    }

    if (!done) {
      /* finally, check if we are on a boundary */
      if (data[edge] & (jedge?J_BNDY:I_BNDY)) {
	done= two_levels? 2 : 4;
	/* flip back into the zone that exists */
	left= -left;
	fwd= -fwd;
	if (!pass2 && (edge!=edge0||left!=left0)) {
	  Cdata start= data[edge]&START_MARK(left);
	  if (start) { data[edge]&=~start; site->count--; }
	}
      }
    }
  }

  site->edge= edge;
  site->n= n;
  site->left= left;
  return done>4? slit_cutter(site, data, done-5, pass2) : done;
}

/* edge_walker assumes that the current edge is being drawn CCW
 * around the current zone.  Since only boundary edges are drawn
 * and we always walk around with the filled region to the left,
 * no edge is ever drawn CW.  We attempt to advance to the next
 * edge on this boundary, but if current second endpoint is not
 * between the two contour levels, we exit back to zone_crosser.
 * Note that we may wind up marking no points.
 * -- edge_walker is never called for single level case */
static int edge_walker(Csite *site, Cdata *data, int pass2)
{
  long edge= site->edge;
  long left= site->left;
  long n= site->n;
  long fwd= FORWARD(left,site->imax);
  long p0= POINT0(edge,fwd);
  long p1= POINT1(edge,fwd);
  int jedge= IS_JEDGE(edge,left);
  long edge0= site->edge0;
  long left0= site->left0;
  int level0= site->level0==2;
  int marked;

  const GpReal *x= pass2? site->x : 0;
  const GpReal *y= pass2? site->y : 0;
  GpReal *xcp= pass2? site->xcp : 0;
  GpReal *ycp= pass2? site->ycp : 0;

  int z0, z1, heads_up= 0;

  for (;;) {
    /* mark endpoint 0 only if value is 1 there, and this is a
     * two level task */
    z0= data[p0]&Z_VALUE;
    z1= data[p1]&Z_VALUE;
    marked= 0;
    if (z0==1) {
      /* mark current boundary point */
      if (pass2) {
	xcp[n]= x[p0];
	ycp[n]= y[p0];
      }
      marked= 1;
    } else if (!n) {
      /* if this is the first point is not between the levels
       * must do the job of the zone_crosser and mark the first cut here,
       * so that it will be marked again by zone_crosser as it closes */
      if (pass2) {
	GpReal zcp= site->zlevel[(z0!=0)];
	zcp= (zcp-site->z[p0])/(site->z[p1]-site->z[p0]);
	xcp[n]= zcp*(x[p1]-x[p0]) + x[p0];
	ycp[n]= zcp*(y[p1]-y[p0]) + y[p0];
      }
      marked= 1;
    }
    if (n) {
      /* check for closure */
      if (level0 && edge==edge0 && left==left0) {
	site->edge= edge;
	site->left= left;
	site->n= n+marked;
	/* if the curve is closing on a hole, need to make a downslit */
	if (fwd<0 && !(data[edge]&(jedge?J_BNDY:I_BNDY)))
	  return slit_cutter(site, data, 0, pass2);
	return 3;
      } else if (pass2) {
	if (heads_up || (fwd<0 && (data[edge]&SLIT_DN))) {
	  site->edge= edge;
	  site->left= left;
	  site->n= n+marked;
	  return slit_cutter(site, data, heads_up, pass2);
	}
      } else {
	/* if this is not first point, clear start mark for this edge */
	Cdata start= data[edge]&START_MARK(left);
	if (start) { data[edge]&=~start; site->count--; }
      }
    }
    if (marked) n++;

    /* if next endpoint not between levels, need to exit to zone_crosser */
    if (z1!=1) {
      site->edge= edge;
      site->left= left;
      site->n= n;
      return (z1!=0);      /* return level closest to p1 */
    }

    /* step to p1 and find next edge
     * -- turn left if possible, else straight, else right
     * -- check for upward slit beginning at same time */
    edge= p1 + (left>0? left : 0);
    if (pass2 && jedge && fwd>0 && (data[edge]&SLIT_UP)) {
      jedge= !jedge;
      heads_up= 1;
    } else if (data[edge]&(jedge?I_BNDY:J_BNDY)) {
      long tmp=fwd; fwd=left; left=-tmp;
      jedge= !jedge;
    } else {
      edge= p1 + (fwd>0? fwd : 0);
      if (pass2 && !jedge && fwd>0 && (data[edge]&SLIT_UP)) {
	heads_up= 1;
      } else if (!(data[edge]&(jedge?J_BNDY:I_BNDY))) {
	edge= p1 - (left<0? left : 0);
	jedge= !jedge;
	{ long tmp=fwd; fwd=-left; left=tmp; }
      }
    }
    p0= p1;
    p1= POINT1(edge,fwd);
  }
}

/* -- slit_cutter is never called for single level case */
static int slit_cutter(Csite *site, Cdata *data, int up, int pass2)
{
  long imax= site->imax;
  long n= site->n;

  const GpReal *x= pass2? site->x : 0;
  const GpReal *y= pass2? site->y : 0;
  GpReal *xcp= pass2? site->xcp : 0;
  GpReal *ycp= pass2? site->ycp : 0;

  if (up) {
    /* upward stroke of slit proceeds up left side of slit until
     * it hits a boundary or a point not between the contour levels
     * -- this never happens on the first pass */
    long p1= site->edge;
    int z1;
    for (;;) {
      z1= data[p1]&Z_VALUE;
      if (z1 != 1) {
	site->edge= p1;
	site->left= -1;
	site->n= n;
	return (z1!=0);
      } else if (data[p1]&J_BNDY) {
	/* this is very unusual case of closing on a mesh hole */
	site->edge= p1;
	site->left= -imax;
	site->n= n;
	return 2;
      }
      xcp[n]= x[p1];
      ycp[n]= y[p1];
      n++;
      p1+= imax;
    }

  } else {
    /* downward stroke proceeds down right side of slit until it
     * hits a boundary or point not between the contour levels */
    long p0= site->edge;
    int z0;
    /* at beginning of first pass, mark first i-edge with SLIT_DN */
    data[p0]|= SLIT_DN;
    p0-= imax;
    for (;;) {
      z0= data[p0]&Z_VALUE;
      if (!pass2) {
	if (z0!=1 || (data[p0]&I_BNDY) || (data[p0+1]&J_BNDY)) {
	  /* at end of first pass, mark final i-edge with SLIT_UP */
	  data[p0+imax]|= SLIT_UP;
	  /* one extra count for splicing at outer curve */
	  site->n= n+1;
	  return 4;   /* return same special value as for OPEN_END */
	}
      } else {
	if (z0 != 1) {
	  site->edge= p0+imax;
	  site->left= 1;
	  site->n= n;
	  return (z0!=0);
	} else if (data[p0+1]&J_BNDY) {
	  site->edge= p0+1;
	  site->left= imax;
	  site->n= n;
	  return 2;
	} else if (data[p0]&I_BNDY) {
	  site->edge= p0;
	  site->left= 1;
	  site->n= n;
	  return 2;
	}
      }
      if (pass2) {
	xcp[n]= x[p0];
	ycp[n]= y[p0];
	n++;
      } else {
	/* on first pass need to count for upstroke as well */
	n+= 2;
      }
      p0-= imax;
    }
  }
}

/* ------------------------------------------------------------------------ */

/* curve_tracer finds the next starting point, then traces the curve,
 * returning the number of points on this curve
 * -- in a two level trace, the return value is negative on the
 *    first pass if the curve closed on a hole
 * -- in a single level trace, the return value is negative on the
 *    first pass if the curve is an incomplete open curve
 * -- a return value of 0 indicates no more curves */
static long curve_tracer(Csite *site, Cdata *data, int pass2)
{
  long imax= site->imax;
  long edge0= site->edge0;
  long left0= site->left0;
  long edge00= site->edge00;
  int two_levels= site->zlevel[1]>site->zlevel[0];
  int level, level0, mark_row;
  long n;

  /* it is possible for a single i-edge to serve as two actual start
   * points, one to the right and one to the left
   * -- for the two level case, this happens on the first pass for
   *    a doubly cut edge, or on a chunking boundary
   * -- for single level case, this is impossible, but a similar
   *    situation involving open curves is handled below
   * a second two start possibility is when the edge0 zone does not
   * exist and both the i-edge and j-edge boundaries are cut
   * yet another possibility is three start points at a junction
   * of chunk cuts
   * -- sigh, several other rare possibilities,
   *    allow for general case, just go in order i1, i0, j1, j0 */
  int two_starts;
  if (left0==1) two_starts= data[edge0]&(I0_START|J1_START|J0_START);
  else if (left0==-1) two_starts= data[edge0]&(J1_START|J0_START);
  else if (left0==imax) two_starts= data[edge0]&J0_START;
  else two_starts= 0;

  if (pass2 || edge0==0) {
    /* zip up to row marked on first pass (or by data_init if edge0==0)
     * -- but not for double start case */
    if (!two_starts) {
      /* final start point marked by ALL_DONE marker */
      int first= (edge0==0 && !pass2);
      long e0= edge0;
      if (data[edge0]&ALL_DONE) return 0;
      while (!(data[edge0]&START_ROW)) edge0+= imax;
      if (e0==edge0) edge0++;  /* two starts handled specially */
      if (first)
	/* if this is the very first start point, we want to remove
	 * the START_ROW marker placed by data_init */
	data[edge0 - edge0%imax]&= ~START_ROW;
    }

  } else {
    /* first pass ends when all potential start points visited */
    if (site->count<=0) {
      /* place ALL_DONE marker for second pass */
      data[edge00]|= ALL_DONE;
      /* reset initial site for second pass */
      site->edge0= site->edge00= site->left0= 0;
      return 0;
    }
    if (!two_starts) edge0++;
  }

  if (two_starts) {
    /* trace second curve with this start immediately */
    if (left0==1 && (data[edge0]&I0_START)) {
      left0= -1;
      level= (data[edge0]&I_BNDY)? 2 : 0;
    } else if ((left0==1 || left0==-1) && (data[edge0]&J1_START)) {
      left0= imax;
      level= 2;
    } else {
      left0= -imax;
      level= 2;
    }

  } else {
    /* usual case is to scan for next start marker
     * -- on second pass, this is at most one row of mesh, but first
     *    pass hits nearly every point of the mesh, since it can't
     *    know in advance which potential start marks removed */
    while (!(data[edge0]&ANY_START)) edge0++;

    if (data[edge0]&I1_START)      left0=  1;
    else if (data[edge0]&I0_START) left0= -1;
    else if (data[edge0]&J1_START) left0=  imax;
    else /*data[edge0]&J0_START*/  left0= -imax;

    if (data[edge0]&(I1_START|I0_START))
      level= (data[edge0]&I_BNDY)? 2 : 0;
    else
      level= 2;
  }

  /* this start marker will not be unmarked, but it has been visited */
  if (!pass2) site->count--;

  /* if this curve starts on a non-boundary i-edge, we need to
   * determine the level */
  if (!level && two_levels)
    level= left0>0?
      ((data[edge0-imax]&Z_VALUE)!=0) : ((data[edge0]&Z_VALUE)!=0);

  /* initialize site for this curve */
  site->edge= site->edge0= edge0;
  site->left= site->left0= left0;
  site->level0= level0= level;  /* for open curve detection only */

  /* single level case just uses zone_crosser */
  if (!two_levels) level= 0;

  /* to generate the curve, alternate between zone_crosser and
   * edge_walker until closure or first call to edge_walker in
   * single level case */
  site->n= 0;
  for (;;) {
    if (level<2)      level= zone_crosser(site, data, level, pass2);
    else if (level<3) level= edge_walker(site, data, pass2);
    else break;
  }
  n= site->n;

  /* single level case may have ended at a boundary rather than closing
   * -- need to recognize this case here in order to place the
   *    OPEN_END mark for zone_crosser, remove this start marker,
   *    and be sure not to make a START_ROW mark for this case
   * two level case may close with slit_cutter, in which case start
   *    must also be removed and no START_ROW mark made
   * -- change sign of return n to inform caller */
  if (!pass2 && level>3 && (two_levels || level0==0)) {
    if (!two_levels) data[edge0]|= OPEN_END;
    data[edge0]&= ~(left0>0? I1_START : I0_START);
    mark_row= 0;  /* do not mark START_ROW */
    n= -n;
  } else {
    if (two_levels) mark_row= !two_starts;
    else mark_row= 1;
  }

  /* on first pass, must apply START_ROW mark in column above previous
   * start marker
   * -- but skip if we just did second of two start case */
  if (!pass2 && mark_row) {
    data[edge0 - (edge0-edge00)%imax]|= START_ROW;
    site->edge00= edge0;
  }

  return n;
}

/* ------------------------------------------------------------------------ */

static void data_init(Csite *site, Cdata *data, int region, long nchunk)
{
  long imax= site->imax;
  long jmax= site->jmax;
  long ijmax= imax*jmax;
  const GpReal *z= site->z;
  GpReal zlev0= site->zlevel[0];
  GpReal zlev1= site->zlevel[1];
  int two_levels= zlev1>zlev0;
  int *reg= site->reg;
  long count= 0;
  int started= 0;
  int ibndy, jbndy, i_was_chunk;

  long icsize= imax-1;
  long jcsize= jmax-1;
  long ichunk, jchunk, irem, jrem, i, j, ij;

  if (nchunk && two_levels) {
    /* figure out chunk sizes
     * -- input nchunk is square root of maximum allowed zones per chunk
     * -- start points for single level case are wrong, so don't try it */
    long inum= (nchunk*nchunk)/(jmax-1);
    long jnum= (nchunk*nchunk)/(imax-1);
    if (inum < nchunk) inum= nchunk;
    if (jnum < nchunk) jnum= nchunk;
    /* ijnum= actual number of chunks,
     * ijrem= number of those chunks needing one more zone (ijcsize+1) */
    inum= (imax-2)/inum + 1;
    icsize= (imax-1)/inum;
    irem=   (imax-1)%inum;
    jnum= (jmax-2)/jnum + 1;
    jcsize= (jmax-1)/jnum;
    jrem=   (jmax-1)%jnum;
    /* convert ijrem into value of i or j at which to begin adding an
     * extra zone */
    irem= (inum-irem)*icsize;
    jrem= (jnum-jrem)*jcsize;
  } else {
    irem= imax;
    jrem= jmax;
  }

  /* do everything in a single pass through the data array to
   * minimize cache faulting (z, reg, and data are potentially
   * very large arrays)
   * access to the z and reg arrays is strictly sequential,
   * but we need two rows (+-imax) of the data array at a time */
  if (z[0]>zlev0) data[0]= (two_levels && z[0]>zlev1)? 2 : 1;
  else data[0]= 0;
  jchunk= 0;
  for (j=ij=0 ; j<jmax ; j++) {
    ichunk= i_was_chunk= 0;
    for (i=0 ; i<imax ; i++,ij++) {
      /* transfer zonal existence from reg to data array
       * -- get these for next row so we can figure existence of
       *    points and j-edges for this row */
      data[ij+imax+1]= 0;
      if (reg) {
	if (region?(reg[ij+imax+1]==region):(reg[ij+imax+1]!=0))
	  data[ij+imax+1]= ZONE_EX;
      } else {
	if (i<imax-1 && j<jmax-1) data[ij+imax+1]= ZONE_EX;
      }

      /* translate z values to 0, 1, 2 flags */
      if (ij<imax) data[ij+1]= 0;
      if (ij<ijmax-1 && z[ij+1]>zlev0)
	data[ij+1]|= (two_levels && z[ij+1]>zlev1)? 2 : 1;

      /* apply edge boundary marks */
      ibndy= i==ichunk || (data[ij]&ZONE_EX)!=(data[ij+1]&ZONE_EX);
      jbndy= j==jchunk || (data[ij]&ZONE_EX)!=(data[ij+imax]&ZONE_EX);
      if (ibndy) data[ij]|= I_BNDY;
      if (jbndy) data[ij]|= J_BNDY;

      /* apply i-edge start marks
       * -- i-edges are only marked when actually cut
       * -- no mark is necessary if one of the j-edges which share
       *    the lower endpoint is also cut
       * -- no I0 mark necessary unless filled region below some cut,
       *    no I1 mark necessary unless filled region above some cut */
      if (j) {
	int v0= (data[ij]&Z_VALUE);
	int vb= (data[ij-imax]&Z_VALUE);
	if (v0!=vb) {         /* i-edge is cut */
	  if (ibndy) {
	    if (data[ij]&ZONE_EX) {
	      data[ij]|= I0_START;
	      count++;
	    }
	    if (data[ij+1]&ZONE_EX) {
	      data[ij]|= I1_START;
	      count++;
	    }
	  } else {
	    int va= (data[ij-1]&Z_VALUE);
	    int vc= (data[ij+1]&Z_VALUE);
	    int vd= (data[ij-imax+1]&Z_VALUE);
	    if (v0!=1 && va!=v0 && (vc!=v0 || vd!=v0) &&
		(data[ij]&ZONE_EX)) {
	      data[ij]|= I0_START;
	      count++;
	    }
	    if (vb!=1 && va==vb && (vc==vb || vd==vb) &&
		(data[ij+1]&ZONE_EX)) {
	      data[ij]|= I1_START;
	      count++;
	    }
	  }
	}
      }

      /* apply j-edge start marks
       * -- j-edges are only marked when they are boundaries
       * -- all cut boundary edges marked
       * -- for two level case, a few uncut edges must be marked
       */
      if (i && jbndy) {
	int v0= (data[ij]&Z_VALUE);
	int vb= (data[ij-1]&Z_VALUE);
	if (v0!=vb) {
	  if (data[ij]&ZONE_EX) {
	    data[ij]|= J0_START;
	    count++;
	  }
	  if (data[ij+imax]&ZONE_EX) {
	    data[ij]|= J1_START;
	    count++;
	  }
	} else if (two_levels && v0==1) {
	  if (data[ij+imax]&ZONE_EX) {
	    if (i_was_chunk || !(data[ij+imax-1]&ZONE_EX)) {
	      /* lower left is a drawn part of boundary */
	      data[ij]|= J1_START;
	      count++;
	    }
	  } else if (data[ij]&ZONE_EX) {
	    if (data[ij+imax-1]&ZONE_EX) {
	      /* weird case of open hole at lower left */
	      data[ij]|= J0_START;
	      count++;
	    }
	  }
	}
      }

      i_was_chunk= (i==ichunk);
      if (i_was_chunk) ichunk+= icsize + (ichunk>=irem);
    }

    if (j==jchunk) jchunk+= jcsize + (jchunk>=jrem);

    /* place first START_ROW marker */
    if (count && !started) {
      data[ij-imax]|= START_ROW;
      started= 1;
    }
  }

  /* place immediate stop mark if nothing found */
  if (!count) data[0]|= ALL_DONE;

  /* initialize site */
  site->edge0= site->edge00= site->edge= 0;
  site->left0= site->left= 0;
  site->n= 0;
  site->count= count;
}

/* ------------------------------------------------------------------------ */

/* here are the interface routines for Gist */

int GaGetScratchS(long n);
short *gasScratch;
static long nScratchP =0, nScratchS = 0;
double *gaxScratch, *gayScratch;
static long nScratch= 0;
static double *xScratch, *yScratch;
char gistError[128];  /* most recent error message */

static Csite gc_site;

static void MMError(void)
{
  strcpy(gistError, "memory manager failed in gist.c function");
}



static long gc_common(long iMax, long jMax, double *x, double *y, int *reg, short *triangle, int region, const double *zz, long nchunk, long *nparts);

extern long GcInit1(long iMax, long jMax, double *x, double *y, int *reg, short *triangle, int region, const double *zz, double lev, long *nparts)
{

  gc_site.zlevel[0]= gc_site.zlevel[1]= lev;
  
  return gc_common(iMax, jMax, x, y, reg, triangle, region, zz, 0L, nparts);
}

long GcInit2(long iMax, long jMax, double *x, double *y, int *reg, short *triangle, int region, const double  *zz, double levs[2], long nchunk, long *nparts)
{
  gc_site.zlevel[0]= levs[0];
  gc_site.zlevel[1]= levs[1];

  *nparts= 0;
  if (gc_site.zlevel[0]==gc_site.zlevel[1]) return 0;
  if (gc_site.zlevel[0] > gc_site.zlevel[1]) {
    GpReal tmp= gc_site.zlevel[0];
    gc_site.zlevel[0]= gc_site.zlevel[1];
    gc_site.zlevel[1]= tmp;
  }
  
  return gc_common(iMax, jMax, x, y, reg, triangle, region, zz, nchunk, nparts);
}

static long gc_common(long iMax, long jMax, double *x, double *y, int *reg, short *triangle, int region, const double  *zz, long nchunk, long *nparts)
{
  long ntotal= 0;
  long n;

  gc_site.imax= iMax;
  gc_site.jmax= jMax;
  gc_site.x= x;
  gc_site.y= y;
  gc_site.reg= reg;
  gc_site.triangle= triangle;
  gc_site.z= zz;

  gc_site.xcp= gc_site.ycp= 0;
  gc_site.n= gc_site.count= 0;

  *nparts= 0;
  /* get scratch space for data array */
  if (GaGetScratchS(gc_site.imax*(gc_site.jmax+1)+1)) return 0;

  /* initialize the data array */
  data_init(&gc_site, gasScratch, region, nchunk);


  /* make first pass to compute required sizes for GcTrace second pass */
  for (;;) {

    n= curve_tracer(&gc_site, gasScratch, 0);

    if (!n) break;
    if (n>0) {
      (*nparts)++;
      //printf("nparts: %ld, %ld\n", *nparts, n);
      ntotal+= n;
    } else {
      ntotal-= n;
    }
  }

  return ntotal;
}

long GcTrace(long *n, GpReal *px, GpReal *py)
{
  long np, ntotal= 0;

  /* make second pass to fill outputs */
  for (;;) {
    gc_site.xcp= px;
    gc_site.ycp= py;

    np= curve_tracer(&gc_site, gasScratch, 1);

    if (!np) break;
    if (np>0) {

      *(n++)=np;
      //printf("n: %ld, %ld\n", *n, np);
      px+= np;
      py+= np;
      ntotal+= np;
    } else {

      /* serious bug */
      ntotal= -1;
      break;
    }
  }

  /* data array is likely big -- go ahead and free it */
  GaFreeScratch();

  return ntotal;
}

/* ------------------------------------------------------------------------ */

int GaFreeScratch(void)
{
  if (nScratchP>0) { free(gaxScratch);  free(gayScratch); gaxScratch=NULL; gayScratch=NULL;}
  if (nScratchS>0) {free(gasScratch); gasScratch=NULL;}
  if (nScratch>0) { free(xScratch);   free(yScratch); xScratch=NULL; yScratch=NULL;}
  nScratchP= nScratchS= nScratch= 0;
  return 0;
}


int GaGetScratchS(long n)
{
  if (n<=nScratchS) return 0;
  if (nScratchS>0) {free(gasScratch); gasScratch=NULL;}
  gasScratch= (short *)malloc(sizeof(short)*n);
  if (!gasScratch) {
    nScratchS= 0;
    MMError();
    return 1;
  }

  nScratchS= n;
  return 0;
}

#include "delaunay_utils.h"
#include <algorithm>
#include <queue>
#include <vector>
#include <iostream>

using namespace std;

int walking_triangles(int start, double targetx, double targety, 
    double *x, double *y, int *nodes, int *neighbors)
{
    int i, j, k, t;

    if (start == -1) start = 0;
    t = start;
    while (1) {
        for (i=0; i<3; i++) {
            j = EDGE0(i);
            k = EDGE1(i);
            if (ONRIGHT(x[INDEX3(nodes,t,j)], y[INDEX3(nodes,t,j)],
                        x[INDEX3(nodes,t,k)], y[INDEX3(nodes,t,k)],
                        targetx, targety)) {
                t = INDEX3(neighbors, t, i);
                if (t < 0) return t; // outside the convex hull
                break;
            }
        }
        if (i == 3) break;
    }

    return t;
}

void getminmax(double *arr, int n, double& minimum, double& maximum)
{
    int i;
    minimum = arr[0];
    maximum = arr[0];
    for (i=1; i<n; i++) {
        if (arr[i] < minimum) {
            minimum = arr[i];
        } else if (arr[i] > maximum) {
            maximum = arr[i];
        }
    }
}

// Find the circumcenter of three 2-D points by Cramer's Rule to find
// the intersection of two perpendicular bisectors of the triangle's 
// edges.
// http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
//
// Return true if successful; return false if points are collinear
bool circumcenter(double x0, double y0,
                  double x1, double y1,
                  double x2, double y2,
                  double& centerx, double& centery)
{
    double D;
    double x0m2, y1m2, x1m2, y0m2;
    double x0p2, y1p2, x1p2, y0p2;
    x0m2 = x0 - x2;
    y1m2 = y1 - y2;
    x1m2 = x1 - x2;
    y0m2 = y0 - y2;
    x0p2 = x0 + x2;
    y1p2 = y1 + y2;
    x1p2 = x1 + x2;
    y0p2 = y0 + y2;

    D = x0m2*y1m2 - x1m2*y0m2;
    if ((D < TOLERANCE_EPS) && (D > -TOLERANCE_EPS)) return false;

    centerx = (((x0m2*x0p2 + y0m2*y0p2)/2*y1m2) 
              - (x1m2*x1p2 + y1m2*y1p2)/2*y0m2) / D;
    centery = (((x1m2*x1p2 + y1m2*y1p2)/2*x0m2)
              - (x0m2*x0p2 + y0m2*y0p2)/2*x1m2) / D;

    return true;
}

double signed_area(double x0, double y0,
                   double x1, double y1,
                   double x2, double y2)
{
    return 0.5*(x0 * (y1 - y2)
              + x1 * (y2 - y0)
              + x2 * (y0 - y1));
}

ConvexPolygon::ConvexPolygon(){
    seeded = false;
}

ConvexPolygon::~ConvexPolygon() {
}

void ConvexPolygon::seed(double x0c, double y0c) {
    this->x0 = x0c;
    this->y0 = y0c;
}

void ConvexPolygon::push(double x, double y) {
    if (!seeded) {
        seed(x, y);
        seeded = true;
        return;
    }
    SeededPoint xy(this->x0, this->y0, x, y);

    this->points.push_back(xy);
}

double ConvexPolygon::area() {
    double A=0.0;
    int i;

    sort(this->points.begin(), this->points.end());
    
    this->points.push_back(SeededPoint(x0, y0, x0, y0));
    int n = (int)this->points.size();

    for (i=0; i<n; i++) {
        int j = i-1;
        if (j<0) j = n-1;
        int k = i+1;
        if (k>=n) k = 0;

        A += points[i].x*(points[k].y - points[j].y);
    }

    A *= 0.5;
    return A;
}


#ifndef _DELAUNAY_UTILS_H
#define _DELAUNAY_UTILS_H

#include <vector>
#include <queue>
#include <functional>

using namespace std;

#define ONRIGHT(x0, y0, x1, y1, x, y) ((y0-y)*(x1-x) > (x0-x)*(y1-y))
#define EDGE0(node) ((node + 1) % 3)
#define EDGE1(node) ((node + 2) % 3)
#define INDEX2(arr,ix,jx) (arr[2*ix+jx])
#define INDEX3(arr,ix,jx) (arr[3*ix+jx])
#define INDEXN(arr,N,ix,jx) (arr[N*ix+jx])
#define SQ(a) ((a)*(a))

#define TOLERANCE_EPS (4e-13)
#define PERTURB_EPS (1e-3)
#define GINORMOUS (1e100)

extern int walking_triangles(int start, double targetx, double targety, 
    double *x, double *y, int *nodes, int *neighbors);
extern void getminmax(double *arr, int n, double& minimum, double& maximum);
extern bool circumcenter(double x0, double y0,
                         double x1, double y1,
                         double x2, double y2,
                         double& centerx, double& centery);
extern double signed_area(double x0, double y0,
                          double x1, double y1,
                          double x2, double y2);

class SeededPoint {
public:
    SeededPoint() {};
    SeededPoint(double x0c, double y0c, double xc, double yc) {
        this->x0 = x0c;
        this->y0 = y0c;
        this->x = xc;
        this->y = yc;
    };
    ~SeededPoint() {};

    double x0, y0;
    double x, y;

    bool operator<(const SeededPoint& p2) const {
        double test = (this->y0-p2.y)*(this->x-p2.x) - (this->x0-p2.x)*(this->y-p2.y);
        if (test == 0) {
            double length1 = SQ(this->x-this->x0) + SQ(this->y-this->y0);
            double length2 = SQ(p2.x-this->x0) + SQ(p2.y-this->y0);

            return (length2 > length1);
        } else return (test < 0);
    }

};

class ConvexPolygon {
public:
    ConvexPolygon();
    ~ConvexPolygon();

    void seed(double x0c, double y0c);
    void push(double x, double y);

    double area();

// private:  // I don't care much for data-hiding
    double x0, y0;
    vector<SeededPoint> points;
    bool seeded;
};


#endif // _DELAUNAY_UTILS_H

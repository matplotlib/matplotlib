#ifndef _NATNEIGHBORS_H
#define _NATNEIGHBORS_H

#include <list>
using namespace std;

class NaturalNeighbors
{
public:
    NaturalNeighbors(int npoints, int ntriangles, double *x, double *y,
        double *centers, int *nodes, int *neighbors);
    ~NaturalNeighbors();

    double interpolate_one(double *z, double targetx, double targety,
        double defvalue, int &start_triangle);

    void interpolate_grid(double *z, 
        double x0, double x1, int xsteps,
        double y0, double y1, int ysteps,
        double *output, double defvalue, int start_triangle);

    void interpolate_unstructured(double *z, int size, 
        double *intx, double *inty, double *output, double defvalue);

private:
    int npoints, ntriangles;
    double *x, *y, *centers, *radii2;
    int *nodes, *neighbors;

    int find_containing_triangle(double targetx, double targety, int start_triangle);
};

#endif // _NATNEIGHBORS_H

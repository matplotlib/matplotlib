/* -*- mode: c++; c-basic-offset: 4 -*- */

#define NO_IMPORT_ARRAY

#include <math.h>

// utilities for irregular grids
void _bin_indices_middle(
    unsigned int *irows, int nrows, const float *ys1, unsigned long ny, float dy, float y_min)
{
    int i, j, j_last;
    unsigned int *rowstart = irows;
    const float *ys2 = ys1 + 1;
    const float *yl = ys1 + ny;
    float yo = y_min + dy / 2.0;
    float ym = 0.5f * (*ys1 + *ys2);
    // y/rows
    j = 0;
    j_last = j;
    for (i = 0; i < nrows; i++, yo += dy, rowstart++) {
        while (ys2 != yl && yo > ym) {
            ys1 = ys2;
            ys2 = ys1 + 1;
            ym = 0.5f * (*ys1 + *ys2);
            j++;
        }
        *rowstart = j - j_last;
        j_last = j;
    }
}

void _bin_indices_middle_linear(float *arows,
                                unsigned int *irows,
                                int nrows,
                                const float *y,
                                unsigned long ny,
                                float dy,
                                float y_min)
{
    int i;
    int ii = 0;
    int iilast = (int)ny - 1;
    float sc = 1 / dy;
    int iy0 = (int)floor(sc * (y[ii] - y_min));
    int iy1 = (int)floor(sc * (y[ii + 1] - y_min));
    float invgap = 1.0f / (iy1 - iy0);
    for (i = 0; i < nrows && i <= iy0; i++) {
        irows[i] = 0;
        arows[i] = 1.0;
    }
    for (; i < nrows; i++) {
        while (i > iy1 && ii < iilast) {
            ii++;
            iy0 = iy1;
            iy1 = (int)floor(sc * (y[ii + 1] - y_min));
            invgap = 1.0f / (iy1 - iy0);
        }
        if (i >= iy0 && i <= iy1) {
            irows[i] = ii;
            arows[i] = (iy1 - i) * invgap;
        } else
            break;
    }
    for (; i < nrows; i++) {
        irows[i] = iilast - 1;
        arows[i] = 0.0;
    }
}

void _bin_indices(int *irows, int nrows, const double *y, unsigned long ny, double sc, double offs)
{
    int i;
    if (sc * (y[ny - 1] - y[0]) > 0) {
        int ii = 0;
        int iilast = (int)ny - 1;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii + 1] - offs));
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii < iilast) {
                ii++;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii + 1] - offs));
            }
            if (i >= iy0 && i <= iy1)
                irows[i] = ii;
            else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    } else {
        int iilast = (int)ny - 1;
        int ii = iilast;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii - 1] - offs));
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii > 1) {
                ii--;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii - 1] - offs));
            }
            if (i >= iy0 && i <= iy1)
                irows[i] = ii - 1;
            else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    }
}

void _bin_indices_linear(
    float *arows, int *irows, int nrows, double *y, unsigned long ny, double sc, double offs)
{
    int i;
    if (sc * (y[ny - 1] - y[0]) > 0) {
        int ii = 0;
        int iilast = (int)ny - 1;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii + 1] - offs));
        float invgap = 1.0 / (iy1 - iy0);
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii < iilast) {
                ii++;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii + 1] - offs));
                invgap = 1.0 / (iy1 - iy0);
            }
            if (i >= iy0 && i <= iy1) {
                irows[i] = ii;
                arows[i] = (iy1 - i) * invgap;
            } else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    } else {
        int iilast = (int)ny - 1;
        int ii = iilast;
        int iy0 = (int)floor(sc * (y[ii] - offs));
        int iy1 = (int)floor(sc * (y[ii - 1] - offs));
        float invgap = 1.0 / (iy1 - iy0);
        for (i = 0; i < nrows && i < iy0; i++) {
            irows[i] = -1;
        }
        for (; i < nrows; i++) {
            while (i > iy1 && ii > 1) {
                ii--;
                iy0 = iy1;
                iy1 = (int)floor(sc * (y[ii - 1] - offs));
                invgap = 1.0 / (iy1 - iy0);
            }
            if (i >= iy0 && i <= iy1) {
                irows[i] = ii - 1;
                arows[i] = (i - iy0) * invgap;
            } else
                break;
        }
        for (; i < nrows; i++) {
            irows[i] = -1;
        }
    }
}

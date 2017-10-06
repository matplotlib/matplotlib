/* -*- mode: c++; c-basic-offset: 4 -*- */

/* image.h
 *
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <vector>


// utilities for irregular grids
void _bin_indices_middle(
    unsigned int *irows, int nrows, const float *ys1, unsigned long ny, float dy, float y_min);
void _bin_indices_middle_linear(float *arows,
                                unsigned int *irows,
                                int nrows,
                                const float *y,
                                unsigned long ny,
                                float dy,
                                float y_min);
void _bin_indices(int *irows, int nrows, const double *y, unsigned long ny, double sc, double offs);
void _bin_indices_linear(
    float *arows, int *irows, int nrows, double *y, unsigned long ny, double sc, double offs);

template <class CoordinateArray, class ColorArray, class OutputArray>
void pcolor(CoordinateArray &x,
            CoordinateArray &y,
            ColorArray &d,
            unsigned int rows,
            unsigned int cols,
            float bounds[4],
            int interpolation,
            OutputArray &out)
{
    if (rows >= 32768 || cols >= 32768) {
        throw std::runtime_error("rows and cols must both be less than 32768");
    }

    float x_min = bounds[0];
    float x_max = bounds[1];
    float y_min = bounds[2];
    float y_max = bounds[3];
    float width = x_max - x_min;
    float height = y_max - y_min;
    float dx = width / ((float)cols);
    float dy = height / ((float)rows);

    // Check we have something to output to
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Cannot scale to zero size");
    }

    if (d.dim(2) != 4) {
        throw std::runtime_error("data must be in RGBA format");
    }

    // Check dimensions match
    unsigned long nx = x.dim(0);
    unsigned long ny = y.dim(0);
    if (nx != (unsigned long)d.dim(1) || ny != (unsigned long)d.dim(0)) {
        throw std::runtime_error("data and axis dimensions do not match");
    }

    // Allocate memory for pointer arrays
    std::vector<unsigned int> rowstarts(rows);
    std::vector<unsigned int> colstarts(cols);

    // Calculate the pointer arrays to map input x to output x
    unsigned int i, j;
    unsigned int *colstart = &colstarts[0];
    unsigned int *rowstart = &rowstarts[0];
    const float *xs1 = x.data();
    const float *ys1 = y.data();

    // Copy data to output buffer
    const unsigned char *start;
    const unsigned char *inposition;
    size_t inrowsize = nx * 4;
    size_t rowsize = cols * 4;
    unsigned char *position = (unsigned char *)out.data();
    unsigned char *oldposition = NULL;
    start = d.data();

    if (interpolation == NEAREST) {
        _bin_indices_middle(colstart, cols, xs1, nx, dx, x_min);
        _bin_indices_middle(rowstart, rows, ys1, ny, dy, y_min);
        for (i = 0; i < rows; i++, rowstart++) {
            if (i > 0 && *rowstart == 0) {
                memcpy(position, oldposition, rowsize * sizeof(unsigned char));
                oldposition = position;
                position += rowsize;
            } else {
                oldposition = position;
                start += *rowstart * inrowsize;
                inposition = start;
                for (j = 0, colstart = &colstarts[0]; j < cols; j++, position += 4, colstart++) {
                    inposition += *colstart * 4;
                    memcpy(position, inposition, 4 * sizeof(unsigned char));
                }
            }
        }
    } else if (interpolation == BILINEAR) {
        std::vector<float> acols(cols);
        std::vector<float> arows(rows);

        _bin_indices_middle_linear(&acols[0], colstart, cols, xs1, nx, dx, x_min);
        _bin_indices_middle_linear(&arows[0], rowstart, rows, ys1, ny, dy, y_min);
        double a00, a01, a10, a11, alpha, beta;

        // Copy data to output buffer
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                alpha = arows[i];
                beta = acols[j];

                a00 = alpha * beta;
                a01 = alpha * (1.0 - beta);
                a10 = (1.0 - alpha) * beta;
                a11 = 1.0 - a00 - a01 - a10;

                for (size_t k = 0; k < 4; ++k) {
                    position[k] =
                        d(rowstart[i], colstart[j], k) * a00 +
                        d(rowstart[i], colstart[j] + 1, k) * a01 +
                        d(rowstart[i] + 1, colstart[j], k) * a10 +
                        d(rowstart[i] + 1, colstart[j] + 1, k) * a11;
                }
                position += 4;
            }
        }
    }
}

template <class CoordinateArray, class ColorArray, class Color, class OutputArray>
void pcolor2(CoordinateArray &x,
             CoordinateArray &y,
             ColorArray &d,
             unsigned int rows,
             unsigned int cols,
             float bounds[4],
             Color &bg,
             OutputArray &out)
{
    double x_left = bounds[0];
    double x_right = bounds[1];
    double y_bot = bounds[2];
    double y_top = bounds[3];

    // Check we have something to output to
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("rows or cols is zero; there are no pixels");
    }

    if (d.dim(2) != 4) {
        throw std::runtime_error("data must be in RGBA format");
    }

    // Check dimensions match
    unsigned long nx = x.dim(0);
    unsigned long ny = y.dim(0);
    if (nx != (unsigned long)d.dim(1) + 1 || ny != (unsigned long)d.dim(0) + 1) {
        throw std::runtime_error("data and axis bin boundary dimensions are incompatible");
    }

    if (bg.dim(0) != 4) {
        throw std::runtime_error("bg must be in RGBA format");
    }

    std::vector<int> irows(rows);
    std::vector<int> jcols(cols);

    // Calculate the pointer arrays to map input x to output x
    size_t i, j;
    const double *x0 = x.data();
    const double *y0 = y.data();
    double sx = cols / (x_right - x_left);
    double sy = rows / (y_top - y_bot);
    _bin_indices(&jcols[0], cols, x0, nx, sx, x_left);
    _bin_indices(&irows[0], rows, y0, ny, sy, y_bot);

    // Copy data to output buffer
    unsigned char *position = (unsigned char *)out.data();

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (irows[i] == -1 || jcols[j] == -1) {
                memcpy(position, (const unsigned char *)bg.data(), 4 * sizeof(unsigned char));
            } else {
                for (size_t k = 0; k < 4; ++k) {
                    position[k] = d(irows[i], jcols[j], k);
                }
            }
            position += 4;
        }
    }
}

#endif

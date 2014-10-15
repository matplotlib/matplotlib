/* -*- mode: c++; c-basic-offset: 4 -*- */

/* image.h
 *
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <vector>

#include "agg_trans_affine.h"
#include "agg_rendering_buffer.h"
#include "agg_color_rgba.h"

class Image
{
  public:
    Image();
    Image(unsigned numrows, unsigned numcols, bool isoutput);
    virtual ~Image();

    static void init_type(void);

    void apply_rotation(double r);
    void apply_scaling(double sx, double sy);
    void apply_translation(double tx, double ty);
    void as_rgba_str(agg::int8u *outbuf);
    void color_conv(int format, agg::int8u *outbuf);
    void reset_matrix();
    void clear();
    void resize(int numcols, int numrows, int norm, double radius);
    void blend_image(Image &im, unsigned ox, unsigned oy, bool apply_alpha, float alpha);
    void set_bg(double r, double g, double b, double a);

    enum {
        NEAREST,
        BILINEAR,
        BICUBIC,
        SPLINE16,
        SPLINE36,
        HANNING,
        HAMMING,
        HERMITE,
        KAISER,
        QUADRIC,
        CATROM,
        GAUSSIAN,
        BESSEL,
        MITCHELL,
        SINC,
        LANCZOS,
        BLACKMAN
    };

    // enum { BICUBIC=0, BILINEAR, BLACKMAN100, BLACKMAN256, BLACKMAN64,
    //   NEAREST, SINC144, SINC256, SINC64, SPLINE16, SPLINE36};
    enum {
        ASPECT_PRESERVE = 0,
        ASPECT_FREE
    };

    agg::int8u *bufferIn;
    agg::rendering_buffer *rbufIn;
    unsigned colsIn, rowsIn;

    agg::int8u *bufferOut;
    agg::rendering_buffer *rbufOut;
    unsigned colsOut, rowsOut;
    unsigned BPP;

    unsigned interpolation, aspect;
    agg::rgba bg;
    bool resample;
    agg::trans_affine srcMatrix, imageMatrix;

  private:
    // prevent copying
    Image(const Image &);
    Image &operator=(const Image &);
};

template <class ArrayType>
Image *from_grey_array(ArrayType &array, bool isoutput)
{
    Image *im = new Image((unsigned)array.dim(0), (unsigned)array.dim(1), isoutput);

    agg::int8u *buffer;
    if (isoutput) {
        buffer = im->bufferOut;
    } else {
        buffer = im->bufferIn;
    }

    agg::int8u gray;
    for (size_t rownum = 0; rownum < (size_t)array.dim(0); rownum++) {
        for (size_t colnum = 0; colnum < (size_t)array.dim(1); colnum++) {
            double val = array(rownum, colnum);

            gray = int(255 * val);
            *buffer++ = gray; // red
            *buffer++ = gray; // green
            *buffer++ = gray; // blue
            *buffer++ = 255;  // alpha
        }
    }

    return im;
}

template <class ArrayType>
Image *from_color_array(ArrayType &array, bool isoutput)
{
    Image *im = new Image((unsigned)array.dim(0), (unsigned)array.dim(1), isoutput);

    agg::int8u *buffer;
    if (isoutput) {
        buffer = im->bufferOut;
    } else {
        buffer = im->bufferIn;
    }

    int rgba = array.dim(2) >= 4;
    double r, g, b;
    double alpha = 1.0;

    for (size_t rownum = 0; rownum < (size_t)array.dim(0); rownum++) {
        for (size_t colnum = 0; colnum < (size_t)array.dim(1); colnum++) {
            typename ArrayType::sub_t::sub_t color = array[rownum][colnum];

            r = color(0);
            g = color(1);
            b = color(2);

            if (rgba) {
                alpha = color(3);
            }

            *buffer++ = int(255 * r);     // red
            *buffer++ = int(255 * g);     // green
            *buffer++ = int(255 * b);     // blue
            *buffer++ = int(255 * alpha); // alpha
        }
    }

    return im;
}

template <class ArrayType>
Image *frombyte(ArrayType &array, bool isoutput)
{
    Image *im = new Image((unsigned)array.dim(0), (unsigned)array.dim(1), isoutput);

    agg::int8u *buffer;
    if (isoutput) {
        buffer = im->bufferOut;
    } else {
        buffer = im->bufferIn;
    }

    int rgba = array.dim(2) >= 4;
    agg::int8u r, g, b;
    agg::int8u alpha = 255;

    for (size_t rownum = 0; rownum < (size_t)array.dim(0); rownum++) {
        for (size_t colnum = 0; colnum < (size_t)array.dim(1); colnum++) {
            typename ArrayType::sub_t::sub_t color = array[rownum][colnum];
            r = color(0);
            g = color(1);
            b = color(2);

            if (rgba) {
                alpha = color(3);
            }

            *buffer++ = r;     // red
            *buffer++ = g;     // green
            *buffer++ = b;     // blue
            *buffer++ = alpha; // alpha
        }
    }

    return im;
}

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

template <class CoordinateArray, class ColorArray>
Image *pcolor(CoordinateArray &x,
              CoordinateArray &y,
              ColorArray &d,
              unsigned int rows,
              unsigned int cols,
              float bounds[4],
              int interpolation)
{
    if (rows >= 32768 || cols >= 32768) {
        throw "rows and cols must both be less than 32768";
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
        throw "Cannot scale to zero size";
    }

    if (d.dim(2) != 4) {
        throw "data must be in RGBA format";
    }

    // Check dimensions match
    unsigned long nx = x.dim(0);
    unsigned long ny = y.dim(0);
    if (nx != (unsigned long)d.dim(1) || ny != (unsigned long)d.dim(0)) {
        throw "data and axis dimensions do not match";
    }

    // Allocate memory for pointer arrays
    std::vector<unsigned int> rowstarts(rows);
    std::vector<unsigned int> colstarts(cols);

    // Create output
    Image *imo = new Image(rows, cols, true);

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
    agg::int8u *position = imo->bufferOut;
    agg::int8u *oldposition = NULL;
    start = d.data();

    if (interpolation == Image::NEAREST) {
        _bin_indices_middle(colstart, cols, xs1, nx, dx, x_min);
        _bin_indices_middle(rowstart, rows, ys1, ny, dy, y_min);
        for (i = 0; i < rows; i++, rowstart++) {
            if (i > 0 && *rowstart == 0) {
                memcpy(position, oldposition, rowsize * sizeof(agg::int8u));
                oldposition = position;
                position += rowsize;
            } else {
                oldposition = position;
                start += *rowstart * inrowsize;
                inposition = start;
                for (j = 0, colstart = &colstarts[0]; j < cols; j++, position += 4, colstart++) {
                    inposition += *colstart * 4;
                    memcpy(position, inposition, 4 * sizeof(agg::int8u));
                }
            }
        }
    } else if (interpolation == Image::BILINEAR) {
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

                typename ColorArray::sub_t::sub_t start00 = d[rowstart[i]][colstart[j]];
                typename ColorArray::sub_t::sub_t start01 = d[rowstart[i]][colstart[j] + 1];
                typename ColorArray::sub_t::sub_t start10 = d[rowstart[i] + 1][colstart[j]];
                typename ColorArray::sub_t::sub_t start11 = d[rowstart[i] + 1][colstart[j] + 1];
                for (size_t k = 0; k < 4; ++k) {
                    position[k] =
                        start00(k) * a00 + start01(k) * a01 + start10(k) * a10 + start11(k) * a11;
                }
                position += 4;
            }
        }
    }

    return imo;
}

template <class CoordinateArray, class ColorArray, class Color>
Image *pcolor2(CoordinateArray &x,
               CoordinateArray &y,
               ColorArray &d,
               unsigned int rows,
               unsigned int cols,
               float bounds[4],
               Color &bg)
{
    double x_left = bounds[0];
    double x_right = bounds[1];
    double y_bot = bounds[2];
    double y_top = bounds[3];

    // Check we have something to output to
    if (rows == 0 || cols == 0) {
        throw "rows or cols is zero; there are no pixels";
    }

    if (d.dim(2) != 4) {
        throw "data must be in RGBA format";
    }

    // Check dimensions match
    unsigned long nx = x.dim(0);
    unsigned long ny = y.dim(0);
    if (nx != (unsigned long)d.dim(1) + 1 || ny != (unsigned long)d.dim(0) + 1) {
        throw "data and axis bin boundary dimensions are incompatible";
    }

    if (bg.dim(0) != 4) {
        throw "bg must be in RGBA format";
    }

    std::vector<int> irows(rows);
    std::vector<int> jcols(cols);

    // Create output
    Image *imo = new Image(rows, cols, true);

    // Calculate the pointer arrays to map input x to output x
    size_t i, j;
    const double *x0 = x.data();
    const double *y0 = y.data();
    double sx = cols / (x_right - x_left);
    double sy = rows / (y_top - y_bot);
    _bin_indices(&jcols[0], cols, x0, nx, sx, x_left);
    _bin_indices(&irows[0], rows, y0, ny, sy, y_bot);

    // Copy data to output buffer
    agg::int8u *position = imo->bufferOut;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (irows[i] == -1 || jcols[j] == -1) {
                memcpy(position, (const agg::int8u *)bg.data(), 4 * sizeof(agg::int8u));
            } else {
                for (size_t k = 0; k < 4; ++k) {
                    position[k] = d(irows[i], jcols[j], k);
                }
            }
            position += 4;
        }
    }

    return imo;
}

#endif

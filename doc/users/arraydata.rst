.. _arraydata:

***********************************************
Plotting 2-D arrays or functions of 2 variables
***********************************************

In this chapter we will address methods for plotting a
scalar function of two variables.  Here are some examples:

* A photographic image, represented as a 2-D array of
  colors; the grid is regular, with each element of the
  array corresponding to a square pixel.

* Earth surface elevation and ocean bottom topography,
  represented as a 2-D array of heights; the grid is
  rectangular in latitude and longitude, but the latitude
  increment may not be uniform--often it will decrease
  toward the poles.

* A mathematical function of two variables, such as a
  bivariate Gaussian probability density.

Note: in this chapter we will assume the data to be plotted
are on a grid.  If you have scalar data of two
variables, but the data are not on a  grid - for
example, sea level at island stations - then you will need
to use an interpolation or other gridding routine
before you can use any of the
plotting methods we will discuss here.

As a 2-D plotting library, matplotlib offers two basic
styles of plot for scalar functions of two variables: an
image style and a contour style.  The image style renders
the data as either a continuously-varying field of color or
a set of contiguous colored quadrilaterals.  Hence, the
image style is a direct representation of the data array.
The contour style is less direct; isolines of the data are
calculated and then either plotted as lines or used to delimit
colored regions.

.. _image_styles:

Image (or pcolor) styles
========================

some text

.. _image:

image
-----

image text and example

.. _pcolor:

pcolor
------

pcolor and pcolorfast, including quadmesh variant

.. _contour:

Contouring
==========

contour and contourf






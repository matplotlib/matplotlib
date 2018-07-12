Changes for 0.86
================

.. code-block:: text

     Matplotlib data is installed into the matplotlib module.
     This is similar to package_data.  This should get rid of
     having to check for many possibilities in _get_data_path().
     The MATPLOTLIBDATA env key is still checked first to allow
     for flexibility.

     1) Separated the color table data from cm.py out into
     a new file, _cm.py, to make it easier to find the actual
     code in cm.py and to add new colormaps. Everything
     from _cm.py is imported by cm.py, so the split should be
     transparent.
     2) Enabled automatic generation of a colormap from
     a list of colors in contour; see modified
     examples/contour_demo.py.
     3) Support for imshow of a masked array, with the
     ability to specify colors (or no color at all) for
     masked regions, and for regions that are above or
     below the normally mapped region.  See
     examples/image_masked.py.
     4) In support of the above, added two new classes,
     ListedColormap, and no_norm, to colors.py, and modified
     the Colormap class to include common functionality. Added
     a clip kwarg to the normalize class.

Customizing matplotlib
======================

matplotlib uses an configuration file ``matplotlibrc`` which is
located in ``matplotlib/mpl-data/matplotlibrc``.  Every time you
install matplotlib, this file will be overwritten, so if you want your
customizations to be saved, please move this file to your ``HOME/.matplotlib``
directory.

You can control the defaults of almost every property in matplotlib:
figure size and dpi, line width, color and style, axes, axis and grid
properties, text and font properties and so on.

You can also dynamically change the defaults in a python script or
interactively from the python shell using the :func:`matplotlib.rc`
command.  For example to change the default line properties, you could
do::

    import matplotlib
    matplotlib.rc('lines', linewidth=2, color='r')


A sample matplotlibrc file
--------------------------

.. literalinclude:: ../mpl_data/matplotlibrc
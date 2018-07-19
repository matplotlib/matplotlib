Changes for 0.65.1
==================

.. code-block:: text

  removed add_axes and add_subplot from backend_bases.  Use
  figure.add_axes and add_subplot instead.  The figure now manages the
  current axes with gca and sca for get and set current axes.  If you
  have code you are porting which called, e.g., figmanager.add_axes, you
  can now simply do figmanager.canvas.figure.add_axes.

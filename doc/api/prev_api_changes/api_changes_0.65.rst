Changes for 0.65
================

.. code-block:: text


  mpl_connect and mpl_disconnect in the MATLAB interface renamed to
  connect and disconnect

  Did away with the text methods for angle since they were ambiguous.
  fontangle could mean fontstyle (oblique, etc) or the rotation of the
  text.  Use style and rotation instead.

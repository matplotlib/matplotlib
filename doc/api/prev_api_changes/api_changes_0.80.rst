Changes for 0.80
================

.. code-block:: text

  - xlim/ylim/axis always return the new limits regardless of
    arguments.  They now take kwargs which allow you to selectively
    change the upper or lower limits while leaving unnamed limits
    unchanged.  See help(xlim) for example

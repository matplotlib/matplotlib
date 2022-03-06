Double-quoted strings in matplotlibrc
-------------------------------------

You can now use double-quotes around strings. This allows using the '#'
character in strings. Without quotes, '#' is interpreted as start of a comment.
In particular, you can now define hex-colors:

.. code-block:: none

   grid.color: "#b0b0b0"

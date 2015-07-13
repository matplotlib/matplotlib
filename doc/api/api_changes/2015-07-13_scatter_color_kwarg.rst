Deprecated the 'color' keyword argument to scatter
``````````````````````````````````````````````````

The ``scatter`` function takes a ``color`` keyword argument, but this argument
is not defined in the documentation, and there was a comment in the code
suggesting it should be deprecated. Thus, the argument has been deprecated and
removed from the examples; instead, users should use the ``c`` keyword argument
to set the color of the scatter points.

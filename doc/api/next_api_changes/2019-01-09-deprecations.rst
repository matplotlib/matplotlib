Deprecations
````````````

The ``matplotlib.sphinxext.mathmpl`` and
``matplotlib.sphinxext.plot_directive`` interfaces have changed from the
(Sphinx-)deprecated function-based interface to a class-based interface.  This
should not affect end users, but the
``matplotlib.sphinxext.mathmpl.math_directive`` and
``matplotlib.sphinxext.plot_directive.plot_directive`` functions are now
deprecated.

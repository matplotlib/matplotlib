New config option for ``matplotlib.sphinxext.plot_directive``: ``plot_skip_execution``
--------------------------------------------------------------------------------------

This configuration option allows users to temporarily skip the execution of all
plot directives, not running the code or generating the plots. It is intended to
be used during development to speed up building documentation that contains many
plot directives.

It can be temporarily enabled from the command line by passing ``-D
plot_skip_execution=1`` to ``sphinx-build``, e.g.,: ``make html O="-D
plot_skip_execution=1"``.

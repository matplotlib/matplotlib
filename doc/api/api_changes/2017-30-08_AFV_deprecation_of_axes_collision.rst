Deprecation of axes collision
-----------------------------

Adding an axes instance to a figure by using the same arguments as for
a previous axes instance currently reuses the earlier instance.  This
behavior has been deprecated in Matplotlib 2.1. In a future version, a
*new* instance will always be created and returned.  Meanwhile, in such
a situation, a deprecation warning is raised by
:class:`~matplotlib.figure.AxesStack`.

This warning can be suppressed, and the future behavior ensured, by passing
a *unique* label to each axes instance.  See the docstring of
:meth:`~matplotlib.figure.Figure.add_axes` for more information.

Additional details on the rationale behind this deprecation can be found
in :issue:`7377` and :issue:`9024`.

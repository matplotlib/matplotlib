Partial ``figsize`` specification at figure creation
----------------------------------------------------

Figure creation now accepts a single ``None`` in ``figsize``.
Passing ``(None, h)`` uses the default width from :rc:`figure.figsize`, and
passing ``(w, None)`` uses the default height.
Passing ``(None, None)`` is invalid and raises a `ValueError`.

For example::

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(None, 4))

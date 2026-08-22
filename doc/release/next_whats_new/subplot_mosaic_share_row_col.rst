``subplot_mosaic`` accepts the same axis-sharing spellings as ``subplots``
--------------------------------------------------------------------------
The *sharex* and *sharey* parameters of `.Figure.subplot_mosaic` and
`.pyplot.subplot_mosaic` now accept the strings ``'none'``, ``'all'``,
``'row'`` and ``'col'`` in addition to the booleans, matching
`.Figure.subplots`.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt

    fig, ax_dict = plt.subplot_mosaic("AB;CD", sharex="col", sharey="row")

Because an Axes of a mosaic may span several rows or columns, it belongs to
every row or column it covers and therefore joins the sharing groups of all of
them.  For instance in ``'AAE;C.E'`` the Axes *E* spans both rows, so
``sharex='row'`` shares the x-axis among *A*, *C* and *E* alike.  The Axes of a
nested mosaic are attributed to the cell of the outer mosaic holding them.

``subplot_mosaic`` no longer provisional
----------------------------------------

The API on `.Figure.subplot_mosaic` and `.pyplot.subplot_mosaic` are now
considered stable and will change under Matplotlib's normal deprecation
process.


Per-subplot keyword arguments  in ``subplot_mosaic``
----------------------------------------------------

It is now possible to pass keyword arguments through to Axes creation in each
specific call to ``add_subplot`` in `.Figure.subplot_mosaic` and
`.pyplot.subplot_mosaic` :

.. plot::
   :include-source: true

   fig, axd = plt.subplot_mosaic(
       "AB;CD",
       per_subplot_kw={
           "A": {"projection": "polar"},
           ("C", "D"): {"xscale": "log"},
           "B": {"projection": "3d"},
       },
   )


This is particularly useful for creating mosaics with mixed projections, but
any keyword arguments can be passed through.

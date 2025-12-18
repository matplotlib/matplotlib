+++++++
Artists
+++++++

Almost all objects you interact with on a Matplotlib plot are called "Artist"
(and are subclasses of the `.Artist` class).  :doc:`Figure <../figure/index>`
and :doc:`Axes <../axes/index>` are Artists, and generally contain
`~.axis.Axis` Artists and Artists that contain data or annotation information.

.. toctree::
    :maxdepth: 2

    artist_intro

.. toctree::
    :maxdepth: 1

    Automated color cycle <color_cycle>
    Optimizing Artists for performance <performance>
    Paths <paths>
    Path effects guide <patheffects_guide>
    Understanding the extent keyword argument of imshow <imshow_extent>
    transforms_tutorial

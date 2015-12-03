Merge JSAnimation
-----------------

Jake Vanderplas' JSAnimation package has been merged into matplotlib. This
adds to matplotlib the `~matplotlib.animation.HTMLWriter` class for
generating a javascript HTML animation, suitable for the IPython notebook.
This can be activated by default by setting the ``animation.html`` rc
parameter to ``jshtml``. One can also call the ``anim_to_jshtml`` function
to manually convert an animation. This can be displayed using IPython's
``HTML`` display class::

    from IPython.display import HTML
    HTML(animation.anim_to_jshtml(anim))

The `~matplotlib.animation.HTMLWriter` class can also be used to generate
an HTML file by asking for the ``html`` writer.

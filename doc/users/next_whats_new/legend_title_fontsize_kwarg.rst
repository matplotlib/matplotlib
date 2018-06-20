Legend now has a title_fontsize kwarg (and rcParam)
---------------------------------------------------

The title for a `.Figure.legend` and `.Axes.legend` can now have its
fontsize set via the ``title_fontsize`` kwarg.  There is also a new
:rc:`legend.title_fontsize`.  Both default to ``None``, which means
the legend title will have the same fontsize as the axes default fontsize
(*not* the legend fontsize, set by the ``fontsize`` kwarg or
:rc:`legend.fontsize`).

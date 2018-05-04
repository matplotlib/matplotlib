Legend now has a title_fontsize kwarg
-------------------------------------

The title for a `.Figure.legend` and `.Axes.legend` can now have its
fontsize set via the ``title_fontsize`` kwarg, defaults to ``None``, which
means the legend title will have the same fontsize as the axes default
fontsize (*not* the legend fontsize, set by the ``fontsize`` kwarg or
:rc:`legend.fontsize`).

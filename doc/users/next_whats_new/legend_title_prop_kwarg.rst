Legend now has a title_prop kwarg
-------------------------------------

The title for a `.Figure.legend` and `.Axes.legend` can now have its
title's font properties set via the ``title_prop`` kwarg, defaults to 
``None``, which means the legend's title will have the font properties
as set by the :rc:`legend.title_fontsize`. Through this new kwarg, one
can set the legend's title font properties through either FontProperties
or dict, for example:

.. plot::

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(range(10))
    ax.legend(title='Points', title_prop={'family': 'serif'}, title_fontsize=22) 

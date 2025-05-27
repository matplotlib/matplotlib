Colormaps support giving colors for bad, under and over values on creation
--------------------------------------------------------------------------

Colormaps gained keyword arguments ``bad``, ``under``, and ``over`` to
specify these values on creation. Previously, these values would have to
be set afterwards using one of `~.Colormap.set_bad`, `~.Colormap.set_under`,
`~.Colormap.set_bad`, `~.Colormap.set_extremes`, `~.Colormap.with_extremes`.

It is recommended to use the new functionality, e.g.::

    cmap = ListedColormap(colors, bad="red", under="darkblue", over="purple")

instead of::

    cmap = ListedColormap(colors).with_extremes(
        bad="red", under="darkblue", over="purple")

or::

   cmap = ListedColormap(colors)
   cmap.set_bad("red")
   cmap.set_under("darkblue")
   cmap.set_over("purple")

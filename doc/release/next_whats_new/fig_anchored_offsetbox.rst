``AnchoredOffsetbox`` and ``AnchoredText`` for ``Figure`` and ``SubFigure``
---------------------------------------------------------------------------

The `.AnchoredOffsetbox` and `.AnchoredText` artists may now be directly added to a
`.Figure` or `.SubFigure` and by default will be anchored relative to their parent
``(Sub)Figure``.


.. plot::
    :include-source: true
    :alt: The left half of the figure contains two text boxes with black outlines.  One in the top left of the figure reads "Anchored to Figure".  One at the bottom and to the left the vertical center line reads "Anchored to SubFigure".  A third box containing a blue circle is halfway up and to the right of the vertical center line.

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox, DrawingArea

    fig = plt.figure()

    sfig1, sfig2 = fig.subfigures(ncols=2)
    sfig1.set_facecolor("lemonchiffon")
    sfig2.set_facecolor("lightcyan")

    fig_text = AnchoredText("Anchored to Figure", loc="upper left")
    fig.add_artist(fig_text)

    sfig_text = AnchoredText("Anchored to SubFigure", loc="lower right")
    sfig1.add_artist(sfig_text)

    area = DrawingArea(width=30, height=30)
    area.add_artist(mpatches.Circle((15, 15), 15, fc="tab:blue"))
    box = AnchoredOffsetbox(child=area, loc="center left")
    sfig2.add_artist(box)

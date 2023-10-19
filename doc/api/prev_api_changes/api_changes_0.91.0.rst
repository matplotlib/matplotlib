
Changes for 0.91.0
==================

* Changed ``cbook.is_file_like`` to ``cbook.is_writable_file_like`` and
  corrected behavior.

* Added *ax* keyword argument to :func:`.pyplot.colorbar` and
  :meth:`.Figure.colorbar` so that one can specify the axes object from
  which space for the colorbar is to be taken, if one does not want to
  make the colorbar axes manually.

* Changed ``cbook.reversed`` so it yields a tuple rather than a (index, tuple).
  This agrees with the Python reversed builtin, and cbook only defines reversed
  if Python doesn't provide the builtin.

* Made skiprows=1 the default on ``csv2rec``

* The gd and paint backends have been deleted.

* The errorbar method and function now accept additional kwargs
  so that upper and lower limits can be indicated by capping the
  bar with a caret instead of a straight line segment.

* The :mod:`matplotlib.dviread` file now has a parser for files like
  psfonts.map and pdftex.map, to map TeX font names to external files.

* The file ``matplotlib.type1font`` contains a new class for Type 1
  fonts.  Currently it simply reads pfa and pfb format files and
  stores the data in a way that is suitable for embedding in pdf
  files. In the future the class might actually parse the font to
  allow e.g.,  subsetting.

* ``matplotlib.ft2font`` now supports ``FT_Attach_File``. In
  practice this can be used to read an afm file in addition to a
  pfa/pfb file, to get metrics and kerning information for a Type 1
  font.

* The ``AFM`` class now supports querying CapHeight and stem
  widths. The get_name_char method now has an isord kwarg like
  get_width_char.

* Changed :func:`.pcolor` default to ``shading='flat'``; but as noted now in
  the docstring, it is preferable to simply use the *edgecolor* keyword
  argument.

* The mathtext font commands (``\cal``, ``\rm``, ``\it``, ``\tt``) now
  behave as TeX does: they are in effect until the next font change
  command or the end of the grouping.  Therefore uses of ``$\cal{R}$``
  should be changed to ``${\cal R}$``.  Alternatively, you may use the
  new LaTeX-style font commands (``\mathcal``, ``\mathrm``,
  ``\mathit``, ``\mathtt``) which do affect the following group,
  e.g., ``$\mathcal{R}$``.

* Text creation commands have a new default linespacing and a new
  ``linespacing`` kwarg, which is a multiple of the maximum vertical
  extent of a line of ordinary text.  The default is 1.2;
  ``linespacing=2`` would be like ordinary double spacing, for example.

* Changed default kwarg in `matplotlib.colors.Normalize` to ``clip=False``;
  clipping silently defeats the purpose of the special over, under,
  and bad values in the colormap, thereby leading to unexpected
  behavior.  The new default should reduce such surprises.

* Made the emit property of :meth:`~matplotlib.axes.Axes.set_xlim` and
  :meth:`~matplotlib.axes.Axes.set_ylim` ``True`` by default; removed
  the Axes custom callback handling into a 'callbacks' attribute which
  is a :class:`~matplotlib.cbook.CallbackRegistry` instance.  This now
  supports the 'xlim_changed' and 'ylim_changed' Axes events.

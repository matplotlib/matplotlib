:orphan:

Introduce new Tableau colors
----------------------------

In its version 10, Tableau `introduced a new palette of categorical colors
<https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782>`__.
Those are now available in matplotlib with the prefix ``tabx:``:
``{'tabx:blue', 'tabx:orange', 'tabx:red', 'tabx:cyan', 'tabx:green',
'tabx:yellow', 'tabx:purple', 'tabx:pink', 'tabx:brown', 'tabx:grey'}``

Those colors are also provided as a new ``tabx10`` colormap. An additional
``tabx20`` colormap with is added.

In general those colors are a little less saturated than those from the default
color cycle. Replacing the default color cycler with those colors can e.g. be
achieved via

::

    cols = plt.cm.tabx10.colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cols)}



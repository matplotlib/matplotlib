Qualitative colormaps
---------------------

Colorbrewer's qualitative/discrete colormaps ("Accent", "Dark2", "Paired",
"Pastel1", "Pastel2", "Set1", "Set2", "Set3") are now implemented as
``ListedColormap`` instead of ``LinearSegmentedColormap``.

To use these for images where categories are specified as integers, for
instance, use::

    plt.imshow(x, cmap='Dark2', norm=colors.NoNorm())

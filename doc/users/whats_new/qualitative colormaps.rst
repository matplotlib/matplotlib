Qualitative colormaps
---------------------

ColorBrewer's "qualitative" colormaps ("Accent", "Dark2", "Paired",
"Pastel1", "Pastel2", "Set1", "Set2", "Set3") were intended for discrete
categorical data, with no implication of value, and therefore have been
converted to ``ListedColormap`` instead of ``LinearSegmentedColormap``, so
the colors will no longer be interpolated and they can be used for
choropleths, labeled image features, etc.

Shared-axes ``subplots`` tick label visibility is now correct for top or left labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When calling ``subplots(..., sharex=True, sharey=True)``, Matplotlib
automatically hides x tick labels for axes not in the first column and y tick
labels for axes not in the last row.  This behavior is incorrect if rcParams
specify that axes should be labeled on the top (``rcParams["xtick.labeltop"] =
True``) or on the right (``rcParams["ytick.labelright"] = True``).

Such cases are now handled correctly (adjusting visibility as needed on the
first row and last column of axes).

Added "legend.framealpha" key to rcParams
`````````````````````````````````````````

Added a key and the corresponding logic to control the default transparency of
legend frames. This feature was written into the docstring of axes.legend(),
but not yet implemented.


Added "figure.titlesize" and "figure.titleweight" keys to rcParams
``````````````````````````````````````````````````````````````````

Two new keys were added to rcParams to control the default font size and weight
used by the figure title (as emitted by ``pyplot.suptitle()``).

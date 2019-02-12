:orphan:

Legend for scatter
------------------

A new method for creating legends for scatter plots has been introduced.
Previously, in order to obtain a legend for a :meth:`~.axes.Axes.scatter`
plot, one could either plot several scatters, each with an individual label,
or create proxy artists to show in the legend manually.
Now, :class:`~.collections.PathCollection` provides a method
:meth:`~.collections.PathCollection.legend_elements` to obtain the handles and labels
for a scatter plot in an automated way. This makes creating a legend for a
scatter plot as easy as::

    scatter = plt.scatter([1,2,3], [4,5,6], c=[7,2,3])
    plt.legend(*scatter.legend_elements())

An example can be found in
:ref:`automatedlegendcreation`.

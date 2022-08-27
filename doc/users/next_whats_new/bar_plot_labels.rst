Easier labelling of bars in bar plot
------------------------------------

The ``label`` argument of `~matplotlib.axes.Axes.bar` can now
be passed a list of labels for the bars.

.. code-block:: python

    import matplotlib.pyplot as plt

    x = ["a", "b", "c"]
    y = [10, 20, 15]

    fig, ax = plt.subplots()
    bar_container = ax.barh(x, y, label=x)
    [bar.get_label() for bar in bar_container]

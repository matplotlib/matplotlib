Order of legend entries in stackplots
-------------------------------------

The order of entries in a legend of a `~matplotlib.axes.Axes.stackplot` can be inverted with the option ``top_to_bottom``. In the below example, a viewer would see the area for ``y3`` above ``y2``, and ``y2`` above ``y1``. Here the legend will be generated in the same way such that ``y3`` appears above ``y2`` and ``y2`` above ``y1``.

.. code-block:: python

   from matplotlib import pyplot as plt
   plt.x = [1, 2, 3, 4, 5]
   y1 = [1, 1, 2, 3, 5]
   y2 = [0, 4, 2, 6, 8]
   y3 = [1, 3, 5, 7, 9]

   y4 = [23, 23, 23, 24, 25]

   y = np.vstack([y1, y2, y3])

   labels = ["Fibonacci ", "Evens", "Odds"]

   fig, ax = plt.subplots()
   ax.plot(x, y4, label="staying above")
   ax.stackplot(x, y1, y2, y3, labels=labels, top_to_bottom=True)
   ax.legend(loc='upper left')
   plt.show()

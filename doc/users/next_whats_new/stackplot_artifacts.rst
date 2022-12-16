Suppress 0 thickness layers in stackplots
-----------------------------------------

When making `.Axes.stackplot` where one of the layers is 0 it may or may not be
desirable for that layer to be visible.  Using the new *where* keyword argument
along with the *interpolate* keyword argument these lines can be suppressed.


.. plot::
   :include-source: true

   import matplotlib.pyplot as plt
   import numpy as np

   fig, (axl, axr) = plt.subplots(1, 2)

   x = np.arange(10)

   colors = ["grey", "blue", "red", "blue"]

   y = np.stack(
       [
           np.linspace(0, 1, 10),
           np.linspace(0, 1, 10),
           [0, 0, 0.25, 0, 0, 0, 0.25, 0, 0, 0],
           np.linspace(0, 1, 10),
       ]
   )

   axl.set_title("No line")
   axl.stackplot(
       x, y,
       colors=colors, edgecolor="face",
       where=(y != 0), interpolate=True,
   )


   axr.set_title("With line")
   axr.stackplot(
       x, y,
       colors=colors, edgecolor="face",
   )

Text can be positioned inside TextBox widget
--------------------------------------------

A new parameter called *textalignment* can be used to control for the position of the text inside the Axes of the TextBox widget.

.. plot::

  from matplotlib import pyplot as plt
  from matplotlib.widgets import TextBox

  box_input = plt.axes([0.2, 0.2, 0.1, 0.075])
  text_box = TextBox(ax=box_input, initial="text", label="", textalignment="center")


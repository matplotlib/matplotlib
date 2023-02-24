"""
==================================
Move x-axis tick labels to the top
==================================

`~.axes.Axes.tick_params` can be used to configure the ticks. *top* and
*labeltop* control the visibility tick lines and labels at the top x-axis.
To move x-axis ticks from bottom to top, we have to activate the top ticks
and deactivate the bottom ticks::

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

.. note::

    If the change should be made for all future plots and not only the current
    Axes, you can adapt the respective config parameters

    - :rc:`xtick.top`
    - :rc:`xtick.labeltop`
    - :rc:`xtick.bottom`
    - :rc:`xtick.labelbottom`

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(10))
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.set_title('x-ticks moved to the top')

plt.show()

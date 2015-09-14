"""

Basic demo showing how to set tick labels to values of a series.

Using ax.set_xticks causes the tick labels to be set on the currently
chosen ticks. However, you may want to allow matplotlib to dynamically 
choose the number of ticks and their spacing.

In this case may be better to determine the tick label from the value
at the tick. The following example shows how to do this.

NB: You may want to combine the solution below with 
`xaxis.set_major_locator(MaxNLocator(integer=True))` to ensure
that tick values are always at integer values, and therefore always use
the appropriate label.

"""



import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
xs = range(26)
ys = range(26)
labels = list('abcdefghijklmnopqrstuvwxyz')

def format_fn(tick_val, tick_pos): 
    return labels[int(tick_val)]

ax.xaxis.set_major_formatter(FuncFormatter(format_fn))

ax.plot(xs, ys)
plt.show()

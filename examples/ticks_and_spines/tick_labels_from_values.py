"""

Basic demo showing how to set tick labels to values of a series.

Using ax.set_xticks causes the tick labels to be set on the currently
chosen ticks. However, you may want to allow matplotlib to dynamically 
choose the number of ticks and their spacing.

In this case may be better to determine the tick label from the value
at the tick. The following example shows how to do this.

NB: The MaxNLocator is used here to ensure that the tick
values take integer values. As such, we need to catch
any IndexErrors in the format function where we have not
defined a label for that particular tick

"""



import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


fig = plt.figure()
ax = fig.add_subplot(111)
xs = range(26)
ys = range(26)
labels = list('abcdefghijklmnopqrstuvwxyz')

def format_fn(tick_val, tick_pos):
    try:
        return labels[int(tick_val)]
    except IndexError:
        # no label for this tick
        return ''

ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(xs, ys)
plt.show()

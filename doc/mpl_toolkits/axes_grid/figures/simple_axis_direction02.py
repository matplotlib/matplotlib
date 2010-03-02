
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid.axislines as axislines

def setup_axes(fig, rect):
    ax = axislines.Subplot(fig, rect)
    fig.add_subplot(ax)

    ax.set_yticks([0.2, 0.8])
    ax.set_xticks([0.2, 0.8])

    return ax
    
fig = plt.figure(1, figsize=(4, 2))
ax = setup_axes(fig, "111")

ax.axis[:].major_ticks.set_tick_out(True)

plt.show()



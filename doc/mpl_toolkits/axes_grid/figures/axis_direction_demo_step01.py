import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def setup_axes(fig, rect):
    ax = axisartist.Subplot(fig, rect)
    fig.add_axes(ax)

    ax.set_ylim(-0.1, 1.5)
    ax.set_yticks([0, 1])

    ax.axis[:].set_visible(False)

    ax.axis["x"] = ax.new_floating_axis(1, 0.5)
    ax.axis["x"].set_axisline_style("->", size=1.5)

    return ax

fig = plt.figure(figsize=(3,2.5))
fig.subplots_adjust(top=0.8)
ax1 = setup_axes(fig, "111")

ax1.axis["x"].set_axis_direction("left")


plt.show()

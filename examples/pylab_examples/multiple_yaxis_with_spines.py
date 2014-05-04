import matplotlib.pyplot as plt


fig, host = plt.subplots()

# Leave room for the additional spine:
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# By default, par2 would now be the last Axes to be drawn,
# and the one to receive mouse and keyboard events.  If we
# want the host to receive events, we can swap its position
# with par2 in the Axes stack with a Figure method:
fig.swap_axes_order(par2, host)

# Or let par1 receive the events:
#fig.swap_axes_order(par2, par1)   # par1 is active

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))

# Show only the right spine.
for sp in par2.spines.itervalues():
    sp.set_visible(False)
par2.spines["right"].set_visible(True)

p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

host.set_xlim(0, 2)
host.set_ylim(0, 2)
par1.set_ylim(0, 4)
par2.set_ylim(1, 65)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines], loc='upper left')

plt.show()

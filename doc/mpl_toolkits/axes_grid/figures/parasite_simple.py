from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.pyplot as plt

fig = plt.figure(1)

host = SubplotHost(fig, 111)
fig.add_subplot(host)

par = host.twinx()

host.set_xlabel("Distance")
host.set_ylabel("Density")
par.set_ylabel("Temperature")

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
p2, = par.plot([0, 1, 2], [0, 3, 2], label="Temperature")

host.axis["left"].label.set_color(p1.get_color())
par.axis["right"].label.set_color(p2.get_color())

host.legend()

plt.show()


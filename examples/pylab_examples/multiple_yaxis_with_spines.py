import matplotlib.pyplot as plt

def make_patch_spines_invisible(ax):
    par2.set_frame_on(True)
    par2.patch.set_visible(False)
    for sp in par2.spines.itervalues():
        sp.set_visible(False)

def make_spine_invisible(ax, direction):
    if direction in ["right", "left"]:
        ax.yaxis.set_ticks_position(direction)
        ax.yaxis.set_label_position(direction)
    elif direction in ["top", "bottom"]:
        ax.xaxis.set_ticks_position(direction)
        ax.xaxis.set_label_position(direction)
    else:
        raise ValueError("Unknown Direction : %s" % (direction,))

    ax.spines[direction].set_visible(True)


if 1:
    fig = plt.figure(1)

    host = fig.add_subplot(111)

    host.set_xlabel("Distance")

    par1 = host.twinx()
    par2 = host.twinx()

    par2.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(par2)
    make_spine_invisible(par2, "right")

    plt.subplots_adjust(right=0.75)


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
    host.legend(lines, [l.get_label() for l in lines])
    plt.show()


import time
import matplotlib.pyplot as plt
import numpy as np


def get_memory():
    "Simulate a function that returns system memory"
    return 100*(0.5 + 0.5*np.sin(0.5*np.pi*time.time()))


def get_cpu():
    "Simulate a function that returns cpu usage"
    return 100*(0.5 + 0.5*np.sin(0.2*np.pi*(time.time() - 0.25)))


def get_net():
    "Simulate a function that returns network bandwidth"
    return 100*(0.5 + 0.5*np.sin(0.7*np.pi*(time.time() - 0.1)))


def get_stats():
    return get_memory(), get_cpu(), get_net()


# turn interactive mode on for dynamic updates.  If you aren't in
# interactive mode, you'll need to use a GUI event handler/timer.
# plt.ion()

fig, ax = plt.subplots()
ind = np.arange(1, 4)
pm, pc, pn = plt.bar(ind, get_stats())
centers = ind + 0.5*pm.get_width()
pm.set_facecolor('r')
pc.set_facecolor('g')
pn.set_facecolor('b')
ax.set_xlim([0.5, 4])
ax.set_xticks(centers)
ax.set_ylim([0, 100])
ax.set_xticklabels(['Memory', 'CPU', 'Bandwidth'])
ax.set_ylabel('Percent usage')
ax.set_title('System Monitor')

for i in range(200):  # run for a little while
    m, c, n = get_stats()

    pm.set_height(m)
    pc.set_height(c)
    pn.set_height(n)
    ax.set_ylim([0, 100])

    plt.draw()

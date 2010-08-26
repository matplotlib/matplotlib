import numpy as np
import matplotlib.pyplot as plt
from animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(np.random.rand(10))
ax.set_ylim(0, 1)

def update(data):
    line.set_ydata(data)
    return line,

def data_gen():
    while True: yield np.random.rand(10)

ani = FuncAnimation(fig, update, data_gen, interval=100)
plt.show()

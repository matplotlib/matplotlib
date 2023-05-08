import matplotlib
import matplotlib.pyplot as plt
from numpy.random import rand
matplotlib.use('tkagg')

fig, ax = plt.subplots()


def user_defined_function(event):
    x, y = round(event.xdata * 10, 1), round(event.ydata + 3, 3)
    return f'({x}, {y})'


ax.plot(rand(100), 'o', hover=user_defined_function)
plt.show()


# Alternative test for testing out string literals as tooltips:

fig, ax = plt.subplots()

ax.plot(rand(3), 'o', hover=['London', 'Paris', 'Barcelona'])
plt.show()

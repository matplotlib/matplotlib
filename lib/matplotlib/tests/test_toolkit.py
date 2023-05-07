import matplotlib
import matplotlib.pyplot as plt
from numpy.random import rand
matplotlib.use('tkagg')


# Run this if it seems like your chanegs aren't being applied. If this does not
# print something along the lines of:
#   3.8.0.dev898+g0a062ed8bf.d20230506 /Users/eslothower/Desktop/matplotlib/lib/
#   matplotlib/__init__.py
# then this means you did not set up matplotlib for development:
#   https://matplotlib.org/stable/devel/development_setup.html

# print(matplotlib.__version__, matplotlib.__file__)

# fig, ax = plt.subplots()
# plt.ylabel('some numbers')


# def user_defined_function(event):
#     return round(event.xdata * 10, 1), round(event.ydata + 3, 3)

# ax.plot(rand(100), 'o', hover=user_defined_function)
# plt.show()

# Alternative test for testing out string literals as tooltips:

fig, ax = plt.subplots()
plt.ylabel('some numbers')

ax.plot(rand(3), 'o', hover=['London','Paris','Barcelona'])
plt.show()

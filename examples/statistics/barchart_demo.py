"""
Demo of multiple bar plots.
"""
import matplotlib.pyplot as plt

values = (20, 35, 30, 35, 27)
pos = (0, 1, 2, 3, 4)

plt.bar(left=pos, height=values, width=0.5, color='r')

plt.title('Example of bar plots')

plt.show()

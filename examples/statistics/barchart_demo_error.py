"""
Demo of multiple bar plot with errorbars
"""
import matplotlib.pyplot as plt

values = (20, 35, 30, 35, 27)
errors = (2, 3, 4, 1, 2)
pos = (0, 1, 2, 3, 4)

plt.bar(left=pos, height=values, width=0.5, color='r', yerr=errors)

plt.title("Bar plot with errors")

plt.show()

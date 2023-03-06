import matplotlib.pyplot as plt

"""""
Should create the same plots, running the saved figures in an
external png difference checker gives the same result after the
new implementation.
"""

fig, ax = plt.subplots()
pc1 = ax.scatter(1, 1, facecolors='none', edgecolors=(1.0, 0.0, 0.0))
pc2 = ax.scatter(1, 1, c=[(1.0, 0.0, 0.0)], facecolors='none')

print("Figure1")
print(pc1.get_facecolors())
print(pc1.get_edgecolors())
print("Figure2")
print(pc2.get_facecolors())
print(pc2.get_edgecolors())

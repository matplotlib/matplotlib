"""
==============================
Plotting categorical variables
==============================

How to use categorical variables in matplotlib.

Many times you want to create a plot that uses categorical variables
in Matplotlib. For example, your data may naturally fall into
several "bins" and you're interested in summarizing the data per
bin. Matplotlib allows you to pass categorical variables directly to
many plotting functions.
"""
import matplotlib.pyplot as plt

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.suptitle('Categorical Plotting')

plt.show()

#!/usr/bin/python
#
# Show how to add a translucent legend

# import pyplot module
import matplotlib.pyplot as plt

# draw 2 crossing lines
plt.plot([0,1], label='going up')
plt.plot([1,0], label='going down')

# add the legend in the middle of the plot
leg = plt.legend(fancybox=True, loc='center')
# set the alpha value of the legend: it will be translucent
leg.get_frame().set_alpha(0.5)

# show the plot
plt.show()

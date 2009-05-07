from matplotlib.pyplot import *

p1, = plot([1,2,3], label="test1")
p2, = plot([3,2,1], label="test2")

l1 = legend([p1], ["Label 1"], loc=1)
l2 = legend([p2], ["Label 2"], loc=4) # this removes l1 from the axes.
gca().add_artist(l1) # add l1 as a separate artist to the axes

show()

from matplotlib.matlab import *

x = 0.9*rand(30)
y = 0.9*rand(30)
s = 0.1*rand(30)
c = rand(30)
scatter(x,y,s,'b')
#axis([0, 1, 0, 1])
show()

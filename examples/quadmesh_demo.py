from matplotlib.mlab import linspace, meshgrid
import matplotlib.numerix as nx
from pylab import figure,show

n = 56
x = linspace(-1.5,1.5,n)
X,Y = meshgrid(x,x);
Qx = nx.cos(Y) - nx.cos(X)
Qz = nx.sin(Y) + nx.sin(X)
Qx = (Qx + 1.1)
Z = nx.sqrt(X**2 + Y**2)/5;
Z = (Z - nx.mlab.amin(Z)) / (nx.mlab.amax(Z) - nx.mlab.amin(Z))

fig = figure()
ax = fig.add_subplot(111)
ax.pcolormesh(Qx,Qz,Z)
show()

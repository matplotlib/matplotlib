from pylab import figure, show, nx, linspace, stineman_interp
x = linspace(0,2*nx.pi,20);
y = nx.sin(x); yp = None 
xi = linspace(x[0],x[-1],100);
yi = stineman_interp(xi,x,y,yp);

fig = figure()
ax = fig.add_subplot(111)
ax.plot(x,y,'ro',xi,yi,'-b.')
show()


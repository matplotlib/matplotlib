from pylab import figure, show, nx


x,y = nx.mlab.randn(2,100)
fig = figure()
ax1 = fig.add_subplot(211)
ax1.xcorr(x, y)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)

ax2 = fig.add_subplot(212)
ax2.acorr(x, normed=True)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)

show()

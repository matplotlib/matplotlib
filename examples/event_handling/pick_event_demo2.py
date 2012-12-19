"""
compute the mean and standard deviation (stddev) of 100 data sets and plot
mean vs stddev.  When you click on one of the mu, sigma points, plot the raw
data from the dataset that generated the mean and stddev.
"""
import numpy
import matplotlib.pyplot as plt


X = numpy.random.rand(100, 1000)
xs = numpy.mean(X, axis=1)
ys = numpy.std(X, axis=1)

fig, ax = plt.subplots()
ax.set_title('click on point to plot time series')
line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance


def onpick(event):

    if event.artist!=line: return True

    N = len(event.ind)
    if not N: return True


    figi = plt.figure()
    for subplotnum, dataind in enumerate(event.ind):
        ax = figi.add_subplot(N,1,subplotnum+1)
        ax.plot(X[dataind])
        ax.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f'%(xs[dataind], ys[dataind]),
                transform=ax.transAxes, va='top')
        ax.set_ylim(-0.5, 1.5)
    figi.show()
    return True

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()






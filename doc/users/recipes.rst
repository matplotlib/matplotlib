.. _recipes:

********************
Our Favorite Recipes
********************

Here is a collection of short tutorials, examples and code snippets
that illustrate some of the useful idioms and tricks to make snazzier
figures and overcome some matplotlib warts.

Fixing common date annoyances
=============================

matplotlib allows you to natively plots python datetime instances, and
for the most part does a good job picking tick locations and string
formats.  There are a couple of things it does not handle so
gracefully, and here are some tricks to help you work around them.
We'll load up some sample date data which contains datetime.date
objects in a numpy record array::

  In [63]: datafile = cbook.get_sample_data('goog.npy')

  In [64]: r = np.load(datafile).view(np.recarray)

  In [65]: r.dtype
  Out[65]: dtype([('date', '|O4'), ('', '|V4'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), ('close', '<f8'), ('volume', '<i8'), ('adj_close', '<f8')])

  In [66]: r.date
  Out[66]:
  array([2004-08-19, 2004-08-20, 2004-08-23, ..., 2008-10-10, 2008-10-13,
	 2008-10-14], dtype=object)

The dtype of the numpy record array for the field 'date' is '|O4'
which means it is a 4-byte python object pointer; in this case the
objects are datetime.date instances, which we can see when we print
some samples in the ipython terminal window.

If you plot the data, you will see that the x tick labels are all
squashed together::

  In [67]: plot(r.date, r.close)
  Out[67]: [<matplotlib.lines.Line2D object at 0x92a6b6c>]

.. plot::

   import matplotlib.cbook as cbook
   datafile = cbook.get_sample_data('goog.npy')
   r = np.load(datafile).view(np.recarray)
   plt.figure()
   plt.plot(r.date, r.close)
   plt.show()

Another annoyance is that if you hover the mouse over a the window and
look in the lower right corner of the matplotlib toolbar at the x and
y coordinates, you see that the x locations are formatted the same way
the tick labels are, eg "Dec 2004".  What we'd like is for the
location in the toolbar to have a higher degree of precision, eg
giving us the exact date out mouse is hovering over.  To fix the first
problem, we can use method:`matplotlib.figure.Figure.autofmt_xdate()`
and to fix the second problem we can use the ``ax.fmt_xdata``
attribute which can be set to any function that takes a position and
returns a string.  matplotlib has a number of date formatters built
im, so we'll use one of those.

.. plot::


   import matplotlib.cbook as cbook
   datafile = cbook.get_sample_data('goog.npy')
   r = np.load(datafile).view(np.recarray)
   fig, ax = plt.subplots(1)
   ax.plot(r.date, r.close)

   # rotate and align the tick labels so they look better
   fig.autofmt_xdate()

   # use a more precise date string for the x axis locations in the
   # toolbar
   import matplotlib.dates as mdates
   ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')



Fill Between and Alpha
======================

The :method:`~matplotlib.axes.Axes.fill_between` function generates a
shaded region between a min and max boundary that is useful for
illustrating ranges.  It has a very handy ``where`` argument to
combine filling with logical ranges, eg to just fill in a curve over
some threshold value.  

At it's most basic level, ``fill_between`` can be use to enhance a
graphs visual appearance. Let's compare two graphs of a financial
times with a simple line plot on the left and a filled line on the
right.

.. plot::
   :include-source:

   import matplotlib.cbook as cbook

   # load up some sample financial data
   datafile = cbook.get_sample_data('goog.npy')
   r = np.load(datafile).view(np.recarray)

   # create two subplots with the shared x and y axes
   fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

   pricemin = r.close.min()

   ax1.plot(r.date, r.close, lw=2)
   ax2.fill_between(r.date, pricemin, r.close, facecolor='blue', alpha=0.5)

   for ax in ax1, ax2:
       ax.grid(True)

   ax1.set_ylabel('price')
   fig.suptitle('Google (GOOG) daily closing price')
   fig.autofmt_xdate()
   plt.show()

The alpha channel is not necessary here, but it can be used to soften
colors for more visually appealing plots.  In other examples, as we'll
see below, the alpha channel is functionally useful as the shaded
regions can overlap and alpha allows you to see both.  Note that the
postscript format does not support alpha (this is a postscript
limitation, not a matplotlib limitation), so when using alpha save
your figures in PNG, PDF or SVG.

Our next example computes two populations of random walkers with a
different mean and standard deviation of the normal distributions from
which there steps are drawn.  We use shared regions to plot +/- one
standard deviation of the mean position of the population.  Here the
alpha channel is useful, not just aesthetic.

.. plot::
   :include-source:

   Nsteps, Nwalkers = 100, 250
   t = np.arange(Nsteps)

   # an Nsteps x Nwalkers array of random walk steps
   S1 = 0.002 + 0.01*np.random.randn(Nsteps, Nwalkers)
   S2 = 0.004 + 0.02*np.random.randn(Nsteps, Nwalkers)

   # an Nsteps x Nwalkers array of random walker positions
   X1 = S1.cumsum(axis=0)
   X2 = S2.cumsum(axis=0)


   # Nsteps length arrays empirical means and standard deviations of both
   # populations over time
   mu1 = X1.mean(axis=1)
   sigma1 = X1.std(axis=1)
   mu2 = X2.mean(axis=1)
   sigma2 = X2.std(axis=1)

   # plot it!
   fig, ax = plt.subplots(1)
   ax.plot(t, mu1, lw=2, label='mean population 1', color='blue')
   ax.plot(t, mu1, lw=2, label='mean population 2', color='yellow')
   ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)
   ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
   ax.set_title('random walkers empirical $\mu$ and $\pm \sigma$ interval')
   ax.legend(loc='upper left')
   ax.set_xlabel('num steps')
   ax.set_ylabel('position')
   ax.grid()
   plt.show()



The where keyword argument is very handy for highlighting certain
regions of the graph.  Where takes a boolean mask the same length as
the x, ymin and ymax arguments, and only fills in the region where the
boolean mask is True.  In the example below, we take a a single random
walker and compute the analytic mean and standard deviation of the
population positions.  The population mean is shown as the black
dashed line, and the plus/minus one sigma deviation from the mean is
showsn as the yellow filled region.  We use the where mask
``X>upper_bound`` to find the region where the walker is above the
one sigma boundary, and shade that region blue.

.. plot::
   :include-source:

   np.random.seed(1234)

   Nsteps = 500
   t = np.arange(Nsteps)

   mu = 0.002
   sigma = 0.01

   # the steps and position
   S = mu + sigma*np.random.randn(Nsteps)
   X = S.cumsum()

   # the 1 sigma upper and lower population bounds
   lower_bound = mu*t - sigma*np.sqrt(t)
   upper_bound = mu*t + sigma*np.sqrt(t)

   fig, ax = plt.subplots(1)
   ax.plot(t, X, lw=2, label='walker position', color='blue')
   ax.plot(t, mu*t, lw=1, label='population mean', color='black', ls='--')
   ax.fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5,
		   label='1 sigma range')
   ax.legend(loc='upper left')

   # here we use the where argument to only fill the region where the
   # walker is above the population 1 sigma boundary
   ax.fill_between(t, upper_bound, X, where=X>upper_bound, facecolor='blue', alpha=0.5)
   ax.set_xlabel('num steps')
   ax.set_ylabel('position')
   ax.grid()
   plt.show()


Another handy use of filled regions is to highlight horizontal or
vertical spans of an axes -- for that matplotlib has some helper
functions :method:`~matplotlib.axes.Axes.axhspan` and
:method:`~matplotlib.axes.Axes.axvspan` and example
:ref:`pylab_examples-axhspan_demo`.


.. _best-practices:

**************
Best Practices
**************

.. contents::
   :backlinks: none


.. _best-practice-many-figures:

How to save many figures
========================

You have figured out how to create a plot from your data and either
show or save the figure, possibly following the tutorial :doc:`/tutorials/introductory/lifecycle`. It looks great and wonderful.::

  data = data_loading_function(fname)
  fig, ax = plt.subplots()
  ax.plot(data.x, data.y, label=data.name)
  ax.legend()
  ax.set(xlabel='Time [months]', ylabel='Profits [$]', title='Sales Forecast')
  fig.savefig('sales_forecast.png')

Now you want to make this plot for each of your data files. One might just
put the above code into a for-loop over all of your filenames. However, you
will eventually encounter a warning ``More than X figures hae been opened.``.
With enough looping, Python can eventually run out of memory, and calling
garbage collection will not help. This is because figures created through
``pyplot`` will also be stored in within the module.

So, the best practice for saving many figures is to recycle the figure.::

  fig = plt.figure()
  for fname in data_file_names:
    data = data_loading_function(fname)
    ax = fig.subplots()
    ax.plot(data.x, data.y, label=data.name)
    ax.legend()
    ax.set(xlabel='Time [months]', ylabel='Profits [$]', title='Sales Forecast')
    fig.savefig('sales_forecast_%s.png' % data.name)
    fig.clear()

Now, only one figure object is ever made, and it is cleaned of any plots
at each iteration, so the memory usage will not grow.


.. _best-practice-plotting-functions:

Creating your own plotting functions
====================================

For nontrivial plots, it may make sense to create your own plotting
functions for easy reuse. Here are some guidelines:

* Have an ``ax`` positional argument as the first argument.
* Avoid creating too many plotting elements in a single function.
* Avoid putting data processing together with plotting.
  Instead, have the plotting function take the processed data
  created by another function. It is OK to put these two functions
  within a convenience function.
* Plotting functions should return all of the artists that it creates.
  Doing so enables customizations by users.

Example::

  def my_plotter(ax, data1, data2, param_dict=None):
      """
      A helper function to make a graph

      Parameters
      ----------
      ax : Axes
          The axes to draw to

      data1 : array
         The x data

      data2 : array
         The y data

      param_dict : dict
         Dictionary of kwargs to pass to ax.plot

      Returns
      -------
      out : list
          list of artists added
      """
      if param_dict is None:
          param_dict = {}
      out = ax.plot(data1, data2, **param_dict)
      return out


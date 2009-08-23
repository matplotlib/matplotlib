.. _mpl-shell:

**********************************
Using matplotlib in a python shell
**********************************

By default, matplotlib defers drawing until the end of the script
because drawing can be an expensive operation, and you may not want
to update the plot every time a single property is changed, only once
after all the properties have changed.

But when working from the python shell, you usually do want to update
the plot with every command, eg, after changing the
:func:`~matplotlib.pyplot.xlabel`, or the marker style of a line.
While this is simple in concept, in practice it can be tricky, because
matplotlib is a graphical user interface application under the hood,
and there are some tricks to make the applications work right in a
python shell.


.. _ipython-pylab:

Ipython to the rescue
=====================

Fortunately, `ipython <http://ipython.scipy.org/dist>`_, an enhanced
interactive python shell, has figured out all of these tricks, and is
matplotlib aware, so when you start ipython in the *pylab* mode.

.. sourcecode:: ipython

    johnh@flag:~> ipython -pylab
    Python 2.4.5 (#4, Apr 12 2008, 09:09:16)
    IPython 0.9.0 -- An enhanced Interactive Python.

      Welcome to pylab, a matplotlib-based Python environment.
      For more information, type 'help(pylab)'.

    In [1]: x = randn(10000)

    In [2]: hist(x, 100)

it sets everything up for you so interactive plotting works as you
would expect it to.  Call :func:`~matplotlib.pyplot.figure` and a
figure window pops up, call :func:`~matplotlib.pyplot.plot` and your
data appears in the figure window.

Note in the example above that we did not import any matplotlib names
because in pylab mode, ipython will import them automatically.
ipython also turns on *interactive* mode for you, which causes every
pyplot command to trigger a figure update, and also provides a
matplotlib aware ``run`` command to run matplotlib scripts
efficiently.  ipython will turn off interactive mode during a ``run``
command, and then restore the interactive state at the end of the
run so you can continue tweaking the figure manually.

There has been a lot of recent work to embed ipython, with pylab
support, into various GUI applications, so check on the ipython
mailing `list
<http://projects.scipy.org/mailman/listinfo/ipython-user>`_ for the
latest status.

.. _other-shells:

Other python interpreters
=========================

If you can't use ipython, and still want to use matplotlib/pylab from
an interactive python shell, eg the plain-ole standard python
interactive interpreter, or the interpreter in your favorite IDE, you
are going to need to understand what a matplotlib backend is
:ref:`what-is-a-backend`.



With the TkAgg backend, that uses the Tkinter user interface toolkit,
you can use matplotlib from an arbitrary python shell.  Just set your
``backend : TkAgg`` and ``interactive : True`` in your
:file:`matplotlibrc` file (see :ref:`customizing-matplotlib`) and fire
up python.  Then::

  >>> from pylab import *
  >>> plot([1,2,3])
  >>> xlabel('hi mom')

should work out of the box.  Note, in batch mode, ie when making
figures from scripts, interactive mode can be slow since it redraws
the figure with each command.  So you may want to think carefully
before making this the default behavior.

For other user interface toolkits and their corresponding matplotlib
backends, the situation is complicated by the GUI mainloop which takes
over the entire process.  The solution is to run the GUI in a separate
thread, and this is the tricky part that ipython solves for all the
major toolkits that matplotlib supports.  There are reports that
upcoming versions of pygtk will place nicely with the standard python
shell, so stay tuned.

.. _controlling-interactive:

Controlling interactive updating
================================

The *interactive* property of the pyplot interface controls whether a
figure canvas is drawn on every pyplot command.  If *interactive* is
*False*, then the figure state is updated on every plot command, but
will only be drawn on explicit calls to
:func:`~matplotlib.pyplot.draw`.  When  *interactive* is
*True*, then every pyplot command triggers a draw.


The pyplot interface provides 4 commands that are useful for
interactive control.

:func:`~matplotlib.pyplot.isinteractive`
    returns the interactive setting *True|False*

:func:`~matplotlib.pyplot.ion`
    turns interactive mode on

:func:`~matplotlib.pyplot.ioff`
    turns interactive mode off

:func:`~matplotlib.pyplot.draw`
    forces a figure redraw

When working with a big figure in which drawing is expensive, you may
want to turn matplotlib's interactive setting off temporarily to avoid
the performance hit::


    >>> #create big-expensive-figure
    >>> ioff()      # turn updates off
    >>> title('now how much would you pay?')
    >>> xticklabels(fontsize=20, color='green')
    >>> draw()      # force a draw
    >>> savefig('alldone', dpi=300)
    >>> close()
    >>> ion()      # turn updating back on
    >>> plot(rand(20), mfc='g', mec='r', ms=40, mew=4, ls='--', lw=3)




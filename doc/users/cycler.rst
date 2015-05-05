.. _cycler_guide:

==========================
 Style/kwarg cycler Guide
==========================

.. currentmodule:: matplotlib.cycler

When plotting more than one line it is common to want to be able to cycle over one
or more artist styles.  For simple cases than can be done with out too much trouble:

.. plot::
   :include-source:

   fig, ax = plt.subplots(tight_layout=True)
   x = np.linspace(0, 2*np.pi, 1024)

   for i, (lw, c) in enumerate(zip(range(4), ['r', 'g', 'b', 'k'])):
      ax.plot(x, np.sin(x - i * np.pi / 4),
              label=r'$\phi = {{{0}}} \pi / 4$'.format(i),
              lw=lw + 1,
              c=c)

   ax.set_xlim([0, 2*np.pi])
   ax.set_title(r'$y=\sin(\theta + \phi)$')
   ax.set_ylabel(r'[arb]')
   ax.set_xlabel(r'$\theta$ [rad]')

   ax.legend(loc=0)

However, if you want to do something more complicated:

.. plot::
   :include-source:

   fig, ax = plt.subplots(tight_layout=True)
   x = np.linspace(0, 2*np.pi, 1024)

   for i, (lw, c) in enumerate(zip(range(4), ['r', 'g', 'b', 'k'])):
      if i % 2:
          ls = '-'
      else:
          ls = '--'
      ax.plot(x, np.sin(x - i * np.pi / 4),
              label=r'$\phi = {{{0}}} \pi / 4$'.format(i),
              lw=lw + 1,
              c=c,
              ls=ls)

   ax.set_xlim([0, 2*np.pi])
   ax.set_title(r'$y=\sin(\theta + \phi)$')
   ax.set_ylabel(r'[arb]')
   ax.set_xlabel(r'$\theta$ [rad]')

   ax.legend(loc=0)

the plotting logic can quickly become very involved.  To address this and allow easy
cycling over arbitrary ``kwargs`` the `~matplotlib.cycler.Cycler` class, a composable
kwarg iterator, was developed.


`~matplotlib.cycler.Cycler`
===========================

The public API of `Cycler` consists of a class
`~matplotlib.cycler.Cycler` and a factory function
`~matplotlib.cycler.cycler`.  The class takes care of the composition and iteration logic while
the function provides a simple interface for creating 'base' `Cycler` objects.

.. autosummary::
   :toctree: generated/

   Cycler
   cycler


A 'base' `Cycler` object is some what useful

.. plot::
   :include-source:

   from matplotlib.cycler import cycler
   from itertools import cycle

   fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
   x = np.arange(10)

   single_cycle = cycler('c', ['r', 'g', 'b'])

   for i, sty in enumerate(single_cycle):
      ax1.plot(x, x*(i+1), **sty)


   for i, sty in zip(range(1, 10), cycle(single_cycle)):
      ax2.plot(x, x*i, **sty)


.. ipython:: python

   from __future__ import print_function
   from matplotlib.cycler import cycler


   color_cycle = cycler('c', ['r', 'g', 'b'])

   color_cycle

   for v in color_cycle:
       print(v)

   len(color_cycle)



However they are most useful when composed.  They can be added

.. ipython:: python

   lw_cycle = cycler('lw', range(1, 5))
   add_cycle = color_cycle + lw_cycle

   lw_cycle
   add_cycle

   for v in add_cycle:
       print(v)

   len(add_cycle)

or multiplied

.. ipython:: python

   prod_cycle = color_cycle * lw_cycle

   color_cycle
   lw_cycle
   prod_cycle

   for v in prod_cycle:
       print(v)

   len(prod_cycle)

The result of composition is another `Cycler` object which allows very
complicated cycles to be defined very succinctly

.. ipython:: python


.. plot::
   :include-source:

   from matplotlib.cycler import cycler
   from itertools import cycle

   fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
   x = np.arange(10)

   single_cycle = cycler('c', ['r', 'g', 'b'])

   for i, sty in enumerate(single_cycle):
      ax1.plot(x, x*(i+1), **sty)


   for i, sty in zip(range(1, 10), cycle(single_cycle)):
      ax2.plot(x, x*i, **sty)

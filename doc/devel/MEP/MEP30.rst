=======================================
 MEP28: Revised pyplot / suggested API
=======================================

.. contents::
   :local:


Status
======
**Discussion**


Branches and Pull requests
==========================

3 half-done attempts


Abstract
========

Matplotlib currently has ~4 APIs.

 1. ``pyplot`` state machine API
 2. the ``OO`` API
 3. user written functions which may take an ``Axes`` or ``Figure``
 4. user written functions that create (and may not directly return)
    ``Axes`` / ``Figure`` objects

This leads to a wide variety of not incorrect but conflicting
behaviors which inhibits interaction between libraries and greatly
confuses users.  This MEP proposes some signatures and decorators.

Taking advantage key-word only arguments in 3 we can suggest a signature of ::

  def plotting_function(data, *, ax, **data_kwargs, **style_kwargs):
      arts = create_artsits(data, **data_kwargs, **style_kwargs)
      for a in arts:
          ax.add_artist(a)
      return arts

or ::

  def plotting_function_by_fig(data, *, fig, **data_kwargs, **style_kwargs):
      ax_lst = fig.subplots(N, M).ravel()
      arts = {}
      for j, ax in enumerate(ax_lst):
          kw = sub_set_kwargs(j, data_kwargs, style_kwargs)
          arts[j] = some_plotting_function(data, ax=ax, **kw))
      return arts


Building on these we can then, in ``pyplot`` provide the decorators like ::

  def ensure_current_axes(func):
      @functools.wraps(func)
      def inner(*args, **kwargs):
          if 'ax' not in kwargs:
	      kwargs['ax'] = plt.gca()

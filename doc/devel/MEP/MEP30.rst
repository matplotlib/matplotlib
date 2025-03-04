=======================================
 MEP30: Revised pyplot / suggested API
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

Matplotlib currently has two main entry points:

1. ``pyplot`` state machine API
2. the ``OO`` API

and two secondary entry points:

1. user written functions that may create (and may not directly
   return) ``Axes`` / ``Figure`` objects
2. user written functions which may take an ``Axes`` or ``Figure``


This leads to a wide variety of not incorrect but conflicting
behaviors which inhibits interaction between libraries and greatly
confuses users.

The ``pyplot`` API is very convenient for quick work in an interactive
terminal working with a small number of single-``Axes`` graphs,
however for complex figures with multiple axes or many open ``Figure``
windows the global 'current axes' quickly becomes unmanageable and
results in plotting commands going to the wrong axes [dig up issue
requesting opting slider axes out of gca consideration].
Additionally, the set of plotting functions in the ``pyplot``
namespace is controlled by Matplotlib upstream and is somewhat
limited.  There have been proposals to provide a 'register' hook [dig
up PR from 2 years ago] into the namespace, the only way for
third-party packages to add to the pyplot namespace is monkey
patching.  Either way, this can be problematic if two packages try to
use the same name and which one gets it depends on import order.  Uses
can write functions that 'feel' like ``pyplot`` functions by using
``pyplot.gca()`` in their code, however this then ties them very
tightly to ``pyplot`` which makes it more difficult to re-use the
functions in embedding applications where the user may not want to
import ``pyplot`` at all.


The ``OO`` API is a bit less concise to work with interactively, it
requires the user to explicitly create an ``Axes`` object to work
with, however it solves the problem of shared global state.  The
``Axes`` objects serve as both nodes in the draw tree and as the
primary namespace for plotting functions.  It is possible (and
encouraged) for users to write functions that take in ``Axes`` /
``Figure`` objects and internally use the ``OO`` API, however these
feel qualitatively different than the 'native' plotting routines which
are ``Axes`` methods.  This can be overcome by ex sub-classing the
``Axes`` or monkey patching methods on to it, however this has the
same problem of clashing third-party packages.

Among third-party functions in the wild (ex ``seaborn``, ``pandas``,
and user code) the return types vary between returning the artist
created during the call, the ``Axes`` objects plotted to, custom
types, and nothing.

Matplotlib is being used in a applications that are well beyond the
original use cases and well beyond the expertise of the current
development team.  To that end, we should make sure it is easy to
write plotting functions that feel "native".  This can then be used to
support an eco-system of ``mplkit-*`` domain specific libraries.  This
libraries will be able to depend on a wider range of libraries (ex
``scipy``, various scikits, and ``pandas``) than core Matplotlib and can
be built around the 'fundamental' data structures of the domain.

To better support the wide range of use cases, in a way that feels
'native', we propose the following changes:

 1. move away from ``Axes`` as the primary namespace for plotting
    routines
 2. provide decorators to easily opt third-party code into the
    ``pyplot`` state machine without much boiler plate.


Signatures
----------

There are two obvious ways to write a function has a required as input
``Axes`` in a way that can be easily wrapped by a decorator::

  def plotting_func(ax, *data_args, **style_kwargs):
      ...

or ::

  def ploting_func(*data_args, ax, **style_kwargs):
      ...

The first case has the advantage that it works in both python2 and
python3.  Calling many functions explicitly passing *ax* it would look
something like ::

  a1 = func1(ax, data1, ...)
  a2 = func2(ax, data2, ...)
  a3 = func3(ax, data3, ...)

which is only one extra space from the status quo of ``ax.func1(data1,
...)``.  However wrapping this in a decorator to provide a default
*ax* requires type checking ::

  def ensure_ax(func):
      @wraps(func)
      def inner(*args, **kwargs):
          if not isinstance(args[0], AxesBase):
	      args = (gca(), ) + args
	  return func(*args, **kwargs)
      return inner

Changing the contents of ``*args`` on the way through seems a bit
awkard and possibly a bit hard to explain.  While we have been
advocating this signature in the docs for a few years, it is not the
pattern used by major third-party extensions.

On the other hand wrapping the second option is simpler to decorate ::

  def ensure_ax(func):
      @wraps(func)
      def inner(*args, **kwargs):
          if 'ax' not in kwargs:
	      kwargs['ax'] = gca()
	  return func(*args, **kwargs)
      return inner

but is a bit more verbose when explicitly passing the *ax* argument ::

  a1 = func1(data1, ..., ax=ax)
  a2 = func2(data2, ..., ax=ax)
  a3 = func3(data3, ..., ax=ax)

which is a few more characters and swaps ``.`` or ``,`` for ``=``.
The axes-as-kwarg pattern matches the API that many third-party
libraries (``pandas``, ``sklean``, ``seaborn``, ``skimage``) are
already using.

It is possible to support both at the user level via a decorator ::

   def ensure_ax_arg(func):
       # modulo signature and docstring hacking
       @wraps(func)
       def inner(*args, **kwargs):
           ax = kwargs.pop('ax', None)

           if len(args):
               if not isinstance(args[0], AxesBase):
                  if ax is None:
                      ax = gca()
                   args = (ax, ) + args

               elif ax is not None:
                   raise ValueError("passed in 2 axes")
           else:
               if ax is None:
                   ax = gca()
               args = (ax, )
           return func(*args, **kwargs)

       return inner

   def ensure_ax_kwarg(func):
       # modulo signature and docstring hacking
       @wraps(func):
       def inner(*args, **kwargs):
           if len(args) and isinstance(args[0], AxesBase):
               ax, *args = args
           else:
               ax = None
           if 'ax' in kwargs and ax is not None:
               raise ValueError("passed in two axes")
           elif 'ax' not in kwargs:
               if ax is None:
                   ax = gca()
               kwargs['ax'] = ax
           return func(*args, **kwargs)
       return inner

but it is not clear if the complexity is worth it.  It would allow the end users to call
plotting functions three ways ::

  a1 = func(*data_args, **style_kwargs)
  a2 = func(ax, *data_args, **style_kwargs)
  a3 = func(*data_args, ax=ax, **style_kwargs)


and allow libraries to internally organize them selves using either of
the above Axes-is-required API.  This avoids bike-shedding over the
API and eliminates the first-party 'special' namespace, but is a bit
magical.


Factories
---------

A design principle which is applied to some parts of the library (ex
``contour`` and ``quiver``) is to separate the logic of create the
artists to be added to the draw tree and logic of adding them to the
draw tree more cleanly.  Than is functions that look like ::

  def artist_factory(*data_args, **style_kwargs):
      ...
      return arts

It may be better to return these as a simple iterable ::

  def artist_factory(*data_args, **style_kwargs) -> List[Artist]:
      ...
      return arts

or as a dictionary::

  def artist_factory(*data_args, **style_kwargs) -> Dict[str, Artist]:
      ...
      return arts

The first case is simpler, but the second case exposes more semantics.

In either case, with a few exceptions where the plotting methods
change other properties of the axes (such as ``imshow`` which sets the
extents and may flip the y-axis), many plotting functions can be
implemented as simple wrappers ::

  def add_to_axes(func):
      # modulo signature and docstring hacking
      @wraps(func)
      def inner(*data_args, ax, **style_wkargs):
          arts = func(*data_args, **style_kwargs)
	  for a in arts.values():
	      ax.add_artist(a)
	  return arts
      return inner

Thus ::

  @ensure_ax_kwarg
  @add_to_axes
  def art_factory(*data_args, **kwargs):
      ...
      return arts

will produce a function which is a first-class.  From a list of factories namespaces
for the three levels can easily be produced::

   func_list = [...]
   factory = SimpleNamespace(**{f.name: f for f in func_list})
   explicit = SimpleNamespace(**{f.name: add_to_ax(getattr(factory, f.name))
                                 for f in func_list})
   implicit = SimpleNamespace(**{f.name: ensure_ax_kwarg(getattr(explicit, f.name))
                                 for f in func_list})

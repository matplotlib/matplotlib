======================
 ``animation`` module
======================

.. automodule:: matplotlib.animation

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: entry


Animation
=========

The easiest way to make a live animation in mpl is to use one of the
`Animation` classes.

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   FuncAnimation
   ArtistAnimation

In both cases it is critical to keep a reference to the instance
object.  The animation is advanced by a timer (typically from the host
GUI framework) which the `Animation` object holds the only reference
to.  If you do not hold a reference to the `Animation` object, it (and
hence the timers), will be garbage collected which will stop the
animation.

To save an animation use to disk use

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Animation.save
   Animation.to_html5_video

See :ref:`ani_writer_classes` below for details about what movie formats are supported.


``FuncAnimation``
-----------------

The inner workings of `FuncAnimation` is more-or-less::

  for d in frames:
     arts = func(d, *fargs)
     fig.canvas.draw_idle()
     plt.pause(interval)


with details to handle 'blitting' (to dramatically improve the live
performance), to be non-blocking, handle repeats, multiple animated
axes, and easily save the animation to a movie file.

'Blitting' is a `old technique
<https://en.wikipedia.org/wiki/Bit_blit>`__ in computer graphics.  The
general gist is to take as existing bit map (in our case a mostly
rasterized figure) and then 'blit' one more artist on top.  Thus, by
managing a saved 'clean' bitmap, we can only re-draw the few artists
that are changing at each frame and possibly save significant amounts of
time.  When using blitting (by passing ``blit=True``) the core loop of
`FuncAnimation` gets a bit more complicated ::

   ax = fig.gca()

   def update_blit(arts):
       fig.canvas.restore_region(bg_cache)
       for a in arts:
           a.axes.draw_artist(a)

       ax.figure.canvas.blit(ax.bbox)

   arts = init_func()

   for a in arts:
      a.set_animated(True)

   fig.canvas.draw()
   bg_cache = fig.canvas.copy_from_bbox(ax.bbox)

   for f in frames:
       arts = func(f, *fargs)
       update_blit(arts)
       plt.pause(interval)

This is of course leaving out many details (such as updating the
background when the figure is resized or fully re-drawn).  However,
this hopefully minimalist example gives a sense of how ``init_func``
and ``func`` are used inside of `FuncAnimation` and the theory of how
'blitting' works.

The expected signature on ``func`` and ``init_func`` is very simple to
keep `FuncAnimation` out of your book keeping and plotting logic, but
this means that the callable objects you pass in must know what
artists they should be working on.  There are several approaches to
handling this, of varying complexity and encapsulation.   The simplest
approach, which works quite well in the case of a script, is to define the
artist at a global scope and let Python sort things out.  For example ::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.animation import FuncAnimation

   fig, ax = plt.subplots()
   xdata, ydata = [], []
   ln, = plt.plot([], [], 'ro', animated=True)

   def init():
       ax.set_xlim(0, 2*np.pi)
       ax.set_ylim(-1, 1)
       return ln,

   def update(frame):
       xdata.append(frame)
       ydata.append(np.sin(frame))
       ln.set_data(xdata, ydata)
       return ln,

   ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                       init_func=init, blit=True)
   plt.show()


The second method is to us `functools.partial` to 'bind' artists to
function.  A third method is to use closures to build up the required
artists and functions.  A fourth method is to create a class.




Examples
~~~~~~~~

.. toctree::
   :maxdepth: 1

   ../examples/animation/animate_decay
   ../examples/animation/bayes_update
   ../examples/animation/double_pendulum_animated
   ../examples/animation/dynamic_image
   ../examples/animation/histogram
   ../examples/animation/rain
   ../examples/animation/random_data
   ../examples/animation/simple_3danim
   ../examples/animation/simple_anim
   ../examples/animation/strip_chart_demo
   ../examples/animation/unchained

``ArtistAnimation``
-------------------


Examples
~~~~~~~~

.. toctree::
   :maxdepth: 1

   ../examples/animation/basic_example
   ../examples/animation/basic_example_writer
   ../examples/animation/dynamic_image2




Writer Classes
==============



.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   AVConvFileWriter
   AVConvWriter
   FFMpegFileWriter
   FFMpegWriter
   ImageMagickFileWriter
   ImageMagickWriter

:ref:`animation-moviewriter`


.. _ani_writer_classes:

Helper Classes
==============


Animation Base Classes
----------------------


.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Animation
   TimedAnimation


Custom Animation classes
------------------------

:ref:`animation-subplots`

Writer Registry
---------------

A module-level registry is provided to map between the name of the
writer and the class to allow a string to be passed to
`Animation.save` instead of a writer instance.

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   MovieWriterRegistry

Writer Base Classes
-------------------

To reduce code duplication base classes

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   MovieWriter
   FileMovieWriter

and mixins are provided

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   AVConvBase
   FFMpegBase
   ImageMagickBase

See the source code for how to easily implement new `MovieWriter`
classes.


Inheritance Diagrams
====================

.. inheritance-diagram:: matplotlib.animation.FuncAnimation matplotlib.animation.ArtistAnimation
   :private-bases:

.. inheritance-diagram:: matplotlib.animation.AVConvFileWriter matplotlib.animation.AVConvWriter matplotlib.animation.FFMpegFileWriter matplotlib.animation.FFMpegWriter matplotlib.animation.ImageMagickFileWriter matplotlib.animation.ImageMagickWriter
   :private-bases:



Deprecated
==========


.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   MencoderBase
   MencoderFileWriter
   MencoderWriter

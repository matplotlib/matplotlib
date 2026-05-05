************************
``matplotlib.animation``
************************

.. automodule:: matplotlib.animation
   :no-members:
   :no-undoc-members:

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: entry


Animation
=========

The easiest way to make a live animation in Matplotlib is to use one of the
`Animation` classes.

.. seealso::
   - :ref:`animations`

.. inheritance-diagram:: matplotlib.animation.FuncAnimation matplotlib.animation.ArtistAnimation
   :parts: 1

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Animation
   FuncAnimation
   ArtistAnimation

In both cases it is critical to keep a reference to the instance
object.  The animation is advanced by a timer (typically from the host
GUI framework) which the `Animation` object holds the only reference
to.  If you do not hold a reference to the `Animation` object, it (and
hence the timers) will be garbage collected which will stop the
animation.

To save an animation use `Animation.save`, `Animation.to_html5_video`,
or `Animation.to_jshtml`.

See :ref:`ani_writer_classes` below for details about what movie formats are
supported.


.. _func-animation:

``FuncAnimation``
-----------------

see the `Animation guide <https://matplotlib.org/stable/users/api/animations.html>`
_for a detailed explanation of FuncAnimation.

``ArtistAnimation``
-------------------

Examples
^^^^^^^^

* :doc:`../gallery/animation/dynamic_image`

Writer Classes
==============

.. inheritance-diagram:: matplotlib.animation.FFMpegFileWriter matplotlib.animation.FFMpegWriter matplotlib.animation.ImageMagickFileWriter matplotlib.animation.ImageMagickWriter matplotlib.animation.PillowWriter matplotlib.animation.HTMLWriter
   :top-classes: matplotlib.animation.AbstractMovieWriter
   :parts: 1

The provided writers fall into a few broad categories.

The Pillow writer relies on the Pillow library to write the animation, keeping
all data in memory.

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   PillowWriter

The HTML writer generates JavaScript-based animations.

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   HTMLWriter

The pipe-based writers stream the captured frames over a pipe to an external
process.  The pipe-based variants tend to be more performant, but may not work
on all systems.

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   FFMpegWriter
   ImageMagickWriter

The file-based writers save temporary files for each frame which are stitched
into a single file at the end.  Although slower, these writers can be easier to
debug.

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   FFMpegFileWriter
   ImageMagickFileWriter

The writer classes provide a way to grab sequential frames from the same
underlying `~matplotlib.figure.Figure`.  They all provide three methods that
must be called in sequence:

- `~.AbstractMovieWriter.setup` prepares the writer (e.g. opening a pipe).
  Pipe-based and file-based writers take different arguments to ``setup()``.
- `~.AbstractMovieWriter.grab_frame` can then be called as often as
  needed to capture a single frame at a time
- `~.AbstractMovieWriter.finish` finalizes the movie and writes the output
  file to disk.

Example::

   moviewriter = MovieWriter(...)
   moviewriter.setup(fig, 'my_movie.ext', dpi=100)
   for j in range(n):
       update_figure(j)
       moviewriter.grab_frame()
   moviewriter.finish()

If using the writer classes directly (not through `Animation.save`), it is
strongly encouraged to use the `~.AbstractMovieWriter.saving` context manager::

  with moviewriter.saving(fig, 'myfile.mp4', dpi=100):
      for j in range(n):
          update_figure(j)
          moviewriter.grab_frame()

to ensure that setup and cleanup are performed as necessary.

Examples
--------

* :doc:`../gallery/animation/frame_grabbing_sgskip`

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

   AbstractMovieWriter
   MovieWriter
   FileMovieWriter

and mixins

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   FFMpegBase
   ImageMagickBase

are provided.

See the source code for how to easily implement new `MovieWriter` classes.

---------------
FuncAnimation
---------------
For detailed explanation of how FuncAnimation works see :ref: `animation_api`.

The inner workings of FuncAnimation is based on the following loo::
    for d in frames:
        artists=func(d, *frags)
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(interval)
This also handles blitting, repeating, 
multiple axes and saving to movie files.

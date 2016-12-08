======================
 ``animation`` module
======================

.. currentmodule:: matplotlib.animation


Animation
=========

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   FuncAnimation
   ArtistAnimation
   Animation.save


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

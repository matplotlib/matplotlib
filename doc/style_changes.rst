Default Style Changes
=====================

Colormap
--------

``matplotlib`` is changing the default colormap and styles in the
upcoming 2.0 release!

The new default color map will be 'viridis' (aka `option
D <http://bids.github.io/colormap/>`_).  For an introduction to color
theory and how 'viridis' was generated watch Nathaniel Smith and
Stéfan van der Walt's talk from SciPy2015

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/xAoljeRJ3lU" frameborder="0" allowfullscreen></iframe>

All four color maps will be included in matplotlib 1.5.


Everything Else
---------------

We have collected proposals to change any and all visual defaults
(including adding new rcParams as needed).

Michael Droettboom and Thomas Caswell will decide on the new default
styles to appear in the upcoming 2.0 release.

A 'classic' style sheet will be provided so reverting to the 1.x
default values will be a single line of python
(`mpl.style.use('classic')`).

.. _color_changes:

*********************
Default Color changes
*********************

As discussed at length elsewhere [insert links], ``jet`` is an
empirically bad color map and should not be the default color map.
Due to the position that changing the appearance of the plot breaks
backward compatibility, this change has been put off for far longer
than it should have been.  In addition to changing the default color
map we plan to take the chance to change the default color-cycle on
plots and to adopt a different color map for filled plots (``imshow``,
``pcolor``, ``contourf``, etc) and for scatter like plots.


Default Heat Map Colormap
-------------------------

The choice of a new color map is fertile ground to bike-shedding ("No,
it should be _this_ color") so we have a proposed set criteria (via
Nathaniel Smith) to evaluate proposed color maps.

- it should be a sequential colormap, because diverging colormaps are
  really misleading unless you know where the "center" of the data is,
  and for a default colormap we generally won't.

- it should be perceptually uniform, i.e., human subjective judgments
  of how far apart nearby colors are should correspond as linearly as
  possible to the difference between the numerical values they
  represent, at least locally.

- it should have a perceptually uniform luminance ramp, i.e. if you
  convert to greyscale it should still be uniform. This is useful both
  in practical terms (greyscale printers are still a thing!) and
  because luminance is a very strong and natural cue to magnitude.

- it should also have some kind of variation in hue, because hue
  variation is a really helpful additional cue to perception, having
  two cues is better than one, and there's no reason not to do it.

- the hue variation should be chosen to produce reasonable results
  even for viewers with the more common types of
  colorblindness. (Which rules out things like red-to-green.)

- For bonus points, it would be nice to choose a hue ramp that still
  works if you throw away the luminance variation, because then we
  could use the version with varying luminance for 2d plots, and the
  version with just hue variation for 3d plots. (In 3d plots you
  really want to reserve the luminance channel for lighting/shading,
  because your brain is *really* good at extracting 3d shape from
  luminance variation. If the 3d surface itself has massively varying
  luminance then this screws up the ability to see shape.)

- Not infringe any existing IP

Example script
++++++++++++++

Proposed Colormaps
++++++++++++++++++

Default Scatter Colormap
------------------------

For heat-map like applications it can be desirable to cover as much of
the luminence scale as possible, however when color mapping markers,
having markers too close to white can be a problem.  For that reason
we propose using a different (but maybe related) color map to the
heat map for marker-based.  The design parameters are the same as
above, only with a more limited luminence variation.


Example script
++++++++++++++
::

   import numpy as np
   import matplotlib.pyplot as plt

   np.random.seed(1234)

   fig, (ax1, ax2) = plt.subplots(1, 2)

   N = 50
   x = np.random.rand(N)
   y = np.random.rand(N)
   colors = np.random.rand(N)
   area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

   ax1.scatter(x, y, s=area, c=colors, alpha=0.5)


   X,Y = np.meshgrid(np.arange(0, 2*np.pi, .2),
                     np.arange(0, 2*np.pi, .2))
   U = np.cos(X)
   V = np.sin(Y)
   Q = ax2.quiver(X, Y, U, V, units='width')
   qd = np.random.rand(np.prod(X.shape))
   Q.set_array(qd)

Proposed Colormaps
++++++++++++++++++

Color Cycle / Qualitative color map
-----------------------------------

When plotting lines it is frequently desirable to plot multiple lines
or artists which need to be distinguishable, but there is no inherent
ordering.


Example script
++++++++++++++
::

   import numpy as np
   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(1, 2)

   x = np.linspace(0, 1, 10)

   for j in range(10):
       ax1.plot(x, x * j)


   th = np.linspace(0, 2*np.pi, 1024)
   for j in np.linspace(0, np.pi, 10):
       ax2.plot(th, np.sin(th + j))

   ax2.set_xlim(0, 2*np.pi)

Proposed Color cycle
++++++++++++++++++++

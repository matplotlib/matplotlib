"""
=================
Blitting Tutorial
=================

'Blitting' is a `standard technique
<https://en.wikipedia.org/wiki/Bit_blit>`__ in computer graphics that
in the context of matplotlib can be used to (drastically) improve
performance of interactive figures.  It is used internally by the
:mod:`~.animation` and :mod:`~.widgets` modules for this reason.

The source of the performance gains is simply not re-doing work we do
not have to.  For example, if the limits of an Axes have not changed,
then there is no reason we should re-draw all of the ticks and
tick-labels (particularly because text is one of the more expensive
things to render).

The procedure to save our work is roughly:

- draw the figure, but exclude an artists marked as 'animated'
- save a copy of the Agg RBGA buffer

In the future, to update the 'animated' artists we

- restore our copy of the RGBA buffer
- redraw only the animated artists
- show the resulting image on the screen

thus saving us from having to re-draw everything which is _not_
animated.

Simple Example
--------------

We can implement this via methods on `.CanvasAgg` and setting
``animated=True`` on our artist.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)

fig, ax = plt.subplots()
# animated=True makes the artist be excluded from normal draw tree
ln, = ax.plot(x, np.sin(x), animated=True)

# stop to admire our empty window axes and ensure it is drawn
plt.pause(.1)

# save a copy of the image sans animated artist
bg = fig.canvas.copy_from_bbox(fig.bbox)
# draw the animated artist
ax.draw_artist(ln)
# show the result to the screen
fig.canvas.blit(fig.bbox)

for j in range(100):
    # put the un-changed background back
    fig.canvas.restore_region(bg)
    # update the artist.
    ln.set_ydata(np.sin(x + (j / 100) * np.pi))
    # re-render the artist
    ax.draw_artist(ln)
    # copy the result to the screen
    fig.canvas.blit(fig.bbox)


###############################################################################
# This example works and shows a simple animation, however because we are only
# grabbing the background once, if the size of dpi of the figure change, the
# background will be invalid and result in incorrect images.  There is also a
# global variable and a fair amount of boiler plate which suggests we should
# wrap this in a class.
#
# Class-based example
# -------------------
#
# We can use a class to encapsulate the boilerplate logic and state of
# restoring the background, drawing the artists, and then blitting the
# result to the screen.  Additionally, we can use the ``'draw_event'``
# callback to capture a new background whenever a full re-draw
# happens.


class BlitManager:

    def __init__(self, canvas, animated_artists):
        """
        Parameters
        ----------
        canvas : CanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~CanvasAgg.copy_from_bbox` and
            `~CanvasAgg.restore_region` methods.

        animated_artists : Optional[List[Artist]]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'
        """
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """Add a artist to be managed

        Parameters
        ----------
        art : Artist
            The artist to be added.  Will be set to 'animated' (just to be safe).
            *art* must be in the figure associated with the canvas this class
            is managing.
        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists
        """
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists
        """
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the old background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the screen
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


###############################################################################
# And now use our class.  This is a slightly more complicated example of the
# first case as we add a text frame counter as well.

# make a new figure
fig, ax = plt.subplots()
# add a line
ln, = ax.plot(x, np.sin(x), animated=True)
# add a frame number
fr_number = ax.annotate('0', (0, 1),
                        xycoords='axes fraction',
                        xytext=(10, -10),
                        textcoords='offset points',
                        ha='left', va='top',
                        animated=True)
bm = BlitManager(fig.canvas, [ln, fr_number])
plt.pause(.1)

for j in range(100):
    # update the artists
    ln.set_ydata(np.sin(x + (j / 100) * np.pi))
    fr_number.set_text('frame: {j}'.format(j=j))
    # tell the blitting manager to do it's thing
    bm.update()

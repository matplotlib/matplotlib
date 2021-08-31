Figure now has ``draw_without_rendering`` method
------------------------------------------------

Some aspects of a figure are only determined at draw-time, such as the exact
position of text artists or deferred computation like automatic data limits.
If you need these values, you can use ``figure.canvas.draw()`` to force a full
draw. However, this has side effects, sometimes requires an open file, and is
doing more work than is needed.

The new `.Figure.draw_without_rendering` method runs all the updates that
``draw()`` does, but skips rendering the figure. It's thus more efficient if you
need the updated values to configure further aspects of the figure.

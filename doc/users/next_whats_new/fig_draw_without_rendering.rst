Figure now has ``draw_without_rendering`` method
------------------------------------------------

Rarely, the user will want to trigger a draw without rendering to either the
screen or a file.  This is useful for determining the final position of artists
on the figure that require a draw, like text artists, or resolve deferred
computation like automatic data limits.  This can be done by
``fig.canvas.draw()``, which forces a full draw and rendering, however this has
side effects, sometimes requires an open file, and is doing more work than is
needed.  The `.Figure.draw_without_rendering` method is provided to run the
code in Matplotlib that updates values that are computed at draw-time and get
accurate dimensions of the Artists more efficiently.

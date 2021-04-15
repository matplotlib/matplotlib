Figure now has draw_no_output method
------------------------------------

Rarely, the user will want to trigger a draw without making output to 
either the screen or a file.  This is useful for determining the final 
position of artists on the figure that require a draw like text.
This could be accomplished via ``fig.canvas.draw()`` but that is 
not user-facing, so a new method on `.Figure.draw_no_output` is provided.  
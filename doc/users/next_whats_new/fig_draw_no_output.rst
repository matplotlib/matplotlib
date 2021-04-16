Figure now has draw_no_output method
------------------------------------

Rarely, the user will want to trigger a draw without making output to 
either the screen or a file.  This is useful for determining the final 
position of artists on the figure that require a draw, like text artists.
This could be accomplished via ``fig.canvas.draw()`` but has side effects,
sometimes requires an open file, and is documented on an object most users 
do not need to access.  The `.Figure.draw_no_output` is provided to trigger 
a draw without pushing to the final output, and with fewer side effects.
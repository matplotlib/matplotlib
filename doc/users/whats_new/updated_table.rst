Updated Table and to control edge visibility
--------------------------------------------
Added the ability to toggle the visibility of lines in Tables.
Functionality added to the table() factory function under the keyword argument "edges".
Values can be the strings "open", "closed", "horizontal", "vertical" or combinations of the letters "L", "R", "T", "B" which represent left, right, top, and bottom respectively.

Example:
    table(..., edges="open")  # No line visible
    table(..., edges="closed")  # All lines visible
    table(..., edges="horizontal")  # Only top and bottom lines visible
    table(..., edges="LT")  # Only left and top lines visible.

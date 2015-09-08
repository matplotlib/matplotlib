Moved ``ignore``, ``set_active``, and ``get_active`` methods to base class ``Widget``
`````````````````````````````````````````````````````````````````````````````````````

Pushes up duplicate methods in child class to parent class to avoid duplication of code.


Adds enable/disable feature to MultiCursor
``````````````````````````````````````````

A MultiCursor object can be disabled (and enabled) after it has been created without destroying the object.
Example::

  multi_cursor.active = False


Improved RectangleSelector and new EllipseSelector Widget
`````````````````````````````````````````````````````````

Adds an `interactive` keyword which enables visible handles for manipulating the shape after it has been drawn.

Adds keyboard modifiers for:

- Moving the existing shape (default key = 'space')
- Making the shape square (default 'shift')
- Make the initial point the center of the shape (default 'control')
- Square and center can be combined

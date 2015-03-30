Moved ``ignore``, ``set_active``, and ``get_active`` methods to base class ``Widget``
`````````````````````````````````````````````````````````````````````````````````
Pushes up duplicate methods in child class to parent class to avoid duplication of code.


Adds enable/disable feature to MultiCursor
``````````````````````````````````````````
A MultiCursor object can be disabled (and enabled) after it has been created without destroying the object. 
Example:
        multi_cursor.active = False

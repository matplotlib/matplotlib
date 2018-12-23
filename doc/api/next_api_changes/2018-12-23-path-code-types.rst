Path code types like ``Path.MOVETO`` are now ``np.uint8`` instead of ``int``
````````````````````````````````````````````````````````````````````````````

``Path.STOP``, ``Path.MOVETO``, ``Path.LINETO``, ``Path.CURVE3``,
``Path.CURVE4`` and ``Path.CLOSEPOLY`` are now of the type ``Path.code_type``
(``np.uint8`` by default) instead of plain ``int``. This makes their type
match the array value type of the ``Path.codes`` array.

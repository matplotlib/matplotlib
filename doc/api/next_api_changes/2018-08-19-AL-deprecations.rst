Deprecations
````````````

The ``\stackrel`` mathtext command is deprecated (it behaved differently
from LaTeX's ``\stackrel``.  To stack two mathtext expressions, use
``\genfrac{left-delim}{right-delim}{fraction-bar-thickness}{}{top}{bottom}``.

Undeprecations
``````````````

The ``obj_type`` kwarg to the ``cbook.deprecated`` decorator is undeprecated.

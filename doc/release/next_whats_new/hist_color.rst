Handle single color for multiple datasets in hist
-----------------

It is now possible to use a single color with multiple datasets in ``hist``.
Up to now, the 'color' keyword argument required one color per dataset.
Using a single color with multiple datasets would previously lead to a ValueError.

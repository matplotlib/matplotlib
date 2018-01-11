Using strings instead of booleans to control grid and tick visibility is deprecated
```````````````````````````````````````````````````````````````````````````````````

Using ``"on"``, ``"off"``, ``"true"``, or ``"false"`` to control grid
and tick visibility has been deprecated.  Instead, use normal booleans
(``True``/``False``) or boolean-likes.  In the future, all non-empty strings
may be interpreted as ``True``.

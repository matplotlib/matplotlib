`cbook.is_numlike` only performs an instance check, `cbook.is_string_like` is deprecated
````````````````````````````````````````````````````````````````````````````````````````

`cbook.is_numlike` now only checks that its argument is an instance of
``(numbers.Number, np.Number)`` and.  In particular, this means that arrays are
now not num-like.

`cbook.is_string_like` and `cbook.is_sequence_of_strings` have been
deprecated.  Use ``isinstance(obj, six.string_types)`` and ``all(isinstance(o,
six.string_types) for o in obj) and not iterable(obj)`` instead.

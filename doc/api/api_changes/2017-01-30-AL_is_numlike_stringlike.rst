`cbook.is_numlike` and `cbook.is_string_like` only perform an instance check
````````````````````````````````````````````````````````````````````````````

`cbook.is_numlike` and `cbook.is_string_like` now only check that
their argument is an instance of ``(numbers.Number, np.Number)`` and
``(six.string_types, np.str_, np.unicode_)`` respectively.  In particular, this
means that arrays are now never num-like or string-like regardless of their
dtype.

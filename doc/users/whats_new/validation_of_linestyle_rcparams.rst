Validation of line style rcParams
---------------------------------

Stricter validation
```````````````````
The validation of rcParams that are related to line styles
(``lines.linestyle``, ``boxplot.*.linestyle``, ``grid.linestyle`` and
``contour.negative_linestyle``) now effectively checks that the values
are valid line styles. Strings like ``dashed`` or ``--`` are accepted,
as well as even-length sequences of on-off ink like ``[1, 1.65]``. In
this latter case, the offset value is handled internally and should *not*
be provided by the user.

The validation is case-insensitive.

Deprecation of the former validators for ``contour.negative_linestyle``
```````````````````````````````````````````````````````````````````````
The new validation scheme replaces the former one used for the
``contour.negative_linestyle`` rcParams, that was limited to ``solid``
and ``dashed`` line styles.

The former public validation functions ``validate_negative_linestyle``
and ``validate_negative_linestyle_legacy`` will be deprecated in 2.1 and
may be removed in 2.3. There are no public functions to replace them.

Examples of use
```````````````
::

    grid.linestyle             : (1, 3)   # loosely dotted grid lines
    contour.negative_linestyle : dashdot  # previously only solid or dashed

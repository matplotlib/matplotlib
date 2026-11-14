``\limits`` and ``\nolimits`` in mathtext
-----------------------------------------
Mathtext now parses the ``\limits`` and ``\nolimits`` commands.  They control
whether the sub- and superscripts of the preceding operator are placed directly
above and below it, or to its side:

- ``\limits`` forces over/under placement, e.g. ``$\int\limits_a^b$`` renders
  the bounds above and below the integral sign.
- ``\nolimits`` forces side placement, e.g. ``$\sum\nolimits_i^n$`` renders the
  bounds to the right of the summation sign.

Previously these commands raised a parse error.

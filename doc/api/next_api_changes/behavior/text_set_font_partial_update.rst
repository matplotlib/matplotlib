``Text.set_font`` now performs a partial update for string arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`.Text.set_font` previously behaved identically to `.Text.set_fontproperties`:
passing a string caused **all** font properties (size, weight, style, etc.) to be
reset to their defaults.  This was surprising given that all other``set_font*`` methods
(`~.Text.set_fontfamily`, `~.Text.set_fontsize`, `~.Text.set_fontweight`, ...)
update only the property they describe.

Starting with this release ``set_font`` performs a *partial* update when given
a string:

* The string is interpreted as a fontconfig pattern (same syntax as before).
* Only the properties explicitly named in the pattern are changed.
* All other font properties (size, weight, style, ...) are preserved.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    t1 = ax.text(0.5, 0.5, "Hello", fontsize=20, fontweight="bold")
    t2 = ax.text(0.5, 0.7, "Hello", fontsize=20, fontweight="bold")

    # Example 1: Set family name:
    t1.set_font("DejaVu Serif")
    # Old behaviour: size and weight would be reset to defaults.
    # New behaviour: only the family is updated; size=20 and bold weight are kept.

    # Example 2: Set fontconfig pattern with multiple properties:
    t2.set_font("DejaVu Serif:italic:size=14")
    # - Old behaviour: weight would be reset to defaults.
    # - New behaviour: family, italics, and size are updated, but bold weight is kept.

For a complete replacement of all font properties (i.e. the previous behaviour)
use `.Text.set_fontproperties` ::

    t.set_fontproperties("DejaVu Serif")   # resets all other properties

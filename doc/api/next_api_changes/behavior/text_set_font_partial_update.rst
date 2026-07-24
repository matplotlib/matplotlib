``Text.set_font`` now performs a partial update for string arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`.Text.set_font` (and its alias ``font=`` in keyword-argument form) previously
behaved identically to `.Text.set_fontproperties`: passing a string caused
**all** font properties to be replaced, resetting size, weight, style, etc. to
their defaults.  This was surprising given that all other ``set_font*`` methods
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
    t = ax.text(0.5, 0.5, "Hello")

    t.set_fontsize(20)
    t.set_fontweight("bold")

    # Old behaviour: size and weight would be reset to defaults.
    # New behaviour: only the family is updated; size=20 and bold weight remain.
    t.set_font("DejaVu Serif")

    # A rich fontconfig pattern updates exactly those properties:
    t.set_font("DejaVu Serif:italic:size=14")

For a complete replacement of all font properties (the previous behaviour)
use `.Text.set_fontproperties` ::

    t.set_fontproperties("DejaVu Serif")   # resets all other properties

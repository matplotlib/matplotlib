``matplotlib.mathtext`` now supports *overset* and *underset* LaTeX symbols
---------------------------------------------------------------------------

`.mathtext`, the default TeX layout engine which is shipped along with
Matplotlib now supports symbols like *overset* and *underset*.

The structure which should be followed: "\overset{body}{annotation}" or
"\underset{body}{annotation}", where *body* would be the text "above" or
"below" the *annotation* - the baseline character.

.. plot::

    math_expr = r"$ x \overset{f}{\rightarrow} y \underset{f}{\leftarrow} z $"
    plt.text(0.4, 0.5, math_expr, usetex=False)

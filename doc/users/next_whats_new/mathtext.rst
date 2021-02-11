``matplotlib.mathtext`` now supports *overset* and *underset* LaTeX symbols
---------------------------------------------------------------------------

`.mathtext` now supports *overset* and *underset*, called as 
``\overset{annotation}{body}`` or ``\underset{annotation}{body}``, where 
*annotation* is the text "above" or "below" the *body*.

.. plot::

    math_expr = r"$ x \overset{f}{\rightarrow} y \underset{f}{\leftarrow} z $"
    plt.text(0.4, 0.5, math_expr, usetex=False)

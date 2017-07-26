r"""
=========================================
The difference between \\dfrac and \\frac
=========================================

In this example, the differences between the \\dfrac and \\frac TeX macros are
illustrated; in particular, the difference between display style and text style
fractions when using Mathtex.

.. versionadded:: 2.1

.. note::
    To use \\dfrac with the LaTeX engine (text.usetex : True), you need to
    import the amsmath package with the text.latex.preamble rc, which is
    an unsupported feature; therefore, it is probably a better idea to just
    use the \\displaystyle option before the \\frac macro to get this behavior
    with the LaTeX engine.

"""

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5.25, 0.75))
fig.text(0.5, 0.3, r'\dfrac: $\dfrac{a}{b}$',
         horizontalalignment='center', verticalalignment='center')
fig.text(0.5, 0.7, r'\frac: $\frac{a}{b}$',
         horizontalalignment='center', verticalalignment='center')
plt.show()

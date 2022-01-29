"""
========================================
Auto font management for various locales
========================================

This program use matplotlib.use_locale_font to put Chinese (Simplified)
characters onto a plot.
"""


import matplotlib.pyplot as plt

plt.use_locale_font()
plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel("$x$")
plt.ylabel("$x^2$")
plt.title("平方关系")
plt.show()

"""
=================================
Ways to set a color's alpha value
=================================

Compare setting alpha by the *alpha* keyword argument and by one of the Matplotlib color
formats. Often, the *alpha* keyword is the only tool needed to add transparency to a
color. In some cases, the *(matplotlib_color, alpha)* color format provides an easy way
to fine-tune the appearance of a Figure.

"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility.
np.random.seed(19680801)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

x_values = [n for n in range(20)]
y_values = np.random.randn(20)

facecolors = ['green' if y > 0 else 'red' for y in y_values]
edgecolors = facecolors

ax1.bar(x_values, y_values, color=facecolors, edgecolor=edgecolors, alpha=0.5)
ax1.set_title("Explicit 'alpha' keyword value\nshared by all bars and edges")


# Normalize y values to get distinct face alpha values.
abs_y = [abs(y) for y in y_values]
face_alphas = [n / max(abs_y) for n in abs_y]
edge_alphas = [1 - alpha for alpha in face_alphas]

colors_with_alphas = list(zip(facecolors, face_alphas))
edgecolors_with_alphas = list(zip(edgecolors, edge_alphas))

ax2.bar(x_values, y_values, color=colors_with_alphas,
        edgecolor=edgecolors_with_alphas)
ax2.set_title('Normalized alphas for\neach bar and each edge')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar`
#    - `matplotlib.pyplot.subplots`

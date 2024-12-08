Dark-mode diverging colormaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three diverging colormaps have been added: "berlin", "managua", and "vanimo".
They are dark-mode diverging colormaps, with minimum lightness at the center,
and maximum at the extremes. These are taken from F. Crameri's Scientific
colour maps version 8.0.1 (DOI: https://doi.org/10.5281/zenodo.1243862).

.. note::
    For `cmap` arguments, we recommend using string colormap names (e.g., `cmap="berlin"`).
    This is both readable and concise. If an explicit colormap instance is needed, use
    `plt.colormaps["berlin"]`. The `plt.cm.*` syntax is less scalable, as it only supports
    built-in colormaps and not user-registered ones.

.. plot::
    :include-source: true
    :alt: Example figures using "imshow" with dark-mode diverging colormaps on positive and negative data. First panel: "berlin" (blue to red with a black center); second panel: "managua" (orange to cyan with a dark purple center); third panel: "vanimo" (pink to green with a black center).

    import numpy as np
    import matplotlib.pyplot as plt

    vals = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(vals, vals)
    img = np.sin(x*y)

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap="berlin")
    ax[1].imshow(img, cmap="managua")
    ax[2].imshow(img, cmap="vanimo")

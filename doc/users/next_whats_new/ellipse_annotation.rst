``ellipse`` boxstyle option for annotations
-------------------------------------------

The ``'ellipse'`` option for boxstyle can now be used to create annotations
with an elliptical outline. It can be used as a closed curve shape for
longer texts instead of the ``'circle'`` boxstyle which can get quite big.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    t = ax.text(0.5, 0.5, "elliptical box",
            ha="center", size=15,
            bbox=dict(boxstyle="ellipse,pad=0.3"))

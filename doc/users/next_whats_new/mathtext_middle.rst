``mathtext`` supports ``\middle``
---------------------------------

The ``\middle`` latex command is now supported by `.mathtext`. It is a
complement to ``\left`` and ``\right`` and used to add a middle sized
separator. Currently, only a single ``\middle`` between ``\left`` and
``\right`` is supported.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    plt.figure(figsize=(2, 1))
    plt.figtext(
        0.05, 0.55,
        r"$\left\{\sum_{i=0}^x a_i \middle | x \in \mathbb{N}\right\}$",
        size=20, math_fontfamily='cm')

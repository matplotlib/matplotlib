New customization of MarkerStyle
--------------------------------

New MarkerStyle parameters allow control of join style and cap style. 
The appearance of individual markers can be further controlled by 
transform supplied during creation.

.. plot::
    :include-source:
    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.suptitle('New markers', fontsize=14)
    for col, (size, rot) in enumerate(zip([2, 5, 10], [0, 45, 90])):
        t = Affine2D().rotate_deg(rot).scale(size)
        ax.plot(col, 0, marker=MarkerStyle("*", transform=t))
    ax.axis("off")
    ax.set_xlim(-0.1, 2.4)

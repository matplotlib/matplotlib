Legend handler for PatchCollection objects
------------------------------------------

PatchCollection objects are now supported in legends. The feature can be used as follows:

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    fig, axs = plt.subplots()
    p1, p2 = Polygon([[0, 0], [100, 100], [200, 0]]), Polygon([[400, 0], [500, 100], [600, 0]])
    p3, p4 = Polygon([[700, 0], [800, 100], [900, 0]]), Polygon([[1000, 0], [1100, 100], [1200, 0]])
    p = PatchCollection([p1, p2], label="a", facecolors='red', edgecolors='black')
    p2 = PatchCollection([p3, p4], label="ab", color='green')
    axs.add_collection(p, autolim=True)
    axs.add_collection(p2, autolim=True)
    axs.set_xlim(right=1200)
    axs.set_ylim(top=100)
    axs.legend()

    plt.show()

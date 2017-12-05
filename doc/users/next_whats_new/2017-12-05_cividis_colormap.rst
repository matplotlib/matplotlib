Cividis colormap
----------------------------

A new colormap, named cividis, has been optimized so that it looks similar to people with a red-green color vision deficiency. For more information, please see our paper (to be published in 2018).

plot::

    import matplotlib.pyplot
    import numpy as np

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(np.rand.random(32,32), cmap='cividis')
    fig.colorbar(pcm)
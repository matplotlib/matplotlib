Cividis colormap
----------------------------

A new dark blue/yellow colormap named 'cividis' was added. Like viridis, cividis is perceptually uniform and colorblind friendly. However, cividis also goes a step further: not only is it usable by colorblind users, it should actually look effectively identical to colorblind and non-colorblind users. For more details, see Nunez J, Anderton C, and Renslow R. (submitted). Optimizing colormaps with consideration for color vision deficiency to enable accurate interpretation of scientific data."

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(np.random.rand(32,32), cmap='cividis')
    fig.colorbar(pcm)
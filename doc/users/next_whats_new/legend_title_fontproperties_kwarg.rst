Font properties of legend title are configurable
------------------------------------------------

Title's font properties can be set via the *title_fontproperties* keyword
argument, for example:

.. plot::

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(range(10),label='point')
    ax.legend(title='Points', title_fontproperties={'family': 'serif', 'size': 20}) 

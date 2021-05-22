New rcParams for legend: set legend labelcolor globally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new :rc:`legend.labelcolor` sets the default *labelcolor* argument for
`.Figure.legend`.  The special values  'linecolor', 'markerfacecolor'
(or 'mfc'), or 'markeredgecolor' (or 'mec') will cause the legend text to match 
the corresponding color of marker. 


.. plot::

    plt.rcParams['legend.labelcolor'] = 'linecolor'

    # Make some fake data.
    a = np.arange(0, 3, .02)
    c = np.exp(a)
    d = c[::-1]

    fig, ax = plt.subplots()
    ax.plot(a, c, 'g--', label='Model length')
    ax.plot(a, d, 'r:', label='Data length')

    ax.legend()

    plt.show()
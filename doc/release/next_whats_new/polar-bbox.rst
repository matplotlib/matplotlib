Polar Axes now fill their bounding box
--------------------------------------

Polar Axes that were not full circles would previously center themselves in a square box
the size of the full circle. Now they only encompass the area required for the limited
wedge in view. This allows layout engines (constrained or tight) to more compactly pack
such Axes.

For example, the Polar Axes in the below image now expands to fill the entire figure,
when it previously would be centered and approximately 50% as large.

.. plot::

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout='constrained',
                           facecolor='#e8f4f2', subplot_kw={'projection': 'polar'})

    theta_min = 0
    theta_max = 180
    theta = np.linspace(0, np.deg2rad(theta_max), 181)
    r = np.arange(len(theta))
    ax.plot(theta, r)

    ax.set_thetamin(theta_min)
    ax.set_thetamax(theta_max)

    ax.set_xlabel('Magnitude', fontsize=15)
    ax.set_ylabel('Angles', fontsize=15)

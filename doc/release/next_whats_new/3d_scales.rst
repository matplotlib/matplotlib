Non-linear scales on 3D axes
----------------------------

Resolving a long-standing issue, 3D axes now support non-linear axis scales
such as 'log', 'symlog', 'logit', 'asinh', and custom 'function' scales, just
like 2D axes. Use `~.Axes3D.set_xscale`, `~.Axes3D.set_yscale`, and
`~.Axes3D.set_zscale` to set the scale for each axis independently.

.. plot::
    :include-source: true
    :alt: A 3D plot with a linear x-axis, logarithmic y-axis, and symlog z-axis.

    import matplotlib.pyplot as plt
    import numpy as np

    # A sine chirp with increasing frequency and amplitude
    x = np.linspace(0, 1, 400)  # time
    y = 10 ** (2 * x)  # frequency, growing exponentially from 1 to 100 Hz
    phase = 2 * np.pi * (10 ** (2 * x) - 1) / (2 * np.log(10))
    z = np.sin(phase) * x ** 2 * 10  # amplitude, growing quadratically

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('Time (linear)')
    ax.set_ylabel('Frequency, Hz (log)')
    ax.set_zlabel('Amplitude (symlog)')

    ax.set_yscale('log')
    ax.set_zscale('symlog')

    plt.show()

See `matplotlib.scale` for details on all available scales and their parameters.

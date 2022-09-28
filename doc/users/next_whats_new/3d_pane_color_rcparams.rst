rcParam for 3D pane color
-------------------------

The rcParams :rc:`axes3d.xaxis.panecolor`, :rc:`axes3d.yaxis.panecolor`,
:rc:`axes3d.zaxis.panecolor` can be used to change the color of the background
panes in 3D plots. Note that it is often beneficial to give them slightly
different shades to obtain a "3D effect" and to make them slightly transparent
(alpha < 1).

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    with plt.rc_context({'axes3d.xaxis.panecolor': (0.9, 0.0, 0.0, 0.5),
                         'axes3d.yaxis.panecolor': (0.7, 0.0, 0.0, 0.5),
                         'axes3d.zaxis.panecolor': (0.8, 0.0, 0.0, 0.5)}):
        fig = plt.figure()
        fig.add_subplot(projection='3d')

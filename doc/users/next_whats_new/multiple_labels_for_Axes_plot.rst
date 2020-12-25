An iterable object with labels can be passed to `.Axes.plot`
------------------------------------------------------------

When plotting multiple datasets by passing 2D data as *y* value to 
`~.Axes.plot`, labels for the datasets can be passed as a list, the 
length matching the number of columns in *y*.

.. plot::

    import matplotlib.pyplot as plt
    
    x = [1, 2, 3]

    y = [[1, 2],
         [2, 5],
         [4, 9]]

    plt.plot(x, y, label=['low', 'high'])
    plt.legend()

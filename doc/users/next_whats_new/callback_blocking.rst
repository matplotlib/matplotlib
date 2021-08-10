``CallbackRegistry`` objects gain a method to temporarily block signals
-----------------------------------------------------------------------

The context manager `~matplotlib.cbook.CallbackRegistry.blocked` can be used
to block callback signals from being processed by the ``CallbackRegistry``.
The optional keyword, *signal*, can be used to block a specific signal
from being processed and let all other signals pass.

.. code-block::

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])

    # Block all interactivity through the canvas callbacks
    with fig.canvas.callbacks.blocked():
        plt.show()

    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])

    # Only block key press events
    with fig.canvas.callbacks.blocked(signal="key_press_event"):
        plt.show()

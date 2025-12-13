``CallbackRegistry.disconnect_func`` to disconnect callbacks by function
-------------------------------------------------------------------------

`.CallbackRegistry` now has a `~.CallbackRegistry.disconnect_func` method that
allows disconnecting a callback by passing the signal and function directly,
instead of needing to track the callback ID returned by
`~.CallbackRegistry.connect`.

.. code-block:: python

    from matplotlib.cbook import CallbackRegistry

    def my_callback(event):
        print(event)

    callbacks = CallbackRegistry()
    callbacks.connect('my_signal', my_callback)

    # Disconnect by function reference instead of callback ID
    callbacks.disconnect_func('my_signal', my_callback)

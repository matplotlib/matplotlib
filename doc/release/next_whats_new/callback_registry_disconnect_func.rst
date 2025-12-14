``CallbackRegistry.disconnect`` allows directly callbacks by function
-------------------------------------------------------------------------

`.CallbackRegistry` now allows directly passing a function and optionally signal to
`~.CallbackRegistry.disconnect` instead of needing to track the callback ID
returned by `~.CallbackRegistry.connect`.

.. code-block:: python

    from matplotlib.cbook import CallbackRegistry

    def my_callback(event):
        print(event)

    callbacks = CallbackRegistry()
    callbacks.connect('my_signal', my_callback)

    # Disconnect by function reference instead of callback ID
    callbacks.disconnect('my_signal', my_callback)

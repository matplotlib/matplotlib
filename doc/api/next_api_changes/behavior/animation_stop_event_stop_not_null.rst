``TimedAnimation``/``Animation`` no longer set to None when an animation stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously upon animation completion or fig close ``.Animation.event_source``
was set to ``None``.  Now the event source is stopped and retained on
the animation.

The following code snippet would before have returned None but would now return a
``matplotlib.backend_bases.TimerBase``.  Code that used None to check for successful
stoppage should be updated.

.. code-block:: python

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, lambda f: [], frames=3, repeat=False)
    # Wait for the animation to finish.
    print(anim.event_source)

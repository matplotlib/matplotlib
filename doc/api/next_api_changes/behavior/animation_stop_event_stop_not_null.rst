``TimedAnimation``/``Animation`` now perform .stop() on mpl_disconnect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously these classes set ``self.event_source = None`` when
``self._fig.canvas.mpl_disconnect`` was called. Now they run
``self.event_source.stop()``.

This hange is for issue `30622 <https://github.com/matplotlib/matplotlib/issues/30622>`__.

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
    anim._start()
    while anim._step():
        pass
    print(anim.event_source)

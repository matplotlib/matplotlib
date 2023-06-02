Default antialiasing behavior changes for ``Text`` and ``Annotation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``matplotlib.pyplot.annotate()`` and ``matplotlib.pyplot.text()`` now support parameter *antialiased* when initializing.
Examples:

.. code-block::

    mpl.text.Text(.5, .5, "foo\nbar", antialiased=True)
    plt.text(0.5, 0.5, '6 inches x 2 inches', antialiased=True)
    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5), antialiased=False)

See "What's New" for more details on usage.

With this new feature, you may want to make sure that you are creating and saving/showing the figure under the same context::

    # previously this was a no-op, now it is what works
    with rccontext(text.antialiased=False):
        fig, ax = plt.subplots()
        ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5))
        fig.savefig('/tmp/test.png')

    # previously this had an effect, now this is a no-op
    fig, ax = plt.subplots()
    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5))
    with rccontext(text.antialiased=False):
        fig.savefig('/tmp/test.png')

Also note that antialiasing for tick labels will be set with :rc:`text.antialiased` when they are created (usually when a ``Figure`` is created) - This means antialiasing for them can no longer be changed by modifying :rc:`text.antialiased`.

Support customizing antialiasing for text and annotation
--------------------------------------------------------
``matplotlib.pyplot.annotate()`` and ``matplotlib.pyplot.text()`` now support parameter *antialiased*.
When *antialiased* is set to ``True``, antialiasing will be applied to the text.
When *antialiased* is set to ``False``, antialiasing will not be applied to the text.
When *antialiased* is not specified, antialiasing will be set by :rc:`text.antialiased` at the creation time of ``Text`` and ``Annotation`` object.
Examples:

.. code-block::

    mpl.text.Text(.5, .5, "foo\nbar", antialiased=True)
    plt.text(0.5, 0.5, '6 inches x 2 inches', antialiased=True)
    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5), antialiased=False)

If the text contains math expression, *antialiased* applies to the whole text.
Examples:

.. code-block::

    # no part will be antialiased for the text below
    plt.text(0.5, 0.25, r"$I'm \sqrt{x}$", antialiased=False)

Also note that antialiasing for tick labels will be set with :rc:`text.antialiased` when they are created (usually when a ``Figure`` is created) and cannot be changed afterwards.

Furthermore, with this new feature, you may want to make sure that you are creating and saving/showing the figure under the same context::

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

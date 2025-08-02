Figures can be attached to and removed from pyplot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Figures can now be attached to and removed from management through pyplot, which in
the background also means a less strict coupling to backends.

In particular, standalone figures (created with the `.Figure` constructor) can now be
registered with the `.pyplot` module by calling ``plt.figure(fig)``. This allows to
show them with ``plt.show()`` as you would do with any figure created with pyplot
factory methods such as ``plt.figure()`` or ``plt.subplots()``.

When closing a shown figure window, the related figure is reset to the standalone
state, i.e. it's not visible to pyplot anymore, but if you still hold a reference
to it, you can continue to work with it (e.g. do ``fig.savefig()``, or re-add it
to pyplot with ``plt.figure(fig)`` and then show it again).

The following is now possible - though the example is exaggerated to show what's
possible. In practice, you'll stick with much simpler versions for better
consistency ::

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # Create a standalone figure
    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([1, 2, 3], [4, 5, 6])

    # Register it with pyplot
    plt.figure(fig)

    # Modify the figure through pyplot
    plt.xlabel("x label")

    # Show the figure
    plt.show()

    # Close the figure window through the GUI

    # Continue to work on the figure
    fig.savefig("my_figure.png")
    ax.set_ylabel("y label")

    # Re-register the figure and show it again
    plt.figure(fig)
    plt.show()

Technical detail: Standalone figures use `.FigureCanvasBase` as canvas. This is
replaced by a backend-dependent subclass when registering with pyplot, and is
reset to `.FigureCanvasBase` when the figure is closed. `.Figure.savefig` uses
the current canvas to save the figure (if possible). Since `.FigureCanvasBase`
is Agg-based any Agg-based backend will create the same file output. There may
be slight differences for non-Agg backends; e.g. if you use "GTK4Cairo" as
interactive backend, ``fig.savefig("file.png")`` may create a slightly different
image depending on whether the figure is registered with pyplot or not. In
general, you should not store a reference to the canvas, but rather always
obtain it from the figure with ``fig.canvas``. This will return the current
canvas, which is either the original `.FigureCanvasBase` or a backend-dependent
subclass, depending on whether the figure is registered with pyplot or not.

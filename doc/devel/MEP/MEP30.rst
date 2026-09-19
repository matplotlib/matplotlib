.. _MEP30:

===============================================================
MEP30: Figure-level Overlay Architecture with Layered Rendering
===============================================================

.. contents::
   :local:

Status
======

**Progress** - Implementation work is currently ongoing.

Branches and Pull requests
==========================

* Implementation PR: `#32199 <https://github.com/matplotlib/matplotlib/pull/32199>`_
* Alternative exploration PR: `#32002 <https://github.com/matplotlib/matplotlib/pull/32002>`_

Abstract
========

Currently, any interactive element in Matplotlib (such as the ``Cursor`` widget
that draws a crosshair) must be rendered as part of the main figure. This means
that even a tiny mouse movement triggers a full redraw of the entire figure,
including all artists, data, labels, and ticks. As a result, simple interactive
tools like crosshairs feel noticeably laggy over heavy data plots.

This MEP introduces a Figure-level layered architecture that allows interactive
overlay artists to be isolated into a separate layer, allowing them to be
redrawn independently from the heavy base content.

Detailed description
====================

When a user moves their mouse over a scatter plot with 1,000,000 data points,
the ``Cursor`` widget updates its crosshair lines. This triggers a stale callback
that marks the entire figure as needing a redraw. Matplotlib then redraws
everything: the million scatter points, the axes, the labels, the ticks, and
finally the two thin crosshair lines.

Usage Example
-------------

The following example moves a ``Cursor`` widget's crosshair lines into the
overlay layer. With 1,000,000 scatter points sitting in the base layer, moving
the mouse is smooth because only the two cursor lines are redrawn on each
mouse event.

.. code-block:: python

    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Cursor

    x = np.random.normal(5, 2, 1000000)
    y = np.random.normal(5, 2, 1000000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.1, color='blue')

    cursor = Cursor(ax, color='red', linewidth=1)

    # Remove the cursor lines from the axes (default base layer)
    cursor.lineh.remove()
    cursor.linev.remove()

    # Re-add them into the overlay layer
    fig.add_artist(cursor.lineh, layer="overlay")
    fig.add_artist(cursor.linev, layer="overlay")

    plt.show()

Implementation
==============

In this approach, layer membership is managed directly by the ``Figure``.

* ``Figure`` keeps track of which artists belong to each layer using an internal
  dictionary (``_children_by_layer``).
* Individual artists do not know anything about layers.
* Artists can be added to a specific layer using the layer argument:
  ``fig.add_artist(line, layer="overlay")``.
* Each layer is rendered into its own ``RendererAgg`` buffer (in QtAgg).

Stale Tracking & Drawing Lifecycle
----------------------------------

Stale state tracking is managed per layer: ``_stale_layers[layer_name]`` is set to
``True`` only when an artist belonging to that specific layer becomes stale.

During a render pass in the QtAgg backend, the canvas checks each layer's stale
status. If a layer is marked dirty (or if the canvas was resized),
``fig._draw_layer()`` re-renders only that layer into its dedicated ``RendererAgg``
buffer and immediately resets ``_stale_layers[layer_name] = False``.

Once all stale layer buffers are updated, ``draw()`` clears the top-level figure
staleness (``fig.stale = False``) before calling ``self.update()`` to schedule Qt's
``paintEvent()``. During ``paintEvent()``, ``QPainter`` composites the clean layer
buffers sequentially onto the screen.

As a result, moving an interactive cursor flags only ``_stale_layers["overlay"] = True``,
leaving the complex base plot cached and completely untouched in memory.

Backward compatibility
======================

* **Public API:** The new ``layer`` argument in ``add_artist()`` defaults to ``None``,
  ensuring all unassigned artists are safely routed to the "base" layer. Similarly,
  calling ``get_children()`` without arguments continues to return every artist in
  the figure across all layers.
* **Backend Compatibility:** Fully backward compatible. For backends that do not
  support multi-pass layer caching (like standard PDF, SVG, PNG, or non-Qt
  backends), ``Figure.draw()`` simply renders each layer sequentially one after
  another.

Performance Trade-offs
----------------------

* **Interactive performance:** For interactive elements, only the affected layer needs
  to be rendered. if the cursor lives in a separate layer, only that layer needs to
  be redrawn when the user moves the mouse. The base layer (with the heavy scatter data)
  is cached in ``RendererAgg`` buffer unchanged.

* **Resize and zoom performance:** These operations are slightly slower than before.
  When the window is resized, every layer has to be fully redrawn and each layer
  requires its own ``RendererAgg`` buffer. Where the old code had one buffer, the new
  code has one per layer.

Alternatives
============

Artist-level flag (``in_overlay``)
---------------------------------------------

Explored in `PR #32002 <https://github.com/matplotlib/matplotlib/pull/32002>`_.

* Each overlay artist has ``in_overlay = True``.
* The ``Figure`` does not know about overlay. All artists remain in the same
  children list.
* When an overlay artist becomes stale, it bypasses the normal stale callback and
  directly calls ``canvas.draw_overlay()``. Because the ``Figure`` has no concept of
  overlay, it cannot mark a specific layer as stale — so the interception has to
  happen inside the ``Artist`` itself.
* During drawing, the backend has to search through the figure using ``figure.findobj()``
  and identify artists for which ``get_in_overlay()`` returns ``True``.

This alternative was rejected because it does not provide proper layer separation.
The backend canvas classes have to look up individual artist attributes through
``findobj()`` traversal of the entire scene graph.

Figure-managed layers ensures layer isolation. Figure is responsible for
maintaining the layer registry and staleness status, making it possible for
backends to operate as lightweight execution engines rendering isolated layer buffers.

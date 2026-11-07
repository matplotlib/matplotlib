.. _MEP30:

===============================================================
MEP30: Figure-level layer Architecture with Layered Rendering
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

This MEP proposes an architecture that supports arbitrary Figure-level rendering
layers. This allows interactive artists to be isolated into separate
layers, enabling them to be redrawn independently from the heavy base content.

Detailed description
====================

Currently, when a user moves their mouse over a scatter plot with 1,000,000
data points, the ``Cursor`` widget updates its crosshair lines. This triggers
a stale callback that marks the entire figure as needing a redraw, forcing
Matplotlib to slowly redraw everything from scratch.

This architecture solves this by grouping artists into completely arbitrary,
string-named layers (e.g., ``"patch"``, ``"base"``, ``"widgets"`` or any other
user-defined name).

**Why Arbitrary Layers?**
Instead of limiting Matplotlib to a strict "base" and "overlay" layer, this
architecture lets users create any number of custom layers simply by naming them.

By adding different interactive widgets to different layers, developers
can completely isolate their redraw cycles. When a user interacts with one widget,
only that widget's layer is redrawn. The other widgets and the heavy background plot
all remain perfectly cached, improving the application's framerate
and responsiveness.

Usage Example
-------------

For a complete demonstration of this architecture in action, see the
``galleries/examples/widgets/cursor_layers.py`` gallery example.

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

* **Public API:** The new ``layer`` argument in ``Figure.add_artist()`` defaults to ``None``,
  ensuring all unassigned artists are safely routed to the ``"base"`` layer.
  Additionally, calling ``get_children()`` without arguments continues to return
  every artist in the figure across all layers.
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

Future Work
===========

Coexistence of Layers and Blitting
----------------------------------
In the current architecture, if a backend supports layers (``supports_layers = True``)
and a widget is initialized with ``useblit=True``, the widget automatically drops
blitting and opts to use the layer system instead.

This is because the two approaches use completely different rendering buffers.
The layer system draws each layer into its own dedicated ``RendererAgg`` buffer
(stored in ``_layer_renderers``). Blitting, on the other hand, draws
dynamic artists into the single ``self.renderer`` buffer.

The approach used in the QtAgg backend first loops over ``_layer_renderers``
and paints each layer to the screen, and then checks if ``self.renderer`` is present.
If yes then alpha-blends it on top as a final step. This means blitting is done on an
transparent renderer buffer which get added to the top of the drawn layers.

To avoid this, blitting is currently disabled when the backend supports layers.

In the future, a clean way must be found for the layer system and
blitting to work together.

Updating Other Built-in Widgets to Support Layers
-------------------------------------------------
In the current architecture, only the ``Cursor`` widget has been updated to
support being drawn in a separate layer.

In the future, all other built-in interactive widgets
(like ``SpanSelector``, ``RectangleSelector``, and ``LassoSelector``) should
be updated to support layers, allowing them to draw in a separate layer
when the backend supports layers.

Expanding Layer Caching to All Interactive Backends
-----------------------------------------------------
Currently, the performance boost from rendering isolated layer buffers is only
implemented in the QtAgg backend. While the layered drawing sequence works across
all backends, the other interactive backends (like ``TkAgg``, ``GTKAgg``,
``MacOSX``, and ``WebAgg``) still fall back to redrawing everything.
A key next step will be porting this per-layer buffering system to the other
interactive backends so all users get the same interactive speedup.

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

Removal of deprecated APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following deprecated APIs have been removed.  Unless a replacement is stated, please
vendor the previous implementation if needed.

- The following methods of `.FigureCanvasBase`: ``pick`` (use ``Figure.pick`` instead),
  ``resize``, ``draw_event``, ``resize_event``, ``close_event``, ``key_press_event``,
  ``key_release_event``, ``pick_event``, ``scroll_event``, ``button_press_event``,
  ``button_release_event``, ``motion_notify_event``, ``leave_notify_event``,
  ``enter_notify_event`` (for all the ``foo_event`` methods, construct the relevant
  `.Event` object and call ``canvas.callbacks.process(event.name, event)`` instead).
- ``ToolBase.destroy`` (connect to ``tool_removed_event`` instead).
- The *cleared* parameter to `.FigureCanvasAgg.get_renderer` (call ``renderer.clear()``
  instead).
- The following methods of `.RendererCairo`: ``set_ctx_from_surface`` and
  ``set_width_height`` (use ``set_context`` instead, which automatically infers the
  canvas size).
- The ``window`` or ``win`` parameters and/or attributes of ``NavigationToolbar2Tk``,
  ``NavigationToolbar2GTK3``, and ``NavigationToolbar2GTK4``, and the ``lastrect``
  attribute of ``NavigationToolbar2Tk``
- The ``error_msg_gtk`` function and the ``icon_filename`` and ``window_icon`` globals
  in ``backend_gtk3``; the ``error_msg_wx`` function in ``backend_wx``.
- ``FigureManagerGTK3Agg`` and ``FigureManagerGTK4Agg`` (use ``FigureManagerGTK3``
  instead); ``RendererGTK3Cairo`` and ``RendererGTK4Cairo``.
- ``NavigationToolbar2Mac.prepare_configure_subplots`` (use
  `~.NavigationToolbar2.configure_subplots` instead).
- ``FigureManagerMac.close``.
- The ``qApp`` global in `.backend_qt` (use ``QtWidgets.QApplication.instance()``
  instead).
- The ``offset_text_height`` method of ``RendererWx``; the ``sizer``, ``figmgr``,
  ``num``, ``toolbar``, ``toolmanager``, ``get_canvas``, and ``get_figure_manager``
  attributes or methods of ``FigureFrameWx`` (use ``frame.GetSizer()``,
  ``frame.canvas.manager``, ``frame.canvas.manager.num``, ``frame.GetToolBar()``,
  ``frame.canvas.manager.toolmanager``, the *canvas_class* constructor parameter, and
  ``frame.canvas.manager``, respectively, instead).
- ``FigureFrameWxAgg`` and ``FigureFrameWxCairo`` (use
  ``FigureFrameWx(..., canvas_class=FigureCanvasWxAgg)`` and
  ``FigureFrameWx(..., canvas_class=FigureCanvasWxCairo)``, respectively, instead).
- The ``filled`` attribute and the ``draw_all`` method of `.Colorbar` (instead of
  ``draw_all``, use ``figure.draw_without_rendering``).
- Calling `.MarkerStyle` without setting the *marker* parameter or setting it to None
  (use ``MarkerStyle("")`` instead).
- Support for third-party canvas classes without a ``required_interactive_framework``
  attribute (this can only occur if the canvas class does not inherit from
  `.FigureCanvasBase`).
- The ``canvas`` and ``background`` attributes of `.MultiCursor`; the
  ``state_modifier_keys`` attribute of selector widgets.
- Passing *useblit*, *horizOn*, or *vertOn* positionally to `.MultiCursor`.
- Support for the ``seaborn-<foo>`` styles; use ``seaborn-v0_8-<foo>`` instead, or
  directly use the seaborn API.

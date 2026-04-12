*********************
``matplotlib.typing``
*********************

.. automodule:: matplotlib.typing
   :no-members:
   :no-undoc-members:

.. types are written out explicitly as `.. autodata::` directives, so that we
   can meaningfully group them in sections.
   test_typing.py::test_typing_aliases_documented ensures that the documented
   types are in sync with the actual types defined in matplotlib.typing.

Color
=====

.. autodata:: matplotlib.typing.ColorType
.. autodata:: matplotlib.typing.RGBColorType
.. autodata:: matplotlib.typing.RGBAColorType
.. autodata:: matplotlib.typing.ColourType
.. autodata:: matplotlib.typing.RGBColourType
.. autodata:: matplotlib.typing.RGBAColourType

Artist styles
=============

.. autodata:: matplotlib.typing.LineStyleType
.. autodata:: matplotlib.typing.DrawStyleType
.. autodata:: matplotlib.typing.MarkEveryType
.. autodata:: matplotlib.typing.MarkerType
.. autodata:: matplotlib.typing.FillStyleType
.. autodata:: matplotlib.typing.CapStyleType
.. autodata:: matplotlib.typing.JoinStyleType

Events
======

.. autodata:: matplotlib.typing.MouseEventType
.. autodata:: matplotlib.typing.KeyEventType
.. autodata:: matplotlib.typing.DrawEventType
.. autodata:: matplotlib.typing.PickEventType
.. autodata:: matplotlib.typing.ResizeEventType
.. autodata:: matplotlib.typing.CloseEventType
.. autodata:: matplotlib.typing.EventType

rcParams and stylesheets
========================

.. autodata:: matplotlib.typing.RcKeyType
.. autodata:: matplotlib.typing.RcGroupKeyType
.. autodata:: matplotlib.typing.RcStyleType

Other types
===========

.. autodata:: matplotlib.typing.CoordsType
.. autodata:: matplotlib.typing.LegendLocType
.. autodata:: matplotlib.typing.LogLevel
.. autodata:: matplotlib.typing.HashableList


.. intentionally undocumented types (one type per row)
   CoordsBaseType

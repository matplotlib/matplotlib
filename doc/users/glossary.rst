========
Glossary
========

.. Note for glossary authors:
   The glossary is primarily intended for Matplotlib's own concepts and
   terminology, e.g. figure, artist, backend, etc. We don't want to list
   general terms like "GUI framework", "event loop" or similar.
   The glossary should contain a short definition of the term, aiming at
   a high-level understanding. Use links to redirect to more comprehensive
   explanations and API reference when possible.

This glossary defines concepts and terminology specific to Matplotlib.

.. glossary::

    Figure
        The outermost container for a Matplotlib graphic. Think of this as the
        sheet of paper to draw on.

        This is impolemented in the class `.Figure`. For more details see
        :ref:`figure-intro`.

    Axes
        This is what is often colloquially called "a plot". A data area with
        :term:`Axis`\ es, i.e. coordinate directions. This includes also
        decoration like title, axis labels, legend and data artists
        like lines, bars etc.

        Since most "plotting operations" are realized as methods on `~.axes.Axes`
        this is the object users will mostly interact with.

        Note: The term *Axes* was taken over from MATLAB. Think of this as
        A container spanned by the *x*- and *y*-axis, including decoration
        and data.

    Axis
        A direction with a scale. The scale defines the mapping from
        data coordinates to screen coordinates. The Axis also includes
        the ticks and axis label.

    Artist
        The base class for all graphical element that can be drawn.
        Examples are Lines, Rectangles, Text, Ticks, Lengend, Axes, ...

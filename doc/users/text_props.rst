.. _text-properties:

Text properties and layout
==========================

The :class:`matplotlib.text.Text` instances have a variety of
properties which can be configured via keyword arguments to the text
commands (e.g., :func:`~matplotlib.pyplot.title`,
:func:`~matplotlib.pyplot.xlabel` and :func:`~matplotlib.pyplot.text`).

==========================  ==============================================================================
Property                    Value Type
==========================  ==============================================================================
alpha			    float
backgroundcolor		    any matplotlib color
bbox			    rectangle prop dict plus key ``'pad'`` which is a pad in points
clip_box		    a matplotlib.transform.Bbox instance
clip_on			    [True | False]
clip_path		    a Path instance and a Transform instance, a Patch
color			    any matplotlib color
family			    [ ``'serif'`` | ``'sans-serif'`` | ``'cursive'`` | ``'fantasy'`` | ``'monospace'`` ]
fontproperties		    a matplotlib.font_manager.FontProperties instance
horizontalalignment or ha   [ ``'center'`` | ``'right'`` | ``'left'`` ]
label			    any string
linespacing		    float
multialignment		    [``'left'`` | ``'right'`` | ``'center'`` ]
name or fontname	    string e.g., [``'Sans'`` | ``'Courier'`` | ``'Helvetica'`` ...]
picker			    [None|float|boolean|callable]
position		    (x,y)
rotation		    [ angle in degrees ``'vertical'`` | ``'horizontal'``
size or fontsize	    [ size in points | relative size, e.g., ``'smaller'``, ``'x-large'`` ]
style or fontstyle	    [ ``'normal'`` | ``'italic'`` | ``'oblique'``]
text			    string or anything printable with '%s' conversion
transform		    a matplotlib.transform transformation instance
variant			    [ ``'normal'`` | ``'small-caps'`` ]
verticalalignment or va	    [ ``'center'`` | ``'top'`` | ``'bottom'`` | ``'baseline'`` ]
visible			    [True | False]
weight or fontweight	    [ ``'normal'`` | ``'bold'`` | ``'heavy'`` | ``'light'`` | ``'ultrabold'`` | ``'ultralight'``]
x			    float
y			    float
zorder			    any number
==========================  ==============================================================================


You can layout text with the alignment arguments
``horizontalalignment``, ``verticalalignment``, and
``multialignment``.  ``horizontalalignment`` controls whether the x
positional argument for the text indicates the left, center or right
side of the text bounding box. ``verticalalignment`` controls whether
the y positional argument for the text indicates the bottom, center or
top side of the text bounding box.  ``multialignment``, for newline
separated strings only, controls whether the different lines are left,
center or right justified.  Here is an example which uses the
:func:`~matplotlib.pyplot.text` command to show the various alignment
possibilities.  The use of ``transform=ax.transAxes`` throughout the
code indicates that the coordinates are given relative to the axes
bounding box, with 0,0 being the lower left of the axes and 1,1 the
upper right.

.. plot:: pyplots/text_layout.py
   :include-source:

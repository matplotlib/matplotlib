.. _whats-new:

==========================
 What's new in matplotlib
==========================

For a list of all of the issues and pull requests since the last
revision, see the :ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 3



New in matplotlib 2.0
=====================

.. note::

   matplotlib 2.0 supports Python 2.7, and 3.4+



Default style changes
---------------------

The major changes in v2.0 are related to overhauling the default styles.

.. toctree::
   :maxdepth: 2

   dflt_style_changes


Improved color conversion API and RGBA support
----------------------------------------------

The :mod:`~matplotlib.colors` gained a new color conversion API with
full support for the alpha channel.  The main public functions are
:func:`~matplotlib.colors.is_color_like`, :func:`matplotlib.colors.to_rgba`,
:func:`matplotlib.colors.to_rgba_array` and :func:`~matplotlib.colors.to_hex`.
RGBA quadruplets are encoded in hex format as `#rrggbbaa`.

A side benefit is that the Qt options editor now allows setting the alpha
channel of the artists as well.


New Configuration (rcParams)
----------------------------

New rcparams added

+---------------------------------+--------------------------------------------------+
| Parameter                       | Description                                      |
+=================================+==================================================+
|`date.autoformatter.year`        | foramt string for 'year' scale dates             |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.month`       | format string for 'month' scale dates            |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.day`         | format string for 'day' scale dates              |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.hour`        | format string for 'hour' scale times             |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.minute`      | format string for 'minute' scale times           |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.second`      | format string for 'second' scale times           |
+---------------------------------+--------------------------------------------------+
|`date.autoformatter.microsecond` | format string for 'microsecond' scale times      |
+---------------------------------+--------------------------------------------------+
|`scatter.marker`                 | default marker for scatter plot                  |
+---------------------------------+--------------------------------------------------+
|`svg.hashsalt`                   | see note                                         |
+---------------------------------+--------------------------------------------------+
|`xtick.top`, `xtick.minor.top`,  | Control where major and minor ticks are drawn.   |
|`xtick.major.top`                | The global values are `and` ed with the          |
|`xtick.bottom`,                  | corosponding major/minor values.                 |
|`xtick.minor.bottom`,            |                                                  |
|`xtick.major.bottom`             |                                                  |
|`ytick.left`, `ytick.minor.left`,|                                                  |
|`ytick.major.left`               |                                                  |
|`ytick.right`,                   |                                                  |
|`ytick.minor.right`,             |                                                  |
|`ytick.major.right`              |                                                  |
+---------------------------------+--------------------------------------------------+
|`hist.bins`                      | the default number of bins to use in             |
|                                 | `~matplotlib.axes.Axes.hist`.  This can be an    |
|                                 | `int`, a list of floats, or ``'auto'`` if numpy  |
|                                 | >= 1.11 is installed.                            |
+---------------------------------+--------------------------------------------------+
|`lines.scale_dashes`             | If the line dash patterns should scale with      |
|                                 | linewidth                                        |
+---------------------------------+--------------------------------------------------+



Added ``svg.hashsalt`` key to rcParams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``svg.hashsalt`` is ``None`` (which it is by default), the svg
backend uses ``uuid4`` to generate the hash salt.  If it is not
``None``, it must be a string that is used as the hash salt instead of
``uuid4``.  This allows for deterministic SVG output.


Removed the ``svg.image_noscale`` rcParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a result of the extensive changes to image handling, the
``svg.image_noscale`` rcParam has been removed.  The same
functionality may be achieved by setting ``interpolation='none'`` on
individual images or globally using the ``image.interpolation``
rcParam.


Qualitative colormaps
---------------------

ColorBrewer's "qualitative" colormaps ("Accent", "Dark2", "Paired",
"Pastel1", "Pastel2", "Set1", "Set2", "Set3") were intended for discrete
categorical data, with no implication of value, and therefore have been
converted to ``ListedColormap`` instead of ``LinearSegmentedColormap``, so
the colors will no longer be interpolated and they can be used for
choropleths, labeled image features, etc.



Axis offset label now responds to `labelcolor`
----------------------------------------------

Axis offset labels are now colored the same as axis tick markers when `labelcolor` is altered.

Improved offset text choice
---------------------------
The default offset-text choice was changed to only use significant digits that
are common to all ticks (e.g. 1231..1239 -> 1230, instead of 1231), except when
they straddle a relatively large multiple of a power of ten, in which case that
multiple is chosen (e.g. 1999..2001->2000).


Style parameter blacklist
-------------------------

In order to prevent unexpected consequences from using a style, style
files are no longer able to set parameters that affect things
unrelated to style.  These parameters include::

  'interactive', 'backend', 'backend.qt4', 'webagg.port',
  'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
  'toolbar', 'timezone', 'datapath', 'figure.max_open_warning',
  'savefig.directory', 'tk.window_focus', 'docstring.hardcopy'


Change in default font
----------------------

The default font used by matplotlib in text has been changed to DejaVu Sans and
DejaVu Serif for the sans-serif and serif families, respectively. The DejaVu
font family is based on the previous matplotlib default --Bitstream Vera-- but
includes a much wider range of characters.

The default mathtext font has been changed from Computer Modern to the DejaVu
family to maintain consistency with regular text. Two new options for the
``mathtext.fontset`` configuration parameter have been added: ``dejavusans``
(default) and ``dejavuserif``. Both of these options use DejaVu glyphs whenever
possible and fall back to STIX symbols when a glyph is not found in DejaVu. To
return to the previous behavior, set the rcParam ``mathtext.fontset`` to ``cm``.


Faster text rendering
---------------------

Rendering text in the Agg backend is now less fuzzy and about 20%
faster to draw.


Improvements for the Qt figure options editor
---------------------------------------------

Various usability improvements were implemented for the Qt figure options
editor, among which:

- Line style entries are now sorted without duplicates.
- The colormap and normalization limits can now be set for images.
- Line edits for floating values now display only as many digits as necessary
  to avoid precision loss.  An important bug was also fixed regarding input
  validation using Qt5 and a locale where the decimal separator is ",".
- The axes selector now uses shorter, more user-friendly names for axes, and
  does not crash if there are no axes.
- Line and image entries using the default labels ("_lineX", "_imageX") are now
  sorted numerically even when there are more than 10 entries.


Improved image support
----------------------

Prior to version 2.0, matplotlib resampled images by first applying
the color map and then resizing the result.  Since the resampling was
performed on the colored image, this introduced colors in the output
image that didn't actually exist in the color map.  Now, images are
resampled first (and entirely in floating-point, if the input image is
floating-point), and then the color map is applied.

In order to make this important change, the image handling code was
almost entirely rewritten.  As a side effect, image resampling uses
less memory and fewer datatype conversions than before.

The experimental private feature where one could "skew" an image by
setting the private member ``_image_skew_coordinate`` has been
removed.  Instead, images will obey the transform of the axes on which
they are drawn.


Support for HiDPI (Retina) displays in the NbAgg and WebAgg backends
--------------------------------------------------------------------

The NbAgg and WebAgg backends will now use the full resolution of your
high-pixel-density display.

Change in the default animation codec
-------------------------------------

The default animation codec has been changed from ``mpeg4`` to ``h264``,
which is more efficient. It can be set via the ``animation.codec`` rcParam.

Deprecated support for mencoder in animation
--------------------------------------------

The use of mencoder for writing video files with mpl is problematic;
switching to ffmpeg is strongly advised.  All support for mencoder
will be removed in version 2.2.


Previous Whats New
==================

.. toctree::
   :glob:
   :maxdepth: 1

   prev_whats_new/whats_new_*

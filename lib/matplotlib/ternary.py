#!/usr/bin/env python
"""Class to create a ternary plot using matplotlib projections
"""
__author__ = "Kevin L. Davies"
__version__ = "2011/10/12"
__license__ = "BSD"

import matplotlib.pyplot as plt # **Is this acceptable?

from matplotlib import rcParams

# **To do:
#   1. Clean up the procedure for setting the colorbar's location.
#   2. Support all of the applicable axes methods
#      (http://matplotlib.sourceforge.net/api/axes_api.html).

class Ternary():
    """Create and manage a set of ternary axes.

    This class is provided for convenience.  It creates all three axes at once
    and allows applicable axes methods to be called from the group.
    """
    def set_title(self, label, fontdict=None, **kwargs):
        """call signature::

          set_title(label, fontdict=None, **kwargs):

        Set the title for the set of ternary axes.

        kwargs are Text properties:
        %(Text)s

        ACCEPTS: str

        .. seealso::

            :meth:`text`
                for information on how override and the optional args work
        """
        self.ab.set_title(label, fontdict=None, **kwargs)

    def set_xticks(self, ticks, minor=False):
        """
        Set the x ticks of the ternary axes with list of *ticks*

        ACCEPTS: sequence of floats
        """
        self.ab.set_xticks(ticks, minor=False)
        # The _sharex system (through twinx()) propagates this to all 3 axes.

    def grid(self, b=None, which='major', axis='both', **kwargs):
        """call signature::

           grid(self, b=None, which='major', axis='both', **kwargs)

        For each of the ternary axes, set the axes grids on or off; *b* is a
        boolean.  (For MATLAB compatibility, *b* may also be a string, 'on' or
        'off'.)

        If *b* is *None* and ``len(kwargs)==0``, toggle the grid state.  If
        *kwargs* are supplied, it is assumed that you want a grid and *b*
        is thus set to *True*.

        *which* can be 'major' (default), 'minor', or 'both' to control
        whether major tick grids, minor tick grids, or both are affected.

        *axis* can be 'both' (default), 'x', or 'y' to control which
        set of gridlines are drawn.

        *kawrgs* are used to set the grid line properties, eg::

           ax.grid(color='r', linestyle='-', linewidth=2)

        Valid :class:`~matplotlib.lines.Line2D` kwargs are

        %(Line2D)s
        """
        self.ab.grid(b=b, which=which, axis=axis, **kwargs)
        self.bc.grid(b=b, which=which, axis=axis, **kwargs)
        self.ca.grid(b=b, which=which, axis=axis, **kwargs)

    def set_axis_bgcolor(self, color):
        """
        set the axes background color

        ACCEPTS: any matplotlib color - see
        :func:`~matplotlib.pyplot.colors`
        """
        self.ab.set_axis_bgcolor(self, color)

    def legend(self, *args, **kwargs):
        """call signature::

          legend(*args, **kwargs)

        Place a legend on the ternary axes at location *loc*.  Labels are a
        sequence of strings and *loc* can be a string or an integer specifying
        the legend location.

        To make a legend with existing lines::

          legend()

        :meth:`legend` by itself will try and build a legend using the label
        property of the lines/patches/collections.  You can set the label of
        a line by doing::

          plot(x, y, label='my data')

        or::

          line.set_label('my data').

        If label is set to '_nolegend_', the item will not be shown in
        legend.

        To automatically generate the legend from labels::

          legend( ('label1', 'label2', 'label3') )

        To make a legend for a list of lines and labels::

          legend( (line1, line2, line3), ('label1', 'label2', 'label3') )

        To make a legend at a given location, using a location argument::

          legend( ('label1', 'label2', 'label3'), loc='upper left')

        or::

          legend( (line1, line2, line3),  ('label1', 'label2', 'label3'), loc=2)

        The location codes are

          ===============   =============
          Location String   Location Code
          ===============   =============
          'best'            0
          'upper right'     1
          'upper left'      2
          'lower left'      3
          'lower right'     4
          'right'           5
          'center left'     6
          'center right'    7
          'lower center'    8
          'upper center'    9
          'center'          10
          ===============   =============


        Users can specify any arbitrary location for the legend using the
        *bbox_to_anchor* keyword argument. bbox_to_anchor can be an instance
        of BboxBase(or its derivatives) or a tuple of 2 or 4 floats.
        For example,

          loc = 'upper right', bbox_to_anchor = (0.5, 0.5)

        will place the legend so that the upper right corner of the legend at
        the center of the axes.

        The legend location can be specified in other coordinate, by using the
        *bbox_transform* keyword.

        The loc itslef can be a 2-tuple giving x,y of the lower-left corner of
        the legend in axes coords (*bbox_to_anchor* is ignored).


        Keyword arguments:

          *prop*: [ None | FontProperties | dict ]
            A :class:`matplotlib.font_manager.FontProperties`
            instance. If *prop* is a dictionary, a new instance will be
            created with *prop*. If *None*, use rc settings.

          *numpoints*: integer
            The number of points in the legend for line

          *scatterpoints*: integer
            The number of points in the legend for scatter plot

          *scatteroffsets*: list of floats
            a list of yoffsets for scatter symbols in legend

          *markerscale*: [ None | scalar ]
            The relative size of legend markers vs. original. If *None*, use rc
            settings.

          *frameon*: [ True | False ]
            if True, draw a frame around the legend.
            The default is set by the rcParam 'legend.frameon'

          *fancybox*: [ None | False | True ]
            if True, draw a frame with a round fancybox.  If None, use rc

          *shadow*: [ None | False | True ]
            If *True*, draw a shadow behind legend. If *None*, use rc settings.

          *ncol* : integer
            number of columns. default is 1

          *mode* : [ "expand" | None ]
            if mode is "expand", the legend will be horizontally expanded
            to fill the axes area (or *bbox_to_anchor*)

          *bbox_to_anchor* : an instance of BboxBase or a tuple of 2 or 4 floats
            the bbox that the legend will be anchored.

          *bbox_transform* : [ an instance of Transform | None ]
            the transform for the bbox. transAxes if None.

          *title* : string
            the legend title

        Padding and spacing between various elements use following
        keywords parameters. These values are measure in font-size
        units. E.g., a fontsize of 10 points and a handlelength=5
        implies a handlelength of 50 points.  Values from rcParams
        will be used if None.

        ================   ==================================================================
        Keyword            Description
        ================   ==================================================================
        borderpad          the fractional whitespace inside the legend border
        labelspacing       the vertical space between the legend entries
        handlelength       the length of the legend handles
        handletextpad      the pad between the legend handle and text
        borderaxespad      the pad between the axes and legend border
        columnspacing      the spacing between columns
        ================   ==================================================================

        .. Note:: Not all kinds of artist are supported by the legend command.
                  See LINK (FIXME) for details.


        **Example:**

        .. plot:: mpl_examples/api/legend_demo.py

        Also see :ref:`plotting-guide-legend`.
        """
        # Combine the legend entries from all three axes.
        ab_legend_handles_labels = self.ab.get_legend_handles_labels()
        bc_legend_handles_labels = self.bc.get_legend_handles_labels()
        ca_legend_handles_labels = self.ca.get_legend_handles_labels()
        lines = (ab_legend_handles_labels[0]
                 + bc_legend_handles_labels[0]
                 + ca_legend_handles_labels[0])
        labels = (ab_legend_handles_labels[1]
                  + bc_legend_handles_labels[1]
                  + ca_legend_handles_labels[1])
        self.ab.legend(lines, labels)

    def colorbar(self, mappable, cax=None, ax=None, **kwargs):
        """Create a colorbar for a ScalarMappable instance.

        Documentation for the pylab thin wrapper:
        %(colorbar_doc)s
        """
        pad = kwargs.pop('pad', 0.1)
        shrink = kwargs.pop('shrink', 1.0)
        fraction = kwargs.pop('fraction', 0.04)
        # This is a hack and the alignment isnt quite right. **Clean it up.
        scaley = rcParams['figure.subplot.top'] - rcParams['figure.subplot.bottom']
        if cax is None:
            cax = self.ab.figure.add_axes([0.74 + pad,
                                          rcParams['figure.subplot.bottom'] + self.ab.elevation,
                                          fraction,
                                          self.ab.height*scaley*shrink - 0.005])
        return self.ab.figure.colorbar(mappable, cax=cax, ax=ax, **kwargs)
#        return self.figure.colorbar(shrink=shrink, pad=pad, *args, **kwargs)

    def __init__(self, ax=None):
        if  ax is None:
            fig = plt.figure()
            self.ab = fig.add_subplot(111, projection='ternaryab')
        else:
            self.ab = ax
        self.bc = self.ab.twinx(projection='ternarybc')
        self.ca = self.ab.twinx(projection='ternaryca')

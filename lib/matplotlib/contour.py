"""
These are classes to support contour plotting and labelling for the Axes class.
"""

from numbers import Integral

import numpy as np
from numpy import ma

import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.text as text
import matplotlib.cbook as cbook
import matplotlib.mathtext as mathtext
import matplotlib.patches as mpatches
import matplotlib.texmanager as texmanager
import matplotlib.transforms as mtransforms

# Import needed for adding manual selection capability to clabel
from matplotlib.blocking_input import BlockingContourLabeler

# We can't use a single line collection for contour because a line
# collection can have only a single line style, and we want to be able to have
# dashed negative contours, for example, and solid positive contours.
# We could use a single polygon collection for filled contours, but it
# seems better to keep line and filled contours similar, with one collection
# per level.


class ClabelText(text.Text):
    """
    Unlike the ordinary text, the get_rotation returns an updated
    angle in the pixel coordinate assuming that the input rotation is
    an angle in data coordinate (or whatever transform set).
    """

    def get_rotation(self):
        new_angle, = self.get_transform().transform_angles(
            [text.Text.get_rotation(self)], [self.get_position()])
        return new_angle


class ContourLabeler:
    """Mixin to provide labelling capability to `.ContourSet`."""

    def clabel(self, levels=None, *,
               fontsize=None, inline=True, inline_spacing=5, fmt='%1.3f',
               colors=None, use_clabeltext=False, manual=False,
               rightside_up=True):
        """
        Label a contour plot.

        Adds labels to line contours in this `.ContourSet` (which inherits from
        this mixin class).

        Parameters
        ----------
        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``cs.levels``. If not given, all levels are labeled.

        fontsize : str or float, optional
            Size in points or relative size e.g., 'smaller', 'x-large'.
            See `.Text.set_size` for accepted string values.

        colors : color-spec, optional
            The label colors:

            - If *None*, the color of each label matches the color of
              the corresponding contour.

            - If one string color, e.g., *colors* = 'r' or *colors* =
              'red', all labels will be plotted in this color.

            - If a tuple of matplotlib color args (string, float, rgb, etc),
              different labels will be plotted in different colors in the order
              specified.

        inline : bool, optional
            If ``True`` the underlying contour is removed where the label is
            placed. Default is ``True``.

        inline_spacing : float, optional
            Space in pixels to leave on each side of label when
            placing inline. Defaults to 5.

            This spacing will be exact for labels at locations where the
            contour is straight, less so for labels on curved contours.

        fmt : str or dict, optional
            A format string for the label. Default is '%1.3f'

            Alternatively, this can be a dictionary matching contour levels
            with arbitrary strings to use for each contour level (i.e.,
            fmt[level]=string), or it can be any callable, such as a
            `.Formatter` instance, that returns a string when called with a
            numeric contour level.

        manual : bool or iterable, optional
            If ``True``, contour labels will be placed manually using
            mouse clicks. Click the first button near a contour to
            add a label, click the second button (or potentially both
            mouse buttons at once) to finish adding labels. The third
            button can be used to remove the last label added, but
            only if labels are not inline. Alternatively, the keyboard
            can be used to select label locations (enter to end label
            placement, delete or backspace act like the third mouse button,
            and any other key will select a label location).

            *manual* can also be an iterable object of (x, y) tuples.
            Contour labels will be created as if mouse is clicked at each
            (x, y) position.

        rightside_up : bool, optional
            If ``True``, label rotations will always be plus
            or minus 90 degrees from level. Default is ``True``.

        use_clabeltext : bool, optional
            If ``True``, `.ClabelText` class (instead of `.Text`) is used to
            create labels. `ClabelText` recalculates rotation angles
            of texts during the drawing time, therefore this can be used if
            aspect of the axes changes. Default is ``False``.

        Returns
        -------
        labels
            A list of `.Text` instances for the labels.
        """

        # clabel basically takes the input arguments and uses them to
        # add a list of "label specific" attributes to the ContourSet
        # object.  These attributes are all of the form label* and names
        # should be fairly self explanatory.
        #
        # Once these attributes are set, clabel passes control to the
        # labels method (case of automatic label placement) or
        # `BlockingContourLabeler` (case of manual label placement).

        self.labelFmt = fmt
        self._use_clabeltext = use_clabeltext
        # Detect if manual selection is desired and remove from argument list.
        self.labelManual = manual
        self.rightside_up = rightside_up

        if levels is None:
            levels = self.levels
            indices = list(range(len(self.cvalues)))
        else:
            levlabs = list(levels)
            indices, levels = [], []
            for i, lev in enumerate(self.levels):
                if lev in levlabs:
                    indices.append(i)
                    levels.append(lev)
            if len(levels) < len(levlabs):
                raise ValueError(f"Specified levels {levlabs} don't match "
                                 f"available levels {self.levels}")
        self.labelLevelList = levels
        self.labelIndiceList = indices

        self.labelFontProps = font_manager.FontProperties()
        self.labelFontProps.set_size(fontsize)
        font_size_pts = self.labelFontProps.get_size_in_points()
        self.labelFontSizeList = [font_size_pts] * len(levels)

        if colors is None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = mcolors.ListedColormap(colors, N=len(self.labelLevelList))
            self.labelCValueList = list(range(len(self.labelLevelList)))
            self.labelMappable = cm.ScalarMappable(cmap=cmap,
                                                   norm=mcolors.NoNorm())

        self.labelXYs = []

        if np.iterable(self.labelManual):
            for x, y in self.labelManual:
                self.add_label_near(x, y, inline, inline_spacing)
        elif self.labelManual:
            print('Select label locations manually using first mouse button.')
            print('End manual selection with second mouse button.')
            if not inline:
                print('Remove last label by clicking third mouse button.')
            blocking_contour_labeler = BlockingContourLabeler(self)
            blocking_contour_labeler(inline, inline_spacing)
        else:
            self.labels(inline, inline_spacing)

        self.labelTextsList = cbook.silent_list('text.Text', self.labelTexts)
        return self.labelTextsList

    def print_label(self, linecontour, labelwidth):
        "Return *False* if contours are too short for a label."
        return (len(linecontour) > 10 * labelwidth
                or (np.ptp(linecontour, axis=0) > 1.2 * labelwidth).any())

    def too_close(self, x, y, lw):
        "Return *True* if a label is already near this location."
        thresh = (1.2 * lw) ** 2
        return any((x - loc[0]) ** 2 + (y - loc[1]) ** 2 < thresh
                   for loc in self.labelXYs)

    def get_label_coords(self, distances, XX, YY, ysize, lw):
        """
        Return x, y, and the index of a label location.

        Labels are plotted at a location with the smallest
        deviation of the contour from a straight line
        unless there is another label nearby, in which case
        the next best place on the contour is picked up.
        If all such candidates are rejected, the beginning
        of the contour is chosen.
        """
        hysize = int(ysize / 2)
        adist = np.argsort(distances)

        for ind in adist:
            x, y = XX[ind][hysize], YY[ind][hysize]
            if self.too_close(x, y, lw):
                continue
            return x, y, ind

        ind = adist[0]
        x, y = XX[ind][hysize], YY[ind][hysize]
        return x, y, ind

    def get_label_width(self, lev, fmt, fsize):
        """
        Return the width of the label in points.
        """
        if not isinstance(lev, str):
            lev = self.get_text(lev, fmt)
        lev, ismath = text.Text()._preprocess_math(lev)
        if ismath == 'TeX':
            lw, _, _ = (texmanager.TexManager()
                        .get_text_width_height_descent(lev, fsize))
        elif ismath:
            if not hasattr(self, '_mathtext_parser'):
                self._mathtext_parser = mathtext.MathTextParser('bitmap')
            img, _ = self._mathtext_parser.parse(lev, dpi=72,
                                                 prop=self.labelFontProps)
            _, lw = np.shape(img)  # at dpi=72, the units are PostScript points
        else:
            # width is much less than "font size"
            lw = len(lev) * fsize * 0.6
        return lw

    def set_label_props(self, label, text, color):
        """Set the label properties - color, fontsize, text."""
        label.set_text(text)
        label.set_color(color)
        label.set_fontproperties(self.labelFontProps)
        label.set_clip_box(self.ax.bbox)

    def get_text(self, lev, fmt):
        """Get the text of the label."""
        if isinstance(lev, str):
            return lev
        else:
            if isinstance(fmt, dict):
                return fmt.get(lev, '%1.3f')
            elif callable(fmt):
                return fmt(lev)
            else:
                return fmt % lev

    def locate_label(self, linecontour, labelwidth):
        """
        Find good place to draw a label (relatively flat part of the contour).
        """

        # Number of contour points
        nsize = len(linecontour)
        if labelwidth > 1:
            xsize = int(np.ceil(nsize / labelwidth))
        else:
            xsize = 1
        if xsize == 1:
            ysize = nsize
        else:
            ysize = int(labelwidth)

        XX = np.resize(linecontour[:, 0], (xsize, ysize))
        YY = np.resize(linecontour[:, 1], (xsize, ysize))
        # I might have fouled up the following:
        yfirst = YY[:, :1]
        ylast = YY[:, -1:]
        xfirst = XX[:, :1]
        xlast = XX[:, -1:]
        s = (yfirst - YY) * (xlast - xfirst) - (xfirst - XX) * (ylast - yfirst)
        L = np.hypot(xlast - xfirst, ylast - yfirst)
        # Ignore warning that divide by zero throws, as this is a valid option
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.sum(np.abs(s) / L, axis=-1)
        x, y, ind = self.get_label_coords(dist, XX, YY, ysize, labelwidth)

        # There must be a more efficient way...
        lc = [tuple(l) for l in linecontour]
        dind = lc.index((x, y))

        return x, y, dind

    def calc_label_rot_and_inline(self, slc, ind, lw, lc=None, spacing=5):
        """
        This function calculates the appropriate label rotation given
        the linecontour coordinates in screen units, the index of the
        label location and the label width.

        It will also break contour and calculate inlining if *lc* is
        not empty (lc defaults to the empty list if None).  *spacing*
        is the space around the label in pixels to leave empty.

        Do both of these tasks at once to avoid calculating path lengths
        multiple times, which is relatively costly.

        The method used here involves calculating the path length
        along the contour in pixel coordinates and then looking
        approximately label width / 2 away from central point to
        determine rotation and then to break contour if desired.
        """

        if lc is None:
            lc = []
        # Half the label width
        hlw = lw / 2.0

        # Check if closed and, if so, rotate contour so label is at edge
        closed = _is_closed_polygon(slc)
        if closed:
            slc = np.r_[slc[ind:-1], slc[:ind + 1]]

            if len(lc):  # Rotate lc also if not empty
                lc = np.r_[lc[ind:-1], lc[:ind + 1]]

            ind = 0

        # Calculate path lengths
        pl = np.zeros(slc.shape[0], dtype=float)
        dx = np.diff(slc, axis=0)
        pl[1:] = np.cumsum(np.hypot(dx[:, 0], dx[:, 1]))
        pl = pl - pl[ind]

        # Use linear interpolation to get points around label
        xi = np.array([-hlw, hlw])
        if closed:  # Look at end also for closed contours
            dp = np.array([pl[-1], 0])
        else:
            dp = np.zeros_like(xi)

        # Get angle of vector between the two ends of the label - must be
        # calculated in pixel space for text rotation to work correctly.
        (dx,), (dy,) = (np.diff(np.interp(dp + xi, pl, slc_col))
                        for slc_col in slc.T)
        rotation = np.rad2deg(np.arctan2(dy, dx))

        if self.rightside_up:
            # Fix angle so text is never upside-down
            rotation = (rotation + 90) % 180 - 90

        # Break contour if desired
        nlc = []
        if len(lc):
            # Expand range by spacing
            xi = dp + xi + np.array([-spacing, spacing])

            # Get (integer) indices near points of interest; use -1 as marker
            # for out of bounds.
            I = np.interp(xi, pl, np.arange(len(pl)), left=-1, right=-1)
            I = [np.floor(I[0]).astype(int), np.ceil(I[1]).astype(int)]
            if I[0] != -1:
                xy1 = [np.interp(xi[0], pl, lc_col) for lc_col in lc.T]
            if I[1] != -1:
                xy2 = [np.interp(xi[1], pl, lc_col) for lc_col in lc.T]

            # Actually break contours
            if closed:
                # This will remove contour if shorter than label
                if all(i != -1 for i in I):
                    nlc.append(np.row_stack([xy2, lc[I[1]:I[0]+1], xy1]))
            else:
                # These will remove pieces of contour if they have length zero
                if I[0] != -1:
                    nlc.append(np.row_stack([lc[:I[0]+1], xy1]))
                if I[1] != -1:
                    nlc.append(np.row_stack([xy2, lc[I[1]:]]))

            # The current implementation removes contours completely
            # covered by labels.  Uncomment line below to keep
            # original contour if this is the preferred behavior.
            # if not len(nlc): nlc = [ lc ]

        return rotation, nlc

    def _get_label_text(self, x, y, rotation):
        dx, dy = self.ax.transData.inverted().transform((x, y))
        t = text.Text(dx, dy, rotation=rotation,
                      horizontalalignment='center',
                      verticalalignment='center')
        return t

    def _get_label_clabeltext(self, x, y, rotation):
        # x, y, rotation is given in pixel coordinate. Convert them to
        # the data coordinate and create a label using ClabelText
        # class. This way, the rotation of the clabel is along the
        # contour line always.
        transDataInv = self.ax.transData.inverted()
        dx, dy = transDataInv.transform((x, y))
        drotation = transDataInv.transform_angles(np.array([rotation]),
                                                  np.array([[x, y]]))
        t = ClabelText(dx, dy, rotation=drotation[0],
                       horizontalalignment='center',
                       verticalalignment='center')

        return t

    def _add_label(self, t, x, y, lev, cvalue):
        color = self.labelMappable.to_rgba(cvalue, alpha=self.alpha)

        _text = self.get_text(lev, self.labelFmt)
        self.set_label_props(t, _text, color)
        self.labelTexts.append(t)
        self.labelCValues.append(cvalue)
        self.labelXYs.append((x, y))

        # Add label to plot here - useful for manual mode label selection
        self.ax.add_artist(t)

    def add_label(self, x, y, rotation, lev, cvalue):
        """
        Add contour label using :class:`~matplotlib.text.Text` class.
        """
        t = self._get_label_text(x, y, rotation)
        self._add_label(t, x, y, lev, cvalue)

    def add_label_clabeltext(self, x, y, rotation, lev, cvalue):
        """
        Add contour label using :class:`ClabelText` class.
        """
        # x, y, rotation is given in pixel coordinate. Convert them to
        # the data coordinate and create a label using ClabelText
        # class. This way, the rotation of the clabel is along the
        # contour line always.
        t = self._get_label_clabeltext(x, y, rotation)
        self._add_label(t, x, y, lev, cvalue)

    def add_label_near(self, x, y, inline=True, inline_spacing=5,
                       transform=None):
        """
        Add a label near the point (x, y). If transform is None
        (default), (x, y) is in data coordinates; if transform is
        False, (x, y) is in display coordinates; otherwise, the
        specified transform will be used to translate (x, y) into
        display coordinates.

        Parameters
        ----------
        x, y : float
            The approximate location of the label.

        inline : bool, optional, default: True
            If *True* remove the segment of the contour beneath the label.

        inline_spacing : int, optional, default: 5
            Space in pixels to leave on each side of label when placing
            inline. This spacing will be exact for labels at locations where
            the contour is straight, less so for labels on curved contours.
        """

        if transform is None:
            transform = self.ax.transData

        if transform:
            x, y = transform.transform((x, y))

        # find the nearest contour _in screen units_
        conmin, segmin, imin, xmin, ymin = self.find_nearest_contour(
            x, y, self.labelIndiceList)[:5]

        # The calc_label_rot_and_inline routine requires that (xmin, ymin)
        # be a vertex in the path. So, if it isn't, add a vertex here

        # grab the paths from the collections
        paths = self.collections[conmin].get_paths()
        # grab the correct segment
        active_path = paths[segmin]
        # grab its vertices
        lc = active_path.vertices
        # sort out where the new vertex should be added data-units
        xcmin = self.ax.transData.inverted().transform([xmin, ymin])
        # if there isn't a vertex close enough
        if not np.allclose(xcmin, lc[imin]):
            # insert new data into the vertex list
            lc = np.r_[lc[:imin], np.array(xcmin)[None, :], lc[imin:]]
            # replace the path with the new one
            paths[segmin] = mpath.Path(lc)

        # Get index of nearest level in subset of levels used for labeling
        lmin = self.labelIndiceList.index(conmin)

        # Coordinates of contour
        paths = self.collections[conmin].get_paths()
        lc = paths[segmin].vertices

        # In pixel/screen space
        slc = self.ax.transData.transform(lc)

        # Get label width for rotating labels and breaking contours
        lw = self.get_label_width(self.labelLevelList[lmin],
                                  self.labelFmt, self.labelFontSizeList[lmin])
        # lw is in points.
        lw *= self.ax.figure.dpi / 72.0  # scale to screen coordinates
        # now lw in pixels

        # Figure out label rotation.
        if inline:
            lcarg = lc
        else:
            lcarg = None
        rotation, nlc = self.calc_label_rot_and_inline(
            slc, imin, lw, lcarg,
            inline_spacing)

        self.add_label(xmin, ymin, rotation, self.labelLevelList[lmin],
                       self.labelCValueList[lmin])

        if inline:
            # Remove old, not looping over paths so we can do this up front
            paths.pop(segmin)

            # Add paths if not empty or single point
            for n in nlc:
                if len(n) > 1:
                    paths.append(mpath.Path(n))

    def pop_label(self, index=-1):
        """Defaults to removing last label, but any index can be supplied"""
        self.labelCValues.pop(index)
        t = self.labelTexts.pop(index)
        t.remove()

    def labels(self, inline, inline_spacing):

        if self._use_clabeltext:
            add_label = self.add_label_clabeltext
        else:
            add_label = self.add_label

        for icon, lev, fsize, cvalue in zip(
                self.labelIndiceList, self.labelLevelList,
                self.labelFontSizeList, self.labelCValueList):

            con = self.collections[icon]
            trans = con.get_transform()
            lw = self.get_label_width(lev, self.labelFmt, fsize)
            lw *= self.ax.figure.dpi / 72.0  # scale to screen coordinates
            additions = []
            paths = con.get_paths()
            for segNum, linepath in enumerate(paths):
                lc = linepath.vertices  # Line contour
                slc0 = trans.transform(lc)  # Line contour in screen coords

                # For closed polygons, add extra point to avoid division by
                # zero in print_label and locate_label.  Other than these
                # functions, this is not necessary and should probably be
                # eventually removed.
                if _is_closed_polygon(lc):
                    slc = np.r_[slc0, slc0[1:2, :]]
                else:
                    slc = slc0

                # Check if long enough for a label
                if self.print_label(slc, lw):
                    x, y, ind = self.locate_label(slc, lw)

                    if inline:
                        lcarg = lc
                    else:
                        lcarg = None
                    rotation, new = self.calc_label_rot_and_inline(
                        slc0, ind, lw, lcarg,
                        inline_spacing)

                    # Actually add the label
                    add_label(x, y, rotation, lev, cvalue)

                    # If inline, add new contours
                    if inline:
                        for n in new:
                            # Add path if not empty or single point
                            if len(n) > 1:
                                additions.append(mpath.Path(n))
                else:  # If not adding label, keep old path
                    additions.append(linepath)

            # After looping over all segments on a contour, replace old paths
            # by new ones if inlining.
            if inline:
                paths[:] = additions


def _find_closest_point_on_leg(p1, p2, p0):
    """Find the closest point to p0 on line segment connecting p1 and p2."""

    # handle degenerate case
    if np.all(p2 == p1):
        d = np.sum((p0 - p1)**2)
        return d, p1

    d21 = p2 - p1
    d01 = p0 - p1

    # project on to line segment to find closest point
    proj = np.dot(d01, d21) / np.dot(d21, d21)
    if proj < 0:
        proj = 0
    if proj > 1:
        proj = 1
    pc = p1 + proj * d21

    # find squared distance
    d = np.sum((pc-p0)**2)

    return d, pc


def _is_closed_polygon(X):
    """
    Return whether first and last object in a sequence are the same. These are
    presumably coordinates on a polygonal curve, in which case this function
    tests if that curve is closed.
    """
    return np.all(X[0] == X[-1])


def _find_closest_point_on_path(lc, point):
    """
    Parameters
    ----------
    lc : coordinates of vertices
    point : coordinates of test point
    """

    # find index of closest vertex for this segment
    ds = np.sum((lc - point[None, :])**2, 1)
    imin = np.argmin(ds)

    dmin = np.inf
    xcmin = None
    legmin = (None, None)

    closed = _is_closed_polygon(lc)

    # build list of legs before and after this vertex
    legs = []
    if imin > 0 or closed:
        legs.append(((imin-1) % len(lc), imin))
    if imin < len(lc) - 1 or closed:
        legs.append((imin, (imin+1) % len(lc)))

    for leg in legs:
        d, xc = _find_closest_point_on_leg(lc[leg[0]], lc[leg[1]], point)
        if d < dmin:
            dmin = d
            xcmin = xc
            legmin = leg

    return (dmin, xcmin, legmin)


class ContourSet(cm.ScalarMappable, ContourLabeler):
    """
    Store a set of contour lines or filled regions.

    User-callable method: `~.axes.Axes.clabel`

    Parameters
    ----------
    ax : `~.axes.Axes`

    levels : [level0, level1, ..., leveln]
        A list of floating point numbers indicating the contour
        levels.

    allsegs : [level0segs, level1segs, ...]
        List of all the polygon segments for all the *levels*.
        For contour lines ``len(allsegs) == len(levels)``, and for
        filled contour regions ``len(allsegs) = len(levels)-1``. The lists
        should look like ::

            level0segs = [polygon0, polygon1, ...]
            polygon0 = [[x0, y0], [x1, y1], ...]

    allkinds : ``None`` or [level0kinds, level1kinds, ...]
        Optional list of all the polygon vertex kinds (code types), as
        described and used in Path. This is used to allow multiply-
        connected paths such as holes within filled polygons.
        If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
        should look like ::

            level0kinds = [polygon0kinds, ...]
            polygon0kinds = [vertexcode0, vertexcode1, ...]

        If *allkinds* is not ``None``, usually all polygons for a
        particular contour level are grouped together so that
        ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

    **kwargs
        Keyword arguments are as described in the docstring of
        `~.axes.Axes.contour`.

    Attributes
    ----------
    ax
        The axes object in which the contours are drawn.

    collections
        A silent_list of LineCollections or PolyCollections.

    levels
        Contour levels.

    layers
        Same as levels for line contours; half-way between
        levels for filled contours.  See :meth:`_process_colors`.
    """

    def __init__(self, ax, *args,
                 levels=None, filled=False, linewidths=None, linestyles=None,
                 alpha=None, origin=None, extent=None,
                 cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                 extend='neither', antialiased=None,
                 **kwargs):
        """
        Draw contour lines or filled regions, depending on
        whether keyword arg *filled* is ``False`` (default) or ``True``.

        Call signature::

            ContourSet(ax, levels, allsegs, [allkinds], **kwargs)

        Parameters
        ----------
        ax : `~.axes.Axes`
            The `~.axes.Axes` object to draw on.

        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the contour
            levels.

        allsegs : [level0segs, level1segs, ...]
            List of all the polygon segments for all the *levels*.
            For contour lines ``len(allsegs) == len(levels)``, and for
            filled contour regions ``len(allsegs) = len(levels)-1``. The lists
            should look like ::

                level0segs = [polygon0, polygon1, ...]
                polygon0 = [[x0, y0], [x1, y1], ...]

        allkinds : [level0kinds, level1kinds, ...], optional
            Optional list of all the polygon vertex kinds (code types), as
            described and used in Path. This is used to allow multiply-
            connected paths such as holes within filled polygons.
            If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
            should look like ::

                level0kinds = [polygon0kinds, ...]
                polygon0kinds = [vertexcode0, vertexcode1, ...]

            If *allkinds* is not ``None``, usually all polygons for a
            particular contour level are grouped together so that
            ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

        **kwargs
            Keyword arguments are as described in the docstring of
            `~axes.Axes.contour`.
        """
        self.ax = ax
        self.levels = levels
        self.filled = filled
        self.linewidths = linewidths
        self.linestyles = linestyles
        self.hatches = kwargs.pop('hatches', [None])
        self.alpha = alpha
        self.origin = origin
        self.extent = extent
        self.colors = colors
        self.extend = extend
        self.antialiased = antialiased
        if self.antialiased is None and self.filled:
            # Eliminate artifacts; we are not stroking the boundaries.
            self.antialiased = False
            # The default for line contours will be taken from the
            # LineCollection default, which uses :rc:`lines.antialiased`.

        self.nchunk = kwargs.pop('nchunk', 0)
        self.locator = kwargs.pop('locator', None)
        if (isinstance(norm, mcolors.LogNorm)
                or isinstance(self.locator, ticker.LogLocator)):
            self.logscale = True
            if norm is None:
                norm = mcolors.LogNorm()
        else:
            self.logscale = False

        cbook._check_in_list([None, 'lower', 'upper', 'image'], origin=origin)
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError(
                "If given, 'extent' must be None or (x0, x1, y0, y1)")
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image':
            self.origin = mpl.rcParams['image.origin']

        self._transform = kwargs.pop('transform', None)

        kwargs = self._process_args(*args, **kwargs)
        self._process_levels()

        if self.colors is not None:
            ncolors = len(self.levels)
            if self.filled:
                ncolors -= 1
            i0 = 0

            # Handle the case where colors are given for the extended
            # parts of the contour.
            extend_min = self.extend in ['min', 'both']
            extend_max = self.extend in ['max', 'both']
            use_set_under_over = False
            # if we are extending the lower end, and we've been given enough
            # colors then skip the first color in the resulting cmap. For the
            # extend_max case we don't need to worry about passing more colors
            # than ncolors as ListedColormap will clip.
            total_levels = ncolors + int(extend_min) + int(extend_max)
            if len(self.colors) == total_levels and (extend_min or extend_max):
                use_set_under_over = True
                if extend_min:
                    i0 = 1

            cmap = mcolors.ListedColormap(self.colors[i0:None], N=ncolors)

            if use_set_under_over:
                if extend_min:
                    cmap.set_under(self.colors[0])
                if extend_max:
                    cmap.set_over(self.colors[-1])

        if self.filled:
            self.collections = cbook.silent_list('mcoll.PathCollection')
        else:
            self.collections = cbook.silent_list('mcoll.LineCollection')
        # label lists must be initialized here
        self.labelTexts = []
        self.labelCValues = []

        kw = {'cmap': cmap}
        if norm is not None:
            kw['norm'] = norm
        # sets self.cmap, norm if needed;
        cm.ScalarMappable.__init__(self, **kw)
        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax
        self._process_colors()

        self.allsegs, self.allkinds = self._get_allsegs_and_allkinds()

        if self.filled:
            if self.linewidths is not None:
                cbook._warn_external('linewidths is ignored by contourf')

            # Lower and upper contour levels.
            lowers, uppers = self._get_lowers_and_uppers()

            # Ensure allkinds can be zipped below.
            if self.allkinds is None:
                self.allkinds = [None] * len(self.allsegs)

            # Default zorder taken from Collection
            zorder = kwargs.pop('zorder', 1)
            for level, level_upper, segs, kinds in \
                    zip(lowers, uppers, self.allsegs, self.allkinds):
                paths = self._make_paths(segs, kinds)

                col = mcoll.PathCollection(
                    paths,
                    antialiaseds=(self.antialiased,),
                    edgecolors='none',
                    alpha=self.alpha,
                    transform=self.get_transform(),
                    zorder=zorder)
                self.ax.add_collection(col, autolim=False)
                self.collections.append(col)
        else:
            tlinewidths = self._process_linewidths()
            self.tlinewidths = tlinewidths
            tlinestyles = self._process_linestyles()
            aa = self.antialiased
            if aa is not None:
                aa = (self.antialiased,)
            # Default zorder taken from LineCollection
            zorder = kwargs.pop('zorder', 2)
            for level, width, lstyle, segs in \
                    zip(self.levels, tlinewidths, tlinestyles, self.allsegs):
                col = mcoll.LineCollection(
                    segs,
                    antialiaseds=aa,
                    linewidths=width,
                    linestyles=[lstyle],
                    alpha=self.alpha,
                    transform=self.get_transform(),
                    zorder=zorder)
                col.set_label('_nolegend_')
                self.ax.add_collection(col, autolim=False)
                self.collections.append(col)

        for col in self.collections:
            col.sticky_edges.x[:] = [self._mins[0], self._maxs[0]]
            col.sticky_edges.y[:] = [self._mins[1], self._maxs[1]]
        self.ax.update_datalim([self._mins, self._maxs])
        self.ax.autoscale_view(tight=True)

        self.changed()  # set the colors

        if kwargs:
            s = ", ".join(map(repr, kwargs))
            cbook._warn_external('The following kwargs were not used by '
                                 'contour: ' + s)

    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform`
        instance used by this ContourSet.
        """
        if self._transform is None:
            self._transform = self.ax.transData
        elif (not isinstance(self._transform, mtransforms.Transform)
              and hasattr(self._transform, '_as_mpl_transform')):
            self._transform = self._transform._as_mpl_transform(self.ax)
        return self._transform

    def __getstate__(self):
        state = self.__dict__.copy()
        # the C object _contour_generator cannot currently be pickled. This
        # isn't a big issue as it is not actually used once the contour has
        # been calculated.
        state['_contour_generator'] = None
        return state

    def legend_elements(self, variable_name='x', str_format=str):
        """
        Return a list of artists and labels suitable for passing through
        to :func:`plt.legend` which represent this ContourSet.

        The labels have the form "0 < x <= 1" stating the data ranges which
        the artists represent.

        Parameters
        ----------
        variable_name : str
            The string used inside the inequality used on the labels.

        str_format : function: float -> str
            Function used to format the numbers in the labels.

        Returns
        -------
        artists : List[`.Artist`]
            A list of the artists.

        labels : List[str]
            A list of the labels.

        """
        artists = []
        labels = []

        if self.filled:
            lowers, uppers = self._get_lowers_and_uppers()
            n_levels = len(self.collections)

            for i, (collection, lower, upper) in enumerate(
                    zip(self.collections, lowers, uppers)):
                patch = mpatches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=collection.get_facecolor()[0],
                    hatch=collection.get_hatch(),
                    alpha=collection.get_alpha())
                artists.append(patch)

                lower = str_format(lower)
                upper = str_format(upper)

                if i == 0 and self.extend in ('min', 'both'):
                    labels.append(fr'${variable_name} \leq {lower}s$')
                elif i == n_levels - 1 and self.extend in ('max', 'both'):
                    labels.append(fr'${variable_name} > {upper}s$')
                else:
                    labels.append(fr'${lower} < {variable_name} \leq {upper}$')
        else:
            for collection, level in zip(self.collections, self.levels):

                patch = mcoll.LineCollection(None)
                patch.update_from(collection)

                artists.append(patch)
                # format the level for insertion into the labels
                level = str_format(level)
                labels.append(fr'${variable_name} = {level}$')

        return artists, labels

    def _process_args(self, *args, **kwargs):
        """
        Process *args* and *kwargs*; override in derived classes.

        Must set self.levels, self.zmin and self.zmax, and update axes
        limits.
        """
        self.levels = args[0]
        self.allsegs = args[1]
        self.allkinds = args[2] if len(args) > 2 else None
        self.zmax = np.max(self.levels)
        self.zmin = np.min(self.levels)

        # Check lengths of levels and allsegs.
        if self.filled:
            if len(self.allsegs) != len(self.levels) - 1:
                raise ValueError('must be one less number of segments as '
                                 'levels')
        else:
            if len(self.allsegs) != len(self.levels):
                raise ValueError('must be same number of segments as levels')

        # Check length of allkinds.
        if (self.allkinds is not None and
                len(self.allkinds) != len(self.allsegs)):
            raise ValueError('allkinds has different length to allsegs')

        # Determine x, y bounds and update axes data limits.
        flatseglist = [s for seg in self.allsegs for s in seg]
        points = np.concatenate(flatseglist, axis=0)
        self._mins = points.min(axis=0)
        self._maxs = points.max(axis=0)

        return kwargs

    def _get_allsegs_and_allkinds(self):
        """
        Override in derived classes to create and return allsegs and allkinds.
        allkinds can be None.
        """
        return self.allsegs, self.allkinds

    def _get_lowers_and_uppers(self):
        """
        Return ``(lowers, uppers)`` for filled contours.
        """
        lowers = self._levels[:-1]
        if self.zmin == lowers[0]:
            # Include minimum values in lowest interval
            lowers = lowers.copy()  # so we don't change self._levels
            if self.logscale:
                lowers[0] = 0.99 * self.zmin
            else:
                lowers[0] -= 1
        uppers = self._levels[1:]
        return (lowers, uppers)

    def _make_paths(self, segs, kinds):
        if kinds is not None:
            return [mpath.Path(seg, codes=kind)
                    for seg, kind in zip(segs, kinds)]
        else:
            return [mpath.Path(seg) for seg in segs]

    def changed(self):
        tcolors = [(tuple(rgba),)
                   for rgba in self.to_rgba(self.cvalues, alpha=self.alpha)]
        self.tcolors = tcolors
        hatches = self.hatches * len(tcolors)
        for color, hatch, collection in zip(tcolors, hatches,
                                            self.collections):
            if self.filled:
                collection.set_facecolor(color)
                # update the collection's hatch (may be None)
                collection.set_hatch(hatch)
            else:
                collection.set_color(color)
        for label, cv in zip(self.labelTexts, self.labelCValues):
            label.set_alpha(self.alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        # add label colors
        cm.ScalarMappable.changed(self)

    def _autolev(self, N):
        """
        Select contour levels to span the data.

        The target number of levels, *N*, is used only when the
        scale is not log and default locator is used.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        """
        if self.locator is None:
            if self.logscale:
                self.locator = ticker.LogLocator()
            else:
                self.locator = ticker.MaxNLocator(N + 1, min_n_ticks=1)

        lev = self.locator.tick_values(self.zmin, self.zmax)

        try:
            if self.locator._symmetric:
                return lev
        except AttributeError:
            pass

        # Trim excess levels the locator may have supplied.
        under = np.nonzero(lev < self.zmin)[0]
        i0 = under[-1] if len(under) else 0
        over = np.nonzero(lev > self.zmax)[0]
        i1 = over[0] + 1 if len(over) else len(lev)
        if self.extend in ('min', 'both'):
            i0 += 1
        if self.extend in ('max', 'both'):
            i1 -= 1

        if i1 - i0 < 3:
            i0, i1 = 0, len(lev)

        return lev[i0:i1]

    def _contour_level_args(self, z, args):
        """
        Determine the contour levels and store in self.levels.
        """
        if self.levels is None:
            if len(args) == 0:
                levels_arg = 7  # Default, hard-wired.
            else:
                levels_arg = args[0]
        else:
            levels_arg = self.levels
        if isinstance(levels_arg, Integral):
            self.levels = self._autolev(levels_arg)
        else:
            self.levels = np.asarray(levels_arg).astype(np.float64)

        if not self.filled:
            inside = (self.levels > self.zmin) & (self.levels < self.zmax)
            levels_in = self.levels[inside]
            if len(levels_in) == 0:
                self.levels = [self.zmin]
                cbook._warn_external(
                    "No contour levels were found within the data range.")

        if self.filled and len(self.levels) < 2:
            raise ValueError("Filled contours require at least 2 levels.")

        if len(self.levels) > 1 and np.min(np.diff(self.levels)) <= 0.0:
            raise ValueError("Contour levels must be increasing")

    def _process_levels(self):
        """
        Assign values to :attr:`layers` based on :attr:`levels`,
        adding extended layers as needed if contours are filled.

        For line contours, layers simply coincide with levels;
        a line is a thin layer.  No extended levels are needed
        with line contours.
        """
        # Make a private _levels to include extended regions; we
        # want to leave the original levels attribute unchanged.
        # (Colorbar needs this even for line contours.)
        self._levels = list(self.levels)

        if self.logscale:
            lower, upper = 1e-250, 1e250
        else:
            lower, upper = -1e250, 1e250

        if self.extend in ('both', 'min'):
            self._levels.insert(0, lower)
        if self.extend in ('both', 'max'):
            self._levels.append(upper)
        self._levels = np.asarray(self._levels)

        if not self.filled:
            self.layers = self.levels
            return

        # Layer values are mid-way between levels in screen space.
        if self.logscale:
            # Avoid overflow by taking sqrt before multiplying.
            self.layers = (np.sqrt(self._levels[:-1])
                           * np.sqrt(self._levels[1:]))
        else:
            self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])

    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the color mapping on the contour levels
        and layers, not on the actual range of the Z values.  This
        means we don't have to worry about bad values in Z, and we
        always have the full dynamic range available for the selected
        levels.

        The color is based on the midpoint of the layer, except for
        extended end layers.  By default, the norm vmin and vmax
        are the extreme values of the non-extended levels.  Hence,
        the layer color extremes are not the extreme values of
        the colormap itself, but approach those values as the number
        of levels increases.  An advantage of this scheme is that
        line contours, when added to filled contours, take on
        colors that are consistent with those of the filled regions;
        for example, a contour line on the boundary between two
        regions will have a color intermediate between those
        of the regions.

        """
        self.monochrome = self.cmap.monochrome
        if self.colors is not None:
            # Generate integers for direct indexing.
            i0, i1 = 0, len(self.levels)
            if self.filled:
                i1 -= 1
                # Out of range indices for over and under:
                if self.extend in ('both', 'min'):
                    i0 -= 1
                if self.extend in ('both', 'max'):
                    i1 += 1
            self.cvalues = list(range(i0, i1))
            self.set_norm(mcolors.NoNorm())
        else:
            self.cvalues = self.layers
        self.set_array(self.levels)
        self.autoscale_None()
        if self.extend in ('both', 'max', 'min'):
            self.norm.clip = False

        # self.tcolors are set by the "changed" method

    def _process_linewidths(self):
        linewidths = self.linewidths
        Nlev = len(self.levels)
        if linewidths is None:
            tlinewidths = [(mpl.rcParams['lines.linewidth'],)] * Nlev
        else:
            if not np.iterable(linewidths):
                linewidths = [linewidths] * Nlev
            else:
                linewidths = list(linewidths)
                if len(linewidths) < Nlev:
                    nreps = int(np.ceil(Nlev / len(linewidths)))
                    linewidths = linewidths * nreps
                if len(linewidths) > Nlev:
                    linewidths = linewidths[:Nlev]
            tlinewidths = [(w,) for w in linewidths]
        return tlinewidths

    def _process_linestyles(self):
        linestyles = self.linestyles
        Nlev = len(self.levels)
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
            if self.monochrome:
                neg_ls = mpl.rcParams['contour.negative_linestyle']
                eps = - (self.zmax - self.zmin) * 1e-15
                for i, lev in enumerate(self.levels):
                    if lev < eps:
                        tlinestyles[i] = neg_ls
        else:
            if isinstance(linestyles, str):
                tlinestyles = [linestyles] * Nlev
            elif np.iterable(linestyles):
                tlinestyles = list(linestyles)
                if len(tlinestyles) < Nlev:
                    nreps = int(np.ceil(Nlev / len(linestyles)))
                    tlinestyles = tlinestyles * nreps
                if len(tlinestyles) > Nlev:
                    tlinestyles = tlinestyles[:Nlev]
            else:
                raise ValueError("Unrecognized type for linestyles kwarg")
        return tlinestyles

    def get_alpha(self):
        """returns alpha to be applied to all ContourSet artists"""
        return self.alpha

    def set_alpha(self, alpha):
        """
        Set the alpha blending value for all ContourSet artists.
        *alpha* must be between 0 (transparent) and 1 (opaque).
        """
        self.alpha = alpha
        self.changed()

    def find_nearest_contour(self, x, y, indices=None, pixel=True):
        """
        Finds contour that is closest to a point.  Defaults to
        measuring distance in pixels (screen space - useful for manual
        contour labeling), but this can be controlled via a keyword
        argument.

        Returns a tuple containing the contour, segment, index of
        segment, x & y of segment point and distance to minimum point.

        Optional keyword arguments:

          *indices*:
            Indexes of contour levels to consider when looking for
            nearest point.  Defaults to using all levels.

          *pixel*:
            If *True*, measure distance in pixel space, if not, measure
            distance in axes space.  Defaults to *True*.

        """

        # This function uses a method that is probably quite
        # inefficient based on converting each contour segment to
        # pixel coordinates and then comparing the given point to
        # those coordinates for each contour.  This will probably be
        # quite slow for complex contours, but for normal use it works
        # sufficiently well that the time is not noticeable.
        # Nonetheless, improvements could probably be made.

        if indices is None:
            indices = list(range(len(self.levels)))

        dmin = np.inf
        conmin = None
        segmin = None
        xmin = None
        ymin = None

        point = np.array([x, y])

        for icon in indices:
            con = self.collections[icon]
            trans = con.get_transform()
            paths = con.get_paths()

            for segNum, linepath in enumerate(paths):
                lc = linepath.vertices
                # transfer all data points to screen coordinates if desired
                if pixel:
                    lc = trans.transform(lc)

                d, xc, leg = _find_closest_point_on_path(lc, point)
                if d < dmin:
                    dmin = d
                    conmin = icon
                    segmin = segNum
                    imin = leg[1]
                    xmin = xc[0]
                    ymin = xc[1]

        return (conmin, segmin, imin, xmin, ymin, dmin)


class QuadContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions.

    User-callable method: `~axes.Axes.clabel`

    Attributes
    ----------
    ax
        The axes object in which the contours are drawn.

    collections
        A silent_list of LineCollections or PolyCollections.

    levels
        Contour levels.

    layers
        Same as levels for line contours; half-way between
        levels for filled contours. See :meth:`_process_colors` method.
    """

    def _process_args(self, *args, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], QuadContourSet):
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._corner_mask = args[0]._corner_mask
            contour_generator = args[0]._contour_generator
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
        else:
            import matplotlib._contour as _contour

            self._corner_mask = kwargs.pop('corner_mask', None)
            if self._corner_mask is None:
                self._corner_mask = mpl.rcParams['contour.corner_mask']

            x, y, z = self._contour_args(args, kwargs)

            _mask = ma.getmask(z)
            if _mask is ma.nomask or not _mask.any():
                _mask = None

            contour_generator = _contour.QuadContourGenerator(
                x, y, z.filled(), _mask, self._corner_mask, self.nchunk)

            t = self.get_transform()

            # if the transform is not trans data, and some part of it
            # contains transData, transform the xs and ys to data coordinates
            if (t != self.ax.transData and
                    any(t.contains_branch_seperately(self.ax.transData))):
                trans_to_data = t - self.ax.transData
                pts = (np.vstack([x.flat, y.flat]).T)
                transformed_pts = trans_to_data.transform(pts)
                x = transformed_pts[..., 0]
                y = transformed_pts[..., 1]

            self._mins = [ma.min(x), ma.min(y)]
            self._maxs = [ma.max(x), ma.max(y)]

        self._contour_generator = contour_generator

        return kwargs

    def _get_allsegs_and_allkinds(self):
        """Compute ``allsegs`` and ``allkinds`` using C extension."""
        allsegs = []
        if self.filled:
            lowers, uppers = self._get_lowers_and_uppers()
            allkinds = []
            for level, level_upper in zip(lowers, uppers):
                vertices, kinds = \
                    self._contour_generator.create_filled_contour(
                        level, level_upper)
                allsegs.append(vertices)
                allkinds.append(kinds)
        else:
            allkinds = None
            for level in self.levels:
                vertices = self._contour_generator.create_contour(level)
                allsegs.append(vertices)
        return allsegs, allkinds

    def _contour_args(self, args, kwargs):
        if self.filled:
            fn = 'contourf'
        else:
            fn = 'contour'
        Nargs = len(args)
        if Nargs <= 2:
            z = ma.asarray(args[0], dtype=np.float64)
            x, y = self._initialize_x_y(z)
            args = args[1:]
        elif Nargs <= 4:
            x, y, z = self._check_xyz(args[:3], kwargs)
            args = args[3:]
        else:
            raise TypeError("Too many arguments to %s; see help(%s)" %
                            (fn, fn))
        z = ma.masked_invalid(z, copy=False)
        self.zmax = float(z.max())
        self.zmin = float(z.min())
        if self.logscale and self.zmin <= 0:
            z = ma.masked_where(z <= 0, z)
            cbook._warn_external('Log scale: values of z <= 0 have been '
                                 'masked')
            self.zmin = float(z.min())
        self._contour_level_args(z, args)
        return (x, y, z)

    def _check_xyz(self, args, kwargs):
        """
        Check that the shapes of the input arrays match; if x and y are 1D,
        convert them to 2D using meshgrid.
        """
        x, y = args[:2]
        kwargs = self.ax._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)
        x = self.ax.convert_xunits(x)
        y = self.ax.convert_yunits(y)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = ma.asarray(args[2], dtype=np.float64)

        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        if z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        Ny, Nx = z.shape

        if x.ndim != y.ndim:
            raise TypeError(f"Number of dimensions of x ({x.ndim}) and y "
                            f"({y.ndim}) do not match")
        if x.ndim == 1:
            nx, = x.shape
            ny, = y.shape
            if nx != Nx:
                raise TypeError(f"Length of x ({nx}) must match number of "
                                f"columns in z ({Nx})")
            if ny != Ny:
                raise TypeError(f"Length of y ({ny}) must match number of "
                                f"rows in z ({Ny})")
            x, y = np.meshgrid(x, y)
        elif x.ndim == 2:
            if x.shape != z.shape:
                raise TypeError(
                    f"Shapes of x {x.shape} and z {z.shape} do not match")
            if y.shape != z.shape:
                raise TypeError(
                    f"Shapes of y {y.shape} and z {z.shape} do not match")
        else:
            raise TypeError(f"Inputs x and y must be 1D or 2D, not {x.ndim}D")

        return x, y, z

    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i, j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        if z.ndim != 2:
            raise TypeError(f"Input z must be 2D, not {z.ndim}D")
        elif z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f"Input z must be at least a (2, 2) shaped array, "
                            f"but has shape {z.shape}")
        else:
            Ny, Nx = z.shape
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                x0, x1, y0, y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)
        # Match image behavior:
        if self.extent is None:
            x0, x1, y0, y1 = (0, Nx, 0, Ny)
        else:
            x0, x1, y0, y1 = self.extent
        dx = (x1 - x0) / Nx
        dy = (y1 - y0) / Ny
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return np.meshgrid(x, y)

    _contour_doc = """
        Plot contours.

        Call signature::

            contour([X, Y,] Z, [levels], **kwargs)

        `.contour` and `.contourf` draw contour lines and filled contours,
        respectively.  Except as noted, function signatures and return values
        are the same for both versions.

        Parameters
        ----------
        X, Y : array-like, optional
            The coordinates of the values in *Z*.

            *X* and *Y* must both be 2-D with the same shape as *Z* (e.g.
            created via `numpy.meshgrid`), or they must both be 1-D such
            that ``len(X) == M`` is the number of columns in *Z* and
            ``len(Y) == N`` is the number of rows in *Z*.

            If not given, they are assumed to be integer indices, i.e.
            ``X = range(M)``, ``Y = range(N)``.

        Z : array-like(N, M)
            The height values over which the contour is drawn.

        levels : int or array-like, optional
            Determines the number and positions of the contour lines / regions.

            If an int *n*, use *n* data intervals; i.e. draw *n+1* contour
            lines. The level heights are automatically chosen.

            If array-like, draw contour lines at the specified levels.
            The values must be in increasing order.

        Returns
        -------
        c : `~.contour.QuadContourSet`

        Other Parameters
        ----------------
        corner_mask : bool, optional
            Enable/disable corner masking, which only has an effect if *Z* is
            a masked array.  If ``False``, any quad touching a masked point is
            masked out.  If ``True``, only the triangular corners of quads
            nearest those points are always masked out, other triangular
            corners comprising three unmasked points are contoured as usual.

            Defaults to :rc:`contour.corner_mask`.

        colors : color string or sequence of colors, optional
            The colors of the levels, i.e. the lines for `.contour` and the
            areas for `.contourf`.

            The sequence is cycled for the levels in ascending order. If the
            sequence is shorter than the number of levels, it's repeated.

            As a shortcut, single color strings may be used in place of
            one-element lists, i.e. ``'red'`` instead of ``['red']`` to color
            all levels with the same color. This shortcut does only work for
            color strings, not for other ways of specifying colors.

            By default (value *None*), the colormap specified by *cmap*
            will be used.

        alpha : float, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        cmap : str or `.Colormap`, optional
            A `.Colormap` instance or registered colormap name. The colormap
            maps the level values to colors.
            Defaults to :rc:`image.cmap`.

            If both *colors* and *cmap* are given, an error is raised.

        norm : `~matplotlib.colors.Normalize`, optional
            If a colormap is used, the `.Normalize` instance scales the level
            values to the canonical colormap range [0, 1] for mapping to
            colors. If not given, the default linear scaling is used.

        vmin, vmax : float, optional
            If not *None*, either or both of these values will be supplied to
            the `.Normalize` instance, overriding the default color scaling
            based on *levels*.

        origin : {*None*, 'upper', 'lower', 'image'}, optional
            Determines the orientation and exact position of *Z* by specifying
            the position of ``Z[0, 0]``.  This is only relevant, if *X*, *Y*
            are not given.

            - *None*: ``Z[0, 0]`` is at X=0, Y=0 in the lower left corner.
            - 'lower': ``Z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
            - 'upper': ``Z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left
              corner.
            - 'image': Use the value from :rc:`image.origin`.

        extent : (x0, x1, y0, y1), optional
            If *origin* is not *None*, then *extent* is interpreted as in
            `.imshow`: it gives the outer pixel boundaries. In this case, the
            position of Z[0, 0] is the center of the pixel, not a corner. If
            *origin* is *None*, then (*x0*, *y0*) is the position of Z[0, 0],
            and (*x1*, *y1*) is the position of Z[-1,-1].

            This argument is ignored if *X* and *Y* are specified in the call
            to contour.

        locator : ticker.Locator subclass, optional
            The locator is used to determine the contour levels if they
            are not given explicitly via *levels*.
            Defaults to `~.ticker.MaxNLocator`.

        extend : {'neither', 'both', 'min', 'max'}, optional, default: \
'neither'
            Determines the ``contourf``-coloring of values that are outside the
            *levels* range.

            If 'neither', values outside the *levels* range are not colored.
            If 'min', 'max' or 'both', color the values below, above or below
            and above the *levels* range.

            Values below ``min(levels)`` and above ``max(levels)`` are mapped
            to the under/over values of the `.Colormap`. Note, that most
            colormaps do not have dedicated colors for these by default, so
            that the over and under values are the edge values of the colormap.
            You may want to set these values explicitly using
            `.Colormap.set_under` and `.Colormap.set_over`.

            .. note::

                An exising `.QuadContourSet` does not get notified if
                properties of its colormap are changed. Therefore, an explicit
                call `.QuadContourSet.changed()` is needed after modifying the
                colormap. The explicit call can be left out, if a colorbar is
                assigned to the `.QuadContourSet` because it internally calls
                `.QuadContourSet.changed()`.

            Example::

                x = np.arange(1, 10)
                y = x.reshape(-1, 1)
                h = x * y

                cs = plt.contourf(h, levels=[10, 30, 50],
                    colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
                cs.cmap.set_over('red')
                cs.cmap.set_under('blue')
                cs.changed()

        xunits, yunits : registered units, optional
            Override axis units by specifying an instance of a
            :class:`matplotlib.units.ConversionInterface`.

        antialiased : bool, optional
            Enable antialiasing, overriding the defaults.  For
            filled contours, the default is *True*.  For line contours,
            it is taken from :rc:`lines.antialiased`.

        nchunk : int >= 0, optional
            If 0, no subdivision of the domain.  Specify a positive integer to
            divide the domain into subdomains of *nchunk* by *nchunk* quads.
            Chunking reduces the maximum length of polygons generated by the
            contouring algorithm which reduces the rendering workload passed
            on to the backend and also requires slightly less RAM.  It can
            however introduce rendering artifacts at chunk boundaries depending
            on the backend, the *antialiased* flag and value of *alpha*.

        linewidths : float or sequence of float, optional
            *Only applies to* `.contour`.

            The line width of the contour lines.

            If a number, all levels will be plotted with this linewidth.

            If a sequence, the levels in ascending order will be plotted with
            the linewidths in the order specified.

            Defaults to :rc:`lines.linewidth`.

        linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
            *Only applies to* `.contour`.

            If *linestyles* is *None*, the default is 'solid' unless the lines
            are monochrome.  In that case, negative contours will take their
            linestyle from :rc:`contour.negative_linestyle` setting.

            *linestyles* can also be an iterable of the above strings
            specifying a set of linestyles to be used. If this
            iterable is shorter than the number of contour levels
            it will be repeated as necessary.

        hatches : List[str], optional
            *Only applies to* `.contourf`.

            A list of cross hatch patterns to use on the filled areas.
            If None, no hatching will be added to the contour.
            Hatching is supported in the PostScript, PDF, SVG and Agg
            backends only.

        Notes
        -----
        1. `.contourf` differs from the MATLAB version in that it does not draw
           the polygon edges. To draw edges, add line contours with calls to
           `.contour`.

        2. `.contourf` fills intervals that are closed at the top; that is, for
           boundaries *z1* and *z2*, the filled region is::

              z1 < Z <= z2

           except for the lowest interval, which is closed on both sides (i.e.
           it includes the lowest value).
        """

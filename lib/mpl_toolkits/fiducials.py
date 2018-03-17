"""
Fiducial/Landmark markers for Matplotlib.

Fiducial are numbered points markers (represented as cross) that are
generaly used for make correspondences between an image pixels and its
equivalent on a physical object. This is usefull for keep positions and
dimensions information between the object and its virtual representation.

Mouse & Keyboard interactions
-----------------------------
This fiducial tool-box for Matplotlib is made for be easy to use directly with
cursor and keyboard.

Create a new fiducial:
    Right-click on the wished position to create a new fiducial.

    If "img_regions_center" parameter is set, fiducial will be placed on the
    centroid of the image region currently bellow the cursor position.

    If not set, it will be placed directly on cursor position.

Delete a fiducial:
    Press DEL to delete all selected fiducials.

    Middle-click on fiducial to delete it.

Select one or more fiducial(s):
    Left-click on fiducial to select it;

    Left-click on no fiducial to clear selection;

    Holding CTRL while left-clicking on many fiducials to perform multiple
    selection. With CTRL held, left-clicking on an already selected fiducial
    unselect it;

    Holding SHIFT while left-clicking on two fiducials to select the entire
    range of fiducials with numbers between theses two fiducials numbers.

    Press CTRL+A to select all fiducials.

Move a Fiducial with cursor:
    Left-click and hold button on a fiducial to move it on wished position
    with cursor and releasing click.

Move selected Fiducial(s) accurately:
    Use arrows keys to move all selected fiducials pixel by pixel in the arrow
    direction.
"""
# TODO:
# - Add fiducial with circle, ellipse, polygon, square, rectangle, ...
# - Keyboard shortcuts: (en/dis)able autocenter, show/hide center
# - Fiducials reordering methods
# - Undo/Redo with CTRL+Z, CTRL+Y keyboard shortcuts

import numpy as np
from matplotlib.colors import colorConverter
from matplotlib import rcParams
try:
    import skimage.measure as measure
    _SKIMAGE_AVAILABLE = True
except ImportError:
    from warnings import warn
    _SKIMAGE_AVAILABLE = False
    warn('"scikit-image" package required for fiducial auto center on image '
         'regions ("img_regions_center" parameter.')


class Fiducials:
    def __init__(self, x, y, axes, color='b', editable=False, name='',
                 show_centroid=False, img_index=0, img_regions_center=None,
                 sync=None):
        """
        Matplotlib Fiducial/Landmark markers class.

        Parameters
        ----------
        x, y : list
            X, Y fiducials initials coordinates. Fiducial link theses
            coordinates as reference, so, are updated dynamically if fiducials
            change on the figure.
        axes : matplotlib.axes
            Axes where draw fiducials.
        color : matplotlib.colors
            Fiducials colors.
        editable : bool, optional
            If "True", Fiducials are editables with cursor and keyboard.
            Default is False.
        name : str
            Fiducials serie name, show near markers.
        show_centroid : bool, optional
            Show the fiducials centroid (As a "C" fiducial).
        img_index : int, optional
            Index of image linked to fiducials in "axes.images" list. Usefull
            only if multiple images on axes. This will select the image to use
            with the 'img_regions_center'
        img_regions_center : str or None, optional
            If not "None" and if at least one image on axes.
            Place new ficucials with cursor on image regions centroid instead
            of cursor coordinates. If "mask", compute regions on image mask (if
            no mask, create mask based on invalids pixels), if "data", compute
            regions on image intensity data.
            Default is "None". 'scikit-image' package required
        sync : list or tupple of Fiducials instances.
            If True, all fiducials instances in list will be updated when
            this instance is changed.
        """
        # Privates
        self._selectedmarkers = []
        self._selectionmode = 0
        self._dragmarker = None
        self._rcpfwd = False
        self._rcpbck = False
        self._todraw = []

        # Get mains attributes
        self.x = x
        self.y = y

        # Initialize artist
        self._artist = axes.scatter(self._x, self._y, 200, marker='+',
                                    picker=5)

        # Initialize Properties
        self.sync = sync
        self.name = name
        self.color = color
        self.img_regions_center = img_regions_center
        self.img_index = img_index
        self.editable = editable
        self.show_centroid = show_centroid  # also run self.update()

    @property
    def artist(self):
        """
        Fiducial artist (read only).
        """
        return self._artist

    @property
    def canvas(self):
        """
        The Canvas instance the Figure resides in (read only).
        """
        return self._artist.figure.canvas

    @property
    def figure(self):
        """
        The Figure instance the fiducials resides in (read only).
        """
        return self._artist.figure

    @property
    def axes(self):
        """
        The Axes instance the fiducials resides in (read only).
        """
        return self._artist.axes

    @property
    def color(self):
        """
        Fiducials color.
        """
        return self._color

    @color.setter
    def color(self, value):
        """
        Set fiducial color.
        """
        # Set color value
        self._color = colorConverter.to_rgba(value)

        # Set artist color
        self._artist.set_color(self._color)

        # Update ficucials
        self._update()

    @property
    def name(self):
        """
        Fiducials serie name.
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        set fiducials serie name.
        """
        # Set color value
        self._name = str(value)

    @property
    def sync(self):
        """
        List of fiducials instance to sync.
        """
        return self._sync

    @sync.setter
    def sync(self, value):
        """
        set list of fiducials instance to sync.
        """
        if value is None:
            self._sync = []
        else:
            for fid in value:
                if not isinstance(fid, Fiducials):
                    raise ValueError("sync must be a list or tupple of "
                                     "'Fiducials' instances")
            # Set color value
            self._sync = value

    @property
    def editable(self):
        """
        If True, Fiducials are editable with mousse and keyboard.
        """
        return self._editable

    @editable.setter
    def editable(self, value):
        """
        Set "editable" value and connect/disconnect events.
        """
        # Set bool value
        if not isinstance(value, bool):
            raise ValueError('bool required')
        self._editable = value

        # Init Connection ID list
        if not hasattr(self, '_eventconnections'):
            self._eventconnections = []

        # Connect/disconnect events
        if self._editable:
            # Connect events to canvas
            events = ('button_press_event', 'button_release_event',
                      'motion_notify_event', 'pick_event', 'key_press_event',
                      'key_release_event')
            for event in events:
                connection = self.canvas.\
                    mpl_connect(event, getattr(self, '_%s' % event))
                self._eventconnections.append(connection)
        else:
            # Disconnect events to canvas
            for connection in self._eventconnections:
                self.canvas.mpl_disconnect(connection)

    @property
    def img_regions_center(self):
        """
        If "mask", create fiducials on mask region centroid on click.

        If "data", create fiducials on data region centroid on click.

        If None, create fiducials on cursor position.
        """
        return self._img_regions_center

    @img_regions_center.setter
    def img_regions_center(self, value):
        """
        set "img_regions_center" value
        """
        error = '"data", "mask", or None required'
        if not _SKIMAGE_AVAILABLE:
            self._img_regions_center = None

        # Check and set value
        elif value is None:
            self._img_regions_center = None
        elif type(value) is not str:
            raise ValueError(error)
        else:
            value = value.lower()
            if value in ('data', 'mask'):
                self._img_regions_center = value
            else:
                raise ValueError(error)

        # Update regions
        self._update_img_regions()

    @property
    def centroid(self):
        """
        tuple (x, y) : Fiducial centroid X,Y coordinates (Read only) .
        """
        return self._centroid

    @property
    def show_centroid(self):
        """
        If True, show fiducials centroid (as a 'C' fiducial).
        """
        return self._show_centroid

    @show_centroid.setter
    def show_centroid(self, value):
        """
        set "show_centroid" value.
        """
        # Set bool value
        if type(value) is not bool:
            raise ValueError('bool required')
        self._show_centroid = value

        # Update ficucials
        self._update()

    @property
    def img_index(self):
        """
        Index of image linked to fiducials in "axes.images" list.
        """
        return self._img_index

    @img_index.setter
    def img_index(self, value):
        """
        Set index of image linked to fiducials in "axes.images" list.
        """
        # Set data
        int_val = int(value)
        axes_img_imax = len(self._artist.axes.images) - 1

        # Default value if no image
        if axes_img_imax == -1:
            self._img_index = 0

        # Check if good value
        elif int_val < 0 or int_val > axes_img_imax:
            raise IndexError('list index out of range')

        # Set value
        else:
            self._img_index = int_val

        # Update regions
        self._update_img_regions()

    @property
    def x(self):
        """
        X fiducials coordinates.
        """
        return self._x

    @x.setter
    def x(self, value):
        """
        set X coordinates.
        """
        # Set values
        if isinstance(value, list):
            self._x = value
        else:
            raise ValueError("Coordinates must be a list")

    @property
    def y(self):
        """
        Y fiducials coordinates.
        """
        return self._y

    @y.setter
    def y(self, value):
        """
        set Y coordinates.
        """
        # Set values
        if isinstance(value, list):
            self._y = value
        else:
            raise ValueError("Coordinates must be a list")

    def _update(self):
        """
        Update fiducials informations
        """
        # Cancel if artist not set
        if not hasattr(self, '_artist') or not hasattr(self, 'show_centroid'):
            return None

        # Remove previous labels
        if hasattr(self, '_labels'):
            for label in self._labels:
                label.remove()

        # Draw labels
        self._labels = []
        for i in range(len(self._x)):
            labxy = (self._x[i], self._y[i])
            self._labels.append(self.axes.annotate('  %s%d' % (self._name, i),
                                                   labxy, labxy,
                                                   color=self._color))

        # Update plot with fiducial X,Y lists
        xy = list(zip(self._x, self._y))

        # Update centroid
        self._updatecentroid()
        if self.show_centroid and len(xy) > 1:
            xy.append(self._centroid)
            self._labels.append(self.axes.annotate('  C',
                                                   self._centroid,
                                                   self._centroid,
                                                   color=self.color,
                                                   weight='bold'))

        # Set heavier linewidth for selected markers
        if self._labels:
            self._artist.set_linewidth([2.0 if x in self._selectedmarkers
                                        else 1.0 for x in range(len(xy))])

        # Update ficucials coordinates
        self._artist.set_offsets(xy)

        # update synched instances
        for fid in self._sync:
            if fid is self:
                continue
            fid._update()
            if fid.canvas is not self.canvas:
                self._todraw.append(fid.canvas)

    def _update_img_regions(self):
        """
        Update regions labels for linked image.
        """
        if (not hasattr(self, '_img_index') or
                not hasattr(self, '_img_regions_center') or
                len(self._artist.axes.images) == 0):
            self._img_regions = None
            return None

        if self._img_regions_center is None:
            self._img_regions = None
            return None

        # label image regions based on mask
        img = self._artist.axes.images[self._img_index].get_array()
        if self._img_regions_center == 'mask':
            img = np.ma.getmask(img)

        if img is np.False_:
            return None
        try:
            # Use scikit-image to compute regions
            # "+ 1" because 0 (background) ignored by regionprops
            self._img_regions = measure.label(img) + 1
        except NameError:
            # Scikit-image not available
            return None

    def draw(self):
        """
        Update fiducials informations and draw canvas.
        """
        # Remove previous labels
        self._update()

        # Draw plot
        self.canvas.draw_idle()

        # Update synched fiducials canvas
        for canvas in self._todraw:
            canvas.draw_idle()
        self._todraw.clear()

    def clear(self):
        """
        Remove all fiducials.
        """
        self._x.clear()
        self._y.clear()
        self._markerselection(0, 'clear')
        self._update()

    def _updatecentroid(self):
        """
        Update fiducials centroid X, Y coordinates.
        """
        if len(self._x) > 0:
            self._centroid = (np.mean(self._x), np.mean(self._y))
        else:
            # No data => no centroid
            self._centroid = (np.nan, np.nan)

    def _markerselection(self, index, mode='select'):
        """
        Select or unselect markers

        Parameters
        ----------
        index: int
            Marker index

        mode: str
            if mode = "select", add index to selection; if mode = "unselect",
            remove index to selection; if mode = "clear", clear all selected
            markers.
        """
        mode = mode.lower()
        if mode == "select":
            if index not in self._selectedmarkers:
                self._selectedmarkers.append(index)
        elif mode == "unselect":
            try:
                self._selectedmarkers.remove(index)
            except ValueError:
                pass
        elif mode == 'clear':
            self._selectedmarkers.clear()

        # Temporaly disable some Matplotlib keyboard shortcuts while fiducials
        # are selected to avoid conflics with arrow keys fiducial moves.
        if self._selectedmarkers:
            if not self._rcpfwd:
                try:
                    rcParams['keymap.forward'].remove('right')
                    self._rcpfwd = True
                except ValueError:
                    self._rcpfwd = False
            if not self._rcpbck:
                try:
                    rcParams['keymap.back'].remove('left')
                    self._rcpbck = True
                except ValueError:
                    self._rcpbck = False
        else:
            if self._rcpfwd:
                self._rcpfwd = False
                rcParams['keymap.forward'].append('right')
            if self._rcpbck:
                self._rcpbck = False
                rcParams['keymap.back'].append('left')

    def _button_press_event(self, event):
        """
        If self.editable is True, activate mousse button press actions.

        Actions
        =======
        Right-click:
            Add ficucial on cursor coordinates or on region centroid.
        """
        # Cancel if not on good axes
        if event.inaxes != self.axes:
            return None

        # Action to do
        if event.button == 3:
            # Create a new fiducial on cursor

            # Determinate X, Y coordinates
            if self.img_regions_center and self._img_regions is not None:
                # Use data region centroid coordinates
                reg = self._img_regions[event.ydata, event.xdata]
                y, x = measure.regionprops(self._img_regions)[reg - 1].centroid
            else:
                # Use cursor coordinates
                x, y = event.xdata, event.ydata

            # Add cursor position to fiducials coordinates
            self._x.append(x), self._y.append(y)

        elif event.button == 1 and not hasattr(event, '_disableevent'):
            if self._selectionmode == 0:
                # Clear selection if not CTRL or SHIFT
                self._markerselection(0, 'clear')
        else:
            # No Action
            return None

        # Update plot
        self.draw()

    def _button_release_event(self, event):
        """
        If self.editable is True, activate mousse button release actions.

        Actions
        =======
        Left-click:
            Stop moving fiducial with cursor.
        """
        if self._dragmarker is not None:
            # Stop moving with cursor
            self._dragmarker = None

        else:
            # No action
            return None

        # Update plot
        self.draw()

    def _motion_notify_event(self, event):
        """
        If self.editable is True, activate mousse motion actions.

        Actions
        =======
        Left-click:
            Move fiducial with cursor.
        """
        # Cancel if not on good axes
        if event.inaxes != self.axes:
            return None

        if self._dragmarker is not None:
            # Move with cursor
            self._x[self._dragmarker] = event.xdata
            self._y[self._dragmarker] = event.ydata

        else:
            # No action
            return None

        # Update plot
        self.draw()

    def _movemarker(self, direction, distance):
        """
        Move selected fiducials of distance in direction.

        Parameters
        ----------
        direction: str
            'x' or 'y.'

        distance: int
            Distance in number of pixels.
        """
        transform = self.axes.get_window_extent().\
            transformed(self.figure.dpi_scale_trans.inverted())

        # Compute way
        if direction == 'x':
            axe = self._x
            vmin, vmax = self.axes.get_xaxis().get_view_interval()
            size = transform.width
        else:
            axe = self._y
            vmin, vmax = self.axes.get_yaxis().get_view_interval()
            size = transform.height

        way = 1 if vmin < vmax else -1

        # Move
        movement = distance * way * abs(vmax - vmin) / (size * self.figure.dpi)
        for i in self._selectedmarkers:
            axe[i] += movement

        # Update plot
        self.draw()

    def _deletemarker(self, index):
        """
        Delete selected fiducials.

        Parameters
        ----------
        index: int
            Index of fiducial to delete.
        """
        # Remove marker from fiducials coordinates
        self._x.pop(index), self._y.pop(index)
        self._markerselection(index, 'unselect')

        # Update selected fiducials indexes
        for j in range(len(self._selectedmarkers)):
            if self._selectedmarkers[j] > index:
                self._selectedmarkers[j] -= 1

    def _key_press_event(self, event):
        """
        If self.editable is True, activate key press actions.

        Actions
        =======
        up, down, right, left:
            Move selected fiducials of one pixel in direction.

        ctrl:
            Activate multi-selection mode.

        shift:
            Activate range-selection mode.

        ctrl+a:
            Select all fiducials.

        delete:
            Delete selected fiducials.
        """
        # Cancel if not on good axes
        if event.inaxes != self.axes:
            return None

        # Move with arrows
        if event.key == 'up':
            self._movemarker('y', 1)
        elif event.key == 'down':
            self._movemarker('y', -1)
        elif event.key == 'right':
            self._movemarker('x', 1)
        elif event.key == 'left':
            self._movemarker('x', -1)

        # Enable selection mode
        elif event.key == 'control':
            self._selectionmode = 1
        elif event.key == 'shift':
            self._selectionmode = 2

        # Select all
        elif event.key == 'ctrl+a':
            self._selectedmarkers = [i for i in range(len(self._x))]
            self.draw()

        # Delete selected fiducials
        elif event.key == 'delete':
            for i in sorted(self._selectedmarkers, reverse=True):
                self._deletemarker(i)
            self.draw()

        # No action
        else:
            return None

    def _key_release_event(self, event):
        """
        If self.editable is True, activate key release actions.

        Actions
        =======
        ctrl, shift:
            Activate single-selection mode.
        """
        if event.key in ('control', 'shift'):
            # Disable selecton mode
            self._selectionmode = 0

    def _pick_event(self, event):
        """
        If self.editable is True, activate mousse button press actions.

        Actions
        =======
        Left-click:
            Start moving fiducial with cursor.

            Select/unselect picked fiducial or more depending selection mode.

        Middle-click:
            Remove picked fiducial.
        """
        # Cancel if not on good artist
        if event.artist is not self._artist:
            return None

        # Get indice of picked fiducial
        if self._show_centroid and event.ind[-1] == len(self._x):
            # If centroid picked, avoid to remove it
            if len(event.ind) <= 1:
                return None
            i = event.ind[-2]
        else:
            i = event.ind[-1]

        # Action to do
        if event.mouseevent.button == 1:
            # Select marker
            if self._selectionmode == 0:
                # Disable previous selection if not CTRL or SHIFT
                self._markerselection(0, 'clear')
                self._markerselection(i, 'select')
                event.mouseevent._disableevent = True

            elif self._selectionmode == 1:
                # Switch selection if CTRL
                if i not in self._selectedmarkers:
                    action = 'select'
                else:
                    action = 'unselect'
                self._markerselection(i, action)

            elif self._selectionmode == 2:
                # Select all the range if SHIFT
                self._markerselection(i, 'select')
                selmin = min(self._selectedmarkers)
                selmax = max(self._selectedmarkers) + 1
                self._markerselection(0, 'clear')
                for j in range(selmin, selmax):
                    self._markerselection(j, 'select')

            # Start moving with cursor
            self._dragmarker = i

        elif event.mouseevent.button == 2:
            # Delete marker
            self._deletemarker(i)

        else:
            # No Action
            return None

        # Update plot
        self.draw()

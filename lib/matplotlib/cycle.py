import itertools
import six
from matplotlib import rcParams


class Cycle:

    def __init__(self):
        """
        Set the initial cycle of styles to be used by the lines of the graph
        """
        self._styles = {'color': None,
                        'linestyle': None,
                        'linewidth': None,
                        'marker': None,
                        'markersize': None,
                        'markeredgewidth': None,
                        'markeredgecolor': None,
                        'markerfacecolor': None,
                        'antialiased': None,
                        'dash_capstyle': None,
                        'solid_capstyle': None,
                        'dash_joinstyle': None,
                        'solid_joinstyle': None,
                        }
        self._styles_list = {}
        self.set_color_cycle()
        self.set_line_cycle()

    def __getstate__(self):
        return {'_styles_list': self._styles_list}

    def __setstate__(self, state):
        self.__init__()
        self.__dict__.update(state)
        for style in self._styles_list.keys():
            self.set_cycle(style, self._styles_list[style])

    def next(self, args={}):
        """
        Returns the next set of line attributes for a line on the graph to use
        *args* is an optional dictionary of style arguments
        Styles that already exist in *args* will not be cycled through
        """
        args = args.copy()
        for style in self._styles.keys():
            if self._styles[style] != None and style not in args:
                args[style] = six.next(self._styles[style])
        return args

    def set_cycle(self, style, slist):
        """
        Set a cycle for a line attribute specified by style, the cycle to be
        used to is specified by slist

        *style* is a key to the _style dictionary
        *slist* is a list of mpl style specifiers
        """
        if self._validate(style, slist):
            self._styles_list[style] = slist
            self._styles[style] = itertools.cycle(slist)

    def _validate(self, style, slist):
        """
        Ensures that the style given is a valid attribute to by cycled over
        If the style is a valid cycle, ensure that the list of specifiers
        given are valid specifiers for that style

        *style* is a key to the _style dictionary
        *plist* is a list of mpl style specifiers
        """
        if type(slist) not in (list, tuple) or slist == []:
            msg = "'slist' must be of type [list | tuple ] and non empty"
            raise ValueError(msg)
        if style not in self._styles.keys():
            msg = "set cycle value error, %s is not a valid style" % style
            raise ValueError(msg)
        param = 'lines.' + style
        for val in slist:
            try:
                rcParams.validate[param](val)
            except ValueError:
                msg = "Set cycle value error, Style %s: %s" % (style, str(val))
                raise ValueError(msg)

        return True

    def clear_cycle(self, style):
        """
        Clears(resets) a cycle for a line attribute specified by style

        *style* is a key to the _style dictionary
        """
        if style not in self._styles.keys():
            msg = "clear cycle value error, %s is not a valid style" % style
            raise ValueError(msg)
        self._styles[style] = None

    def clear_all_cycle(self):
        """
        Clears (resets) all cycles for the lines on a plot
        """
        for style in self._styles.keys():
            self._styles[style] = None

    def set_line_cycle(self, llist=None):
        """
        Sets a line style cycle to be used for the lines on the graph, if none are
        specified the default line style cycle will be used
        """
        if llist is None:
            llist = rcParams['axes.line_cycle']
        self.set_cycle('linestyle', llist)

    def set_color_cycle(self, clist=None):
        """
        Sets a color cycle to be used for the lines on the graph, if none are
        specified the default color cycle will be used
        """
        if clist is None:
            clist = rcParams['axes.color_cycle']
        self.set_cycle('color', clist)

    def get_next_color(self):
        """
        Return the next color or defaults to rcParams if none
        """
        try:
            return six.next(self._styles['color'])
        except TypeError:
            return rcParams['lines.color']

    def get_next_linestyle(self):
        """
        Return the next linestyle or defaults to rcParams if none
        """
        try:
            return six.next(self._styles['linestyle'])
        except TypeError:
            return rcParams['lines.linestyle']

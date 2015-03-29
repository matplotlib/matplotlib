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
        self.set_color_cycle()
        self.set_line_cycle()

    def next(self):
        """
        Returns the next set of line attributes for a line on the graph to use
        """
        args = {}
        for i in self._styles.keys():
            if self._styles[i] != None:
                args[i] = six.next(self._styles[i])
        return args

    def set_cycle(self, style, slist):
        """
        Set a cycle for a line attribute specified by style, the cycle to be
        used to is specified by slist
        
        *style* is a key to the _style dictionary
        *slist* is a list of mpl style specifiers
        """
        if self.validate(style, slist):
            self._styles[style] = itertools.cycle(slist)

    def validate(self, style, plist):
        """
        Ensures that the style given is a valid attribute to by cycled over
        If the style is a valid cycle, ensure that the list of specifiers
        given are valid specifiers for that style
        
        *style* is a key to the _style dictionary
        *plist* is a list of mpl style specifiers
        """
        if style not in self._styles.keys():
            raise ValueError(
                "Set cycle value error, %s is not a valid style" % style)
        param = 'lines.' + style
        for val in set(plist):
            try:
                rcParams.validate[param](val)
            except ValueError:
                raise ValueError(
                    "Set cycle value error, Style %s: %s" % (style, str(val)))
        return True

    def clear_cycle(self, style):
        """
        Clears(resets) a cycle for a line attribute specified by style
        
        *style* is a key to the _style dictionary
        """
        self._styles[style] = None

    def clear_all_cycle(self):
        """
        Clears (resets) all cycles for the lines on a plot
        """
        for style in self._styles.keys():
            self._styles[style] = None

    def get_next_color(self):
        """
        Return the next color to be used by a line
        """
        return six.next(self._styles['color'])

    def set_color_cycle(self, clist=None):
        """
        Sets a color cycle to be used for the lines on the graph, if none are
        specified the default color cycle will be used
        """
        if clist is None:
            clist = rcParams['axes.color_cycle']
        self._styles['color'] = itertools.cycle(clist)

    def set_line_cycle(self, llist=None):
        """
        Sets a line style cycle to be used for the lines on the graph, if none are
        specified the default line style cycle will be used
        """
        if llist is None:
            llist = rcParams['axes.line_cycle']
        self._styles['linestyle'] = itertools.cycle(llist)

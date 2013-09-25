"""

Some gtk specific tools and widgets

   * rec2gtk          : put record array in GTK treeview - requires gtk

Example usage

    import matplotlib.mlab as mlab
    import mpl_toolkits.gtktools as gtktools

    r = mlab.csv2rec('somefile.csv', checkrows=0)

    formatd = dict(
        weight = mlab.FormatFloat(2),
        change = mlab.FormatPercent(2),
        cost   = mlab.FormatThousands(2),
        )


    exceltools.rec2excel(r, 'test.xls', formatd=formatd)
    mlab.rec2csv(r, 'test.csv', formatd=formatd)


    import gtk
    scroll = gtktools.rec2gtk(r, formatd=formatd)
    win = gtk.Window()
    win.set_size_request(600,800)
    win.add(scroll)
    win.show_all()
    gtk.main()

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import xrange, zip

import copy
import gtk, gobject
import numpy as npy
import matplotlib.cbook as cbook
import matplotlib.mlab as mlab


def error_message(msg, parent=None, title=None):
    """
    create an error message dialog with string msg.  Optionally set
    the parent widget and dialog title
    """

    dialog = gtk.MessageDialog(
        parent         = None,
        type           = gtk.MESSAGE_ERROR,
        buttons        = gtk.BUTTONS_OK,
        message_format = msg)
    if parent is not None:
        dialog.set_transient_for(parent)
    if title is not None:
        dialog.set_title(title)
    else:
        dialog.set_title('Error!')
    dialog.show()
    dialog.run()
    dialog.destroy()
    return None

def simple_message(msg, parent=None, title=None):
    """
    create a simple message dialog with string msg.  Optionally set
    the parent widget and dialog title
    """
    dialog = gtk.MessageDialog(
        parent         = None,
        type           = gtk.MESSAGE_INFO,
        buttons        = gtk.BUTTONS_OK,
        message_format = msg)
    if parent is not None:
        dialog.set_transient_for(parent)
    if title is not None:
        dialog.set_title(title)
    dialog.show()
    dialog.run()
    dialog.destroy()
    return None


def gtkformat_factory(format, colnum):
    """
    copy the format, perform any overrides, and attach an gtk style attrs


    xalign = 0.
    cell = None

    """
    if format is None: return None
    format = copy.copy(format)
    format.xalign = 0.
    format.cell = None

    def negative_red_cell(column, cell, model, thisiter):
        val = model.get_value(thisiter, colnum)
        try: val = float(val)
        except: cell.set_property('foreground', 'black')
        else:
            if val<0:
                cell.set_property('foreground', 'red')
            else:
                cell.set_property('foreground', 'black')


    if isinstance(format, mlab.FormatFloat) or isinstance(format, mlab.FormatInt):
        format.cell = negative_red_cell
        format.xalign = 1.
    elif isinstance(format, mlab.FormatDate):
        format.xalign = 1.
    return format



class SortedStringsScrolledWindow(gtk.ScrolledWindow):
    """
    A simple treeview/liststore assuming all columns are strings.
    Supports ascending/descending sort by clicking on column header
    """

    def __init__(self, colheaders, formatterd=None):
        """
        xalignd if not None, is a dict mapping col header to xalignent (default 1)

        formatterd if not None, is a dict mapping col header to a ColumnFormatter
        """


        gtk.ScrolledWindow.__init__(self)
        self.colheaders = colheaders
        self.seq = None # not initialized with accts
        self.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.set_policy(gtk.POLICY_AUTOMATIC,
                        gtk.POLICY_AUTOMATIC)

        types = [gobject.TYPE_STRING] * len(colheaders)
        model = self.model = gtk.ListStore(*types)


        treeview = gtk.TreeView(self.model)
        treeview.show()
        treeview.get_selection().set_mode(gtk.SELECTION_MULTIPLE)
        treeview.set_rules_hint(True)


        class Clicked:
            def __init__(self, parent, i):
                self.parent = parent
                self.i = i
                self.num = 0

            def __call__(self, column):
                ind = []
                dsu = []
                for rownum, thisiter in enumerate(self.parent.iters):
                    val = model.get_value(thisiter, self.i)
                    try: val = float(val.strip().rstrip('%'))
                    except ValueError: pass
                    if mlab.safe_isnan(val): val = npy.inf # force nan to sort uniquely
                    dsu.append((val, rownum))
                dsu.sort()
                if not self.num%2: dsu.reverse()

                vals, otherind = list(zip(*dsu))
                ind.extend(otherind)

                self.parent.model.reorder(ind)
                newiters = []
                for i in ind:
                    newiters.append(self.parent.iters[i])
                self.parent.iters = newiters[:]
                for i, thisiter in enumerate(self.parent.iters):
                    key = tuple([self.parent.model.get_value(thisiter, j) for j in range(len(colheaders))])
                    self.parent.rownumd[i] = key

                self.num+=1


        if formatterd is None:
            formatterd = dict()

        formatterd = formatterd.copy()

        for i, header in enumerate(colheaders):
            renderer = gtk.CellRendererText()
            if header not in formatterd:
                formatterd[header] = ColumnFormatter()
            formatter = formatterd[header]

            column = gtk.TreeViewColumn(header, renderer, text=i)
            renderer.set_property('xalign', formatter.xalign)
            renderer.set_property('editable', True)
            renderer.connect("edited", self.position_edited, i)
            column.connect('clicked', Clicked(self, i))
            column.set_property('clickable', True)

            if formatter.cell is not None:
                column.set_cell_data_func(renderer, formatter.cell)

            treeview.append_column(column)



        self.formatterd = formatterd
        self.lastcol = column
        self.add(treeview)
        self.treeview = treeview
        self.clear()

    def position_edited(self, renderer, path, newtext, position):
        #print path, position
        self.model[path][position] = newtext

    def clear(self):
        self.iterd = dict()
        self.iters = []        # an ordered list of iters
        self.rownumd = dict()  # a map from rownum -> symbol
        self.model.clear()
        self.datad = dict()


    def flat(self, row):
        seq = []
        for i,val in enumerate(row):
            formatter = self.formatterd.get(self.colheaders[i])
            seq.extend([i,formatter.tostr(val)])
        return seq

    def __delete_selected(self, *unused): # untested


        keyd = dict([(thisiter, key) for key, thisiter in self.iterd.values()])
        for row in self.get_selected():
            key = tuple(row)
            thisiter = self.iterd[key]
            self.model.remove(thisiter)
            del self.datad[key]
            del self.iterd[key]
            self.iters.remove(thisiter)

        for i, thisiter in enumerate(self.iters):
            self.rownumd[i] = keyd[thisiter]



    def delete_row(self, row):
        key = tuple(row)
        thisiter = self.iterd[key]
        self.model.remove(thisiter)


        del self.datad[key]
        del self.iterd[key]
        self.rownumd[len(self.iters)] = key
        self.iters.remove(thisiter)

        for rownum, thiskey in list(six.iteritems(self.rownumd)):
            if thiskey==key: del self.rownumd[rownum]

    def add_row(self, row):
        thisiter = self.model.append()
        self.model.set(thisiter, *self.flat(row))
        key = tuple(row)
        self.datad[key] = row
        self.iterd[key] = thisiter
        self.rownumd[len(self.iters)] = key
        self.iters.append(thisiter)

    def update_row(self, rownum, newrow):
        key = self.rownumd[rownum]
        thisiter = self.iterd[key]
        newkey = tuple(newrow)

        self.rownumd[rownum] = newkey
        del self.datad[key]
        del self.iterd[key]
        self.datad[newkey] = newrow
        self.iterd[newkey] = thisiter


        self.model.set(thisiter, *self.flat(newrow))

    def get_row(self, rownum):
        key = self.rownumd[rownum]
        return self.datad[key]

    def get_selected(self):
        selected = []
        def foreach(model, path, iter, selected):
            selected.append(model.get_value(iter, 0))

        self.treeview.get_selection().selected_foreach(foreach, selected)
        return selected



def rec2gtk(r, formatd=None, rownum=0, autowin=True):
    """
    formatd is a dictionary mapping dtype name -> mlab.Format instances

    This function creates a SortedStringsScrolledWindow (derived
    from gtk.ScrolledWindow) and returns it.  if autowin is True,
    a gtk.Window is created, attached to the
    SortedStringsScrolledWindow instance, shown and returned.  If
    autowin=False, the caller is responsible for adding the
    SortedStringsScrolledWindow instance to a gtk widget and
    showing it.
    """



    if formatd is None:
        formatd = dict()

    formats = []
    for i, name in enumerate(r.dtype.names):
        dt = r.dtype[name]
        format = formatd.get(name)
        if format is None:
            format = mlab.defaultformatd.get(dt.type, mlab.FormatObj())
        #print 'gtk fmt factory', i, name, format, type(format)
        format = gtkformat_factory(format, i)
        formatd[name] = format


    colheaders = r.dtype.names
    scroll = SortedStringsScrolledWindow(colheaders, formatd)

    ind = npy.arange(len(r.dtype.names))
    for row in r:
        scroll.add_row(row)


    if autowin:
        win = gtk.Window()
        win.set_default_size(800,600)
        #win.set_geometry_hints(scroll)
        win.add(scroll)
        win.show_all()
        scroll.win = win

    return scroll


class RecListStore(gtk.ListStore):
    """
    A liststore as a model of an editable record array.

    attributes:

     * r - the record array with the edited values

     * formatd - the list of mlab.FormatObj instances, with gtk attachments

     * stringd - a dict mapping dtype names to a list of valid strings for the combo drop downs

     * callbacks - a matplotlib.cbook.CallbackRegistry.  Connect to the cell_changed with

        def mycallback(liststore, rownum, colname, oldval, newval):
           print('verify: old=%s, new=%s, rec=%s'%(oldval, newval, liststore.r[rownum][colname]))

        cid = liststore.callbacks.connect('cell_changed', mycallback)

        """
    def __init__(self, r, formatd=None, stringd=None):
        """
        r is a numpy record array

        formatd is a dict mapping dtype name to mlab.FormatObj instances

        stringd, if not None, is a dict mapping dtype names to a list of
        valid strings for a combo drop down editor
        """

        if stringd is None:
            stringd = dict()

        if formatd is None:
            formatd = mlab.get_formatd(r)

        self.stringd = stringd
        self.callbacks = cbook.CallbackRegistry(['cell_changed'])

        self.r = r

        self.headers = r.dtype.names
        self.formats = [gtkformat_factory(formatd.get(name, mlab.FormatObj()),i)
                        for i,name in enumerate(self.headers)]

        # use the gtk attached versions
        self.formatd = formatd = dict(zip(self.headers, self.formats))
        types = []
        for format in self.formats:
            if isinstance(format, mlab.FormatBool):
                types.append(gobject.TYPE_BOOLEAN)
            else:
                types.append(gobject.TYPE_STRING)

        self.combod = dict()
        if len(stringd):
            types.extend([gobject.TYPE_INT]*len(stringd))

            keys = list(six.iterkeys(stringd))
            keys.sort()

            valid = set(r.dtype.names)
            for ikey, key in enumerate(keys):
                assert(key in valid)
                combostore = gtk.ListStore(gobject.TYPE_STRING)
                for s in stringd[key]:
                    combostore.append([s])
                self.combod[key] = combostore, len(self.headers)+ikey


        gtk.ListStore.__init__(self, *types)

        for row in r:
            vals = []
            for formatter, val in zip(self.formats, row):
                if isinstance(formatter, mlab.FormatBool):
                    vals.append(val)
                else:
                    vals.append(formatter.tostr(val))
            if len(stringd):
                # todo, get correct index here?
                vals.extend([0]*len(stringd))
            self.append(vals)


    def position_edited(self, renderer, path, newtext, position):

        position = int(position)
        format = self.formats[position]

        rownum = int(path)
        colname = self.headers[position]
        oldval = self.r[rownum][colname]
        try: newval = format.fromstr(newtext)
        except ValueError:
            msg = cbook.exception_to_str('Error converting "%s"'%newtext)
            error_message(msg, title='Error')
            return
        self.r[rownum][colname] = newval

        self[path][position] = format.tostr(newval)


        self.callbacks.process('cell_changed', self, rownum, colname, oldval, newval)

    def position_toggled(self, cellrenderer, path, position):
        position = int(position)
        format = self.formats[position]

        newval = not cellrenderer.get_active()

        rownum = int(path)
        colname = self.headers[position]
        oldval = self.r[rownum][colname]
        self.r[rownum][colname] = newval

        self[path][position] = newval

        self.callbacks.process('cell_changed', self, rownum, colname, oldval, newval)





class RecTreeView(gtk.TreeView):
    """
    An editable tree view widget for record arrays
    """
    def __init__(self, recliststore, constant=None):
        """
        build a gtk.TreeView to edit a RecListStore

        constant, if not None, is a list of dtype names which are not editable
        """
        self.recliststore = recliststore

        gtk.TreeView.__init__(self, recliststore)

        combostrings = set(recliststore.stringd.keys())


        if constant is None:
            constant = []

        constant = set(constant)

        for i, header in enumerate(recliststore.headers):
            formatter = recliststore.formatd[header]
            coltype =  recliststore.get_column_type(i)

            if coltype==gobject.TYPE_BOOLEAN:
                renderer = gtk.CellRendererToggle()
                if header not in constant:
                    renderer.connect("toggled", recliststore.position_toggled, i)
                    renderer.set_property('activatable', True)

            elif header in combostrings:
                 renderer = gtk.CellRendererCombo()
                 renderer.connect("edited", recliststore.position_edited, i)
                 combostore, listind = recliststore.combod[header]
                 renderer.set_property("model", combostore)
                 renderer.set_property('editable', True)
            else:
                renderer = gtk.CellRendererText()
                if header not in constant:
                    renderer.connect("edited", recliststore.position_edited, i)
                    renderer.set_property('editable', True)


                if formatter is not None:
                    renderer.set_property('xalign', formatter.xalign)



            tvcol = gtk.TreeViewColumn(header)
            self.append_column(tvcol)
            tvcol.pack_start(renderer, True)

            if coltype == gobject.TYPE_STRING:
                tvcol.add_attribute(renderer, 'text', i)
                if header in combostrings:
                    combostore, listind = recliststore.combod[header]
                    tvcol.add_attribute(renderer, 'text-column', listind)
            elif coltype == gobject.TYPE_BOOLEAN:
                tvcol.add_attribute(renderer, 'active', i)


            if formatter is not None and formatter.cell is not None:
                tvcol.set_cell_data_func(renderer, formatter.cell)




        self.connect("button-release-event", self.on_selection_changed)
        #self.set_grid_lines(gtk.TREE_VIEW_GRID_LINES_BOTH)

        self.get_selection().set_mode(gtk.SELECTION_BROWSE)
        self.get_selection().set_select_function(self.on_select)


    def on_select(self, *args):
        return False

    def on_selection_changed(self, *args):
        (path, col) = self.get_cursor()
        ren = col.get_cell_renderers()[0]
        if isinstance(ren, gtk.CellRendererText):
            self.set_cursor_on_cell(path, col, ren, start_editing=True)

def edit_recarray(r, formatd=None, stringd=None, constant=None, autowin=True):
    """
    create a RecListStore and RecTreeView and return them.

    If autowin is True, create a gtk.Window, insert the treeview into
    it, and return it (return value will be (liststore, treeview, win)

    See RecListStore and RecTreeView for a description of the keyword args
    """

    liststore = RecListStore(r, formatd=formatd, stringd=stringd)
    treeview = RecTreeView(liststore, constant=constant)

    if autowin:
        win = gtk.Window()
        win.add(treeview)
        win.show_all()
        return liststore, treeview, win
    else:
        return liststore, treeview




if __name__=='__main__':

    import datetime
    import gtk
    import numpy as np
    import matplotlib.mlab as mlab
    N = 10
    today = datetime.date.today()
    dates = [today+datetime.timedelta(days=i) for i in range(N)] # datetimes
    weekdays = [d.strftime('%a') for d in dates]                 # strings
    gains = np.random.randn(N)                                   # floats
    prices = np.random.rand(N)*1e7                               # big numbers
    up = gains>0                                                 # bools
    clientid = list(xrange(N))                                   # ints

    r = np.rec.fromarrays([clientid, dates, weekdays, gains, prices, up],
                          names='clientid,date,weekdays,gains,prices,up')

    # some custom formatters
    formatd = mlab.get_formatd(r)
    formatd['date'] = mlab.FormatDate('%Y-%m-%d')
    formatd['prices'] = mlab.FormatMillions(precision=1)
    formatd['gain'] = mlab.FormatPercent(precision=2)

    # use a drop down combo for weekdays
    stringd = dict(weekdays=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    constant = ['clientid']   # block editing of this field


    liststore = RecListStore(r, formatd=formatd, stringd=stringd)
    treeview = RecTreeView(liststore, constant=constant)

    def mycallback(liststore, rownum, colname, oldval, newval):
        print('verify: old=%s, new=%s, rec=%s'%(oldval, newval, liststore.r[rownum][colname]))

    liststore.callbacks.connect('cell_changed', mycallback)

    win = gtk.Window()
    win.set_title('with full customization')
    win.add(treeview)
    win.show_all()

    # or you just use the defaults
    r2 = r.copy()
    ls, tv, win2 = edit_recarray(r2)
    win2.set_title('with all defaults')

    gtk.main()

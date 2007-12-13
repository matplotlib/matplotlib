"""

Some gtk specific tools and widgets

   * rec2gtk          : put record array in GTK treeview - requires gtk

Example usage

    import matplotlib.mlab as mlab
    import matplotlib.toolkits.gtktools as gtktools
    
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
import copy
import gtk, gobject
import numpy as npy
import matplotlib.cbook as cbook
import matplotlib.mlab as mlab

def gtkformat_factory(format, colnum):
    """
    copy the format, perform any overrides, and attach an gtk style attrs


    xalign = 0.
    cell = None

    """

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
                    if npy.isnan(val): val = npy.inf # force nan to sort uniquely
                    dsu.append((val, rownum))
                dsu.sort()
                if not self.num%2: dsu.reverse()

                vals, otherind = zip(*dsu)
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

        for rownum, thiskey in self.rownumd.items():
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
    save record array r to excel pyExcelerator worksheet ws
    starting at rownum.  if ws is string like, assume it is a
    filename and save to it

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
        win.add(scroll)
        win.show_all()
        scroll.win = win

    return scroll


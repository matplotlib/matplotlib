"""
GUI Neutral widgets

All of these widgets require you to predefine an Axes instance and
pass that as the first arg.  matplotlib doesn't try to be too smart in
layout -- you have to figure out how wide and tall you want your Axes
to be to accommodate your widget.
"""

from matplotlib.mlab import linspace, dist
from matplotlib.patches import Circle
from matplotlib.numerix import array

class Widget:
    """
    OK, I couldn't resist; abstract base class for mpl GUI neutral
    widgets
    """
    
    drawon = True
    eventson = True
        
        


class Button(Widget):
    """
    A GUI neutral button

    The following attributes are accesible

      ax    - the Axes the button renders into
      label - a text.Text instance
      color - the color of the button when not hovering
      hovercolor - the color of the button when hovering

    Call "on_clicked" to connect to the button
    """

    def __init__(self, ax, label, image=None,
                 color=0.85, hovercolor=0.95):
        """
        ax is the Axes instance the button will be placed into

        label is a string which is the button text

        image if not None, is an image to place in the button -- can
          be any legal arg to imshow (array, matplotlib Image
          instance, or PIL image)

        color is the color of the button when not activated

        hovercolor is the color of the button when the mouse is over
          it

        """
        if image is not None:
            ax.imshow(image)
        self.label = ax.text(0.5, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='center',
                             transform=ax.transAxes)

        self.cnt = 0
        self.observers = {}
        self.ax = ax

        def silent(*args):
            'turn off the coords reporting for ax'
            return ''
        ax.format_coord = silent
        
        ax.figure.canvas.mpl_connect('button_press_event', self._click)
        ax.figure.canvas.mpl_connect('motion_notify_event', self._motion)

        ax.set_axis_bgcolor(color)
        ax.set_xticks([])
        ax.set_yticks([])        
        self.color = color
        self.hovercolor = hovercolor        
        
        self._lastcolor = color

    def _click(self, event):
        if event.inaxes != self.ax: return
        if not self.eventson: return
        for cid, func in self.observers.items():
            func(event)

    def _motion(self, event):
        if event.inaxes==self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.set_axis_bgcolor(c)
            self._lastcolor = c
            if self.drawon: self.ax.figure.canvas.draw()
        
    def on_clicked(self, func):
        """
        When the button is clicked, call this func with event

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1        
        return cid

    def disconnect(self, cid):
        'remove the observer with connection id cid'
        try: del self.observers[cid]
        except KeyError: pass
        
        

class Slider(Widget):
    """
    A slider representing a floating point range

    The following attributes are defined 
      ax     : the slider axes.Axes instance
      val    : the current slider value
      vline  : a Line2D instance representing the initial value
      poly   : A patch.Polygon instance which is the slider
      valfmt : the format string for formatting the slider text
      label  : a text.Text instance, the slider label
      closedmin : whether the slider is closed on the minimum 
      closedmax : whether the slider is closed on the maximum
      slidermin : another slider - if not None, this slider must be > slidermin
      slidermax : another slider - if not None, this slider must be < slidermax

      
    Call on_changed to connect to the slider event
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None, slidermax=None):
        """
        Create a slider from valmin to valmax in axes ax;

        label is the slider label valinit is the slider initial position

        valfmt is used to format the slider value

        closedmin and closedmax indicated whether the slider interval is closed

        slidermin and slidermax can be used to contrain the value of
          this slider to the values of other sliders.
          """
        self.ax = ax
        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axvspan(valmin,valinit,0,1)
        
        self.vline = ax.axvline(valinit,0,1, color='r', lw=1)

        
        self.valfmt=valfmt
        ax.set_yticks([])
        ax.set_xlim((valmin, valmax))
        ax.set_xticks([])
        
        ax.figure.canvas.mpl_connect('button_press_event', self._update)
        self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                             verticalalignment='center',
                             horizontalalignment='right')

        self.valtext = ax.text(1.02, 0.5, valfmt%valinit,
                               transform=ax.transAxes,
                               verticalalignment='center',
                               horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax


    def _update(self, event):
        'update the slider position'
        if event.button !=1: return
        if event.inaxes != self.ax: return
        val = event.xdata
        if not self.closedmin and val<=self.valmin: return
        if not self.closedmax and val>=self.valmax: return        

        if self.slidermin is not None:
            if val<=self.slidermin.val: return

        if self.slidermax is not None:
            if val>=self.slidermax.val: return

        self._set_val(val)
        
    def _set_val(self, val):
        self.poly.xy[-1] = val, 0
        self.poly.xy[-2] = val, 1        
        self.valtext.set_text(self.valfmt%val)
        if self.drawon: self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: return
        for cid, func in self.observers.items():
            func(val)


        
    def on_changed(self, func):
        """
        When the slider valud is changed, call this func with the new
        slider position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1        
        return cid

    def disconnect(self, cid):
        'remove the observer with connection id cid'
        try: del self.observers[cid]
        except KeyError: pass

    def reset(self):
        "reset the slider to the initial value"
        self._set_val(self.valinit)        





class RadioButtons(Widget):
    """
    A GUI neutral radio button

    The following attributes are exposed

     ax - the Axes instance the buttons are in
     activecolor - the color of the button when clicked
     labels - a list of text.Text instances
     circles - a list of patch.Circle instances

    Connect to the RadioButtons with the on_clicked method
    """
    def __init__(self, ax, labels, active=0, activecolor='blue'):
        """
        Add radio buttons to axes.Axes instance ax

        labels is a len(buttons) list of labels as strings

        active is the index into labels for the button that is active

        activecolor is the color of the button when clicked
        """
        self.activecolor = activecolor

        def silent(*args):
            'turn off the coords reporting for ax'
            return ''
        ax.format_coord = silent
        

        ax.set_xticks([])
        ax.set_yticks([])
        
        dy = 1./(len(labels)+1)
        ys = linspace(1-dy, dy, len(labels))
        cnt = 0
        axcolor = ax.get_axis_bgcolor()

        self.labels = []
        self.circles = []
        for y, label in zip(ys, labels):
            t = ax.text(0.25, y, label, transform=ax.transAxes,
                        horizontalalignment='left',
                        verticalalignment='center')

            if cnt==active:
                facecolor = activecolor
            else:
                facecolor = axcolor
                
            p = Circle(xy=(0.15, y), radius=0.05, facecolor=facecolor,
                       transform=ax.transAxes)

            
            self.labels.append(t)
            self.circles.append(p)
            ax.add_patch(p)
            cnt += 1

        ax.figure.canvas.mpl_connect('button_press_event', self._clicked)
        self.ax = ax


        self.cnt = 0
        self.observers = {}
        
    def _clicked(self, event):
        if event.button !=1 : return
        if event.inaxes != self.ax: return
        xy = self.ax.transAxes.inverse_xy_tup((event.x, event.y))
        pclicked = array([xy[0], xy[1]])
        def inside(p):
            pcirc = array([p.center[0], p.center[1]])
            return dist(pclicked, pcirc) < p.radius

        for p,t in zip(self.circles, self.labels):
            if t.get_window_extent().contains(event.x, event.y) or inside(p):
                inp = p
                thist = t
                break
        else: return

        for p in self.circles:
            if p==inp: color = self.activecolor
            else: color = self.ax.get_axis_bgcolor()
            p.set_facecolor(color)



        if self.drawon: self.ax.figure.canvas.draw() 

        if not self.eventson: return
        for cid, func in self.observers.items():
            func(thist.get_text())


    def on_clicked(self, func):
        """
        When the button is clicked, call this func with button label

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1        
        return cid

    def disconnect(self, cid):
        'remove the observer with connection id cid'
        try: del self.observers[cid]
        except KeyError: pass
                
class SubplotTool(Widget):
    """
    A tool to adjust to subplot params of fig
    """
    def __init__(self, targetfig, toolfig):
        """
        targetfig is the figure to adjust

        toolfig is the figure to embed the the subplot tool into.  If
        None, a default pylab figure will be created.  If you are
        using this from the GUI
        """
        self.targetfig = targetfig



        toolfig.subplots_adjust(left=0.2, right=0.9)

        class toolbarfmt:
            def __init__(self, slider):
                self.slider = slider
                
            def __call__(self, x, y):
                fmt = '%s=%s'%(self.slider.label.get_text(), self.slider.valfmt)
                return fmt%x

        self.axleft = toolfig.add_subplot(711)
        self.axleft.set_title('Click on slider to adjust subplot param')
        
        self.sliderleft = Slider(self.axleft, 'left', 0, 1, targetfig.subplotpars.left, closedmax=False)
        self.sliderleft.on_changed(self.funcleft)
        self.axleft.format_coord = toolbarfmt(self.sliderleft)


        self.axbottom = toolfig.add_subplot(712)
        self.sliderbottom = Slider(self.axbottom, 'bottom', 0, 1, targetfig.subplotpars.bottom, closedmax=False)
        self.sliderbottom.on_changed(self.funcbottom)
        self.axbottom.format_coord = toolbarfmt(self.sliderbottom)
        
        self.axright = toolfig.add_subplot(713)
        self.sliderright = Slider(self.axright, 'right', 0, 1, targetfig.subplotpars.right, closedmin=False)
        self.sliderright.on_changed(self.funcright)
        self.axright.format_coord = toolbarfmt(self.sliderright)
        
        self.axtop = toolfig.add_subplot(714)
        self.slidertop = Slider(self.axtop, 'top', 0, 1, targetfig.subplotpars.top, closedmin=False)
        self.slidertop.on_changed(self.functop)
        self.axtop.format_coord = toolbarfmt(self.slidertop)

        
        self.axwspace = toolfig.add_subplot(715)
        self.sliderwspace = Slider(self.axwspace, 'wspace', 0, 1, targetfig.subplotpars.wspace, closedmax=False)
        self.sliderwspace.on_changed(self.funcwspace)
        self.axwspace.format_coord = toolbarfmt(self.sliderwspace)
        
        self.axhspace = toolfig.add_subplot(716)
        self.sliderhspace = Slider(self.axhspace, 'hspace', 0, 1, targetfig.subplotpars.hspace, closedmax=False)
        self.sliderhspace.on_changed(self.funchspace)
        self.axhspace.format_coord = toolbarfmt(self.sliderhspace)


        # constraints
        self.sliderleft.slidermax = self.sliderright
        self.sliderright.slidermin = self.sliderleft
        self.sliderbottom.slidermax = self.slidertop
        self.slidertop.slidermin = self.sliderbottom
        

        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonrestore = Button(bax, 'Restore')

        sliders = (self.sliderleft, self.sliderbottom, self.sliderright,
                   self.slidertop, self.sliderwspace, self.sliderhspace, )

        def func(event):
            thisdrawon = self.drawon

            self.drawon = False

            # store the drawon state of each slider
            bs = [] 
            for slider in sliders:
                bs.append(slider.drawon)
                slider.drawon = False

            # reset the slider to the initial position
            for slider in sliders:
                slider.reset()

            # reset drawon
            for slider, b in zip(sliders, bs):
                slider.drawon = b

            # draw the canvas
            self.drawon = thisdrawon
            if self.drawon:
                toolfig.canvas.draw()
                self.targetfig.canvas.draw()

                
        self.buttonrestore.on_clicked(func)


    def funcleft(self, val):
        self.targetfig.subplots_adjust(left=val)
        if self.drawon: self.targetfig.canvas.draw()

    def funcright(self, val):
        self.targetfig.subplots_adjust(right=val)
        if self.drawon: self.targetfig.canvas.draw()

    def funcbottom(self, val):
        self.targetfig.subplots_adjust(bottom=val)
        if self.drawon: self.targetfig.canvas.draw()

    def functop(self, val):
        self.targetfig.subplots_adjust(top=val)
        if self.drawon: self.targetfig.canvas.draw()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val)
        if self.drawon: self.targetfig.canvas.draw()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val)
        if self.drawon: self.targetfig.canvas.draw()

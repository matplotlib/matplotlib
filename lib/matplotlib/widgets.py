"""
GUI Neutral widgets
"""
class Widget:
    """
    OK, I couldn't resist; abstract base class for mpl GUI neutral
    widgets
    """
    


class Button(Widget):
    def __init__(self, ax, label, image=None,
                 color=0.85, hovercolor=0.95):

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
        
        ax.figure.canvas.mpl_connect('button_press_event', self.click)
        ax.figure.canvas.mpl_connect('motion_notify_event', self.motion)

        ax.set_axis_bgcolor(color)
        ax.set_xticks([])
        ax.set_yticks([])        
        self.color = color
        self.hovercolor = hovercolor        
        
        self._lastcolor = color

    def click(self, event):
        if event.inaxes != self.ax: return
        for cid, func in self.observers.items():
            func(event)

    def motion(self, event):
        if event.inaxes==self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.set_axis_bgcolor(c)
            self._lastcolor = c
            self.ax.figure.canvas.draw()
        
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
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None, slidermax=None):
        """
        Create a slider from valmin to valmax in axes ax;

        label is the slider label valinit is the slider initial position
        valfmt is used to format the slider value closedmin and
        closedmax indicated whether the slider interval is open or
        closed

        Attributes exposed are
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
        
        ax.figure.canvas.mpl_connect('button_press_event', self.update)
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

    def update(self, event):
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
        self.ax.figure.canvas.draw()
        for cid, func in self.observers.items():
            func(val)

        self.val = val
        
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

class SubplotTool:
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
            print 'clicked'
            for slider in sliders:
                slider.reset()
                
        self.buttonrestore.on_clicked(func)


    def funcleft(self, val):
        self.targetfig.subplots_adjust(left=val)
        self.targetfig.canvas.draw()

    def funcright(self, val):
        self.targetfig.subplots_adjust(right=val)
        self.targetfig.canvas.draw()

    def funcbottom(self, val):
        self.targetfig.subplots_adjust(bottom=val)
        self.targetfig.canvas.draw()

    def functop(self, val):
        self.targetfig.subplots_adjust(top=val)
        self.targetfig.canvas.draw()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val)
        self.targetfig.canvas.draw()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val)
        self.targetfig.canvas.draw()

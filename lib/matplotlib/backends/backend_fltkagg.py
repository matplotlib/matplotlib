"""
A backend for FLTK

Copyright: Gregory Lielens, Free Field Technologies SA and
           John D. Hunter 2004

This code is released under the matplotlib license

"""

from __future__ import division

import os, sys, math

import fltk as Fltk
from backend_agg import FigureCanvasAgg

import os.path

import matplotlib

from matplotlib import rcParams, verbose
from matplotlib.cbook import is_string_like,  enumerate
from matplotlib.backend_bases import \
     RendererBase, GraphicsContextBase, FigureManagerBase, FigureCanvasBase,\
     NavigationToolbar2, cursors, MplEvent
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
from matplotlib.numerix import asarray
import matplotlib.windowing as windowing


import thread,time

Fl_running=thread.allocate_lock()
def Fltk_run_interactive():
    global Fl_running
    if Fl_running.acquire(0):
      while True:
        Fltk.Fl.check()
        time.sleep(0.005)
    else:
        print "fl loop already running"  

# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 75

cursord= {  
    cursors.HAND:     Fltk.FL_CURSOR_HAND,
    cursors.POINTER:     Fltk.FL_CURSOR_ARROW,
    cursors.SELECT_REGION:   Fltk.FL_CURSOR_CROSS,
    cursors.MOVE:   Fltk.FL_CURSOR_MOVE
    }

def error_msg_fltk(msg, parent=None):
    Fltk.fl_message(msg)
     

def draw_if_interactive():
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw()

def show(mainloop=False):
    """
    Show all the figures and enter the fltk mainloop

    This should be the last line of your script
    """
    for manager in Gcf.get_all_fig_managers():
        manager.show()
    #mainloop, if an fltk program exist no need to call that
    #threaded (and interractive) version    
    if show._needmain and mainloop:
        Fltk.Fl.run()
        show._needmain = False
    else:
        thread.start_new_thread(Fltk_run_interactive,())
        show._needmain = False
show._needmain = True


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    figure = Figure(*args, **kwargs)
    canvas = FigureCanvasFltkAgg(figure)   
    window = Fltk.Fl_Double_Window(10,10,30,30)
    window.end()    
    figManager = FigureManagerFltkAgg(canvas, num, window)
    return figManager
    
def fltk_event2mpl_event(s,source, need_pos=False, need_button=False, need_key=False):
    special={Fltk.FL_Shift_R:'shift',
             Fltk.FL_Shift_L:'shift',
             Fltk.FL_Control_R:'control',
             Fltk.FL_Control_L:'control',
             Fltk.FL_Control_R:'control',
             Fltk.FL_Control_L:'control',
             65515:'win',    
             65516:'win',    
             }
                 
    thisEvent = MplEvent(s,source) 
    thisEvent.x = Fltk.Fl.event_x()
    # flipy so y=0 is bottom of canvas
    thisEvent.y = source.figure.bbox.height() - Fltk.Fl.event_y()
    if need_pos:  
        thisEvent.inaxes = None
        for a in source.figure.get_axes():
            if a.in_axes(thisEvent.x, thisEvent.y):
                thisEvent.inaxes = a
                xdata, ydata = a.transData.inverse_xy_tup((thisEvent.x, thisEvent.y))
                thisEvent.xdata  = xdata
                thisEvent.ydata  = ydata
                break
                
    if need_button:
        b1=Fltk.Fl.event_button1()
        b2=Fltk.Fl.event_button2()
        b3=Fltk.Fl.event_button3()
        thisEvent.button = None
        if b1:
         thisEvent.button = 1
        if b2:
         thisEvent.button = 2
        if b3:
         thisEvent.button = 3
    else:
        thisEvent.button = None
    
    ikey= Fltk.Fl.event_key()   
    if need_key and Fltk.Fl.event_key(ikey):  
        if(ikey<=255):
          thisEvent.key=chr(ikey)
        else:
          try:  
            thisEvent.key=special[ikey]   
          except:
            thisEvent.key=None  
    else:
        thisEvent.key=None
    return thisEvent  

class FltkCanvas(Fltk.Fl_Widget):
     
    def __init__(self,x,y,w,h,l,source):
        Fltk.Fl_Widget.__init__(self, 0, 0, w, h, "canvas")
        self._source=source
        self._oldsize=(None,None) 
        self.button_press_event=None
        self.button_double_press_event=None
        self.button_release_event=None
        self.button_drag_event=None
        self.motion_notify_event=None
        self.key_press_event=None
        self.key_release_event=None
        self._draw_overlay = False
        
        
    def draw(self): 
        newsize=(self.w(),self.h())
        if(self._oldsize !=newsize):
            self._oldsize =newsize
            self._source.resize(newsize)
            self._source.draw()
        t1,t2,w,h = self._source.figure.bbox.get_bounds()
        Fltk.fl_draw_image(self._source.buffer_rgba(),0,0,int(w),int(h),4,0)
                    
    def handle(self, event):
        self.window().make_current()
        if event == Fltk.FL_PUSH:
            if self._draw_overlay:
                self._oldx=Fltk.Fl.event_x()
                self._oldy=Fltk.Fl.event_y()
            if Fltk.Fl.event_clicks() and self.button_double_press_event:
                self.double_press_event(fltk_event2mpl_event('button_double_press_event',self._source,True,True,True))
                return 1    
            elif self.button_press_event:
                self.button_press_event(fltk_event2mpl_event('button_press_event',self._source,True,True,True))  
                return 1  
        elif event == Fltk.FL_ENTER:
            return 1     
        elif event == Fltk.FL_LEAVE:
            return 1     
        elif event == Fltk.FL_MOVE:
            if self.motion_notify_event:
                self.motion_notify_event(fltk_event2mpl_event('motion_notify_event',self._source,True,False,True))  
                return 1     
        elif event == Fltk.FL_DRAG:
            if self._draw_overlay:
                self._dx=Fltk.Fl.event_x()-self._oldx
                self._dy=Fltk.Fl.event_y()-self._oldy
                Fltk.fl_overlay_rect(self._oldx,self._oldy,self._dx,self._dy)
            if self.motion_notify_event:
                self.motion_notify_event(fltk_event2mpl_event('motion_notify_event',self._source,True,True,True))  
                return 1    
        elif event == Fltk.FL_RELEASE:
            if self._draw_overlay:
                Fltk.fl_overlay_clear()
            if self.button_release_event:
                self.button_release_event(fltk_event2mpl_event('button_release_event',self._source,True,True,True))   
                return 1  
        else:
            return 0


class FigureCanvasFltkAgg(FigureCanvasAgg):
    def __init__(self, figure):
        FigureCanvasAgg.__init__(self,figure)
        t1,t2,w,h = self.figure.bbox.get_bounds()
        w, h = int(w), int(h)
        self.canvas=FltkCanvas(0, 0, w, h, "canvas",self)
        #self.draw()

    def resize(self,size):
        w, h = size
        # compute desired figure size in inches
        dpival = self.figure.dpi.get()
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_figsize_inches(winch,hinch)

    def draw(self):
        FigureCanvasAgg.draw(self)
        self.canvas.redraw()


    show = draw

    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):

        agg = self.switch_backends(FigureCanvasAgg)
        agg.print_figure(filename, dpi, facecolor, edgecolor, orientation)
        
    def mpl_connect(self,s,func):
        if(s=='button_press_event'):
            self.canvas.button_press_event=func
        elif(s=='button_double_press_event'):
            self.canvas.button_double_press_event=func    
        elif(s=='button_release_event'):
            self.canvas.button_release_event=func   
        elif(s=='motion_notify_event'):
            self.canvas.motion_notify_event=func    
            self.canvas.button_drag_event=func   
        elif(s=='key_press_event'):
            self.canvas.key_press_event=func  
        elif(s=='key_release_event'):
            self.canvas.key_release_event=func
        return s    
            
    def mpl_disconnect(self,s):
        if(s=='button_press_event'):
            self.canvas.button_press_event=None
        elif(s=='button_double_press_event'):
            self.canvas.button_double_press_event=None    
        elif(s=='button_release_event'):
            self.canvas.button_release_event=None    
        elif(s=='motion_notify_event'):
            self.canvas.motion_notify_event=None   
            self.canvas.button_drag_event=None     
        elif(s=='key_press_event'):
            self.canvas.key_press_event=None  
        elif(s=='key_release_event'):
            self.canvas.key_release_event=None
        return None
    
    def widget(self):
        return self.canvas

def destroy_figure(ptr,figman):
    figman.window.hide()
    Gcf.destroy(figman._num)
    
class FigureManagerFltkAgg(FigureManagerBase):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The fltk.Toolbar
    window      : The fltk.Window
    """
    def __init__(self, canvas, num, window):
        FigureManagerBase.__init__(self, canvas, num)   
        #Fltk container window
        t1,t2,w,h = canvas.figure.bbox.get_bounds()
        w, h = int(w), int(h)
        self.window = window
        self.window.size(w,h+30)
        self.window_title="Figure %d" % num
        self.window.label(self.window_title)
        self.window.size_range(350,200)
        self.window.callback(destroy_figure,self)
        self.canvas = canvas
        self._num =  num
        if matplotlib.rcParams['toolbar']=='classic':
            self.toolbar = NavigationToolbar( canvas, self )
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2FltkAgg( canvas, self )
        else:
            self.toolbar = None
        self.window.add_resizable(canvas.widget())
        if self.toolbar:
            self.window.add(self.toolbar.widget())
            self.toolbar.update()
        self.window.show()

        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'        
            if self.toolbar != None: self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

    def resize(self, event):
        width, height = event.width, event.height
        self.toolbar.configure(width=width) # , height=height)

    def show(self):
        _focus = windowing.FocusManager()
        self.canvas.draw()
        self.window.redraw()
        

class AxisMenu:
    def __init__(self, toolbar):
        self.toolbar=toolbar
        self._naxes = toolbar.naxes
        self._mbutton = Fltk.Fl_Menu_Button(0,0,50,10,"Axes")
        self._mbutton.add("Select All",0,select_all,self,0)
        self._mbutton.add("Invert All",0,invert_all,self,Fltk.FL_MENU_DIVIDER)
        self._axis_txt=[]
        self._axis_var=[]
        for i in range(self._naxes):
            self._axis_txt.append("Axis %d" % (i+1))
            self._mbutton.add(self._axis_txt[i],0,set_active,self,Fltk.FL_MENU_TOGGLE)
        for i in range(self._naxes):
            self._axis_var.append(self._mbutton.find_item(self._axis_txt[i]))
            self._axis_var[i].set()
    def adjust(self, naxes):
        if self._naxes < naxes:
            for i in range(self._naxes, naxes):
                self._axis_txt.append("Axis %d" % (i+1))
                self._mbutton.add(self._axis_txt[i],0,set_active,self,Fltk.FL_MENU_TOGGLE)
            for i in range(self._naxes, naxes):
                self._axis_var.append(self._mbutton.find_item(self._axis_txt[i]))
                self._axis_var[i].set()
        elif self._naxes > naxes:
            for i in range(self._naxes-1, naxes-1, -1):
                self._mbutton.remove(i+2)
            if(naxes):    
                self._axis_var=self._axis_var[:naxes-1]
                self._axis_txt=self._axis_txt[:naxes-1]
            else:
                self._axis_var=[]
                self._axis_txt=[]
        self._naxes = naxes
        set_active(0,self)
       
    def widget(self):
        return self._mbutton	     

    def get_indices(self):
        a = [i for i in range(len(self._axis_var)) if self._axis_var[i].value()]
        return a

def set_active(ptr,amenu):
    amenu.toolbar.set_active(amenu.get_indices())

def invert_all(ptr,amenu):
    for a in amenu._axis_var:
        if not a.value(): a.set()
    set_active(ptr,amenu)

def select_all(ptr,amenu):
    for a in amenu._axis_var:
        a.set()
    set_active(ptr,amenu)

class FLTKButton:
    def __init__(self, text, file, command,argument,type="classic"):
        file = os.path.join(rcParams['datapath'], file)
        self.im = Fltk.Fl_PNM_Image(file)
        size=26
        if type=="repeat":
            self.b = Fltk.Fl_Repeat_Button(0,0,size,10)
            self.b.box(Fltk.FL_THIN_UP_BOX)
        elif type=="classic":
            self.b = Fltk.Fl_Button(0,0,size,10)
            self.b.box(Fltk.FL_THIN_UP_BOX)
        elif type=="light":
            self.b = Fltk.Fl_Light_Button(0,0,size+20,10)
            self.b.box(Fltk.FL_THIN_UP_BOX)
        elif type=="pushed":
            self.b = Fltk.Fl_Button(0,0,size,10)
            self.b.box(Fltk.FL_UP_BOX)
            self.b.down_box(Fltk.FL_DOWN_BOX)
            self.b.type(Fltk.FL_TOGGLE_BUTTON)
        self.tooltiptext=text+" "
        self.b.tooltip(self.tooltiptext)
        self.b.callback(command,argument)
        self.b.image(self.im)
        self.b.deimage(self.im)
        self.type=type
    
    def widget(self):
        return self.b
        
class NavigationToolbar:
    """
    Public attriubutes

      canvas   - the FigureCanvas  (FigureCanvasFltkAgg = customised fltk.Widget)
      

    """
            
    def __init__(self, canvas, figman):
        #xmin, xmax = canvas.figure.bbox.intervalx().get_bounds()
        #height, width = 50, xmax-xmin
        self.canvas = canvas
        self.figman = figman

        Fltk.Fl_File_Icon.load_system_icons()
        self._fc = Fltk.Fl_File_Chooser( ".", "*", Fltk.Fl_File_Chooser.CREATE, "Save Figure" )
        self._fc.hide()
        t1,t2,w,h = canvas.figure.bbox.get_bounds()
        w, h = int(w), int(h)
        self._group = Fltk.Fl_Pack(0,h+2,1000,26)
        self._group.type(Fltk.FL_HORIZONTAL)
        self._axes=self.canvas.figure.axes
        self.naxes = len(self._axes)
        self.omenu = AxisMenu( toolbar=self)

        self.bLeft = FLTKButton(
            text="Left", file="stock_left.ppm",
            command=pan,argument=(self,1,'x'),type="repeat")

        self.bRight = FLTKButton(
            text="Right", file="stock_right.ppm",
            command=pan,argument=(self,-1,'x'),type="repeat")

        self.bZoomInX = FLTKButton(
            text="ZoomInX",file="stock_zoom-in.ppm",
            command=zoom,argument=(self,1,'x'),type="repeat")
        
        self.bZoomOutX = FLTKButton(
            text="ZoomOutX", file="stock_zoom-out.ppm",
            command=zoom, argument=(self,-1,'x'),type="repeat")

        self.bUp = FLTKButton(
            text="Up", file="stock_up.ppm",
            command=pan,argument=(self,1,'y'),type="repeat")
        
        self.bDown = FLTKButton(
            text="Down", file="stock_down.ppm",
            command=pan,argument=(self,-1,'y'),type="repeat")

        self.bZoomInY = FLTKButton(
            text="ZoomInY", file="stock_zoom-in.ppm",
            command=zoom,argument=(self,1,'y'),type="repeat")

        self.bZoomOutY = FLTKButton(
            text="ZoomOutY",file="stock_zoom-out.ppm",
            command=zoom, argument=(self,-1,'y'),type="repeat")
        
        self.bSave = FLTKButton(
            text="Save", file="stock_save_as.ppm",
            command=save_figure, argument=self)

	self._group.end()

    def widget(self):
        return self._group    

    def close(self):
        Gcf.destroy(self.figman._num)
        
    def set_active(self, ind):
        self._ind = ind
        self._active = [ self._axes[i] for i in self._ind ]

    def update(self):
        self._axes = self.canvas.figure.axes
        naxes = len(self._axes)
        self.omenu.adjust(naxes)

def pan(ptr, arg):
    base,direction,axe=arg
    for a in base._active:
        if(axe=='x'):	
            a.panx(direction)
        else:
            a.pany(direction)
    base.figman.show()
 
def zoom(ptr, arg):
    base,direction,axe=arg
    for a in base._active:
        if(axe=='x'):	
             a.zoomx(direction)
        else:
            a.zoomy(direction)
    base.figman.show()



def save_figure(ptr,base):
    file_chooser=base._fc
    file_chooser.show()
    while file_chooser.visible() : 
        Fltk.Fl.wait()  
    fname=None
    if(file_chooser.count() and file_chooser.value(0) != None):
        fname=""
        (status,fname)=Fltk.fl_filename_absolute(fname, 1024, file_chooser.value(0))

    if fname is None: # Cancel
        return
    #start from last directory
    lastDir = os.path.dirname(fname) 
    file_chooser.directory(lastDir)

    try:
        base.canvas.print_figure(fname)
    except IOError, msg:                
        err = '\n'.join(map(str, msg))
        msg = 'Failed to save %s: Error msg was\n\n%s' % (
            fname, err)
        error_msg_fltk(msg)
        
class NavigationToolbar2FltkAgg(NavigationToolbar2):
    """
    Public attriubutes

      canvas   - the FigureCanvas  
      figman   - the Figure manager

    """
        
    def __init__(self, canvas, figman):
        self.canvas = canvas
        self.figman = figman
        NavigationToolbar2.__init__(self, canvas)
        self.pan_selected=False
        self.zoom_selected=False
    
    def set_cursor(self, cursor):
        Fltk.fl_cursor(cursord[cursor],Fltk.FL_BLACK,Fltk.FL_WHITE)  
    
    def dynamic_update(self):
        self.canvas.draw()
        
    def pan(self,*args):
        self.pan_selected=not  self.pan_selected
        self.zoom_selected = False
        self.canvas.canvas._draw_overlay= False
        if self.pan_selected:
            self.bPan.widget().value(1)
        else:    
            self.bPan.widget().value(0)
        if self.zoom_selected:
            self.bZoom.widget().value(1)
        else:    
            self.bZoom.widget().value(0)
        NavigationToolbar2.pan(self,args)        
    
    def zoom(self,*args):
        self.zoom_selected=not  self.zoom_selected
        self.canvas.canvas._draw_overlay=self.zoom_selected
        self.pan_selected = False
        if self.pan_selected:
            self.bPan.widget().value(1)
        else:    
            self.bPan.widget().value(0)
        if self.zoom_selected:
            self.bZoom.widget().value(1)
        else:    
            self.bZoom.widget().value(0)
        NavigationToolbar2.zoom(self,args)        
        
    def _init_toolbar(self):
        Fltk.Fl_File_Icon.load_system_icons()
        self._fc = Fltk.Fl_File_Chooser( ".", "*", Fltk.Fl_File_Chooser.CREATE, "Save Figure" )
        self._fc.hide()
        t1,t2,w,h = self.canvas.figure.bbox.get_bounds()
        w, h = int(w), int(h)
        self._group = Fltk.Fl_Pack(0,h+2,1000,26)
        self._group.type(Fltk.FL_HORIZONTAL)
        self._axes=self.canvas.figure.axes
        self.naxes = len(self._axes)
        self.omenu = AxisMenu( toolbar=self)

        self.bHome = FLTKButton(
            text="Home", file="home.ppm",
            command=self.home,argument=self)

        self.bBack = FLTKButton(
            text="Back", file="back.ppm",
            command=self.back,argument=self)

        self.bForward = FLTKButton(
            text="Forward", file="forward.ppm",
            command=self.forward,argument=self)

        self.bPan = FLTKButton(
            text="Pan/Zoom",file="move.ppm",
            command=self.pan,argument=self,type="pushed")
            
        self.bZoom = FLTKButton(
            text="Zoom to rectangle",file="zoom_to_rect.ppm",
            command=self.zoom,argument=self,type="pushed")
            
        self.bSave = FLTKButton(
            text="Save", file="filesave.ppm",
            command=save_figure, argument=self)

        self._group.end()
        self.message = Fltk.Fl_Output(0,0,w,8)
        self._group.add_resizable(self.message)
        self.update()

    def widget(self):
        return self._group    

    def close(self):
        Gcf.destroy(self.figman._num)
        
    def set_active(self, ind):
        self._ind = ind
        self._active = [ self._axes[i] for i in self._ind ]

    def update(self):
        self._axes = self.canvas.figure.axes
        naxes = len(self._axes)
        self.omenu.adjust(naxes)
        NavigationToolbar2.update(self)
        
    def set_message(self, s):
        self.message.value(s)



FigureManager = FigureManagerFltkAgg
error_msg = error_msg_fltk
if matplotlib.is_interactive():
    show()

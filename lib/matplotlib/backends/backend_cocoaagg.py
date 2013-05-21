from __future__ import division, print_function
"""
 backend_cocoaagg.py

 A native Cocoa backend via PyObjC in OSX.

 Author: Charles Moad (cmoad@users.sourceforge.net)

 Notes:
  - Requires PyObjC (currently testing v1.3.7)
  - The Tk backend works nicely on OSX.  This code
    primarily serves as an example of embedding a
    matplotlib rendering context into a cocoa app
    using a NSImageView.
"""

import os, sys

try:
    import objc
except ImportError:
    raise ImportError('The CococaAgg backend required PyObjC to be installed!')

from Foundation import *
from AppKit import *
from PyObjCTools import NibClassBuilder, AppHelper

from matplotlib import cbook
cbook.warn_deprecated(
    '1.3',
    message="The CocoaAgg backend is not a fully-functioning backend. "
            "It may be removed in matplotlib 1.4.")

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureManagerBase, FigureCanvasBase
from matplotlib.backend_bases import ShowBase

from backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf

mplBundle = NSBundle.bundleWithPath_(os.path.dirname(__file__))


def new_figure_manager(num, *args, **kwargs):
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass( *args, **kwargs )
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasCocoaAgg(figure)
    return FigureManagerCocoaAgg(canvas, num)


## Below is the original show() function:
#def show():
#    for manager in Gcf.get_all_fig_managers():
#        manager.show()
#
## It appears that this backend is unusual in having a separate
## run function invoked for each figure, instead of a single
## mainloop.  Presumably there is no blocking at all.
##
## Using the Show class below should cause no difference in
## behavior.

class Show(ShowBase):
    def mainloop(self):
        pass

show = Show()

def draw_if_interactive():
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.show()

class FigureCanvasCocoaAgg(FigureCanvasAgg):
    def draw(self):
        FigureCanvasAgg.draw(self)

    def blit(self, bbox):
        pass

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__



NibClassBuilder.extractClasses('Matplotlib.nib', mplBundle)

class MatplotlibController(NibClassBuilder.AutoBaseClass):
    # available outlets:
    #  NSWindow plotWindow
    #  PlotView plotView

    def awakeFromNib(self):
        # Get a reference to the active canvas
        NSApp().setDelegate_(self)
        self.app = NSApp()
        self.canvas = Gcf.get_active().canvas
        self.plotView.canvas = self.canvas
        self.canvas.plotView = self.plotView

        self.plotWindow.setAcceptsMouseMovedEvents_(True)
        self.plotWindow.makeKeyAndOrderFront_(self)
        self.plotWindow.setDelegate_(self)#.plotView)

        self.plotView.setImageFrameStyle_(NSImageFrameGroove)
        self.plotView.image_ = NSImage.alloc().initWithSize_((0,0))
        self.plotView.setImage_(self.plotView.image_)

        # Make imageview first responder for key events
        self.plotWindow.makeFirstResponder_(self.plotView)

        # Force the first update
        self.plotView.windowDidResize_(self)

    def windowDidResize_(self, sender):
        self.plotView.windowDidResize_(sender)

    def windowShouldClose_(self, sender):
        #NSApplication.sharedApplication().stop_(self)
        self.app.stop_(self)
        return objc.YES

    def saveFigure_(self, sender):
        p = NSSavePanel.savePanel()
        if(p.runModal() == NSFileHandlingPanelOKButton):
            self.canvas.print_figure(p.filename())

    def printFigure_(self, sender):
        op = NSPrintOperation.printOperationWithView_(self.plotView)
        op.runOperation()

class PlotWindow(NibClassBuilder.AutoBaseClass):
    pass

class PlotView(NibClassBuilder.AutoBaseClass):
    def updatePlot(self):
        w,h = self.canvas.get_width_height()

        # Remove all previous images
        for i in xrange(self.image_.representations().count()):
            self.image_.removeRepresentation_(self.image_.representations().objectAtIndex_(i))

        self.image_.setSize_((w,h))

        brep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            (self.canvas.buffer_rgba(),'','','',''), # Image data
            w, # width
            h, # height
            8, # bits per pixel
            4, # components per pixel
            True, # has alpha?
            False, # is planar?
            NSCalibratedRGBColorSpace, # color space
            w*4, # row bytes
            32) # bits per pixel

        self.image_.addRepresentation_(brep)
        self.setNeedsDisplay_(True)

    def windowDidResize_(self, sender):
        w,h = self.bounds().size
        dpi = self.canvas.figure.dpi
        self.canvas.figure.set_size_inches(w / dpi, h / dpi)
        self.canvas.draw()
        self.updatePlot()

    def mouseDown_(self, event):
        dblclick = (event.clickCount() == 2)
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        type = event.type()
        if (type == NSLeftMouseDown):
            button = 1
        else:
            print('Unknown mouse event type:', type, file=sys.stderr)
            button = -1
        self.canvas.button_press_event(loc.x, loc.y, button, dblclick=dblclick)
        self.updatePlot()

    def mouseDragged_(self, event):
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        self.canvas.motion_notify_event(loc.x, loc.y)
        self.updatePlot()

    def mouseUp_(self, event):
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        type = event.type()
        if (type == NSLeftMouseUp):
            button = 1
        else:
            print('Unknown mouse event type:', type, file=sys.stderr)
            button = -1
        self.canvas.button_release_event(loc.x, loc.y, button)
        self.updatePlot()

    def keyDown_(self, event):
        self.canvas.key_press_event(event.characters())
        self.updatePlot()

    def keyUp_(self, event):
        self.canvas.key_release_event(event.characters())
        self.updatePlot()

class MPLBootstrap(NSObject):
    # Loads the nib containing the PlotWindow and PlotView
    def startWithBundle_(self, bundle):
        #NSApplicationLoad()
        if not bundle.loadNibFile_externalNameTable_withZone_('Matplotlib.nib', {}, None):
            print('Unable to load Matplotlib Cocoa UI!', file=sys.stderr)
            sys.exit()

class FigureManagerCocoaAgg(FigureManagerBase):
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)

        try:
            WMEnable('Matplotlib')
        except:
            # MULTIPLE FIGURES ARE BUGGY!
            pass # If there are multiple figures we only need to enable once
        #self.bootstrap = MPLBootstrap.alloc().init().performSelectorOnMainThread_withObject_waitUntilDone_(
        #        'startWithBundle:',
        #        mplBundle,
        #        False)

    def show(self):
        # Load a new PlotWindow
        self.bootstrap = MPLBootstrap.alloc().init().performSelectorOnMainThread_withObject_waitUntilDone_(
                'startWithBundle:',
                mplBundle,
                False)
        NSApplication.sharedApplication().run()


FigureManager = FigureManagerCocoaAgg

#### Everything below taken from PyObjC examples
#### This is a hack to allow python scripts to access
#### the window manager without running pythonw.
def S(*args):
    return ''.join(args)

OSErr = objc._C_SHT
OUTPSN = 'o^{ProcessSerialNumber=LL}'
INPSN = 'n^{ProcessSerialNumber=LL}'
FUNCTIONS=[
    # These two are public API
    ( u'GetCurrentProcess', S(OSErr, OUTPSN) ),
    ( u'SetFrontProcess', S(OSErr, INPSN) ),
    # This is undocumented SPI
    ( u'CPSSetProcessName', S(OSErr, INPSN, objc._C_CHARPTR) ),
    ( u'CPSEnableForegroundOperation', S(OSErr, INPSN) ),
]
def WMEnable(name='Python'):
    if isinstance(name, unicode):
        name = name.encode('utf8')
    mainBundle = NSBundle.mainBundle()
    bPath = os.path.split(os.path.split(os.path.split(sys.executable)[0])[0])[0]
    if mainBundle.bundlePath() == bPath:
        return True
    bndl = NSBundle.bundleWithPath_(objc.pathForFramework('/System/Library/Frameworks/ApplicationServices.framework'))
    if bndl is None:
        print('ApplicationServices missing', file=sys.stderr)
        return False
    d = {}
    objc.loadBundleFunctions(bndl, d, FUNCTIONS)
    for (fn, sig) in FUNCTIONS:
        if fn not in d:
            print('Missing', fn, file=sys.stderr)
            return False
    err, psn = d['GetCurrentProcess']()
    if err:
        print('GetCurrentProcess', (err, psn), file=sys.stderr)
        return False
    err = d['CPSSetProcessName'](psn, name)
    if err:
        print('CPSSetProcessName', (err, psn), file=sys.stderr)
        return False
    err = d['CPSEnableForegroundOperation'](psn)
    if err:
        #print >>sys.stderr, 'CPSEnableForegroundOperation', (err, psn)
        return False
    err = d['SetFrontProcess'](psn)
    if err:
        print('SetFrontProcess', (err, psn), file=sys.stderr)
        return False
    return True

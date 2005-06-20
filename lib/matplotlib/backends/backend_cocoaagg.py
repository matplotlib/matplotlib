from __future__ import division
"""
 backend_cocoaagg.py

 A native Cocoa backend via PyObjC in OSX.

 Author: Charles Moad (cmoad@indiana.edu)

 Notes:
  - THIS IS STILL IN DEVELOPMENT!
  - Requires PyObjC (currently testing v1.3.6)
"""

import os, sys

try:
    import objc
except:
    print >>sys.stderr, 'The CococaAgg backend required PyObjC to be installed!'
    sys.exit()

from Foundation import *
from AppKit import *
from PyObjCTools import NibClassBuilder, AppHelper

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureManagerBase
from backend_agg import FigureCanvasAgg
import pylab

DEBUG = True

mplBundle = NSBundle.bundleWithPath_(matplotlib.get_data_path())

def new_figure_manager(num, *args, **kwargs):
    if DEBUG: print >>sys.stderr, 'new_figure_manager'
    thisFig = Figure( *args, **kwargs )
    canvas = FigureCanvasAgg(thisFig)
    return FigureManagerCocoaAgg(canvas, num)
    
def show():
    if DEBUG: print >>sys.stderr, 'show'
    # Let the cocoa run loop take over
    NSApplication.sharedApplication().run()
	
def draw_if_interactive():
    if matplotlib.is_interactive():
        print >>sys.stderr, 'Interactive not implemented yet'
        
NibClassBuilder.extractClasses('Matplotlib.nib', mplBundle)

class MatplotlibController(NibClassBuilder.AutoBaseClass):
    # available outlets:
    #  NSWindow plotWindow
    #  PlotView plotView
    
    def awakeFromNib(self):
	if DEBUG: print 'MPLController awakeFromNib'
        # Get a reference to the active canvas
        self.canvas = pylab.get_current_fig_manager().canvas
	self.plotWindow.plotView = self.plotView
	self.plotView.canvas = self.canvas
	
	# Make imageview first responder for key events
	self.plotWindow.makeFirstResponder_(self.plotView)

	# Issue a resize to update plot
	self.plotWindow.windowDidResize_(self)

    def saveFigure_(self, sender):
        pass

    def quit_(self, sender):
        pass

class PlotWindow(NibClassBuilder.AutoBaseClass):
    def awakeFromNib(self):
	if DEBUG: print 'PlotWindow awakeFromNib'
	self.setAcceptsMouseMovedEvents_(True)
	self.useOptimizedDrawing_(True)
	self.makeKeyAndOrderFront_(self)
	self.setDelegate_(self)

    def windowDidResize_(self, sender):
        w,h = self.plotView.bounds().size
        dpi = self.plotView.canvas.figure.dpi.get()
        self.plotView.canvas.figure.set_figsize_inches(w / dpi, h / dpi)
        self.plotView.updatePlot()

class PlotView(NibClassBuilder.AutoBaseClass):
    def awakeFromNib(self):
	if DEBUG: print 'PlotView awakeFromNib'
	self.setImageFrameStyle_(NSImageFrameGroove)

    def updatePlot(self):
        self.canvas.draw() # tell the agg to render

        w,h = self.canvas.get_width_height()
        
        image = NSImage.alloc().initWithSize_((w,h))
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

        image.addRepresentation_(brep)
        self.setImage_(image)
        self.setNeedsDisplay_(True)

    def mouseDown_(self, event):
	loc = self.convertPoint_fromView_(event.locationInWindow(), None)
	type = event.type()
	if DEBUG: print >>sys.stderr, 'mouseDown_:', loc, type
	if (type == NSLeftMouseDown):
	    button = 1
	else:
	    print >>sys.stderr, 'Unknown mouse event type:', type
	    button = -1
	self.canvas.button_press_event(loc.x, loc.y, button)
	self.updatePlot()

    def mouseDragged_(self, event):
	loc = self.convertPoint_fromView_(event.locationInWindow(), None)
	if DEBUG: print >>sys.stderr, 'mouseDragged_:', loc
	self.canvas.motion_notify_event(loc.x, loc.y)
	self.updatePlot()

    def mouseUp_(self, event):
	loc = self.convertPoint_fromView_(event.locationInWindow(), None)
	type = event.type()
	if DEBUG: print >>sys.stderr, 'mouseUp_:', loc, type
	if (type == NSLeftMouseUp):
	    button = 1
	else:
	    print >>sys.stderr, 'Unknown mouse event type:', type
	    button = -1
	self.canvas.button_release_event(loc.x, loc.y, button)
	self.updatePlot()

    def keyDown_(self, event):
	if DEBUG: print >>sys.stderr, 'keyDown_', event.keyCode()
	self.canvas.key_press_event(event.keyCode())

    def keyUp_(self, event):
	if DEBUG: print >>sys.stderr, 'keyUp_', event.keyCode()
	self.canvas.key_release_event(event.keyCode())

class MPLBootstrap(NSObject):
    # Loads the nib containing the PlotWindow and PlotView
    def startWithBundle_(self, bundle):
	NSApplicationLoad()
	if not bundle.loadNibFile_externalNameTable_withZone_(
	    'Matplotlib.nib',
	    {},
	    None):
	    print >>sys.stderr, 'Unable to load Matplotlib Cocoa UI!'
	    sys.exit()

class FigureManagerCocoaAgg(FigureManagerBase):
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)

	# If there are multiple figures we only need to enable once
        try:
	    WMEnable('Matplotlib')
	except:
	    pass

	# Load a new PlotWindow
	MPLBootstrap.alloc().init().performSelectorOnMainThread_withObject_waitUntilDone_(
	    'startWithBundle:',
	    mplBundle,
	    False)
            
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
        print >>sys.stderr, 'ApplicationServices missing'
        return False
    d = {}
    objc.loadBundleFunctions(bndl, d, FUNCTIONS)
    for (fn, sig) in FUNCTIONS:
        if fn not in d:
            print >>sys.stderr, 'Missing', fn
            return False
    err, psn = d['GetCurrentProcess']()
    if err:
        print >>sys.stderr, 'GetCurrentProcess', (err, psn)
        return False
    err = d['CPSSetProcessName'](psn, name)
    if err:
        print >>sys.stderr, 'CPSSetProcessName', (err, psn)
        return False
    err = d['CPSEnableForegroundOperation'](psn)
    if err:
        #print >>sys.stderr, 'CPSEnableForegroundOperation', (err, psn)
        return False
    err = d['SetFrontProcess'](psn)
    if err:
        print >>sys.stderr, 'SetFrontProcess', (err, psn)
        return False
    return True


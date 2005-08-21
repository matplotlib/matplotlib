from __future__ import division
"""
 backend_cocoaagg.py

 A native Cocoa backend via PyObjC in OSX.

 Author: Charles Moad (cmoad@users.sourceforge.net)

 Notes:
  - THIS IS STILL IN DEVELOPMENT!
  - Requires PyObjC (currently testing v1.3.7)
"""

import os, sys

try:
    import objc
except:
    print >>sys.stderr, 'The CococaAgg backend required PyObjC to be installed!'
    print >>sys.stderr, '  (currently testing v1.3.7)'
    sys.exit()

from Foundation import *
from AppKit import *
from PyObjCTools import NibClassBuilder, AppHelper

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureManagerBase
from backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf
import pylab

mplBundle = NSBundle.bundleWithPath_(matplotlib.get_data_path())

def new_figure_manager(num, *args, **kwargs):
    thisFig = Figure( *args, **kwargs )
    canvas = FigureCanvasCocoaAgg(thisFig)
    return FigureManagerCocoaAgg(canvas, num)

def show():
    for manager in Gcf.get_all_fig_managers():
        manager.show()
    # Let the cocoa run loop take over
    NSApplication.sharedApplication().run()

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

NibClassBuilder.extractClasses('Matplotlib.nib', mplBundle)

class MatplotlibController(NibClassBuilder.AutoBaseClass):
    # available outlets:
    #  NSWindow plotWindow
    #  PlotView plotView
    
    def awakeFromNib(self):
        # Get a reference to the active canvas
        self.canvas = pylab.get_current_fig_manager().canvas
	self.plotView.canvas = self.canvas
	self.canvas.plotView = self.plotView
	
	self.plotWindow.setAcceptsMouseMovedEvents_(True)
	self.plotWindow.makeKeyAndOrderFront_(self)
	self.plotWindow.setDelegate_(self.plotView)

	self.plotView.setImageFrameStyle_(NSImageFrameGroove)
        self.plotView.image = NSImage.alloc().initWithSize_((0,0))
	self.plotView.setImage_(self.plotView.image)

	# Make imageview first responder for key events
	self.plotWindow.makeFirstResponder_(self.plotView)

	# Force the first update
	self.plotView.windowDidResize_(self)

    def saveFigure_(self, sender):
	print >>sys.stderr, 'Not Implented Yet'

class PlotWindow(NibClassBuilder.AutoBaseClass):
    pass

class PlotView(NibClassBuilder.AutoBaseClass):
    def updatePlot(self):
        w,h = self.canvas.get_width_height()
	
	# Remove all previous images
	for i in xrange(self.image.representations().count()):
	    self.image.removeRepresentation_(self.image.representations().objectAtIndex_(i))
	
	self.image.setSize_((w,h))

	brep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
	    (self.canvas.buffer_rgba(0,0),'','','',''), # Image data
	    w, # width
	    h, # height
	    8, # bits per pixel
	    4, # components per pixel
	    True, # has alpha?
	    False, # is planar?
	    NSCalibratedRGBColorSpace, # color space
	    w*4, # row bytes
	    32) # bits per pixel

        self.image.addRepresentation_(brep)
        self.setNeedsDisplay_(True)

    def windowDidResize_(self, sender):
        w,h = self.bounds().size
        dpi = self.canvas.figure.dpi.get()
        self.canvas.figure.set_figsize_inches(w / dpi, h / dpi)
	self.canvas.draw()
        self.updatePlot()

    def mouseDown_(self, event):
	loc = self.convertPoint_fromView_(event.locationInWindow(), None)
	type = event.type()
	if (type == NSLeftMouseDown):
	    button = 1
	else:
	    print >>sys.stderr, 'Unknown mouse event type:', type
	    button = -1
	self.canvas.button_press_event(loc.x, loc.y, button)
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
	    print >>sys.stderr, 'Unknown mouse event type:', type
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

        try:
	    WMEnable('Matplotlib')
	except:
	    # MULTIPLE FIGURES ARE BUGGY!
	    pass # If there are multiple figures we only need to enable once

    def show(self):
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


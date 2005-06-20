from __future__ import division
"""
 backend_cocoaagg.py

 A native Cocoa backend via PyObjC in OSX.

 Author: Charles Moad (cmoad@indiana.edu)

 Notes:
  - THIS IS STILL IN DEVELOPMENT!
  - Requires PyObjC (currently testing v1.3.6)
  - Only works with 10.3 at this time (10.4 is high priority)
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

mplBundle = NSBundle.bundleWithPath_(matplotlib.get_data_path())

def new_figure_manager(num, *args, **kwargs):
    thisFig = Figure( *args, **kwargs )
    canvas = FigureCanvasAgg(thisFig)
    return FigureManagerCocoaAgg(canvas, num)
    
def show():
    # Let the cocoa run loop take over
    NSApplication.sharedApplication().run()
	
def draw_if_interactive():
    if matplotlib.is_interactive():
        print >>sys.stderr, 'Not implemented yet'
        
NibClassBuilder.extractClasses('Matplotlib.nib', mplBundle)

class MatplotlibController(NibClassBuilder.AutoBaseClass):
    # available outlets:
    #  NSWindow plotWindow
    #  NSImageView plotView
    
    def awakeFromNib(self):
        self.plotView.setImageFrameStyle_(NSImageFrameGroove)

	self.plotWindow.setAcceptsMouseMovedEvents_(True)
	self.plotWindow.useOptimizedDrawing_(True)
	self.plotWindow.makeKeyAndOrderFront_(self)

        # Get a reference to the active canvas
        self.canvas = pylab.get_current_fig_manager().canvas

        # Issue a update
        self.windowDidResize_(self)

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
        self.plotView.setImage_(image)
        self.plotView.setNeedsDisplay_(True)
	
    def saveFigure_(self, sender):
        pass

    def quit_(self, sender):
        pass

    def windowDidResize_(self, sender):
        w,h = self.plotView.frame().size
        dpi = self.canvas.figure.dpi.get()
        self.canvas.figure.set_figsize_inches(w / dpi, h / dpi)
        self.updatePlot()

class MPLBootstrap(NSObject):
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
        if not WMEnable('Matplotlib'):
            print >>sys.stderr, 'Unable to hook to native window manager!'
            sys.exit()

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
        print >>sys.stderr, 'CPSEnableForegroundOperation', (err, psn)
        return False
    err = d['SetFrontProcess'](psn)
    if err:
        print >>sys.stderr, 'SetFrontProcess', (err, psn)
        return False
    return True


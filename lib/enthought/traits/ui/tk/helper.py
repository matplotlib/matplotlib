#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: David C. Morrill
# Date: 12/02/2004
# Description: Helper functions used to define Tkinter based trait editors and
#              trait editor factories.
#
#  Symbols defined: bitmap_cache
#                   TkDelegate
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import wx
import sys

from os.path   import join, dirname, abspath 
from constants import standard_bitmap_width, screen_dx, screen_dy

#-------------------------------------------------------------------------------
#  Data:
#-------------------------------------------------------------------------------

# Bitmap cache dictionary (indexed by filename):
_bitmap_cache = {}

### NOTE: This needs major improvements:
app_path    = None
traits_path = None
  
#-------------------------------------------------------------------------------
#  'TkDelegate' class:
#-------------------------------------------------------------------------------

class TkDelegate ( object ):
    
   #----------------------------------------------------------------------------
   #  Initialize the object:
   #----------------------------------------------------------------------------
   
   def __init__ ( self, delegate = None, **kw ):
       self.delegate = delegate
       for name, value in kw.items():
           setattr( self, name, value )
   
   #----------------------------------------------------------------------------
   #  Return the handle method for the delegate:
   #----------------------------------------------------------------------------
   
   def __call__ ( self ):
       return self.on_event
    
   #----------------------------------------------------------------------------
   #  Handle an event:
   #----------------------------------------------------------------------------
   
   def on_event ( self, *args ):
       self.delegate( self, *args )
       
#-------------------------------------------------------------------------------
#  Convert an image file name to a cached bitmap:
#-------------------------------------------------------------------------------

def bitmap_cache ( name, standard_size, path = None ):
    global app_path, traits_path
    if path is None:
        if traits_path is None:
           import  enthought.traits.ui.wx
           traits_path = join( dirname( enthought.traits.ui.wx.__file__ ), 
                               'images' )
        path = traits_path
    elif path == '':
        if app_path is None:
            app_path = join( dirname( sys.argv[0] ), '..', 'images' )
        path = app_path
    filename = abspath( join( path, name.replace( ' ', '_' ).lower() + '.gif' ))
    bitmap   = _bitmap_cache.get( filename + ('*'[ not standard_size: ]) )
    if bitmap is not None:
        return bitmap
    std_bitmap = bitmap = wx.BitmapFromImage( wx.Image( filename ) )
    _bitmap_cache[ filename ] = bitmap
    dx = bitmap.GetWidth()
    if dx < standard_bitmap_width:
        dy = bitmap.GetHeight()
        std_bitmap = wx.EmptyBitmap( standard_bitmap_width, dy )
        dc1 = wx.MemoryDC()
        dc2 = wx.MemoryDC()
        dc1.SelectObject( std_bitmap )
        dc2.SelectObject( bitmap )
        dc1.SetPen( wx.TRANSPARENT_PEN )
        dc1.SetBrush( wx.WHITE_BRUSH )
        dc1.DrawRectangle( 0, 0, standard_bitmap_width, dy )
        dc1.Blit( (standard_bitmap_width - dx) / 2, 0, dx, dy, dc2, 0, 0 ) 
    _bitmap_cache[ filename + '*' ] = std_bitmap
    if standard_size:
        return std_bitmap
    return bitmap

#-------------------------------------------------------------------------------
#  Positions one window near another:
#-------------------------------------------------------------------------------

def position_near ( origin, target, offset_x = 0, offset_y = 0, 
                                    align_x  = 1, align_y  = 1 ):
    """ Positions one window near another.
    """
    # Calculate the target window position relative to the origin window:                                         
    x, y     = origin.ClientToScreenXY( 0, 0 )
    dx, dy   = target.GetSizeTuple()
    odx, ody = origin.GetSizeTuple()
    if align_x < 0:
        x = x + odx - dx
    if align_y < 0:
        y = y + ody - dy
    x += offset_x
    y += offset_y
    
    # Make sure the target window will be on the screen:
    if (x + dx) > screen_dx:
       x = screen_dx - dx
    if x < 0:
       x = 0
    if (y + dy) > screen_dy:
       y = screen_dy - dy
    if y < 0:
       y = 0
       
    # Position the target window:
    target.SetPosition( wx.Point( x, y ) )
    
#-------------------------------------------------------------------------------
#  Returns an appropriate width for a wxChoice widget based upon the list of
#  values it contains:
#-------------------------------------------------------------------------------
    
def choice_width ( values ):
    """ Returns an appropriate width for a wxChoice widget based upon the list 
        of values it contains:
    """
    return max( [ len( x ) for x in values ] ) * 6


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
# Description: Defines a Tkinter ImageControl widget that is used by various
#              trait editors to display trait values iconically.
#
#  Symbols defined: ImageControl
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
       
import tk

#-------------------------------------------------------------------------------
#  'ImageControl' class:
#-------------------------------------------------------------------------------
       
class ImageControl ( wx.Window ):

    # Pens used to draw the 'selection' marker:
    _selectedPenDark = wx.Pen( 
        wx.SystemSettings_GetColour( wx.SYS_COLOUR_3DSHADOW ), 1, 
        wx.SOLID )
    _selectedPenLight = wx.Pen( 
        wx.SystemSettings_GetColour( wx.SYS_COLOUR_3DHIGHLIGHT ), 1, 
        wx.SOLID )
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
     
    def __init__ ( self, parent, bitmap, selected = None, handler = None ):
        """ Initializes the object.
        """
        wx.Window.__init__( self, parent, -1, 
                            size = wx.Size( bitmap.GetWidth()  + 10, 
                                            bitmap.GetHeight() + 10 ) )
        self._bitmap      = bitmap
        self._selected    = selected
        self._handler     = handler
        self._mouse_over  = False
        self._button_down = False
 
        # Set up the 'paint' event handler:
        wx.EVT_PAINT( self, self._on_paint )
        
        # Set up mouse event handlers:
        wx.EVT_LEFT_DOWN(    self, self._on_left_down )
        wx.EVT_LEFT_UP(      self, self._on_left_up )
        wx.EVT_ENTER_WINDOW( self, self._on_enter )
        wx.EVT_LEAVE_WINDOW( self, self._on_leave )
        
    #---------------------------------------------------------------------------
    #  Gets/Sets the current selection state of the image: 
    #---------------------------------------------------------------------------
        
    def Selected ( self, selected = None ):
        """ Gets/Sets the current selection state of the image.
        """
        if selected is not None:
            selected = (selected != 0)
            if selected != self._selected:
                if selected:
                    for control in self.GetParent().GetChildren():
                        if (isinstance( control, ImageControl ) and 
                            control.Selected()):
                            control.Selected( False )
                            break
                self._selected = selected
                self.Refresh()
        return self._selected
        
    #---------------------------------------------------------------------------
    #  Gets/Sets the current bitmap image: 
    #---------------------------------------------------------------------------
        
    def Bitmap ( self, bitmap = None ):
        if bitmap is not None:
            if bitmap != self._bitmap:
                self._bitmap = bitmap
                self.Refresh()
        return self._bitmap
        
    #---------------------------------------------------------------------------
    #  Gets/Sets the current click handler:
    #---------------------------------------------------------------------------
        
    def Handler ( self, handler = None ):
        """ Gets/Sets the current click handler.
        """
        if handler is not None:
            if handler != self._handler:
                self._handler = handler
                self.Refresh()
        return self._handler
    
    #---------------------------------------------------------------------------
    #  Handles the mouse entering the control: 
    #---------------------------------------------------------------------------
        
    def _on_enter ( self, event = None ):
        """ Handles the mouse entering the control.
        """
        if self._selected is not None:
            self._mouse_over = True
            self.Refresh()
        
    #---------------------------------------------------------------------------
    #  Handles the mouse leaving the control: 
    #---------------------------------------------------------------------------
    
    def _on_leave ( self, event = None ):
        """ Handles the mouse leaving the control.
        """
        if self._mouse_over:
            self._mouse_over = False
            self.Refresh()
    
    #---------------------------------------------------------------------------
    #  Handles the user pressing the mouse button: 
    #---------------------------------------------------------------------------
           
    def _on_left_down ( self, event = None ):
        """ Handles the user pressing the mouse button.
        """
        if self._selected is not None:
            self.CaptureMouse()    
            self._button_down = True
            self.Refresh()
 
    #---------------------------------------------------------------------------
    #  Handles the user clicking the control: 
    #---------------------------------------------------------------------------
    
    def _on_left_up ( self, event = None ):
        """ Handles the user clicking the control.
        """
        need_refresh = self._button_down
        if need_refresh:
            self.ReleaseMouse()    
            self._button_down = False
 
        if self._selected is not None:
            wdx, wdy = self.GetClientSizeTuple()
            x        = event.GetX()
            y        = event.GetY()
            if (0 <= x < wdx) and (0 <= y < wdy):
                if self._selected != -1:
                    self.Selected( True )
                elif need_refresh:
                    self.Refresh()
                if self._handler is not None:
                    self._handler( self )
                return
          
        if need_refresh:
            self.Refresh()
           
    #---------------------------------------------------------------------------
    #  Handles the control being re-painted: 
    #---------------------------------------------------------------------------
    
    def _on_paint ( self, event = None ):
        """ Handles the control being re-painted.
        """
        wdc      = wx.PaintDC( self )
        wdx, wdy = self.GetClientSizeTuple()
        bitmap   = self._bitmap
        bdx      = bitmap.GetWidth()
        bdy      = bitmap.GetHeight()
        wdc.DrawBitmap( bitmap, (wdx - bdx) / 2, (wdy - bdy) / 2, True )
            
        pens = [ self._selectedPenLight, self._selectedPenDark ]
        bd   = self._button_down
        if self._mouse_over:
            wdc.SetBrush( wx.TRANSPARENT_BRUSH )
            wdc.SetPen( pens[ bd ] )
            wdc.DrawLine( 0, 0, wdx, 0 )     
            wdc.DrawLine( 0, 1, 0, wdy )
            wdc.SetPen( pens[ 1 - bd ] )
            wdc.DrawLine( wdx - 1, 1, wdx - 1, wdy )
            wdc.DrawLine( 1, wdy - 1, wdx - 1, wdy - 1 )
            
        if self._selected == True:
            wdc.SetBrush( wx.TRANSPARENT_BRUSH )
            wdc.SetPen( pens[ bd ] )
            wdc.DrawLine( 1, 1, wdx - 1, 1 )
            wdc.DrawLine( 1, 1, 1, wdy - 1 )
            wdc.DrawLine( 2, 2, wdx - 2, 2 )
            wdc.DrawLine( 2, 2, 2, wdy - 2 )
            wdc.SetPen( pens[ 1 - bd ] )
            wdc.DrawLine( wdx - 2, 2, wdx - 2, wdy - 1 )
            wdc.DrawLine( 2, wdy - 2, wdx - 2, wdy - 2 )
            wdc.DrawLine( wdx - 3, 3, wdx - 3, wdy - 2 )
            wdc.DrawLine( 3, wdy - 3, wdx - 3, wdy - 3 )       


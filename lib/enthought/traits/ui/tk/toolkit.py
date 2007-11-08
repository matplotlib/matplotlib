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
# Description: Define the concrete implementations of the traits Toolkit
#              interface for the Tkinter user interface toolkit.
#
#  Symbols defined: GUIToolkit
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

# Make sure that Tkinter is installed:
import tk

from enthought.traits.ui.toolkit import Toolkit
from helper                      import position_near
from constants                   import screen_dx, screen_dy

#-------------------------------------------------------------------------------
#  'GUIToolkit' class:
#-------------------------------------------------------------------------------
    
class GUIToolkit ( Toolkit ):
    
    #---------------------------------------------------------------------------
    #  Create Tkinter specific user interfaces using information from the
    #  specified UI object:
    #---------------------------------------------------------------------------
    
    def ui_panel ( self, ui, parent ):
        """ Creates a Tkinter panel-based user interface using information 
            from the specified UI object.
        """
        import ui_panel
        ui_panel.ui_panel( ui, parent )
    
    def ui_subpanel ( self, ui, parent ):
        """ Creates a Tkinter subpanel-based user interface using information 
            from the specified UI object.
        """
        import ui_panel
        ui_panel.ui_subpanel( ui, parent )
    
    def ui_livemodal ( self, ui, parent ):
        """ Creates a Tkinter modal 'live update' dialog user interface using 
            information from the specified UI object.
        """
        import ui_live
        ui_live.ui_livemodal( ui, parent )
    
    def ui_live ( self, ui, parent ):
        """ Creates a Tkinter non-modal 'live update' window user interface 
            using information from the specified UI object.
        """
        import ui_live
        ui_live.ui_live( ui, parent )
    
    def ui_modal ( self, ui, parent ):
        """ Creates a Tkinter modal dialog user interface using information 
            from the specified UI object.
        """
        import ui_modal
        ui_modal.ui_modal( ui, parent )
    
    def ui_nonmodal ( self, ui, parent ):
        """ Creates a Tkinter non-modal dialog user interface using 
            information from the specified UI object.
        """
        import ui_nonmodal
        ui_nonmodal.ui_nonmodal( ui, parent )
    
    def ui_wizard ( self, ui, parent ):
        """ Creates a Tkinter wizard dialog user interface using information 
            from the specified UI object.
        """
        import ui_wizard
        ui_wizard.ui_wizard( ui, parent )
        
    def view_application ( self, context, view, kind = None ):        
        """ Creates a GUI toolkit specific modal dialog user interface that 
            runs as a complete application using information from the 
            specified View object.
        """
        import view_application
        return view_application.view_application( context, view, kind )
    
    #---------------------------------------------------------------------------
    #  Positions the associated dialog window on the display:
    #---------------------------------------------------------------------------
        
    def position ( self, ui ):
        """ Positions the associated dialog window on the display.
        """
        view   = ui.view
        window = ui.control

        # Set up the default position of the window:
        parent = window.GetParent()
        if parent is None:
           window.Centre( wx.BOTH )
        else:
           position_near( parent, window, offset_y = -30 )
        
        # Calculate the correct width and height for the window:
        cur_width  = window.winfo_width()
        cur_height = window.winfo_height()
        width      = view.width
        height     = view.height
        
        if width < 0.0:
            width = cur_width
        elif width <= 1.0:
            width = int( width * screen_dx )
        else:
            width = int( width )
            
        if height < 0.0:
            height = cur_height
        elif height <= 1.0:
            height = int( height * screen_dy )
        else:
            height = int( height )
            
        # Calculate the correct position for the window:
        x = view.x
        y = view.y
        
        if x < -99999.0:
            x = (screen_dx - width) / 2
        elif x <= -1.0:
            x = screen_dx - width + int( x ) + 1
        elif x < 0.0:
            x = screen_dx - width + int( x * screen_dx )
        elif x <= 1.0:
            x = int( x * screen_dx )
        else:
            x = int( x )
        
        if y < -99999.0:
            y = (screen_dy - height) / 2
        elif y <= -1.0:
            y = screen_dy - height + int( y ) + 1
        elif x < 0.0:
            y = screen_dy - height + int( y * screen_dy )
        elif y <= 1.0:
            y = int( y * screen_dy )
        else:
            y = int( y )
            
        # Position and size the window as requested:
        window.geometry( '%dx%d+%d+%d' % ( width, height, x, y ) )
        
    #---------------------------------------------------------------------------
    #  'EditorFactory' factory methods:
    #---------------------------------------------------------------------------
    
    # Boolean:
    def boolean_editor ( self, *args, **traits ):
        import boolean_editor as be
        return be.ToolkitEditorFactory( *args, **traits )
        
    # Button:
    def button_editor ( self, *args, **traits ):
        import button_editor as be
        return be.ToolkitEditorFactory( *args, **traits )
        
    # Check list:
    def check_list_editor ( self, *args, **traits ):
        import check_list_editor as cle
        return cle.ToolkitEditorFactory( *args, **traits )
        
    # Color:
    def color_editor ( self, *args, **traits ):
        import color_editor as ce
        return ce.ToolkitEditorFactory( *args, **traits )
        
    # RGB Color:
    def rgb_color_editor ( self, *args, **traits ):
        import rgb_color_editor as rgbce
        return rgbce.ToolkitEditorFactory( *args, **traits )
        
    # Compound:
    def compound_editor ( self, *args, **traits ):
        import compound_editor as ce
        return ce.ToolkitEditorFactory( *args, **traits )
        
    # Directory:
    def directory_editor ( self, *args, **traits ):
        import directory_editor as de
        return de.ToolkitEditorFactory( *args, **traits)
        
    # Enum(eration):
    def enum_editor ( self, *args, **traits ):
        import enum_editor as ee
        return ee.ToolkitEditorFactory( *args, **traits )
        
    # File:
    def file_editor ( self, *args, **traits ):
        import file_editor as fe
        return fe.ToolkitEditorFactory( *args, **traits )
        
    # Font:
    def font_editor ( self, *args, **traits ):
        import font_editor as fe
        return fe.ToolkitEditorFactory( *args, **traits )
        
    # Image enum(eration):
    def image_enum_editor ( self, *args, **traits ):
        import image_enum_editor as iee
        return iee.ToolkitEditorFactory( *args, **traits )
        
    # Instance:
    def instance_editor ( self, *args, **traits ):
        import instance_editor as ie
        return ie.ToolkitEditorFactory( *args, **traits )
        
    # List:
    def list_editor ( self, *args, **traits ):
        import list_editor as le
        return le.ToolkitEditorFactory( *args, **traits )
        
    # Range:
    def range_editor ( self, *args, **traits ):
        import range_editor as re
        return re.ToolkitEditorFactory( *args, **traits )
        
    # Text:
    def text_editor ( self, *args, **traits ):
        import text_editor as te
        return te.ToolkitEditorFactory( *args, **traits )
        
    # Tree:
    def tree_editor ( self, *args, **traits ):
        import tree_editor as te
        return te.ToolkitEditorFactory( *args, **traits )
        

#-------------------------------------------------------------------------------
#
#  Define the stub functions used for creating concrete implementations of the
#  standard EditorFactory subclasses supplied with the Traits package.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: toolkit
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits import HasTraits, HasPrivateTraits, TraitError
from ui_traits        import SequenceTypes

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# List of implemented UI toolkits:
TraitUIToolkits = [ 'wx', 'null' ]

#-------------------------------------------------------------------------------
#  Data:
#-------------------------------------------------------------------------------

# The current GUI toolkit object being used:
_toolkit = None

#-------------------------------------------------------------------------------
#  Low-level GUI toolkit selection function:
#-------------------------------------------------------------------------------

def toolkit ( *toolkits ):
    global _toolkit
    
    if len( toolkits ) == 0:
        if _toolkit is not None:
            return _toolkit
        toolkits = TraitUIToolkits
    for toolkit_name in toolkits:   
        try:
            package  = 'enthought.traits.ui.' + toolkit_name
            module   = __import__( package )
            _toolkit = getattr( module.traits.ui, toolkit_name ).toolkit
            return _toolkit
        except ImportError:
            pass
    else:
        raise TraitError, ("Could not find any UI toolkit called: %s" % 
                           ', '.join( toolkits ))
    
#-------------------------------------------------------------------------------
#  'Toolkit' class (abstract base class):
#-------------------------------------------------------------------------------
    
class Toolkit ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Create GUI toolkit specific user interfaces using information from the
    #  specified UI object:
    #---------------------------------------------------------------------------
    
    def ui_panel ( self, ui, parent ):
        """ Creates a GUI toolkit specific panel-based user interface using 
            information from the specified UI object.
        """
        raise NotImplementedError
    
    def ui_subpanel ( self, ui, parent ):
        """ Creates a GUI toolkit specific subpanel-based user interface using 
            information from the specified UI object.
        """
        raise NotImplementedError
    
    def ui_livemodal ( self, ui, parent ):
        """ Creates a GUI toolkit specific modal 'live update' dialog user 
            interface using information from the specified UI object.
        """
        raise NotImplementedError
    
    def ui_live ( self, ui, parent ):
        """ Creates a GUI toolkit specific non-modal 'live update' window user 
            interface using information from the specified UI object.
        """
        raise NotImplementedError
    
    def ui_modal ( self, ui, parent ):
        """ Creates a GUI toolkit specific modal dialog user interface using 
            information from the specified UI object.
        """
        raise NotImplementedError
    
    def ui_nonmodal ( self, ui, parent ):
        """ Creates a GUI toolkit specific non-modal dialog user interface using 
            information from the specified UI object.
        """
        raise NotImplementedError
    
    def ui_wizard ( self, ui, parent ):
        """ Creates a GUI toolkit specific wizard dialog user interface using 
            information from the specified UI object.
        """
        raise NotImplementedError
        
    def view_application ( self, context, view, kind = None, handler = None ):        
        """ Creates a GUI toolkit specific modal dialog user interface that 
            runs as a complete application using information from the 
            specified View object.
        """
        raise NotImplementedError
    
    #---------------------------------------------------------------------------
    #  Positions the associated dialog window on the display:
    #---------------------------------------------------------------------------
        
    def position ( self, ui ):
        """ Positions the associated dialog window on the display.
        """
        raise NotImplementedError
        
    #---------------------------------------------------------------------------
    #  Shows a 'Help' window for a specified UI and control:    
    #---------------------------------------------------------------------------
                
    def show_help ( self, ui, control ):
        """ Shows a 'Help' window for a specified UI and control.
        """
        raise NotImplementedError
        
    #---------------------------------------------------------------------------
    #  Saves user preference information associated with a UI window:  
    #---------------------------------------------------------------------------
                
    def save_window ( self, ui ):
        """ Saves user preference information associated with a UI window.
        """
        raise NotImplementedError
        
    #---------------------------------------------------------------------------
    #  GUI toolkit dependent trait definitions:
    #---------------------------------------------------------------------------
        
    def color_trait ( self, *args, **traits ):
        raise NotImplementedError
        
    def rgb_color_trait ( self, *args, **traits ):
        raise NotImplementedError
        
    def rgba_color_trait ( self, *args, **traits ):
        raise NotImplementedError
        
    def font_trait ( self, *args, **traits ):
        raise NotImplementedError
        
    def kiva_font_trait ( self, *args, **traits ):
        raise NotImplementedError
        
    #---------------------------------------------------------------------------
    #  'EditorFactory' factory methods:
    #---------------------------------------------------------------------------
    
    def boolean_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def button_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def check_list_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def code_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def color_editor ( self, *args, **traits ):
        raise NotImplementedError
       
    def compound_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def directory_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def enable_rgba_color_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def enum_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def file_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def font_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def kiva_font_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def image_enum_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def instance_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def list_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def plot_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def range_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def rgb_color_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def rgba_color_editor ( self, *args, **traits ):
        raise NotImplementedError
         
    def text_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def tree_editor ( self, *args, **traits ):
        raise NotImplementedError
        
    def tuple_editor ( self, *args, **traits ):
        raise NotImplementedError
        

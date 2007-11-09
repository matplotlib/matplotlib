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
# Date: 10/21/2004
# Description: Define the Tkinter implementation of the various boolean editors
#              and the boolean editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api import Dict, Str, Any, true
from editor           import Editor
from text_editor      import SimpleEditor         as TextEditor
from text_editor      import ToolkitEditorFactory as EditorFactory
from helper           import TkDelegate

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Map from user input text to other value:
mapping_trait = Dict( Str, Any, { 'True':  True,
                                  'true':  True,
                                  't':     True,
                                  'yes':   True,
                                  'y':     True,
                                  'False': False,
                                  'false': False,
                                  'f':     False,
                                  'no':    False,
                                  'n':     False,
                    } )

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
       
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # These definitions override definitions in the 'text_editor' version:
    mapping = mapping_trait  # Dictionary mapping user input to other values
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description ) 
    
    def text_editor ( self, ui, object, name, description, parent ):
        return TextEditor( parent,
                           factory     = self, 
                           ui          = ui, 
                           object      = object, 
                           name        = name, 
                           description = description ) 
    
    def readonly_editor ( self, ui, object, name, description, parent ):
        return ReadonlyEditor( parent,
                               factory     = self, 
                               ui          = ui, 
                               object      = object, 
                               name        = name, 
                               description = description ) 
                                      
#-------------------------------------------------------------------------------
#  'SimpleEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        var          = tk.IntVar()
        self.control = control = tk.Checkbutton( parent, 
                                                 text     = '',
                                                 variable = var,
                                                 anchor   = 'w' )
        control.configure( command = TkDelegate( self.update_object,
                                                 var = var )() )

    #---------------------------------------------------------------------------
    #  Handles the user clicking on the checkbox:
    #---------------------------------------------------------------------------
 
    def update_object ( self, delegate ):
        """ Handles the user clicking on the checkbox.
        """
        self.value = (delegate.var.get() != 0)
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        self.control.cget( 'variable' ).set( self.value )
                                      
#-------------------------------------------------------------------------------
#  'ReadonlyEditor' class:
#-------------------------------------------------------------------------------
                               
class ReadonlyEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = tk.Label( parent, text = '', anchor = 'e' )
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #
    #  (Should normally be overridden in a subclass)
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        self.control.configure( text = [ 'False', 'True' ][ self.value ] )


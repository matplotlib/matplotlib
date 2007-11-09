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
# Description: Define the Tkinter implementation of the various button editors
#              and the button editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api            import Any, Str
from editor_factory              import EditorFactory
from editor                      import Editor
from enthought.traits.trait_base import user_name_for

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    value = Any( 0 )  # Value to set when button is clicked 
    label = Str       # Optional label for button
    
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
        label = self.factory.label
        if label == '':
            label = user_name_for( self.name )
        self.control = tk.Button( parent, text    = label, 
                                          command = self.update_object )
       
    #---------------------------------------------------------------------------
    #  Handles the user clicking the button by setting the value on the object:
    #---------------------------------------------------------------------------
        
    def update_object ( self, event ):
        """ Handles the user clicking the button by setting the factory value 
            on the object.
        """
        self.value = self.factory.value
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        pass


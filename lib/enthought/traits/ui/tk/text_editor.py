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
# Description: Define the Tkinter implementation of the various text editors and
#              the text editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api import Dict, Str, Any, true, false, TraitError
from editor           import Editor
from editor_factory   import EditorFactory
from constants        import OKColor, ErrorColor
from helper           import TkDelegate

#-------------------------------------------------------------------------------
#  Define a simple identity mapping:
#-------------------------------------------------------------------------------

class _Identity ( object ):
    
    def __call__ ( self, value ):    
        return value

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Map from user input text to other value:
mapping_trait = Dict( Str, Any )

# Function used to evaluate textual user input:
evaluate_trait = Any( _Identity() )

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
       
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    mapping   = mapping_trait  # Dictionary mapping user input to other values
    auto_set  = true           # Is user input set on every keystroke?
    enter_set = false          # Is user input set on enter key?
    password  = false          # Is user input unreadable (i.e. password)?
    evaluate  = evaluate_trait # Function used to evaluate textual user input
    
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
        factory       = self.factory
        var           = tk.StringVar()
        update_object = TkDelegate( self.update_object, var = var )
        if factory.password:
            control = tk.Entry( parent, textvariable = var, show = '*' )
        else:
            control = tk.Entry( parent, textvariable = var )
        if factory.enter_set:
            control.bind( '<Return>', update_object )
        control.bind( '<FocusOut>', update_object )
        if factory.auto_set:
            control.bind( '<Any-KeyPress>', update_object )
        self.control = control

    #---------------------------------------------------------------------------
    #  Handles the user entering input data in the edit control:
    #---------------------------------------------------------------------------
  
    def update_object ( self, delegate ):
        """ Handles the user entering input data in the edit control.
        """
        try:
            self.value = self._get_value()
            self.control.configure( bg = OKColor )
            #self.control.update()
        except TraitError, excp:
            pass
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        var = self.control.cget('textvariable' )
        if self.factory.evaluate( var.get() ) != self.value:
            var.set( self.str_value )
            #self.control.update()

    #---------------------------------------------------------------------------
    #  Gets the actual value corresponding to what the user typed:
    #---------------------------------------------------------------------------
 
    def _get_value ( self ):
        """ Gets the actual value corresponding to what the user typed.
        """
        value = self.control.cget( 'textvariable' ).get()
        try:
            value = self.factory.evaluate( value )
        except:
            pass
        return self.factory.mapping.get( value, value )
        
    #---------------------------------------------------------------------------
    #  Handles an error that occurs while setting the object's trait value:
    #---------------------------------------------------------------------------
        
    def error ( self, excp ):
        """ Handles an error that occurs while setting the object's trait value.
        """
        self.control.configure( bg = ErrorColor )
        #self.control.update()
        

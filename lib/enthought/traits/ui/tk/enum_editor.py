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
# Description: Define the Tkinter implementation of the various enumeration
#              editors and the enumeration editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from string                        import ascii_lowercase
from editor                        import Editor
from editor_factory                import EditorFactory
from enthought.traits.api              import Any, Range, TraitError, CTrait, \
                                          TraitHandler
from enthought.traits.ui.ui_traits import SequenceTypes

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    values = Any            # Values to enumerate (can be a list, tuple, dict,
                            # or a CTrait or TraitHandler than is 'mapped')
    cols   = Range( 1, 20 ) # Number of columns to use when displayed as a grid
    
    #---------------------------------------------------------------------------
    #  Performs any initialization needed after all constructor traits have 
    #  been set:
    #---------------------------------------------------------------------------
     
    def init ( self ):
        """ Performs any initialization needed after all constructor traits 
            have been set.
        """
        values      = self.values
        type_values = type( values )
        
        if type_values is dict:
            data = [ ( str( n ), v ) for n, v in values.items() ]
            data.sort( lambda x, y: cmp( x[0], y[0] ) )
            col = data[0][0].find( ':' ) + 1
            if col > 0:
                data = [ ( n[ col: ], v ) for n, v in data ]
        else:
            if not type_values in SequenceTypes:
                handler = values
                if isinstance( handler, CTrait ):
                    handler = handler.handler
                if not isinstance( handler, TraitHandler ):
                    raise TraitError, "Invalid value for 'values' specified"
                if handler.is_mapped:
                    data = [ ( str( n ), n ) for n in handler.map.keys() ]
                    data.sort( lambda x, y: cmp( x[0], y[0] ) )
                else:
                    data = [ ( str( v ), v ) for v in handler.values ]
            else:
                data = [ ( str( v ), v ) for v in values ]
        
        self._names   = [ x[0] for x in data ]
        self._mapping = _mapping = {}
        for name, value in data:
            _mapping[ name ] = value
    
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
    
    def custom_editor ( self, ui, object, name, description, parent ):
        return CustomEditor( parent,
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
        self.control = wx.Choice( parent, -1, wx.Point( 0, 0 ), 
                                  wx.Size( 100, 20 ), self.factory._names )
        wx.EVT_CHOICE( parent, self.control.GetId(), self.update_object )
        self.update_editor()

    #---------------------------------------------------------------------------
    #  Handles the user selecting a new value from the combo box:
    #---------------------------------------------------------------------------
  
    def update_object ( self, event ):
        """ Handles the user selecting a new value from the combo box.
        """
        self.value = self.factory._mapping[ event.GetString() ]
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        try:
            self.control.SetStringSelection( self.str_value )
        except:
            pass
                                      
#-------------------------------------------------------------------------------
#  'CustomEditor' class:
#-------------------------------------------------------------------------------
                               
class CustomEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        # Create a panel to hold all of the radio buttons:
        self.control = panel = wx.Panel( parent, -1 )
                              
        # Get the current trait value:
        cur_name = self.str_value
                           
        # Create a sizer to manage the radio buttons:
        names   = self.factory._names
        mapping = self.factory._mapping
        n       = len( names )
        cols    = self.factory.cols
        rows    = (n + cols - 1) / cols
        incr    = [ n / cols ] * cols
        rem     = n % cols
        for i in range( cols ):
            incr[i] += (rem > i)
        incr[-1] = -(reduce( lambda x, y: x + y, incr[:-1], 0 ) - 1)
        if cols > 1:
            sizer = wx.GridSizer( 0, cols, 2, 4 )
        else:
            sizer = wx.BoxSizer( wx.VERTICAL )
        
        # Add the set of all possible choices:
        style = wx.RB_GROUP
        index = 0
        for i in range( rows ):
            for j in range( cols ):
                if n > 0:
                    name = label = names[ index ]
                    if label[:1] in ascii_lowercase:
                        label = label.capitalize()
                    control = wx.RadioButton( panel, -1, label, style = style )
                    control.value = mapping[ name ]
                    style         = 0
                    control.SetValue( name == cur_name )
                    wx.EVT_RADIOBUTTON( panel, control.GetId(), 
                                        self.update_object )
                    self.set_tooltip( control )
                    index += incr[j]
                    n     -= 1
                else:
                    control = wx.RadioButton( panel, -1, '' )
                    control.value = ''
                    control.Show( False )
                sizer.Add( control, 0, wx.NORTH, 5 )
     
        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )
   
    #---------------------------------------------------------------------------
    #  Handles the user clicking one of the 'custom' radio buttons:
    #---------------------------------------------------------------------------
    
    def update_object ( self, event ):
        """ Handles the user clicking one of the 'custom' radio buttons.
        """
        self.value = event.GetEventObject().value 
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        value = self.value
        for button in self.control.GetChildren():
            button.SetValue( button.value == value )
        

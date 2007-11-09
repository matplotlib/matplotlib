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
# Description: Define the Tkinter implementation of the various check list
#              editors and the check list editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api import Range, List, TraitError
from editor_factory   import EditorFactory
from editor_factory   import TextEditor as BaseTextEditor
from editor           import Editor

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    cols   = Range( 1, 20 ) # Number of columns to use when displayed as a grid
    values = List           # List of possible checklist values (either a list
                            # of strings, or a list of 2-element sequences:
                            # ( value, label ))

    #---------------------------------------------------------------------------
    #  Performs any initialization needed after all constructor traits have
    #  been set:
    #---------------------------------------------------------------------------

    def init ( self, handler = None ):
        """ Performs any initialization needed after all constructor traits
            have been set.
        """
        values = self.values
        if isinstance(values[0], basestring):
           values = [ ( x, x.capitalize() ) for x in values ]
        self._values = [ x[0] for x in values ]
        self._names  = [ x[1] for x in values ]

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

    def text_editor ( self, ui, object, name, description, parent ):
        return TextEditor( parent,
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
        self.control = wx.Choice( parent, -1,
                                  wx.Point( 0, 0 ), wx.Size( 100, 20 ),
                                  self.factory._names )
        wx.EVT_CHOICE( parent, self.control.GetId(), self.update_object )
        self.update_editor()

    #----------------------------------------------------------------------------
    #  Handles the user selecting a new value from the combo box:
    #----------------------------------------------------------------------------

    def update_object ( self, event ):
        """ Handles the user selecting a new value from the combo box.
        """
        value = self.factory._values[
                    self.factory._names.index( event.GetString() ) ]
        if type( self.value ) is not str:
           value = [ value ]
        self.value = value

    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------

    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the
            editor.
        """
        try:
            self.control.SetSelection( self.factory._values.index(
                                            parse_value( self.value )[0] ) )
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
        cur_value = parse_value( self.value )

        # Create a sizer to manage the radio buttons:
        labels = self.factory._names
        values = self.factory._values
        n      = len( labels )
        cols   = self.factory.cols
        rows   = (n + cols - 1) / cols
        incr   = [ n / cols ] * cols
        rem    = n % cols
        for i in range( cols ):
            incr[i] += (rem > i)
        incr[-1] = -(reduce( lambda x, y: x + y, incr[:-1], 0 ) - 1)
        if cols > 1:
           sizer = wx.GridSizer( 0, cols, 2, 4 )
        else:
           sizer = wx.BoxSizer( wx.VERTICAL )

        # Add the set of all possible choices:
        index = 0
        for i in range( rows ):
            for j in range( cols ):
                if n > 0:
                    label   = labels[ index ]
                    control = wx.CheckBox( panel, -1, label )
                    control.value = value = values[ index ]
                    control.SetValue( value in cur_value )
                    wx.EVT_CHECKBOX( panel, control.GetId(), self.update_object)
                    index += incr[j]
                    n     -= 1
                else:
                   control = wx.CheckBox( panel, -1, '' )
                   control.Show( False )
                sizer.Add( control, 0, wx.NORTH, 5 )

        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

    #---------------------------------------------------------------------------
    #  Handles the user clicking one of the 'custom' check boxes:
    #---------------------------------------------------------------------------

    def update_object ( self, event ):
        """ Handles the user clicking one of the 'custom' check boxes.
        """
        control   = event.GetEventObject()
        cur_value = parse_value( self.value )
        if control.GetValue():
            cur_value.append( control.value )
        else:
            cur_value.remove( control.value )
        if isinstance(self.value, basestring):
            cur_value = ','.join( cur_value )
        self.value = cur_value

    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------

    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the
            editor.
        """
        new_values = parse_value( self.value )
        for control in self.control.GetChildren():
            if control.IsShown():
               control.SetValue( control.value in new_values )

#-------------------------------------------------------------------------------
#  'TextEditor' class:
#-------------------------------------------------------------------------------

class TextEditor ( BaseTextEditor ):

    #---------------------------------------------------------------------------
    #  Handles the user changing the contents of the edit control:
    #---------------------------------------------------------------------------

    def update_object ( self, event ):
        """ Handles the user changing the contents of the edit control.
        """
        try:
            value = self.control.GetValue()
            value = eval( value )
        except:
            pass
        try:
            self.value = value
        except TraitError, excp:
            pass

#-------------------------------------------------------------------------------
#  Parse a value into a list:
#-------------------------------------------------------------------------------

def parse_value ( value ):
    if value is None:
       return []
    if type( value ) is not str:
       return value[:]
    return [ x.strip() for x in value.split( ',' ) ]


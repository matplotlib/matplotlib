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
# Description: Define the Tkinter implementation of the various range editors
#              and the range editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from math             import log10                           
from enthought.traits.api import CTrait, TraitError, Property, Range, Str, true, \
                             false
from editor_factory   import EditorFactory, TextEditor
from editor           import Editor
from constants        import OKColor, ErrorColor

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    cols       = Range( 1, 20 ) # Number of columns when displayed as an enum
    auto_set   = true           # Is user input set on every keystroke?
    enter_set  = false          # Is user input set on enter key?
    low_label  = Str            # Label for low end of range
    high_label = Str            # Label for high end of range
    is_float   = true           # Is the range float (or int)?
        
    #---------------------------------------------------------------------------
    #  Performs any initialization needed after all constructor traits have 
    #  been set:
    #---------------------------------------------------------------------------
     
    def init ( self, handler = None ):
        """ Performs any initialization needed after all constructor traits 
            have been set.
        """
        if handler is not None:
            if isinstance( handler, CTrait ):
                handler = handler.handler
            self.low  = handler.low
            self.high = handler.high
            
    #---------------------------------------------------------------------------
    #  Define the 'low' and 'high' traits:
    #---------------------------------------------------------------------------
            
    def _get_low ( self ):
        return self._low
        
    def _set_low ( self, low ):
        self._low     = low
        self.is_float = (type( low ) is float)
        if self.low_label == '':
            self.low_label = str( low  )
            
    def _get_high ( self ):
        return self._high
        
    def _set_high ( self, high ):
        self._high    = high
        self.is_float = (type( high ) is float)
        if self.high_label == '':
            self.high_label = str( high )
        
    low  = Property( _get_low,  _set_low  )
    high = Property( _get_high, _set_high )
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        if self.is_float or (abs( self.high - self.low ) <= 100):
            return SimpleSliderEditor( parent,
                                       factory     = self, 
                                       ui          = ui, 
                                       object      = object, 
                                       name        = name, 
                                       description = description )
        return SimpleSpinEditor( parent,
                                 factory     = self, 
                                 ui          = ui, 
                                 object      = object, 
                                 name        = name, 
                                 description = description ) 
    
    def custom_editor ( self, ui, object, name, description, parent ):
        if self.is_float or (abs( self.high - self.low ) > 15):
           return self.simple_editor( ui, object, name, description, parent )
           
        if self._enum is None:
            import enum_editor
            self._enum = enum_editor.ToolkitEditorFactory( 
                              values = range( self.low, self.high + 1 ), 
                              cols   = self.cols )
        return self._enum.custom_editor( ui, object, name, description, parent )
    
    def text_editor ( self, ui, object, name, description, parent ):
        return RangeTextEditor( parent,
                                factory     = self, 
                                ui          = ui, 
                                object      = object, 
                                name        = name, 
                                description = description ) 
                                      
#-------------------------------------------------------------------------------
#  'SimpleSliderEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleSliderEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        factory = self.factory
        low     = factory.low
        high    = factory.high
        self._format = '%d'
        if factory.is_float:
           self._format = '%%.%df' % max( 0, 4 - int( log10( high - low ) ) )
        self.control = panel = wx.Panel( parent, -1 )
        sizer  = wx.BoxSizer( wx.HORIZONTAL )
        fvalue = self.value
        try:
           fvalue_text = self._format % fvalue
           1 / (low <= fvalue <= high)
        except:
           fvalue_text = ''
           fvalue      = low 
        ivalue   = int( (float( fvalue - low ) / (high - low)) * 10000 )
        label_lo = wx.StaticText( panel, -1, '999.999',
                      style = wx.ALIGN_RIGHT | wx.ST_NO_AUTORESIZE )
        sizer.Add( label_lo, 0, wx.ALIGN_CENTER )
        panel.slider = slider = wx.Slider( panel, -1, ivalue, 0, 10000,
                                   size   = wx.Size( 100, 20 ),
                                   style  = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS )
        slider.SetTickFreq( 1000, 1 )
        slider.SetPageSize( 1000 )
        slider.SetLineSize( 100 )
        wx.EVT_SCROLL( slider, self.update_object_on_scroll )
        sizer.Add( slider, 1, wx.EXPAND )
        label_hi = wx.StaticText( panel, -1, '999.999' )
        sizer.Add( label_hi, 0, wx.ALIGN_CENTER )
        if factory.enter_set:
            panel.text = text = wx.TextCtrl( panel, -1, fvalue_text,
                                             size  = wx.Size( 60, 20 ),
                                             style = wx.TE_PROCESS_ENTER )
            wx.EVT_TEXT_ENTER( panel, text.GetId(), 
                               self.update_object_on_enter )
        else:                                               
            panel.text = text = wx.TextCtrl( panel, -1, fvalue_text,
                                             size  = wx.Size( 60, 20 ) )
        wx.EVT_KILL_FOCUS( text, self.update_object_on_enter )
        sizer.Add( text, 0, wx.LEFT | wx.EXPAND, 8 )
        
        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )
        label_lo.SetLabel( self.factory.low_label  )
        label_hi.SetLabel( self.factory.high_label )
        self.set_tooltip( slider )
        self.set_tooltip( label_lo )
        self.set_tooltip( label_hi )
        self.set_tooltip( text )
       
    #---------------------------------------------------------------------------
    #  Handles the user changing the current slider value: 
    #---------------------------------------------------------------------------
    
    def update_object_on_scroll ( self, event ):
        """ Handles the user changing the current slider value.
        """
        value = self.factory.low + ((float( event.GetPosition() ) / 10000.0) * 
                              (self.factory.high - self.factory.low))
        self.control.text.SetValue( self._format % value )
        event_type = event.GetEventType()
        if ((event_type == wx.wxEVT_SCROLL_ENDSCROLL) or
            (self.factory.auto_set and 
             (event_type == wx.wxEVT_SCROLL_THUMBTRACK))):
            if self.factory.is_float:
                self.value = value
            else:
                self.value = int( value )

    #---------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #---------------------------------------------------------------------------
 
    def update_object_on_enter ( self, event ):
        try:
            self.value = value = eval( self.control.text.GetValue().strip() )
            self.control.slider.SetValue( 
                int( ((float( value ) - self.factory.low) / 
                     (self.factory.high - self.factory.low)) * 10000 ) )
        except TraitError, excp:
            pass
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        value = self.value
        low   = self.factory.low
        high  = self.factory.high
        try:
            text = self._format % value
            1 / (low <= value <= high)
        except:
            text  = ''
            value = low
        ivalue = int( (float( value - low ) / (high - low)) * 10000.0 )
        self.control.text.SetValue( text )
        self.control.slider.SetValue( ivalue )
                                      
#-------------------------------------------------------------------------------
#  'SimpleSpinEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleSpinEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = wx.SpinCtrl( parent, -1, self.str_value,
                                    min     = self.factory.low,
                                    max     = self.factory.high,
                                    initial = self.value )
        wx.EVT_SPINCTRL( parent, self.control.GetId(), self.update_object )
        wx.EVT_TEXT(     parent, self.control.GetId(), self.update_object )

    #---------------------------------------------------------------------------
    #  Handle the user selecting a new value from the spin control:
    #---------------------------------------------------------------------------
  
    def update_object ( self, event ):
        self.value = self.control.GetValue()
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        try:
            self.control.SetValue( int( self.value ) )
        except:
            pass
                                      
#-------------------------------------------------------------------------------
#  'RangeTextEditor' class:
#-------------------------------------------------------------------------------
                               
class RangeTextEditor ( TextEditor ):

    #---------------------------------------------------------------------------
    #  Handles the user entering input data in the edit control:
    #---------------------------------------------------------------------------
  
    def update_object ( self, event ):
        """ Handles the user entering input data in the edit control.
        """
        try:
            self.value = eval( self.control.GetValue() )
            self.control.SetBackgroundColour( OKColor )
        except:
            self.control.SetBackgroundColour( ErrorColor )
        self.control.Refresh()
       

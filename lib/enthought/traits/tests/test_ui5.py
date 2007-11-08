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
# Author: David C. Morrill Date: 11/02/2004 Description: Test case for Traits
# User Interface
# ------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import wx

from enthought.traits.api    import Trait, HasTraits, Str, Int, Range, List, Event,\
                                File, Directory, true
from enthought.traits.ui.api import View, Handler, Item, CheckListEditor, \
                                ButtonEditor, FileEditor, DirectoryEditor, \
                                ImageEnumEditor
from enthought.traits.api import Color, Font

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

origin_values = [ 'top left', 'top right', 'bottom left', 'bottom right' ]
        
#-------------------------------------------------------------------------------
#  'Instance' class:
#-------------------------------------------------------------------------------
            
class Instance ( HasTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    integer_text = Int( 1 )
    enumeration  = Trait( 'one', 'two', 'three', 'four', 'five', 'six', 
                          cols = 3 )
    float_range  = Range( 0.0, 10.0, 10.0 )
    int_range    = Range( 1, 5 )
    boolean      = true
    
    view         = View( 'integer_text', 'enumeration', 'float_range', 
                         'int_range', 'boolean' )

#-------------------------------------------------------------------------------
#  'TraitsTestHandler' class:
#-------------------------------------------------------------------------------
                      
class TraitsTestHandler ( Handler ):
    
    def object_enabled_changed ( self, info ):
        enabled = info.object.enabled
        for i in range( 1, 63 ):
            getattr( info, 'f%d' % i ).enabled = enabled

#-------------------------------------------------------------------------------
#  'TraitsTest' class
#-------------------------------------------------------------------------------

class TraitsTest ( HasTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    enabled      = true
    integer_text = Int( 1 )
    enumeration  = Trait( 'one', 'two', 'three', 'four', 'five', 'six', 
                          cols = 3 )
    float_range  = Range( 0.0, 10.0, 10.0 )
    int_range    = Range( 1, 6 )
    int_range2   = Range( 1, 50 )
    compound     = Trait( 1, Range( 1, 6 ), 
                          'one', 'two', 'three', 'four', 'five', 'six' )
    boolean      = true
    instance     = Trait( Instance() )
    color        = Color( 'cyan' )
    font         = Font()
    check_list   = List( editor = CheckListEditor( 
                                   values = [ 'one', 'two', 'three', 'four' ],
                                   cols   = 4 ) )
    list         = List( Str, [ 'East of Eden', 'The Grapes of Wrath', 
                                'Of Mice and Men' ] )
    button       = Event( 0, editor = ButtonEditor( label = 'Click' ) )                                  
    file         = File()
    directory    = Directory()
    image_enum   = Trait( editor = ImageEnumEditor( values = origin_values,
                                                   suffix = '_origin', 
                                                   cols   = 4,
                                                   klass  = Instance ),
                         *origin_values )
                         
    #---------------------------------------------------------------------------
    #  View definitions: 
    #---------------------------------------------------------------------------
        
    view = View( 
                 ( '|{Enum}', 
                   ( 'enabled', ),
                   ( '|<[Enumeration]',  'f1:enumeration[Simple]',  '_',
                                         'f2:enumeration[Custom]@', '_',
                                         'f3:enumeration[Text]*',   '_',
                                         'f4:enumeration[Readonly]~' ),
                   ( '|<[Check List]',   'f5:check_list[Simple]',  '_',
                                         'f6:check_list[Custom]@', '_',
                                         'f7:check_list[Text]*',   '_',
                                         'f8:check_list[Readonly]~' )
                 ),                      
                 ( '|{Range}',           
                   ( '|<[Float Range]',  'f9:float_range[Simple]',  '_',
                                         'f10:float_range[Custom]@', '_',
                                         'f11:float_range[Text]*',   '_',
                                         'f12:float_range[Readonly]~' ),
                   ( '|<[Int Range]',    'f13:int_range[Simple]',  '_',
                                         'f14:int_range[Custom]@', '_',
                                         'f15:int_range[Text]*',   '_',
                                         'f16:int_range[Readonly]~' ),
                   ( '|<[Int Range 2]',  'f17:int_range2[Simple]',  '_', 
                                         'f18:int_range2[Custom]@', '_',
                                         'f19:int_range2[Text]*',   '_',
                                         'f20:int_range2[Readonly]~' )
                 ),                      
                 ( '|{Misc}',            
                   ( '|<[Integer Text]', 'f21:integer_text[Simple]',  '_',
                                         'f22:integer_text[Custom]@', '_',
                                         'f23:integer_text[Text]*',   '_',
                                         'f24:integer_text[Readonly]~' ),
                   ( '|<[Compound]',     'f25:compound[Simple]',  '_',
                                         'f26:compound[Custom]@', '_',
                                         'f27:compound[Text]*',   '_',
                                         'f28:compound[Readonly]~' ),
                   ( '|<[Boolean]',      'f29:boolean[Simple]',  '_',
                                         'f30:boolean[Custom]@', '_',
                                         'f31:boolean[Text]*',   '_',
                                         'f32:boolean[Readonly]~' )
                 ),                      
                 ( '|{Color/Font}',                        
                   ( '|<[Color]',        'f33:color[Simple]',  '_', 
                                         'f34:color[Custom]@', '_',
                                         'f35:color[Text]*',   '_',
                                         'f36:color[Readonly]~' ),
                   ( '|<[Font]',         'f37:font[Simple]', '_',
                                         'f38:font[Custom]@', '_',
                                         'f39:font[Text]*',   '_',
                                         'f40:font[Readonly]~' )
                 ),                      
                 ( '|{List}',            
                   ( '|<[List]',         'f41:list[Simple]',  '_',
                                         'f42:list[Custom]@', '_',
                                         'f43:list[Text]*',   '_',
                                         'f44:list[Readonly]~' )
                 ),                      
                 ( '|{Button}',          
                   ( '|<[Button]',       'f45:button[Simple]',  '_',
                                         'f46:button[Custom]@' ),
#                                        'button[Text]*', 
#                                        'button[Readonly]~' ),
                   ( '|<[Image Enum]',   'f47:image_enum[Simple]',  '_',
                                         'f48:image_enum[Custom]@', '_',
                                         'f49:image_enum[Text]*',   '_',
                                         'f50:image_enum[Readonly]~' ),
                   ( '|<[Instance]',     'f51:instance[Simple]',  '_',
                                         'f52:instance[Custom]@', '_',
                                         'f53:instance[Text]*',   '_',
                                         'f54:instance[Readonly]~' ),
                 ),                      
                 ( '|{File}',            
                                         
                   ( '|<[File]',         'f55:file[Simple]',  '_',
                                         'f56:file[Custom]@', '_',
                                         'f57:file[Text]*',   '_',
                                         'f58:file[Readonly]~', ),
                   ( '|<[Directory]',    'f59:directory[Simple]',  '_', 
                                         'f60:directory[Custom]@', '_',
                                         'f61:directory[Text]*',   '_',
                                         'f62:directory[Readonly]~' )
                 ),
                 apply   = True,
                 revert  = True,
                 undo    = True,
                 ok      = True,
                 handler = TraitsTestHandler()
               )

#-------------------------------------------------------------------------------
#  'TraitSheetApp' class:  
#-------------------------------------------------------------------------------

class TraitSheetApp ( wx.App ):

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, object ):
        self.object = object
        wx.InitAllImageHandlers()
        wx.App.__init__( self, 1, 'debug.log' )
        self.MainLoop()
    
    #---------------------------------------------------------------------------
    #  Handle application initialization:
    #---------------------------------------------------------------------------
 
    def OnInit ( self ):
        ui = self.object.edit_traits( kind = 'live' )
        ui = self.object.edit_traits( kind = 'modal' )
        ui = self.object.edit_traits( kind = 'nonmodal' )
        ui = self.object.edit_traits( kind = 'wizard' )
        self.SetTopWindow( ui.control )
        return True
    
#-------------------------------------------------------------------------------
#  Main program:
#-------------------------------------------------------------------------------
    
TraitSheetApp( TraitsTest() )


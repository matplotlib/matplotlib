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
                                File, Directory, true, Color, Font, Enum
from enthought.traits.ui.api import View, Handler, Item, CheckListEditor, \
                                ButtonEditor, FileEditor, DirectoryEditor, \
                                ImageEnumEditor

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
    enumeration  = Enum( 'one', 'two', 'three', 'four', 'five', 'six', 
                         cols = 3 )
    float_range  = Range( 0.0, 10.0, 10.0 )
    int_range    = Range( 1, 5 )
    boolean      = true
    
    view         = View( 'integer_text', 'enumeration', 'float_range', 
                         'int_range', 'boolean' )

#-------------------------------------------------------------------------------
#  'TraitsTest' class
#-------------------------------------------------------------------------------

class TraitsTest ( HasTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    integer_text = Int( 1 )
    enumeration  = Enum( 'one', 'two', 'three', 'four', 'five', 'six', 
                         cols = 3 )
    float_range  = Range( 0.0, 10.0, 10.0 )
    int_range    = Range( 1, 6 )
    int_range2   = Range( 1, 50 )
    compound     = Trait( 1, Range( 1, 6 ), 
                          'one', 'two', 'three', 'four', 'five', 'six' )
    boolean      = true
    instance     = Trait( Instance() )
    color        = Color
    font         = Font
    check_list   = List( editor = CheckListEditor( 
                                   values = [ 'one', 'two', 'three', 'four' ],
                                   cols   = 4 ) )
    list         = List( Str, [ 'East of Eden', 'The Grapes of Wrath', 
                                'Of Mice and Men' ] )
    button       = Event( 0, editor = ButtonEditor( label = 'Click' ) )                                  
    file         = File
    directory    = Directory
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
                   ( '|<[Enumeration]',  'enumeration[Simple]',  '_',
                                         'enumeration[Custom]@', '_',
                                         'enumeration[Text]*',   '_',
                                         'enumeration[Readonly]~' ),
                   ( '|<[Check List]',   'check_list[Simple]',  '_',
                                         'check_list[Custom]@', '_',
                                         'check_list[Text]*',   '_',
                                         'check_list[Readonly]~' )
                 ),                      
                 ( '|{Range}',           
                   ( '|<[Float Range]',  'float_range[Simple]',  '_',
                                         'float_range[Custom]@', '_',
                                         'float_range[Text]*',   '_',
                                         'float_range[Readonly]~' ),
                   ( '|<[Int Range]',    'int_range[Simple]',  '_',
                                         'int_range[Custom]@', '_',
                                         'int_range[Text]*',   '_',
                                         'int_range[Readonly]~' ),
                   ( '|<[Int Range 2]',  'int_range2[Simple]',  '_', 
                                         'int_range2[Custom]@', '_',
                                         'int_range2[Text]*',   '_',
                                         'int_range2[Readonly]~' )
                 ),                      
                 ( '|{Misc}',            
                   ( '|<[Integer Text]', 'integer_text[Simple]',  '_',
                                         'integer_text[Custom]@', '_',
                                         'integer_text[Text]*',   '_',
                                         'integer_text[Readonly]~' ),
                   ( '|<[Compound]',     'compound[Simple]',  '_',
                                         'compound[Custom]@', '_',
                                         'compound[Text]*',   '_',
                                         'compound[Readonly]~' ),
                   ( '|<[Boolean]',      'boolean[Simple]',  '_',
                                         'boolean[Custom]@', '_',
                                         'boolean[Text]*',   '_',
                                         'boolean[Readonly]~' )
                 ),                      
                 ( '|{Color/Font}',                        
                   ( '|<[Color]',        'color[Simple]',  '_', 
                                         'color[Custom]@', '_',
                                         'color[Text]*',   '_',
                                         'color[Readonly]~' ),
                   ( '|<[Font]',          'font[Simple]', '_',
                                         'font[Custom]@', '_',
                                         'font[Text]*',   '_',
                                         'font[Readonly]~' )
                 ),                      
                 ( '|{List}',            
                   ( '|<[List]',         'list[Simple]',  '_',
                                         'list[Custom]@', '_',
                                         'list[Text]*',   '_',
                                         'list[Readonly]~' )
                 ),                      
                 ( '|{Button}',          
                   ( '|<[Button]',       'button[Simple]',  '_',
                                         'button[Custom]@' ),
#                                        'button[Text]*', 
#                                        'button[Readonly]~' ),
                   ( '|<[Image Enum]',   'image_enum[Simple]',  '_',
                                         'image_enum[Custom]@', '_',
                                         'image_enum[Text]*',   '_',
                                         'image_enum[Readonly]~' ),
                   ( '|<[Instance]',     'instance[Simple]',  '_',
                                         'instance[Custom]@', '_',
                                         'instance[Text]*',   '_',
                                         'instance[Readonly]~' ),
                 ),                      
                 ( '|{File}',            
                                         
                   ( '|<[File]',         'file[Simple]',  '_',
                                         'file[Custom]@', '_',
                                         'file[Text]*',   '_',
                                         'file[Readonly]~', ),
                   ( '|<[Directory]',    'directory[Simple]',  '_', 
                                         'directory[Custom]@', '_',
                                         'directory[Text]*',   '_',
                                         'directory[Readonly]~' )
                 ),
                 apply  = True,
                 revert = True,
                 undo   = True,
                 ok     = True
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
        ui = self.object.edit_traits( kind = 'modal' )
        ui = self.object.edit_traits( kind = 'wizard' )
        ui = self.object.edit_traits( kind = 'nonmodal' )
        ui = self.object.edit_traits( kind = 'live' )
        self.SetTopWindow( ui.control )
        return True
    
#-------------------------------------------------------------------------------
#  Main program:
#-------------------------------------------------------------------------------
    
if __name__ == '__main__':
    TraitSheetApp( TraitsTest() )


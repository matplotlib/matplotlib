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
# Description: Define the Tkinter implementation of the various list editors and
#              the list editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from constants        import scrollbar_dx
from editor_factory   import EditorFactory
from editor           import Editor
from enthought.traits.api import Trait, HasTraits, TraitHandler, Range, Str
from helper           import bitmap_cache
from menu             import MakeMenu
from image_control    import ImageControl

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Require the value to be a TraitHandler object:
handler_trait = Trait( TraitHandler )

# The visible number of rows displayed:
rows_trait = Range( 1, 50, 5,
                    desc = "the number of list rows to display" )

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    trait_handler = handler_trait  # The trait handler for each list item
    rows          = rows_trait     # Number of list rows to display
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description,
                             kind        = 'simple_editor' )
    
    def custom_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description,
                             kind        = 'custom_editor' )
    
    def text_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description,
                             kind        = 'text_editor' )
    
    def readonly_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description,
                             kind        = 'readonly_editor' )
                                      
#-------------------------------------------------------------------------------
#  'SimpleEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleEditor ( Editor ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    kind = Str  # The kind of editor to create for each list item
    
    #---------------------------------------------------------------------------
    #  Normal list item menu:
    #---------------------------------------------------------------------------
    
    list_menu = """
       Add Before     [_menu_before]: self.add_before()
       Add After      [_menu_after]:  self.add_after()
       ---
       Delete         [_menu_delete]: self.delete_item()
       ---
       Move Up        [_menu_up]:     self.move_up()
       Move Down      [_menu_down]:   self.move_down()
       Move to Top    [_menu_top]:    self.move_top() 
       Move to Bottom [_menu_bottom]: self.move_bottom()
    """
 
    #---------------------------------------------------------------------------
    #  Empty list item menu:
    #---------------------------------------------------------------------------
    
    empty_list_menu = """
       Add: self.add_empty()
    """
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        # Create a scrolled window to hold all of the list item controls:
        self.control = wx.ScrolledWindow( parent, -1 )
        self.control.SetAutoLayout( True )
        
        # Remember the editor to use for each individual list item:
        self._editor = getattr( 
                           self.factory.trait_handler.item_trait.get_editor(), 
                           self.kind )
                     
        # Set up the additional 'list items changed' event handler needed for
        # a list based trait:
        self.object.on_trait_change( self.update_editor_item, 
                                     self.name + '_items', dispatch = 'ui' )
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        list_pane = self.control
        editor    = self._editor

        # Disconnext the editor from any control about to be destroyed:        
        for control in list_pane.GetChildren():
            if hasattr( control, '_editor' ):
                control._editor.control = None
                
        # Get rid of any previous contents:
        list_pane.SetSizer( None )
        list_pane.DestroyChildren()
        
        # Create all of the list item trait editors:
        trait_handler = self.factory.trait_handler
        resizable     = ((trait_handler.minlen != trait_handler.maxlen) and
                         (self.kind != 'readonly_editor'))
        item_trait    = trait_handler.item_trait
        list_sizer    = wx.FlexGridSizer( 0, 1 + resizable, 0, 0 )
        list_sizer.AddGrowableCol( resizable )
        values        = self.value
        index         = 0
        width, height = 100, 18
        is_fake       = (resizable and (len( values ) == 0))
        if is_fake:
            values = [ item_trait.default_value()[1] ]
            
        for value in values:
            width = height = 0
            if resizable:       
                control = ImageControl( list_pane, 
                                        bitmap_cache( 'list_editor', False ),
                                        -1, self.popup_menu )                                   
                width, height = control.GetSize()
                width += 4
            try:
                proxy    = ListItemProxy( self.object, self.name, index, 
                                          item_trait, value )
                peditor  = editor( self.ui, proxy, 'value', self.description, 
                                   list_pane )
                pcontrol = peditor.control
                pcontrol.proxy = proxy
                if resizable:
                    control.proxy = proxy
            except:
                if not is_fake:
                    raise
                pcontrol = wx.Button( list_pane, -1, 'sample' )
            width2, height2 = pcontrol.GetSize()
            width += width2
            height = max( height, height2 )
            if resizable:
                list_sizer.Add( control, 0, wx.LEFT | wx.RIGHT, 2 )
            list_sizer.Add( pcontrol, 1, wx.EXPAND )
            index += 1
            
        list_pane.SetSizer( list_sizer )
        
        if is_fake:
           self._cur_control = control   
           self.empty_list()
           control.Destroy()             
           pcontrol.Destroy()
           
        rows = [ self.factory.rows, 1 ][ self.kind == 'simple_editor' ]
        list_pane.SetSize( wx.Size( 
             width + ((trait_handler.maxlen > rows) * scrollbar_dx), 
             height * rows ) )
        list_pane.SetScrollRate( 16, height )
        list_pane.SetVirtualSize( list_sizer.GetMinSize() )
        list_pane.GetParent().Layout()
        
    #---------------------------------------------------------------------------
    #  Updates the editor when an item in the object trait changes external to 
    #  the editor:
    #---------------------------------------------------------------------------
        
    def update_editor_item ( self, event ):
        """ Updates the editor when an item in the object trait changes external 
            to the editor:
        """
        # If this is not a simple, single item update, rebuild entire editor:
        if (len( event.removed ) != 1) or (len( event.added ) != 1):
            self.update_editor()
        
        # Otherwise, find the proxy for this index and update it with the 
        # changed value: 
        for control in self.control.GetChildren():
            proxy = control.proxy
            if proxy.index == event.index:
                proxy.value = event.added[0]
                break

    #---------------------------------------------------------------------------
    #  Creates an empty list entry (so the user can add a new item):
    #---------------------------------------------------------------------------
           
    def empty_list ( self ):
        """ Creates an empty list entry (so the user can add a new item).
        """
        control = ImageControl( self.control, 
                                bitmap_cache( 'list_editor', False ),
                                -1, self.popup_empty_menu )                                   
        control.is_empty = True
        proxy    = ListItemProxy( self.object, self.name, -1, None, None )
        pcontrol = wx.StaticText( self.control, -1, '   (Empty List)' )
        pcontrol.proxy = control.proxy = proxy
        self.reload_sizer( [ ( control, pcontrol ) ] )
  
    #---------------------------------------------------------------------------
    #  Reloads the layout from the specified list of ( button, proxy ) pairs:
    #---------------------------------------------------------------------------
          
    def reload_sizer ( self, controls, extra = 0 ):
        """ Reloads the layout from the specified list of ( button, proxy ) 
            pairs.
        """
        sizer = self.control.GetSizer()
        for i in xrange( 2 * len( controls ) + extra ):
            sizer.Remove( 0 )
        index = 0
        for control, pcontrol in controls:
            sizer.Add( control,  0, wx.LEFT | wx.RIGHT, 2 )
            sizer.Add( pcontrol, 1, wx.EXPAND )
            control.proxy.index = index
            index += 1
        sizer.Layout()
        self.control.SetVirtualSize( sizer.GetMinSize() )
       
    #---------------------------------------------------------------------------
    #  Returns the associated object list and current item index:
    #---------------------------------------------------------------------------
     
    def get_info ( self ):
        """ Returns the associated object list and current item index.
        """
        proxy = self._cur_control.proxy
        return ( proxy.list(), proxy.index )
        
    #---------------------------------------------------------------------------
    #  Displays the empty list editor popup menu:
    #---------------------------------------------------------------------------
    
    def popup_empty_menu ( self, control ):
        """ Displays the empty list editor popup menu.
        """
        self._cur_control = control
        control.PopupMenuXY( MakeMenu( self.empty_list_menu, self, True, 
                                       control ).menu, 0, 0 )
       
    #---------------------------------------------------------------------------
    #  Displays the list editor popup menu:
    #---------------------------------------------------------------------------
    
    def popup_menu ( self, control ):
        """ Displays the list editor popup menu.
        """
        self._cur_control = control
        proxy    = control.proxy
        index    = proxy.index
        menu     = MakeMenu( self.list_menu, self, True, control ).menu
        len_list = len( proxy.list() )
        not_full = (len_list < self.factory.trait_handler.maxlen)
        self._menu_before.enabled( not_full )
        self._menu_after.enabled(  not_full )
        self._menu_delete.enabled( len_list > self.factory.trait_handler.minlen )
        self._menu_up.enabled(  index > 0 )
        self._menu_top.enabled( index > 0 )
        self._menu_down.enabled(   index < (len_list - 1) )
        self._menu_bottom.enabled( index < (len_list - 1) )
        control.PopupMenuXY( menu, 0, 0 )

    #---------------------------------------------------------------------------
    #  Adds a new value at the specified list index:
    #---------------------------------------------------------------------------
           
    def add_item ( self, offset ):
        """ Adds a new value at the specified list index.
        """
        list, index = self.get_info()
        index      += offset 
        item_trait  = self.factory.trait_handler.item_trait
        value       = item_trait.default_value()[1]
        self.value  = list[:index] + [ value ] + list[index:]
        
    #---------------------------------------------------------------------------
    #  Inserts a new item before the current item:
    #---------------------------------------------------------------------------
           
    def add_before ( self ):
        """ Inserts a new item before the current item.
        """
        self.add_item( 0 )
        
    #---------------------------------------------------------------------------
    #  Inserts a new item after the current item:
    #---------------------------------------------------------------------------
    
    def add_after ( self ):
        """ Inserts a new item after the current item.
        """
        self.add_item( 1 )
        
    #---------------------------------------------------------------------------
    #  Adds a new item when the list is empty:
    #---------------------------------------------------------------------------
    
    def add_empty ( self ):
        """ Adds a new item when the list is empty.
        """
        self.add_item( 0 )
        self.delete_item()
        
    #---------------------------------------------------------------------------
    #  Delete the current item:
    #---------------------------------------------------------------------------
    
    def delete_item ( self ):
        """ Delete the current item.
        """
        list, index = self.get_info()
        self.value  = list[:index] + list[index+1:]
        
    #---------------------------------------------------------------------------
    #  Move the current item up one in the list:
    #---------------------------------------------------------------------------
       
    def move_up ( self ):
        """ Move the current item up one in the list.
        """
        list, index = self.get_info()
        self.value  = (list[:index-1] + [ list[index], list[index-1] ] + 
                       list[index+1:])
       
    #---------------------------------------------------------------------------
    #  Moves the current item down one in the list:
    #---------------------------------------------------------------------------
    
    def move_down ( self ):
        """ Moves the current item down one in the list.
        """
        list, index = self.get_info()
        self.value  = (list[:index] + [ list[index+1], list[index] ] + 
                       list[index+2:])
        
    #---------------------------------------------------------------------------
    #  Moves the current item to the top of the list:
    #---------------------------------------------------------------------------
    
    def move_top ( self ):
        """ Moves the current item to the top of the list.
        """
        list, index = self.get_info()
        self.value  = [ list[index] ] + list[:index] + list[index+1:]
         
    #---------------------------------------------------------------------------
    #  Moves the current item to the bottom of the list:
    #---------------------------------------------------------------------------
    
    def move_bottom ( self ):
        """ Moves the current item to the bottom of the list.
        """
        list, index = self.get_info()
        self.value  = list[:index] + list[index+1:] + [ list[index] ] 
   
#-------------------------------------------------------------------------------
#  'ListItemProxy' class:
#-------------------------------------------------------------------------------
       
class ListItemProxy ( HasTraits ):

    def __init__ ( self, object, name, index, trait, value ):
        HasTraits.__init__( self )
        self.inited = False
        self.object = object
        self.name   = name
        self.index  = index
        if trait is not None:
            self.add_trait( 'value', trait )
            self.value  = value
        self.inited = True
        
    def list ( self ):
        return getattr( self.object, self.name )
        
    def _value_changed ( self, old_value, new_value ):
        if self.inited:
            self.list()[ self.index ] = new_value     
        

#-------------------------------------------------------------------------------
#
#  Define the tree node descriptor used by the tree editor and tree editor 
#  factory classes.
#
#  Written by: David C. Morrill
#
#  Date: 12/03/2004
#
#  Symbols defined: TreeNode
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import copy

from inspect                     import stack

from matplotlib.enthought.traits            import HasTraits, HasPrivateTraits, Str, List,\
                                        Callable, Instance, Any, true, false
from matplotlib.enthought.traits.trait_base import SequenceTypes                                
from matplotlib.enthought.traits.ui         import View

from matplotlib.enthought.resource          import resource_path

#-------------------------------------------------------------------------------
#  'TreeNode' class:
#-------------------------------------------------------------------------------

class TreeNode ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    children   = Str   # Name of trait containing children ('' = Leaf)
    label      = Str   # Name of trait containing label ('=label' = constant)
    name       = Str   # Name to use when a new instance is being created
    rename     = true  # Can the object's children be renamed?
    copy       = true  # Can the object's children be copied?
    delete     = true  # Can the object's children be deleted?
    insert     = true  # Can children be inserted (or just appended)?
    auto_open  = false # Automatically open ( i.e. expand) tree nodes?
    auto_close = false # Automatically close sibling tree nodes?
    add        = List( Any ) # List of object classes than can be added/copied
    move       = List( Any ) # List of object classes that can be moved
    node_for   = List( Any ) # List of object classes the node applies to
    formatter  = Callable    # Function for formatting label
    on_select  = Callable    # Function for handling selecting an object 
    on_dclick  = Callable    # Function for handling double_clicking an object 
    view       = Instance( View ) # View to use for editing object
    menu       = Any              # Right click context menu 
    icon_item  = Str( '<item>' )  # Name of leaf item icon
    icon_group = Str( '<group>' ) # Name of group item icon
    icon_open  = Str( '<open>' )  # Name of opened group item icon
    icon_path  = Str              # Resource path used to locate node icon
    
    # fixme: The 'menu' trait should really be defined as:
    #        Instance( 'enthought.traits.ui.menu.MenuBar' ), but it doesn't work
    #        right currently.
    
    #---------------------------------------------------------------------------
    #  Initializes the object:  
    #---------------------------------------------------------------------------
        
    def __init__ ( self, **traits ):
        super( TreeNode, self ).__init__( **traits )
        if self.icon_path == '':
            self.icon_path = resource_path()
    
#---- Overridable methods: -----------------------------------------------------

    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:  
    #---------------------------------------------------------------------------

    def allows_children ( self, object ):
        """ Returns whether chidren of this object are allowed or not.
        """
        return (self.children != '')
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def has_children ( self, object ):
        """ Returns whether or not the object has children.
        """
        return (len( self.get_children( object ) ) > 0)
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def get_children ( self, object ):
        """ Gets the object's children.
        """
        return getattr( object, self.children )
        
    #---------------------------------------------------------------------------
    #  Appends a child to the object's children:  
    #---------------------------------------------------------------------------
                
    def append_child ( self, object, child ):
        """ Appends a child to the object's children.
        """
        getattr( object, self.children ).append( child )
        
    #---------------------------------------------------------------------------
    #  Inserts a child into the object's children:  
    #---------------------------------------------------------------------------
                
    def insert_child ( self, object, index, child ):
        """ Inserts a child into the object's children.
        """
        getattr( object, self.children )[ index: index ] = [ child ]
        
    #---------------------------------------------------------------------------
    #  Deletes a child at a specified index from the object's children:  
    #---------------------------------------------------------------------------
                
    def delete_child ( self, object, index ):
        """ Deletes a child at a specified index from the object's children.
        """
        del getattr( object, self.children )[ index ]
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children replaced' on a specified  
    #  object:  
    #---------------------------------------------------------------------------

    def when_children_replaced ( self, object, listener, remove ):
        """ Sets up/Tears down a listener for 'children replaced' on a specified  
            object.
        """
        object.on_trait_change( listener, self.children, remove = remove )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified 
    #  object:  
    #---------------------------------------------------------------------------
                
    def when_children_changed ( self, object, listener, remove ):
        """ Sets up/Tears down a listener for 'children changed' on a specified
            object.
        """
        object.on_trait_change( listener, self.children + '_items', 
                                remove = remove )
        
    #---------------------------------------------------------------------------
    #  Gets the label to display for a specified object:    
    #---------------------------------------------------------------------------
        
    def get_label ( self, object ):
        """ Gets the label to display for a specified object.
        """
        label = self.label
        if label[:1] == '=':
            return label[1:]
        label = getattr( object, label )
        if self.formatter is None:
            return label
        return self.formatter( object, label )
        
    #---------------------------------------------------------------------------
    #  Sets the label for a specified object:  
    #---------------------------------------------------------------------------
                
    def set_label ( self, object, label ):        
        """ Sets the label for a specified object.
        """
        label_name = self.label
        if label_name[:1] != '=':
            setattr( object, label_name, label )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'label changed' on a specified object:
    #---------------------------------------------------------------------------
                
    def when_label_changed ( self, object, listener, remove ):
        """ Sets up/Tears down a listener for 'label changed' on a specified
            object.
        """
        label = self.label
        if label[:1] != '=':
            object.on_trait_change( listener, label, remove = remove )
        
    #---------------------------------------------------------------------------
    #  Returns the icon for a specified object:  
    #---------------------------------------------------------------------------
                
    def get_icon ( self, object, is_expanded ):
        """ Returns the icon for a specified object.
        """
        if not self.allows_children( object ):
            return self.icon_item
        if is_expanded:
            return self.icon_open
        return self.icon_group
        
    #---------------------------------------------------------------------------
    #  Returns the path used to locate an object's icon:  
    #---------------------------------------------------------------------------
                
    def get_icon_path ( self, object ):
        """ Returns the path used to locate an object's icon.
        """
        return self.icon_path

    #---------------------------------------------------------------------------
    #  Returns the name to use when adding a new object instance (displayed in
    #  the 'New' submenu):
    #---------------------------------------------------------------------------
                
    def get_name ( self, object ):
        """ Returns the name to use when adding a new object instance
            (displayed in the 'New' submenu).
        """
        return self.name
        
    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:  
    #---------------------------------------------------------------------------
                
    def get_view ( self, object ):
        """ Gets the View to use when editing an object.
        """
        return self.view   
        
    #---------------------------------------------------------------------------
    #  Returns the right-click context menu for an object:  
    #---------------------------------------------------------------------------
                
    def get_menu ( self, object ):
        """ Returns the right-click context menu for an object.
        """
        return self.menu        

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be renamed:
    #---------------------------------------------------------------------------
                
    def can_rename ( self, object ):
        """ Returns whether or not the object's children can be renamed.
        """
        return self.rename

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------
    
    def can_copy ( self, object ):
        """ Returns whether or not the object's children can be copied.
        """
        return self.copy

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------
    
    def can_delete ( self, object ):
        """ Returns whether or not the object's children can be deleted.
        """
        return self.delete

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------
    
    def can_insert ( self, object ):
        """ Returns whether or not the object's children can be inserted (or 
            just appended).
        """
        return self.insert

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def can_auto_open ( self, object ):
        """ Returns whether or not the object's children should be auto-opened.
        """
        return self.auto_open

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def can_auto_close ( self, object ):
        """ Returns whether or not the object's children should be auto-closed.
        """
        return self.auto_close
        
    #---------------------------------------------------------------------------
    #  Returns whether or not this is the node that should handle a specified
    #  object:
    #---------------------------------------------------------------------------
                
    def is_node_for ( self, object ):
        """ Returns whether or not this is the node that should handle a 
            specified object.
        """
        return isinstance( object, tuple( self.node_for ) )
 
    #---------------------------------------------------------------------------
    #  Returns whether a given 'add_object' can be added to an object:
    #---------------------------------------------------------------------------
                                
    def can_add ( self, object, add_object ):
        """ Returns whether a given object is droppable on the node.
        """
        if isinstance( add_object, tuple( self.add ) ):
            return True
        for item in self.move:
            if type( item ) in SequenceTypes:
                item = item[0]
            if isinstance( add_object, item ):
                return True
        return False
        
    #---------------------------------------------------------------------------
    #  Returns the list of classes that can be added to the object:
    #---------------------------------------------------------------------------
    
    def get_add ( self, object ):
        """ Returns the list of classes that can be added to the object.
        """
        return self.add
                           
    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:    
    #---------------------------------------------------------------------------
                                                      
    def drop_object ( self, object, dropped_object ):
        """ Returns a droppable version of a specified object.
        """
        if isinstance( dropped_object, tuple( self.add ) ):
            return dropped_object
            
        for item in self.move:
            if type( item ) in SequenceTypes:
                if isinstance( dropped_object, item[0] ):
                    return item[1]( object, dropped_object )
            else:
                if isinstance( dropped_object, item ):
                    return dropped_object
        
    #---------------------------------------------------------------------------
    #  Handles an object being selected:    
    #---------------------------------------------------------------------------
                
    def select ( self, object ):
        """ Handles an object being selected.
        """
        if self.on_select is not None:
            self.on_select( object )
            return None
        return True
        
    #---------------------------------------------------------------------------
    #  Handles an object being double-clicked:
    #---------------------------------------------------------------------------
       
    def dclick ( self, object ):
        """ Handles an object being double-clicked.
        """
        if self.on_dclick is not None:
            self.on_dclick( object )
            return None
        return True
    
#----- Private methods: --------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #  Returns whether an object has any children:    
    #---------------------------------------------------------------------------
                        
    def _has_children ( self, object ):
        """ Returns whether an object has any children.
        """
        return (self.allows_children( object ) and self.has_children( object )) 
                
    #---------------------------------------------------------------------------
    #  Returns whether a given object is droppable on the node:    
    #---------------------------------------------------------------------------
                                
    def _is_droppable ( self, object, add_object, for_insert ):
        """ Returns whether a given object is droppable on the node.
        """
        if for_insert and (not self.can_insert( object )):
            return False
        return self.can_add( object, add_object )
                           
    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:    
    #---------------------------------------------------------------------------
                                                      
    def _drop_object ( self, object, dropped_object, make_copy = True ):
        new_object = self.drop_object( object, dropped_object )
        if (new_object is not dropped_object) or (not make_copy):
            return new_object
        return copy.deepcopy( new_object )
        
#-------------------------------------------------------------------------------
#  'ObjectTreeNode' class  
#-------------------------------------------------------------------------------
                
class ObjectTreeNode ( TreeNode ):

    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:  
    #---------------------------------------------------------------------------

    def allows_children ( self, object ):
        """ Returns whether chidren of this object are allowed or not.
        """
        return object.tno_allows_children( self )
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def has_children ( self, object ):
        """ Returns whether or not the object has children.
        """
        return object.tno_has_children( self )
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def get_children ( self, object ):
        """ Gets the object's children.
        """
        return object.tno_get_children( self )
        
    #---------------------------------------------------------------------------
    #  Appends a child to the object's children:  
    #---------------------------------------------------------------------------
                
    def append_child ( self, object, child ):
        """ Appends a child to the object's children.
        """
        return object.tno_append_child( self, child )
        
    #---------------------------------------------------------------------------
    #  Inserts a child into the object's children:  
    #---------------------------------------------------------------------------
                
    def insert_child ( self, object, index, child ):
        """ Inserts a child into the object's children.
        """
        return object.tno_insert_child( self, index, child )
        
    #---------------------------------------------------------------------------
    #  Deletes a child at a specified index from the object's children:  
    #---------------------------------------------------------------------------
                
    def delete_child ( self, object, index ):
        """ Deletes a child at a specified index from the object's children.
        """
        return object.tno_delete_child( self, index )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children replaced' on a specified  
    #  object:  
    #---------------------------------------------------------------------------

    def when_children_replaced ( self, object, listener, remove ):
        """ Sets up/Tears down a listener for 'children replaced' on a specified  
            object.
        """
        return object.tno_when_children_replaced( self, listener, remove )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified 
    #  object:  
    #---------------------------------------------------------------------------
                
    def when_children_changed ( self, object, listener, remove ):
        """ Sets up/Tears down a listener for 'children changed' on a specified
            object.
        """
        return object.tno_when_children_changed( self, listener, remove )
        
    #---------------------------------------------------------------------------
    #  Gets the label to display for a specified object:    
    #---------------------------------------------------------------------------
        
    def get_label ( self, object ):
        """ Gets the label to display for a specified object.
        """
        return object.tno_get_label( self )
        
    #---------------------------------------------------------------------------
    #  Sets the label for a specified object:  
    #---------------------------------------------------------------------------
                
    def set_label ( self, object, label ):        
        """ Sets the label for a specified object.
        """
        return object.tno_set_label( self, label )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'label changed' on a specified object:
    #---------------------------------------------------------------------------
                
    def when_label_changed ( self, object, listener, remove ):
        """ Sets up/Tears down a listener for 'label changed' on a specified
            object.
        """
        return object.tno_when_label_changed( self, listener, remove )
        
    #---------------------------------------------------------------------------
    #  Returns the icon for a specified object:  
    #---------------------------------------------------------------------------
                
    def get_icon ( self, object, is_expanded ):
        """ Returns the icon for a specified object.
        """
        return object.tno_get_icon( self, is_expanded )
        
    #---------------------------------------------------------------------------
    #  Returns the path used to locate an object's icon:  
    #---------------------------------------------------------------------------
                
    def get_icon_path ( self, object ):
        """ Returns the path used to locate an object's icon.
        """
        return object.tno_get_icon_path( self )

    #---------------------------------------------------------------------------
    #  Returns the name to use when adding a new object instance (displayed in
    #  the 'New' submenu):
    #---------------------------------------------------------------------------
                
    def get_name ( self, object ):
        """ Returns the name to use when adding a new object instance
            (displayed in the 'New' submenu).
        """
        return object.tno_get_name( self )
        
    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:  
    #---------------------------------------------------------------------------
                
    def get_view ( self, object ):
        """ Gets the View to use when editing an object.
        """
        return object.tno_get_view( self )
        
    #---------------------------------------------------------------------------
    #  Returns the right-click context menu for an object:  
    #---------------------------------------------------------------------------
                
    def get_menu ( self, object ):
        """ Returns the right-click context menu for an object.
        """
        return object.tno_get_menu( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be renamed:
    #---------------------------------------------------------------------------
                
    def can_rename ( self, object ):
        """ Returns whether or not the object's children can be renamed.
        """
        return object.tno_can_rename( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------
    
    def can_copy ( self, object ):
        """ Returns whether or not the object's children can be copied.
        """
        return object.tno_can_copy( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------
    
    def can_delete ( self, object ):
        """ Returns whether or not the object's children can be deleted.
        """
        return object.tno_can_delete( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------
    
    def can_insert ( self, object ):
        """ Returns whether or not the object's children can be inserted (or 
            just appended).
        """
        return object.tno_can_insert( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def can_auto_open ( self, object ):
        """ Returns whether or not the object's children should be auto-opened.
        """
        return object.tno_can_auto_open( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def can_auto_close ( self, object ):
        """ Returns whether or not the object's children should be auto-closed.
        """
        return object.tno_can_auto_close( self )
        
    #---------------------------------------------------------------------------
    #  Returns whether or not this is the node that should handle a specified
    #  object:
    #---------------------------------------------------------------------------
                
    def is_node_for ( self, object ):
        """ Returns whether or not this is the node that should handle a 
            specified object.
        """
        return object.tno_is_node_for( self )
 
    #---------------------------------------------------------------------------
    #  Returns whether a given 'add_object' can be added to an object:
    #---------------------------------------------------------------------------
                                
    def can_add ( self, object, add_object ):
        """ Returns whether a given object is droppable on the node.
        """
        return object.tno_can_add( self, add_object )
        
    #---------------------------------------------------------------------------
    #  Returns the list of classes that can be added to the object:
    #---------------------------------------------------------------------------
    
    def get_add ( self, object ):
        """ Returns the list of classes that can be added to the object.
        """
        return object.tno_get_add( self )
                           
    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:    
    #---------------------------------------------------------------------------
                                                      
    def drop_object ( self, object, dropped_object ):
        """ Returns a droppable version of a specified object.
        """
        return object.tno_drop_object( self, dropped_object )
        
    #---------------------------------------------------------------------------
    #  Handles an object being selected:    
    #---------------------------------------------------------------------------
                
    def select ( self, object ):
        """ Handles an object being selected.
        """
        return object.tno_select( self )
        
    #---------------------------------------------------------------------------
    #  Handles an object being double-clicked:
    #---------------------------------------------------------------------------
       
    def dclick ( self, object ):
        """ Handles an object being double-clicked.
        """
        return object.tno_dclick( self )
    
#-------------------------------------------------------------------------------
#  'TreeNodeObject' class:  
#-------------------------------------------------------------------------------
                
class TreeNodeObject ( HasTraits ):

    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:  
    #---------------------------------------------------------------------------

    def tno_allows_children ( self, node ):
        """ Returns whether chidren of this object are allowed or not.
        """
        return (node.children != '')
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether or not the object has children.
        """
        return (len( self.tno_get_children( node ) ) > 0)
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def tno_get_children ( self, node ):
        """ Gets the object's children.
        """
        return getattr( self, node.children )
        
    #---------------------------------------------------------------------------
    #  Appends a child to the object's children:  
    #---------------------------------------------------------------------------
                
    def tno_append_child ( self, node, child ):
        """ Appends a child to the object's children.
        """
        getattr( self, node.children ).append( child )
        
    #---------------------------------------------------------------------------
    #  Inserts a child into the object's children:  
    #---------------------------------------------------------------------------
                
    def tno_insert_child ( self, node, index, child ):
        """ Inserts a child into the object's children.
        """
        getattr( self, node.children )[ index: index ] = [ child ]
        
    #---------------------------------------------------------------------------
    #  Deletes a child at a specified index from the object's children:  
    #---------------------------------------------------------------------------
                
    def tno_delete_child ( self, node, index ):
        """ Deletes a child at a specified index from the object's children.
        """
        del getattr( self, node.children )[ index ]
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children replaced' on a specified  
    #  object:  
    #---------------------------------------------------------------------------

    def tno_when_children_replaced ( self, node, listener, remove ):
        """ Sets up/Tears down a listener for 'children replaced' on a specified  
            object.
        """
        self.on_trait_change( listener, node.children, remove = remove )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified 
    #  object:  
    #---------------------------------------------------------------------------
                
    def tno_when_children_changed ( self, node, listener, remove ):
        """ Sets up/Tears down a listener for 'children changed' on a specified
            object.
        """
        self.on_trait_change( listener, node.children + '_items', 
                              remove = remove )
        
    #---------------------------------------------------------------------------
    #  Gets the label to display for a specified object:    
    #---------------------------------------------------------------------------
        
    def tno_get_label ( self, node ):
        """ Gets the label to display for a specified object.
        """
        label = node.label
        if label[:1] == '=':
            return label[1:]
        label = getattr( self, label )
        if node.formatter is None:
            return label
        return node.formatter( self, label )
        
    #---------------------------------------------------------------------------
    #  Sets the label for a specified node:  
    #---------------------------------------------------------------------------
                
    def tno_set_label ( self, node, label ):        
        """ Sets the label for a specified object.
        """
        label_name = node.label
        if label_name[:1] != '=':
            setattr( self, label_name, label )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'label changed' on a specified object:
    #---------------------------------------------------------------------------
                
    def tno_when_label_changed ( self, node, listener, remove ):
        """ Sets up/Tears down a listener for 'label changed' on a specified
            object.
        """
        label = node.label
        if label[:1] != '=':
            self.on_trait_change( listener, label, remove = remove )
        
    #---------------------------------------------------------------------------
    #  Returns the icon for a specified object:  
    #---------------------------------------------------------------------------
                
    def tno_get_icon ( self, node, is_expanded ):
        """ Returns the icon for a specified object.
        """
        if not self.tno_allows_children( node ):
            return node.icon_item
        if is_expanded:
            return node.icon_open
        return node.icon_group
        
    #---------------------------------------------------------------------------
    #  Returns the path used to locate an object's icon:  
    #---------------------------------------------------------------------------
                
    def tno_get_icon_path ( self, node ):
        """ Returns the path used to locate an object's icon.
        """
        return node.icon_path

    #---------------------------------------------------------------------------
    #  Returns the name to use when adding a new object instance (displayed in
    #  the 'New' submenu):
    #---------------------------------------------------------------------------
                
    def tno_get_name ( self, node ):
        """ Returns the name to use when adding a new object instance
            (displayed in the 'New' submenu).
        """
        return node.name
        
    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:  
    #---------------------------------------------------------------------------
                
    def tno_get_view ( self, node ):
        """ Gets the View to use when editing an object.
        """
        return node.view   
        
    #---------------------------------------------------------------------------
    #  Returns the right-click context menu for an object:  
    #---------------------------------------------------------------------------
                
    def tno_get_menu ( self, node ):
        """ Returns the right-click context menu for an object.
        """
        return node.menu        

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be renamed:
    #---------------------------------------------------------------------------
                
    def tno_can_rename ( self, node ):
        """ Returns whether or not the object's children can be renamed.
        """
        return node.rename

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------
    
    def tno_can_copy ( self, node ):
        """ Returns whether or not the object's children can be copied.
        """
        return node.copy

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------
    
    def tno_can_delete ( self, node ):
        """ Returns whether or not the object's children can be deleted.
        """
        return node.delete

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------
    
    def tno_can_insert ( self, node ):
        """ Returns whether or not the object's children can be inserted (or 
            just appended).
        """
        return node.insert

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def tno_can_auto_open ( self, node ):
        """ Returns whether or not the object's children should be auto-opened.
        """
        return node.auto_open

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def tno_can_auto_close ( self, node ):
        """ Returns whether or not the object's children should be auto-closed.
        """
        return node.auto_close
        
    #---------------------------------------------------------------------------
    #  Returns whether or not this is the node that should handle a specified
    #  object:
    #---------------------------------------------------------------------------
                
    def tno_is_node_for ( self, node ):
        """ Returns whether or not this is the node that should handle a 
            specified object.
        """
        return isinstance( self, tuple( node.node_for ) )
 
    #---------------------------------------------------------------------------
    #  Returns whether a given 'add_object' can be added to an object:
    #---------------------------------------------------------------------------
                                
    def tno_can_add ( self, node, add_object ):
        """ Returns whether a given object is droppable on the node.
        """
        if isinstance( add_object, tuple( node.add ) ):
            return True
        for item in node.move:
            if type( item ) in SequenceTypes:
                item = item[0]
            if isinstance( add_object, item ):
                return True
        return False
        
    #---------------------------------------------------------------------------
    #  Returns the list of classes that can be added to the object:
    #---------------------------------------------------------------------------
    
    def tno_get_add ( self, node ):
        """ Returns the list of classes that can be added to the object.
        """
        return node.add
                           
    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:    
    #---------------------------------------------------------------------------
                                                      
    def tno_drop_object ( self, node, dropped_object ):
        """ Returns a droppable version of a specified object.
        """
        if isinstance( dropped_object, tuple( node.add ) ):
            return dropped_object
            
        for item in node.move:
            if type( item ) in SequenceTypes:
                if isinstance( dropped_object, item[0] ):
                    return item[1]( self, dropped_object )
            else:
                if isinstance( dropped_object, item ):
                    return dropped_object
        
    #---------------------------------------------------------------------------
    #  Handles an object being selected:    
    #---------------------------------------------------------------------------
                
    def tno_select ( self, node ):
        """ Handles an object being selected.
        """
        if node.on_select is not None:
            node.on_select( self )
            return None
        return True
        
    #---------------------------------------------------------------------------
    #  Handles an object being double-clicked:
    #---------------------------------------------------------------------------
       
    def tno_dclick ( self, node ):
        """ Handles an object being double-clicked.
        """
        if node.on_dclick is not None:
            node.on_dclick( self )
            return None
        return True
    

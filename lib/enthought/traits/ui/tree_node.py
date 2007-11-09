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
# Date: 12/03/2004
#
#  Symbols defined: TreeNode
#
#------------------------------------------------------------------------------
""" Defines the tree node descriptor used by the tree editor and tree editor
factory classes.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import copy

from enthought.traits.api \
    import HasTraits, HasPrivateTraits, Str, List, Callable, Instance, Any, \
           true, false

from enthought.traits.trait_base \
    import SequenceTypes

from enthought.traits.ui.view \
    import View

#-------------------------------------------------------------------------------
#  'TreeNode' class:
#-------------------------------------------------------------------------------

class TreeNode ( HasPrivateTraits ):
    """ Represents a tree node. Used by the tree editor and tree editor factory
    classes.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Name of trait containing children ('' = Leaf)
    children = Str

    # Constant or name of a trait containing the label to use; constants
    # are preceded by '=', for example: label = '=A Label'.
    label = Str

    # Name to use for a new instance
    name = Str

    # Can the object's children be renamed?
    rename = true

    # Can the object be renamed?
    rename_me = true

    # Can the object's children be copied?
    copy = true

    # Can the object's children be deleted?
    delete = true

    # Can the object be deleted (if its parent allows it)?
    delete_me = true

    # Can children be inserted (vs. appended)?
    insert = true

    # Should tree nodes be automatically opened (expanded)?
    auto_open = false

    # Automatically close sibling tree nodes?
    auto_close = false

    # List of object classes than can be created or copied
    add = List( Any )

    # List of object classes that can be moved
    move = List( Any )

    # List of object classes that the node applies to
    node_for = List( Any )

    # Function for formatting the label
    formatter = Callable

    # Function for handling selecting an object
    on_select = Callable

    # Function for handling double-clicking an object
    on_dclick = Callable

    # View to use for editing the object
    view = Instance( View )

    # Right-click context menu
    menu = Any

    # Name of leaf item icon
    icon_item = Str( '<item>' )

    # Name of group item icon
    icon_group = Str( '<group>' )

    # Name of opened group item icon
    icon_open = Str( '<open>' )

    # Resource path used to locate the node icon
    icon_path = Str

    # fixme: The 'menu' trait should really be defined as:
    #        Instance( 'enthought.traits.ui.menu.MenuBar' ), but it doesn't work
    #        right currently.

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        super( TreeNode, self ).__init__( **traits )
        if self.icon_path == '':
            # FIXME: The following try...except was done in an attempt to avoid
            # having to declare traits as being dependent on enthought.resource.
            # There might be a better way to isolate them.
            try:
                from enthought.resource.api import resource_path
                self.icon_path = resource_path()
            except ImportError:
                pass


#---- Overridable methods: -----------------------------------------------------

    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:
    #---------------------------------------------------------------------------

    def allows_children ( self, object ):
        """ Returns whether this object can have children.
        """
        return (self.children != '')

    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:
    #---------------------------------------------------------------------------

    def has_children ( self, object ):
        """ Returns whether the object has children.
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
    #  Confirms that a specified object can be deleted or not:
    #  Result = True:  Delete object with no further prompting
    #         = False: Do not delete object
    #         = other: Take default action (may prompt user to confirm delete)
    #---------------------------------------------------------------------------

    def confirm_delete ( self, object ):
        """ Checks whether a specified object can be deleted.

        Returns
        -------
        * **True** if the object should be deleted with no further prompting.
        * **False** if the object should not be deleted.
        * Anything else: Caller should take its default action (which might
          include prompting the user to confirm deletion).
        """
        return None

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
        """ Sets up or removes a listener for children being replaced on a
        specified object.
        """
        object.on_trait_change( listener, self.children, remove = remove,
                                dispatch = 'ui' )

    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified
    #  object:
    #---------------------------------------------------------------------------

    def when_children_changed ( self, object, listener, remove ):
        """ Sets up or removes a listener for children being changed on a
        specified object.
        """
        object.on_trait_change( listener, self.children + '_items',
                                remove = remove, dispatch = 'ui' )

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
        """ Sets up or removes a listener for the label being changed on a
        specified object.
        """
        label = self.label
        if label[:1] != '=':
            object.on_trait_change( listener, label, remove = remove,
                                    dispatch = 'ui' )

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
            (displayed in the "New" submenu).
        """
        return self.name

    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:
    #---------------------------------------------------------------------------

    def get_view ( self, object ):
        """ Gets the view to use when editing an object.
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
        """ Returns whether the object's children can be renamed.
        """
        return self.rename

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be renamed:
    #---------------------------------------------------------------------------

    def can_rename_me ( self, object ):
        """ Returns whether the object can be renamed.
        """
        return self.rename_me

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------

    def can_copy ( self, object ):
        """ Returns whether the object's children can be copied.
        """
        return self.copy

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------

    def can_delete ( self, object ):
        """ Returns whether the object's children can be deleted.
        """
        return self.delete

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be deleted:
    #---------------------------------------------------------------------------

    def can_delete_me ( self, object ):
        """ Returns whether the object can be deleted.
        """
        return self.delete_me

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------

    def can_insert ( self, object ):
        """ Returns whether the object's children can be inserted (vs.
        appended).
        """
        return self.insert

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def can_auto_open ( self, object ):
        """ Returns whether the object's children should be automatically
        opened.
        """
        return self.auto_open

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def can_auto_close ( self, object ):
        """ Returns whether the object's children should be automatically
        closed.
        """
        return self.auto_close

    #---------------------------------------------------------------------------
    #  Returns whether or not this is the node that should handle a specified
    #  object:
    #---------------------------------------------------------------------------

    def is_node_for ( self, object ):
        """ Returns whether this is the node that handles a specified object.
        """
        return isinstance( object, tuple( self.node_for ) )

    #---------------------------------------------------------------------------
    #  Returns whether a given 'add_object' can be added to an object:
    #---------------------------------------------------------------------------

    def can_add ( self, object, add_object ):
        """ Returns whether a given object is droppable on the node.
        """
        klass = self._class_for( add_object )
        if self.is_addable( klass ):
            return True

        for item in self.move:
            if type( item ) in SequenceTypes:
                item = item[0]
            if issubclass( klass, item ):
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
    #  Returns the 'draggable' version of a specified object:
    #---------------------------------------------------------------------------

    def get_drag_object ( self, object ):
        """ Returns a draggable version of a specified object.
        """
        return object

    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:
    #---------------------------------------------------------------------------

    def drop_object ( self, object, dropped_object ):
        """ Returns a droppable version of a specified object.
        """
        klass = self._class_for( dropped_object )
        if self.is_addable( klass ):
            return dropped_object

        for item in self.move:
            if type( item ) in SequenceTypes:
                if issubclass( klass, item[0] ):
                    return item[1]( object, dropped_object )
            elif issubclass( klass, item ):
                return dropped_object

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

    #---------------------------------------------------------------------------
    #  Returns whether or not a specified object class can be added to the node:
    #---------------------------------------------------------------------------

    def is_addable ( self, klass ):
        """ Returns whether a specified object class can be added to the node.
        """
        for item in self.add:
            if type( item ) in SequenceTypes:
                item = item[0]
            if issubclass( klass, item ):
                return True

        return False

    #---------------------------------------------------------------------------
    #  Returns the class of an object:
    #---------------------------------------------------------------------------

    def _class_for ( self, object ):
        """ Returns the class of an object.
        """
        if isinstance( object, type ):
            return object

        return object.__class__

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
        """ Returns a droppable version of a specified object.
        """
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
        """ Returns whether this object can have children.
        """
        return object.tno_allows_children( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:
    #---------------------------------------------------------------------------

    def has_children ( self, object ):
        """ Returns whether the object has children.
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
    #  Confirms that a specified object can be deleted or not:
    #  Result = True:  Delete object with no further prompting
    #         = False: Do not delete object
    #         = other: Take default action (may prompt user to confirm delete)
    #---------------------------------------------------------------------------

    def confirm_delete ( self, object ):
        """ Checks whether a specified object can be deleted.

        Returns
        -------
        * **True** if the object should be deleted with no further prompting.
        * **False** if the object should not be deleted.
        * Anything else: Caller should take its default action (which might
          include prompting the user to confirm deletion).
        """
        return object.tno_confirm_delete( self )

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
        """ Sets up or removes a listener for children being replaced on a
        specified object.
        """
        return object.tno_when_children_replaced( self, listener, remove )

    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified
    #  object:
    #---------------------------------------------------------------------------

    def when_children_changed ( self, object, listener, remove ):
        """ Sets up or removes a listener for children being changed on a
        specified object.
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
        """ Sets up or removes a listener for the label being changed on a
        specified object.
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
            (displayed in the "New" submenu).
        """
        return object.tno_get_name( self )

    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:
    #---------------------------------------------------------------------------

    def get_view ( self, object ):
        """ Gets the view to use when editing an object.
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
        """ Returns whether the object's children can be renamed.
        """
        return object.tno_can_rename( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be renamed:
    #---------------------------------------------------------------------------

    def can_rename_me ( self, object ):
        """ Returns whether the object can be renamed.
        """
        return object.tno_can_rename_me( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------

    def can_copy ( self, object ):
        """ Returns whether the object's children can be copied.
        """
        return object.tno_can_copy( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------

    def can_delete ( self, object ):
        """ Returns whether the object's children can be deleted.
        """
        return object.tno_can_delete( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be deleted:
    #---------------------------------------------------------------------------

    def can_delete_me ( self, object ):
        """ Returns whether the object can be deleted.
        """
        return object.tno_can_delete_me( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------

    def can_insert ( self, object ):
        """ Returns whether the object's children can be inserted (vs.
        appended).
        """
        return object.tno_can_insert( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def can_auto_open ( self, object ):
        """ Returns whether the object's children should be automatically
        opened.
        """
        return object.tno_can_auto_open( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def can_auto_close ( self, object ):
        """ Returns whether the object's children should be automatically
        closed.
        """
        return object.tno_can_auto_close( self )

    #---------------------------------------------------------------------------
    #  Returns whether or not this is the node that should handle a specified
    #  object:
    #---------------------------------------------------------------------------

    def is_node_for ( self, object ):
        """ Returns whether this is the node that should handle a
            specified object.
        """
        if isinstance( object, TreeNodeObject ):
            return object.tno_is_node_for( self )
        return False

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
    #  Returns the 'draggable' version of a specified object:
    #---------------------------------------------------------------------------

    def get_drag_object ( self, object ):
        """ Returns a draggable version of a specified object.
        """
        return object.tno_get_drag_object( self )

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

class TreeNodeObject ( HasPrivateTraits ):
    """ Represents the object that corresponds to a tree node.
    """
    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:
    #---------------------------------------------------------------------------

    def tno_allows_children ( self, node ):
        """ Returns whether this object allows children.
        """
        return (node.children != '')

    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether this object has children.
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
    #  Confirms that a specified object can be deleted or not:
    #  Result = True:  Delete object with no further prompting
    #         = False: Do not delete object
    #         = other: Take default action (may prompt user to confirm delete)
    #---------------------------------------------------------------------------

    def tno_confirm_delete ( self, node ):
        """ Checks whether a specified object can be deleted.

        Returns
        -------
        * **True** if the object should be deleted with no further prompting.
        * **False** if the object should not be deleted.
        * Anything else: Caller should take its default action (which might
          include prompting the user to confirm deletion).
        """
        return None

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
        """ Sets up or removes a listener for children being replaced on a
        specified object.
        """
        self.on_trait_change( listener, node.children, remove = remove,
                              dispatch = 'ui' )

    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified
    #  object:
    #---------------------------------------------------------------------------

    def tno_when_children_changed ( self, node, listener, remove ):
        """ Sets up or removes a listener for children being changed on a
        specified object.
        """
        self.on_trait_change( listener, node.children + '_items',
                              remove = remove, dispatch = 'ui' )

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
        """ Sets up or removes a listener for  the label being changed on a
        specified object.
        """
        label = node.label
        if label[:1] != '=':
            self.on_trait_change( listener, label, remove = remove,
                                  dispatch = 'ui' )

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
            (displayed in the "New" submenu).
        """
        return node.name

    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:
    #---------------------------------------------------------------------------

    def tno_get_view ( self, node ):
        """ Gets the view to use when editing an object.
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
        """ Returns whether the object's children can be renamed.
        """
        return node.rename

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be renamed:
    #---------------------------------------------------------------------------

    def tno_can_rename_me ( self, node ):
        """ Returns whether the object can be renamed.
        """
        return node.rename_me

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------

    def tno_can_copy ( self, node ):
        """ Returns whether the object's children can be copied.
        """
        return node.copy

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------

    def tno_can_delete ( self, node ):
        """ Returns whether the object's children can be deleted.
        """
        return node.delete

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be deleted:
    #---------------------------------------------------------------------------

    def tno_can_delete_me ( self, node ):
        """ Returns whether the object can be deleted.
        """
        return node.delete_me

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------

    def tno_can_insert ( self, node ):
        """ Returns whether the object's children can be inserted (vs.
        appended).
        """
        return node.insert

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def tno_can_auto_open ( self, node ):
        """ Returns whether the object's children should be automatically
        opened.
        """
        return node.auto_open

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def tno_can_auto_close ( self, node ):
        """ Returns whether the object's children should be automatically
        closed.
        """
        return node.auto_close

    #---------------------------------------------------------------------------
    #  Returns whether or not this is the node that should handle a specified
    #  object:
    #---------------------------------------------------------------------------

    def tno_is_node_for ( self, node ):
        """ Returns whether this is the node that should handle a
            specified object.
        """
        return isinstance( self, tuple( node.node_for ) )

    #---------------------------------------------------------------------------
    #  Returns whether a given 'add_object' can be added to an object:
    #---------------------------------------------------------------------------

    def tno_can_add ( self, node, add_object ):
        """ Returns whether a given object is droppable on the node.
        """
        klass = node._class_for( add_object )
        if node.is_addable( klass ):
            return True

        for item in node.move:
            if type( item ) in SequenceTypes:
                item = item[0]
            if issubclass( klass, item ):
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
    #  Returns the 'draggable' version of a specified object:
    #---------------------------------------------------------------------------

    def tno_get_drag_object ( self, node ):
        """ Returns a draggable version of a specified object.
        """
        return self

    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:
    #---------------------------------------------------------------------------

    def tno_drop_object ( self, node, dropped_object ):
        """ Returns a droppable version of a specified object.
        """
        if node.is_addable( dropped_object ):
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

#-------------------------------------------------------------------------------
#  'MultiTreeNode' object:
#-------------------------------------------------------------------------------

class MultiTreeNode ( TreeNode ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # TreeNode that applies to the base object itself
    root_node = Instance( TreeNode )

    # List of TreeNodes (one for each sub-item list)
    nodes     = List( TreeNode )

    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:
    #---------------------------------------------------------------------------

    def allows_children ( self, object ):
        """ Returns whether this object can have children (True for this
        class).
        """
        return True

    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:
    #---------------------------------------------------------------------------

    def has_children ( self, object ):
        """ Returns whether this object has children (True for this class).
        """
        return True

    #---------------------------------------------------------------------------
    #  Gets the object's children:
    #---------------------------------------------------------------------------

    def get_children ( self, object ):
        """ Gets the object's children.
        """
        return [ ( object, node ) for node in self.nodes ]

    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children replaced' on a specified
    #  object:
    #---------------------------------------------------------------------------

    def when_children_replaced ( self, object, listener, remove ):
        """ Sets up or removes a listener for children being replaced on a
        specified object.
        """
        pass

    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified
    #  object:
    #---------------------------------------------------------------------------

    def when_children_changed ( self, object, listener, remove ):
        """ Sets up or removes a listener for children being changed on a
        specified object.
        """
        pass

    #---------------------------------------------------------------------------
    #  Gets the label to display for a specified object:
    #---------------------------------------------------------------------------

    def get_label ( self, object ):
        """ Gets the label to display for a specified object.
        """
        return self.root_node.get_label( object )

    #---------------------------------------------------------------------------
    #  Sets the label for a specified object:
    #---------------------------------------------------------------------------

    def set_label ( self, object, label ):
        """ Sets the label for a specified object.
        """
        return self.root_node.set_label( object, label )

    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'label changed' on a specified object:
    #---------------------------------------------------------------------------

    def when_label_changed ( self, object, listener, remove ):
        """ Sets up or removes a listener for the label being changed on a
        specified object.
        """
        return self.root_node.when_label_changed( object, listener, remove )

    #---------------------------------------------------------------------------
    #  Returns the icon for a specified object:
    #---------------------------------------------------------------------------

    def get_icon ( self, object, is_expanded ):
        """ Returns the icon for a specified object.
        """
        return self.root_node.get_icon( object, is_expanded )

    #---------------------------------------------------------------------------
    #  Returns the path used to locate an object's icon:
    #---------------------------------------------------------------------------

    def get_icon_path ( self, object ):
        """ Returns the path used to locate an object's icon.
        """
        return self.root_node.get_icon_path( object )

    #---------------------------------------------------------------------------
    #  Returns the name to use when adding a new object instance (displayed in
    #  the 'New' submenu):
    #---------------------------------------------------------------------------

    def get_name ( self, object ):
        """ Returns the name to use when adding a new object instance
            (displayed in the "New" submenu).
        """
        return self.root_node.get_name( object )

    #---------------------------------------------------------------------------
    #  Gets the View to use when editing an object:
    #---------------------------------------------------------------------------

    def get_view ( self, object ):
        """ Gets the view to use when editing an object.
        """
        return self.root_node.get_view( object )

    #---------------------------------------------------------------------------
    #  Returns the right-click context menu for an object:
    #---------------------------------------------------------------------------

    def get_menu ( self, object ):
        """ Returns the right-click context menu for an object.
        """
        return self.root_node.get_menu( object )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be renamed:
    #---------------------------------------------------------------------------

    def can_rename ( self, object ):
        """ Returns whether the object's children can be renamed (False for
        this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be renamed:
    #---------------------------------------------------------------------------

    def can_rename_me ( self, object ):
        """ Returns whether the object can be renamed (False for this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------

    def can_copy ( self, object ):
        """ Returns whether the object's children can be copied.
        """
        return self.root_node.can_copy( object )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------

    def can_delete ( self, object ):
        """ Returns whether the object's children can be deleted (False for
        this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object can be deleted:
    #---------------------------------------------------------------------------

    def can_delete_me ( self, object ):
        """ Returns whether the object can be deleted (True for this class).
        """
        return True

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------

    def can_insert ( self, object ):
        """ Returns whether the object's children can be inserted (False,
        meaning that children are appended, for this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-opened:
    #---------------------------------------------------------------------------

    def can_auto_open ( self, object ):
        """ Returns whether the object's children should be automatically
        opened.
        """
        return self.root_node.can_auto_open( object )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children should be auto-closed:
    #---------------------------------------------------------------------------

    def can_auto_close ( self, object ):
        """ Returns whether the object's children should be automatically
        closed.
        """
        return self.root_node.can_auto_close( object )

    #---------------------------------------------------------------------------
    #  Returns whether a given 'add_object' can be added to an object:
    #---------------------------------------------------------------------------

    def can_add ( self, object, add_object ):
        """ Returns whether a given object is droppable on the node (False for
        this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns the list of classes that can be added to the object:
    #---------------------------------------------------------------------------

    def get_add ( self, object ):
        """ Returns the list of classes that can be added to the object.
        """
        return []

    #-------------------------------------------------------------------------------
    #  Returns the 'draggable' version of a specified object:
    #-------------------------------------------------------------------------------

    def get_drag_object ( self, object ):
        """ Returns a draggable version of a specified object.
        """
        return self.root_node.get_drag_object( object )

    #---------------------------------------------------------------------------
    #  Returns a droppable version of a specified object:
    #---------------------------------------------------------------------------

    def drop_object ( self, object, dropped_object ):
        """ Returns a droppable version of a specified object.
        """
        return self.root_node.drop_object( object, dropped_object )

    #---------------------------------------------------------------------------
    #  Handles an object being selected:
    #---------------------------------------------------------------------------

    def select ( self, object ):
        """ Handles an object being selected.
        """
        return self.root_node.select( object )

    #---------------------------------------------------------------------------
    #  Handles an object being double-clicked:
    #---------------------------------------------------------------------------

    def dclick ( self, object ):
        """ Handles an object being double-clicked.
        """
        return self.root_node.dclick( object )


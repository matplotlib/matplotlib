#------------------------------------------------------------------------------
#
#  Copyright (c) 2006, Enthought, Inc.
#  All rights reserved.
#  
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#  Thanks for using Enthought open source!
#  
#  Author: David C. Morrill
#
#  Date: 01/05/2006
#
# 
# 
#------------------------------------------------------------------------------
""" Defines tree node classes and editors for various types of values.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from numpy import array
    
from enthought.traits.api \
    import HasTraits, HasPrivateTraits, Instance, List, Any, Str, false
    
from enthought.traits.ui.api \
    import View, Item, TreeEditor, TreeNode, TreeNodeObject, ObjectTreeNode
    
#-------------------------------------------------------------------------------
#  'SingleValueTreeNodeObject' class:  
#-------------------------------------------------------------------------------

class SingleValueTreeNodeObject ( TreeNodeObject ):
    """ A tree node for objects of types that have a single value.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    # The parent of this node    
    parent = Instance( TreeNodeObject )
    
    # Name of the value
    name = Str
    
    # User-specified override of the default label
    label = Str
    
    # The value itself
    value = Any
    
    # Is the value readonly?
    readonly = false

    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:  
    #---------------------------------------------------------------------------

    def tno_allows_children ( self, node ):
        """ Returns whether this object can have children (False for this 
        class).
        """
        return False
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether the object has children (False for this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be renamed:
    #---------------------------------------------------------------------------
                
    def tno_can_rename ( self, node ):
        """ Returns whether the object's children can be renamed (False for 
        this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be copied:
    #---------------------------------------------------------------------------
    
    def tno_can_copy ( self, node ):
        """ Returns whether the object's children can be copied (True for this
        class).
        """
        return True

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------
    
    def tno_can_delete ( self, node ):
        """ Returns whether the object's children can be deleted (False for 
        this class).
        """
        return False

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------
    
    def tno_can_insert ( self, node ):
        """ Returns whether the object's children can be inserted (False, 
        meaning children are appended, for this class).
        """
        return False
        
    #---------------------------------------------------------------------------
    #  Returns the icon for a specified object:  
    #---------------------------------------------------------------------------
                
    def tno_get_icon ( self, node, is_expanded ):
        """ Returns the icon for a specified object.
        """
        return self.__class__.__name__[ : -4 ].lower() + '_node'
        
    #---------------------------------------------------------------------------
    #  Sets the label for a specified node:  
    #---------------------------------------------------------------------------
                
    def tno_set_label ( self, node, label ):        
        """ Sets the label for a specified object.
        """
        if label == '?':
            label = ''
        self.label = label
    
    #---------------------------------------------------------------------------
    #  Gets the label to display for a specified object:    
    #---------------------------------------------------------------------------
        
    def tno_get_label ( self, node ):
        """ Gets the label to display for a specified object.
        """
        if self.label != '':
            return self.label
            
        if self.name == '':
            return self.format_value( self.value )
            
        return '%s: %s' % ( self.name, self.format_value( self.value ) )
    
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        return repr( value )
        
    #---------------------------------------------------------------------------
    #  Returns the correct node type for a specified value:  
    #---------------------------------------------------------------------------
                
    def node_for ( self, name, value ): 
        """ Returns the correct node type for a specified value.
        """
        for type, node in BasicTypes:
            if isinstance( value, type ):
                break
        else:
            node = OtherNode
            if hasattr( value, '__class__' ):
                node = ObjectNode
            
        return node( parent   = self, 
                     name     = name,
                     value    = value,
                     readonly = self.readonly )

#-------------------------------------------------------------------------------
#  'MultiValueTreeNodeObject' class:  
#-------------------------------------------------------------------------------

class MultiValueTreeNodeObject ( SingleValueTreeNodeObject ):
    """ A tree node for objects of types that have multiple values.
    """
    #---------------------------------------------------------------------------
    #  Returns whether chidren of this object are allowed or not:  
    #---------------------------------------------------------------------------

    def tno_allows_children ( self, node ):
        """ Returns whether this object can have children (True for this class).
        """
        return True
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether the object has children (True for this class).
        """
        return True
        
#-------------------------------------------------------------------------------
#  'StringNode' class:  
#-------------------------------------------------------------------------------
                
class StringNode ( SingleValueTreeNodeObject ):
    """ A tree node for strings.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        n = len( value )
        if len( value ) > 80:
            value = '%s...%s' % ( value[ :42 ], value[ -35: ] )
            
        return '%s [%d]' % ( repr( value ), n )
        
#-------------------------------------------------------------------------------
#  'NoneNode' class:  
#-------------------------------------------------------------------------------
                
class NoneNode ( SingleValueTreeNodeObject ):
    """ A tree node for None values.
    """
    pass
        
#-------------------------------------------------------------------------------
#  'BoolNode' class:  
#-------------------------------------------------------------------------------
                
class BoolNode ( SingleValueTreeNodeObject ):
    """ A tree node for Boolean values.
    """
    pass
        
#-------------------------------------------------------------------------------
#  'IntNode' class:  
#-------------------------------------------------------------------------------
                
class IntNode ( SingleValueTreeNodeObject ):
    """ A tree node for integer values.
    """
    pass
        
#-------------------------------------------------------------------------------
#  'FloatNode' class:  
#-------------------------------------------------------------------------------
                
class FloatNode ( SingleValueTreeNodeObject ):
    """ A tree node for floating point values.
    """
    pass
        
#-------------------------------------------------------------------------------
#  'ComplexNode' class:  
#-------------------------------------------------------------------------------
                
class ComplexNode ( SingleValueTreeNodeObject ):
    """ A tree node for complex number values.
    """
    pass
    
#-------------------------------------------------------------------------------
#  'OtherNode' class:  
#-------------------------------------------------------------------------------
                
class OtherNode ( SingleValueTreeNodeObject ):
    """ A tree node for single-value types for which there is not another
    node type.
    """
    pass
    
#-------------------------------------------------------------------------------
#  'TupleNode' class:  
#-------------------------------------------------------------------------------
        
class TupleNode ( MultiValueTreeNodeObject ):
    """ A tree node for tuples.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        return 'Tuple(%d)' % len( value )    
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether the object has children, based on the length of 
        the tuple.
        """
        return (len( self.value ) > 0)
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def tno_get_children ( self, node ):
        """ Gets the object's children.
        """
        node_for = self.node_for
        value    = self.value
        if len( value ) > 500:
            return ([ node_for( '[%d]' % i, x )
                      for i, x in enumerate( value[ : 250 ] ) ] +
                    [ StringNode( value = '...', readonly = True ) ] +   
                    [ node_for( '[%d]' % i, x )
                      for i, x in enumerate( value[ -250: ] ) ])
        
        return [ node_for( '[%d]' % i, x ) for i, x in enumerate( value ) ]
    
#-------------------------------------------------------------------------------
#  'ListNode' class:  
#-------------------------------------------------------------------------------
        
class ListNode ( TupleNode ):
    """ A tree node for lists.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        return 'List(%d)' % len( value )

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------
    
    def tno_can_delete ( self, node ):
        """ Returns whether the object's children can be deleted.
        """
        return (not self.readonly)

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be inserted (or just
    #  appended):
    #---------------------------------------------------------------------------
    
    def tno_can_insert ( self, node ):
        """ Returns whether the object's children can be inserted (vs. 
        appended).
        """
        return (not self.readonly)
        
#-------------------------------------------------------------------------------
#  'ArrayNode' class:  
#-------------------------------------------------------------------------------
        
class ArrayNode ( TupleNode ):
    """ A tree node for arrays.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        return 'Array(%s)' % ','.join( [ str( n ) for n in value.shape ] )
    
#-------------------------------------------------------------------------------
#  'DictNode' class:  
#-------------------------------------------------------------------------------
        
class DictNode ( TupleNode ):
    """ A tree node for dictionaries.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        return 'Dict(%d)' % len( value ) 
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def tno_get_children ( self, node ):
        """ Gets the object's children.
        """
        node_for = self.node_for
        items    = [ ( repr( k ), v ) for k, v in self.value.items() ]
        items.sort( lambda l, r: cmp( l[0], r[0] ) )
        if len( items ) > 500:
            return ([ node_for( '[%s]' % k, v ) for k, v in items[: 250 ] ] +
                    [ StringNode( value = '...', readonly = True ) ]        +
                    [ node_for( '[%s]' % k, v ) for k, v in items[ -250: ] ])
            
        return [ node_for( '[%s]' % k, v ) for k, v in items ]

    #---------------------------------------------------------------------------
    #  Returns whether or not the object's children can be deleted:
    #---------------------------------------------------------------------------
    
    def tno_can_delete ( self, node ):
        """ Returns whether the object's children can be deleted.
        """
        return (not self.readonly)
    
#-------------------------------------------------------------------------------
#  'ObjectNode' class:  
#-------------------------------------------------------------------------------
        
class ObjectNode ( MultiValueTreeNodeObject ):
    """ A tree node for objects.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        try:
            klass = value.__class__.__name__
        except:
            klass = '???'
        return '%s(0x%08X)' % ( klass, id( value ) )
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether the object has children.
        """
        try:
            return (len( self.value.__dict__ ) > 0)
        except:
            return False
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def tno_get_children ( self, node ):
        """ Gets the object's children.
        """
        items = [ ( k, v ) for k, v in self.value.__dict__.items() ]
        items.sort( lambda l, r: cmp( l[0], r[0] ) )
        return [ self.node_for( '.' + k, v ) for k, v in items ]
    
#-------------------------------------------------------------------------------
#  'TraitsNode' class:  
#-------------------------------------------------------------------------------
        
class TraitsNode ( ObjectNode ):
    """ A tree node for traits.
    """
    #---------------------------------------------------------------------------
    #  Returns whether or not the object has children:  
    #---------------------------------------------------------------------------

    def tno_has_children ( self, node ):
        """ Returns whether the object has children.
        """
        return (len( self._get_names() ) > 0)
    
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def tno_get_children ( self, node ):
        """ Gets the object's children.
        """
        names = self._get_names()
        names.sort()
        value    = self.value
        node_for = self.node_for
        nodes    = []
        for name in names:
            try:
                item_value = getattr( value, name, '<unknown>' )
            except Exception, excp:
                item_value = '<%s>' % excp 
            nodes.append( node_for( '.' + name, item_value ) )
        
        return nodes
                 
    #---------------------------------------------------------------------------
    #  Gets the names of all defined traits/attributes:  
    #---------------------------------------------------------------------------
                                  
    def _get_names ( self ):
        """ Gets the names of all defined traits or attributes.
        """
        value = self.value
        names = {}
        for name in value.trait_names( type = lambda x: x != 'event' ):
            names[ name ] = None
        for name in value.__dict__.keys(): 
            names[ name ] = None
        return names.keys()
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children replaced' on a specified  
    #  object:  
    #---------------------------------------------------------------------------

    def tno_when_children_replaced ( self, node, listener, remove ):
        """ Sets up or removes a listener for children being replaced on a 
        specified object.
        """
        self._listener = listener
        self.value.on_trait_change( self._children_replaced, remove = remove,
                                    dispatch = 'ui' )
        
    def _children_replaced ( self ):
        self._listener( self )
        
    #---------------------------------------------------------------------------
    #  Sets up/Tears down a listener for 'children changed' on a specified 
    #  object:  
    #---------------------------------------------------------------------------
                
    def tno_when_children_changed ( self, node, listener, remove ):
        """ Sets up or removes a listener for children being changed on a 
        specified object.
        """
        pass
    
#-------------------------------------------------------------------------------
#  'RootNode' class:  
#-------------------------------------------------------------------------------
        
class RootNode ( MultiValueTreeNodeObject ):
    """ A root node.
    """
    #---------------------------------------------------------------------------
    #  Returns the formatted version of the value:  
    #---------------------------------------------------------------------------
        
    def format_value ( self, value ):
        """ Returns the formatted version of the value.
        """
        return ''
        
    #---------------------------------------------------------------------------
    #  Gets the object's children:  
    #---------------------------------------------------------------------------

    def tno_get_children ( self, node ):
        """ Gets the object's children.
        """
        return [ self.node_for( '', self.value ) ]
    
#-------------------------------------------------------------------------------
#  Define the mapping of object types to nodes:
#-------------------------------------------------------------------------------

# The mapping of object types to nodes
BasicTypes = ( 
    ( type( None ),           NoneNode ),
    ( str,                    StringNode ),
    ( unicode,                StringNode ),
    ( bool,                   BoolNode ),
    ( int,                    IntNode ),
    ( float,                  FloatNode ),
    ( complex,                ComplexNode ),
    ( tuple,                  TupleNode ),
    ( list,                   ListNode ),
    ( dict,                   DictNode ),
    ( type( array( [ 1 ] ) ), ArrayNode ),
    ( HasTraits,              TraitsNode )
)    
    
#-------------------------------------------------------------------------------
#  '_ValueTree' class:  
#-------------------------------------------------------------------------------
        
class _ValueTree ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # List of arbitrary Python values contained in the tree:
    values = List( SingleValueTreeNodeObject )
    
#-------------------------------------------------------------------------------
#  Defines the value tree editor(s):  
#-------------------------------------------------------------------------------

# Nodes in a value tree
value_tree_nodes = [
    ObjectTreeNode( 
        node_for = [ NoneNode, StringNode, BoolNode, IntNode, FloatNode, 
                     ComplexNode, OtherNode, TupleNode, ListNode, ArrayNode,
                     DictNode, ObjectNode, TraitsNode, RootNode ] )
]

# Editor for a value tree
value_tree_editor = TreeEditor(
    auto_open = 3,
    hide_root = True,
    editable  = False,
    nodes     = value_tree_nodes
)

# Editor for a value tree with a root
value_tree_editor_with_root = TreeEditor(
    auto_open = 3,
    editable  = False,
    nodes     = [
        ObjectTreeNode( 
            node_for = [ NoneNode, StringNode, BoolNode, IntNode, FloatNode, 
                         ComplexNode, OtherNode, TupleNode, ListNode, ArrayNode,
                         DictNode, ObjectNode, TraitsNode, RootNode ] ),
        TreeNode( node_for = [ _ValueTree ],
                  auto_open  = True,
                  children   = 'values',
                  move       = [ SingleValueTreeNodeObject ],
                  copy       = False,
                  label      = '=Values',
                  icon_group = 'traits_node',
                  icon_open  = 'traits_node' ) 
    ]
)
    
#-------------------------------------------------------------------------------
#  Defines a 'ValueTree' trait:  
#-------------------------------------------------------------------------------
        
# Trait for a value tree
ValueTree = Instance( _ValueTree, (), editor = value_tree_editor_with_root )


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
# Date: 10/07/2004
#------------------------------------------------------------------------------
""" Manager for Undo and Redo history for Traits user interface support.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from operator \
    import isSequenceType

from enthought.traits.api \
    import HasStrictTraits, HasPrivateTraits, HasTraits, Trait, List, Int, Str,\
           Any, Event, Property, Instance, false

from enthought.traits.trait_base \
    import enumerate

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

NumericTypes = ( int, long, float, complex )
SimpleTypes  = ( str, unicode, int, long, float, complex )

#-------------------------------------------------------------------------------
#  'AbstractUndoItem' class:
#-------------------------------------------------------------------------------

class AbstractUndoItem ( HasPrivateTraits ):
    """ Abstract base class for undo items.
    """
    #---------------------------------------------------------------------------
    #  Undoes the change:
    #---------------------------------------------------------------------------

    def undo ( self ):
        """ Undoes the change.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Re-does the change:
    #---------------------------------------------------------------------------

    def redo ( self ):
        """ Re-does the change.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Merges two undo items if possible:
    #---------------------------------------------------------------------------

    def merge_undo ( self, undo_item ):
        """ Merges two undo items if possible.
        """
        return False

#-------------------------------------------------------------------------------
#  'UndoItem' class:
#-------------------------------------------------------------------------------

class UndoItem ( AbstractUndoItem ):
    """ A change to an object trait, which can be undone.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Object the change occurred on
    object    = Trait( HasTraits )  
    # Name of the trait that changed
    name      = Str                 
    # Old value of the changed trait
    old_value = Property            
    # New value of the changed trait
    new_value = Property            

    #---------------------------------------------------------------------------
    #  Implementation of the 'old_value' and 'new_value' properties:
    #---------------------------------------------------------------------------

    def _get_old_value ( self ):
        return self._old_value

    def _set_old_value ( self, value ):
        if isinstance( value, list ):
            value = value[:]
        self._old_value = value

    def _get_new_value ( self ):
        return self._new_value

    def _set_new_value ( self, value ):
        if isinstance( value, list ):
            value = value[:]
        self._new_value = value

    #---------------------------------------------------------------------------
    #  Undoes the change:
    #---------------------------------------------------------------------------

    def undo ( self ):
        """ Undoes the change.
        """
        try:
            setattr( self.object, self.name, self.old_value )
        except:
            pass

    #---------------------------------------------------------------------------
    #  Re-does the change:
    #---------------------------------------------------------------------------

    def redo ( self ):
        """ Re-does the change.
        """
        try:
            setattr( self.object, self.name, self.new_value )
        except:
            pass

    #---------------------------------------------------------------------------
    #  Merges two undo items if possible:
    #---------------------------------------------------------------------------

    def merge_undo ( self, undo_item ):
        """ Merges two undo items if possible.
        """
        # Undo items are potentially mergeable only if they are of the same
        # class and refer to the same object trait, so check that first:
        if (isinstance( undo_item, self.__class__ ) and
           (self.object is undo_item.object) and
           (self.name == undo_item.name)):
            v1 = self.new_value
            v2 = undo_item.new_value
            t1 = type( v1 )
            if t1 is type( v2 ):

                if isinstance(t1, basestring):
                    # Merge two undo items if they have new values which are
                    # strings which only differ by one character (corresponding
                    # to a single character insertion, deletion or replacement
                    # operation in a text editor):
                    n1 = len( v1 )
                    n2 = len( v2 )
                    n  = min( n1, n2 )
                    i  = 0
                    while (i < n) and (v1[i] == v2[i]):
                        i += 1
                    if v1[i + (n2 <= n1):] == v2[i + (n2 >= n1):]:
                        self.new_value = v2
                        return True

                elif isSequenceType( v1 ):
                    # Merge sequence types only if a single element has changed
                    # from the 'original' value, and the element type is a
                    # simple Python type:
                    v1 = self.old_value
                    if isSequenceType( v1 ):
                        # Note: wxColour says it's a sequence type, but it
                        # doesn't support 'len', so we handle the exception
                        # just in case other classes have similar behavior:
                        try:
                            if len( v1 ) == len( v2 ):
                                diffs = 0
                                for i, item in enumerate( v1 ):
                                    titem = type( item )
                                    item2 = v2[i]
                                    if ((titem not in SimpleTypes)   or
                                        (titem is not type( item2 )) or
                                        (item != item2)):
                                        diffs += 1
                                        if diffs >= 2:
                                            return False
                                if diffs == 0:
                                    return False
                                self.new_value = v2
                                return True
                        except:
                            pass

                elif t1 in NumericTypes:
                    # Always merge simple numeric trait changes:
                    self.new_value = v2
                    return True
        return False

    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' form of the object:
    #---------------------------------------------------------------------------

    def __repr__ ( self ):
        """ Returns a "pretty print" form of the object.
        """
        n  = self.name
        cn = self.object.__class__.__name__
        return 'undo( %s.%s = %s )\nredo( %s.%s = %s )' % (
                      cn, n, self.old_value, cn, n, self.new_value )

#-------------------------------------------------------------------------------
#  'ListUndoItem' class:
#-------------------------------------------------------------------------------

class ListUndoItem ( AbstractUndoItem ):
    """ A change to a list, which can be undone.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Object that the change occurred on
    object    = Trait( HasTraits )  
    # Name of the trait that changed
    name      = Str                 
    # Starting index
    index     = Int                 
    # Items added to the list
    added     = List                
    # Items removed from the list
    removed   = List                

    #---------------------------------------------------------------------------
    #  Undoes the change:
    #---------------------------------------------------------------------------

    def undo ( self ):
        """ Undoes the change.
        """
        try:
            list = getattr( self.object, self.name )
            list[ self.index: (self.index + len( self.added )) ] = self.removed
        except:
            pass

    #---------------------------------------------------------------------------
    #  Re-does the change:
    #---------------------------------------------------------------------------

    def redo ( self ):
        """ Re-does the change.
        """
        try:
            list = getattr( self.object, self.name )
            list[ self.index: (self.index + len( self.removed )) ] = self.added
        except:
            pass

    #---------------------------------------------------------------------------
    #  Merges two undo items if possible:
    #---------------------------------------------------------------------------

    def merge_undo ( self, undo_item ):
        """ Merges two undo items if possible.
        """
        # Discard undo items that are identical to us. This is to eliminate
        # the same undo item being created by multiple listeners monitoring the
        # same list for changes:
        if (isinstance( undo_item, self.__class__ )        and
           (self.object is undo_item.object)               and
           (self.name  == undo_item.name)                  and
           (self.index == undo_item.index)):
            added   = undo_item.added
            removed = undo_item.removed
            if ((len( self.added )   == len( added )) and
                (len( self.removed ) == len( removed ))):
                for i, item in enumerate( self.added ):
                    if item is not added[i]:
                        break
                else:
                    for i, item in enumerate( self.removed ):
                        if item is not removed[i]:
                            break
                    else:
                        return True
        return False

    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' form of the object:
    #---------------------------------------------------------------------------

    def __repr__ ( self ):
        """ Returns a 'pretty print' form of the object.
        """
        return 'undo( %s.%s[%d:%d] = %s )' % (
                self.object.__class__.__name__, self.name, self.index,
                self.index + len( self.removed ), self.added )

#-------------------------------------------------------------------------------
#  'UndoHistory' class:
#-------------------------------------------------------------------------------

class UndoHistory ( HasStrictTraits ):
    """ Manages a list of undoable changes.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # List of accumulated undo changes
    history  = List           
    # The current position in the list
    now      = Int            
    # Fired when state changes to undoable
    undoable = Event( False ) 
    # Fired when state changes to redoable
    redoable = Event( False ) 
    # Can an action be undone?
    can_undo = Property       
    # Can an action be redone?
    can_redo = Property       

    #---------------------------------------------------------------------------
    #  Adds an UndoItem to the history:
    #---------------------------------------------------------------------------

    def add ( self, undo_item, extend = False ):
        """ Adds an UndoItem to the history.
        """
        if extend:
            self.extend( undo_item )
            return

        # Try to merge the new undo item with the previous item if allowed:
        now = self.now
        if now > 0:
            previous = self.history[ now - 1 ]
            if (len( previous ) == 1) and previous[0].merge_undo( undo_item ):
                self.history[ now: ] = []
                return

        old_len = len( self.history )
        self.history[ now: ] = [ [ undo_item ] ]
        self.now += 1
        if self.now == 1:
            self.undoable = True
        if self.now <= old_len:
            self.redoable = False

    #---------------------------------------------------------------------------
    #  Extends the most recent 'undo' item:
    #---------------------------------------------------------------------------

    def extend ( self, undo_item ):
        """ Extends the undo history. 
        
        If possible the method merges the new UndoItem with the last item in 
        the history; otherwise, it appends the new item.
        """
        if self.now > 0:
            undo_list =  self.history[ self.now - 1 ]
            if not undo_list[-1].merge_undo( undo_item ):
                undo_list.append( undo_item )

    #---------------------------------------------------------------------------
    #  Undo an operation:
    #---------------------------------------------------------------------------

    def undo ( self ):
        """ Undoes an operation.
        """
        if self.can_undo:
            self.now -= 1
            items = self.history[ self.now ]
            for i in range( len( items ) - 1, -1, -1 ):
                items[i].undo()
            if self.now == 0:
                self.undoable = False
            if self.now == (len( self.history ) - 1):
                self.redoable = True

    #---------------------------------------------------------------------------
    #  Redo an operation:
    #---------------------------------------------------------------------------

    def redo ( self ):
        """ Redoes an operation.
        """
        if self.can_redo:
            self.now += 1
            for item in self.history[ self.now - 1 ]:
                item.redo()
            if self.now == 1:
                self.undoable = True
            if self.now == len( self.history ):
                self.redoable = False

    #---------------------------------------------------------------------------
    #  Reverts all changes made so far and clears the history:
    #---------------------------------------------------------------------------

    def revert ( self ):
        """ Reverts all changes made so far and clears the history.
        """
        history = self.history[ : self.now ]
        self.clear()
        for i in range( len( history ) - 1, -1, -1 ):
            items = history[i]
            for j in range( len( items ) - 1, -1, -1 ):
                items[j].undo()

    #---------------------------------------------------------------------------
    #  Clears the undo history
    #---------------------------------------------------------------------------

    def clear ( self ):
        """ Clears the undo history.
        """
        old_len  = len( self.history )
        old_now  = self.now
        self.now = 0
        del self.history[:]
        if old_now > 0:
            self.undoable = False
        if old_now < old_len:
            self.redoable = False

    #---------------------------------------------------------------------------
    #  Are there any undoable operations?
    #---------------------------------------------------------------------------

    def _get_can_undo ( self ):
        """ Are there any undoable operations?
        """
        return self.now > 0

    #---------------------------------------------------------------------------
    #  Are there any redoable operations?
    #---------------------------------------------------------------------------

    def _get_can_redo ( self ):
        """ Are there any redoable operations?
        """
        return self.now < len( self.history )

#-------------------------------------------------------------------------------
#  'UndoHistoryUndoItem' class:
#-------------------------------------------------------------------------------

class UndoHistoryUndoItem ( AbstractUndoItem ):
    """ An undo item for the undo history.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # The undo history to undo or redo
    history = Instance( UndoHistory )

    #---------------------------------------------------------------------------
    #  Undoes the change:
    #---------------------------------------------------------------------------

    def undo ( self ):
        """ Undoes the change.
        """
        history = self.history
        for i in range( history.now - 1, -1, -1 ):
            items = history.history[i]
            for j in range( len( items ) - 1, -1, -1 ):
                items[j].undo()

    #---------------------------------------------------------------------------
    #  Re-does the change:
    #---------------------------------------------------------------------------

    def redo ( self ):
        """ Re-does the change.
        """
        history = self.history
        for i in range( 0, history.now ):
            for item in history.history[i]:
                item.redo()


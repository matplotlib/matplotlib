#-------------------------------------------------------------------------------
#
#  Define the abstract Editor class used to represent an object trait editing 
#  control in a traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: Editor 
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits import Trait, HasPrivateTraits, ReadOnly, Any, Property, \
                             Undefined, true, false, TraitError
from editor_factory   import EditorFactory
from undo             import UndoItem

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Reference to an EditorFactory object:
factory_trait = Trait( EditorFactory )

#-------------------------------------------------------------------------------
#  'Editor' abstract base class:
#-------------------------------------------------------------------------------

class Editor ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    ui          = ReadOnly # The UI (user interface) this editor is part of
    object      = ReadOnly # The object this editor is editing
    name        = ReadOnly # The name of the trait this editor is editing
    old_value   = ReadOnly # Original value of object.name
    description = ReadOnly # Text description of the object trait being edited 
    control     = Any      # The GUI widget defined by this editor
    enabled     = true     # Whether or not the underlying GUI widget is enabled
    factory     = factory_trait # The EditorFactory used to create this editor
    updating    = false    # Is the editor updating the object.name value?
#   value       = Any      # Current value for object.name
#   str_value   = Str      # Current value of object trait as a string
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, parent, **traits ):
        """ Initializes the object.
        """
        HasPrivateTraits.__init__( self, **traits )
        try:
            self.old_value = getattr( self.object, self.name )
        except AttributeError:
            # Getting the attribute will fail for 'Event' traits:
            self.old_value = Undefined
        self.object.on_trait_change( self._update_editor, self.name )
        self.init( parent )
        self.update_editor()
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        raise NotImplementedError
        
    #---------------------------------------------------------------------------
    #  Disposes of the contents of an editor:    
    #---------------------------------------------------------------------------
                
    def dispose ( self ):
        """ Disposes of the contents of an editor.
        """
        self.object.on_trait_change( self._update_editor, self.name, 
                                     remove = True )
       
    #---------------------------------------------------------------------------
    #  Gets/Sets the associated object trait's value:
    #---------------------------------------------------------------------------
    
    def _get_value ( self ):
        return getattr( self.object, self.name )
        
    def _set_value ( self, value ):
        self.ui.do_undoable( self.__set_value, value )
        
    value = Property( _get_value, _set_value )
    
    def __set_value ( self, value ):    
        try:
            self.ui.handler.setattr( self.object, self.name, value )
        except TraitError, excp:
            self.error( excp )
            raise
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified object trait value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, value ):
        """ Returns the text representation of a specified object trait value.
        """
        return str( value )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of the object trait:
    #---------------------------------------------------------------------------
  
    def _str_value ( self ):
        """ Returns the text representation of the object trait.
        """
        return self.string_value( getattr( self.object, self.name ) )
        
    str_value = Property( _str_value )
  
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified value:
    #---------------------------------------------------------------------------
  
    def _str ( self, value ):
        """ Returns the text representation of a specified value.
        """
        return str( value )
        
    #---------------------------------------------------------------------------
    #  Handles an error that occurs while setting the object's trait value:
    #
    #  (Should normally be overridden in a subclass)
    #---------------------------------------------------------------------------
        
    def error ( self, excp ):
        """ Handles an error that occurs while setting the object's trait value.
        """
        pass
        
    #---------------------------------------------------------------------------
    #  Performs updates when the object trait changes:
    #---------------------------------------------------------------------------
        
    def _update_editor ( self, object, name, old_value, new_value ):
        """ Performs updates when the object trait changes.
        """
        # If the editor has gone away for some reason, disconnect and exit:
        if self.control is None:
            object.on_trait_change( self._update_editor, name, remove = True )
            return
            
        # Log the change that was made:
        self.log_change( self.get_undo_item, object, name, 
                                             old_value, new_value )
                    
        # Update the editor control to reflect the current object state:                    
        self.update_editor()
        
    #---------------------------------------------------------------------------
    #  Logs a change made in the editor:    
    #---------------------------------------------------------------------------
                
    def log_change ( self, undo_factory, *undo_args ): 
        """ Logs a change made in the editor.
        """
        # Indicate that the contents of the user interface have been changed:
        ui          = self.ui
        ui.modified = True
        
        # Create an undo history entry if we are maintaining a history:
        undoable = ui._undoable
        if undoable >= 0:
            history = ui.history
            if history is not Undefined:
                item = undo_factory( *undo_args )
                if item is not None:
                    if undoable == history.now: 
                        # Create a new undo transaction:
                        history.add( item )
                    else:
                        # Extend the most recent undo transaction:
                        history.extend( item )
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #
    #  (Should normally be overridden in a subclass)
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        pass
        
    #---------------------------------------------------------------------------
    #  Creates an undo history entry:   
    #
    #  (Can be overridden in a subclass for special value types)
    #---------------------------------------------------------------------------
           
    def get_undo_item ( self, object, name, old_value, new_value ):
        """ Creates an undo history entry.
        """
        return UndoItem( object    = object,
                         name      = name,
                         old_value = old_value,
                         new_value = new_value ) 

#-- UI preference save/restore interface ---------------------------------------

    #---------------------------------------------------------------------------
    #  Restores any saved user preference information associated with the 
    #  editor:
    #---------------------------------------------------------------------------
            
    def restore_prefs ( self, prefs ):
        """ Restores any saved user preference information associated with the 
            editor.
        """
        pass
            
    #---------------------------------------------------------------------------
    #  Returns any user preference information associated with the editor:
    #---------------------------------------------------------------------------
            
    def save_prefs ( self ):
        """ Returns any user preference information associated with the editor.
        """
        return None
        
#-- End UI preference save/restore interface -----------------------------------                         


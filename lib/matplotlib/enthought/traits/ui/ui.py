#-------------------------------------------------------------------------------
#
#  Define the UI class used to represent an active traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: UI
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import shelve
import os

from types            import FunctionType
from editor           import Editor
from view_elements    import ViewElements
from matplotlib.enthought.traits import Trait, HasPrivateTraits, ReadOnly, DictStrAny, \
                             Any, List, Int, TraitError, false, Property, \
                             Undefined
from toolkit          import toolkit
from ui_info          import UIInfo
from matplotlib.enthought.traits.trait_base import traits_home                             

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# ViewElements the View is associated with:
view_elements_trait = Trait( None, ViewElements,
     desc = 'the ViewElements collection this UI resolves Include items from' )

#-------------------------------------------------------------------------------
#  'UI' class:
#-------------------------------------------------------------------------------

class UI ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    view_elements = view_elements_trait # The ViewElements object this UI
                          # resolves Include items from
    context  = DictStrAny # Context objects the UI is editing     
    handler  = ReadOnly   # Handler object used for event handling
    view     = ReadOnly   # View template used to construct the UI 
    control  = Any        # Panel/Dialog associated with the UI
    owner    = Any        # Toolkit specific object which 'owns' 'control'
    info     = ReadOnly   # UIInfo object containing the context/editor objects
    result   = ReadOnly   # Result from a modal/wizard dialog
    history  = ReadOnly   # Undo/Redo history
    modified = false      # Have any modifications been made to the UI contents?
    
    # Private traits:
    _context      = DictStrAny # Original context when used with modal dialog
    _revert       = DictStrAny # Copy of original context used for 'revert'
    _defined      = List   # List of methods to call once user interface created
    _enabled      = List   # List of (enabled_when,Editor) pairs
    _checked      = List   # List of (checked_when,Editor) pairs
    _search       = List   # Search stack used while building a user interface
    _dispatchers  = List   # List of dispatchable Handler methods
    _editors      = List   # List of editors used to build the user interface
    _names        = List   # List of names bound to 'info' object
    _active_group = Int    # Index of currently active group in the ui
    _groups       = Property   # List of top-level groups used to build the ui
    _undoable     = Int( -1 )  # Undoable action nesting level count
    
    # List of 'disposable' traits:
    disposable_traits = [ 
        'control', 'history', '_context', '_revert', '_defined', '_enabled',     
        '_checked', '_search', '_dispatchers', '_editors', '_names',
        '_active_group', '_groups', '_undoable'
    ]
    
    #---------------------------------------------------------------------------
    #  Initializes the object: 
    #---------------------------------------------------------------------------
        
    def __init__ ( self, **traits ):
        """ Initializes the object.
        """
        super( UI, self ).__init__( **traits )
        self.info = UIInfo( ui = self )
    
    #---------------------------------------------------------------------------
    #  Creates a user interface from the associated View template object:
    #---------------------------------------------------------------------------
    
    def ui ( self, parent, kind ):
        """ Creates a user interface from the associated View template object.
        """
        getattr( toolkit(), 'ui_' + kind )( self, parent )
        
    #---------------------------------------------------------------------------
    #  Disposes of the contents of a user interface:    
    #---------------------------------------------------------------------------
                                
    def dispose ( self ):
        """ Disposes of the contents of a user interface.
        """
        # Save the user preference information for the user interface:
        toolkit().save_window( self )
        
        # Finish disposing of the user interface:
        self.finish()
        
    #---------------------------------------------------------------------------
    #  Finishes a user interface:    
    #---------------------------------------------------------------------------
                                
    def finish ( self, result = None ):
        """ Finishes a user interface.
        """
        if result is not None:
            self.result = result
            
        for editor in self._editors:
            editor.dispose()
            
        self.control.DestroyChildren()
        self.control.Destroy()
        
        dict = self.__dict__
        for name in self.disposable_traits:
            if name in dict:
                del dict[ name ]
                
        self.handler.closed( self.info, self.result )
        
    #---------------------------------------------------------------------------
    #  Find the definition of the specified Include object in the current user
    #  interface building context:
    #---------------------------------------------------------------------------
    
    def find ( self, include ):
        """ Find the definition of the specified Include object in the current 
            user interface building context.
        """
        # Try to use our ViewElements objects:
        ve = self.view_elements
        
        # If none specified, try to get it from the UI context:
        if ve is None:
            if len( self.context ) == 1:
                obj = self.context.values()[0]
            elif 'object' in self.context:
                obj = self.context[ 'object' ]
            else:
                # Couldn't find a context object to use, so give up:
                return None
                
            # Otherwise, use the context object's ViewElements:
            ve = obj.trait_view_elements()
            
        # Ask the ViewElements to find the requested item for us:
        return ve.find( include.id, self._search )
        
    #---------------------------------------------------------------------------
    #  Returns the current search stack level:  
    #---------------------------------------------------------------------------
                
    def push_level ( self ):
        """ Returns the current search stack level.
        """
        return len( self._search )

    #---------------------------------------------------------------------------
    #  Restores a previously pushed search stack level: 
    #---------------------------------------------------------------------------
    
    def pop_level ( self, level ):
        """ Restores a previously pushed search stack level.
        """
        del self._search[ : len( self._search ) - level ]        
        
    #---------------------------------------------------------------------------
    #  Performs all post user interface creation processing:
    #---------------------------------------------------------------------------
        
    def prepare_ui ( self ):
        """ Performs all post user interface creation processing.
        """
        # Invoke all of the editor 'name_defined' methods we've accumulated:
        info = self.info
        for method in self._defined:
            method( info )
            
        # Then reset the list, since we don't need it anymore:
        del self._defined[:]
        
        # Invoke the handler's 'init' method, and abort if it indicates failure:
        handler = self.handler
        if not handler.init( info ):
            self.result = False
            raise TraitError, 'User interface creation aborted'
            
        # For each Handler method whose name is of the form 
        # 'object_name_changed', where 'object' is the name of an object in the 
        # UI's 'context', create a trait notification handler that will call 
        # the method whenever 'object's 'name' trait changes. Also invoke the
        # method immediately so initial user interface state can be correctly 
        # set:
        context = self.context
        for name, method in self._each_method( handler ):
            if name[-8:] == '_changed':
                prefix = name[:-8]
                col    = prefix.find( '_', 1 )
                if col >= 0:
                    object = context.get( prefix[:col] )
                    if object is not None:
                        dispatcher = Dispatcher( getattr( handler, name ), 
                                                 info )
                        self._dispatchers.append( dispatcher )
                        object.on_trait_change( dispatcher.dispatch,
                                                prefix[ col + 1: ] )
                        method( handler, info )
                        
        # If there are any Editor object's whose 'enabled' or 'checked' state is 
        # controlled by an 'enabled_when' or 'checked_when' expression, set up 
        # an 'anytrait' changed notification handler on each object in the 
        # 'context' that will cause the 'enabled' or 'checked' state of each 
        # affected Editor to be set. Also trigger the evaluation immediately, so 
        # the enabled or checked state of each Editor can be correctly 
        # initialized:
        if (len( self._enabled ) + len( self._checked )) > 0:
            for object in context.values():
                object.on_trait_change( self._evaluate_when )
            self._evaluate_when()
            
    #---------------------------------------------------------------------------
    #  Restores any saved user preference information associated with the UI:
    #---------------------------------------------------------------------------
            
    def restore_prefs ( self ):
        """ Restores any saved user preference information associated with the 
            UI.
        """
        id = self.view.id
        if id != '':
            db = self._get_ui_db()
            if db is not None:
                ui_prefs = db.get( id )
                db.close()
                if isinstance( ui_prefs, dict ):
                    info = self.info
                    for name in self._names:
                        editor = getattr( info, name, None )
                        if isinstance( editor, Editor ):
                           prefs = ui_prefs.get( name )
                           if prefs != None:
                               editor.restore_prefs( prefs )
                    return ui_prefs.get( '' )
        return None
            
    #---------------------------------------------------------------------------
    #  Saves any user preference information associated with the UI:
    #---------------------------------------------------------------------------
            
    def save_prefs ( self, prefs = None ):
        """ Saves any user preference information associated with the UI.
        """
        id = self.view.id
        if id != '':
            db = self._get_ui_db( mode = 'c' )
            if db is not None:
                ui_prefs = {}
                if prefs is not None:
                    ui_prefs[''] = prefs 
                info = self.info
                for name in self._names:
                    editor = getattr( info, name, None )
                    if isinstance( editor, Editor ):
                       prefs = editor.save_prefs()
                       if prefs != None:
                           ui_prefs[ name ] = prefs
                db[ id ] = ui_prefs
                db.close()
                        
    #---------------------------------------------------------------------------
    #  Gets a reference to the traits UI preference database:
    #---------------------------------------------------------------------------
                        
    def _get_ui_db ( self, mode = 'r' ):
        try:
            return shelve.open( os.path.join( traits_home(), 'traits_ui' ), 
                                flag = mode, protocol = -1 )
        except:
            return None
        
    #---------------------------------------------------------------------------
    #  Adds a Handler method to the list of methods to be called once the user 
    #  interface has been constructed:
    #---------------------------------------------------------------------------
    
    def add_defined ( self, method ):
        """ Adds a Handler method to the list of methods to be called once the 
            user interface has been constructed.
        """
        self._defined.append( method )
        
    #---------------------------------------------------------------------------
    #  Add's a conditionally enabled Editor object to the list of monitored 
    #  'enabled_when' objects:
    #---------------------------------------------------------------------------
        
    def add_enabled ( self, enabled_when, editor ):        
        """ Add's a conditionally enabled Editor object to the list of monitored 
            'enabled_when' objects.
        """
        self._enabled.append( ( compile( enabled_when, '<string>', 'eval' ), 
                                editor ) )
        
    #---------------------------------------------------------------------------
    #  Add's a conditionally checked (menu/toolbar) Editor object to the list of 
    #  monitored 'checked_when' objects:
    #---------------------------------------------------------------------------
        
    def add_checked ( self, checked_when, editor ):        
        """ Add's a conditionally enabled (menu) Editor object to the list of 
            monitored 'checked_when' objects.
        """
        self._checked.append( ( compile( checked_when, '<string>', 'eval' ), 
                                editor ) )

    #---------------------------------------------------------------------------
    #  Performs an 'undoable' action:    
    #---------------------------------------------------------------------------
                                     
    def do_undoable ( self, action, *args, **kw ):
        undoable = self._undoable
        try:
            if (undoable == -1) and (self.history is not Undefined) and \
                   (self.history is not None):
                self._undoable = self.history.now
            action( *args, **kw )
        finally:
            if undoable == -1:
                self._undoable = -1
        
    #---------------------------------------------------------------------------
    #  Generates each (name, method) pair for a specified object:
    #---------------------------------------------------------------------------
        
    def _each_method ( self, object ):
        """ Generates each (name, method) pair for a specified object.
        """
        dic = {}
        for klass in object.__class__.__mro__:
            for name, method in klass.__dict__.items():
                if (type( method ) is FunctionType) and (name not in dic):
                    dic[ name ] = True
                    yield ( name, method )
                    
    #---------------------------------------------------------------------------
    #  Evaluates an expression in the UI's 'context' and returns the result:    
    #---------------------------------------------------------------------------
                                        
    def eval_when ( self, when, result = True ):
        """ Evaluates an expression in the UI's 'context' and returns the
            result.
        """
        context = self.context
        context[ 'ui' ] = self
        try:
            result = eval( when, globals(), context )
        except:
            # fixme: Should the exception be logged somewhere?
            pass
        del context[ 'ui' ]
        return result
        
    #---------------------------------------------------------------------------
    #  Sets the 'enabled' and/or 'checked' state for all Editors controlled by 
    #  an 'enabled_when' or 'checked_when' expression:
    #---------------------------------------------------------------------------
                
    def _evaluate_when ( self ):
        """ Sets the 'enabled' state for all Editors controlled by an 
            'enabled_when' expression.
        """
        self._evaluate_condition( self._enabled, 'enabled' )
        self._evaluate_condition( self._checked, 'checked' )
        
    #---------------------------------------------------------------------------
    #  Evaluates a list of ( eval, editor ) pairs and sets a specified trait on
    #  each editor to reflect the boolean truth of the expression evaluated:    
    #---------------------------------------------------------------------------
                
    def _evaluate_condition ( self, conditions, trait ):                
        context = self.context
        context[ 'ui' ] = self
        for when, editor in conditions:
            value = True
            try:
                if not eval( when, globals(), context ):
                    value = False
            except:
                # fixme: Should the exception be logged somewhere?
                pass
            setattr( editor, trait, value )
        del context[ 'ui' ]
        
    #---------------------------------------------------------------------------
    #  Returns the top-level Groups for the view (after resolving Includes):  
    #---------------------------------------------------------------------------
    
    def _get__groups ( self ):
        if self.__groups is None:
            shadow_group  = self.view.content.get_shadow( self )
            self.__groups = shadow_group.get_content()
        return self.__groups
                
#-------------------------------------------------------------------------------
#  'Dispatcher' class:
#-------------------------------------------------------------------------------
                
class Dispatcher ( object ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, method, info ):
        """ Initializes the object.
        """
        self.method = method
        self.info   = info
        
    #---------------------------------------------------------------------------
    #  Dispatches the method:
    #---------------------------------------------------------------------------
        
    def dispatch ( self ):
        self.method( self.info )

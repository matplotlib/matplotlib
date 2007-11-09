#-------------------------------------------------------------------------------
#
#  Written by: David C. Morrill
#
#  Date: 05/20/2005
#
#  (c) Copyright 2005 by Enthought, Inc.
#
#  Classes defined: KeyBinding, KeyBindings
#
#-------------------------------------------------------------------------------
""" Defines KeyBinding and KeyBindings classes, which manage the mapping of
keystroke events into method calls on controller objects that are supplied by
the application.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import TraitError, HasStrictTraits, Str, List, Any, Instance, Event

from enthought.traits.ui.api \
    import View, Item, ListEditor, KeyBindingEditor, toolkit

#-------------------------------------------------------------------------------
#  Key binding trait definition:
#-------------------------------------------------------------------------------

# Trait definition for key bindings
Binding = Str( event = 'binding', editor = KeyBindingEditor() )
    
#-------------------------------------------------------------------------------
#  'KeyBinding' class:  
#-------------------------------------------------------------------------------

class KeyBinding ( HasStrictTraits ):
    """ Binds one or two keystrokes to a method.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    # First key binding
    binding1 = Binding
    
    # Second key binding
    binding2 = Binding
        
    # Description of what application function the method performs
    description = Str
    
    # Name of controller method the key is bound to
    method_name = Str
    
    # KeyBindings object that "owns" the KeyBinding
    owner = Instance( 'KeyBindings' )
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------
    
    traits_view = View( [ 'binding1', 'binding2', 'description~#', '-<>' ] )
    
    #---------------------------------------------------------------------------
    #  Handles a binding trait being changed:  
    #---------------------------------------------------------------------------
    
    def _binding_changed ( self ):
        if self.owner is not None:
            self.owner.binding_modified = self

#-------------------------------------------------------------------------------
#  'KeyBindings' class:  
#-------------------------------------------------------------------------------
                      
class KeyBindings ( HasStrictTraits ):
    """ A set of key bindings.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # Set of defined key bindings (added dynamically)
    #bindings = List( KeyBinding )
    
    # Optional prefix to add to each method name
    prefix = Str
    
    # Optional suffix to add to each method name
    suffix = Str
    
    # Event fired when one of the contained KeyBinding objects is changed
    binding_modified = Event( KeyBinding )
    
    # Control that currently has the focus (if any)
    focus_owner = Any
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------
        
    traits_view = View( [ Item( 'bindings@#', 
                                editor = ListEditor( style = 'custom' ) ),
                          '|{Click on a first or second column entry, then '
                          'press the key to assign to the corresponding '
                          'function}<>' ],
                        title     = 'Update Key Bindings',
                        kind      = 'livemodal',
                        resizable = True,
                        width     = 0.4,
                        height    = 0.4,
                        help      = False )
                        
    #---------------------------------------------------------------------------
    #  Initializes the object:  
    #---------------------------------------------------------------------------
                              
    def __init__ ( self, *bindings, **traits ):
        super( KeyBindings, self ).__init__( **traits )
        n = len( bindings )
        self.add_trait( 'bindings', List( KeyBinding, minlen = n, 
                                                      maxlen = n, 
                                                      mode   = 'list' ) )
        self.bindings = [ binding.set( owner = self ) for binding in bindings ]
    
    #---------------------------------------------------------------------------
    #  Processes a keyboard event:  
    #---------------------------------------------------------------------------
        
    def do ( self, event, controller, *args ):
        """ Processes a keyboard event.
        """
        key_name = toolkit().key_event_to_name( event )
        for binding in self.bindings:
            if (key_name == binding.binding1) or (key_name == binding.binding2):
                method_name = '%s%s%s' % ( 
                              self.prefix, binding.method_name, self.suffix )
                return (getattr( controller, method_name )( *args ) != 
                        False)
        return False
                
    #---------------------------------------------------------------------------
    #  Merges another set of key bindings into this set:  
    #---------------------------------------------------------------------------
                                
    def merge ( self, key_bindings ):
        """ Merges another set of key bindings into this set.
        """
        binding_dic = {}
        for binding in self.bindings:
            binding_dic[ binding.method_name ] = binding
            
        for binding in key_bindings.bindings:
            binding2 = binding_dic.get( binding.method_name )
            if binding2 is not None:
                binding2.binding1 = binding.binding1
                binding2.binding2 = binding.binding2
                
    #---------------------------------------------------------------------------
    #  Returns the current binding for a specified key (if any):
    #---------------------------------------------------------------------------
                                
    def key_binding_for ( self, binding, key_name ):
        """ Returns the current binding for a specified key (if any).
        """
        if key_name != '':
            for a_binding in self.bindings:
                if ((a_binding is not binding) and
                    ((key_name == a_binding.binding1) or 
                     (key_name == a_binding.binding2))):
                    return a_binding
        return None
                
    #---------------------------------------------------------------------------
    #  Handles a binding being changed:  
    #---------------------------------------------------------------------------
                                
    def _binding_modified_changed ( self, binding ):
        binding1 = binding.binding1
        binding2 = binding.binding2
        for a_binding in self.bindings:
            if binding is not a_binding:
                if binding1 == a_binding.binding1:
                    a_binding.binding1 = ''
                if binding1 == a_binding.binding2:
                    a_binding.binding2 = ''
                if binding2 == a_binding.binding1:
                    a_binding.binding1 = ''
                if binding2 == a_binding.binding2:
                    a_binding.binding2 = ''
                    
    #---------------------------------------------------------------------------
    #  Handles the focus owner being changed:  
    #---------------------------------------------------------------------------
                                        
    def _focus_owner_changed ( self, old, new ):
        if old is not None:
            old.border_size = 0
            
#-- object overrides -----------------------------------------------------------

    #---------------------------------------------------------------------------
    #  Restores the state of a previously pickled object:  
    #---------------------------------------------------------------------------

    def __setstate__ ( self, state ):
        """ Restores the state of a previously pickled object.
        """
        n = len( state[ 'bindings' ] )
        self.add_trait( 'bindings', List( KeyBinding, minlen = n, maxlen = n ) )
        self.__dict__.update( state )
        self.bindings = self.bindings[:]


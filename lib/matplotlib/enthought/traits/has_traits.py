#-------------------------------------------------------------------------------
#
#  Defines the HasTraits class along with several useful subclasses and the
#  associated meta-classes.
#
#  Written by: David C. Morrill
#
#  Original Date:                         06/21/2002
#  Rewritten as a C-based type extension: 06/21/2004
#
#  Symbols defined: HasTraits
#                   HasStrictTraits
#                   HasPrivateTraits
#
#  (c) Copyright 2002, 2004 by Enthought, Inc.
#
#------------------------------------------------------------------------------- 
               
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import sys
import copy

from types           import FunctionType, MethodType
from ctraits         import CHasTraits, CTraitMethod
from traits          import Trait, CTrait, Python, warning, deprecate, Any, \
                            Event, Disallow, TraitFactory, trait_factory, \
                            Property, ForwardProperty, __newobj__
from trait_notifiers import StaticAnyTraitChangeNotifyWrapper, \
                            StaticTraitChangeNotifyWrapper, \
                            TraitChangeNotifyWrapper
from trait_base      import Missing, enumerate
from trait_errors    import TraitError                            
                           
#-------------------------------------------------------------------------------
#  Deferred definitions:
#
#  The following classes have a 'chicken and the egg' definition problem. They 
#  require Traits to work, because they subclass Traits, but the Traits 
#  meta-class programming support uses them, so Traits can't be subclassed
#  until they are defined.
#
#-------------------------------------------------------------------------------

class ViewElement ( object ):
    pass

def ViewElements ( ): 
    return None
       
#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------
                
WrapperTypes     = ( StaticAnyTraitChangeNotifyWrapper, 
                     StaticTraitChangeNotifyWrapper )
                     
MethodTypes      = ( MethodType,   CTraitMethod )                  
FunctionTypes    = ( FunctionType, CTraitMethod )    

# Class dictionary entries used to save trait and view information and 
# definitions:
BaseTraits     = '__base_traits__'
ClassTraits    = '__class_traits__'
PrefixTraits   = '__prefix_traits__'
ViewTraits     = '__view_traits__'
SubclassTraits = '__subclass_traits__'

# The default traits View name:
DefaultTraitsView = 'traits_view'
            
#-------------------------------------------------------------------------------
#  Function used for removing 'event' traits from a list of traits:
#-------------------------------------------------------------------------------
            
def _is_not_event ( value ):
    return value != 'event'

#-------------------------------------------------------------------------------
#  Creates a clone of a specified trait:
#-------------------------------------------------------------------------------

def _clone_trait ( clone ):
    trait = CTrait( 0 )
    trait.clone( clone )
    if clone.__dict__ is not None:
        trait.__dict__ = clone.__dict__.copy()
    return trait
        
#-------------------------------------------------------------------------------
#  Gets the definition of a specified method (if any):
#-------------------------------------------------------------------------------

def _get_method ( cls, method ):
    result = getattr( cls, method, None )
    if (result is not None) and isinstance( result, MethodTypes ):
        return result
    return None
    
def _get_def ( class_name, class_dict, bases, method ):
    if method[0:2] == '__':
        method = '_%s%s' % ( class_name, method )
    result = class_dict.get( method )
    if (result is not None) and isinstance( result, FunctionTypes ):
        return result
    for base in bases:
        result = getattr( base, method, None )
        if (result is not None) and isinstance( result, MethodTypes ):
            return result
    return None

#-------------------------------------------------------------------------------
#  '_SimpleTest' class:
#-------------------------------------------------------------------------------

class _SimpleTest:
    def __init__ ( self, value ): self.value = value
    def __call__ ( self, test  ): return test == self.value
        
#-------------------------------------------------------------------------------
#  Checks if a function can be converted to a 'trait method' (and convert it if
#  possible):
#-------------------------------------------------------------------------------

def _check_method ( cls, class_dict, name, func ):
    method_name  = name
    return_trait = Any
    col          = name.find( '__' )
    if col >= 1:
        type_name    = name[ : col ]
        method_name  = name[ col + 2: ]
        return_trait = globals().get( type_name )
        if not isinstance( return_trait, CTrait ):
            try:
                return_trait = SpecialNames.get( type_name.lower() )
            except:
                return_trait = Any
                method_name  = name
    has_traits = (method_name != name)
    arg_traits = []
    defaults   = func.func_defaults
    if defaults is not None:
        for trait in defaults:
            if isinstance( trait, CTrait ):
                has_traits = True
            elif isinstance( trait, TraitFactory ):
                has_traits = True
                trait      = trait_factory( trait )
            else:
                trait = Any( trait )
            arg_traits.append( trait )
    if has_traits:
        code       = func.func_code
        var_names  = code.co_varnames
        arg_traits = (([ Missing ] * (code.co_argcount - len( arg_traits ))) + 
                      arg_traits)
        traits     = []
        for i, trait in enumerate( arg_traits ):
            traits.append( var_names[i] )
            traits.append( trait )
        del class_dict[ name ]
        class_dict[ method_name ] = CTraitMethod( method_name, func, 
                                          tuple( [ return_trait ] + traits ) )

#-------------------------------------------------------------------------------
#  Returns the trait corresponding to a specified value:
#-------------------------------------------------------------------------------

def _trait_for ( trait ):
    if isinstance( trait, CTrait ):
        return trait
    elif isinstance( trait, TraitFactory ):
        return trait_factory( trait )
    else:
        return Trait( trait )
        
#-------------------------------------------------------------------------------
#  Returns the 'mapped trait' definition for a mapped trait:
#-------------------------------------------------------------------------------
        
def _mapped_trait_for ( trait ):        
    default_value = trait.default_value()[1]
    try:
        default_value = trait.handler.mapped_value( default_value ) 
    except:
        pass
    return Any( default_value, is_base = False )
    
#-------------------------------------------------------------------------------
#  Adds a list of handlers to a specified notifiers list:
#-------------------------------------------------------------------------------
        
def _add_notifiers ( notifiers, handlers ):
    for handler in handlers:
        if not isinstance( handler, WrapperTypes ):
            handler = StaticTraitChangeNotifyWrapper( handler )
        notifiers.append( handler )
            
#-------------------------------------------------------------------------------
#  Adds any specified event handlers defined for a trait by a class:  
#-------------------------------------------------------------------------------

def _add_event_handlers ( trait, cls, handlers ):
    events = trait.event
    if events is not None:
        if type( events ) is str:
            events = [ events ]
        for event in events:
            handlers.append( _get_method( cls, '_%s_changed' % event ) )
            handlers.append( _get_method( cls, '_%s_fired'   % event ) )
            
#-------------------------------------------------------------------------------
#  Returns the method associated with a particular class property getter/setter:
#-------------------------------------------------------------------------------
            
def _property_method ( class_dict, name ):
    
    method = class_dict.get( name )
    if method is not None:
        del class_dict[ name ]
    return method
                                          
#-------------------------------------------------------------------------------
#  Defines a factory function for creating type checked methods:
#-------------------------------------------------------------------------------
                                          
def trait_method ( func, return_type, **arg_types ):
    # Make the sure the first argument is a function:
    if type( func ) is not FunctionType:
        if type( return_type ) is not FunctionType:
            raise TypeError, "First or second argument must be a function."
        else:
            func, return_type = return_type, func
        
    # Make sure the return type is a trait (if not, coerce it to one):
    return_type = _trait_for( return_type )
            
    # Make up the list of arguments defined by the function we are wrapping:
    code       = func.func_code
    arg_count  = code.co_argcount
    var_names  = code.co_varnames[ : arg_count ]
    defaults   = func.func_defaults or ()
    defaults   = ( Missing, ) * (arg_count - len( defaults )) + defaults
    arg_traits = []
    for i, name in enumerate( var_names ):
        try:
            trait = arg_types[ name ]
            del arg_types[ name ]
        except:
            # fixme: Should this be a hard error (i.e. missing parameter type?)
            trait = Any
        arg_traits.append( name )
        arg_traits.append( Trait( defaults[i], _trait_for( trait ) ) )
    
    # Make sure there are no unaccounted for type parameters left over:
    if len( arg_types ) > 0:
        names = arg_types.keys()
        if len( names ) == 1:
            raise TraitError, ("The '%s' keyword defines a type for an "
                               "argument which '%s' does not have." % ( 
                               names[0], func.func_name ))
        else:
            names.sort()
            raise TraitError, ("The %s keywords define types for arguments "
                               "which '%s' does not have." % ( 
                               ', '.join( [ "'%s'" % name for name in names ] ),
                               func.func_name ))
                
    # Otherwise, return a method wrapper for the function:
    return CTraitMethod( func.func_name, func, 
                                         tuple( [ return_type ] + arg_traits ) )
                                         
#-------------------------------------------------------------------------------
#  Defines a method 'decorator' for adding type checking to methods:
#-------------------------------------------------------------------------------

def _add_assignment_advisor ( callback, depth = 2 ):

     frame      = sys._getframe( depth )
     old_trace  = [ frame.f_trace ]
     old_locals = frame.f_locals.copy()

     def tracer ( frm, event, arg ):

         if event == 'call':
             if old_trace[0]:
                 return old_trace[0]( frm, event, arg )
             else:
                 return None
         try:
             if frm is frame and event != 'exception':
                 for k, v in frm.f_locals.items():
                     if k not in old_locals:
                         del frm.f_locals[k]
                         break
                     elif old_locals[k] is not v:
                         frm.f_locals[k] = old_locals[k]
                         break
                 else:
                     return tracer

                 callback( frm, k, v )

         finally:
             if old_trace[0]:
                 old_trace[0] = old_trace[0]( frm, event, arg )

         frm.f_trace = old_trace[0]
         sys.settrace( old_trace[0] )
         return old_trace[0]

     frame.f_trace = tracer
     sys.settrace( tracer )

def method ( return_type = Any, *arg_types, **kwarg_types ):
    
    # The following is a 'hack' to get around what seems to be a Python bug
    # that does not pass 'return_type' and 'arg_types' through to the scope of 
    # 'callback' below:
    kwarg_types[''] = ( return_type, arg_types )

    def callback ( frame, method_name, func ):
        
        # This undoes the work of the 'hack' described above:
        return_type, arg_types = kwarg_types['']
        del kwarg_types['']
        
        # Add a 'fake' positional argument as a place holder for 'self':
        arg_types = ( Any, ) + arg_types
         
        # Make the sure the first argument is a function:
        if type( func ) is not FunctionType:
            raise TypeError, ("'method' must immediately precede a method "
                              "definition.")
            
        # Make sure the return type is a trait (if not, coerce it to one):
        return_type = _trait_for( return_type )
                
        # Make up the list of arguments defined by the function we are wrapping:
        code       = func.func_code
        func_name  = func.func_name
        arg_count  = code.co_argcount
        var_names  = code.co_varnames[ : arg_count ]
        defaults   = func.func_defaults or ()
        defaults   = ( Missing, ) * (arg_count - len( defaults )) + defaults
        arg_traits = []
        n          = len( arg_types )
        if n > len( var_names ):
            raise TraitError, ("Too many positional argument types specified "
                               "in the method signature for %s" % func_name)
        for i, name in enumerate( var_names ):
            if (i > 0) and (i < n):
                if name in kwarg_types:
                    raise TraitError, ("The '%s' argument is defined by both "
                                       "a positional and keyword argument in "
                                       "the method signature for %s" % 
                                       ( name, func_name ) )
                trait = arg_types[i]
            else:
                try:
                    trait = kwarg_types[ name ]
                    del kwarg_types[ name ]
                except:
                    # fixme: Should this be an error (missing parameter type?)
                    trait = Any
            arg_traits.append( name )
            arg_traits.append( Trait( defaults[i], _trait_for( trait ) ) )
        
        # Make sure there are no unaccounted for type parameters left over:
        if len( kwarg_types ) > 0:
            names = kwarg_types.keys()
            if len( names ) == 1:
                raise TraitError, ("The '%s' method signature keyword defines "
                                   "a type for an argument which '%s' does not "
                                   "have." % ( names[0], func_name ))
            else:
                names.sort()
                raise TraitError, ("The %s method signature keywords define "
                          "types for arguments which '%s' does not have." % ( 
                          ', '.join( [ "'%s'" % name for name in names ] ),
                          func_name ))
                    
        # Otherwise, return a method wrapper for the function:
        frame.f_locals[ method_name ] = CTraitMethod( func_name, func, 
                                         tuple( [ return_type ] + arg_traits ) )

    _add_assignment_advisor( callback )
     
#-------------------------------------------------------------------------------
#  'MetaHasTraits' class:
#-------------------------------------------------------------------------------

# This really should be 'HasTraits', but its not defined yet:
_HasTraits = None

class MetaHasTraits ( type ):
    
    def __new__ ( cls, class_name, bases, class_dict ):
        MetaHasTraitsObject( cls, class_name, bases, class_dict, False )
            
        # Finish building the class using the updated class dictionary:
        klass = type.__new__( cls, class_name, bases, class_dict )
        if _HasTraits is not None:
            for base in bases:
                if issubclass( base, _HasTraits ):
                    getattr( base, SubclassTraits ).append( klass )
        setattr( klass, SubclassTraits, [] )
        return klass

#-------------------------------------------------------------------------------
#  'MetaHasTraitsObject' class:  
#-------------------------------------------------------------------------------
                
class MetaHasTraitsObject ( object ):
                
    def __init__ ( self, cls, class_name, bases, class_dict, is_category ):
        """ Processes all of the traits related data in the class dictionary.
        """
        # Create the various class dictionaries, lists and objects needed to 
        # hold trait and view information and definitions:
        base_traits   = {}
        class_traits  = {}
        prefix_traits = {}
        prefix_list   = []
        view_elements = ViewElements()
        
        # Move all 'old style' trait definitions to the appropriate trait 
        # class dictionaries:
        traits = class_dict.get( '__traits__' )
        if traits is not None:
            warning( "Use of '__traits__' in class %s is deprecated, "
                     "use new style trait definitions instead." %
                     class_name )
            del class_dict[ '__traits__' ]
            for name, trait in traits.items():
                if not isinstance( trait, CTrait ):
                    trait = Trait( trait )
                if name[ -1: ] != '*':
                    base_traits[ name ] = class_traits[ name ] = trait
                    handler = trait.handler
                    if handler is not None:
                        if handler.has_items:
                            class_traits[ name + '_items' ] = \
                                    handler.items_event()
                        if handler.is_mapped:
                            class_traits[ name + '_' ] = _mapped_trait_for( 
                                                                         trait )
                else:
                    name = name[:-1]
                    prefix_list.append( name )
                    prefix_traits[ name ] = trait
                    
        # Move all trait definitions from the class dictionary to the 
        # appropriate trait class dictionaries:
        for name, value in class_dict.items():
            rc = isinstance( value, CTrait )
            if (not rc) and isinstance( value, TraitFactory ):
                value = trait_factory( value )
                rc    = isinstance( value, CTrait )
            if (not rc) and isinstance( value, ForwardProperty ):
                rc       = True
                validate = _property_method( class_dict, '_validate_' + name )
                if validate is None:
                    validate = value.validate
                value = Property( 
                            _property_method( class_dict, '_get_' + name ),
                            _property_method( class_dict, '_set_' + name ),
                            validate, True, **value.metadata )
            if rc:
                del class_dict[ name ]
                if name[-1:] != '_':
                    base_traits[ name ] = class_traits[ name ] = value
                    handler = value.handler
                    if handler is not None:
                        if handler.has_items:
                            class_traits[ name + '_items' ] = \
                                    handler.items_event()
                        if handler.is_mapped:
                            class_traits[ name + '_' ] = _mapped_trait_for( 
                                                                         value )
                else:
                    name = name[:-1]
                    prefix_list.append( name )
                    prefix_traits[ name ] = value
            elif isinstance( value, FunctionType ):
                _check_method( cls, class_dict, name, value )
            elif isinstance( value, property ):
                class_traits[ name ] = generic_trait
                
            # Handle any view elements found in the class:                
            elif isinstance( value, ViewElement ):
                
                # Add the view element to the class's 'ViewElements' if it is 
                # not already defined (duplicate definitions are errors):
                if name in view_elements.content:
                    raise TraitError, \
                          "Duplicate definition for view element '%s'" % name
                view_elements.content[ name ] = value
                
                # Replace all substitutable view sub elements with 'Include'
                # objects, and add the sustituted items to the 'ViewElements':
                value.replace_include( view_elements )
                    
                # Remove the view element from the class definition:
                del class_dict[ name ]
                    
        # Process all base classes:                    
        for base in bases:
            
            # Merge any base class trait definitions:
            base_base_traits = base.__dict__.get( BaseTraits )
            if base_base_traits is not None:
                        
                # Merge base traits:
                for name, value in base_base_traits.items():
                    if name not in base_traits:
                        base_traits[ name ] = value
                    elif is_category:
                        raise TraitError, ("Cannot override '%s' trait "
                                           "definition in a category" % name)
                
                # Merge class traits:
                for name, value in base.__dict__.get( ClassTraits ).items():
                    if name not in class_traits:
                        class_traits[ name ] = value
                    elif is_category:
                        raise TraitError, ("Cannot override '%s' trait "
                                           "definition in a category" % name)
    
                # Merge prefix traits:
                base_prefix_traits = base.__dict__.get( PrefixTraits )
                for name in base_prefix_traits['*']:
                    if name not in prefix_list:
                        prefix_list.append( name )
                        prefix_traits[ name ] = base_prefix_traits[ name ]
                    elif is_category:
                        raise TraitError, ("Cannot override '%s_' trait "
                                           "definition in a category" % name)
            
            # If the base class has a 'ViewElements' object defined, add it to 
            # the 'parents' list of this class's 'ViewElements':
            parent_view_elements = base.__dict__.get( ViewTraits )
            if parent_view_elements is not None:
                view_elements.parents.append( parent_view_elements )
                        
        # Make sure there is a definition for 'undefined' traits:
        if (prefix_traits.get( '' ) is None) and (not is_category):
            prefix_list.append( '' )
            prefix_traits[''] = Python
            
        # Save a link to the prefix_list:
        prefix_traits['*'] = prefix_list
                    
        # Make sure the trait prefixes are sorted longest to shortest
        # so that we can easily bind dynamic traits to the longest matching
        # prefix:
        prefix_list.sort( lambda x, y: len( y ) - len( x ) )

        # If there is an 'anytrait_changed' event handler, wrap it so that
        # it can be attached to all traits in the class:  
        if deprecate < 2:
            anytrait = _get_def( class_name, class_dict, bases, 
                                 'anytrait_changed' )
        if (deprecate >= 2) or (anytrait is None):
            anytrait = _get_def( class_name, class_dict, bases, 
                                 '_anytrait_changed' )
        if anytrait is not None:
            anytrait = StaticAnyTraitChangeNotifyWrapper( anytrait )
            
            # Save it in the prefix traits dictionary so that any dynamically 
            # created traits (e.g. 'prefix traits') can re-use it:
            prefix_traits['@'] = anytrait
            
        # Make one final pass over the class traits dictionary, making sure 
        # all static trait notification handlers are attached to a 'cloned' 
        # copy of the original trait:
        for name, trait in class_traits.items():
            if deprecate >= 2:
                changex = None
            else:
                changex = _get_def( class_name, class_dict, bases, 
                                    '%s_changed' % name )
            handlers = [ anytrait, changex,
                         _get_def( class_name, class_dict, bases, 
                                   '_%s_changed' % name ),
                         _get_def( class_name, class_dict, bases, 
                                   '_%s_fired' % name ) ]
            events = trait.event
            if events is not None:
                if type( events ) is str:
                    events = [ events ]
                for event in events:
                    handlers.append( _get_def( class_name, class_dict, bases, 
                                               '_%s_changed' % event ) )
                    handlers.append( _get_def( class_name, class_dict, bases, 
                                               '_%s_fired' % event ) )
            handlers = [ h for h in handlers if h is not None ]
            default  = _get_def( class_name, class_dict, bases, 
                                 '_%s_default' % name )
            if (len( handlers ) > 0) or (default is not None):
                if changex is not None:
                    warning( "Use of 'def %s_changed' in class %s is "
                             "deprecated, use 'def _%s_changed' instead." % ( 
                             name, class_name, name ) ) 
                class_traits[ name ] = trait = _clone_trait( trait )
                if len( handlers ) > 0:
                    _add_notifiers( trait._notifiers( 1 ), handlers )
                if default is not None:
                    trait.default_value( 8, default )
        
        # Add the traits meta-data to the class:
        self.add_traits_meta_data( bases, class_dict, base_traits, class_traits,
                                   prefix_traits, view_elements )                                            
    
    #---------------------------------------------------------------------------
    #  Adds the traits meta-data to the class:   
    #---------------------------------------------------------------------------
           
    def add_traits_meta_data ( self, bases, class_dict, base_traits, 
                               class_traits,  prefix_traits, view_elements ):
        class_dict[ BaseTraits   ] = base_traits
        class_dict[ ClassTraits  ] = class_traits
        class_dict[ PrefixTraits ] = prefix_traits
        class_dict[ ViewTraits   ] = view_elements

#-------------------------------------------------------------------------------
#  Manages the list of trait instance monitors: 
#-------------------------------------------------------------------------------

# List of new trait instance monitors:
_HasTraits_monitors = []

def _trait_monitor_index ( cls, handler ):
    type_handler = type( handler )
    for i, _cls, _handler in enumerate( _HasTraits_monitors ):
        if type_handler is type( _handler ): 
            if ((type_handler is MethodType) and 
                (handler.im_self is not None)):
                if ((handler.__name__ == _handler.__name__) and
                    (handler.im_self is _handler.im_self)):
                   return i
            elif handler == _handler:
                return i
    return -1
        
#-------------------------------------------------------------------------------
#  'HasTraits' class:
#-------------------------------------------------------------------------------

class HasTraits ( CHasTraits ):
    
    __metaclass__ = MetaHasTraits 
    
    # Trait definitions:
    trait_added = Event( str )
        
    #---------------------------------------------------------------------------
    #  Adds/Removes a trait instance creation monitor: 
    #---------------------------------------------------------------------------
             
    def trait_monitor ( cls, handler, remove = False ):
        global _HasTraits_monitors
        
        index = _trait_monitor_index( cls, handler )
        if remove:
            if index >= 0:
                del _HasTraits_monitors[ index ]
            return
        
        if index < 0:
            _HasTraits_monitors.append( ( cls, handler ) )
        
    trait_monitor = classmethod( trait_monitor )
        
    #---------------------------------------------------------------------------
    #  Add a new class trait (i.e. applies to all instances and subclasses): 
    #---------------------------------------------------------------------------
    
    def add_class_trait ( cls, name, *trait ):
        
        # Make sure a trait argument was specified:
        if len( trait ) == 0:
            raise ValueError, 'No trait definition was specified.'
            
        # Make sure only valid traits get added:
        if len( trait ) > 1:
            trait = Trait( *trait )
        else:
            trait = _trait_for( trait[0] )
            
        # Add the trait to the class:
        cls._add_class_trait( name, trait, False )
        
        # Also add the trait to all subclasses of this class:
        for subclass in cls.trait_subclasses( True ):
            subclass._add_class_trait( name, trait, True )
            
    add_class_trait = classmethod( add_class_trait )
            
    def _add_class_trait ( cls, name, trait, is_subclass ):
        # Get a reference to the class's dictionary and 'prefix' traits:
        class_dict    = cls.__dict__
        prefix_traits = class_dict[ PrefixTraits ]
            
        # See if the trait is a 'prefix' trait:
        if name[-1:] == '_':
            name = name[:-1]
            if name in prefix_traits:
                if is_subclass:
                    return
                raise TraitError, "The '%s_' trait is already defined." % name
            prefix_traits[ name ] = trait
                
            # Otherwise, add it to the list of known prefixes:
            prefix_list = prefix_traits['*']
            prefix_list.append( name )
            
            # Resort the list from longest to shortest:
            prefix_list.sort( lambda x, y: len( y ) - len( x ) )
            
            return
            
        # Check to see if the trait is already defined:
        class_traits = class_dict[ ClassTraits ]
        if class_traits.get( name ) is not None:
            if is_subclass:
                return
            raise TraitError, "The '%s' trait is aleady defined." % name
            
        # Check to see if the trait has additional sub-traits that need to be
        # defined also:
        handler = trait.handler
        if handler is not None:
            if handler.has_items:
                cls.add_class_trait( name + '_items', handler.items_event() )
            if handler.is_mapped:
                cls.add_class_trait( name + '_', _mapped_trait_for( trait ) )
            
        # Make the new trait inheritable (if allowed):
        if trait.is_base is not False:
            class_dict[ BaseTraits ][ name ] = trait
        
        # See if there are any static notifiers defined:
        if deprecate >= 2:
            changex = None
        else:
            changex = _get_method( cls, '%s_changed'  % name )
            
        handlers = [ changex,
                     _get_method( cls, '_%s_changed' % name ),
                     _get_method( cls, '_%s_fired'   % name ) ]
                     
        # Add any special trait defined event handlers:
        _add_event_handlers( trait, cls, handlers )
        
        # Add the 'anytrait' handler (if any):
        handlers.append( prefix_traits.get( '@' ) )
        
        # Filter out any 'None' values:
        handlers = [ h for h in handlers if h is not None ]
        
        # If there are and handlers, add them to the trait's notifier's list:
        if len( handlers ) > 0:
            if changex is not None:
                warning( "'def %s_changed' is deprecated, use "
                         "'def _%s_changed' instead." % ( name, name ) ) 
            trait = _clone_trait( trait ) 
            _add_notifiers( trait._notifiers( 1 ), handlers )
                
        # Finally, add the new trait to the class trait dictionary:
        class_traits[ name ] = trait
        
    _add_class_trait = classmethod( _add_class_trait )
        
    #---------------------------------------------------------------------------
    #  Returns the immediate (or all) subclasses of this class:   
    #---------------------------------------------------------------------------
    
    def trait_subclasses ( cls, all = False ):
        """ Returns the immediate (or all) subclasses of this class.
        """
        if not all:
            return getattr( cls, SubclassTraits )[:]
        return cls._trait_subclasses( [] )
        
    trait_subclasses = classmethod( trait_subclasses )
        
    def _trait_subclasses ( cls, subclasses ):
        for subclass in getattr( cls, SubclassTraits ):         
            if subclass not in subclasses:
                subclasses.append( subclass )
                subclass._trait_subclasses( subclasses )
        return subclasses
                            
    _trait_subclasses = classmethod( _trait_subclasses )

    #---------------------------------------------------------------------------
    #  Initialize the trait values of an object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        global _HasTraits_monitors
        
        # Define any traits specified in the constructor:
        for name, value in traits.items():
            setattr( self, name, value )
            
        # Notify any interested monitors that a new object has been created:
        for cls, handler in _HasTraits_monitors:
            if isinstance( self, cls ):
                handler( self )
                
    #---------------------------------------------------------------------------
    #  Prepares an object to be pickled:
    #---------------------------------------------------------------------------
                
    def __reduce_ex__ ( self, protocol ):
        state = self.__dict__
        try:
            getstate = self.__getstate__
            if getstate is not None:
                state = getstate()
        except:
            pass
        return ( __newobj__, ( self.__class__, ), state )

    #---------------------------------------------------------------------------
    #  Shortcut for setting object traits:
    #---------------------------------------------------------------------------

    def set ( self, **traits ):
        """ Shortcut for setting object traits.
        """
        for name, value in traits.items():
            setattr( self, name, value )
        return self
       
    #---------------------------------------------------------------------------
    #  Resets some or all of an object's traits to their default values:
    #---------------------------------------------------------------------------
             
    def reset_traits ( self, traits = None ):
        """ Resets some or all of an object's traits to their default values.
        """
        unresetable = []
        if traits is None:
            traits = self.trait_names()
        for name in traits:
            try:
                delattr( self, name )
            except AttributeError:
                unresetable.append( name )
        return unresetable
        
    #---------------------------------------------------------------------------
    #  Returns the list of trait names to copy/clone by default:  
    #---------------------------------------------------------------------------
                
    def copyable_trait_names ( self ):
        """ Returns the list of trait names to copy/clone by default.
        """
        return self.trait_names()
 
    #---------------------------------------------------------------------------
    #  Copies another object's traits into this one:
    #---------------------------------------------------------------------------
 
    def copy_traits ( self, other, traits = None, memo = None ):
        """ Copies another object's traits into this one.
        """
        if traits is None:
            traits = self.copyable_trait_names()
            
        unassignable = []
        deferred     = []
        for name in traits:
            try:
                trait = self.trait( name )
                if trait.type == 'delegate':
                    deferred.append( name )
                    continue
                base_trait = other.base_trait( name )
                if base_trait.type == 'event':
                    continue
                value     = getattr( other, name )
                copy_type = base_trait.copy
                if copy_type == 'deep':
                    if memo is None:
                        value = copy.deepcopy( value )
                    else:
                        value = copy.deepcopy( value, memo )
                elif copy_type == 'shallow':
                    value = copy.copy( value )
                setattr( self, name, value )
            except:
                unassignable.append( name )
                
        for name in deferred:
            try:
                value     = getattr( other, name )
                copy_type = other.base_trait( name ).copy
                if copy_type == 'deep':
                    if memo is None:
                        value = copy.deepcopy( value )
                    else:
                        value = copy.deepcopy( value, memo )
                elif copy_type == 'shallow':
                    value = copy.copy( value )
                setattr( self, name, value )
            except:
                unassignable.append( name )
        return unassignable
        
    #---------------------------------------------------------------------------
    #  Clones a new object from this one, optionally copying only a specified 
    #  set of traits:
    #---------------------------------------------------------------------------
        
    def clone_traits ( self, traits = None, memo = None ):
        """ Clones a new object from this one, optionally copying only a 
            specified set of traits.
        """
        if memo is None:
            memo = {}
        new = self.__new__( self.__class__ )
        new.copy_traits( self, traits, memo )
        return new
        
    #---------------------------------------------------------------------------
    #  Creates a deep copy of the object:
    #---------------------------------------------------------------------------
       
    def __deepcopy__ ( self, memo ):
        """ Creates a deep copy of the object.
        """
        id_self = id( self )
        if id_self in memo:
            return memo[ id_self ]
        memo[ id_self ] = result = self.clone_traits( memo = memo )
        return result
        
    #---------------------------------------------------------------------------
    #  Edits the object's traits:
    #---------------------------------------------------------------------------
    
    def edit_traits ( self, view    = None, parent = None, kind = None, 
                            context = None ): 
        """ Edits the object's traits.
        """
        if context is None:
            context = self
        view = self.trait_view( view )
        return view.ui( context, parent, kind, self.trait_view_elements() )
        
    #---------------------------------------------------------------------------
    #  Gets or sets a ViewElement associated with an object's class:
    #---------------------------------------------------------------------------
        
    def trait_view ( self, name = None, view_element = None ):
        """ Gets or sets a ViewElement associated with an object's class.
        """
        # If a view element was passed instead of a name or None, return it:
        if isinstance( name, ViewElement ):
            return name
            
        # Get the ViewElements object associated with the class:
        view_elements = self.trait_view_elements()
        
        if name:
            if view_element is None:
                # If only a name was specified, return the ViewElement it 
                # matches, if any:
                return view_elements.find( name )
                
            # Otherwise, save the specified ViewElement under the name 
            # specified:
            view_elements.content[ name ] = view_element
            return
            
        # Get the default view/view name:
        name = self.default_traits_view()
        
        # If the default is a View, return it:
        if isinstance( name, ViewElement ):
            return name
          
        # Otherwise, get all View objects associated with the object's class:
        names = view_elements.filter_by()
        
        # If the specified default name is in the list, return its View:
        if name in names:
            return view_elements.find( name )
        
        # If there is only one View, return it:
        if len( names ) == 1:
            return view_elements.find( names[0] )
            
        # Otherwise, create and return a View based on the set of editable 
        # traits defined for the object:
        from matplotlib.enthought.traits.ui import View
        return View( self.editable_traits() )
        
    #---------------------------------------------------------------------------
    #  Return the default traits view/name:  
    #---------------------------------------------------------------------------
                
    def default_traits_view ( self ):
        """ Return the default traits view/name.
        """
        return DefaultTraitsView
        
    #---------------------------------------------------------------------------
    #  Gets the list of names of ViewElements associated with the object's
    #  class that are of a specified ViewElement type:
    #---------------------------------------------------------------------------
        
    def trait_views ( self, klass = None ):
        return self.__class__.__dict__[ ViewTraits ].filter_by( klass ) 

    #---------------------------------------------------------------------------
    #  Returns the ViewElements object associated with the object's class:   
    #---------------------------------------------------------------------------
                    
    def trait_view_elements ( self ):
        """ Returns the ViewElements object associated with the object's class.
        """
        return self.__class__.__dict__[ ViewTraits ]
        
    #---------------------------------------------------------------------------
    #  Configure the object's traits:
    #---------------------------------------------------------------------------
    
    def configure_traits ( self, filename = None, view    = None, 
                                 kind     = None, edit    = True, 
                                 context  = None, handler = None ):
        if filename is not None:
            fd = None
            try:
                import cPickle
                fd = open( filename, 'rb' )
                self.copy_traits( cPickle.Unpickler( fd ).load() )
            except:
                if fd is not None:
                    fd.close()
           
        if edit:
            from matplotlib.enthought.traits.ui import toolkit
            if context is None:
                context = self
            rc = toolkit().view_application( context, self.trait_view( view ), 
                                             kind, handler )
            if rc and (filename is not None):
                fd = None
                try:
                    import cPickle
                    fd = open( filename, 'wb' )
                    cPickle.Pickler( fd, True ).dump( self )
                except:
                    if fd is not None:
                        fd.close()
                    return False
                 
        return True
            
    #---------------------------------------------------------------------------
    #  Return the list of editable traits:
    #---------------------------------------------------------------------------
    
    def editable_traits ( self ):
        try:
            # Use the object's specified editable traits:
            result = self.__editable_traits__ 
            warning( "Use of '_editable_traits__' is deprecated, use View "
                     "objects instead." )
            return result
        except:
            # Otherwise, derive it from all of the object's non-event traits:
            names = self.trait_names( type = _is_not_event )
            names.sort()
            return names
      
    #---------------------------------------------------------------------------
    #  Pretty print the traits of an object:
    #---------------------------------------------------------------------------
 
    def print_traits ( self, show_help = False ):
        
        names = self.trait_names( type = _is_not_event )
        if len( names ) == 0:
           print ''
           return
           
        result = []
        pad    = max( [ len( x ) for x in names ] ) + 1
        maxval = 78 - pad
        names.sort()
        
        for name in names:
            try:
                value = repr( getattr( self, name ) ).replace( '\n', '\\n' )
                if len( value ) > maxval:
                    value = '%s...%s' % ( value[: (maxval - 2) / 2 ],
                                          value[ -((maxval - 3) / 2): ] ) 
            except:
                value = '<undefined>'
            lname = (name + ':').ljust( pad )
            if show_help:   
                result.append( '%s %s\n   The value must be %s.' % (
                       lname, value, self.base_trait( name ).setter.info() ) )
            else:
                result.append( '%s %s' % ( lname, value ) )
                
        print '\n'.join( result )
       
    #---------------------------------------------------------------------------
    #  Add/Remove a handler for a specified trait being changed:
    #
    #  If no name is specified, the handler will be invoked for any trait 
    #  change.
    #---------------------------------------------------------------------------
    
    def on_trait_change ( self, handler, name = None, remove = False ):
        
        if type( name ) is list:
            for name_i in name:
                self.on_trait_change( handler, name_i, remove )
            return
            
        name = name or 'anytrait'
            
        if remove:
            if name == 'anytrait':
                notifiers = self._notifiers( 0 )
            else:
                trait = self._trait( name, 1 )
                if trait is None:
                    return
                notifiers = trait._notifiers( 0 )
            if notifiers is not None:
                for i, notifier in enumerate( notifiers ):
                    if notifier.equals( handler ):
                        notifier.dispose()
                        del notifiers[i]
                        break
            return
        
        if name == 'anytrait':
            notifiers = self._notifiers( 1 )
        else:
            notifiers = self._trait( name, 2 )._notifiers( 1 )
        for notifier in notifiers:
            if notifier.equals( handler ):
                break
        else:
            notifiers.append( TraitChangeNotifyWrapper( handler, notifiers ) )
            
    # Make 'on_trait_event' a synonym for 'on_trait_change':
    on_trait_event = on_trait_change
       
    #---------------------------------------------------------------------------
    #  Synchronize the value of two traits:
    #---------------------------------------------------------------------------
    
    def sync_trait ( self, trait_name, object, alias = None, mutual = True ):
        
        if alias is None:
            alias = trait_name
        
        # Synchronize the values by setting the passed in object's trait to 
        # have the same value we have:
        setattr( object, alias, getattr( self, trait_name ) )
        
        # Now hook up the events:
        self.on_trait_change( lambda value: setattr( object, alias, value ),
                              trait_name )
                              
        if mutual:
            object.on_trait_change( 
                   lambda value: setattr( self, trait_name, value ), alias )
        
    #---------------------------------------------------------------------------
    #  Add a new trait: 
    #---------------------------------------------------------------------------
    
    def add_trait ( self, name, *trait ):
        
        # Make sure a trait argument was specified:
        if len( trait ) == 0:
            raise ValueError, 'No trait definition was specified.'
            
        # Make sure only valid traits get added:
        if len( trait ) > 1:
            trait = Trait( *trait )
        else:
            trait = _trait_for( trait[0] )
            
        # Check to see if the trait has additional sub-traits that need to be
        # defined also:
        handler = trait.handler
        if handler is not None:
            if handler.has_items:
                self.add_trait( name + '_items', handler.items_event() )
            if handler.is_mapped:
                self.add_trait( name + '_', _mapped_trait_for( trait ) )
        
        # See if there already is a class or instance trait with the same name:
        old_trait = self._trait( name, 0 )
        
        # Get the object's instance trait dictionary and add a clone of the new
        # trait to it:
        itrait_dict = self._instance_traits()
        itrait_dict[ name ] = trait = _clone_trait( trait )
        
        # If there already was a trait with the same name:
        if old_trait is not None:
            # Copy the old traits notifiers into the new trait:
            old_notifiers = old_trait._notifiers( 0 )
            if old_notifiers is not None:
                trait._notifiers( 1 ).extend( old_notifiers )
        else:
            # Otherwise, see if there are any static notifiers that should be 
            # applied to the trait:
            cls = self.__class__
            if deprecate >= 2:
                changex = None
            else:
                changex = _get_method( cls, '%s_changed'  % name )
                
            handlers = [ changex,
                         _get_method( cls, '_%s_changed' % name ),
                         _get_method( cls, '_%s_fired'   % name ) ]
                      
            # Add any special trait defined event handlers:
            _add_event_handlers( trait, cls, handlers )
            
            # Add the 'anytrait' handler (if any):
            handlers.append( self.__prefix_traits__.get( '@' ) )
            
            # Filter out any 'None' values:
            handlers = [ h for h in handlers if h is not None ]
                    
            # If there are any static notifiers, attach them to the trait:
            if len( handlers ) > 0:
                if changex is not None:
                    warning( "'def %s_changed' is deprecated, use "
                             "'def _%s_changed' instead." % ( name, name ) ) 
                _add_notifiers( trait._notifiers( 1 ), handlers )
                
        # If this was a new trait, fire the 'trait_added' event:
        if old_trait is None:
            self.trait_added = name
            
    #---------------------------------------------------------------------------
    #  Returns the trait definition of a specified trait: 
    #---------------------------------------------------------------------------
                
    def trait ( self, name, force = False ):
        mode = 0
        if force:
            mode = -1
        return self._trait( name, mode )
            
    #---------------------------------------------------------------------------
    #  Returns the base trait definition of a specified trait: 
    #---------------------------------------------------------------------------
                
    def base_trait ( self, name ):
        return self._trait( name, -2 )
       
    #---------------------------------------------------------------------------
    #  Return a dictionary of all traits which match a set of metadata:
    #---------------------------------------------------------------------------
    
    def traits ( self, **metadata ):
        
        if len( metadata ) == 0:
            return self.__base_traits__.copy()
            
        result = {}
        
        for meta_name, meta_eval in metadata.items():
            if type( meta_eval ) is not FunctionType:
                metadata[ meta_name ] = SimpleTest( meta_eval )
                
        for name, trait in self.__base_traits__.items():
            for meta_name, meta_eval in metadata.items():
                if not meta_eval( getattr( trait, meta_name ) ):
                    break
            else:
                result[ name ] = trait
                
        return result
       
    #---------------------------------------------------------------------------
    #  Return a list of all trait names which match a set of metadata:
    #---------------------------------------------------------------------------
    
    def trait_names ( self, **metadata ):
        
        if len( metadata ) == 0:
            return self.__base_traits__.keys()
            
        result = []
        
        for meta_name, meta_eval in metadata.items():
            if type( meta_eval ) is not FunctionType:
                metadata[ meta_name ] = _SimpleTest( meta_eval )
                
        for name, trait in self.__base_traits__.items():
            for meta_name, meta_eval in metadata.items():
                if not meta_eval( getattr( trait, meta_name ) ):
                    break
            else:
                result.append( name )
                
        return result
        
    def _trait_names ( self, **metadata ):
        warning( "'HasTraits._trait_names' is deprecated, use 'trait_names' "
                 "instead." )
        return self.trait_names( **metadata )
    
    #---------------------------------------------------------------------------
    #  Get a trait editor for the specified trait:
    #---------------------------------------------------------------------------
    
    def simple_trait_editor ( self, parent, trait_name, 
                                    description = '',
                                    handler     = None ): 
        return self._trait_editor( 'simple', parent, trait_name, description, 
                                   handler )
    
    def custom_trait_editor ( self, parent, trait_name, 
                                    description = '',
                                    handler     = None ): 
        return self._trait_editor( 'custom', parent, trait_name, description, 
                                   handler )
    
    def text_trait_editor ( self, parent, trait_name, 
                                  description = '',
                                  handler     = None ): 
        return self._trait_editor( 'text', parent, trait_name, description, 
                                   handler )
    
    def readonly_trait_editor ( self, parent, trait_name, 
                                      description = '',
                                      handler     = None ): 
        return self._trait_editor( 'readonly', parent, trait_name, description, 
                                   handler )
                                   
    def _trait_editor ( self, prefix, parent, trait_name, description, handler ):
        if handler is None:
            from trait_sheet import default_trait_sheet_handler
            handler = default_trait_sheet_handler 
        trait = self.trait( trait_name )
        return getattr( trait.get_editor(), prefix + '_editor' )( self, 
                     trait_name, description, handler, parent )
       
    #---------------------------------------------------------------------------
    #  Returns the trait definition for a specified name when there is no
    #  explicit definition in the class:
    #---------------------------------------------------------------------------
    
    def __prefix_trait__ ( self, name ):
        # Never create prefix traits for names of the form '__xxx__':
        if (name[:2] == '__') and (name[-2:] == '__'):
            raise AttributeError, "'%s' object has no attribute '%s'" % ( 
                                  self.__class__.__name__, name )
        
        # Handle the special case of 'delegated' traits:
        if name[-1:] == '_':
           trait = self._trait( name[:-1], 0 )
           if (trait is not None) and (trait.type == 'delegate'):
               return _clone_trait( trait )
        
        prefix_traits = self.__prefix_traits__
        for prefix in prefix_traits['*']:
            if prefix == name[ :len( prefix ) ]:
                # If we found a match, use its trait as a template for a new
                # trait:
                trait = prefix_traits[ prefix ]
                
                # Get any change notifiers that apply to the trait:
                cls = self.__class__
                if deprecate >= 2:
                    changex = None
                else:
                    changex = _get_method( cls, '%s_changed'  % name )
            
                handlers = [ changex,
                             _get_method( cls, '_%s_changed' % name ),
                             _get_method( cls, '_%s_fired'   % name ) ]
                             
                # Add any special trait defined event handlers:
                _add_event_handlers( trait, cls, handlers )
                
                # Add the 'anytrait' handler (if any):
                handlers.append( prefix_traits.get( '@' ) )
                
                # Filter out any 'None' values:
                handlers = [ h for h in handlers if h is not None ]
                
                # If there are any handlers, add them to the trait's notifier's 
                # list:
                if len( handlers ) > 0:
                    if changex is not None:
                        warning( "'def %s_changed' is deprecated, use "
                                 "'def _%s_changed' instead." % ( name, name ) ) 
                    trait = _clone_trait( trait )
                    _add_notifiers( trait._notifiers( 1 ), handlers )
                        
                return trait
                
        # There should ALWAYS be a prefix match in the trait classes, since ''
        # is at the end of the list, so we should never get here:
        raise SystemError, ("Trait class look-up failed for attribute '%s' "
                            "for an object of type '%s'") % ( 
                            name, self.__class__.__name__ )

# Patch the definition of _HasTraits to be the real 'HasTraits':                            
_HasTraits = HasTraits

#-------------------------------------------------------------------------------
#  'HasStrictTraits' class:
#-------------------------------------------------------------------------------
   
class HasStrictTraits ( HasTraits ):
    
    _ = Disallow   # Disallow access to any traits not explicitly defined
    
#-------------------------------------------------------------------------------
#  'HasPrivateTraits' class:
#-------------------------------------------------------------------------------
    
class HasPrivateTraits ( HasTraits ):
    
    __ = Any       # Make 'private' traits (leading '_') have no type checking
    _  = Disallow  # Disallow access to all other traits not explicitly defined
    
#-------------------------------------------------------------------------------
#  Deprecated classes:
#-------------------------------------------------------------------------------
    
class HasDynamicTraits ( HasTraits ):

    def __init__ ( self, **traits ):
        warning( "'HasDynamicTraits' is deprecated, use 'HasTraits' instead." )
        HasTraits.__init__( self, **traits ) 


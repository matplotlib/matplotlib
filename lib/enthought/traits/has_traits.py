#------------------------------------------------------------------------------
#
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#  Thanks for using Enthought open source!
#
#  Author:        David C. Morrill, Prabhu Ramachandran
#  Original Date: 06/21/2002
#  Symbols defined: HasTraits
#                   HasStrictTraits
#                   HasPrivateTraits
#
#  Rewritten as a C-based type extension: 06/21/2004
#
#------------------------------------------------------------------------------
"""
Defines the HasTraits class, along with several useful subclasses and associated
metaclasses.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import sys
import copy as copy_module
import weakref

from cPickle \
    import dumps

from types \
    import FunctionType, MethodType

from ctraits \
    import CHasTraits, CTraitMethod, _HasTraits_monitors

from traits \
    import Trait, CTrait, Python, Any, Event, Enum, Disallow, TraitFactory, \
           trait_factory, Property, ForwardProperty, generic_trait, false, \
           __newobj__, SpecialNames

from trait_notifiers \
    import StaticAnyTraitChangeNotifyWrapper, StaticTraitChangeNotifyWrapper, \
           TraitChangeNotifyWrapper, UITraitChangeNotifyWrapper, \
           NewTraitChangeNotifyWrapper

from trait_base \
    import Missing, enumerate, SequenceTypes

from trait_errors \
    import TraitError

#-------------------------------------------------------------------------------
#  Deferred definitions:
#
#  The following classes have a 'chicken and the egg' definition problem. They
#  require Traits to work, because they subclass Traits, but the Traits
#  meta-class programming support uses them, so Traits can't be subclassed
#  until they are defined.
#
#  Note: We need to look at whether the Category support could be used to
#        allow us to implement this better.
#
#-------------------------------------------------------------------------------

class ViewElement ( object ):
    pass

def ViewElements ( ):
    return None

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

WrapperTypes   = ( StaticAnyTraitChangeNotifyWrapper,
                   StaticTraitChangeNotifyWrapper )

MethodTypes    = ( MethodType,   CTraitMethod )
FunctionTypes  = ( FunctionType, CTraitMethod )

# Class dictionary entries used to save trait and view information and
# definitions:

BaseTraits     = '__base_traits__'
ClassTraits    = '__class_traits__'
PrefixTraits   = '__prefix_traits__'
ViewTraits     = '__view_traits__'
SubclassTraits = '__subclass_traits__'
InstanceTraits = '__instance_traits__'

# The default Traits View name
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
#  Returns a dictionary of potential 'Instance' or 'List(Instance)' handlers:
#-------------------------------------------------------------------------------

def _get_instance_handlers ( class_dict, bases ):
    """ Returns a dictionary of potential 'Instance' or 'List(Instance)'
        handlers.
    """
    # Create the results dictionary:
    instance_traits = {}

    # Merge all of the base class information into the result:
    for base in bases:
        for name, base_arg_lists in base.__dict__.get( InstanceTraits ).items():
            arg_lists = instance_traits.get( name )
            if arg_lists is None:
                instance_traits[ name ] = base_arg_lists[:]
            else:
                for arg_list in base_arg_lists:
                    if arg_list not in arg_lists:
                        arg_lists.append( arg_list )

    # Merge in the information from the class dictionary:
    for name, value in class_dict.items():
        if (name[:1] == '_') and isinstance( value, FunctionTypes ):
            n   = 13
            col = name.find( '_changed_for_' )
            if col < 2:
                 n   = 11
                 col = name.find( '_fired_for_' )
            if col >= 2:
                key = name[ col + n: ]
                if key != '':
                    arg_list  = ( name, name[ 1: col ] )
                    arg_lists = instance_traits.get( key )
                    if arg_lists is None:
                        instance_traits[ key ] = arg_lists = []
                    if arg_list not in arg_lists:
                        arg_lists.append( arg_list )

    # Return the dictionary of possible arg_lists:
    return instance_traits

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
            return_trait = SpecialNames.get( type_name.lower() )
            if return_trait is None:
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
        if isinstance(events, basestring):
            events = [ events ]
        for event in events:
            handlers.append( _get_method( cls, '_%s_changed' % event ) )
            handlers.append( _get_method( cls, '_%s_fired'   % event ) )

#-------------------------------------------------------------------------------
#  Returns the method associated with a particular class property getter/setter:
#-------------------------------------------------------------------------------

def _property_method ( class_dict, name ):
    return class_dict.get( name )

#-------------------------------------------------------------------------------
#  Defines a factory function for creating type checked methods:
#-------------------------------------------------------------------------------

def trait_method ( func, return_type, **arg_types ):
    """ Factory function for creating type-checked methods.

    Parameters
    ----------
    func : function
        The method to be type-checkd
    return_type : trait or a value that can be converted to a trait using Trait()
        The return type of the method
    arg_types : zero or more '*keyword* = *trait*' pairs
        Argument names and types of parameters of the type-checked method. The
        *trait* portion of each pair must be a trait or a value that can be
        converted to a trait using Trait().
    """
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
    """ Declares that the method defined immediately following a call to this
    function is type-checked.

    Parameters
    ----------
    return_type : type
        The type returned by the type-checked method. Must be either a trait
        or a value that can be converted to a trait using the Trait()
        function. The default of Any means that the return value is not
        type-checked.
    arg_types : zero or more types
        The types of positional parameters of the type-checked method. Each
        value must either a trait or a value that can be converted to a trait
        using the Trait()  function.
    kwarg_types : zero or more *keyword* = *type* pairs
        Type names and types of keyword parameters of the type-checked method.
        The *type* portion of the parameter must be either a trait or a value
        that can be converted to a trait using the Trait() function.

    Description
    -----------
    Whenever the type-checked method is called, the method() function ensures
    that each parameter passed to the method of the type specified by
    *arg_types* and *kwarg_types*, and that the return value is of the type
    specified by *return_type*. It is an error to specify both positional and
    keyword definitions for the same method parameter. If a parameter defined by
    the type-checked method is not referenced in the method() call, the
    parameter is not type-checked (i.e., its type is implicitly set to Any).
    If the call to method() signature contains an *arg_types* or *kwarg_types*
    parameter that does not correspond to a parameter in the type-checked method
    definition, a TraitError exception is raised.
    """
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

# This really should be 'HasTraits', but it's not defined yet:
_HasTraits = None

class MetaHasTraits ( type ):
    ### JMS: Need a docstring here.
    # All registered class creation listeners.
    #
    # { Str class_name : Callable listener }
    _listeners = {}

    def __new__ ( cls, class_name, bases, class_dict ):
        MetaHasTraitsObject( cls, class_name, bases, class_dict, False )

        # Finish building the class using the updated class dictionary:
        klass = type.__new__( cls, class_name, bases, class_dict )
        if _HasTraits is not None:
            for base in bases:
                if issubclass( base, _HasTraits ):
                    getattr( base, SubclassTraits ).append( klass )
        setattr( klass, SubclassTraits, [] )

        # Call all listeners that registered for this specific class:
        name = '%s.%s' % ( klass.__module__, klass.__name__ )
        for listener in MetaHasTraits._listeners.get( name, [] ):
            listener( klass )

        # Call all listeners that registered for ANY class:
        for listener in MetaHasTraits._listeners.get( '', [] ):
            listener( klass )

        return klass

    def add_listener ( cls, listener, class_name = '' ):
        """ Adds a class creation listener.

        If the class name is the empty string then the listener will be called
        when *any* class is created.
        """
        MetaHasTraits._listeners.setdefault( class_name, [] ).append( listener )

    add_listener = classmethod( add_listener )

    def remove_listener ( cls, listener, class_name = '' ):
        """ Removes a class creation listener.
        """
        MetaHasTraits._listeners[ class_name ].remove( listener )

    remove_listener = classmethod( remove_listener )

#-------------------------------------------------------------------------------
#  'MetaHasTraitsObject' class:
#-------------------------------------------------------------------------------

class MetaHasTraitsObject ( object ):
    ### JMS: Need a docstring here.
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

        # Create a list of just those base classes that derive from HasTraits:
        hastraits_bases = [ base for base in bases
                            if base.__dict__.get( ClassTraits ) is not None ]

        # Create a list of all inherited trait dictionaries:
        inherited_class_traits = [ base.__dict__.get( ClassTraits )
                                   for base in hastraits_bases ]

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
                            validate, True, value.handler, **value.metadata )
            if rc:
                del class_dict[ name ]
                if name[-1:] != '_':
                    base_traits[ name ] = class_traits[ name ] = value
                    handler = value.handler
                    if handler is not None:
                        if handler.has_items:
                            items_trait = handler.items_event()
                            if value.instance_handler=='_list_changed_handler':
                                items_trait = _clone_trait( items_trait )
                                items_trait.instance_handler = \
                                    '_list_items_changed_handler'
                            class_traits[ name + '_items' ] = items_trait
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
            else:
                for ct in inherited_class_traits:
                    if name in ct:
                        class_traits[ name ] = value = ct[ name ]( value )
                        del class_dict[ name ]
                        handler = value.handler
                        if (handler is not None) and handler.is_mapped:
                            class_traits[ name + '_' ] = _mapped_trait_for(
                                                                         value )
                        break

        # Process all HasTraits base classes:
        migrated_properties = {}
        for base in hastraits_bases:
            base_dict = base.__dict__

            # Merge base traits:
            for name, value in base_dict.get( BaseTraits ).items():
                if name not in base_traits:
                    property_info = value.property()
                    if property_info is not None:
                        key = id( value )
                        migrated_properties[ key ] = value = \
                            self.migrate_property( name, value, property_info,
                                                   class_dict )
                    base_traits[ name ] = value
                elif is_category:
                    raise TraitError, ("Cannot override '%s' trait "
                                       "definition in a category" % name)

            # Merge class traits:
            for name, value in base_dict.get( ClassTraits ).items():
                if name not in class_traits:
                    property_info = value.property()
                    if property_info is not None:
                        new_value = migrated_properties.get( id( value ) )
                        if new_value is not None:
                            value = new_value
                        else:
                            value = self.migrate_property( name, value,
                                                     property_info, class_dict )
                    class_traits[ name ] = value
                elif is_category:
                    raise TraitError, ("Cannot override '%s' trait "
                                       "definition in a category" % name)

            # Merge prefix traits:
            base_prefix_traits = base_dict.get( PrefixTraits )
            for name in base_prefix_traits['*']:
                if name not in prefix_list:
                    prefix_list.append( name )
                    prefix_traits[ name ] = base_prefix_traits[ name ]
                elif is_category:
                    raise TraitError, ("Cannot override '%s_' trait "
                                       "definition in a category" % name)

            # If the base class has a 'ViewElements' object defined, add it to
            # the 'parents' list of this class's 'ViewElements':
            parent_view_elements = base_dict.get( ViewTraits )
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

        # Get the list of all possible 'Instance'/'List(Instance)' handlers:
        instance_traits = _get_instance_handlers( class_dict, hastraits_bases )

        # If there is an 'anytrait_changed' event handler, wrap it so that
        # it can be attached to all traits in the class:
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
        cloned = {}
        for name in class_traits.keys():
            trait    = class_traits[ name ]
            handlers = [ anytrait,
                         _get_def( class_name, class_dict, bases,
                                   '_%s_changed' % name ),
                         _get_def( class_name, class_dict, bases,
                                   '_%s_fired' % name ) ]

            # Check for an 'Instance' or 'List(Instance)' trait with defined
            # handlers:
            instance_handler = trait.instance_handler
            if ((instance_handler is not None) and
                (name in instance_traits) or
                ((instance_handler == '_list_items_changed_handler') and
                 (name[-6:] == '_items') and
                 (name[:-6] in instance_traits))):
                handlers.append( getattr( HasTraits, instance_handler ) )

            events = trait.event
            if events is not None:
                if isinstance(events, basestring):
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
                if name not in cloned:
                    cloned[ name ] = None
                    class_traits[ name ] = trait = _clone_trait( trait )
                if len( handlers ) > 0:
                    _add_notifiers( trait._notifiers( 1 ), handlers )
                if default is not None:
                    trait.default_value( 8, default )

            # Handle the case of properties whose value depends upon the value
            # of other traits:
            if (trait.type == 'property') and (trait.depends_on is not None):
                cached = trait.cached
                if cached is True:
                    cached = '_' + name
                depends_on = trait.depends_on
                if isinstance( depends_on, basestring ):
                    depends_on = [ depends_on ]
                for name2 in depends_on:
                    dot_name = None
                    col      = name2.find( '.' )
                    if col >= 0:
                        dot_name = name2[ col + 1: ].strip()
                        name2    = name2[ : col ].strip()
                    trait2 = class_traits.get( name2 )
                    if trait2 is not None:
                        if name2 not in cloned:
                            cloned[ name2 ] = None
                            class_traits[ name2 ] = trait2 = _clone_trait(
                                                                        trait2 )
                        handlers = [ self._property_dependency_handler( name,
                                                            cached, dot_name ) ]
                        _add_notifiers( trait2._notifiers( 1 ), handlers )

                        # Handle the case of a list dependency:
                        if dot_name is None:
                            handler = trait2.handler
                            if ((handler is not None) and
                                (handler.default_value_type == 5)):
                                name2 += '_items'
                                trait2 = class_traits.get( name2 )
                                if trait2 is not None:
                                    if name2 not in cloned:
                                        cloned[ name2 ] = None
                                        class_traits[ name2 ] = trait2 = \
                                            _clone_trait( trait2 )
                                    _add_notifiers( trait2._notifiers( 1 ),
                                        handlers )

        # Add the traits meta-data to the class:
        self.add_traits_meta_data(
            bases, class_dict, base_traits, class_traits, instance_traits,
            prefix_traits, view_elements )

    #---------------------------------------------------------------------------
    #  Returns a property dependency handler:
    #---------------------------------------------------------------------------

    def _property_dependency_handler ( self, name, cached, dot_name ):
        if cached:
            def handler ( object ):
                old = getattr( object, name )
                setattr( object, cached, None )
                object.trait_property_changed( name, old,
                                               getattr( object, name ) )
        else:
            def handler ( object ):
                object.trait_property_changed( name, None,
                                               getattr( object, name ) )

        if dot_name is None:
            return handler

        return self._property_dependency_handler2( name, handler, dot_name,
                                                   cached )

    def _property_dependency_handler2 ( self, name, inner_handler, dot_name,
                                        cached ):
        def handler ( object, ignore, old, new ):
            inner_handler( object )
            object._trait_property_dependency( name, old, new, dot_name,
                                               cached )

        return handler

    #---------------------------------------------------------------------------
    #  Adds the traits meta-data to the class:
    #---------------------------------------------------------------------------

    def add_traits_meta_data ( self, bases, class_dict, base_traits,
                               class_traits,  instance_traits, prefix_traits,
                               view_elements ):
        """ Adds the Traits metadata to the class dictionary.
        """
        class_dict[ BaseTraits     ] = base_traits
        class_dict[ ClassTraits    ] = class_traits
        class_dict[ InstanceTraits ] = instance_traits
        class_dict[ PrefixTraits   ] = prefix_traits
        class_dict[ ViewTraits     ] = view_elements

    #---------------------------------------------------------------------------
    #  Migrates an existing property to the class being defined (allowing for
    #  method overrides):
    #---------------------------------------------------------------------------

    def migrate_property ( self, name, property, property_info, class_dict ):
        """ Migrates an existing property to the class being defined
        (allowing for method overrides).
        """
        get = _property_method( class_dict, '_get_' + name )
        set = _property_method( class_dict, '_set_' + name )
        val = _property_method( class_dict,
                                '_validate_' + name )
        if ((get is not None) or (set is not None) or (val is not None)):
            old_get, old_set, old_val = property_info
            return Property( get or old_get, set or old_set, val or old_val,
                             True, **property.__dict__ )
        return property

#-------------------------------------------------------------------------------
#  Manages the list of trait instance monitors:
#-------------------------------------------------------------------------------

def _trait_monitor_index ( cls, handler ):
    global _HasTraits_monitors
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
#  'PropertyDependency' class:
#-------------------------------------------------------------------------------

class PropertyDependency ( object ):
    ### JMS: Need a docstring here.
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, obj, name, cached ):
        self.obj    = obj
        self.name   = name
        self.cached = cached

    #---------------------------------------------------------------------------
    #  Handles a value the property depends on being changed:
    #---------------------------------------------------------------------------

    def _changed ( self ):
        old = None
        obj = self.obj
        if self.cached is not None:
            old = getattr( obj, self.name )
            setattr( obj, self.cached, None )
        obj.trait_property_changed( self.name, old, getattr( obj, self.name ) )

    #---------------------------------------------------------------------------
    #  Sets up a listener:
    #---------------------------------------------------------------------------

    def set_listener ( self, obj, name, remove ):
        """ Sets up a listener.
        """
        if obj is not None:
            obj.on_trait_change( self._changed, name, remove = remove )
            if obj._is_list_trait( name ):
                obj.on_trait_change( self._changed, name + '_items',
                                     remove = remove )

#-------------------------------------------------------------------------------
#  'HasTraits' class:
#-------------------------------------------------------------------------------

class HasTraits ( CHasTraits ):
    """ Enables any Python class derived from it to have trait atttributes.

    Most of the methods of HasTraits operated by default only on the trait
    attributes explicitly defined in the class definition. They do not operate
    on trait attributes defined by way of wildcards or by calling
    **add_trait()**.
    For example::

        >>>class Person(HasTraits):
        ...    name = Str
        ...    age  = Int
        ...    temp_ = Any
        >>>bob = Person()
        >>>bob.temp_lunch = 'sandwich'
        >>>bob.add_trait('favorite_sport', Str('football'))
        >>>print bob.trait_names()
        ['trait_added', 'age', 'name']

    In this example, the trait_names() method returns only the *age* and
    *name* attributes defined on the Person class. (The **trait_added**
    attribute is an explicit trait event defined on the HasTraits class.)
    The wildcard attribute *temp_lunch* and the dynamically-added trait
    attribute *favorite_sport* are not listed.
    """
    __metaclass__ = MetaHasTraits

    #-- Class Variables --------------------------------------------------------

    # Mapping from dispatch type to notification wrapper class type:
    wrappers = {
        'same': TraitChangeNotifyWrapper,
        'ui':   UITraitChangeNotifyWrapper,
        'new':  NewTraitChangeNotifyWrapper
    }

    #-- Trait Definitions ------------------------------------------------------

    # An event fired when a new trait is dynamically added to the object:
    trait_added = Event( basestring )

    # An event that can be fired to indicate that the state of the object has
    # been modified:
    trait_modified = Event

    #---------------------------------------------------------------------------
    #  Adds/Removes a trait instance creation monitor:
    #---------------------------------------------------------------------------

    def trait_monitor ( cls, handler, remove = False ):
        """Adds or removes the specified *handler* from the list of active
        monitors.

        Parameters
        ----------
        handler : function
            The function to add or remove as a monitor.
        remove : boolean
            Flag indicating whether to remove (True) or add the specified
            handler as a monitor for this class.

        Description
        -----------
        If *remove* is omitted or False, the specified handler is added to
        the list of active monitors; if *remove* is True, the handler is
        removed from the active monitor list.

        """
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
        """ Adds a named trait attribute to this class.

        Parameters
        ----------
        name : string
            Name of the attribute to add
        trait : a trait or a value that can be converted to a trait using Trait()
            Trait definition of the attribute. It can be a single value or
            a list equivalent to an argument list for the Trait() function

        """

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
        handlers = [ _get_method( cls, '_%s_changed' % name ),
                     _get_method( cls, '_%s_fired'   % name ) ]

        # Add any special trait defined event handlers:
        _add_event_handlers( trait, cls, handlers )

        # Add the 'anytrait' handler (if any):
        handlers.append( prefix_traits.get( '@' ) )

        # Filter out any 'None' values:
        handlers = [ h for h in handlers if h is not None ]

        # If there are and handlers, add them to the trait's notifier's list:
        if len( handlers ) > 0:
            trait = _clone_trait( trait )
            _add_notifiers( trait._notifiers( 1 ), handlers )

        # Finally, add the new trait to the class trait dictionary:
        class_traits[ name ] = trait

    _add_class_trait = classmethod( _add_class_trait )

    #---------------------------------------------------------------------------
    #  Adds a 'category' to the class:
    #---------------------------------------------------------------------------

    def add_trait_category ( cls, category ):
        """ Adds a trait category to a class.
        """
        if issubclass( category, HasTraits ):
            cls._add_trait_category( getattr( category, BaseTraits ),
                                     getattr( category, ClassTraits ),
                                     getattr( category, InstanceTraits ),
                                     getattr( category, PrefixTraits ),
                                     getattr( category, ViewTraits, None ) )

        # Copy all methods that are not already in the class from the category:
        for subcls in category.__mro__:
            for name, value in subcls.__dict__.items():
                if not hasattr( cls, name ):
                    setattr( cls, name, value )

    add_trait_category = classmethod( add_trait_category )

    #---------------------------------------------------------------------------
    #  Adds a 'category' to the class:
    #---------------------------------------------------------------------------

    def _add_trait_category ( cls, base_traits, class_traits, instance_traits,
                                   prefix_traits, view_elements ):
        # Update the class and each of the existing subclasses:
        for subclass in [ cls ] + cls.trait_subclasses( True ):

            subclass_traits = getattr( subclass, BaseTraits )
            for name, value in base_traits.items():
                subclass_traits.setdefault( name, value )

            subclass_traits = getattr( subclass, ClassTraits )
            for name, value in class_traits.items():
                subclass_traits.setdefault( name, value )

            subclass_traits = getattr( subclass, InstanceTraits )
            for name, arg_lists in instance_traits.items():
                subclass_arg_lists = subclass_traits.get( name )
                if subclass_arg_lists is None:
                    subclass_traits[ name ] = arg_lists[:]
                else:
                    for arg_list in arg_lists:
                        if arg_list not in subclass_arg_lists:
                            subclass_arg_lists.append( arg_list )

            subclass_traits = getattr( subclass, PrefixTraits )
            subclass_list   = subclass_traits['*']
            changed         = False
            for name, value in prefix_traits.items():
                if name not in subclass_traits:
                    subclass_traits[ name ] = value
                    subclass_list.append( name )
                    changed = True

            # Resort the list from longest to shortest (if necessary):
            if changed:
                subclass_list.sort( lambda x, y: len( y ) - len( x ) )

        # Copy all our new view elements into the base class's ViewElements:
        if view_elements is not None:
            content = view_elements.content
            if len( content ) > 0:
                base_ve = getattr( cls, ViewTraits, None )
                if base_ve is None:
                    base_ve = ViewElements()
                    setattr( cls, ViewTraits, base_ve )
                base_ve_content = base_ve.content
                for name, value in content.items():
                    base_ve_content.setdefault( name, value )

    _add_trait_category = classmethod( _add_trait_category )

    #---------------------------------------------------------------------------
    #  Sets a trait notification dispatch handler:
    #---------------------------------------------------------------------------

    def set_trait_dispatch_handler ( cls, name, klass, override = False ):
        """ Sets a trait notification dispatch handler.
        """
        try:
            if issubclass( klass, TraitChangeNotifyWrapper ):
                if (not override) and (name in cls.wrappers):
                    raise TraitError, ("A dispatch handler called '%s' has "
                                       "already been defined." % name)
                cls.wrappers[ name ] = klass
                return
        except TypeError:
            pass
        raise TraitError, ('%s is not a subclass of TraitChangeNotifyWrapper.' %
                           klass)

    set_trait_dispatch_handler = classmethod( set_trait_dispatch_handler )

    #---------------------------------------------------------------------------
    #  Returns the immediate (or all) subclasses of this class:
    #---------------------------------------------------------------------------

    def trait_subclasses ( cls, all = False ):
        """ Returns a list of the immediate (or all) subclasses of this class.

        Parameters
        ----------
        all : Boolean
            Indicates whether to return all subclasses of this class. If
            False, only immediate subclasses are returned.

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
    #  Prepares an object to be pickled:
    #---------------------------------------------------------------------------

    def __getstate__ (self):
        """ Returns dictionary of traits to pickle.

        Subclasses should override __getstate__ and use the follow pattern
        to remove items that should not be persisted::

            def __getstate__(self):
                state = super(X,self).__getstate__()
                for key in ['foo', 'bar']:
                    if state.has_key(key):
                        del state[key]
                return state
        """
        state   = self.__dict__.copy()
        restore = []
        for name in state.keys():
            if ((len( name ) > 1) and (name[-1] == '_') and
                (name[:-1] in state)):
                try:
                    dumps( state[ name ] )
                except:
                    del state[ name ]
                    restore.append( name[:-1] )
        if len( restore ) > 0:
            state[ '__HasTraits_restore__' ] = restore

        return state

    def __reduce_ex__ ( self, protocol ):
        state = self.__getstate__()
        return ( __newobj__, ( self.__class__, ), state )

    #---------------------------------------------------------------------------
    #  Restores the previously pickled state of an object:
    #---------------------------------------------------------------------------

    def __setstate__ ( self, state ):
        restore = state.get( '__HasTraits_restore__' )
        if restore is not None:
            del state[ '__HasTraits_restore__' ]
            values = [ ( name, state[ name ] ) for name in restore ]
            for name in restore:
                del state[ name ]
            self.__dict__.update( state )
            for name, value in values:
                setattr( self, name, value )
        else:
            self.__dict__.update( state )

    #---------------------------------------------------------------------------
    #  Shortcut for retrieving the value of a list of traits:
    #---------------------------------------------------------------------------

    def get ( self, *names, **metadata ):
        """ Shortcut for getting object trait attributes.

        Parameters
        ----------
        names : list of trait attribute names
            Trait attributes whose values are requested

        Returns
        -------
        A dictionary whose keys are the names passed as arguments and whose
        values are the corresponding trait values

        Description
        -----------
        Looks up the value of each trait whose name is passed as an argument
        and returns a dictionary containing the resulting name/value pairs.
        If any name does not correspond to a defined trait, it is not included
        in the result.

        If no names are specified, the result is a dictionary containing
        name/value pairs for *all* traits defined on the object.
        """

        result = {}
        n      = len( names )
        if (n == 1) and (type( names[0] ) in SequenceTypes):
            names = names[0]
        elif n == 0:
            names = self.trait_names( **metadata )
        for name in names:
            value = getattr( self, name, Missing )
            if value is not Missing:
                result[ name ] = value
        return result

    #---------------------------------------------------------------------------
    #  Shortcut for setting object traits:
    #---------------------------------------------------------------------------

    def set ( self, trait_change_notify = True, **traits ):
        """ Shortcut for setting object trait attributes.

        Parameters
        ----------
        traits : list of key/value pairs
            Trait attributes and their values to be set

        Returns
        -------
        self
            The method returns this object, after setting attributes.

        Description
        -----------
        Treats each keyword argument to the method as the name of a trait
        attribute and sets the corresponding trait attribute to the value
        specified. This is a useful shorthand when a number of trait attributes
        need to be set on an object, or a trait attribute value needs to be set
        in a lambda function. For example, you can write::

            person.set(name='Bill', age=27)

        instead of::

            person.name = 'Bill'
            person.age = 27

        """
        if not trait_change_notify:
            self._trait_change_notify( False )
            try:
                for name, value in traits.items():
                    setattr( self, name, value )
            finally:
                self._trait_change_notify( True )
        else:
            for name, value in traits.items():
                setattr( self, name, value )

        return self

    #---------------------------------------------------------------------------
    #  Resets some or all of an object's traits to their default values:
    #---------------------------------------------------------------------------

    def reset_traits ( self, traits = None, **metadata ):
        """ Resets some or all of an object's trait attributes to their default values.

        Parameters
        ----------
        traits : list of strings
            Names of trait attributes to reset

        Returns
        -------
        A list of attributes that the method was unable to reset, which is empty
        if all the attributes were successfully reset.

        Description
        -----------
        Resets each of the traits whose names are specified in the *traits* list
        to their default values. If *traits* is None or omitted, the method
        resets all explicitly-defined object trait attributes to their default
        values. Note that this does not affect wildcard trait attraibutes or
        trait attributes added via add_trait(), unless they are explicitly
        named in *traits*.

        """
        unresetable = []
        if traits is None:
            traits = self.trait_names( **metadata )
        for name in traits:
            try:
                delattr( self, name )
            except AttributeError:
                unresetable.append( name )
        return unresetable

    #---------------------------------------------------------------------------
    #  Returns the list of trait names to copy/clone by default:
    #---------------------------------------------------------------------------

    def copyable_trait_names ( self, **metadata ):
        """ Returns the list of trait names to copy or clone by default.
        """
        return self.trait_names( **metadata )

    #---------------------------------------------------------------------------
    #  Returns the list of all trait names, including implicitly defined
    #  traits:
    #---------------------------------------------------------------------------

    def all_trait_names ( self ):
        """ Returns the list of all trait names, including implicitly defined
            traits.
        """
        return self.__class_traits__.keys()

    #---------------------------------------------------------------------------
    #  Copies another object's traits into this one:
    #---------------------------------------------------------------------------

    def copy_traits ( self, other, traits = None, memo = None, copy = None,
                            **metadata ):
        """ Copies another object's trait attributes into this one.

        Parameters
        ----------
        other : object
            The object whose trait attribute values should be copied.
        traits : list of strings
            A list of names of trait attributes to copy. If None or
            unspecified, the set of names returned by trait_names() is used.
            If 'all' or an empty list, the set of names returned by
            all_trait_names() is used.
        memo : dictionary
            A dictionary of objects that have already been copied.
        copy : None | 'deep' | 'shallow'
            The type of copy to perform on any trait that does not have explicit
            'copy' metadata. A value of None means 'copy reference'.

        Returns
        -------
        A list of attributes that the method was unable to copy,
        which is empty if all the attributes were successfully copied.

        """

        if traits is None:
            traits = self.copyable_trait_names( **metadata )
        elif (traits == 'all') or (len( traits ) == 0):
            traits = self.all_trait_names()
            if memo is not None:
                memo[ 'traits_to_copy' ] = 'all'

        unassignable = []
        deferred     = []
        deep_copy    = (copy == 'deep')
        shallow_copy = (copy == 'shallow')
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
                if copy_type == 'shallow':
                    value = copy_module.copy( value )
                elif copy_type == 'ref':
                    pass
                elif (copy_type == 'deep') or deep_copy:
                    if memo is None:
                        value = copy_module.deepcopy( value )
                    else:
                        value = copy_module.deepcopy( value, memo )
                elif shallow_copy:
                    value = copy_module.copy( value )

                setattr( self, name, value )
            except:
                unassignable.append( name )

        for name in deferred:
            try:
                value     = getattr( other, name )
                copy_type = other.base_trait( name ).copy
                if copy_type == 'shallow':
                    value = copy_module.copy( value )
                elif copy_type == 'ref':
                    pass
                elif (copy_type == 'deep') or deep_copy:
                    if memo is None:
                        value = copy_module.deepcopy( value )
                    else:
                        value = copy_module.deepcopy( value, memo )
                elif shallow_copy:
                    value = copy_module.copy( value )

                setattr( self, name, value )
            except:
                unassignable.append( name )
        return unassignable

    #---------------------------------------------------------------------------
    #  Clones a new object from this one, optionally copying only a specified
    #  set of traits:
    #---------------------------------------------------------------------------

    def clone_traits ( self, traits = None, memo = None, copy = None,
                             **metadata ):
        """ Clones a new object from this one, optionally copying only a
        specified set of traits.

        Parameters
        ----------
        traits : list of strings
            The names of the trait attributes to copy.
        memo : dictionary
            A dictionary of objects that have already been copied.
        copy : None | 'deep' | 'shallow'
            The type of copy to perform on any trait that does not have explicit
            'copy' metadata. A value of None means 'copy reference'.

        Returns
        -------
        The newly cloned object.

        Description
        -----------
        Creates a new object that is a clone of the current object. If *traits*
        is None (the default), then all explicit trait attributes defined
        for this object are cloned. If *traits* is 'all' or an empty list, the
        list of traits returned by all_trait_names() is used; otherwise,
        *traits* must be a list of the names of the trait attributes to be
        cloned.
        """
        if memo is None:
            memo = {}
        memo[ 'traits_copy_mode' ] = copy
        new = self.__new__( self.__class__ )
        memo[ id( self ) ] = new
        new.copy_traits( self, traits, memo, copy, **metadata )
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

        result = self.clone_traits( memo   = memo,
                                    traits = memo.get( 'traits_to_copy' ),
                                    copy   = memo.get( 'traits_copy_mode' ) )

        return result

    #---------------------------------------------------------------------------
    #  Edits the object's traits:
    #---------------------------------------------------------------------------

    def edit_traits ( self, view       = None, parent  = None,
                            kind       = None, context = None,
                            handler    = None, id      = '',
                            scrollable = None, **args ):
        """ Displays a user interface window for editing trait attribute values.

        Parameters
        ----------
        view : view or string
            A View object (or its name) that defines a user interface for
            editing trait attribute values of the current object. If the view is
            defined as an attribute on this class, use the name of the attribute.
            Otherwise, use a reference to the view object. If this attribute is
            not specified, the View object returned by trait_view() is used.
        parent : window handle
            A user interface component to use as the parent window for the
            object's UI window.
        kind : string
            The type of user interface window to create. See the
            **enthought.traits.ui.view.kind_trait** trait for values and
            their meanings. If *kind* is unspecified or None, the **kind**
            attribute of the View object is used.
        context : object or dictionary
            A single object or a dictionary of string/object pairs, whose trait
            attributes are to be edited. If not specified, the current object is
            used.
        handler : Handler object
            A handler object used for event handling in the dialog box. If
            None, the default handler for Traits UI is used.
        id : string
            A unique ID for persisting preferences about this user interface,
            such as size and position. If not specified, no user preferences
            are saved.
        scrollable : Boolean
            Indicates whether the dialog box should be scrollable. When set to
            True, scroll bars appear on the dialog box if it is not large enough
            to display all of the items in the view at one time.
        """
        if context is None:
            context = self
        view = self.trait_view( view )
        return view.ui( context, parent, kind, self.trait_view_elements(),
                        handler, id, scrollable, args )

    #---------------------------------------------------------------------------
    #  Returns the default context to use for editing/configuring traits:
    #---------------------------------------------------------------------------

    def trait_context ( self ):
        """ Returns the default context to use for editing or configuring traits.
        """
        return { 'object': self }

    #---------------------------------------------------------------------------
    #  Gets or sets a ViewElement associated with an object's class:
    #---------------------------------------------------------------------------

    def trait_view ( self, name = None, view_element = None ):
        """ Gets or sets a ViewElement associated with an object's class.

        Parameters
        ----------
        name : string
            Name of a view element
        view_element : a ViewElement object
            View element to associate

        Returns
        -------
        A view element.

        Description
        -----------
        If both *name* and *view_element* are specified, the view element is
        associated with *name* for the current object's class. (That is,
        *view_element* is added to the ViewElements object associated with
        the current object's class, indexed by *name*.)

        If only *name* is specified, the function returns the view element
        object associated with *name*, or None if *name* has no associated
        view element. View elements retrieved by this function are those that
        are bound to a class attribute in the class definition, or that are
        associated with a name by a previous call to this method.

        If neither *name* nor *view_element* is specified, the method returns a
        View object, based on the following order of preference:

        1. If there is a View object named ``traits_view`` associated with the
           current object, it is returned.
        2. If there is exactly one View object associated the current
           object, it is returned.
        3. Otherwise, it returns a View object containing items for all the
           non-event trait attributes on the current object.

        """
        return self.__class__._trait_view( name, view_element,
                            self.default_traits_view, self.trait_view_elements,
                            self.editable_traits )

    def class_trait_view ( cls, name = None, view_element = None ):
        return cls._trait_view( name, view_element,
                  cls.class_default_traits_view, cls.class_trait_view_elements,
                  cls.class_editable_traits )

    class_trait_view = classmethod( class_trait_view )

    #---------------------------------------------------------------------------
    #  Gets or sets a ViewElement associated with an object's class:
    #---------------------------------------------------------------------------

    def _trait_view ( cls, name, view_element, default_name, view_elements,
                           editable_traits ):
        """ Gets or sets a ViewElement associated with an object's class.
        """
        # If a view element was passed instead of a name or None, return it:
        if isinstance( name, ViewElement ):
            return name

        # Get the ViewElements object associated with the class:
        view_elements = view_elements()

        # The following test should only succeed for objects created before
        # traits has been fully initialized (such as the default Handler):
        if view_elements is None:
            return None

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
        name = default_name()

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
        from enthought.traits.ui.api import View
        return View( editable_traits() )

    _trait_view = classmethod( _trait_view )

    #---------------------------------------------------------------------------
    #  Return the default traits view/name:
    #---------------------------------------------------------------------------

    def default_traits_view ( self ):
        """ Returns the name of the default traits view for the object's class.
        """
        return self.__class__.class_default_traits_view()

    #---------------------------------------------------------------------------
    #  Return the default traits view/name:
    #---------------------------------------------------------------------------

    def class_default_traits_view ( cls ):
        """ Returns the name of the default traits view for the class.
        """
        return DefaultTraitsView

    class_default_traits_view = classmethod( class_default_traits_view )

    #---------------------------------------------------------------------------
    #  Gets the list of names of ViewElements associated with the object's
    #  class that are of a specified ViewElement type:
    #---------------------------------------------------------------------------

    def trait_views ( self, klass = None ):
        """ Returns a list of the names of all view elements associated with the
        current object's class.

        Parameters
        ----------
        klass : a class
            A class, such that all returned names must correspond to instances
            of this class. Possible values include:

            * Group
            * Item
            * View
            * ViewElement
            * ViewSubElement

        Description
        -----------
        If *klass* is specified, the list of names is filtered such that only
        objects that are instances of the specified class are returned.
        """
        return self.__class__.__dict__[ ViewTraits ].filter_by( klass )

    #---------------------------------------------------------------------------
    #  Returns the ViewElements object associated with the object's class:
    #---------------------------------------------------------------------------

    def trait_view_elements ( self ):
        """ Returns the ViewElements object associated with the object's
        class.

        The returned object can be used to access all the view elements
        associated with the class.
        """
        return self.__class__.class_trait_view_elements()

    def class_trait_view_elements ( cls ):
        """ Returns the ViewElements object associated with the class.

        The returned object can be used to access all the view elements
        associated with the class.
        """
        return cls.__dict__[ ViewTraits ]

    class_trait_view_elements = classmethod( class_trait_view_elements )

    #---------------------------------------------------------------------------
    #  Configure the object's traits:
    #---------------------------------------------------------------------------

    def configure_traits ( self, filename   = None, view       = None,
                                 kind       = None, edit       = True,
                                 context    = None, handler    = None,
                                 id         = '',   scrollable = None, **args ):
        ### JMS: Is it correct to assume that nonmodel options for 'kind'
        ###      behave modally when called from this method?
        """Creates and displays a dialog box for editing values of trait
        attributes, as if it were a complete, self-contained GUI application.

        Parameters
        ----------
        filename : string
            The name (including path) of a file that contains a pickled
            representation of the current object. When this parameter is
            specified, the method reads the corresponding file (if it exists)
            to restore the saved values of the object's traits before displaying
            them. If the user confirms the dialog box (by clicking **OK**),
            the new values are written to the file. If this parameter is not
            specified, the values are loaded from the in-memory object, and are
            not persisted when the dialog box is closed.
        view : view or string
            A View object (or its name) that defines a user interface for
            editing trait attribute values of the current object. If the view is
            defined as an attribute on this class, use the name of the attribute.
            Otherwise, use a reference to the view object. If this attribute is
            not specified, the View object returned by trait_view() is used.
        kind : string
            The type of user interface window to create. See the
            **enthought.traits.ui.view.kind_trait** trait for values and
            their meanings. If *kind* is unspecified or None, the **kind**
            attribute of the View object is used.
        edit : Boolean
            Indicates whether to display a user interface. If *filename*
            specifies an existing file, setting *edit* to False loads the
            saved values from that file into the object without requiring
            user interaction.
        context : object or dictionary
            A single object or a dictionary of string/object pairs, whose trait
            attributes are to be edited. If not specified, the current object is
            used
        handler : Handler object
            A handler object used for event handling in the dialog box. If
            None, the default handler for Traits UI is used.
        id : string
            A unique ID for persisting preferences about this user interface,
            such as size and position. If not specified, no user preferences
            are saved.
        scrollable : Boolean
            Indicates whether the dialog box should be scrollable. When set to
            True, scroll bars appear on the dialog box if it is not large enough
            to display all of the items in the view at one time.

        Description
        -----------
        This method is intended for use in applications that do not normally
        have a GUI. Control does not resume in the calling application until
        the user closes the dialog box.

        The method attempts to open and unpickle the contents of *filename*
        before displaying the dialog box. When editing is complete, the method
        attempts to pickle the updated contents of the object back to *filename*.
        If the file referenced by *filename* does not exist, the object is not
        modified before displaying the dialog box. If *filename* is unspecified
        or None, no pickling or unpickling occurs.

        If *edit* is True (the default), a dialog box for editing the
        current object is displayed. If *edit* is False or None, no
        dialog box is displayed. You can use ``edit=False`` if you want the
        object to be restored from the contents of *filename*, without being
        modified by the user.
        """
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
            from enthought.traits.ui.api import toolkit
            if context is None:
                context = self
            rc = toolkit().view_application( context, self.trait_view( view ),
                                           kind, handler, id, scrollable, args )
            if rc and (filename is not None):
                fd = None
                try:
                    import cPickle
                    fd = open( filename, 'wb' )
                    cPickle.Pickler( fd, True ).dump( self )
                except:
                    if fd is not None:
                        fd.close()
            return rc

        return True

    #---------------------------------------------------------------------------
    #  Return the list of editable traits:
    #---------------------------------------------------------------------------

    def editable_traits ( self ):
        """Returns an alphabetically sorted list of the names of non-event
        trait attributes associated with the current object.
        """
        names = self.trait_names( type = _is_not_event )
        names.sort()
        return names

    def class_editable_traits ( cls ):
        """Returns an alphabetically sorted list of the names of non-event
        trait attributes associated with the current class.
        """
        names = cls.class_trait_names( type = _is_not_event )
        names.sort()
        return names

    class_editable_traits = classmethod( class_editable_traits )

    #---------------------------------------------------------------------------
    #  Pretty print the traits of an object:
    #---------------------------------------------------------------------------

    def print_traits ( self, show_help = False, **metadata ):
        """Prints the values of all explicitly-defined, non-event trait
        attributes on the current object, in an easily readable format.

        Parameters
        ----------
        show_help : boolean
            Indicates whether to display additional descriptive information.
        """

        if len( metadata ) > 0:
            names = self.trait_names( **metadata )
        else:
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

    def on_trait_change ( self, handler, name = None, remove = False,
                                dispatch = 'same' ):
        """Causes the object to invoke a handler whenever a trait attribute
        is modified, or removes the association.

        Parameters
        ----------
        handler : function
            A trait notification function for the *name* trait attribute.
        name : string
            The name of the trait attribute whose value changes trigger the
            notification
        remove : Boolean
            If True, removes the previously-set association between
            *handler* and *name*; if False (the default), creates the
            association.

        Description
        -----------
        Multiple handlers can be defined for the same object, or even for the
        same trait attribute on the same object. If *name* is not specified or
        is None, *handler* is invoked when any trait attribute on the
        object is changed.
        """

        if type( name ) is list:
            for name_i in name:
                self.on_trait_change( handler, name_i, remove, dispatch )
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
            notifiers.append( self.wrappers[ dispatch ]( handler, notifiers ) )

    # Make 'on_trait_event' a synonym for 'on_trait_change':
    on_trait_event = on_trait_change

    #---------------------------------------------------------------------------
    #  Synchronize the value of two traits:
    #---------------------------------------------------------------------------

    def sync_trait ( self, trait_name, object, alias = None, mutual = True,
                           remove = False ):
        """Synchronizes the value of a trait attribute on this object with a
        trait attribute on another object.

        Parameters
        ----------
        name : string
            Name of the trait attribute on this object
        object : object
            The object with which to synchronize
        alias : string
            Name of the trait attribute on *other*; if None or omitted, same
            as *name*.
        mutual : Boolean or integer
            Indicates whether synchronization is mutual (True or non-zero)
            or one-way (False or zero)
        remove : Boolean or integer
            Indicates whether sychronization is being added (False or zero)
            or removed (True or non-zero)

        Description
        -----------
        In mutual synchronization, any change to the value of the specified
        trait attribute of either object results in the same value being
        assigned to the corresponding trait attribute of the other object.
        In one-way synchronization, any change to the value of the attribute
        on this object causes the corresponding trait attribute of *object* to
        be updated, but not vice versa.
        """
        if alias is None:
            alias = trait_name

        is_list = (self._is_list_trait( trait_name ) and
                   object._is_list_trait( alias ))

        if remove:
            info = self._get_sync_trait_info()
            dic  = info.get( trait_name )
            if dic is not None:
                key = ( id( object ), alias )
                if key in dic:
                    del dic[ key ]
                    if len( dic ) == 0:
                        del info[ trait_name ]
                        self.on_trait_change( self._sync_trait_modified,
                            trait_name, remove = True )
                        if is_list:
                            self.on_trait_change(
                                self._sync_trait_items_modified,
                                trait_name + '_items', remove = True )

            if mutual:
                object.sync_trait( alias, self, trait_name, False, True )

            return

        value = ( weakref.ref( object, self._sync_trait_listener_deleted ),
                  alias )
        dic   = self._get_sync_trait_info().setdefault( trait_name, {} )
        key   = ( id( object ), alias )
        if key not in dic:
            if len( dic ) == 0:
                self.on_trait_change( self._sync_trait_modified, trait_name )
                if is_list:
                    self.on_trait_change( self._sync_trait_items_modified,
                                          trait_name + '_items' )
            dic[ key ] = value
            setattr( object, alias, getattr( self, trait_name ) )

        if mutual:
            object.sync_trait( alias, self, trait_name, False )

    def _get_sync_trait_info ( self ):
        info = getattr( self, '__sync_trait__', None )
        if info is None:
            self.__dict__[ '__sync_trait__' ] = info = {}
            info[ '' ] = {}

        return info

    def _sync_trait_modified ( self, object, name, old, new ):
        info   = self.__sync_trait__
        locked = info[ '' ]
        locked[ name ] = None
        for object, object_name in info[ name ].values():
            object = object()
            if object_name not in object._get_sync_trait_info()[ '' ]:
                try:
                    setattr( object, object_name, new )
                except:
                    pass
        del locked[ name ]

    def _sync_trait_items_modified ( self, object, name, old, event ):
        n0     = event.index
        n1     = n0 + len( event.removed )
        name   = name[:-6]
        info   = self.__sync_trait__
        locked = info[ '' ]
        locked[ name ] = None
        for object, object_name in info[ name ].values():
            object = object()
            if object_name not in object._get_sync_trait_info()[ '' ]:
                try:
                    getattr( object, object_name )[ n0: n1 ] = event.added
                except:
                    pass
        del locked[ name ]

    def _sync_trait_listener_deleted ( self, ref ):
        info = self.__sync_trait__
        for key, dic in info.items():
            if key != '':
                for name, value in dic.items():
                    if ref is value[0]:
                        del dic[ name ]
                        if len( dic ) == 0:
                            del info[ key ]

    def _is_list_trait ( self, trait_name ):
        handler = self.base_trait( trait_name ).handler
        return ((handler is not None) and (handler.default_value_type == 5))

    def _trait_property_dependency ( self, name, old, new, dot_name, cached ):
        info = self.__dict__.get( '__depends_on_traits__' )
        if info is None:
            info = self.__dict__[ '__depends_on_traits__' ] = {}
        pd = info.get( name )
        if pd is None:
            pd = info[ name ] = PropertyDependency( self, name, cached )
        pd.set_listener( old, dot_name, True )
        pd.set_listener( new, dot_name, False )

    #---------------------------------------------------------------------------
    #  Add a new trait:
    #---------------------------------------------------------------------------

    def add_trait ( self, name, *trait ):
        """Adds a trait attribute to this object.

        Parameters
        ----------
        name : string
            Name of the attribute to add
        trait : trait or a value that can be converted to a trait by Trait()
            Trait definition for *name*. If more than one value is specified,
            it is equivalent to passing the entire list of values to Trait().

        """

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
            cls      = self.__class__
            handlers = [ _get_method( cls, '_%s_changed' % name ),
                         _get_method( cls, '_%s_fired'   % name ) ]

            # Add any special trait defined event handlers:
            _add_event_handlers( trait, cls, handlers )

            # Add the 'anytrait' handler (if any):
            handlers.append( self.__prefix_traits__.get( '@' ) )

            # Filter out any 'None' values:
            handlers = [ h for h in handlers if h is not None ]

            # If there are any static notifiers, attach them to the trait:
            if len( handlers ) > 0:
                _add_notifiers( trait._notifiers( 1 ), handlers )

        # If this was a new trait, fire the 'trait_added' event:
        if old_trait is None:
            self.trait_added = name

    #---------------------------------------------------------------------------
    #  Remove an existing trait:
    #---------------------------------------------------------------------------

    def remove_trait ( self, name ):
        """Removes a trait attribute from this object.

        Parameters
        ----------
        name : string
            Name of the attribute to remove
        """
        # Get the trait definition:
        trait = self._trait( name, 0 )
        if trait is not None:

            # Check to see if the trait has additional sub-traits that need to
            # be removed also:
            handler = trait.handler
            if handler is not None:
                if handler.has_items:
                    self.remove_trait( name + '_items' )
                if handler.is_mapped:
                    self.remove_trait( name + '_' )

            # Remove the trait value from the object dictionary as well:
            if name in self.__dict__:
                del self.__dict__[ name ]

            # Get the object's instance trait dictionary and remove the trait
            # from it:
            itrait_dict = self._instance_traits()
            if name in itrait_dict:
                del itrait_dict[ name ]
                return True

        return False

    #---------------------------------------------------------------------------
    #  Returns the trait definition of a specified trait:
    #---------------------------------------------------------------------------

    def trait ( self, name, force = False, copy = False ):
        """Returns the trait definition for the *name* trait attribute.

        Parameters
        ----------
        name : string
            Name of the attribute whose trait definition is to be returned
        force : Boolean
            Indicates whether to return a trait definition if *name* is
            not explicitly defined
        copy : Boolean
            Indicates whether to return the original trait definition or a copy

        Description
        -----------
        If *force* is False (the default) and *name* is the name of an
        implicitly defined trait attribute that has never been referenced
        explicitly (i.e., has not yet been defined), the result is None. In
        all other cases, the result is the trait definition object associated
        with *name*.

        If *copy* is True, and a valid trait definition is found for *name*,
        a copy of the trait found is returned. In all other cases, the trait
        definition found is returned unmodified (the default).
        """
        mode = 0
        if force:
            mode = -1
        result = self._trait( name, mode )
        if (not copy) or (result is None):
            return result

        return  _clone_trait( result )

    #---------------------------------------------------------------------------
    #  Returns the base trait definition of a specified trait:
    #---------------------------------------------------------------------------

    def base_trait ( self, name ):
        """Returns the base trait definition for a trait attribute.

        Parameters
        ----------
        name : string
            Name of the attribute whose trait definition is returned.

        Description
        -----------
        This method is similar to the trait() method, and returns a
        different result only in the case where the trait attribute defined by
        *name* is a delegate. In this case, the base_trait() method follows the
        delegation chain until a non-delegated trait attribute is reached, and
        returns the definition of that attribute's trait as the result.
        """
        return self._trait( name, -2 )

    #---------------------------------------------------------------------------
    #  Validates whether or not a specified value is legal for a specified
    # trait and returns the validated value if valid:
    #---------------------------------------------------------------------------

    def validate_trait ( self, name, value ):
        """ Validates whether a value is legal for a trait.

        Returns the validated value if it is valid.
        """
        return self.base_trait( name ).validate( self, name, value )

    #---------------------------------------------------------------------------
    #  Return a dictionary of all traits which match a set of metadata:
    #---------------------------------------------------------------------------

    def traits ( self, **metadata ):
        """Returns a dictionary containing the definitions of all of the trait
        attributes of this object that match the set of *metadata* criteria.

        Parameters
        ----------
        metadata : dictionary
            Criteria for selecting trait attributes

        Description
        -----------
        The keys of the returned dictionary are the trait attribute names, and
        the values are their corresponding trait definition objects.

        If no *metadata* information is specified, then all explicitly defined
        trait attributes defined for the object are returned.

        Otherwise, the *metadata* keyword dictionary is assumed to define a set
        of search criteria for selecting trait attributes of interest. The
        *metadata* dictionary keys correspond to the names of trait metadata
        attributes to examine, and the values correspond to the values the
        metadata attribute must have in order to be included in the search
        results.

        The *metadata* values either may be simple Python values like strings or
        integers, or may be lambda expressions or functions that return True
        if the trait attribute is to be included in the result. A lambda
        expression or function must receive a single argument, which is the
        value of the trait metadata attribute being tested. If more than one
        metadata keyword is specified, a trait attribute must match the metadata
        values of all keywords to be included in the result.
        """
        traits = self.__base_traits__.copy()
        for name in self.__dict__.keys():
            if name not in traits:
                trait = self.trait( name )
                if trait is not None:
                    traits[ name ] = trait

        if len( metadata ) == 0:
            return traits

        for meta_name, meta_eval in metadata.items():
            if type( meta_eval ) is not FunctionType:
                metadata[ meta_name ] = _SimpleTest( meta_eval )

        result = {}
        for name, trait in traits.items():
            for meta_name, meta_eval in metadata.items():
                if not meta_eval( getattr( trait, meta_name ) ):
                    break
            else:
                result[ name ] = trait

        return result

    #---------------------------------------------------------------------------
    #  Return a dictionary of all traits which match a set of metadata:
    #---------------------------------------------------------------------------

    def class_traits ( cls, **metadata ):
        """Returns a dictionary containing the definitions of all of the trait
        attributes of the class that match the set of *metadata* criteria.

        Parameters
        ----------
        metadata : dictionary
            Criteria for selecting trait attributes

        Description
        -----------
        The keys of the returned dictionary are the trait attribute names, and
        the values are their corresponding trait definition objects.

        If no *metadata* information is specified, then all explicitly defined
        trait attributes defined for the class are returned.

        Otherwise, the *metadata* keyword dictionary is assumed to define a set
        of search criteria for selecting trait attributes of interest. The
        *metadata* dictionary keys correspond to the names of trait metadata
        attributes to examine, and the values correspond to the values the
        metadata attribute must have in order to be included in the search
        results.

        The *metadata* values either may be simple Python values like strings or
        integers, or may be lambda expressions or functions that return **True**
        if the trait attribute is to be included in the result. A lambda
        expression or function must receive a single argument, which is the
        value of the trait metadata attribute being tested. If more than one
        metadata keyword is specified, a trait attribute must match the metadata
        values of all keywords to be included in the result.
        """
        if len( metadata ) == 0:
            return cls.__base_traits__.copy()

        result = {}

        for meta_name, meta_eval in metadata.items():
            if type( meta_eval ) is not FunctionType:
                metadata[ meta_name ] = _SimpleTest( meta_eval )

        for name, trait in cls.__base_traits__.items():
            for meta_name, meta_eval in metadata.items():
                if not meta_eval( getattr( trait, meta_name ) ):
                    break
            else:
                result[ name ] = trait

        return result

    class_traits = classmethod( class_traits )

    #---------------------------------------------------------------------------
    #  Return a list of all trait names which match a set of metadata:
    #---------------------------------------------------------------------------

    def trait_names ( self, **metadata ):
        """Returns a list of the names of all trait attributes whose definitions
        match the set of *metadata* criteria specified.

        Parameters
        ----------
        metadata : dictionary
            Criteria for selecting trait attributes

        Description
        -----------
        This method is similar to the traits() method, but returns only the
        names of the matching trait attributes, not the trait definitions.
        """
        return self.traits( **metadata ).keys()

    def class_trait_names ( cls, **metadata ):
        """Returns a list of the names of all trait attributes whose definitions
        match the set of *metadata* criteria specified.

        Parameters
        ----------
        metadata : dictionary
            Criteria for selecting trait attributes

        Description
        -----------
        This method is similar to the traits() method, but returns only the
        names of the matching trait attributes, not the trait definitions.
        """
        return cls.class_traits( **metadata ).keys()

    class_trait_names = classmethod( class_trait_names )

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
                cls      = self.__class__
                handlers = [ _get_method( cls, '_%s_changed' % name ),
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
                    trait = _clone_trait( trait )
                    _add_notifiers( trait._notifiers( 1 ), handlers )

                return trait

        # There should ALWAYS be a prefix match in the trait classes, since ''
        # is at the end of the list, so we should never get here:
        raise SystemError, ("Trait class look-up failed for attribute '%s' "
                            "for an object of type '%s'") % (
                            name, self.__class__.__name__ )

    #---------------------------------------------------------------------------
    #  Adds/Removes (Java-style) event listeners to an object:
    #---------------------------------------------------------------------------

    def add_trait_listener ( self, object, prefix = '' ):
        self._trait_listener( object, prefix, False )

    def remove_trait_listener ( self, object, prefix = '' ):
        self._trait_listener( object, prefix, True )

    def _trait_listener ( self, object, prefix, remove ):
        if prefix[-1:] != '_':
            prefix += '_'
        n      = len( prefix )
        traits = self.__base_traits__
        for name in self._each_trait_method( object ):
            if name[:n] == prefix:
                if name[-8:] == '_changed':
                    short_name = name[n:-8]
                    if short_name in traits:
                        self.on_trait_change( getattr( object, name ),
                                              short_name, remove = remove )
                    elif short_name == 'anytrait':
                        self.on_trait_change( getattr( object, name ),
                                              remove = remove )
                elif name[:-6] == '_fired':
                    short_name = name[n:-6]
                    if short_name in traits:
                        self.on_trait_change( getattr( object, name ),
                                              short_name, remove = remove )
                    elif short_name == 'anytrait':
                        self.on_trait_change( getattr( object, name ),
                                              remove = remove )

    #---------------------------------------------------------------------------
    #  Generates each (name, method) pair for a specified object:
    #---------------------------------------------------------------------------

    def _each_trait_method ( self, object ):
        """ Generates each (name, method) pair for a specified object.
        """
        dic = {}
        for klass in object.__class__.__mro__:
            for name, method in klass.__dict__.items():
                if (type( method ) is FunctionType) and (name not in dic):
                    dic[ name ] = True
                    yield name

    #---------------------------------------------------------------------------
    #  Handles adding/removing listeners for a generic 'Instance' trait:
    #---------------------------------------------------------------------------

    def _instance_changed_handler ( self, name, old, new ):
        """ Handles adding/removing listeners for a generic 'Instance' trait.
        """
        arg_lists = self._get_instance_handlers( name )

        if old is not None:
            for args in arg_lists:
                old.on_trait_change( remove = True, *args )

        if new is not None:
            for args in arg_lists:
                new.on_trait_change( *args )

    #---------------------------------------------------------------------------
    #  Handles adding/removing listeners for a generic 'List( Instance )' trait:
    #---------------------------------------------------------------------------

    def _list_changed_handler ( self, name, old, new ):
        """ Handles adding/removing listeners for a generic 'List( Instance )'
            trait.
        """
        arg_lists = self._get_instance_handlers( name )

        for item in old:
            for args in arg_lists:
                item.on_trait_change( remove = True, *args )

        for item in new:
            for args in arg_lists:
                item.on_trait_change( *args )

    def _list_items_changed_handler ( self, name, not_used, event ):
        """ Handles adding/removing listeners for a generic 'List( Instance )'
            trait.
        """
        arg_lists = self._get_instance_handlers( name[:-6] )

        for item in event.removed:
            for args in arg_lists:
                item.on_trait_change( remove = True, *args )

        for item in event.added:
            for args in arg_lists:
                item.on_trait_change( *args )

    #---------------------------------------------------------------------------
    #  Returns a list of ( name, method ) pairs for a specified 'Instance' or
    #  'List( Instance )' trait name:
    #---------------------------------------------------------------------------

    def _get_instance_handlers ( self, name ):
        """ Returns a list of ( name, method ) pairs for a specified 'Instance'
            or 'List( Instance )' trait name:
        """
        return [ ( getattr( self, method_name ), item_name )
                 for method_name, item_name in
                     self.__class__.__instance_traits__[ name ] ]

# Patch the definition of _HasTraits to be the real 'HasTraits':
_HasTraits = HasTraits

#-------------------------------------------------------------------------------
#  'HasStrictTraits' class:
#-------------------------------------------------------------------------------

class HasStrictTraits ( HasTraits ):
    """ This class guarantees that any object attribute that does not have an
    explicit or wildcard trait definition results in an exception.

    This feature can be useful in cases where a more rigorous software
    engineering approach is being used than is typical for Python programs. It
    also helps prevent typos and spelling mistakes in attribute names from
    going unnoticed; a misspelled attribute name typically causes an exception.
    """
    _ = Disallow   # Disallow access to any traits not explicitly defined

#-------------------------------------------------------------------------------
#  'HasPrivateTraits' class:
#-------------------------------------------------------------------------------

class HasPrivateTraits ( HasTraits ):
    """ This class ensures that any public object attribute that does not have
    an explicit or wildcard trait definition results in an exception, but
    "private" attributes (whose names start with '_') have an initial value of
    **None**, and are not type-checked.

    This feature is useful in cases where a class needs private attributes to
    keep track of its internal object state, which are not part of the class's
    public API. Such attributes do not need to be type-checked, because they are
    manipulated only by the (presumably correct) methods of the class itself.
    """
    # Make 'private' traits (leading '_') have no type checking:
    __ = Any( private = True )

    # Disallow access to all other traits not explicitly defined:
    _  = Disallow

#-------------------------------------------------------------------------------
#  Singleton classes with traits:
#
#  This code is based on a recipe taken from:
#      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66531
#  Specifically, the implementation of Oren Tirosh is used.
#-------------------------------------------------------------------------------

class SingletonHasTraits ( HasTraits ):
    """ Singleton class that support trait attributes.
    """
    def __new__ ( cls, *args, **traits ):
        if not '_the_instance' in cls.__dict__:
            cls._the_instance = HasTraits.__new__( cls, *args, **traits )
        return cls._the_instance

class SingletonHasStrictTraits ( HasStrictTraits ):
    """ Singleton class that supports strict trait attributes.

        Non-trait attributes generate an exception.
    """
    def __new__ ( cls, *args, **traits ):
        return SingletonHasTraits.__new__( cls, *args, **traits )

class SingletonHasPrivateTraits ( HasPrivateTraits ):
    """ Singleton class that supports trait attributes, with private attributes
        being unchecked.
    """
    def __new__ ( cls, *args, **traits ):
        return SingletonHasTraits.__new__( cls, *args, **traits )

#-------------------------------------------------------------------------------
#  Defines a 'vetoable' request object and an associated event:
#-------------------------------------------------------------------------------

class Vetoable ( HasStrictTraits ):

    # Should the request be vetoed? (Can only be set to 'True')
    veto = false

    def _veto_changed ( self, state ):
        self._trait_veto_notify( state )

VetoableEvent = Event( Vetoable )

#-------------------------------------------------------------------------------
#  'traits_super' class:
#-------------------------------------------------------------------------------

class traits_super ( super ):

    def __getattribute__ ( self, name ):
        try:
            return super( traits_super, self ).__getattribute__( name )
        except:
            return self._noop

    def _noop ( self, *args, **kw ):
        pass


#------------------------------------------------------------------------------
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#  Thanks for using Enthought open source!
#
#  Author: David C. Morrill
#  Original Date: 06/21/2002
#
#  Rewritten as a C-based type extension: 06/21/2004
#------------------------------------------------------------------------------
"""
Defines the 'core' traits for the Traits package. A trait is a type definition
that can be used for normal Python object attributes, giving the attributes
some additional characteristics:

Initialization:
    Traits have predefined values that do not need to be explicitly
    initialized in the class constructor or elsewhere.
Validation:
    Trait attributes have flexible, type-checked values.
Delegation:
    Trait attributes' values can be delegated to other objects.
Notification:
    Trait attributes can automatically notify interested parties when
    their values change.
Visualization:
    Trait attributes can automatically construct (automatic or
    programmer-defined) user interfaces that allow their values to be
    edited or displayed)

Note: 'trait' is a synonym for 'property', but is used instead of the
word 'property' to differentiate it from the Python language 'property'
feature.

"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import sys
import trait_handlers

from ctraits \
    import cTrait, CTraitMethod

from trait_base \
    import SequenceTypes, Self, Undefined, Missing, TypeTypes, class_of, \
           add_article, enumerate, BooleanType, get_module_name

from trait_errors \
    import TraitError

from trait_handlers \
    import TraitHandler, TraitInstance, TraitList, TraitDict, TraitFunction, \
           TraitType, TraitCastType, TraitEnum, TraitCompound, TraitMap, \
           TraitString, ThisClass, TraitRange, TraitTuple, TraitCallable, \
           TraitExpression, TraitWeakRef

from types \
    import NoneType, IntType, LongType, FloatType, ComplexType, StringType, \
           UnicodeType, ListType, TupleType, DictType, FunctionType, \
           ClassType, ModuleType, MethodType, InstanceType, TypeType

#-------------------------------------------------------------------------------
#  Editor factory functions:
#-------------------------------------------------------------------------------

PasswordEditor      = None
MultilineTextEditor = None
SourceCodeEditor    = None
HTMLTextEditor      = None
PythonShellEditor   = None

def password_editor ( ):
    """ Factory function that returns an editor for passwords.
    """
    global PasswordEditor

    if PasswordEditor is None:
        from enthought.traits.ui.api import TextEditor
        PasswordEditor = TextEditor( password = True )

    return PasswordEditor

def multi_line_text_editor ( ):
    """ Factory function that returns a text editor for multi-line strings.
    """
    global MultilineTextEditor

    if MultilineTextEditor is None:
        from enthought.traits.ui.api import TextEditor
        MultilineTextEditor = TextEditor( multi_line = True )

    return MultilineTextEditor

def code_editor ( ):
    """ Factory function that returns an editor that treats a multi-line string
    as source code.
    """
    global SourceCodeEditor

    if SourceCodeEditor is None:
        from enthought.traits.ui.api import CodeEditor
        SourceCodeEditor = CodeEditor()

    return SourceCodeEditor

def html_editor ( ):
    """ Factory function for an "editor" that displays a multi-line string as
    interpreted HTML.
    """
    global HTMLTextEditor

    if HTMLTextEditor is None:
        from enthought.traits.ui.api import HTMLEditor
        HTMLTextEditor = HTMLEditor()

    return HTMLTextEditor

def shell_editor ( ):
    """ Factory function that returns a Python shell for editing Python values.
    """
    global PythonShellEditor

    if PythonShellEditor is None:
        from enthought.traits.ui.api import ShellEditor
        PythonShellEditor = ShellEditor()

    return PythonShellEditor

#-------------------------------------------------------------------------------
#  'CTrait' class (extends the underlying cTrait c-based type):
#-------------------------------------------------------------------------------

class CTrait ( cTrait ):
    """ Extends the underlying C-based cTrait type.
    """
    #---------------------------------------------------------------------------
    #  Allows a derivative trait to be defined from this one:
    #---------------------------------------------------------------------------

    def __call__ ( self, *args, **metadata ):
        if 'parent' not in metadata:
            metadata[ 'parent' ] = self
        return Trait( *(args + ( self, )), **metadata )

    #---------------------------------------------------------------------------
    #  Returns the user interface editor associated with the trait:
    #---------------------------------------------------------------------------

    def get_editor ( self ):
        """ Returns the user interface editor associated with the trait.
        """
        from enthought.traits.ui.api import EditorFactory

        # See if we have an editor:
        editor = self.editor
        if editor is None:

            # Else see if the trait handler has an editor:
            handler = self.handler
            if handler is not None:
                editor = handler.get_editor( self )

            # If not, give up and use a default text editor:
            if editor is None:
                from enthought.traits.ui.api import TextEditor
                editor = TextEditor

        # If the result is not an EditoryFactory:
        if not isinstance( editor, EditorFactory ):
            # Then it should be a factory for creating them:
            args   = ()
            traits = {}
            if type( editor ) in SequenceTypes:
                for item in editor[:]:
                    if type( item ) in SequenceTypes:
                        args = tuple( item )
                    elif isinstance( item, dict ):
                        traits = item
                        if traits.get( 'trait', 0 ) is None:
                            traits = traits.copy()
                            traits[ 'trait' ] = self
                    else:
                        editor = item
            editor = editor( *args, **traits )

        # Cache the result:
        self.editor = editor

        # Return the resulting EditorFactory object:
        return editor

    #---------------------------------------------------------------------------
    #  Returns the help text for a trait:
    #---------------------------------------------------------------------------

    def get_help ( self, full = True ):
        """ Returns the help text for a trait.

        Parameters
        ----------
        full : Boolean
            Indicates whether to return the value of the *help* attribute of
            the trait itself.

        Description
        -----------
        If *full* is False or the trait does not have a **help** string,
        the returned string is constructed from the **desc** attribute on the
        trait and the **info** string on the trait's handler.
        """
        if full:
            help = self.help
            if help is not None:
                return help
        handler = self.handler
        if handler is not None:
            info = 'must be %s.' % handler.info()
        else:
            info = 'may be any value.'
        desc = self.desc
        if self.desc is None:
            return info.capitalize()
        return 'Specifies %s and %s' % ( desc, info )

    #---------------------------------------------------------------------------
    #  Returns the pickleable form of a CTrait object:
    #---------------------------------------------------------------------------

    def __reduce_ex__ ( self, protocol ):
        return ( __newobj__, ( self.__class__, 0 ), self.__getstate__() )

# Make sure the Python-level version of the trait class is known to all
# interested parties:
import ctraits
ctraits._ctrait( CTrait )

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

ConstantTypes    = ( NoneType, IntType, LongType, FloatType, ComplexType,
                     StringType, UnicodeType )

PythonTypes      = ( StringType,   UnicodeType,  IntType,    LongType,
                     FloatType,    ComplexType,  ListType,   TupleType,
                     DictType,     FunctionType, MethodType, ClassType,
                     InstanceType, TypeType,     NoneType )

CallableTypes    = ( FunctionType, MethodType )

TraitTypes       = ( TraitHandler, CTrait )

MutableTypes     = ( list, dict )

DefaultValues = {
    StringType:  '',
    UnicodeType: u'',
    IntType:     0,
    LongType:    0L,
    FloatType:   0.0,
    ComplexType: 0j,
    ListType:    [],
    TupleType:   (),
    DictType:    {},
    BooleanType: False
}

DefaultValueSpecial = [ Missing, Self ]
DefaultValueTypes   = [ ListType, DictType ]

#-------------------------------------------------------------------------------
#  Function used to unpickle new-style objects:
#-------------------------------------------------------------------------------

def __newobj__ ( cls, *args ):
    """ Unpickles new-style objects.
    """
    return cls.__new__( cls, *args )

#-------------------------------------------------------------------------------
#  Returns the type of default value specified:
#-------------------------------------------------------------------------------

def _default_value_type ( default_value ):
    try:
        return DefaultValueSpecial.index( default_value ) + 1
    except:
        try:
            return DefaultValueTypes.index( type( default_value ) ) + 3
        except:
            return 0

#-------------------------------------------------------------------------------
#  Returns the correct argument count for a specified function or method:
#-------------------------------------------------------------------------------

def _arg_count ( func ):
    if (type( func ) is MethodType) and (func.im_self is not None):
        return func.func_code.co_argcount - 1
    return func.func_code.co_argcount

#-------------------------------------------------------------------------------
#  'TraitFactory' class:
#-------------------------------------------------------------------------------

class TraitFactory ( object ):
    ### Need a docstring here.
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, maker_function = None ):
        if maker_function is not None:
            self.maker_function = maker_function

    #---------------------------------------------------------------------------
    #  Creates a CTrait instance:
    #---------------------------------------------------------------------------

    def __call__ ( self, *args, **metadata ):
        return self.maker_function( *args, **metadata )

#-------------------------------------------------------------------------------
#  Returns a trait created from a TraitFactory instance:
#-------------------------------------------------------------------------------

_trait_factory_instances = {}

def trait_factory ( trait ):
    global _trait_factory_instances

    tid = id( trait )
    if tid not in _trait_factory_instances:
        _trait_factory_instances[ tid ] = trait()
    return _trait_factory_instances[ tid ]

#-------------------------------------------------------------------------------
#  Casts a CTrait or TraitFactory to a CTrait but returns None if it is neither:
#-------------------------------------------------------------------------------

def trait_cast ( something ):
    """ Casts a CTrait or TraitFactory to a CTrait but returns None if it is
        neither.
    """
    if isinstance( something, CTrait ):
        return something
    if isinstance( something, TraitFactory ):
        return trait_factory( something )
    return None

#-------------------------------------------------------------------------------
#  Returns a trait derived from its input:
#-------------------------------------------------------------------------------

def trait_from ( something ):
    """ Returns a trait derived from its input.
    """
    if isinstance( something, CTrait ):
        return something
    if something is None:
        something = Any
    if isinstance( something, TraitFactory ):
        return trait_factory( something )
    return Trait( something )

# Patch the reference to 'trait_from' in 'trait_handlers.py':
trait_handlers.trait_from = trait_from

#-------------------------------------------------------------------------------
#  Define special 'factory' functions for creating common traits:
#-------------------------------------------------------------------------------

def Any ( value = None, **metadata ):
    """ Returns a trait that does no type-checking.

    Parameters
    ----------
    value : any
        Default value

    Default Value
    -------------
    *value* or None

    Description
    -----------
    A trait attribute defined with this trait is different from a normal
    Python attribute, because if it is referenced before assignment, it returns
    *value*, rather than generating an error.

    """

    metadata[ 'type' ] = 'trait'
    trait = CTrait( 0 )
    trait.default_value( _default_value_type( value ), value )
    trait.rich_comparison( metadata.get( 'rich_compare', True ) )
    trait.__dict__ = metadata.copy()
    return trait

Any = TraitFactory( Any )

#--- 'Coerced' traits ----------------------------------------------------------

def Int ( value = 0, **metadata ):
    """ Returns a trait whose value must be a plain integer.

    Parameters
    ----------
    value: integer
        Default value

    Default Value
    -------------
    *value* or 0
    """
    return Trait( value, TraitType( int ), **metadata )

Int = TraitFactory( Int )

def Long ( value = 0L, **metadata ):
    """ Returns a trait whose value must be a long integer.

    Parameters
    ----------
    value : long integer
        Default value

    Default Value
    -------------
    *value* or 0L
    """
    return Trait( value, TraitType( long ), **metadata )

Long = TraitFactory( Long )

def Float ( value = 0.0, **metadata ):
    """ Returns a trait whose value must be a floating point number.

    Parameters
    ----------
    value : floating point number
        Default value

    Default Value
    -------------
    *value* or 0.0
    """
    return Trait( value, TraitType( float ), **metadata )

Float = TraitFactory( Float )

def Complex ( value = 0.0 + 0.0j, **metadata ):
    """ Returns a trait whose value must be a complex number.

    Parameters
    ----------
    value : complex number
        Default value

    Default Value
    -------------
    *value* or 0.0 + 0.0j
    """

    return Trait( value, TraitType( complex ), **metadata )

Complex = TraitFactory( Complex )

def Str ( value = '', **metadata ):
    """ Returns a trait whose value must be a string.

    Parameters
    ----------
    value : string
        Default value

    Default Value
    -------------
    *value* or ''
    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitType( str ), TraitType( unicode ), **metadata )

Str = TraitFactory( Str )

def Unicode ( value = u'', **metadata ):
    """ Returns a trait whose value must be a Unicode string.

    Parameters
    ----------
    value : a Unicode string
        Default value

    Default Value
    -------------
    *value* or u''
    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitType( unicode ), **metadata )

Unicode = TraitFactory( Unicode )

def Bool ( value = False, **metadata ):
    """ Returns a trait whose value must be a Boolean.

    Parameters
    ----------
    value : Boolean
        Default value

    Default Value
    -------------
    *value* or False
    """
    return Trait( value, TraitType( bool ), **metadata )

Bool = TraitFactory( Bool )

#--- 'Cast' traits -------------------------------------------------------------

def CInt ( value = 0, **metadata ):
    """ Returns a trait whose value must be able to be cast to an integer.

    Parameters
    ----------
    value : anything that can be cast to an integer
        Default value

    Default Value
    -------------
    *value* or 0
    """
    return Trait( value, TraitCastType( int ), **metadata )

CInt = TraitFactory( CInt )

def CLong ( value = 0L, **metadata ):
    """ Returns a trait whose value must be able to be cast to a long integer.

    Parameters
    ----------
    value : anything that can be cast to a long integer
        Default value

    Default Value
    -------------
    *value* or 0L
    """
    return Trait( value, TraitCastType( long ), **metadata )

CLong = TraitFactory( CLong )

def CFloat ( value = 0.0, **metadata ):
    """ Returns a trait whose value must be able to be cast to a floating point number.

    Parameters
    ----------
    value : anything that can be cast to a float
        Default value

    Default Value
    -------------
    *value* or 0.0
    """
    return Trait( value, TraitCastType( float ), **metadata )

CFloat = TraitFactory( CFloat )

def CComplex ( value = 0.0+0.0j, **metadata ):
    """ Returns a trait whose value must be able to be cast to a complex number.

    Parameters
    ----------
    value : anything that can be cast to a complex number
        Default value

    Default Value
    -------------
    *value* or 0.0+0.0j
    """
    return Trait( value, TraitCastType( complex ), **metadata )

CComplex = TraitFactory( CComplex )

def CStr ( value = '', **metadata ):
    """ Returns a trait whose value must be able to be cast to a string.

    Parameters
    ----------
    value : anything that can be cast to a string
        Default value

    Default Value
    -------------
    *value* or ''
    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitCastType( str ), **metadata )

CStr = TraitFactory( CStr )

def CUnicode ( value = u'', **metadata ):
    """ Returns a trait whose value must be able to be cast to a Unicode string.

    Parameters
    ----------
    value : anything that can be cast to a Unicode string
        Default value

    Default Value
    -------------
    *value* or u''
    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitCastType( unicode ), **metadata )

CUnicode = TraitFactory( CUnicode )

def CBool ( value = False, **metadata ):
    """ Returns a trait whose value must be of a type that can be cast to a Boolean.

    Parameters
    ----------
    value : anything that can be cast to a Boolean
        Default value

    Default Value
    -------------
    *value* or False
    """
    return Trait( value, TraitCastType( bool ), **metadata )

CBool = TraitFactory( CBool )

#--- 'sequence' and 'mapping' traits -------------------------------------------

def List ( trait = None, value = None, minlen = 0, maxlen = sys.maxint,
           items = True, **metadata ):
    """ Returns a trait whose value must be a list whose items are of the
        specified trait type.

    Parameters
    ----------
    trait : a trait or a value that can be converted to a trait using Trait()
        The type of item that the list contains. If not specified, the list can
        contain items of any type.
    value :
        Default value for the list
    minlen : integer
        The minimum length of a list that can be assigned to the trait.
    maxlen : integer
        The maximum length of a list that can be assigned to the trait.

    The length of the list assigned to the trait must be such that::

        minlen <= len(list) <= maxlen

    Default Value
    -------------
    *value* or None
    """
    metadata.setdefault( 'copy', 'deep' )
    if isinstance( trait, SequenceTypes ):
        trait, value = value, list( trait )
    if value is None:
        value = []
    handler = TraitList( trait, minlen, maxlen, items )
    if handler.item_trait.instance_handler == '_instance_changed_handler':
        metadata.setdefault( 'instance_handler', '_list_changed_handler' )
    return Trait( value, handler, **metadata )

List = TraitFactory( List )

def Tuple ( *traits, **metadata ):
    """ Returns a trait whose value must be a tuple of specified trait types.

    Parameters
    ----------
    traits : zero or more arguments
        Definition of the default and allowed tuples. If the first item of
        *traits* is a tuple, it is used as the default value.
        The remaining argument list is used to form a tuple that constrains the
        values assigned to the returned trait. The trait's value must be a tuple
        of the same length as the remaining argument list, whose elements must
        match the types specified by the corresponding items of the remaining
        argument list.

    Default Value
    -------------
     1. If no arguments are specified, the default value is ().
     2. If a tuple is specified as the first argument, it is the default value.
     3. If a tuple is not specified as the first argument, the default value is
        a tuple whose length is the length of the argument list, and whose values
        are the default values for the corresponding trait types.

    Example for case #2::

        mytuple = Tuple(('Fred', 'Betty', 5))

    The trait's value must be a 3-element tuple whose first and second elements
    are strings, and whose third element is an integer. The default value is
    ('Fred', 'Betty', 5).

    Example for case #3::

        mytuple = Tuple('Fred', 'Betty', 5)

    The trait's value must be a 3-element tuple whose first and second elements
    are strings, and whose third element is an integer. The default value is
    ('','',0).

    """
    if len( traits ) == 0:
        return Trait( (), TraitType( tuple ), **metadata )
    value = None
    if isinstance( traits[0], tuple ):
        value, traits = traits[0], traits[1:]
        if len( traits ) == 0:
            traits = [ Trait( element ) for element in value ]
    tt = TraitTuple( *traits )
    if value is None:
        value = tuple( [ trait.default_value()[1] for trait in tt.traits ] )
    return Trait( value, tt, **metadata )

Tuple = TraitFactory( Tuple )

def Dict ( key_trait = None, value_trait = None, value = None, items = True,
           **metadata ):
    """ Returns a trait whose value must be a dictionary, optionally with
    specified types for keys and values.

    Parameters
    ----------
    key_trait : a trait or a value that can be converted to a trait using Trait()
        The trait type for keys in the dictionary; if not specified, any values
        can be used as keys.
    value_trait : a trait or a value that can be converted to a trait using Trait()
        The trait type for values in the dictionary; if not specified, any
        values can be used as dictionary values.
    value : a dictionary
        The default value for the returned trait
    items : Boolean
        Indicates whether the value contains items

    Default Value
    -------------
    *value* or {}
    """
    if isinstance( key_trait, dict ):
        key_trait, value_trait, value = value_trait, value, key_trait
    if value is None:
        value = {}
    return Trait( value, TraitDict( key_trait, value_trait, items ),
                  **metadata )

Dict = TraitFactory( Dict )

#--- 'array' traits ------------------------------------------------------------

def Array ( dtype = None, shape = None, value = None, **metadata ):
    """ Returns a trait whose value must be a numpy array.

    Parameters
    ----------
    dtype : a numpy dtype (e.g. float)
        The type of elements in the array; if omitted, no type-checking is
        performed on assigned values.
    shape : a tuple
        Describes the required shape of any assigned value. Wildcards and ranges
        are allowed. The value None within the *shape* tuple means that the
        corresponding dimension is not checked. (For example,
        ``shape=(None,3)`` means that the first dimension can be any size, but
        the second must be 3.) A two-element tuple within the *shape* tuple means
        that the dimension must be in the specified range. The second element
        can be None to indicate that there is no upper bound. (For example,
        ``shape=((3,5),(2,None))`` means that the first dimension must be in the
        range 3 to 5 (inclusive), and the second dimension must be at least 2.)
    value : numpy array
        A default value for the array

    Default Value
    -------------
    *value* or ``zeros(min(shape))``, where ``min(shape)`` refers to the minimum
    shape allowed by the array. If *shape* is not specified, the minimum shape
    is (0,).

    Description
    -----------
    An Array trait allows only upcasting of assigned values that are already
    numpy arrays. It automatically casts tuples and lists of the right shape
    to the specified *dtype* (just like numpy's **array** does).
    """
    return _Array( dtype, shape, value, coerce = False, **metadata )

Array = TraitFactory( Array )

def CArray ( dtype = None, shape = None, value = None, **metadata ):
    """ Returns a trait whose value must be a numpy array, with casting allowed.

    Parameters
    ----------
    dtype : a numpy dtype (e.g. float)
        The type of elements in the array; if omitted, no type-checking is
        performed on assigned values.
    shape : a tuple
        Describes the required shape of any assigned value. Wildcards and ranges
        are allowed. The value None within the *shape* tuple means that the
        corresponding dimension is not checked. (For example,
        ``shape=(None,3)`` means that the first dimension can be any size, but
        the second must be 3.) A two-element tuple within the *shape* tuple means
        that the dimension must be in the specified range. The second element
        can be None to indicate that there is no upper bound. (For example,
        ``shape=((3,5),(2,None))`` means that the first dimension must be in the
        range 3 to 5 (inclusive), and the second dimension must be at least 2.)
    value : numpy array
        A default value for the array

    Default Value
    -------------
    *value* or ``zeros(min(shape))``, where ``min(shape)`` refers to the minimum
    shape allowed by the array. If *shape* is not specified, the minimum shape
    is (0,).

    Description
    -----------
    The trait returned by CArray() is similar to that returned by Array(),
    except that it allows both upcasting and downcasting of assigned values
    that are already numpy arrays. It automatically casts tuples and lists
    of the right shape to the specified *dtype* (just like numpy's
    **array** does).
    """
    return _Array( dtype, shape, value, coerce = True, **metadata )

CArray = TraitFactory( CArray )

def _Array ( dtype = None, shape = None, value = None, coerce = False,
             typecode = None, **metadata ):
    metadata[ 'array' ] = True
    try:
        from trait_numeric import TraitArray
        import numpy
        import warnings
    except ImportError:
        raise TraitError( "Using Array or CArray trait types requires the "
                          "numpy package to be installed." )

    # Normally use object identity to detect array values changing:
    metadata.setdefault( 'rich_compare', False )

    if type( typecode ) in SequenceTypes:
        shape, typecode = typecode, shape

    if typecode is not None:
        warnings.warn("typecode is a deprecated argument; use dtype instead",
            DeprecationWarning)
        if dtype is not None and dtype != typecode:
            raise TraitError( "Inconsistent usage of the dtype and typecode "
                              "arguments; use dtype alone" )
        else:
            dtype = typecode

    if dtype is not None:
        try:
            # Convert the argument into an actual numpy dtype object.
            dtype = numpy.dtype(dtype)
        except TypeError, e:
            raise TraitError( "could not convert %r to a numpy dtype" % dtype)

    if shape is not None:
        if isinstance( shape, SequenceTypes ):
            for item in shape:
                if ((item is None) or (type( item ) is int) or
                    (isinstance( item, SequenceTypes ) and
                     (len( item ) == 2) and
                     (type( item[0] ) is int) and (item[0] >= 0) and
                     ((item[1] is None) or ((type( item[1] ) is int) and
                       (item[0] <= item[1]))))):
                    continue
                raise TraitError, "shape should be a list or tuple"
        else:
            raise TraitError, "shape should be a list or tuple"

        if (len( shape ) == 2) and (metadata.get( 'editor' ) is None):
            from enthought.traits.ui.api import ArrayEditor
            metadata[ 'editor' ] = ArrayEditor

    if value is None:
        if shape is None:
            value = numpy.zeros( ( 0, ), dtype )
        else:
            size = []
            for item in shape:
                if item is None:
                    item = 1
                elif type( item ) in SequenceTypes:
                    item = item[0]
                size.append( item )
            value = numpy.zeros( size, dtype )

    return Trait( value, TraitArray(shape=shape, coerce=coerce, dtype=dtype),
                                    **metadata )

#--- 'instance' traits ---------------------------------------------------------

def Instance ( klass, args = None, kw = None, allow_none = True, **metadata ):
    """ Returns a trait whose value must be an instance of a specified class,
    or one of its subclasses.

    Parameters
    ----------
    klass : class or instance
        The object that forms the basis for the trait; if it is an instance,
        then trait values must be instances of the same class or a
        subclass. This object is not the default value, even if it is an
        instance.
    args : tuple
        Positional arguments for generating the default value
    kw : dictionary
        Keyword arguments for generating the default value
    allow_none : Boolean
        Indicates whether None is allowed as a value

    Default Value
    -------------
    **None** if *klass* is an instance or if it is a class and *args* and *kw*
    are not specified. Otherwise, the default value is the instance obtained by
    calling ``klass(*args, **kw)``. Note that the constructor call is performed
    each time a default value is assigned, so each default value assigned is a
    unique instance.
    """
    metadata.setdefault( 'copy', 'deep' )
    metadata.setdefault( 'instance_handler', '_instance_changed_handler' )
    ti_klass = TraitInstance( klass, or_none = allow_none,
                              module = get_module_name() )
    if (args is None) and (kw is None):
        return Trait( ti_klass, **metadata )
    if kw is None:
        if type( args ) is dict:
            kw   = args
            args = ()
    elif type( kw ) is not dict:
        raise TraitError, "The 'kw' argument must be a dictionary"
    elif args is None:
        args = ()
    if type( args ) is not tuple:
        return Trait( args, ti_klass, **metadata )
    return Trait( _InstanceArgs( args, kw ), ti_klass, **metadata )

class _InstanceArgs ( object ):

    def __init__ ( self, args, kw ):
        self.args = args
        self.kw   = kw

def WeakRef ( klass = 'enthought.traits.HasTraits', allow_none = False,
              **metadata ):
    """ Returns a trait whose values must be instances of the same type
    (or a subclass) of the specified *klass*, which can be a class or an
    instance.

    Only a weak reference is maintained to any object assigned to a WeakRef
    trait. If no other references exist to the assigned value, the value may
    be garbage collected, in which case the value of the trait becomes None.
    In all other cases, the value returned by the trait is the original object.

    Parameters
    ----------
    klass : class or instance
        The object that forms the basis for the traitl If *klass* is omitted,
        then values must be an instance of HasTraits.
    allow_none : Boolean
        Indicates whether None is allowed to be assigned

    Default Value
    -------------
    **None** (even if allow_none==False)
    """

    metadata.setdefault( 'copy', 'ref' )
    ti_klass       = TraitWeakRef( klass, or_none = allow_none,
                                   module = get_module_name() )
    trait          = CTrait( 4 )
    trait.__dict__ = metadata.copy()
    trait.property( ti_klass._get,     _arg_count( ti_klass._get ),
                    ti_klass._set,     _arg_count( ti_klass._set ),
                    ti_klass.validate, _arg_count( ti_klass.validate ) )
    return trait

WeakRef = TraitFactory( WeakRef )

#--- 'creates a run-time default value' ----------------------------------------

class Default ( object ):
    """ Generates a value the first time it is accessed.

    A Default object can be used anywhere a default trait value would normally
    be specified, to generate a default value dynamically.
    """
    def __init__ ( self, func = None, args = (), kw = None ):
        self.default_value = ( func, args, kw )

#--- 'string' traits -----------------------------------------------------------

def Regex ( value = '', regex = '.*', **metadata ):
    """ Returns a trait whose value must be a string that matches a regular
        expression.

    Parameters
    ----------
    value : string
        The default value of the trait
    regex : string
        The regular expression that the trait value must match.

    Default Value
    -------------
    *value* or ''
    """
    return Trait( value, TraitString( regex = regex ), **metadata )

Regex = TraitFactory( Regex )

def String ( value = '', minlen = 0, maxlen = sys.maxint, regex = '',
             **metadata ):
    """ Returns a trait whose value must be a string, optionally of constrained
    length or matching a regular expression.

    Parameters
    ----------
    value : string
        Default value for the trait
    minlen : integer
        Minimum length of the string value
    maxlen : integer
        Maximum length of the string value
    regex : string
        A regular expression

    Default Value
    -------------
    *value* or ''
    """
    return Trait( value, TraitString( minlen = minlen,
                                      maxlen = maxlen,
                                      regex  = regex ), **metadata )

String = TraitFactory( String )

def Code ( value = '', minlen = 0, maxlen = sys.maxint, regex = '',
               **metadata ):
    """ Returns a trait whose value must be a string, optionally of constrained
    length or matching a regular expression.

    The trait returned by this function is indentical to that returned by
    String(), except that by default it uses a CodeEditor in TraitsUI views.

    Parameters
    ----------
    value : string
        Default value for the trait
    minlen : integer
        Minimum length of the string value
    maxlen : integer
        Maximum length of the string value
    regex : string
        A regular expression

    Default Value
    -------------
    *value* or ''

    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = code_editor
    return Trait( value, TraitString( minlen = minlen, maxlen = maxlen,
                                      regex  = regex ), **metadata )

Code = TraitFactory( Code )

def HTML ( value = '', **metadata ):
    """ Returns a trait whose value must be a string.

    The trait returned by this function is indentical to that returned by
    String(), except that by default it is parsed and displayed as HTML in
    TraitsUI views. The validation of the value does not enforce HTML syntax.

    Parameters
    ----------
    value : string
        Default value for the trait

    Default Value
    -------------
    *value* or ''

    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = html_editor
    return Trait( value, TraitString(), **metadata )

HTML = TraitFactory( HTML )

def Password ( value = '', minlen = 0, maxlen = sys.maxint, regex = '',
               **metadata ):
    """ Returns a trait whose value must be a string, optionally of constrained
    length or matching a regular expression.

    The trait returned by this function is indentical to that returned by
    String(), except that by default it uses a PasswordEditor in TraitsUI views,
    which obscures text entered by the user.

    Parameters
    ----------
    value : string
        Default value for the trait
    minlen : integer
        Minimum length of the string value
    maxlen : integer
        Maximum length of the string value
    regex : string
        A regular expression

    Default Value
    -------------
    *value* or ''

    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = password_editor
    return Trait( value, TraitString( minlen = minlen, maxlen = maxlen,
                                      regex  = regex ), **metadata )

Password = TraitFactory( Password )

def Expression ( value = '0', **metadata ):
    """ Returns a trait whose value must be a valid Python expression.

    Parameters
    ----------
    value : string
        The default value of the trait

    Default Value
    -------------
    *value* or '0'

    Description
    -----------
    The compiled form of a valid expression is stored as the mapped value of
    the trait.
    """
    return Trait( value, TraitExpression(), **metadata )

Expression = TraitFactory( Expression )

def PythonValue ( value = None, **metadata ):
    """ Returns a trait whose value can be of any type, and whose editor is
    a Python shell.

    Parameters
    ----------
    value : any
        The default value for the trait

    Default Value
    -------------
    *value* or None
    """
    if 'editor' not in metadata:
        metadata[ 'editor' ] = shell_editor()
    return Any( value, **metadata )

PythonValue = TraitFactory( PythonValue )

#--- 'file' traits -----------------------------------------------------------

def File ( value = '', filter = None, auto_set = False, **metadata ):
    """ Returns a trait whose value must be the name of a file.

    Parameters
    ----------
    value : string
        The default value for the trait
    filter : string
        A wildcard string to filter filenames in the file dialog box used by
        the attribute trait editor.
    auto_set : Boolean
        Indicates whether the file dialog box updates its selection after every
        key stroke.

    Default Value
    -------------
    *value* or ''
    """
    from enthought.traits.ui.editors import FileEditor

    return Str( value, editor = FileEditor( filter   = filter or [],
                                            auto_set = auto_set ),
                **metadata )

File = TraitFactory( File )


def Directory ( value = '', auto_set = False, **metadata ):
    """ Returns a trait whose value must be the name of a directory.

    Parameters
    ----------
    value : string
        The default value for the trait
    auto_set : Boolean
        Indicates whether the file dialog box updates its selection after every
        key stroke.

    Default Value
    -------------
    *value* or ''
    """
    from enthought.traits.ui.editors import DirectoryEditor

    return Str( value, editor = DirectoryEditor( auto_set = auto_set ),
                **metadata )

Directory = TraitFactory( Directory )

#-------------------------------------------------------------------------------
#  Factory function for creating range traits:
#-------------------------------------------------------------------------------

def Range ( low = None, high = None, value = None,
            exclude_low = False, exclude_high = False, **metadata ):
    """ Returns a trait whose numeric value must be in a specified range.

    Parameters
    ----------
    low : integer or float
        The low end of the range.
    high : integer or float
        The high end of the range.
    value : integer or float
        The default value of the trait
    exclude_low : Boolean
        Indicates whether the low end of the range is exclusive.
    exclude_high : Boolean
        Indicates whether the high end of the range is exclusive.

    The *low*, *high*, and *value* arguments must be of the same type (integer
    or float).

    Default Value
    -------------
    *value*; if *value* is None or omitted, the default value is *low*,
    unless *low* is None or omitted, in which case the default value is
    *high*.
    """
    if value is None:
        if low is not None:
            value = low
        else:
            value = high
    return Trait( value, TraitRange( low, high, exclude_low, exclude_high ),
                  **metadata )

Range = TraitFactory( Range )

#-------------------------------------------------------------------------------
#  Factory function for creating enumerated value traits:
#-------------------------------------------------------------------------------

def Enum ( *values, **metadata ):
    """ Returns a trait whose value must be one of an enumerated list.

    Parameters
    ----------
    values : list or tuple
        The list of valid values

    Default Value
    -------------
    values[0]
    """
    dv = values[0]
    if (len( values ) == 2) and (type( values[1] ) in SequenceTypes):
        values = values[1]
    return Trait( dv, TraitEnum( *values ), **metadata )

Enum = TraitFactory( Enum )

#-------------------------------------------------------------------------------
#  Factory function for creating constant traits:
#-------------------------------------------------------------------------------

def Constant ( value, **metadata ):
    """ Returns a read-only trait whose value is *value*.

    Parameters
    ----------
    value : any type except a list or dictionary
        The default value for the trait

    Default Value
    -------------
    *value*

    Description
    -----------
    Traits of this type are very space efficient (and fast) because *value* is
    not stored in each instance using the trait, but only in the trait object
    itself. The *value* cannot be a list or dictionary, because those types have
    mutable values.
    """
    if type( value ) in MutableTypes:
        raise TraitError, \
              "Cannot define a constant using a mutable list or dictionary"
    metadata[ 'type' ] = 'constant'
    return Trait( value, **metadata )

#-------------------------------------------------------------------------------
#  Factory function for creating C-based events:
#-------------------------------------------------------------------------------

def Event ( *value_type, **metadata ):
    """ Returns a trait event whose assigned value must meet the specified criteria.

    Parameters
    ----------
    value_type : any valid arguments for Trait()
        Specifies the criteria for successful assignment to the trait event.

    Default Value
    -------------
    No default value because events do not store values.
    """
    metadata[ 'type' ] = 'event';
    result = Trait( *value_type, **metadata )
    if 'instance_handler' in result.__dict__:
        del result.instance_handler
    return result

Event = TraitFactory( Event )

def Button ( label = '', image = None, style = 'button',
             orientation = 'vertical', width_padding = 7, height_padding = 5,
             **metadata ):
    """ Returns a trait event whose editor is a button.

    Parameters
    ----------
    label : string
        The label for the button
    image : enthought.pyface.ImageResource
        An image to display on the button
    style : one of: 'button', 'radio', 'toolbar', 'checkbox'
        The style of button to display
    orientation : one of: 'horizontal', 'vertical'
        The orientation of the label relative to the image
    width_padding : integer between 0 and 31
        Extra padding (in pixels) added to the left and right sides of the button
    height_padding : integer between 0 and 31
        Extra padding (in pixels) added to the top and bottom of the button

    Default Value
    -------------
    No default value because events do not store values.
    """

    from enthought.traits.ui.api import ButtonEditor

    return Event( editor = ButtonEditor(
                               label          = label,
                               image          = image,
                               style          = style,
                               orientation    = orientation,
                               width_padding  = width_padding,
                               height_padding = height_padding,
                               **metadata ) )

Button = TraitFactory( Button )

def ToolbarButton ( label = '', image = None, style = 'toolbar',
                    orientation = 'vertical', width_padding = 2,
                    height_padding = 2, **metadata ):
    """ Returns a trait even whose editor is a toolbar button.

    Parameters
    ----------
    label : string
        The label for the button
    image : enthought.pyface.ImageResource
        An image to display on the button
    style : one of: 'button', 'radio', 'toolbar', 'checkbox'
        The style of button to display
    orientation : one of: 'horizontal', 'vertical'
        The orientation of the label relative to the image
    width_padding : integer between 0 and 31
        Extra padding (in pixels) added to the left and right sides of the button
    height_padding : integer between 0 and 31
        Extra padding (in pixels) added to the top and bottom of the button

    Default Value
    -------------
    No default value because events do not store values.

    """
    return Button( label, image, style, orientation, width_padding,
                   height_padding, **metadata )

ToolbarButton = TraitFactory( ToolbarButton )

def UIDebugger ( **metadata ):
    ### JMS: Surely there's more to say about this...
    """ Returns a trait event whose editor is a button that opens debugger window.

    Default Value
    -------------
    No default value because events do not store values.
    """
    # FIXME: This import requires us to use a wx backend!  This is certainly
    # not what we want to do!
    from enthought.traits.ui.wx.ui_debug_editor import ToolkitEditorFactory

    return Event( editor = ToolkitEditorFactory(), **metadata )

UIDebugger = TraitFactory( UIDebugger )

#  Handle circular module dependencies:
trait_handlers.Event = Event

#-------------------------------------------------------------------------------
#  Factory function for creating C-based traits:
#-------------------------------------------------------------------------------

def Trait ( *value_type, **metadata ):
    """ Creates a trait definition.

    Parameters
    ----------
    This function accepts a variety of forms of parameter lists:

    +-------------------+---------------+-------------------------------------+
    | Format            | Example       | Description                         |
    +===================+===============+=====================================+
    | Trait(*default*)  | Trait(150.0)  | The type of the trait is inferred   |
    |                   |               | from the type of the default value, |
    |                   |               | which must be in *ConstantTypes*.   |
    +-------------------+---------------+-------------------------------------+
    | Trait(*default*,  | Trait(None,   | The trait accepts any of the        |
    | *other1*,         | 0, 1, 2,      | enumerated values, with the first   |
    | *other2*, ...)    | 'many')       | value being the default value. The  |
    |                   |               | values must be of types in          |
    |                   |               | *ConstantTypes*, but they need not  |
    |                   |               | be of the same type. The *default*  |
    |                   |               | value is not valid for assignment   |
    |                   |               | unless it is repeated later in the  |
    |                   |               | list.                               |
    +-------------------+---------------+-------------------------------------+
    | Trait([*default*, | Trait([None,  | Similar to the previous format, but |
    | *other1*,         | 0, 1, 2,      | takes an explicit list or a list    |
    | *other2*, ...])   | 'many'])      | variable.                           |
    +-------------------+---------------+-------------------------------------+
    | Trait(*type*)     | Trait(Int)    | The *type* parameter must be a name |
    |                   |               | of a Python type (see               |
    |                   |               | *PythonTypes*). Assigned values     |
    |                   |               | must be of exactly the specified    |
    |                   |               | type; no casting or coercion is     |
    |                   |               | performed. The default value is the |
    |                   |               | appropriate form of zero, False,    |
    |                   |               | or emtpy string, set or sequence.   |
    +-------------------+---------------+-------------------------------------+
    | Trait(*class*)    |::             | Values must be instances of *class* |
    |                   |               | or of a subclass of *class*. The    |
    |                   | class MyClass:| default value is None, but None     |
    |                   |    pass       | cannot be assigned as a value.      |
    |                   | foo = Trait(  |                                     |
    |                   | MyClass)      |                                     |
    +-------------------+---------------+-------------------------------------+
    | Trait(None,       |::             | Similar to the previous format, but |
    | *class*)          |               | None *can* be assigned as a value.  |
    |                   | class MyClass:|                                     |
    |                   |   pass        |                                     |
    |                   | foo = Trait(  |                                     |
    |                   | None, MyClass)|                                     |
    +-------------------+---------------+-------------------------------------+
    | Trait(*instance*) |::             | Values must be instances of the     |
    |                   |               | same class as *instance*, or of a   |
    |                   | class MyClass:| subclass of that class. The         |
    |                   |    pass       | specified instance is the default   |
    |                   | i = MyClass() | value.                              |
    |                   | foo =         |                                     |
    |                   |   Trait(i)    |                                     |
    +-------------------+---------------+-------------------------------------+
    | Trait(*handler*)  | Trait(        | Assignment to this trait is         |
    |                   | TraitEnum     | validated by an object derived from |
    |                   |               | **enthought.traits.TraitHandler**.  |
    +-------------------+---------------+-------------------------------------+
    | Trait(*default*,  | Trait(0.0, 0.0| This is the most general form of    |
    | { *type* |        | 'stuff',      | the function. The notation:         |
    | *constant* |      | TupleType)    | ``{...|...|...}+`` means a list of  |
    | *dict* | *class* ||               | one or more of any of the items     |
    | *function* |      |               | listed between the braces. Thus, the|
    | *handler* |       |               | most general form of the function   |
    | *trait* }+ )      |               | consists of a default value,        |
    |                   |               | followed by one or more of several  |
    |                   |               | possible items. A trait defined by  |
    |                   |               | multiple items is called a          |
    |                   |               | "compound" trait.                   |
    +-------------------+---------------+-------------------------------------+

    All forms of the Trait function accept both predefined and arbitrary
    keyword arguments. The value of each keyword argument becomes bound to the
    resulting trait object as the value of an attribute having the same name
    as the keyword. This feature lets you associate metadata with a trait.

    The following predefined keywords are accepted:

    desc : string
        Describes the intended meaning of the trait. It is used in
        exception messages and fly-over help in user interfaces.
    label : string
        Provides a human-readable name for the trait. It is used to label user
        interface editors for traits.
    editor : instance of a subclass of enthought.traits.api.Editor
        Object to use when creating a user interface editor for the trait. See
        the "Traits UI User Guide" for more information on trait editors.
    rich_compare : Boolean
        Indicates whether the basis for considering a trait attribute value to
        have changed is a "rich" comparison (True, the default), or simple
        object identity (False). This attribute can be useful in cases
        where a detailed comparison of two objects is very expensive, or where
        you do not care whether the details of an object change, as long as the
        same object is used.

    """
    return _TraitMaker( *value_type, **metadata ).as_ctrait()

#  Handle circular module dependencies:
trait_handlers.Trait = Trait

#-------------------------------------------------------------------------------
#  '_TraitMaker' class:
#-------------------------------------------------------------------------------

class _TraitMaker ( object ):

    # Ctrait type map for special trait types:
    type_map = {
       'event':    2,
       'constant': 7
    }

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, *value_type, **metadata ):
        metadata.setdefault( 'type', 'trait' )
        self.define( *value_type, **metadata )

    #---------------------------------------------------------------------------
    #  Define the trait:
    #---------------------------------------------------------------------------

    def define ( self, *value_type, **metadata ):
        default_value_type = -1
        default_value      = handler = clone = None
        if len( value_type ) > 0:
            default_value = value_type[0]
            value_type    = value_type[1:]
            if ((len( value_type ) == 0) and
                (type( default_value ) in SequenceTypes)):
                default_value, value_type = default_value[0], default_value
            if len( value_type ) == 0:
                if isinstance( default_value, TraitFactory ):
                    default_value = trait_factory( default_value )
                if default_value in PythonTypes:
                    handler       = TraitType( default_value )
                    default_value = DefaultValues.get( default_value )
                elif isinstance( default_value, CTrait ):
                    clone = default_value
                    default_value_type, default_value = clone.default_value()
                    metadata[ 'type' ] = clone.type
                elif isinstance( default_value, TraitHandler ):
                    handler       = default_value
                    default_value = None
                elif default_value is ThisClass:
                    handler       = ThisClass()
                    default_value = None
                else:
                    typeValue = type( default_value )
                    if isinstance(default_value, basestring):
                        string_options = self.extract( metadata, 'min_len',
                                                       'max_len', 'regex' )
                        if len( string_options ) == 0:
                            handler = TraitCastType( typeValue )
                        else:
                            handler = TraitString( **string_options )
                    elif typeValue in TypeTypes:
                        handler = TraitCastType( typeValue )
                    else:
                        metadata.setdefault( 'instance_handler',
                                             '_instance_changed_handler' )
                        handler = TraitInstance( default_value )
                        if default_value is handler.aClass:
                            default_value = DefaultValues.get( default_value )
            else:
                enum  = []
                other = []
                map   = {}
                self.do_list( value_type, enum, map, other )
                if (((len( enum )  == 1) and (enum[0] is None)) and
                    ((len( other ) == 1) and
                     isinstance( other[0], TraitInstance ))):
                    enum = []
                    other[0].allow_none()
                    metadata.setdefault( 'instance_handler',
                                         '_instance_changed_handler' )
                if len( enum ) > 0:
                    if (((len( map ) + len( other )) == 0) and
                        (default_value not in enum)):
                        enum.insert( 0, default_value )
                    other.append( TraitEnum( enum ) )
                if len( map ) > 0:
                    other.append( TraitMap( map ) )
                if len( other ) == 0:
                    handler = TraitHandler()
                elif len( other ) == 1:
                    handler = other[0]
                    if isinstance( handler, CTrait ):
                        clone, handler = handler, None
                        metadata[ 'type' ] = clone.type
                    elif isinstance( handler, TraitInstance ):
                        metadata.setdefault( 'instance_handler',
                                             '_instance_changed_handler' )
                        if default_value is None:
                            handler.allow_none()
                        elif isinstance( default_value, _InstanceArgs ):
                            default_value_type = 7
                            default_value = ( handler.create_default_value,
                                default_value.args, default_value.kw )
                        elif (len( enum ) == 0) and (len( map ) == 0):
                            aClass    = handler.aClass
                            typeValue = type( default_value )
                            if typeValue is dict:
                                default_value_type = 7
                                default_value = ( aClass, (), default_value )
                            elif not isinstance( default_value, aClass ):
                                if typeValue is not tuple:
                                    default_value = ( default_value, )
                                default_value_type = 7
                                default_value = ( aClass, default_value, None )
                else:
                    for i, item in enumerate( other ):
                        if isinstance( item, CTrait ):
                            if item.type != 'trait':
                                raise TraitError, ("Cannot create a complex "
                                    "trait containing %s trait." %
                                    add_article( item.type ) )
                            handler = item.handler
                            if handler is None:
                                break
                            other[i] = handler
                    else:
                        handler = TraitCompound( other )

        # Save the results:
        self.handler = handler
        self.clone   = clone
        if default_value_type < 0:
            if isinstance( default_value, Default ):
                default_value_type = 7
                default_value      = default_value.default_value
            else:
                if (handler is None) and (clone is not None):
                    handler = clone.handler
                if handler is not None:
                    default_value_type = handler.default_value_type
                    if default_value_type >= 0:
                        if hasattr( handler, 'default_value' ):
                            default_value = handler.default_value(default_value)
                    else:
                        try:
                            default_value = handler.validate( None, '',
                                                              default_value )
                        except:
                            pass
                if default_value_type < 0:
                    default_value_type = _default_value_type( default_value )
        self.default_value_type = default_value_type
        self.default_value      = default_value
        self.metadata           = metadata.copy()

    #---------------------------------------------------------------------------
    #  Determine the correct TraitHandler for each item in a list:
    #---------------------------------------------------------------------------

    def do_list ( self, list, enum, map, other ):
        for item in list:
            if item in PythonTypes:
                other.append( TraitType( item ) )
            else:
                if isinstance( item, TraitFactory ):
                    item = trait_factory( item )
                typeItem = type( item )
                if typeItem in ConstantTypes:
                    enum.append( item )
                elif typeItem in SequenceTypes:
                    self.do_list( item, enum, map, other )
                elif typeItem is DictType:
                    map.update( item )
                elif typeItem in CallableTypes:
                    other.append( TraitFunction( item ) )
                elif item is ThisClass:
                    other.append( ThisClass() )
                elif isinstance( item, TraitTypes ):
                    other.append( item )
                else:
                    other.append( TraitInstance( item ) )

    #---------------------------------------------------------------------------
    #  Returns a properly initialized 'CTrait' instance:
    #---------------------------------------------------------------------------

    def as_ctrait ( self ):
        metadata = self.metadata
        trait    = CTrait( self.type_map.get( metadata.get( 'type' ), 0 ) )
        clone    = self.clone
        if clone is not None:
            trait.clone( clone )
            if clone.__dict__ is not None:
                trait.__dict__ = clone.__dict__.copy()
        trait.default_value( self.default_value_type, self.default_value )
        handler = self.handler
        if handler is not None:
            trait.handler = handler
            if hasattr( handler, 'fast_validate' ):
                trait.set_validate( handler.fast_validate )
            else:
                trait.set_validate( handler.validate )
            if hasattr( handler, 'post_setattr' ):
                trait.post_setattr = handler.post_setattr
        trait.rich_comparison( metadata.get( 'rich_compare', True ) )
        if len( metadata ) > 0:
            if trait.__dict__ is None:
                trait.__dict__ = metadata
            else:
                trait.__dict__.update( metadata )
        return trait

    #---------------------------------------------------------------------------
    #  Extract a set of keywords from a dictionary:
    #---------------------------------------------------------------------------

    def extract ( self, from_dict, *keys ):
        to_dict = {}
        for key in keys:
            if key in from_dict:
                to_dict[ key ] = from_dict[ key ]
                del from_dict[ key ]
        return to_dict

#-------------------------------------------------------------------------------
#  Factory function for creating traits with standard Python behavior:
#-------------------------------------------------------------------------------

def TraitPython ( **metadata ):
    """ Returns a trait that has standard Python behavior.
    """
    metadata.setdefault( 'type', 'python' )
    trait = CTrait( 1 )
    trait.default_value( 0, Undefined )
    trait.__dict__ = metadata.copy()
    return trait

#-------------------------------------------------------------------------------
#  Factory function for creating C-based trait delegates:
#-------------------------------------------------------------------------------

def Delegate ( delegate, prefix = '', modify = False, **metadata ):
    """ Creates a "delegator" trait, whose definition and default value are
    delegated to a *delegate* trait attribute on another object.

    Parameters
    ----------
    delegate : string
        Name of the attribute on the current object which references the object
        that is the trait's delegate
    prefix : string
        A prefix or substitution applied to the original attribute when looking
        up the delegated attribute
    modify : Boolean
        Indicates whether changes are made to the delegate attribute,
        rather than to the delegator attribute

    Description
    -----------
    An object containing a delegator trait attribute must contain a second
    attribute that references the object containing the delegate trait attribute.
    The name of this second attribute is passed as the *delegate* argument to
    the Delegate() function.

    The following rules govern the application of the prefix parameter:

    * If *prefix* is empty or omitted, the delegation is to an attribute of
      the delegate object with the same name as the delegator attribute.
    * If *prefix* is a valid Python attribute name, then the delegation is
      to an attribute whose name is the value of *prefix*.
    * If *prefix* ends with an asterisk ('*') and is longer than one
      character, then the delegation is to an attribute whose name is the
      value of *prefix*, minus the trailing asterisk, prepended to the
      delegator attribute name.
    * If *prefix* is equal to a single asterisk, the delegation is to an
      attribute whose name is the value of the delegator object's
      __prefix__ attribute prepended to delegator attribute name.

    If *modify* is True, then any changes to the delegator attribute are
    actually applied to the delegate attribute.

    """
    metadata.setdefault( 'type', 'delegate' )
    if prefix == '':
        prefix_type = 0
    elif prefix[-1:] != '*':
        prefix_type = 1
    else:
        prefix = prefix[:-1]
        if prefix != '':
            prefix_type = 2
        else:
            prefix_type = 3
    trait = CTrait( 3 )
    trait.delegate( delegate, prefix, prefix_type, modify )
    trait.__dict__ = metadata.copy()
    return trait

#-------------------------------------------------------------------------------
#  Factory function for creating C-based trait properties:
#-------------------------------------------------------------------------------

def Property ( fget = None, fset = None, fvalidate = None, force = False,
               handler = None, trait = None, **metadata ):
    ### JMS: Need more detail in docstring
    """ Returns a trait whose value is a Python property.

    Parameters
    ----------
    fget : function
        The "getter" function for the property
    fset : function
        The "setter" function for the property
    fvalidate : function
        The validation function for the property
    force : Boolean
        Indicates whether to force WHAT?
    handler : function
        A trait handler function for the trait
    trait : a trait definition or value that can be converted to a trait
        A trait definition that constrains the values of the property trait

    Description
    -----------
    If no getter or setter functions are specified, it is assumed that they
    are defined elsewhere on the class whose attribute this trait is
    assigned to. For example::

        class Bar(HasTraits):
            foo = Property(Float)
            # Shadow trait attribute
            _foo = Float

            def _set_foo(self, x):
                self._foo = x

            def _get_foo(self):
                return self._foo

    """
    metadata[ 'type' ] = 'property'

    # If no parameters specified, must be a forward reference (if not forced):
    if (not force) and (fset is None):
        sum = ((fget      is not None) +
               (fvalidate is not None) +
               (trait     is not None))
        if sum <= 1:
            if sum == 0:
                return ForwardProperty( metadata )
            handler = None
            if fget is not None:
                trait = fget
            if trait is not None:
                trait = trait_cast( trait )
                if trait is not None:
                    fvalidate = handler = trait.handler
                    if fvalidate is not None:
                        fvalidate = handler.validate
            if (fvalidate is not None) or (trait is not None):
                if 'editor' not in metadata:
                    if (trait is not None) and (trait.editor is not None):
                        metadata[ 'editor' ] = trait.editor
                return ForwardProperty( metadata, fvalidate, handler )

    if fget is None:
        if fset is None:
            fget = _undefined_get
            fset = _undefined_set
        else:
            fget = _write_only
    elif fset is None:
        fset = _read_only

    if trait is not None:
        trait   = trait_cast( trait )
        handler = trait.handler
        if (fvalidate is None) and (handler is not None):
            fvalidate = handler.validate
        if ('editor' not in metadata) and (trait.editor is not None):
            metadata[ 'editor' ] = trait.editor

    n     = 0
    trait = CTrait( 4 )
    trait.__dict__ = metadata.copy()
    if fvalidate is not None:
        n = _arg_count( fvalidate )
    trait.property( fget,      _arg_count( fget ),
                    fset,      _arg_count( fset ),
                    fvalidate, n )
    trait.handler = handler
    return trait

Property = TraitFactory( Property )

class ForwardProperty ( object ):
    """ Used to implement Property traits where accessor functions are defined
    implicitly on the class.
    """
    def __init__ ( self, metadata, validate = None, handler = None ):
        self.metadata = metadata.copy()
        self.validate = validate
        self.handler  = handler

#-------------------------------------------------------------------------------
#  Property error handling functions:
#-------------------------------------------------------------------------------

def _write_only ( object, name ):
    raise TraitError, "The '%s' trait of %s instance is 'write only'." % (
                      name, class_of( object ) )

def _read_only ( object, name, value ):
    raise TraitError, "The '%s' trait of %s instance is 'read only'." % (
                      name, class_of( object ) )

def _undefined_get ( object, name ):
    raise TraitError, ("The '%s' trait of %s instance is a property that has "
                       "no 'get' or 'set' method") % (
                       name, class_of( object ) )

def _undefined_set ( object, name, value ):
    _undefined_get( object, name )

#-------------------------------------------------------------------------------
#  Dictionary used to handler return type mapping special cases:
#-------------------------------------------------------------------------------

SpecialNames = {
   'int':     trait_factory( Int ),
   'long':    trait_factory( Long ),
   'float':   trait_factory( Float ),
   'complex': trait_factory( Complex ),
   'str':     trait_factory( Str ),
   'unicode': trait_factory( Unicode ),
   'bool':    trait_factory( Bool ),
   'list':    trait_factory( List ),
   'tuple':   trait_factory( Tuple ),
   'dict':    trait_factory( Dict )
}

#-------------------------------------------------------------------------------
#  Create predefined, reusable trait instances:
#-------------------------------------------------------------------------------

# Synonym for Bool; default value is False.
false           = Bool

# Boolean values only; default value is True.
true            = Bool( True )

# Function values only (i.e., types.FunctionType); default value is None.
Function        = Trait( FunctionType )

# Method values only (i.e., types.MethodType); default value is None.
Method          = Trait( MethodType )

# Class values (old-style, i.e., type.ClassType) only; default value is None.
Class           = Trait( ClassType )

# Module values only (i.e., types.ModuleType); default value is None.
Module          = Trait( ModuleType )

# Type values only (i.e., types.TypeType); default value is None.
Type            = Trait( TypeType )

# Allows only class values of the same class (or a subclass) as the object
# containing the trait attribute; default value is None.
This            = Trait( ThisClass )

# Same as This; default value is the object containing the trait attribute
# defined using this trait..
self            = Trait( Self, ThisClass )

# Either(A,B,...,Z) allows any of the traits A,B,...,Z. ('Either' is
# grammatically imprecise, but it reads better than 'AnyOf' or 'OneOf'.)
Either          = lambda *args: Trait(None, *args)

# This trait provides behavior identical to a standard Python attribute.
# That is, it allows any value to be assigned, and raises an ValueError if
# an attempt is made to get the value before one has been assigned. It has no
# default value. This trait is most often used in conjunction with wildcard
# naming. See the *Traits User Manual* for details on wildcards.
Python          = TraitPython()

# Prevents any value from being assigned or read.
# That is, any attempt to get or set the value of the trait attribute raises
# an exception. This trait is most often used in conjunction with wildcard
# naming, for example, to catch spelling mistakes in attribute names. See the
# *Traits User Manual* for details on wildcards.
Disallow        = CTrait( 5 )

# This trait is write-once, and then read-only.
# The initial value of the attribute is the special, singleton object
# Undefined. The trait allows any value to be assigned to the attribute
# if the current value is the Undefined object. Once any other value is
# assigned, no further assignment is allowed. Normally, the initial assignment
# to the attribute is performed in the class constructor, based on information
# passed to the constructor. If the read-only value is known in advance of
# run time, use the Constant() function instead of ReadOnly to define
# the trait.
ReadOnly        = CTrait( 6 )
ReadOnly.default_value( 0, Undefined )  # This allows it to be written once

# Allows any value to be assigned; no type-checking is performed.
# Default value is Undefined.
undefined       = Any( Undefined )

# Indicates that a parameter is missing from a type-checked method signature.
# Allows any value to be assigned; no type-checking is performed; default value
# is the singleton Missing object.
# See **enthought.traits.has_traits.method()**.
missing         = CTrait( 0 )
missing.handler = TraitHandler()
missing.default_value( 1, Missing )

# Generic trait with 'object' behavior
generic_trait   = CTrait( 8 )
# Callable values; default is None.
Callable        = Trait( TraitCallable(), copy = 'ref' )


# List traits:

# List of integer values; default value is [].
ListInt        = List( int )
# List of float values; default value is [].
ListFloat      = List( float )
# List of string values; default value is [].
ListStr        = List( str )
# List of Unicode string values; default value is [].
ListUnicode    = List( unicode )
# List of complex values; default value is [].
ListComplex    = List( complex )
# List of Boolean values; default value is [].
ListBool       = List( bool )
# List of function values; default value is [].
ListFunction   = List( FunctionType )
# List of method values; default value is [].
ListMethod     = List( MethodType )
# List of class values; default value is [].
ListClass      = List( ClassType )
# List of instance values; default value is [].
ListInstance   = List( InstanceType )
# List of container type values; default value is [].
ListThis       = List( ThisClass )


# Dictionary traits:

# Only a dictionary of string:Any values can be assigned; only string keys can
# be inserted. The default value is {}.
DictStrAny     = Dict( str, Any )
# Only a dictionary of string:string values can be assigned; only string keys
# with string values can be inserted. The default value is {}.
DictStrStr     = Dict( str, str )
# Only a dictionary of string:integer values can be assigned; only string keys
# with integer values can be inserted. The default value is {}.
DictStrInt     = Dict( str, int )
# Only a dictionary of string:long-integer values can be assigned; only string
# keys with long-integer values can be inserted. The default value is {}.
DictStrLong    = Dict( str, long )
# Only a dictionary of string:float values can be assigned; only string keys
# with float values can be inserted. The default value is {}.
DictStrFloat   = Dict( str, float )
# Only a dictionary of string:Boolean values can be assigned; only string keys
# with Boolean values can be inserted. The default value is {}.
DictStrBool    = Dict( str, bool )
# Only a dictionary of string:list values can be assigned; only string keys
# with list values can be assigned. The default value is {}.
DictStrList    = Dict( str, list )

#-------------------------------------------------------------------------------
#  User interface related color and font traits:
#-------------------------------------------------------------------------------

def Color ( *args, **metadata ):
    """ Returns a trait whose value must be a GUI toolkit-specific color.

    Description
    -----------
    For wxPython, the returned trait accepts any of the following values:

    * A wx.Colour instance
    * A wx.ColourPtr instance
    * an integer whose hexadecimal form is 0x*RRGGBB*, where *RR* is the red
      value, *GG* is the green value, and *BB* is the blue value

    Default Value
    -------------
    For wxPython, 0x000000 (that is, white)
    """
    from enthought.traits.ui.api import ColorTrait
    return ColorTrait( *args, **metadata )

Color = TraitFactory( Color )

def RGBColor ( *args, **metadata ):
    """ Returns a trait whose value must be a GUI toolkit-specific RGB-based color.

    Description
    -----------
    For wxPython, the returned trait accepts any of the following values:

    * A tuple of the form (*r*, *g*, *b*), in which *r*, *g*, and *b* represent
      red, green, and blue values, respectively, and are floats in the range
      from 0.0 to 1.0
    * An integer whose hexadecimal form is 0x*RRGGBB*, where *RR* is the red
      value, *GG* is the green value, and *BB* is the blue value

    Default Value
    -------------
    For wxPython, (0.0, 0.0, 0.0) (that is, white)
    """
    from enthought.traits.ui.api import RGBColorTrait
    return RGBColorTrait( *args, **metadata )

RGBColor = TraitFactory( RGBColor )

def Font ( *args, **metadata ):
    """ Returns a trait whose value must be a GUI toolkit-specific font.

    Description
    -----------
    For wxPython, the returned trait accepts any of the following:

    * a wx.Font instance
    * a wx.FontPtr instance
    * a string describing the font, including one or more of the font family,
      size, weight, style, and typeface name.

    Default Value
    -------------
    For wxPython, 'Arial 10'
    """
    from enthought.traits.ui.api import FontTrait
    return FontTrait( *args, **metadata )

Font = TraitFactory( Font )


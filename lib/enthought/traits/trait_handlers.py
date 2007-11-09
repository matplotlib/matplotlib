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
# Date: 06/21/2002
#
#  Symbols defined: TraitHandler
#                   TraitRange
#                   TraitType
#                   TraitString
#                   TraitInstance
#                   TraitFunction
#                   TraitEnum
#                   TraitPrefixList
#                   TraitMap
#                   TraitPrefixMap
#                   TraitCompound
#                   TraitList
#                   TraitDict
#
#  Refactored into a separate module: 07/04/2003
#------------------------------------------------------------------------------
""" Defines the base TraitHandler class and a standard set of TraitHandler
subclasses for use with the Traits package.

A trait handler mediates the assignment of values to object traits. It verifies
(via its validate() method) that a specified value is consistent with the
object trait, and generates a TraitError exception if it is not consistent.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import logging
import sys
import re
import copy

from types \
    import InstanceType, TypeType, FunctionType, MethodType

from weakref \
    import ref

from ctraits \
    import CTraitMethod

from trait_base \
    import strx, SequenceTypes, Undefined, TypeTypes, ClassTypes, \
           CoercableTypes, class_of, enumerate

from trait_errors \
    import TraitError


# Setup a logger for this module.
logger = logging.getLogger(__name__)


# Patched by 'traits.py' once class is defined!
Trait = Event = None

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

RangeTypes    = ( int, long, float )

CallableTypes = ( FunctionType, MethodType, CTraitMethod )

#-------------------------------------------------------------------------------
#  Forward references:
#-------------------------------------------------------------------------------

trait_from = None  # Patched by 'traits.py' when real 'trait_from' is defined

#-------------------------------------------------------------------------------
#  'TraitHandler' class (base class for all trait handlers):
#-------------------------------------------------------------------------------

class TraitHandler ( object ):
    """ The task of this class and its subclasses is to verify the correctness
    of values assigned to object trait attributes.

    This class is an alternative to trait validator functions. A trait handler
    has several advantages over a trait validator function, due to being an
    object:

        * Trait handlers have constructors and state. Therefore, you can use
          them to create *parameterized types*.
        * Trait handlers can have multiple methods, whereas validator functions
          can have only one callable interface. This feature allows more
          flexibility in their implementation, and allows them to handle a
          wider range of cases, such as interactions with other components.

    The only method of TraitHandler that *must* be implemented by subclasses
    is validate().
    """

    default_value_type = -1
    has_items          = False
    is_mapped          = False
    editor             = None

    __traits_metadata__ = {
        'type': 'trait'
    }

    def validate ( self, object, name, value ):
        """ Verifies whether a new value assigned to a trait attribute is valid.

        Parameters
        ----------
        object : object
            The object whose attribute is being assigned
        name : string
            The name of the attribute being assigned
        value
            The proposed new value for the attribute

        Returns
        -------
        If the new value is valid, this method must return either the original
        value passed to it, or an alternate value to be assigned in place of the
        original value. Whatever value this method returns is the actual value
        assigned to *object.name*.

        Description
        -----------
        This method *must* be implemented by subclasses of TraitHandler. It is
        called whenever a new value is assigned to a trait attribute defined
        using this trait handler.

        If the value received by validate() is not valid for the trait
        attribute, the method must called the predefined error() method to
        raise a TraitError exception

        """
        raise TraitError, (
              "The '%s' trait of %s instance has an unknown type. "
              "Contact the developer to correct the problem." % (
              name, class_of( object ) ) )

    def is_valid ( self, object, name, value ):
        try:
            self.validate( object, name, value )
            return True
        except:
            return False

    def error ( self, object, name, value ):
        """Raises a TraitError exception.

        Parameters
        ----------
        object : object
            The object whose attribute is being assigned
        name : string
            The name of the attribute being assigned
        value
            The proposed new value for the attribute

        Description
        -----------
        This method is called by the validate() method when an assigned value
        is not valid. Raising a TraitError exception either notifies the user of
        the problem, or, in the case of compound traits, provides a chance for
        another trait handler to handle to validate the value.
        """
        raise TraitError, ( object, name, self.info(), value )

    def arg_error ( self, method, arg_num, object, name, value ):
        """ Raises a TraitError exception to notify the user that a method on
        an instance received a positional argument of an incorrect type.

        Parameters
        ----------
        method : function
            The method that encountered the error
        arg_num : integer
            The position of the incorrect argument in the argument list
        object : object
            The object whose method was called
        name : string
            The name of the parameter corresponding to the incorrect argument
        value
            The value passed to the argument

        Description
        -----------
        This method can be called when type-checking a method.
        """
        raise TraitError, ("The '%s' parameter (argument %d) of the %s method "
                           "of %s instance must be %s, but a value of %s was "
                           "specified." % ( name, arg_num, method.tm_name,
                           class_of( object ), self.info(), value ) )

    def keyword_error ( self, method, object, name, value ):
        """ Raises a TraitError exception to notify the user that a method on
        an instance received a keyword argument of an incorrect type.

        Parameters
        ----------
        method : function
            The method that encountered the error
        object : object
            The object whose method was called
        name : string
            The name of the parameter corresponding to the incorrect argument
        value
            The value passed to the argument

        Description
        -----------
        This method can be called when type-checking a method.
        """
        raise TraitError, ("The '%s' keyword argument of the %s method of "
                           "%s instance must be %s, but a value of %s was "
                           "specified." % ( name, method.tm_name,
                           class_of( object ), self.info(), value ) )

    def missing_arg_error ( self, method, arg_num, object, name ):
        """ Raises a TraitError exception to notify the user that a method on
        an instance failed to receive a required positional argument.

        Parameters
        ----------
        method : function
            The method that encountered the error
        arg_num : integer
            The position of the incorrect argument in the argument list
        object : object
            The object whose method was called
        name : string
            The name of the parameter corresponding to the incorrect argument

        Description
        -----------
        This method can be called when type-checking a method.
        """
        raise TraitError, ("The '%s' parameter (argument %d) of the %s method "
                           "of %s instance must be specified, but was omitted."
                           % ( name, arg_num, method.tm_name,
                               class_of( object ) ) )

    def dup_arg_error ( self, method, arg_num, object, name ):
        """ Raises a TraitError exception to notify the user that a method on
        an instance received an argument as both a keyword argument and a
        positional argument.

        Parameters
        ----------
        method : function
            The method that encountered the error
        arg_num : integer
            The position of the incorrect argument in the argument list
        object : object
            The object whose method was called
        name : string
            The name of the parameter corresponding to the incorrect argument

        Description
        -----------
        This method can be called when type-checking a method.
        """
        raise TraitError, ("The '%s' parameter (argument %d) of the %s method "
                           "of %s instance was specified as both a positional "
                           "and keyword value."
                           % ( name, arg_num, method.tm_name,
                               class_of( object ) ) )

    def return_error ( self, method, object, value ):
        """ Raises a TraitError exception to notify the user that a method on
        an instance returned a value of incorrect type.

        Parameters
        ----------
        method : function
            The method that encountered the error
        object : object
            The object whose method was called
        value
            The value returned by the method

        Description
        -----------
        This method can be called when type-checking a method.
        """
        raise TraitError, ("The result of the %s method of %s instance must "
                           "be %s, but a value of %s was returned." % (
                           method.tm_name, class_of( object ), self.info(),
                           value ) )

    def info ( self ):
        """Must return a string describing the type of value accepted by the
        trait handler.

        The string should be a phrase describing the type defined by the
        TraitHandler subclass, rather than a complete sentence. For example, use
        the phrase, "a square sprocket" instead of the sentence, "The value must
        be a square sprocket." The value returned by info() is combined with
        other information whenever an error occurs and therefore makes more
        sense to the user if the result is a phrase. The info() method is
        similar in purpose and use to the **info** attribute of a validator
        function.

        Note that the result can include information specific to the particular
        trait handler instance. For example, TraitRange instances return a
        string indicating the range of values acceptable to the handler (e.g.,
        "an integer in the range from 1 to 9"). If the info() method is not
        overridden, the default method returns the string 'a legal value'.
        """
        return 'a legal value'

    def repr ( self, value ):
        """ Returns a printable representation of a value.

        Parameters
        ----------
        value
            The value to be printed

        Description
        -----------
        If *value* is an instance, the method returns the printable
        representation of the instance's class.
        """
        if type( value ) is InstanceType:
            return 'class '  + value.__class__.__name__
        return repr( value )

    def get_editor ( self, trait ):
        """ Returns a trait editor that allows the user to modify the *trait*
        trait.

        Parameters
        ----------
        trait : trait
            The trait to be edited

        Description
        -----------
        This method only needs to be specified if traits defined using this
        trait handler require a non-default trait editor in trait user
        interfaces. The default implementation of this method returns a trait
        editor that allows the user to type an arbitrary string as the value.

        For more information on trait user interfaces, refer to the chapter,
        "GUIs for Trait Attributes: Visualization" in the *Traits User Manual*.
        """
        if self.editor is None:
            from enthought.traits.ui.api import TextEditor
            self.editor = TextEditor()
        return self.editor

    def metadata ( self ):
        """ Returns the metadata dictionary for the current object.
        """
        return getattr( self, '__traits_metadata__', {} )

#-------------------------------------------------------------------------------
#  'TraitRange' class:
#-------------------------------------------------------------------------------

class TraitRange ( TraitHandler ):
    """Ensures that a trait attribute lies within a specified numeric range.

    TraitRange is the underlying handler for the predefined Range() trait
    factory.

    Any value assigned to a trait containing a TraitRange handler must be of the
    correct type and in the numeric range defined by the TraitRange instance.
    No automatic coercion takes place. For example::

        class Person(HasTraits):
            age = Trait(0, TraitRange(0, 150))
            weight = Trait(0.0, TraitRange(0.0, None))

    This example defines a Person class, which has an **age** trait
    attribute, which must be an integer/long in the range from 0 to 150, and a
    **weight** trait attribute, which must be a non-negative float value.
    """
    def __init__ ( self, low = None, high = None,
                         exclude_low = False, exclude_high = False ):
        """ Creates a TraitRange handler.

        Parameters
        ----------
        low : number
            The minimum value that the trait can accept
        high : number
            The maximum value that the trait can accept
        exclude_low : Boolean
            Should the *low* value be exclusive (or inclusive)
        exclude_high : Boolean
            Should the *high* value be exclusive (or inclusive)

        Description
        -----------
        The *low* and *high* values must be of the same Python numeric type,
        either ``int``, ``long`` or ``float``. Alternatively, one of the values
        may be None, to indicate that that portion of the range is
        unbounded. The *exclude_low* and *exclude_high* values can be used to
        specify whether the *low* and *high* values should be exclusive (or
        inclusive).
        """

        vtype             = type( high )
        if (low is not None) and (vtype is not float):
            vtype = type( low )
        if vtype not in RangeTypes:
            raise TraitError, ("TraitRange can only be use for int, long or "
                               "float values, but a value of type %s was "
                               "specified." % vtype)
        if vtype is float:
            self.validate = self.float_validate
            kind           = 4
            self.type_desc = 'a floating point number'
            if low is not None:
                low = float( low )
            if high is not None:
                high = float( high )
        elif vtype is long:
            self.validate = self.long_validate
            self.type_desc = 'a long integer'
            if low is not None:
                low = long( low )
            if high is not None:
                high = long( high )
        else:
            self.validate = self.int_validate
            kind = 3
            self.type_desc = 'an integer'
            if low is not None:
                low = int( low )
            if high is not None:
                high = int( high )
        exclude_mask = 0
        if exclude_low:
            exclude_mask |= 1
        if exclude_high:
            exclude_mask |= 2
        if vtype is not long:
            self.fast_validate = ( kind, low, high, exclude_mask )

        # Assign type-corrected arguments to handler attributes
        self.low          = low
        self.high         = high
        self.exclude_low  = exclude_low
        self.exclude_high = exclude_high


    def float_validate ( self, object, name, value ):
        try:
            if (isinstance( value, RangeTypes ) and
                ((self.low  is None) or
                 (self.exclude_low and (self.low < value)) or
                 ((not self.exclude_low) and (self.low <= value))) and
                ((self.high is None) or
                 (self.exclude_high and (self.high > value)) or
                 ((not self.exclude_high) and (self.high >= value)))):
               return float( value )
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def int_validate ( self, object, name, value ):
        try:
            if (isinstance( value, int ) and
                ((self.low  is None) or
                 (self.exclude_low and (self.low < value)) or
                 ((not self.exclude_low) and (self.low <= value))) and
                ((self.high is None) or
                 (self.exclude_high and (self.high > value)) or
                 ((not self.exclude_high) and (self.high >= value)))):
               return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def long_validate ( self, object, name, value ):
        try:
            if (isinstance( value, long ) and
                ((self.low  is None) or
                 (self.exclude_low and (self.low < value)) or
                 ((not self.exclude_low) and (self.low <= value))) and
                ((self.high is None) or
                 (self.exclude_high and (self.high > value)) or
                 ((not self.exclude_high) and (self.high >= value)))):
               return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        if self.low is None:
            if self.high is None:
                return self.type_desc
            return '%s <%s %s' % (
                   self.type_desc, '='[ self.exclude_high: ], self.high )
        elif self.high is None:
            return  '%s >%s %s' % (
                    self.type_desc, '='[ self.exclude_low: ], self.low )
        return '%s <%s %s <%s %s' % (
               self.low, '='[ self.exclude_low: ], self.type_desc,
               '='[ self.exclude_high: ], self.high )

    def get_editor ( self, trait ):
        auto_set = trait.auto_set
        if auto_set is None:
            auto_set = True
        from enthought.traits.ui.api import RangeEditor
        return RangeEditor( self,
                            mode       = trait.mode or 'auto',
                            cols       = trait.cols or 3,
                            auto_set   = auto_set,
                            enter_set  = trait.enter_set or False,
                            low_label  = trait.low  or '',
                            high_label = trait.high or '' )

#-------------------------------------------------------------------------------
#  'TraitString' class:
#-------------------------------------------------------------------------------

class TraitString ( TraitHandler ):
    """ Ensures that a trait attribute value is a string that satisfied some
    additional, optional constraints.

    The optional constraints include minimum and maximum lengths, and a regular
    expression that the string must match.

    If the value assigned to the trait attribute is a Python numeric type, the
    TraitString handler first coerces the value to a string. Values of other
    non-string types result in a TraitError being raised. The handler then
    makes sure that the resulting string is within the specified length range
    and that it matches the regular expression.
    """
    def __init__ ( self, minlen = 0, maxlen = sys.maxint, regex = '' ):
        """ Creates a TraitString handler.

        Parameters
        ----------
        minlen : integer
            The minimum length allowed for the string
        maxlen : integer
            The maximum length allowed for the string
        regex : string
            A Python regular expression that the string must match

        """
        self.minlen = max( 0, minlen )
        self.maxlen = max( self.minlen, maxlen )
        self.regex  = regex
        self._init()

    def _init ( self ):
        if self.regex != '':
            self.match = re.compile( self.regex ).match
            if (self.minlen == 0) and (self.maxlen == sys.maxint):
                self.validate = self.validate_regex
        elif (self.minlen == 0) and (self.maxlen == sys.maxint):
            self.validate = self.validate_str
        else:
            self.validate = self.validate_len

    def validate ( self, object, name, value ):
        try:
            value = strx( value )
            if ((self.minlen <= len( value ) <= self.maxlen) and
                (self.match( value ) is not None)):
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def validate_str ( self, object, name, value ):
        try:
            return strx( value )
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def validate_len ( self, object, name, value ):
        try:
            value = strx( value )
            if self.minlen <= len( value ) <= self.maxlen:
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def validate_regex ( self, object, name, value ):
        try:
            value = strx( value )
            if self.match( value ) is not None:
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        msg = ''
        if (self.minlen != 0) and (self.maxlen != sys.maxint):
            msg = ' between %d and %d characters long' % (
                  self.minlen, self.maxlen )
        elif self.maxlen != sys.maxint:
            msg = ' <= %d characters long' % self.maxlen
        elif self.minlen != 0:
            msg = ' >= %d characters long' % self.minlen
        if self.regex != '':
            if msg != '':
                msg += ' and'
            msg += (" matching the pattern '%s'" % self.regex)
        return 'a string' + msg

    def __getstate__ ( self ):
        result = self.__dict__.copy()
        for name in [ 'validate', 'match' ]:
            if name in result:
                del result[ name ]
        return result

    def __setstate__ ( self, state ):
        self.__dict__.update( state )
        self._init()

#-------------------------------------------------------------------------------
#  'TraitType' class:
#-------------------------------------------------------------------------------

class TraitType ( TraitHandler ):
    """Ensures that a value assigned to a trait attribute is of a specified
    Python type, or can be coerced to the specified type.

    TraitType is the underlying handler for the predefined traits and factories
    for Python simple types. The TraitType class is also an example of a
    parameterized type, because the single TraitType class allows creating
    instances that check for totally different sets of values. For example::

        class Person(HasTraits):
            name = Trait('', TraitType(''))
            weight = Trait(0.0, TraitType(float))

    In this example, the **name** attribute must be of type ``str`` (string),
    while the **weight** attribute must be of type ``float``, although both are
    based on instances of the TraitType class. Note that this example is
    essentially the same as writing::

        class Person(HasTraits):
            name = Trait('')
            weight = Trait(0.0)

    This simpler form is automatically changed by the Trait() function into
    the first form, based on TraitType instances, when the trait attributes
    are defined.

    For attributes based on TraitType instances, if a value that is assigned is
    not of the type defined for the trait, a TraitError exception is raised.
    However, in certain cases, if the value can be coerced to the required type,
    then the coerced value is assigned to the attribute. Only *widening*
    coercions are allowed, to avoid any possible loss of precision. The
    following table lists the allowed coercions.

    ============ =================
     Trait Type   Coercible Types
    ============ =================
    complex      float, int
    float        int
    long         int
    unicode      str
    ============ =================
    """
    def __init__ ( self, aType ):
        """ Creates a TraitType handler.

        Parameters
        ----------
        aType : type
            Either a Python type (e.g., ``str`` or types.StringType) or a
            Python value (e.g., 'cat')

        Description
        -----------
        If *aType* is a value, it is mapped to its corresponding type. For
        example, the string 'cat' is automatically mapped to ``str`` (i.e.,
        types.StringType).
        """
        if not isinstance( aType, TypeType ):
            aType = type( aType )
        self.aType = aType
        try:
            self.fast_validate = CoercableTypes[ aType ]
        except:
            self.fast_validate = ( 11, aType )

    def validate ( self, object, name, value ):
        fv = self.fast_validate
        tv = type( value )

        # If the value is already the desired type, then return it:
        if tv is fv[1]:
            return value

        # Else see if it is one of the coercable types:
        for typei in fv[2:]:
            if tv is typei:
                # Return the coerced value:
                return fv[1]( value )

        # Otherwise, raise an exception:
        if tv is InstanceType:
            kind = class_of( value )
        else:
            kind = repr( value )
        self.error( object, name, '%s (i.e. %s)' % ( str( tv )[1:-1], kind ) )

    def info ( self ):
        return 'a value of %s' % str( self.aType )[1:-1]

    def get_editor ( self, trait ):

        # Make the special case of a 'bool' type use the boolean editor:
        if self.aType is bool:
            if self.editor is None:
                from enthought.traits.ui.api import BooleanEditor
                self.editor = BooleanEditor()
            return self.editor

        # Otherwise, map all other types to a text editor:
        auto_set = trait.auto_set
        if auto_set is None:
            auto_set = True
        from enthought.traits.ui.api import TextEditor
        return TextEditor( auto_set  = auto_set,
                           enter_set = trait.enter_set or False,
                           evaluate  = self.fast_validate[1] )

#-------------------------------------------------------------------------------
#  'TraitCastType' class:
#-------------------------------------------------------------------------------

class TraitCastType ( TraitType ):
    """Ensures that a value assigned to a trait attribute is of a specified
    Python type, or can be cast to the specified type.

    This class is similar to TraitType, but uses casting rather than coercion.
    Values are cast by calling the type with the value to be assigned as an
    argument. When casting is performed, the result of the cast is the value
    assigned to the trait attribute.

    Any trait that uses a TraitCastType instance in its definition ensures that
    its value is of the type associated with the TraitCastType instance. For
    example::

        class Person(HasTraits):
            name = Trait('', TraitCastType(''))
            weight = Trait(0.0, TraitCastType(float))

    In this example, the **name** trait must be of type ``str`` (string), while
    the **weight** trait must be of type ``float``. Note that this example is
    essentially the same as writing::

        class Person(HasTraits):
            name = CStr
            weight = CFloat

    To understand the difference between TraitType and TraitCastType (and also
    between Float and CFloat), consider the following example::

        >>>class Person(HasTraits):
        ...    weight = Float
        ...    cweight = CFloat
        >>>
        >>>bill = Person()
        >>>bill.weight = 180    # OK, coerced to 180.0
        >>>bill.cweight = 180   # OK, cast to 180.0
        >>>bill.weight = '180'  # Error, invalid coercion
        >>>bill.cweight = '180' # OK, cast to float('180')
    """
    def __init__ ( self, aType ):
        """ Creates a TraitCastType handler.

        Parameters
        ----------
        aType : type
            Either a Python type (e.g., ``str`` or types.StringType) or a
            Python value (e.g., ``'cat``)

        Description
        -----------
        If *aType* is a Python value, it is automatically mapped to its
        corresponding Python type. For example, the string 'cat' is
        automatically mapped to ``str`` (i.e., types.StringType).

        """
        if not isinstance( aType, TypeType ):
            aType = type( aType )
        self.aType = aType
        self.fast_validate = ( 12, aType )

    def validate ( self, object, name, value ):

        # If the value is already the desired type, then return it:
        if type( value ) is self.aType:
            return value

        # Else try to cast it to the specified type:
        try:
            return self.aType( value )
        except:
            # Otherwise, raise an exception:
            tv = type( value )
            if tv is InstanceType:
                kind = class_of( value )
            else:
                kind = repr( value )
            self.error( object, name, '%s (i.e. %s)' % (
                                      str( tv )[1:-1], kind ) )

#-------------------------------------------------------------------------------
#  'ThisClass' class:
#-------------------------------------------------------------------------------

class ThisClass ( TraitHandler ):
    """Ensures that the trait attribute values belong to the same class (or
    a subclass) as the object containing the trait attribute.

    ThisClass is the underlying handler for the predefined traits **This** and
    **self**, and the elements of ListThis.
    """
    def __init__ ( self, or_none = 0 ):
        """Creates a ThisClass handler.

        Parameters
        ----------
        or_none : boolean
            Flag indicating whether None is accepted as a valid value
            (True or non-zero) or not (False or 0)
        """
        if or_none is None:
            self.allow_none()
            self.fast_validate = ( 2, None )
        else:
            self.fast_validate = ( 2, )

    def validate ( self, object, name, value ):
        if isinstance( value, object.__class__ ):
            return value
        self.validate_failed( object, name, value )

    def validate_none ( self, object, name, value ):
        if isinstance( value, object.__class__ ) or (value is None):
            return value
        self.validate_failed( object, name, value )

    def info ( self ):
        return 'an instance of the same type as the receiver'

    def info_none ( self ):
        return 'an instance of the same type as the receiver or None'

    def validate_failed ( self, object, name, value ):
        kind = type( value )
        if kind is InstanceType:
            msg = 'class %s' % value.__class__.__name__
        else:
            msg = '%s (i.e. %s)' % ( str( kind )[1:-1], repr( value ) )
        self.error( object, name, msg )

    def get_editor ( self, trait ):
        if self.editor is None:
            from enthought.traits.ui.api import InstanceEditor
            self.editor = InstanceEditor( label = trait.label or '',
                                          view  = trait.view  or '',
                                          kind  = trait.kind  or 'live' )
        return self.editor

#-------------------------------------------------------------------------------
#  'TraitInstance' class:
#-------------------------------------------------------------------------------

class TraitInstance ( ThisClass ):
    """Ensures that trait attribute values belong to a specified Python class
    or type.

    TraitInstance is the underlying handler for the predefined trait
    **Instance** and the elements of ListInstance.

    Any trait that uses a TraitInstance handler ensures that its values belong
    to the specified type or class (or one of its subclasses). For example::

        class employee(HasTraits):
            manager = Trait(None, TraitInstance(Employee, True))

    This example defines a class Employee, which has a **manager** trait
    attribute, which accepts either None or an instance of Employee
    as its value.

    TraitInstance ensures that assigned values are exactly of the type specified
    (i.e., no coercion is performed).
    """
    def __init__ ( self, aClass, or_none = False, module = '' ):
        """Creates a TraitInstance handler.

        Parameters
        ----------
        aClass : class or type
            A Python class, an instance of a Python class, or a Python type
        or_none: boolean
            Flag indicating whether None is accepted as a valid value
            (True or non-zero) or not (False or 0)
        module : module
            The module that the class belongs to

        Description
        -----------
        If *aClass* is an instance, it is mapped to the class it is an instance
        of.
        """
        if aClass is None:
            aClass, or_none = or_none, aClass
        self.or_none = (or_none != False)
        self.module  = module
        if isinstance(aClass, basestring):
            self.aClass = aClass
        else:
            if not isinstance( aClass, ClassTypes ):
                aClass = aClass.__class__
            self.aClass = aClass
            self.set_fast_validate()

    def allow_none ( self ):
        self.or_none = True
        if hasattr( self, 'fast_validate' ):
            self.set_fast_validate()

    def set_fast_validate ( self ):
        fast_validate = [ 1, self.aClass ]
        if self.or_none:
            fast_validate = [ 1, None, self.aClass ]
        if self.aClass in TypeTypes:
            fast_validate[0] = 0
        self.fast_validate = tuple( fast_validate )

    def validate ( self, object, name, value ):
        if value is None:
            if self.or_none:
                return value
            else:
                self.validate_failed( object, name, value )
        if isinstance(self.aClass, basestring):
            self.resolve_class( object, name, value )
        if isinstance( value, self.aClass ):
            return value
        self.validate_failed( object, name, value )

    def info ( self ):
        aClass = self.aClass
        if type( aClass ) is not str:
            aClass = aClass.__name__
        result = class_of( aClass )
        if self.or_none is None:
            return result + ' or None'
        return result

    def resolve_class ( self, object, name, value ):
        aClass = self.find_class()
        if aClass is None:
            self.validate_failed( object, name, value )
        self.aClass = aClass

        # fixme: The following is quite ugly, because it wants to try and fix
        # the trait referencing this handler to use the 'fast path' now that the
        # actual class has been resolved. The problem is finding the trait,
        # especially in the case of List(Instance('foo')), where the
        # object.base_trait(...) value is the List trait, not the Instance
        # trait, so we need to check for this and pull out the List
        # 'item_trait'. Obviously this does not extend well to other traits
        # containing nested trait references (Dict?)...
        self.set_fast_validate()
        trait   = object.base_trait( name )
        handler = trait.handler
        if (handler is not self) and hasattr( handler, 'item_trait' ):
            trait = handler.item_trait
        trait.set_validate( self.fast_validate )

    def find_class ( self ):
        module = self.module
        aClass = self.aClass
        col    = aClass.rfind( '.' )
        if col >= 0:
            module = aClass[ : col ]
            aClass = aClass[ col + 1: ]
        theClass = getattr( sys.modules.get( module ), aClass, None )
        if (theClass is None) and (col >= 0):
            try:
                mod = __import__( module )
                for component in module.split( '.' )[1:]:
                    mod = getattr( mod, component )
                theClass = getattr( mod, aClass, None )
            except:
                pass
        return theClass

    def create_default_value ( self, *args, **kw ):
        aClass = self.aClass
        if isinstance(aClass, basestring):
            aClass = self.find_class()
            if aClass is None:
                raise TraitError, 'Unable to locate class: ' + self.aClass
        return aClass( *args, **kw )

#-------------------------------------------------------------------------------
#  'TraitWeakRef' class:
#-------------------------------------------------------------------------------

class TraitWeakRef ( TraitInstance ):

    def _get ( self, object, name ):
        value = getattr( object, name + '_', None )
        if value is not None:
            return value()
        return None

    def _set ( self, object, name, value ):
        if value is not None:
            value = ref( value )
        object.__dict__[ name + '_' ] = value

    def resolve_class ( self, object, name, value ):
        # fixme: We have to override this method to prevent the 'fast validate'
        # from being set up, since the trait using this is a 'property' style
        # trait which is not currently compatible with the 'fast_validate'
        # style (causes internal Python SystemError messages).
        aClass = self.find_class()
        if aClass is None:
            self.validate_failed( object, name, value )
        self.aClass = aClass

#-------------------------------------------------------------------------------
#  'TraitClass' class:
#-------------------------------------------------------------------------------

class TraitClass ( TraitHandler ):
    """Ensures that trait attribute values are subclasses of a specified class
    (or the class itself).

    A value is valid if it is a subclass of the specified class (including the
    class itself), or it is a string that is equivalent to the name of a valid
    class.
    """
    def __init__ ( self, aClass ):
        """Creates a TraitClass handler.

        Parameters
        ----------
        aClass : class
            A Python class

        Description
        -----------
        If *aClass* is an instance, it is mapped to the class it is an instance
        of.
        """
        if type( aClass ) is InstanceType:
            aClass = aClass.__class__
        self.aClass = aClass

    def validate ( self, object, name, value ):
        try:
            if isinstance(value, basestring):
                value = value.strip()
                col   = value.rfind( '.' )
                if col >= 0:
                    module_name = value[:col]
                    class_name  = value[col + 1:]
                    module      = sys.modules.get( module_name )
                    if module is None:
                        exec( 'import ' + module_name )
                        module = sys.modules[ module_name ]
                    value = getattr( module, class_name )
                else:
                    value = globals().get( value )

            if issubclass( value, self.aClass ):
                return value
        except:
            pass

        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return 'a subclass of ' + self.aClass.__name__

#-------------------------------------------------------------------------------
#  'TraitFunction' class:
#-------------------------------------------------------------------------------

class TraitFunction ( TraitHandler ):
    """Ensures that assigned trait attribute values are acceptable to a
    specified validator function.

    TraitFunction is the underlying handler for the predefined trait
    **Function**, and for the use of function references as arguments to the
    Trait() function.
    """
    def __init__ ( self, aFunc ):
        """ Creates a TraitFunction handler.

        Parameters
        ----------
        aFunc : function
            A function to validate trait attribute values

        Description
        -----------
        The signature of the function passed as an argument must be of the
        form *function* ( *object*, *name*, *value* ). The function must
        verify that *value* is a legal value for the *name* trait attribute
        of *object*. If it is, the value returned by the fucntion is the
        actual value assigned to the trait attribute. If it is not, the
        function must raise a TraitError exception.
        """
        if not isinstance( aFunc, CallableTypes ):
            raise TraitError, "Argument must be callable."
        self.aFunc = aFunc
        self.fast_validate = ( 13, aFunc )

    def validate ( self, object, name, value ):
        try:
            return self.aFunc( object, name, value )
        except TraitError:
            self.error( object, name, self.repr( value ) )

    def info ( self ):
        try:
            return self.aFunc.info
        except:
            if self.aFunc.__doc__:
                return self.aFunc.__doc__
            return 'a legal value'

#-------------------------------------------------------------------------------
#  'TraitEnum' class:
#-------------------------------------------------------------------------------

class TraitEnum ( TraitHandler ):
    """ Ensures that a value assigned to a trait attribute is a member of a
    specified list of values.

    TraitEnum is the underlying handler for the forms of the Trait() function
    that take a list of possible values
    """
    def __init__ ( self, *values ):
        """ Creates a TraitEnum handler.

        Parameters
        ----------
        values : list or tuple
            Enumeration of all legal values for a trait

        Description
        -----------
        The list of legal values can be provided as a list of values. That is,
        ``TraitEnum([1, 2, 3])`` and ``TraitEnum(1, 2, 3)`` are equivalent. For
        example::

            class Flower(HasTraits):
                color = Trait('white', TraitEnum(['white', 'yellow', 'red']))
                kind  = Trait('annual', TraitEnum('annual', 'perennial'))

        This example defines a Flower class, which has a **color** trait
        attribute, which can have as its value, one of the three strings,
        'white', 'yellow', or 'red', and a **kind** trait attribute, which can
        have as its value, either of the strings 'annual' or 'perennial'. This
        is equivalent to the following class definition::

            class Flower(HasTraits):
                color = Trait(['white', 'yellow', 'red'])
                kind  = Trait('annual', 'perennial')

        The Trait() function automatically maps traits of the form shown in
        this example to the form shown in the preceding example whenever it
        encounters them in a trait definition.
        """
        if (len( values ) == 1) and (type( values[0] ) in SequenceTypes):
            values = values[0]
        self.values        = tuple( values )
        self.fast_validate = ( 5, self.values )

    def validate ( self, object, name, value ):
        if value in self.values:
            return value
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return ' or '.join( [ repr( x ) for x in self.values ] )

    def get_editor ( self, trait ):
        from enthought.traits.ui.api import EnumEditor
        return EnumEditor( values   = self,
                           cols     = trait.cols or 3,
                           evaluate = trait.evaluate,
                           mode     = trait.mode or 'radio' )

#-------------------------------------------------------------------------------
#  'TraitPrefixList' class:
#-------------------------------------------------------------------------------

class TraitPrefixList ( TraitHandler ):
    """Ensures that a value assigned to a trait attribute is a member of a list
    of specified string values, or is a unique prefix of one of those values.

    TraitPrefixList is a variation on TraitEnum. The values that can be
    assigned to a trait attribute defined using a TraitPrefixList handler is the
    set of all strings supplied to the TraitPrefixList constructor, as well as
    any unique prefix of those strings. That is, if the set of strings supplied
    to the constructor is described by [*s*\ :sub:`1`\ , *s*\ :sub:`2`\ , ...,
    *s*\ :sub:`n`\ ], then the string *v* is a valid value for the trait if
    *v* == *s*\ :sub:`i[:j]` for one and only one pair of values (i, j). If *v*
    is a valid value, then the actual value assigned to the trait attribute is
    the corresponding *s*\ :sub:`i` value that *v* matched. For example::

        class Person(HasTraits):
            married = Trait('no', TraitPrefixList('yes', 'no')

    The Person class has a **married** trait that accepts any of the
    strings 'y', 'ye', 'yes', 'n', or 'no' as valid values. However, the actual
    values assigned as the value of the trait attribute are limited to either
    'yes' or 'no'. That is, if the value 'y' is assigned to the **married**
    attribute, the actual value assigned will be 'yes'.

    Note that the algorithm used by TraitPrefixList in determining whether a
    string is a valid value is fairly efficient in terms of both time and space,
    and is not based on a brute force set of comparisons.

    """
    def __init__ ( self, *values ):
        """ Creates a TraitPrefixList handler.

        Parameters
        ----------
        values : list or tuple of strings
            Enumeration of all legal values for a trait

        Description
        -----------
        As with TraitEnum, the list of legal values can be provided as a list
        of values.  That is, ``TraitPrefixList(['one', 'two', 'three'])`` and
        ``TraitPrefixList('one', 'two', 'three')`` are equivalent.
        """
        if (len( values ) == 1) and (type( values[0] ) in SequenceTypes):
            values = values[0]
        self.values  = values[:]
        self.values_ = values_ = {}
        for key in values:
            values_[ key ] = key
        self.fast_validate = ( 10, values_, self.validate )

    def validate ( self, object, name, value ):
        try:
            if not self.values_.has_key( value ):
                match = None
                n     = len( value )
                for key in self.values:
                    if value == key[:n]:
                        if match is not None:
                           match = None
                           break
                        match = key
                if match is None:
                    self.error( object, name, self.repr( value ) )
                self.values_[ value ] = match
            return self.values_[ value ]
        except:
            self.error( object, name, self.repr( value ) )

    def info ( self ):
        return (' or '.join( [ repr( x ) for x in self.values ] ) +
                ' (or any unique prefix)')

    def get_editor ( self, trait ):
        from enthought.traits.ui.api import EnumEditor
        return EnumEditor( values = self,
                           cols   = trait.cols or 3  )

    def __getstate__ ( self ):
        result = self.__dict__.copy()
        if 'fast_validate' in result:
            del result[ 'fast_validate' ]
        return result

#-------------------------------------------------------------------------------
#  'TraitMap' class:
#-------------------------------------------------------------------------------

class TraitMap ( TraitHandler ):
    """Checks that the value assigned to a trait attribute is a key of a
    specified dictionary, and also assigns the dictionary value corresponding
    to that key to a *shadow* attribute.

    A trait attribute that uses a TraitMap handler is called *mapped* trait
    attribute. In practice, this means that the resulting object actually
    contains two attributes: one whose value is a key of the TraitMap
    dictionary, and the other whose value is the corresponding value of the
    TraitMap dictionary. The name of the shadow attribute is simply the base
    attribute name with an underscore ('_') appended. Mapped trait attributes
    can be used to allow a variety of user-friendly input values to be mapped to
    a set of internal, program-friendly values. For example::

        >>>class Person(HasTraits):
        ...    married = Trait('yes', TraitMap({'yes': 1, 'no': 0 })
        >>>
        >>>bob = Person()
        >>>print bob.married
        yes
        >>>print bob.married_
        1

    In this example, the default value of the **married** attribute of the
    Person class is 'yes'. Because this attribute is defined using
    TraitPrefixList, instances of Person have another attribute,
    **married_**, whose default value is 1, the dictionary value corresponding
    to the key 'yes'.
    """
    is_mapped = True

    def __init__ ( self, map ):
        """ Creates a TraitMap handler.

        Parameters
        ----------
        map : dictionary
            A dictionary whose keys are valid values for the trait attribute,
            and whose corresponding values are the values for the shadow
            trait attribute.
        """
        self.map = map
        self.fast_validate = ( 6, map )

    def validate ( self, object, name, value ):
        try:
            if self.map.has_key( value ):
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def mapped_value ( self, value ):
        return self.map[ value ]

    def post_setattr ( self, object, name, value ):
        try:
            setattr( object, name + '_', self.mapped_value( value ) )
        except:
            # We don't need a fancy error message, because this exception
            # should always be caught by a TraitCompound handler:
            raise TraitError, 'Unmappable'

    def info ( self ):
        keys = [ repr( x ) for x in self.map.keys() ]
        keys.sort()
        return ' or '.join( keys )

    def get_editor ( self, trait ):
        from enthought.traits.ui.api import EnumEditor
        return EnumEditor( values = self,
                           cols   = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitPrefixMap' class:
#-------------------------------------------------------------------------------

class TraitPrefixMap ( TraitMap ):
    """A cross between the TraitPrefixList and TraitMap classes.

    Like TraitMap, TraitPrefixMap is created using a dictionary, but in this
    case, the keys of the dictionary must be strings. Like TraitPrefixList,
    a string *v* is a valid value for the trait attribute if it is a prefix of
    one and only one key *k* in the dictionary. The actual values assigned to
    the trait attribute is *k*, and its corresponding mapped attribute is
    *map*[*k*].
    """
    def __init__ ( self, map ):
        """Creates a TraitPrefixMap handler.

        Parameters
        ----------
        map : dictionary
            A dictionary whose keys are strings that are valid values for the
            trait attribute, and whose corresponding values are the values for
            the shadow trait attribute.
        """
        self.map  = map
        self._map = _map = {}
        for key in map.keys():
            _map[ key ] = key
        self.fast_validate = ( 10, _map, self.validate )

    def validate ( self, object, name, value ):
        try:
            if not self._map.has_key( value ):
                match = None
                n     = len( value )
                for key in self.map.keys():
                    if value == key[:n]:
                        if match is not None:
                           match = None
                           break
                        match = key
                if match is None:
                    self.error( object, name, self.repr( value ) )
                self._map[ value ] = match
            return self._map[ value ]
        except:
            self.error( object, name, self.repr( value ) )

    def info ( self ):
        return TraitMap.info( self ) + ' (or any unique prefix)'

#-------------------------------------------------------------------------------
#  'TraitExpression' class:
#-------------------------------------------------------------------------------

class TraitExpression ( TraitHandler ):
    """ Ensures that a value assigned to a trait attribute is a valid Python
        expression. The compiled form of a valid expression is stored as the
        mapped value of the trait.
    """

    def validate ( self, object, name, value ):
        try:
            compile( value, '<string>', 'eval' )
            return value
        except:
            self.error( object, name, self.repr( value ) )

    def post_setattr ( self, object, name, value ):
        object.__dict__[ name + '_' ] = compile( value, '<string>', 'eval' )

    def info ( self ):
        return 'a valid Python expression'

#-------------------------------------------------------------------------------
#  'TraitCompound' class:
#-------------------------------------------------------------------------------

class TraitCompound ( TraitHandler ):
    """ Provides a logical-OR combination of other trait handlers.

    This class provides a means of creating complex trait definitions by
    combining several simpler trait definitions. TraitCompound is the underlying
    handler for the general forms of the Trait() function.

    A value is a valid value for a trait attribute based on a TraitCompound
    instance if the value is valid for at least one of the TraitHandler or
    trait objects supplied to the constructor. In addition, if at least one of
    the TraitHandler or trait objects is mapped (e.g., based on a TraitMap or
    TraitPrefixMap instance), then the TraitCompound is also mapped. In this
    case, any non-mapped traits or trait handlers use identity mapping.

    """
    def __init__ ( self, *handlers ):
        """ Creates a TraitCompound handler.

        Parameters
        ----------
        handlers : list or tuple of TraitHandler or trait objects
            The trait handlers to be combined

        Description
        -----------
        The TraitHandler or trait objects can be provided directly as
        arguments to the constructor.
        """
        if (len( handlers ) == 1) and (type( handlers[0] ) in SequenceTypes):
            handlers = handlers[0]
        self.handlers       = handlers
        mapped_handlers     = []
        post_setattrs       = []
        self.validates      = validates = []
        self.slow_validates = slow_validates = []
        fast_validates      = []
        self.reversable     = True
        for handler in handlers:
            if hasattr( handler, 'fast_validate' ):
                validates.append( handler.validate )
                fv = handler.fast_validate
                if fv[0] == 7:
                    # If this is a nested complex fast validator, expand its
                    # contents and adds its list to our list:
                    fast_validates.extend( fv[1] )
                else:
                    # Else just add the entire validator to the list:
                    fast_validates.append( fv )
            else:
                slow_validates.append( handler.validate )
            if hasattr( handler, 'post_setattr' ):
                post_setattrs.append( handler.post_setattr )
            if handler.is_mapped:
                self.is_mapped = True
                mapped_handlers.append( handler )
                self.mapped_handlers = mapped_handlers
            else:
                self.reversable = False
            if handler.has_items:
                self.has_items = True

        # If there are any fast validators, then we create a 'complex' fast
        # validator that composites them:
        if len( fast_validates ) > 0:
            # If there are any 'slow' validators, add a special handler at
            # the end of the fast validator list to handle them:
            if len( slow_validates ) > 0:
                fast_validates.append( ( 8, self ) )
            # Create the 'complex' fast validator:
            self.fast_validate = ( 7, tuple( fast_validates ) )

        if len( post_setattrs ) > 0:
            self.post_setattrs = post_setattrs
            self.post_setattr  = self._post_setattr

    def validate ( self, object, name, value ):
        for validate in self.validates:
            try:
               return validate( object, name, value )
            except TraitError:
               pass
        return self.slow_validate( object, name, value )

    def slow_validate ( self, object, name, value ):
        for validate in self.slow_validates:
            try:
               return validate( object, name, value )
            except TraitError:
               pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return ' or '.join( [ x.info() for x in self.handlers ] )

    def mapped_value ( self, value ):
        for handler in self.mapped_handlers:
            try:
                return handler.mapped_value( value )
            except:
                pass
        return value

    def _post_setattr ( self, object, name, value ):
        for post_setattr in self.post_setattrs:
            try:
                post_setattr( object, name, value )
                return
            except TraitError:
               pass
        setattr( object, name + '_', value )

    def get_editor ( self, trait ):
        from enthought.traits.ui.api import TextEditor, CompoundEditor

        the_editors = [ x.get_editor( trait ) for x in self.handlers ]
        text_editor = TextEditor()
        count       = 0
        editors     = []
        for editor in the_editors:
            if isinstance( text_editor, editor.__class__ ):
                count += 1
                if count > 1:
                    continue
            editors.append( editor )
        return CompoundEditor( editors = editors )

    def items_event ( self ):
        return items_event()

#-------------------------------------------------------------------------------
#  'TraitTuple' class:
#-------------------------------------------------------------------------------

class TraitTuple ( TraitHandler ):
    """ Ensures that values assigned to a trait attribute are tuples of a
    specified length, with elements that are of specified types.

    TraitTuple is the underlying handler for the predefined trait **Tuple**,
    and the trait factory Tuple().

    For example::

        rank = Range(1, 13)
        suit = Trait('Hearts', 'Diamonds', 'Spades', 'Clubs')
        class Card(HasTraits):
            value = Trait(TraitTuple(rank, suit))

    This example defines a Card class, which has a **value** trait attribute,
    which must be a tuple of two elments. The first element must be an integer
    in the range from 1 to 13, and the second element must be one of the four
    strings, 'Hearts', 'Diamonds', 'Spades', or 'Clubs'.
    """
    def __init__ ( self, *args ):
        """ Creates a TraitTuple handler.

        Parameters
        ----------
        args : list of traits
            Each *trait*\ :sub:`i` specifies the type that the *i*\ th element of a
            tuple must be.

        Description
        -----------
        Each *trait*\ :sub:`i` must be either a trait, or a value that can be
        converted to a trait using the Trait() function. The resulting
        trait handler accepts values that are tuples of the same length as
        *args*, and whose *i*\ th element is of the type specified by
        *trait*\ :sub:`i`.
        """
        self.traits = tuple( [ trait_from( arg ) for arg in args ] )
        self.fast_validate = ( 9, self.traits )

    def validate ( self, object, name, value ):
        try:
            if isinstance( value, tuple ):
                traits = self.traits
                if len( value ) == len( traits ):
                    values = []
                    for i, trait in enumerate( traits ):
                        values.append( trait.handler.validate( object, name,
                                                               value[i] ) )
                    return tuple( values )
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return 'a tuple of the form: (%s)' % (', '.join(
               [ self._trait_info( trait ) for trait in self.traits ] ))

    def _trait_info ( self, trait ):
        handler = trait.handler
        if handler is None:
            return 'any value'
        else:
            return handler.info()

    def get_editor ( self, trait ):
        from enthought.traits.ui.api import TupleEditor
        return TupleEditor( traits = self.traits,
                            labels = trait.labels or [],
                            cols   = trait.cols   or 1  )

#-------------------------------------------------------------------------------
#  'TraitCallable' class:
#-------------------------------------------------------------------------------

class TraitCallable ( TraitHandler ):
    """Ensures that the value of a trait attribute is a callable Python object
    (usually a function or method).
    """
    def validate ( self, object, name, value ):
        if (value is None) or callable( value ):
            return value
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return 'a callable value'

#-------------------------------------------------------------------------------
#  'TraitListEvent' class:
#-------------------------------------------------------------------------------

class TraitListEvent ( object ):

    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, index, removed = None, added = None ):
        self.index = index
        if removed is None:
            self.removed = []
        else:
            self.removed = removed
        if added is None:
            self.added = []
        else:
            self.added = added

#-------------------------------------------------------------------------------
#  'TraitList' class:
#-------------------------------------------------------------------------------

class TraitList ( TraitHandler ):
    """ Ensures that a value assigned to a trait attribute is a list containing
    elements of a specified type, and that the length of the list is also
    within a specified range.

    TraitList also makes sure that any changes made to the list after it is
    assigned to the trait attribute do not violate the list's type and length
    constraints. TraitList is the underlying handler for the predefined
    list-based traits.

    For example::

        class Card(HasTraits):
            pass
        class Hand(HasTraits):
            cards = Trait([], TraitList(Trait(Card), maxlen=52))

    This example defines a Hand class, which has a **cards** trait attribute,
    which is a list of Card objects and can have from 0 to 52 items in the
    list.
    """
    info_trait         = None
    default_value_type = 5
    _items_event       = None

    def __init__ ( self, trait = None, minlen = 0, maxlen = sys.maxint,
                         has_items = True ):
        """ Creates a TraitList handler.

        Parameters
        ----------
        trait : trait
            The type of items the list can contain
        minlen : integer
            The minimum length of the list
        maxlen : integer
            The maximum length of the list
        has_items : boolean
            Flag indicating whether the list contains elements

        Description
        -----------
        If *trait* is None or omitted, then no type checking is performed
        on any items in the list; otherwise, *trait* must be either a trait, or
        a value that can be converted to a trait using the Trait() function.

        """
        self.item_trait = trait_from( trait )
        self.minlen     = max( 0, minlen )
        self.maxlen     = max( minlen, maxlen )
        self.has_items  = has_items

    def clone ( self ):
        return TraitList( self.item_trait, self.minlen, self.maxlen,
                          self.has_items )

    def validate ( self, object, name, value ):
        if (isinstance( value, list ) and
           (self.minlen <= len( value ) <= self.maxlen)):
            return TraitListObject( self, object, name, value )
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        if self.minlen == 0:
            if self.maxlen == sys.maxint:
                size = 'items'
            else:
                size = 'at most %d items' % self.maxlen
        else:
            if self.maxlen == sys.maxint:
                size = 'at least %d items' % self.minlen
            else:
                size = 'from %s to %s items' % (
                       self.minlen, self.maxlen )
        handler = self.item_trait.handler
        if handler is None:
            info = ''
        else:
            info = ' each of which is %s' % handler.info()
        return 'a list of %s%s' % ( size, info )

    def get_editor ( self, trait ):
        handler = self.item_trait.handler
        if isinstance( handler, TraitInstance ) and (trait.mode != 'list'):
            from enthought.traits.api import HasTraits
            if issubclass( handler.aClass, HasTraits ):
                try:
                    object = handler.aClass()
                    from enthought.traits.ui.table_column import ObjectColumn
                    from enthought.traits.ui.table_filter import \
                         EvalFilterTemplate, RuleFilterTemplate, \
                         MenuFilterTemplate, EvalTableFilter
                    from enthought.traits.ui.api import TableEditor
                    return TableEditor(
                            columns = [ ObjectColumn( name = name )
                                        for name in object.editable_traits() ],
                            filters     = [ RuleFilterTemplate,
                                            MenuFilterTemplate,
                                            EvalFilterTemplate ],
                            edit_view   = '',
                            orientation = 'vertical',
                            search      = EvalTableFilter(),
                            deletable   = True,
                            row_factory = handler.aClass )
                except:
                    pass

        from enthought.traits.ui.api import ListEditor
        return ListEditor( trait_handler = self,
                           rows          = trait.rows or 5,
                           use_notebook  = trait.use_notebook is True,
                           page_name     = trait.page_name or '' )

    def items_event ( self ):
        return items_event()

def items_event ( ):
    if TraitList._items_event is None:
        TraitList._items_event = Event( TraitListEvent, is_base = False )
    return TraitList._items_event

#-------------------------------------------------------------------------------
#  'TraitListObject' class:
#-------------------------------------------------------------------------------

class TraitListObject ( list ):

    def __init__ ( self, trait, object, name, value ):
        self.trait      = trait
        self.object     = ref( object )
        self.name       = name
        self.name_items = None
        if trait.has_items:
            self.name_items = name + '_items'

        # Do the validated 'setslice' assignment without raising an
        # 'items_changed' event:
        if trait.minlen <= len( value ) <= trait.maxlen:
            try:
                handler = trait.item_trait.handler
                if handler is not None:
                    validate = handler.validate
                    value    = [ validate( object, name, val ) for val in value]
                list.__setslice__( self, 0, 0, value )
                return
            except TraitError, excp:
                excp.set_prefix( 'Each element of the' )
                raise excp
        self.len_error( len( value ) )

    def __deepcopy__ ( self, memo ):
        id_self = id( self )
        if id_self in memo:
            return memo[ id_self ]
        memo[ id_self ] = result = TraitListObject( self.trait, self.object(),
                         self.name, [ copy.deepcopy( x, memo ) for x in self ] )
        return result

    def __setitem__ ( self, key, value ):
        try:
            removed = [ self[ key ] ]
        except:
            pass
        try:
            handler = self.trait.item_trait.handler
            object  = self.object()
            if handler is not None:
                value = handler.validate( object, self.name, value )
            list.__setitem__( self, key, value )
            if self.name_items is not None:
                if key < 0:
                    key = len( self ) + key
                setattr( object, self.name_items,
                         TraitListEvent( key, removed, [ value ] ) )
        except TraitError, excp:
            excp.set_prefix( 'Each element of the' )
            raise excp

    def __setslice__ ( self, i, j, values ):
        try:
            delta = len( values ) - (min( j, len( self ) ) - max( 0, i ))
        except:
            raise TypeError, 'must assign sequence (not "%s") to slice' % (
                             values.__class__.__name__ )
        if self.trait.minlen <= (len(self) + delta) <= self.trait.maxlen:
            try:
                object  = self.object()
                name    = self.name
                trait   = self.trait.item_trait
                removed = self[ i: j ]
                handler = trait.handler
                if handler is not None:
                    validate = handler.validate
                    values   = [ validate( object, name, value )
                                 for value in values ]
                list.__setslice__( self, i, j, values )
                if self.name_items is not None:
                   setattr( object, self.name_items,
                            TraitListEvent( max( 0, i ), removed, values ) )
                return
            except TraitError, excp:
                excp.set_prefix( 'Each element of the' )
                raise excp
        self.len_error( len( self ) + delta )

    def __delitem__ ( self, key ):
        if self.trait.minlen <= (len( self ) - 1):
            try:
                removed = [ self[ key ] ]
            except:
                pass
            list.__delitem__( self, key )
            if self.name_items is not None:
                if key < 0:
                    key = len( self ) + key + 1
                setattr( self.object(), self.name_items,
                         TraitListEvent( key, removed ) )
            return
        self.len_error( len( self ) - 1 )

    def __delslice__ ( self, i, j ):
        delta = min( j, len( self ) ) - max( 0, i )
        if self.trait.minlen <= (len( self ) - delta):
            removed = self[ i: j ]
            list.__delslice__( self, i, j )
            if self.name_items is not None:
                setattr( self.object(), self.name_items,
                         TraitListEvent( max( 0, i ), removed ) )
            return
        self.len_error( len( self ) - delta )

    def append ( self, value ):
        # It can happen that the 'trait' attribute is not defined when
        # unpickling an object that was pickled using pickle protocol 2, which
        # calls 'append' on the list before the 'setstate' method has been
        # called. So we check for an undefined 'trait' attribute, and if it is
        # undefined, do the append with no type checking, since it presumably
        # is occuring as the result of an unpickling operation that should be
        # setting previously validated values:
        trait = getattr( self, 'trait', None )
        if trait is None:
            list.append( self, value )
            return

        if trait.minlen <= (len( self ) + 1) <= trait.maxlen:
            try:
                handler = trait.item_trait.handler
                object  = self.object()
                if handler is not None:
                    value = handler.validate( object, self.name, value )
                list.append( self, value )
                if self.name_items is not None:
                    setattr( object, self.name_items,
                            TraitListEvent( len( self ) - 1, None, [ value ] ) )
                return
            except TraitError, excp:
                excp.set_prefix( 'Each element of the' )
                raise excp
        self.len_error( len( self ) + 1 )

    def insert ( self, index, value ):
        if self.trait.minlen <= (len( self ) + 1) <= self.trait.maxlen:
            try:
                handler = self.trait.item_trait.handler
                object  = self.object()
                if handler is not None:
                    value = handler.validate( object, self.name, value )
                list.insert( self, index, value )
                if self.name_items is not None:
                    if index < 0:
                        index = len( self ) + index - 1
                    setattr( object, self.name_items,
                             TraitListEvent( index, None, [ value ] ) )
                return
            except TraitError, excp:
                excp.set_prefix( 'Each element of the' )
                raise excp
        self.len_error( len( self ) + 1 )

    def extend ( self, xlist ):
        try:
            len_xlist = len( xlist )
        except:
            raise TypeError, "list.extend() argument must be iterable"

        # See the long comment in the 'append' method for more information:
        trait = getattr( self, 'trait', None )
        if trait is None:
            list.extend( self, xlist )
            return

        if (trait.minlen <= (len( self ) + len_xlist) <= trait.maxlen):
            object  = self.object()
            name    = self.name
            handler = trait.item_trait.handler
            try:
                if handler is not None:
                    validate = handler.validate
                    xlist    = [ validate( object, name, value )
                                 for value in xlist ]
                list.extend( self, xlist )
                if self.name_items is not None:
                    setattr( object, self.name_items,
                             TraitListEvent( len( self ) - len( xlist ), None,
                                             xlist ) )
                return
            except TraitError, excp:
                excp.set_prefix( 'The elements of the' )
                raise excp
        self.len_error( len( self ) + len( xlist ) )

    def remove ( self, value ):
        if self.trait.minlen < len( self ):
            try:
                index   = self.index( value )
                removed = [ self[ index ] ]
            except:
                pass
            list.remove( self, value )
            if self.name_items is not None:
                setattr( self.object(), self.name_items,
                         TraitListEvent( index, removed ) )
        else:
            self.len_error( len( self ) - 1 )

    def len_error ( self, len ):
        raise TraitError, ( "The '%s' trait of %s instance must be %s, "
                  "but you attempted to change its length to %d element%s." % (
                  self.name, class_of( self.object() ), self.trait.info(),
                  len, 's'[ len == 1: ] ) )

    def sort ( self, cmpfunc = None ):
        removed = self[:]
        list.sort( self, cmpfunc )
        if self.name_items is not None:
            setattr( self.object(), self.name_items,
                     TraitListEvent( 0, removed, self[:] ) )

    def reverse ( self ):
        removed = self[:]
        if len( self ) > 1:
            list.reverse( self )
            if self.name_items is not None:
                setattr( self.object(), self.name_items,
                         TraitListEvent( 0, removed, self[:] ) )

    def pop ( self, *args ):
        if self.trait.minlen < len( self ):
            if len( args ) > 0:
                index = args[0]
            else:
                index = -1
            try:
                removed = [ self[ index ] ]
            except:
                pass
            result = list.pop( self, *args )
            if self.name_items is not None:
                if index < 0:
                    index = len( self ) + index + 1
                setattr( self.object(), self.name_items,
                         TraitListEvent( index, removed ) )
            return result
        else:
            self.len_error( len( self ) - 1 )

    def rename ( self, name ):
        trait = self.object()._trait( name, 0 )
        if trait is not None:
            self.name  = name
            self.trait = trait.handler

    def __getstate__ ( self ):
        result = self.__dict__.copy()
        result[ 'object' ] = self.object()
        if 'trait' in result:
            del result[ 'trait' ]

        return result

    def __setstate__ ( self, state ):
        self.__dict__.update( state )
        self.object = ref( self.object )
        self.rename( self.name )


class TraitDictEvent ( object ):

    def __init__ ( self, added = None, changed = None, removed = None ):
        """
        Parameters
        ----------
        added : dictionary
            New keys and values
        changed : dictionary
            Updated keys and their previous values
        removed : dictionary
            Old keys and values that were just removed
        """
        # Construct new empty dicts every time instead of using a default value
        # in the method argument, just in case someone gets the bright idea of
        # modifying the dict they get in-place.
        if added:
            self.added = added
        else:
            self.added = {}
        if changed:
            self.changed = changed
        else:
            self.changed = {}
        if removed:
            self.removed = removed
        else:
            self.removed = {}
        return

#-------------------------------------------------------------------------------
#  'TraitDict' class:
#-------------------------------------------------------------------------------

class TraitDict ( TraitHandler ):
    """ Ensures that values assigned to a trait attribute are dictionaries whose
    keys and values are of specified types.

    TraitDict also makes sure that any changes to keys or values made that are
    made after the dictionary is assigned to the trait attribute satisfy the
    type constraints. TraitDict is the underlying handler for the
    dictionary-based predefined traits, and the Dict() trait factory.

     For example::

            class WorkoutClass(HasTraits):
                member_weights = Trait({}, TraitDict(str, float))

    This example defines a WorkoutClass class containing a *member_weights*
    trait attribute whose value must be a dictionary containing keys that
    are strings (i.e., the members' names) and whose associated values must
    be floats (i.e., their most recently recorded weight).
    """
    info_trait         = None
    default_value_type = 6
    _items_event       = None

    def __init__ ( self, key_trait = None, value_trait = None,
                         has_items = True ):
        """ Creates a TraitDict handler.

        Parameters
        ----------
        key_trait : trait
            The type for the dictionary keys
        value_trait : trait
            The type for the dictionary values
        has_items : boolean
            Flag indicating whether the dictionary contains entries

        Description
        -----------
        If *key_trait* is None or omitted, the keys in the dictionary can
        be of any type. Otherwise, *key_trait* must be either a trait, or a
        value that can be converted to a trait using the Trait() function. In
        this case, all dictionary keys are checked to ensure that they are of
        the type specified by *key_trait*.

        If *value_trait* is None or omitted, the values in the dictionary
        can be of any type. Otherwise, *value_trait* must be either a trait, or
        a value that can be converted to a trait using the Trait() function.
        In this case, all dictionary values are checked to ensure that they are
        of the type specified by *value_trait*.

        """
        self.key_trait   = trait_from( key_trait )
        self.value_trait = trait_from( value_trait )
        self.has_items   = has_items
        handler          = self.value_trait.handler
        if (handler is not None) and handler.has_items:
            handler = handler.clone()
            handler.has_items = False
        self.value_handler = handler

    def clone ( self ):
        return TraitDict( self.key_trait, self.value_trait, self.has_items )

    def validate ( self, object, name, value ):
        if isinstance( value, dict ):
            return TraitDictObject( self, object, name, value )
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        extra   = ''
        handler = self.key_trait.handler
        if handler is not None:
            extra = ' with keys which are %s' % handler.info()
        handler = self.value_handler
        if handler is not None:
            if extra == '':
                extra = ' with'
            else:
                extra += ' and'
            extra += ' values which are %s' % handler.info()
        return 'a dictionary%s' % extra

    def get_editor ( self, trait ):
        if self.editor is None:
            from enthought.traits.ui.api import TextEditor
            self.editor = TextEditor( evaluate = eval )
        return self.editor

    def items_event ( self ):
        if TraitDict._items_event is None:
            TraitDict._items_event = Event( TraitDictEvent, is_base = False )
        return TraitDict._items_event

#-------------------------------------------------------------------------------
#  'TraitDictObject' class:
#-------------------------------------------------------------------------------

class TraitDictObject ( dict ):

    def __init__ ( self, trait, object, name, value ):
        self.trait      = trait
        self.object     = ref( object )
        self.name       = name
        self.name_items = None
        if trait.has_items:
            self.name_items = name + '_items'
        if len( value ) > 0:
            dict.update( self, self._validate_dic( value ) )

    def __setitem__ ( self, key, value ):
        # See the long comment in the TraitListObject 'append' method for more
        # information about why this code is necessary:
        trait = getattr( self, 'trait', None )
        if trait is None:
            dict.__setitem__( self, key, value )
            return

        object = self.object()
        try:
            handler = trait.key_trait.handler
            if handler is not None:
                key = handler.validate( object, self.name, key )
        except TraitError, excp:
            excp.set_prefix( 'Each key of the' )
            raise excp

        try:
            handler = trait.value_handler
            if handler is not None:
                value = handler.validate( object, self.name, value )
            if self.name_items is not None:
                if dict.has_key(self, key):
                    added   = None
                    changed = { key: self[ key ] }
                else:
                    added   = { key: value }
                    changed = None
            dict.__setitem__( self, key, value )
            if self.name_items is not None:
                setattr( object, self.name_items,
                         TraitDictEvent( added, changed ) )
        except TraitError, excp:
            excp.set_prefix( 'Each value of the' )
            raise excp

    def __delitem__ ( self, key ):
        if self.name_items is not None:
            removed = { key: self[ key ] }
        dict.__delitem__( self, key )
        if self.name_items is not None:
            setattr( self.object(), self.name_items,
                     TraitDictEvent( removed = removed ) )

    def clear ( self ):
        if len( self ) > 0:
            if self.name_items is not None:
                removed = self.copy()
            dict.clear( self )
            if self.name_items is not None:
                setattr( self.object(), self.name_items,
                         TraitDictEvent( removed = removed ) )

    def update ( self, dic ):
        if len( dic ) > 0:
            new_dic = self._validate_dic( dic )

            if self.name_items is not None:
                added   = {}
                changed = {}
                for key, value in new_dic.iteritems():
                    if key in self:
                        changed[ key ] = self[ key ]
                    else:
                        added[ key ] = value
                dict.update( self, new_dic )
                setattr( self.object(), self.name_items,
                         TraitDictEvent( added = added, changed = changed ) )
            else:
                dict.update( self, new_dic )

    def setdefault ( self, key, value = None ):
        if self.has_key( key ):
            return self[ key ]

        self[ key ] = value
        result      = self[ key ]
        if self.name_items is not None:
            setattr( self.object(), self.name_items,
                     TraitDictEvent( added = { key: result } ) )
        return result

    def pop ( self, key, value = Undefined ):
        if (value is Undefined) or self.has_key( key ):
            result = dict.pop( self, key )
            if self.name_items is not None:
                setattr( self.object(), self.name_items,
                         TraitDictEvent( removed = { key: result } ) )
            return result
        return value

    def popitem ( self ):
        result = dict.popitem( self )
        if self.name_items is not None:
            setattr( self.object(), self.name_items,
                     TraitDictEvent( removed = { result[0]: result[1] } ) )
        return result

    def rename ( self, name ):
        trait = self.object()._trait( name, 0 )
        if trait is not None:
            self.name  = name
            self.trait = trait.handler
        else:
            logger.debug( "rename: No 'trait' in %s for '%s'" %
                          ( self.object(), name ) )

    def __getstate__ ( self ):
        result = self.__dict__.copy()
        result[ 'object' ] = self.object()
        #del result[ 'trait' ]
        if 'trait' not in result:
            logger.debug( "__getstate__: No 'trait' in %s for '%s'" %
                          ( self.object(), self.name ) )
        else:
            del result[ 'trait' ]
        return result

    def __setstate__ ( self, state ):
        self.__dict__.update( state )
        self.object = ref( self.object )
        self.rename( self.name )

#-- Private Methods ------------------------------------------------------------

    def _validate_dic ( self, dic ):
        name    = self.name
        new_dic = {}

        try:
            key_validate = self.trait.key_trait.handler.validate
        except:
            key_validate = lambda object, name, key: key

        try:
            value_validate = self.trait.value_trait.handler.validate
        except:
            value_validate = lambda object, name, value: value

        object = self.object()
        for key, value in dic.iteritems():
            try:
                key = key_validate( object, name, key )
            except TraitError, excp:
                excp.set_prefix( 'Each key of the' )
                raise excp

            try:
                value = value_validate( object, name, value )
            except TraitError, excp:
                excp.set_prefix( 'Each value of the' )
                raise excp

            new_dic[ key ] = value

        return new_dic

#-------------------------------------------------------------------------------
#  Tell the C-based traits module about 'TraitListObject' and 'TraitDictObject':
#-------------------------------------------------------------------------------

import ctraits
ctraits._list_classes( TraitListObject, TraitDictObject )


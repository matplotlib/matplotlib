#-------------------------------------------------------------------------------
#
#  Define the base 'TraitHandler' class and a standard set of TraitHandler
#  subclasses for use with the 'traits' package.
#
#  A trait handler mediates the assignment of values to object traits. It
#  verifies (through the TraitHandler's 'validate' method) that a specified
#  value is consistent with the object trait, and generates a 'TraitError'
#  exception if not.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  Refactored into a separate module: 07/04/2003
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
#  (c) Copyright 2002, 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
           
import sys           
import re
import copy
           
from types        import StringType, InstanceType, TypeType, FunctionType, \
                         MethodType
from ctraits      import CTraitMethod                            
from trait_base   import strx, SequenceTypes, Undefined, TypeTypes, ClassTypes,\
                         CoercableTypes, class_of, enumerate           
from trait_errors import TraitError
                 
# Patched by 'traits.py' once class is defined!     
Trait   = Event = None  
warning = None

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

RangeTypes    = ( int, float )

CallableTypes = ( FunctionType, MethodType, CTraitMethod )

#-------------------------------------------------------------------------------
#  Forward references:
#-------------------------------------------------------------------------------

trait_from = None  # Patched by 'traits.py' when real 'trait_from' is defined
          
#-------------------------------------------------------------------------------
#  'TraitHandler' class (base class for all trait handlers):
#-------------------------------------------------------------------------------

class TraitHandler ( object ):
    
    default_value_type = -1
    has_items          = False
    is_mapped          = False
    editor             = None
    
    __traits_metadata__ = {
        'type': 'trait'
    }

    def validate ( self, object, name, value ):
        raise TraitError, ( 
              "The '%s' trait of %s instance has an unknown type. "
              "Contact the developer to correct the problem." % (
              name, class_of( object ) ) )
 
    def error ( self, object, name, value ):
        raise TraitError, ( object, name, self.info(), value )
        
    def arg_error ( self, method, arg_num, object, name, value ):
        raise TraitError, ("The '%s' parameter (argument %d) of the %s method "
                           "of %s instance must be %s, but a value of %s was "
                           "specified." % ( name, arg_num, method.tm_name,
                           class_of( object ), self.info(), value ) )
        
    def keyword_error ( self, method, object, name, value ):
        raise TraitError, ("The '%s' keyword argument of the %s method of "
                           "%s instance must be %s, but a value of %s was "
                           "specified." % ( name, method.tm_name,
                           class_of( object ), self.info(), value ) )
        
    def missing_arg_error ( self, method, arg_num, object, name ):
        raise TraitError, ("The '%s' parameter (argument %d) of the %s method "
                           "of %s instance must be specified, but was omitted."
                           % ( name, arg_num, method.tm_name, 
                               class_of( object ) ) )
        
    def dup_arg_error ( self, method, arg_num, object, name ):
        raise TraitError, ("The '%s' parameter (argument %d) of the %s method "
                           "of %s instance was specified as both a positional "
                           "and keyword value."
                           % ( name, arg_num, method.tm_name, 
                               class_of( object ) ) )
        
    def return_error ( self, method, object, value ):
        raise TraitError, ("The result of the %s method of %s instance must "
                           "be %s, but a value of %s was returned." % ( 
                           method.tm_name, class_of( object ), self.info(), 
                           value ) )
 
    def info ( self ):
        return 'a legal value'
 
    def repr ( self, value ):
        if type( value ) is InstanceType:
            return 'class '  + value.__class__.__name__
        return repr( value )
               
    def get_editor ( self, trait ):
        if self.editor is None:
            from matplotlib.enthought.traits.ui import TextEditor
            self.editor = TextEditor()
        return self.editor
        
    def metadata ( self ):
        return getattr( self, '__traits_metadata__', {} )

#-------------------------------------------------------------------------------
#  'TraitRange' class:
#-------------------------------------------------------------------------------

class TraitRange ( TraitHandler ):

    def __init__ ( self, low = None, high = None ):
        self.low  = low
        self.high = high
        vtype     = type( high )
        if (low is not None) and (vtype is not float):
            vtype = type( low )
        if vtype not in RangeTypes:
            raise TraitError, ("TraitRange can only be use for int or float "
                               "values, but a value of type %s was specified." %
                               vtype)
        if vtype is float:
            self.validate = self.float_validate
            kind           = 4
            self.type_desc = 'a floating point number'
            if low is not None:
                low = float( low )
            if high is not None:
                high = float( high )
        else:
            self.validate = self.int_validate
            kind = 3
            self.type_desc = 'an integer'
            if low is not None:
                low = int( low )
            if high is not None:
                high = int( high )
        self.fast_validate = ( kind, low, high )
 
    def float_validate ( self, object, name, value ):
        try:
            if (isinstance( value, RangeTypes ) and
                ((self.low  is None) or (self.low  <= value)) and 
                ((self.high is None) or (self.high >= value))):
               return float( value )
        except:
            pass
        self.error( object, name, self.repr( value ) )
 
    def int_validate ( self, object, name, value ):
        try:
            if (isinstance( value, int ) and
                ((self.low  is None) or (self.low  <= value)) and 
                ((self.high is None) or (self.high >= value))):
               return value
        except:
            pass
        self.error( object, name, self.repr( value ) )
 
    def info ( self ):
        if self.low is None:
            if self.high is None:
                return self.type_desc
            return '%s <= %s' % ( self.type_desc, self.high )
        elif self.high is None:
            return  '%s >= %s' % ( self.type_desc, self.low )
        return '%s in the range from %s to %s' % (
               self.type_desc, self.low, self.high )
               
    def get_editor ( self, trait ):
        auto_set = trait.auto_set
        if auto_set is None:
            auto_set = True
        from matplotlib.enthought.traits.ui import RangeEditor
        return RangeEditor( self, 
                            cols       = trait.cols or 3, 
                            auto_set   = auto_set,
                            enter_set  = trait.enter_set or False,
                            low_label  = trait.low  or '',
                            high_label = trait.high or '' )

#-------------------------------------------------------------------------------
#  'TraitString' class:
#-------------------------------------------------------------------------------

class TraitString ( TraitHandler ):

    def __init__ ( self, minlen = 0, maxlen = sys.maxint, regex = '' ):
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

    def __init__ ( self, aType ):
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
                from matplotlib.enthought.traits.ui import BooleanEditor
                self.editor = BooleanEditor()
            return self.editor
            
        # Otherwise, map all other types to a text editor:
        auto_set = trait.auto_set
        if auto_set is None:
            auto_set = True
        from matplotlib.enthought.traits.ui import TextEditor
        return TextEditor( auto_set  = auto_set,
                           enter_set = trait.enter_set or False,
                           evaluate  = self.fast_validate[1] )

#-------------------------------------------------------------------------------
#  'TraitCastType' class:
#-------------------------------------------------------------------------------

class TraitCastType ( TraitType ):

    def __init__ ( self, aType ):
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

    def __init__ ( self, or_none = 0 ):
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
            from matplotlib.enthought.traits.ui import InstanceEditor
            self.editor = InstanceEditor( label = trait.label or '',
                                          view  = trait.view  or '',
                                          kind  = trait.kind  or 'live' )
        return self.editor
        
class TraitThisClass ( ThisClass ):

    def __init__ ( self, or_none = 0 ):
        warning( "Use of 'TraitThisClass' is deprecated, use 'ThisClass' "
                 "instead." )
        ThisClass.__init__( self, or_none )  

#-------------------------------------------------------------------------------
#  'TraitInstance' class:
#-------------------------------------------------------------------------------

class TraitInstance ( ThisClass ):

    def __init__ ( self, aClass, or_none = False, module = '' ):
        if aClass is None:
            aClass, or_none = or_none, aClass
        self.or_none = (or_none != False)
        self.module  = module
        if type( aClass ) is str:
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
        if type( self.aClass ) is str:
            self.resolve_class( object, name, value )
        if (isinstance( value, self.aClass ) or 
            (self.or_none and (value is None))):
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
        self.set_fast_validate()
        trait = object.base_trait( name )
        trait.validate( self.fast_validate )
        
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
                theClass = getattr( __import__( module ), aClass, None )
            except:
                pass
        return theClass            
        
    def create_default_value ( self, *args, **kw ):
        aClass = self.aClass        
        if type( aClass ) is str:
            aClass = self.find_class()
            if aClass is None:
                raise TraitError, 'Unable to locate class: ' + self.aClass
        return aClass( *args, **kw )

#-------------------------------------------------------------------------------
#  'TraitClass' class:
#-------------------------------------------------------------------------------

class TraitClass ( TraitHandler ):

    def __init__ ( self, aClass ):
        if type( aClass ) is InstanceType:
            aClass = aClass.__class__
        self.aClass = aClass
 
    def validate ( self, object, name, value ):
        try:
            if type( value ) == StringType:
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

    def __init__ ( self, aFunc ):
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

    def __init__ ( self, *values ):
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
        from matplotlib.enthought.traits.ui import EnumEditor
        return EnumEditor( values = self, 
                           cols   = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitPrefixList' class:
#-------------------------------------------------------------------------------

class TraitPrefixList ( TraitHandler ):

    def __init__ ( self, *values ):
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
        from matplotlib.enthought.traits.ui import EnumEditor
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
    
    is_mapped = True

    def __init__ ( self, map ):
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
        from matplotlib.enthought.traits.ui import EnumEditor
        return EnumEditor( values = self, 
                           cols   = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitPrefixMap' class:
#-------------------------------------------------------------------------------

class TraitPrefixMap ( TraitMap ):

    def __init__ ( self, map ):
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
#  'TraitCompound' class:
#-------------------------------------------------------------------------------

class TraitCompound ( TraitHandler ):

    def __init__ ( self, *handlers ):
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
        from matplotlib.enthought.traits.ui import TextEditor, CompoundEditor
        
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

#-------------------------------------------------------------------------------
#  'TraitComplex' class:  (deprecated, will be removed)
#-------------------------------------------------------------------------------

class TraitComplex ( TraitCompound ):

    def __init__ ( self, *handlers ):
        warning( 
           "'Use of 'TraitComplex' is deprecated, use 'TraitCompound' instead" )
        TraitCompound.__init__( self, *handlers )

#-------------------------------------------------------------------------------
#  'TraitTuple' class:
#-------------------------------------------------------------------------------

class TraitTuple ( TraitHandler ):
    
    def __init__ ( self, *args ):
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
        from matplotlib.enthought.traits.ui import TupleEditor
        return TupleEditor( traits = self.traits,
                            labels = trait.labels or [],
                            cols   = trait.cols   or 1  )
   
#-------------------------------------------------------------------------------
#  'TraitAny' class: (This class is deprecated)
#-------------------------------------------------------------------------------

class TraitAny ( TraitHandler ):

    def validate ( self, object, name, value ):
        return value

    def info ( self ):
        return 'any value'
   
#-------------------------------------------------------------------------------
#  'TraitCallable' class:
#-------------------------------------------------------------------------------

class TraitCallable ( TraitHandler ):

   def validate ( self, object, name, value ):
       if callable( value ):
           return value
       self.error( object, name, self.repr( value ) ) 

   def info ( self ):
       return 'a callable value'
       
#-------------------------------------------------------------------------------
#  'TraitListEvent' class:
#-------------------------------------------------------------------------------
       
class TraitListEvent:
    
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

    info_trait         = None
    default_value_type = 5
    _items_event       = None 
    
    def __init__ ( self, trait = None, minlen = 0, maxlen = sys.maxint,
                         has_items = True ):
        self.item_trait = trait_from( trait )
        self.minlen     = max( 0, minlen )
        self.maxlen     = max( minlen, maxlen )
        self.has_items  = has_items
 
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
                size = 'from %d to %d items' % ( 
                       self.minlen, self.maxlen )
        handler = self.item_trait.handler
        if handler is None:
            info = ''
        else:
            info = ' which are %s' % handler.info()
        return 'a list of %s%s' % ( size, info ) 
               
    def get_editor ( self, trait ):
        from matplotlib.enthought.traits.ui import ListEditor
        return ListEditor( trait_handler = self, 
                           rows          = trait.rows or 5,
                           use_notebook  = trait.use_notebook is True,
                           page_name     = trait.page_name or '' )
        
    def items_event ( self ):
        if TraitList._items_event is None:
            TraitList._items_event = Event( TraitListEvent, is_base = False )
        return TraitList._items_event
       
#-------------------------------------------------------------------------------
#  'TraitListObject' class:
#-------------------------------------------------------------------------------

class TraitListObject ( list ):
    
    def __init__ ( self, trait, object, name, value ):
        self.trait      = trait
        self.object     = object
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
        memo[ id_self ] = result = TraitListObject( self.trait, self.object, 
                        self.name, [ copy.deepcopy( x, memo ) for x in self ]  )          
        return result

    def __setitem__ ( self, key, value ):
        try:
            removed = [ self[ key ] ]
        except:
            pass
        try:
            handler = self.trait.item_trait.handler
            if handler is not None:
                value = handler.validate( self.object, self.name, value )
            list.__setitem__( self, key, value )
            if self.name_items is not None:
                if key < 0:
                    key = len( self ) + key
                setattr( self.object, self.name_items, 
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
                object  = self.object
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
                   setattr( self.object, self.name_items, 
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
                setattr( self.object, self.name_items, 
                         TraitListEvent( key, removed ) )
            return
        self.len_error( len( self ) - 1 )
    
    def __delslice__ ( self, i, j ):
        delta = min( j, len( self ) ) - max( 0, i )
        if self.trait.minlen <= (len( self ) - delta):
            removed = self[ i: j ]
            list.__delslice__( self, i, j )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 
                         TraitListEvent( max( 0, i ), removed ) )
            return
        self.len_error( len( self ) - delta )
        
    def append ( self, value ):
        if self.trait.minlen <= (len( self ) + 1) <= self.trait.maxlen:
            try:
                handler = self.trait.item_trait.handler
                if handler is not None:
                    value = handler.validate( self.object, self.name, value )
                list.append( self, value )
                if self.name_items is not None:
                    setattr( self.object, self.name_items, 
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
                if handler is not None:
                    value = handler.validate( self.object, self.name, value )
                list.insert( self, index, value )
                if self.name_items is not None:
                    if index < 0:
                        index = len( self ) + index - 1
                    setattr( self.object, self.name_items, 
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
        if (self.trait.minlen <= (len( self ) + len_xlist) <= 
            self.trait.maxlen):
            object  = self.object
            name    = self.name
            handler = self.trait.item_trait.handler
            try:
                if handler is not None:
                    validate = handler.validate
                    xlist    = [ validate( object, name, value )
                                 for value in xlist ]
                list.extend( self, xlist )
                if self.name_items is not None:
                    setattr( self.object, self.name_items, 
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
                setattr( self.object, self.name_items, 
                         TraitListEvent( index, removed ) )
        else:
            self.len_error( len( self ) - 1 )
        
    def len_error ( self, len ):
        raise TraitError, ( "The '%s' trait of %s instance must be %s, "
                  "but you attempted to change its length to %d element%s." % (
                  self.name, class_of( self.object ), self.trait.info(),
                  len, 's'[ len == 1: ] ) )
                  
    def sort ( self, cmpfunc = None ):
        removed = self[:]
        list.sort( self, cmpfunc )
        if self.name_items is not None:
            setattr( self.object, self.name_items, 
                     TraitListEvent( 0, removed, self[:] ) )
                  
    def reverse ( self ):
        removed = self[:]
        if len( self ) > 1:
            list.reverse( self )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 
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
                setattr( self.object, self.name_items, 
                         TraitListEvent( index, removed ) )
            return result
        else:
            self.len_error( len( self ) - 1 )
    
    def __getstate__ ( self ):
        result = self.__dict__.copy()
        del result[ 'trait' ]
        return result
        
    def __setstate__ ( self, state ):
        self.__dict__.update( state )
        self.trait = self.object._trait( self.name, 0 ).handler

#-------------------------------------------------------------------------------
#  'TraitDict' class:
#-------------------------------------------------------------------------------

class TraitDict ( TraitHandler ):
    
    info_trait         = None
    default_value_type = 6
    _items_event       = None
 
    def __init__ ( self, key_trait = None, value_trait = None, 
                         has_items = True ):
        self.key_trait   = trait_from( key_trait )
        self.value_trait = trait_from( value_trait )
        self.has_items   = has_items
 
    def validate ( self, object, name, value ):
        if isinstance( value, dict ):
            return TraitDictObject( self, object, name, value )
        self.error( object, name, self.repr( value ) ) 
 
    def info ( self ):
        extra   = ''
        handler = self.key_trait.handler
        if handler is not None:
            extra = ' with keys which are %s' % handler.info()
        handler = self.value_trait.handler
        if handler is not None:
            if extra == '':
                extra = ' with'
            else:
                extra += ' and'
            extra += ' values which are %s' % handler.info()
        return 'a dictionary%s' % extra
               
    def get_editor ( self, trait ):
        if self.editor is None:
            from matplotlib.enthought.traits.ui import TextEditor
            self.editor = TextEditor( evaluate = eval )
        return self.editor
        
    def items_event ( self ):
        if TraitDict._items_event is None:
            TraitDict._items_event = Event( 0, is_base = False )
        return TraitDict._items_event

#-------------------------------------------------------------------------------
#  'TraitDictObject' class:
#-------------------------------------------------------------------------------

class TraitDictObject ( dict ):
    
    def __init__ ( self, trait, object, name, value ):
        self.trait      = trait
        self.object     = object
        self.name       = name
        self.name_items = None
        if trait.has_items:
            self.name_items = name + '_items'
        if len( value ) > 0:
            dict.update( self, value )
        
    def __setitem__ ( self, key, value ):
        try:
            handler = self.trait.key_trait.handler
            if handler is not None:
                key = self.trait.key_trait.handler.validate( 
                                           self.object, self.name, key )
        except TraitError, excp:
            excp.set_prefix( 'Each key of the' )
            raise excp
        except AttributeError:
            # This is to handle the fact that on unpickling a serialized
            # dictionary, the items are simply stored in the dictionary before
            # '__setstate__' is called:
            dict.__setitem__( self, key, value ) 
            return
        try:                                          
            handler = self.trait.value_trait.handler
            if handler is not None:
                value = handler.validate( self.object, self.name, value )
            dict.__setitem__( self, key, value ) 
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
        except TraitError, excp:
            excp.set_prefix( 'Each value of the' )
            raise excp
    
    def __delitem__ ( self, key ):
        dict.__delitem__( self, key )
        if self.name_items is not None:
            setattr( self.object, self.name_items, 0 )
           
    def clear ( self ):
        if len( self ) > 0:
            dict.clear( self )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
        
    def update ( self, dic ):
        if len( dic ) > 0:
            dict.update( self, dic )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
      
    def setdefault ( self, key, value = None ):
        if self.has_key( key ):
            return self[ key ]
        self[ key ] = value
        return value
      
    def pop ( self, key, value = Undefined ):
        if (value is Undefined) or self.has_key( key ):
            result = dict.pop( key )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
            return result
        return value
           
    def popitem ( self ):
        result = dict.popitem( self )
        if self.name_items is not None:
            setattr( self.object, self.name_items, 0 )
        return result
    
    def __getstate__ ( self ):
        result = self.__dict__.copy()
        del result[ 'trait' ]
        return result
        
    def __setstate__ ( self, state ):
        self.__dict__.update( state )
        self.trait = self.object._trait( self.name, 0 ).handler
            
#-------------------------------------------------------------------------------
#  Tell the C-based traits module about 'TraitListObject' and 'TraitDictObject': 
#-------------------------------------------------------------------------------
        
import ctraits
ctraits._list_classes( TraitListObject, TraitDictObject )


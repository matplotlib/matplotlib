#-------------------------------------------------------------------------------
#
#  Define common, low-level capabilities needed by the 'traits' package.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  Refactored into a separate module: 07/04/2003
#
#  Symbols defined: SequenceTypes
#                   Undefined
#                   trait_editors
#                   class_of
#
#  (c) Copyright 2002, 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from __future__ import generators

import os
import sys

from os.path import exists, join 
from string  import lowercase, uppercase
from types   import ListType, TupleType, DictType, StringType, UnicodeType, \
                    IntType, LongType, FloatType, ComplexType, ClassType, \
                    TypeType
                    
#-------------------------------------------------------------------------------
#  Provide Python 2.3+ compatible definitions (if necessary):  
#-------------------------------------------------------------------------------
                                        
try: 
    from types import BooleanType
except ImportError: 
    BooleanType = IntType

def _enumerate ( seq ):
    for i in xrange( len( seq) ):
        yield i, seq[i]
try:
    enumerate = enumerate
except:
    enumerate = _enumerate
del _enumerate    
    
#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------
                  
ClassTypes    = ( ClassType, TypeType )                  

SequenceTypes = ( ListType, TupleType )

ComplexTypes  = ( float, int )

TypeTypes     = ( StringType,   UnicodeType,  IntType,    LongType,
                  FloatType,    ComplexType,  ListType,   TupleType,
                  DictType,     BooleanType )

TraitNotifier = '__trait_notifier__'            
       
#-------------------------------------------------------------------------------
#  Singleton 'Undefined' object (used as undefined trait name and/or value):
#-------------------------------------------------------------------------------

class _Undefined ( object ):

   def __repr__ ( self ):
       return '<undefined>'
       
Undefined = _Undefined()

# Tell the C-base code about the 'Undefined' objects:
import ctraits
ctraits._undefined( Undefined )
       
#-------------------------------------------------------------------------------
#  Singleton 'Missing' object (used as missing method argument marker):
#-------------------------------------------------------------------------------

class _Missing ( object ):

   def __repr__ ( self ):
       return '<missing>'
       
Missing = _Missing()

#-------------------------------------------------------------------------------
#  Singleton 'Self' object (used as object reference to current 'object'):
#-------------------------------------------------------------------------------

class _Self ( object ):

   def __repr__ ( self ):
       return '<self>'
       
Self = _Self()       

#-------------------------------------------------------------------------------
#  Define a special 'string' coercion function:
#-------------------------------------------------------------------------------

def strx ( arg ):
    if type( arg ) in StringTypes:
       return str( arg )
    raise TypeError
    
#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

StringTypes  = ( StringType, UnicodeType, IntType, LongType, FloatType,
                 ComplexType )

#-------------------------------------------------------------------------------
#  Define a mapping of coercable types:
#-------------------------------------------------------------------------------

CoercableTypes = { 
    LongType:    ( 11, long, int ),
    FloatType:   ( 11, float, int ),
    ComplexType: ( 11, complex, float, int ),
    UnicodeType: ( 11, unicode, str )
}
    
#-------------------------------------------------------------------------------
#  Return a string containing the class name of an object with the correct
#  article (a or an) preceding it (e.g. 'an Image', 'a PlotValue'):
#-------------------------------------------------------------------------------

def class_of ( object ):
    if type( object ) is StringType:
       return add_article( object )
    return add_article( object.__class__.__name__ )
    
#-------------------------------------------------------------------------------
#  Return a string containing the right article (i.e. 'a' or 'an') prefixed to 
#  a specified string:
#-------------------------------------------------------------------------------

def add_article ( name ):
    if name[:1].lower() in 'aeiou':
       return 'an ' + name
    return 'a ' + name
    
#----------------------------------------------------------------------------
#  Return a 'user-friendly' name for a specified trait:
#----------------------------------------------------------------------------

def user_name_for ( name ):
    name       = name.replace( '_', ' ' ).capitalize()
    result     = ''
    last_lower = 0
    for c in name:
        if (c in uppercase) and last_lower:
           result += ' '
        last_lower = (c in lowercase)
        result    += c
    return result
        
#-------------------------------------------------------------------------------
#  Gets the path to the traits home directory:
#-------------------------------------------------------------------------------
 
_traits_home = None

def traits_home ( ):
    """ Gets the path to the traits home directory.
    """
    global _traits_home 
    
    if _traits_home is None:
        home = _verify_path( os.environ.get( 'HOME' ) or '\\home' )
        _traits_home = _verify_path( join( home, '.traits' ) )
    return _traits_home
        
#-------------------------------------------------------------------------------
#  Verify that a specified path exists, and try to create it if it doesn't:    
#-------------------------------------------------------------------------------
            
def _verify_path ( path ):        
    """ Verify that a specified path exists, and try to create it if it 
        doesn't.
    """
    if not exists( path ):
        try:
            os.mkdir( path )
        except:
            pass
    return path


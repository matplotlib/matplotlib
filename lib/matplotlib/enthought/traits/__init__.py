#-------------------------------------------------------------------------------
#
#  Define a 'traits' package that allows other classes to easily define
#  'type-checked' and/or 'delegated' traits for their instances.
#
#  Note: A 'trait' is similar to a 'property', but is used instead of the 
#  word 'property' to differentiate it from the Python language 'property'
#  feature.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

from info_traits  import __doc__
from trait_base   import Undefined, Missing, Self
from trait_errors import TraitError, DelegationError 
from category     import Category   
from trait_db     import tdb
from traits       import Event, TraitEvent, List, Dict, Tuple, Range, \
     Constant, CTrait, Trait, TraitPython, Delegate, Property, TraitProperty
from traits import Any, AnyValue, Int, Long, Float, Str, Unicode, Complex, \
     Bool, CInt, CLong, CFloat, CStr, CUnicode, CComplex, CBool, false, true, \
     Regex, String, Password, File, Directory, Function, Method, Class 
from traits import Instance, Module, Type, This, self, Python, Disallow, \
     ReadOnly, undefined, missing, ListInt, ListFloat, ListStr, ListUnicode, \
     ListComplex, ListBool, ListFunction, ListMethod, ListClass, ListInstance
from traits import ListThis, DictStrAny, DictStrStr, DictStrInt, DictStrLong, \
     DictStrFloat, DictStrBool, DictStrList, TraitFactory, Callable, Array, \
     CArray, Enum, Code, Default
from traits         import Color, RGBColor, RGBAColor, Font, KivaFont     
from has_traits     import method, HasTraits, HasStrictTraits, HasPrivateTraits
from trait_handlers import TraitHandler, TraitRange, TraitString, TraitType, \
     TraitCastType, TraitInstance, ThisClass, TraitClass, TraitFunction
from trait_handlers import TraitEnum, TraitPrefixList, TraitMap, \
     TraitPrefixMap, TraitCompound, TraitList, TraitDict

#-------------------------------------------------------------------------------
#  Deprecated values:
#-------------------------------------------------------------------------------

from traits         import TraitEvent, TraitDelegate, TraitDelegateSynched, \
                           DefaultPythonTrait
from has_traits     import HasDynamicTraits
from trait_handlers import TraitComplex


                            

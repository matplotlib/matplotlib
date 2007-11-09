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
# Date: 12/13/2004
#------------------------------------------------------------------------------
""" Trait definitions related to the numpy library.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from trait_base \
    import SequenceTypes, enumerate
    
from trait_handlers \
    import TraitHandler
    
from traits \
    import Str, Any
    
from traits \
    import Int as TInt
    
from traits \
    import Float as TFloat

import warnings


#-------------------------------------------------------------------------------
#  Deferred imports from numpy:
#-------------------------------------------------------------------------------

ndarray = None
asarray = None


#-------------------------------------------------------------------------------
#  numpy dtype mapping:
#-------------------------------------------------------------------------------

def dtype2trait( dtype ):
    """ Get the corresponding trait for a numpy dtype.
    """

    import numpy

    if dtype.char in numpy.typecodes['Float']:
        return TFloat
    elif dtype.char in numpy.typecodes['AllInteger']:
        return TInt
    elif dtype.char[0] == 'S':
        return Str
    else:
        return Any


#-------------------------------------------------------------------------------
#  'TraitArray' class:
#-------------------------------------------------------------------------------

class TraitArray ( TraitHandler ):
    """ Handles assignment to traits based on Numpy arrays.
    """
    default_value_type = 7

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, dtype = None, shape = None, coerce = False,
                   typecode = None):
        global ndarray, asarray
        
        # Initialize module-level globals
        try:
            import numpy
        except ImportError:
            raise TraitError( "Using Array or CArray trait types requires the "
                              "numpy package to be installed." )
        
        from numpy import array, asarray, ndarray, zeros

        if typecode is not None:
            warnings.warn("typecode is a deprecated argument; use dtype instead",
                          DeprecationWarning)
            if dtype is not None and dtype != typecode:
                raise TraitError("Inconsistent usage of the dtype and typecode "
                                 "arguments; use dtype alone.")
            else:
                dtype = typecode
        
        if dtype is not None:
            try:
                dtype = numpy.dtype(dtype)
            except TypeError, e:
                raise TraitError("could not convert %r to a numpy dtype" % dtype)

        self.dtype  = dtype
        self.shape  = shape
        self.coerce = coerce
 
    #---------------------------------------------------------------------------
    #  Validates that a value is legal for the trait:
    #---------------------------------------------------------------------------
        
    def validate ( self, object, name, value ):
        """ Validates that a value is legal for the trait.
        """
        #try:
        if 1:
            # Make sure the value is an array:
            type_value = type( value )
            if not isinstance( value, ndarray ): 
                if not isinstance(value, (list,tuple)):
                    self.error( object, name, self.repr( value ) ) 
                if self.dtype is not None:
                    value = asarray(value, self.dtype)
                else:
                    value = asarray(value)
                
            # Make sure the array is of the right type:
            if ((self.dtype is not None) and 
                (value.dtype != self.dtype)):
                if self.coerce:
                    value = value.astype( self.dtype )
                else:
                    value = asarray( value, self.dtype )
                    
            # If no shape requirements, then return the value:
            trait_shape = self.shape
            if trait_shape is None:
                return value
                
            # Else make sure that the value's shape is compatible:
            value_shape = value.shape
            if len( trait_shape ) == len( value_shape ):
                for i, dim in enumerate( value_shape ):
                    item = trait_shape[i]
                    if item is not None:
                        if type( item ) is int:
                            if dim != item:
                                break
                        elif ((dim < item[0]) or 
                              ((item[1] is not None) and (dim > item[1]))):
                            break
                else:
                    return value

        #    print "*** pass through"
        #except Exception, e:
        #    print "*** exception:", e
        self.error( object, name, self.repr( value ) ) 

    #---------------------------------------------------------------------------
    #  Returns the default value constructor for the type (called from the
    #  trait factory):
    #---------------------------------------------------------------------------
        
    def default_value ( self, value ):
        """ Returns the default value constructor for the type (called from the
            trait factory).
        """
        return ( self.copy_default_value, 
                 ( self.validate( None, None, value ), ), None )

    #---------------------------------------------------------------------------
    #  Returns a copy of the default value (called from the C code on first
    #  reference to a trait with no current value):
    #---------------------------------------------------------------------------
                  
    def copy_default_value ( self, value ):
        """ Returns a copy of the default value (called from the C code on 
            first reference to a trait with no current value).
        """
        return value.copy()        

    #---------------------------------------------------------------------------
    #  Returns descriptive information about the trait:
    #---------------------------------------------------------------------------
        
    def info ( self ):
        """ Returns descriptive information about the trait.
        """
        dtype = shape = ''
        
        if self.shape is not None:
            shape = []
            for item in self.shape:
                if item is None:
                    item = '*'
                elif type( item ) is not int:
                    if item[1] is None:
                        item = '%d..' % item[0]
                    else:
                        item = '%d..%d' % item
                shape.append( item )
            shape = ' with shape %s' % ( tuple( shape ), )
             
        if self.dtype is not None:
            # FIXME: restore nicer descriptions of dtypes.
            dtype = ' of %s values' % self.dtype
            
        return 'an array%s%s' % ( dtype, shape )

    #---------------------------------------------------------------------------
    #  Gets the trait editor associated with the trait:
    #---------------------------------------------------------------------------

    def get_editor ( self, trait ):
        """ Gets the trait editor associated with the trait.
        """
        from enthought.traits.ui.api import TupleEditor
        
        if self.dtype is None:
            traits = TFloat
        else:
            traits = dtype2trait(self.dtype)
            
        return TupleEditor( traits = traits,
                            labels = trait.labels or [],
                            cols   = trait.cols   or 1  )
        

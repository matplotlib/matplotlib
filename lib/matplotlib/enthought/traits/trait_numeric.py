#-------------------------------------------------------------------------------
#  
#  Numeric related trait definitions.
#  
#  Written by: David C. MOrrill
#  
#  Date: 12/13/2004
#  
#  (c) Copyright 2004 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from trait_base     import SequenceTypes, enumerate
from trait_handlers import TraitHandler
from traits         import Str, Any
from traits         import Int as TInt
from traits         import Float as TFloat

from Numeric import array, asarray, zeros, Character, UnsignedInt8, Int, Int0, \
                    Int8, Int16, Int32, Float, Float0, Float8, Float16, \
                    Float32, Float64, Complex, Complex0, Complex8, Complex16, \
                    Complex32, Complex64, PyObject
                    
try:
    from Numeric import Int64
except:
    Int64 = None
                    
try:
    from Numeric import Int128
except:
    Int128 = None
                    
try:
    from Numeric import Float128
except:
    Float128 = None
                    
try:
    from Numeric import Complex128
except:
    Complex128 = None

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

ArrayType = type( array( [ 1 ] ) )

Typecodes = { 
    Character:    ( 'character', Str ),
    UnsignedInt8: ( 'unsigned 8 bit int', TInt ), 
    Int:          ( 'int', TInt ),
    Int0:         ( 'int', TInt ),
    Int8:         ( '8 bit int', TInt ),
    Int16:        ( '16 bit int', TInt ),
    Int32:        ( '32 bit int', TInt ),
    Int64:        ( '64 bit int', TInt ),
    Float:        ( 'float', TFloat ),
    Float0:       ( 'float', TFloat ),
    Float8:       ( '8 bit float', TFloat ),
    Float16:      ( '16 bit float', TFloat ),
    Float32:      ( '32 bit float', TFloat ),
    Float64:      ( '64 bit float', TFloat ),
    Float128:     ( '128 bit float', TFloat ),
    Complex:      ( 'complex', Any ),
    Complex0:     ( 'complex', Any ),
    Complex8:     ( '8 bit complex', Any ),
    Complex16:    ( '16 bit complex', Any ),
    Complex32:    ( '32 bit complex', Any ), 
    Complex64:    ( '64 bit complex', Any ), 
    Complex128:   ( '128 bit complex', Any ),
    PyObject:     ( 'object', Any )
}
 
#-------------------------------------------------------------------------------
#  'TraitArray' class:
#-------------------------------------------------------------------------------

class TraitArray ( TraitHandler ):
    
    default_value_type = 7

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, typecode = None, shape = None, coerce = False ):
        self.typecode = typecode
        self.shape    = shape
        self.coerce   = coerce
 
    #---------------------------------------------------------------------------
    #  Validates that a value is legal for the trait:
    #---------------------------------------------------------------------------
        
    def validate ( self, object, name, value ):
        """ Validates that a value is legal for the trait.
        """
        try:
            # Make sure the value is an array:
            type_value = type( value )
            if type_value is not ArrayType:
                if type_value not in SequenceTypes:
                    self.error( object, name, self.repr( value ) ) 
                value = asarray( value, self.typecode )
                
            # Make sure the array is of the right type:
            if ((self.typecode is not None) and 
                (value.typecode() != self.typecode)):
                if self.coerce:
                    value = value.astype( self.typecode )
                else:
                    value = asarray( value, self.typecode )
                    
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
        except:
            pass
        self.error( object, name, self.repr( value ) ) 

    #---------------------------------------------------------------------------
    #  Returns the default value constructor for the type (called from the
    #  trait factory):
    #---------------------------------------------------------------------------
        
    def default_value ( self, value ):
        """ Returns the default value constructor for the type (called from the
            trait factory.
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
        typecode = shape = ''
        
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
             
        if self.typecode is not None:
            typecode = ' of %s values' % Typecodes[ self.typecode ][0]
            
        return 'an array%s%s' % ( typecode, shape )

    #---------------------------------------------------------------------------
    #  Gets the trait editor associated with the trait:
    #---------------------------------------------------------------------------

    def get_editor ( self, trait ):
        """ Gets the trait editor associated with the trait.
        """
        from matplotlib.enthought.traits.ui import TupleEditor
        
        if self.typecode is None:
            traits = TFloat
        else:
            traits = Typecodes[ self.typecode ][1]
            
        return TupleEditor( traits = traits,
                            labels = trait.labels or [],
                            cols   = trait.cols   or 1  )
        

#-------------------------------------------------------------------------------
#
#  Definition of common traits used within the traits.ui package.
#
#  Written by: David C. Morrill
#
#  Date: 10/14/2004
#
#  Symbols defined: 
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits import Trait, TraitPrefixList, Delegate, Str

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# User interface element styles:
style_trait = Trait( 'simple',
                     TraitPrefixList( 'simple', 'custom', 'text', 'readonly' ),
                     cols = 4 )
                     
# The default object being edited trait:                     
object_trait = Str( 'object' )                     

# Delegate a trait value to the object's 'container':                      
container_delegate = Delegate( 'container' )

#-------------------------------------------------------------------------------
#  Other definitions:
#-------------------------------------------------------------------------------

SequenceTypes = ( tuple, list )

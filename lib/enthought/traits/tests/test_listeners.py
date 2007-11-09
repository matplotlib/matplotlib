#-------------------------------------------------------------------------------
#  
#  Test the 'add_trait_listener', 'remove_trait_listener' interface to  
#  the HasTraits class.
#  
#  Written by: David C. Morrill 
#  
#  Date: 09/07/2005
#  
#  (c) Copyright 2005 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:  
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasTraits, Str, Int, Float
    
#-------------------------------------------------------------------------------
#  'GenerateEvents' class:  
#-------------------------------------------------------------------------------
        
class GenerateEvents ( HasTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    name   = Str
    age    = Int
    weight = Float
    
#-------------------------------------------------------------------------------
#  'ListenEvents' class:  
#-------------------------------------------------------------------------------

class ListenEvents ( HasTraits ):
    
    #---------------------------------------------------------------------------
    #  'GenerateEvents' event interface:  
    #---------------------------------------------------------------------------
        
    def _name_changed ( self, object, name, old, new ):
        print "_name_changed:", object, name, old, new
        
    def _age_changed ( self, object, name, old, new ):
        print "_age_changed:", object, name, old, new
        
    def _weight_changed ( self, object, name, old, new ):
        print "_weight_changed:", object, name, old, new
        
    def alt_name_changed ( self, object, name, old, new ):
        print "alt_name_changed:", object, name, old, new
        
    def alt_weight_changed ( self, object, name, old, new ):
        print "alt_weight_changed:", object, name, old, new
        
#-------------------------------------------------------------------------------
#  Run the tests:  
#-------------------------------------------------------------------------------
                
ge = GenerateEvents()
le = ListenEvents()
print 'Starting test: No Listeners'
ge.set( name = 'Joe', age = 22, weight = 152.0 )
print 'Adding default listener'
ge.add_trait_listener( le )
ge.set( name = 'Mike', age = 34, weight = 178.0 )
print 'Adding alternate listener'
ge.add_trait_listener( le, 'alt' )
ge.set( name = 'Gertrude', age = 39, weight = 108.0 )
print 'Removing default listener'
ge.remove_trait_listener( le )
ge.set( name = 'Sally', age = 46, weight = 118.0 )
print 'Removing alternate listener'
ge.remove_trait_listener( le, 'alt' )
ge.set( name = 'Ralph', age = 29, weight = 198.0 )
print 'Test Completed'


#-------------------------------------------------------------------------------
#
#  Adds a Cocoa-like 'category' capability to Traits-base classes.
#
#  Written by: David C. Morrill
#
#  Date: 11/06/2004
#
#  (c) Copyright 2002, 2004 by Enthought, Inc.
#
#------------------------------------------------------------------------------- 

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from has_traits import MetaHasTraits, MetaHasTraitsObject, BaseTraits, \
                       ClassTraits, PrefixTraits, ViewTraits  

#-------------------------------------------------------------------------------
#  'MetaCategory' class:
#-------------------------------------------------------------------------------

class MetaCategory ( MetaHasTraits ):
    
    def __new__ ( cls, class_name, bases, class_dict ):

        # Make sure the correct usage is being applied:        
        if len( bases ) > 2:
            raise TypeError, \
                  "Correct usage is: class FooCategory(Category,Foo):"
                  
        # Process any traits-related information in the class dictionary:
        MetaCategoryObject( cls, class_name, bases, class_dict, True )
                  
        # Move all remaining items in our class dictionary to the base class's 
        # dictionary:
        if len( bases ) == 2:
            category_class = bases[1]
            for name, value in class_dict.items():
                if name != '__module__':
                    setattr( category_class, name, value )
                    del class_dict[ name ]
            
        # Finish building the class using the updated class dictionary:
        return type.__new__( cls, class_name, bases, class_dict )
        
#-------------------------------------------------------------------------------
#  'MetaCategoryObject' class:  
#-------------------------------------------------------------------------------
                
class MetaCategoryObject ( MetaHasTraitsObject ):
    
    #---------------------------------------------------------------------------
    #  Adds the traits meta-data to the class:   
    #---------------------------------------------------------------------------
           
    def add_traits_meta_data ( self, bases, class_dict, base_traits, 
                               class_traits, prefix_traits, view_elements ):
        if len( bases ) == 2:
            category_class = bases[1]
            
            # Update the base class's traits with the new ones:
            if len( base_traits ) > 0:
                getattr( category_class, BaseTraits ).update( base_traits )
            if len( class_traits ) > 0:
                getattr( category_class, ClassTraits ).update( class_traits )
            if len( prefix_traits ) > 0:
                getattr( category_class, PrefixTraits ).update( prefix_traits )
            
            # Copy all our view elements into the base class's ViewElements:
            if view_elements is not None:
                content = view_elements.content
                if len( content ) > 0:
                    base_ve = getattr( category_class, ViewTraits, None )
                    if base_ve is None:
                        setattr( category_class, ViewTraits, view_elements )
                    else:
                        base_ve.content.update( content )
                        
            # Update each of the existing subclasses as well:
            for subclass in category_class.trait_subclasses( True ):
                
                subclass_traits = getattr( subclass, BaseTraits )
                for name, value in base_traits.items():
                    subclass_traits.setdefault( name, value )
                    
                subclass_traits = getattr( subclass, ClassTraits )
                for name, value in class_traits.items():
                    subclass_traits.setdefault( name, value )
                    
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
        else:
            MetaHasTraitsObject.add_traits_meta_data( self, bases,
                   class_dict, base_traits, class_traits,  prefix_traits, 
                   view_elements )  
        
#-------------------------------------------------------------------------------
#  'Category' class:
#-------------------------------------------------------------------------------

class Category ( object ):
    
    __metaclass__ = MetaCategory 


#-------------------------------------------------------------------------------
#
#  Define the Group class used to represent a group of items used in a
#  traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: Group
#                   ShadowGroup
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from string           import find

from matplotlib.enthought.traits import Trait, TraitPrefixList, TraitError, ReadOnly, \
                             Delegate, Undefined, List, Str, Range, true, false
from matplotlib.enthought.traits.trait_base import enumerate                             
                             
from view_element     import ViewSubElement
from item             import Item
from include          import Include
from ui_traits        import SequenceTypes, container_delegate

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Group orientation trait:
Orientation = Trait( 'vertical', 
                     TraitPrefixList( 'vertical', 'horizontal' ) )
                           
# Delegate trait to the object being 'shadowed':
ShadowDelegate = Delegate( 'shadow' )

# Amount of padding to add around item:
Padding = Range( 0, 15, desc = 'amount of padding to add around each item' )

#-------------------------------------------------------------------------------
#  'Group' class:
#-------------------------------------------------------------------------------

class Group ( ViewSubElement ):
    
    #---------------------------------------------------------------------------
    # Trait definitions:
    #---------------------------------------------------------------------------
    
    content      = List( ViewSubElement ) # ViewSubElement objects in group
    id           = Str               # Name of the group
    label        = Str               # User interface label for the group
    object       = container_delegate# Default context object for group items 
    style        = container_delegate# Default style of items in the group 
    orientation  = Orientation       # Spatial orientation of the group
    show_border  = false             # Should a border be drawn around group?
    show_labels  = true              # Should labels be added to items in group?
    show_left    = true              # Should labels be shown on left(or right)?
    selected     = false             # Is group the initially selected page?
    splitter     = false             # Are items separated by splitter bars?
    help         = Str               # Optional help text (for top-level group)
    defined_when = Str               # Pre-condition for defining the group
    enabled_when = Str               # Pre-condition for enabling the group
    padding      = Padding           # Amount of padding to add around each item
     
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, *values, **traits ):
        """ Initializes the object.
        """
        ViewSubElement.__init__( self, **traits )
        
        # Initialize the contents of the Group (if not done by parent):
        content = self.content
        if content is Undefined:
            self.content = content = []
            
        # Process all of the data passed to the constructor:
        for value in values:
            if isinstance( value, ViewSubElement ): 
                # Standard element type:
                content.append( value )
            elif type( value ) in SequenceTypes:
                # Map (...) or [...] to a Group():
                content.append( Group( *value ) )
            elif type( value ) is str:
                if value[0:1] in '-|':
                    # Parse Group trait options if specified as a string:
                    self._parse( value )
                elif (value[:1] == '<') and (value[-1:] == '>'):
                    # Convert string to an Include value:
                    content.append( Include( value[1:-1].strip() ) )
                else:
                    # Else let the Item class try to make sense of it:
                    content.append( Item( value ) )
            else:
                raise TypeError, "Unrecognized argument type: %s" % value
                
        # Make sure this Group is the container for all its children:
        for item in content:
            item.container = self
            
    #---------------------------------------------------------------------------
    #  Returns whether or not the object is replacable by an Include object:
    #---------------------------------------------------------------------------
            
    def is_includable ( self ):
        """ Returns whether or not the object is replacable by an Include 
            object.
        """
        return (self.id != '')
    
    #---------------------------------------------------------------------------
    #  Replaces any items which have an 'id' with an Include object with the 
    #  same 'id', and puts the object with the 'id' into the specified 
    #  ViewElements object: 
    #---------------------------------------------------------------------------
    
    def replace_include ( self, view_elements ):
        """ Replaces any items which have an 'id' with an Include object with 
            the same 'id', and puts the object with the 'id' into the specified 
            ViewElements object.
        """
        for i, item in enumerate( self.content ):
            if item.is_includable():
                id = item.id
                if id in view_elements.content:
                    raise TraitError, \
                          "Duplicate definition for view element '%s'" % id
                self.content[ i ] = Include( id )
                view_elements.content[ id ] = item
            item.replace_include( view_elements )
    
    #---------------------------------------------------------------------------
    #  Returns a ShadowGroup for the Group which recursively resolves all
    #  imbedded Include objects and which replaces all imbedded Group objects
    #  with a corresponding ShadowGroup:
    #---------------------------------------------------------------------------
                
    def get_shadow ( self, ui ):
        """ Returns a ShadowGroup for the Group which recursively resolves all
            imbedded Include objects and which replaces each imbedded Group 
            object with a corresponding ShadowGroup.
        """
        content = []
        groups  = 0
        level   = ui.push_level()
        for value in self.content:
            # Recursively replace Include objects:
            while isinstance( value, Include ):
                value = ui.find( value )
                
            # Convert Group objects to ShadowGroup objects, but include Item
            # objects as is (ignore any 'None' values caused by a failed 
            # Include): 
            if isinstance( value, Group ):
                if self._defined_when( ui, value ):
                    content.append( value.get_shadow( ui ) )
                    groups += 1
            elif isinstance( value, Item ):
                if self._defined_when( ui, value ):
                    content.append( value )
                    
            ui.pop_level( level )
                    
        # Return the ShadowGroup:
        return ShadowGroup( shadow = self, content = content, groups = groups )
        
    #---------------------------------------------------------------------------
    #  Returns whether the object should be defined in the user interface:
    #---------------------------------------------------------------------------
        
    def _defined_when ( self, ui, value ):
        """ Returns whether the object should be defined in the user interface.
        """
        if value.defined_when == '':
            return True
        return ui.eval_when( value.defined_when )
            
    #---------------------------------------------------------------------------
    #  Parses Group options specified as a string:
    #---------------------------------------------------------------------------
                
    def _parse ( self, value ):
        """ Parses Group options specified as a string.
        """
        # Override the defaults, since we only allow 'True' values to be
        # specified:
        self.show_border = self.show_labels = self.show_left = False
        
        # Parse all of the single or multi-character options:
        value = self._parse_label( value )
        value = self._parse_style( value )
        value = self._option( value, '-', 'orientation', 'horizontal' )
        value = self._option( value, '|', 'orientation', 'vertical' )
        value = self._option( value, '=', 'splitter',     True )
        value = self._option( value, '>', 'show_labels',  True )
        value = self._option( value, '<', 'show_left',    True )
        value = self._option( value, '!', 'selected',     True )
        
        show_labels      = not (self.show_labels and self.show_left)
        self.show_left   = not self.show_labels
        self.show_labels = show_labels
        
        # Parse all of the punctuation based sub-string options:
        value = self._split( 'id', value, ':', find,  0, 1 )
        if value != '':
            self.object = value
            
    #---------------------------------------------------------------------------
    #  Handles a label being found in the string definition:
    #---------------------------------------------------------------------------
            
    def _parsed_label ( self ):
        """ Handles a label being found in the string definition.
        """
        self.show_border = True
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Group:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a 'pretty print' version of the Group.
        """
        return "( %s, %s )" % ( 
                   ', '.join( [ item.__repr__() for item in self.content ] ),
                   self._repr_group() )
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Group traits:
    #---------------------------------------------------------------------------
                   
    def _repr_group ( self ):
        """ Returns a 'pretty print' version of the Group traits.
        """
        return '"%s%s%s%s%s%s%s%s%s%s%s"' % ( 
                   self._repr_option( self.orientation, 'horizontal', '-' ),
                   self._repr_option( self.orientation, 'vertical',   '|' ),
                   self._repr_option( self.show_border,     True,     '[]' ),
                   self._repr_option( self.show_labels and 
                                      self.show_left,       True,     '<' ),
                   self._repr_option( self.show_labels and 
                                      (not self.show_left), True,     '>' ),
                   self._repr_option( self.show_labels,     False,    '<>' ),
                   self._repr_option( self.selected,        True,     '!' ),
                   self._repr_value( self.id, '', ':' ), 
                   self._repr_value( self.object, '', '', 'object' ), 
                   self._repr_value( self.label,'=' ),
                   self._repr_value( self.style, ';', '', 'simple' ) )

#-------------------------------------------------------------------------------
#  'ShadowGroup' class:
#-------------------------------------------------------------------------------

class ShadowGroup ( Group ):
    
    #---------------------------------------------------------------------------
    # Trait definitions:
    #---------------------------------------------------------------------------
 
    shadow       = ReadOnly        # Group object this is a 'shadow' for
    groups       = ReadOnly        # Number of ShadowGroups in 'content'
    id           = ShadowDelegate  # Name of the group
    label        = ShadowDelegate  # User interface label for the group
    object       = ShadowDelegate  # Default context object for group items 
    style        = ShadowDelegate  # Default style of items in the group 
    orientation  = ShadowDelegate  # Spatial orientation of the group
    show_border  = ShadowDelegate  # Should a border be drawn around group?
    show_labels  = ShadowDelegate  # Should labels be added to items in group?
    show_left    = ShadowDelegate  # Should labels be shown on left(or right)?
    selected     = ShadowDelegate  # Is group the initially selected page?
    splitter     = ShadowDelegate  # Are items separated by splitter bars?
    help         = ShadowDelegate  # Optional help text (for top-level group)
    defined_when = ShadowDelegate  # Pre-condition for defining the group
    enabled_when = ShadowDelegate  # Pre-condition for enabling the group
    padding      = ShadowDelegate  # Amount of padding to add around each item
            
    #---------------------------------------------------------------------------
    #  Returns the contents of the ShadowGroup within a specified user interface
    #  building context. This makes sure that all Group types are of the same
    #  type (i.e. Group or Item) and that all Include objects have been replaced
    #  by their substituted values:
    #---------------------------------------------------------------------------
    
    def get_content ( self, allow_groups = True ):
        """ Returns the contents of the Group within a specified user interface
            building context. This makes sure that all Group types are of the 
            same type (i.e. Group or Item) and that all Include objects have 
            been replaced by their substituted values.
        """
        # Make a copy of the content:
        result = self.content[:]
                
        # If result includes any ShadowGroups and they are not allowed, 
        # replace them:
        if ((self.groups != 0) and 
            ((self.groups != len( result )) or (not allow_groups))):
            i = 0
            while i < len( result ):
                value = result[i]
                if isinstance( value, ShadowGroup ):
                    items         = value.get_content( False )
                    result[i:i+1] = items
                    i += len( items )
                else:
                    i += 1
                    
        # Return the resulting list of objects:
        return result
                   

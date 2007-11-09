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
# Date: 10/07/2004
#  Symbols defined: Group
#                   ShadowGroup
#------------------------------------------------------------------------------
""" Defines the Group class used to represent a group of items used in a 
Traits-based user interface.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from string \
    import find

from enthought.traits.api \
    import Trait, TraitPrefixList, TraitError, ReadOnly, Delegate, Undefined, \
           List, Str, Range, true, false

from enthought.traits.trait_base \
    import enumerate

from view_element \
    import ViewSubElement

from item \
    import Item

from include \
    import Include

from ui_traits \
    import SequenceTypes, container_delegate

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Group orientation trait
Orientation = Trait( 'vertical',
                     TraitPrefixList( 'vertical', 'horizontal' ) )

# Group layout trait
Layout = Trait( 'normal',
                TraitPrefixList( 'normal', 'split', 'tabbed', 'flow' ) )

# Delegate trait to the object being "shadowed"
ShadowDelegate = Delegate( 'shadow' )

# Amount of padding to add around item
Padding = Range( 0, 15, desc = 'amount of padding to add around each item' )

#-------------------------------------------------------------------------------
#  'Group' class:
#-------------------------------------------------------------------------------

class Group ( ViewSubElement ):
    """Represents a grouping of items in a user interface view.
    """
    #---------------------------------------------------------------------------
    # Trait definitions:
    #---------------------------------------------------------------------------

    # A list of Group, Item, and Include objects in this group.
    content = List( ViewSubElement )

    # A unique identifier for the group. 
    id = Str

    # User interface label for the group. How the label is displayed depends 
    # on the **show_border** attribute, and on the **layout** attribute of
    # the group's parent group or view.
    label = Str

    # Default context object for group items.
    object = container_delegate

    # Default editor style of items in the group.
    style = container_delegate

    # Default docking style of items in group.
    dock = container_delegate

    # Default image to display on notebook tabs.
    image = container_delegate

    # Category of elements dragged from view.
    export = container_delegate

    # Spatial orientation of the group's elements. Can be 'vertical' (default)
    # or 'horizontal'.
    orientation = Orientation

    # Layout style of the group, which can be one of the following:
    #
    # * 'normal' (default): Sub-groups are displayed sequentially in a single 
    #   panel.
    # * 'flow': Sub-groups are displayed sequentially, and then "wrap" when
    #   they exceed the available space in the **orientation** direction.
    # * 'split': Sub-groups are displayed in a single panel, separated by
    #   "splitter bars", which the user can drag to adjust the amount of space
    #   for each sub-group.
    # * 'tabbed': Each sub-group appears on a separate tab, labeled with the
    #   sub-group's *label* text, if any.
    #
    # This attribute is ignored for groups that contain only items, or contain
    # only one sub-group.
    layout = Layout

    # The number of columns in the group
    columns = Range( 1, 10 )

    # Should a border be drawn around group? If set to True, the **label** text
    # is embedded in the border. If set to False, the label appears as a banner
    # above the elements of the group.
    show_border = false

    # Should labels be added to items in group? Only items that are directly
    # contained in the group are affected. That is, if the group contains
    # a sub-group, the display of labels in the sub-group is not affected by
    # the attribute on this group.
    show_labels = true

    # Should labels be shown to the left of items (True) or the right (False)?
    # Only items that are directly contained in the group are affected. That is,
    # if the group contains a sub-group, the display of labels in the sub-group
    # is not affected by the attribute in this group. If **show_labels** is
    # False, this attribute is irrelevant.
    show_left = true

    # Is this group the tab that is initially selected? If True, the group's
    # tab is displayed when the view is opened. If the **layout** of the group's
    # parent is not 'tabbed', this attribute is ignored.
    selected = false

    # Should the group use extra space along its parent group's layout
    # orientation?
    springy = false

    # Optional help text (for top-level group). This help text appears in the
    # View-level help window (created by the default help handler), for any
    # View that contains *only* this group. Group-level help is ignored for
    # nested groups and multiple top-level groups
    help = Str

    # Pre-condition for including the groupin the display. If the expression
    # evaluates to False, the group is not defined in the display. Conditions
    # for **defined_when** are evaluated only once, when the display is first
    # constructed. Use this attribute for conditions based on attributes that
    # vary from object to object, but that do not change over time. 
    defined_when = Str

    # Pre-condition for showing the group. If the expression evaluates to False,
    # the group and its items are not visible (and they disappear if they were
    # previously visible). If the value evaluates to True, the group and items
    # become visible. All **visible_when** conditions are checked each time 
    # that any trait value is edited in the display. Therefore, you can use 
    # **visible_when** conditions to hide or show groups in response to user
    # input.
    visible_when = Str

    # Pre-condition for enabling the group. If the expression evaluates to False,
    # the group is disabled, that is, none of the widgets accept input. All
    # **enabled_when** conditions are checked each time that any trait value
    # is edited in the display. Therefore, you can use **enabled_when** 
    # conditions to enable or disable groups in response to user input.
    enabled_when = Str

    # Amount of padding (in pixels) to add around each item in the group. The
    # value must be an integer between 0 and 15. (Unlike the Item class, the 
    # Group class does not support negative padding.) The padding for any 
    # individual widget is the sum of the padding for its Group, the padding 
    # for its Item, and the default spacing determined by the toolkit. 
    padding = Padding

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, *values, **traits ):
        """ Initializes the group object.
        """
        super( ViewSubElement, self ).__init__( **traits )

        content = self.content

        # Process any embedded Group options first:
        for value in values:
            if (isinstance(value, basestring)) and (value[0:1] in '-|'):
                # Parse Group trait options if specified as a string:
                self._parse( value )

        # Process all of the data passed to the constructor:
        for value in values:
            if isinstance( value, ViewSubElement ):
                content.append( value )
            elif type( value ) in SequenceTypes:
                # Map (...) or [...] to a Group():
                content.append( Group( *value ) )
            elif isinstance(value, basestring):
                if value[0:1] in '-|':
                    # We've already parsed Group trait options above:
                    pass
                elif (value[:1] == '<') and (value[-1:] == '>'):
                    # Convert string to an Include value:
                    content.append( Include( value[1:-1].strip() ) )
                else:
                    # Else let the Item class try to make sense of it:
                    content.append( Item( value ) )
            else:
                raise TypeError, "Unrecognized argument type: %s" % value

        # Make sure this Group is the container for all its children:
        self.set_container()

    #---------------------------------------------------------------------------
    #  Gets the label to use for a specified Group in a specified UI:
    #---------------------------------------------------------------------------

    def get_label ( self, ui ):
        """ Gets the label to use this group.
        """
        if self.label != '':
            return self.label

        return 'Group'

    #---------------------------------------------------------------------------
    #  Returns whether or not the object is replacable by an Include object:
    #---------------------------------------------------------------------------

    def is_includable ( self ):
        """ Returns a Boolean value indicating whether the object is replacable
        by an Include object.
        """
        return (self.id != '')

    #---------------------------------------------------------------------------
    #  Replaces any items which have an 'id' with an Include object with the
    #  same 'id', and puts the object with the 'id' into the specified
    #  ViewElements object:
    #---------------------------------------------------------------------------

    def replace_include ( self, view_elements ):
        """ Replaces any items that have an **id** attribute with an Include
        object with the same ID value, and puts the object with the ID
        into the specified ViewElements object.

        Parameters
        ----------
        view_elements : ViewElements object
            A set of Group, Item, and Include objects
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
        """ Returns a ShadowGroup object for the current Group object, which
        recursively resolves all embedded Include objects and which replaces
        each embedded Group object with a corresponding ShadowGroup.
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
    #  Sets the correct container for the content:
    #---------------------------------------------------------------------------

    def set_container ( self ):
        """ Sets the correct container for the content.
        """
        for item in self.content:
            item.container = self

    #---------------------------------------------------------------------------
    #  Returns whether the object should be defined in the user interface:
    #---------------------------------------------------------------------------

    def _defined_when ( self, ui, value ):
        """ Should the object be defined in the user interface?
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
        value, empty = self._parse_label( value )
        value = self._parse_style( value )
        value = self._option( value, '-', 'orientation', 'horizontal' )
        value = self._option( value, '|', 'orientation', 'vertical' )
        value = self._option( value, '=', 'layout',      'split' )
        value = self._option( value, '^', 'layout',      'tabbed' )
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
        """ Returns a "pretty print" version of the Group.
        """
        return "[ %s, %s ]" % (
                   ', '.join( [ item.__repr__() for item in self.content ] ),
                   self._repr_group() )

    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Group traits:
    #---------------------------------------------------------------------------

    def _repr_group ( self ):
        """ Returns a "pretty print" version of the Group's traits.
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
#  'HGroup' class:
#-------------------------------------------------------------------------------

class HGroup ( Group ):
    """ A group whose items are laid out horizontally.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    orientation = 'horizontal'

#-------------------------------------------------------------------------------
#  'VGroup' class:
#-------------------------------------------------------------------------------

class VGroup ( Group ):
    """ A group whose items are laid out vertically.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    orientation = 'vertical'

#-------------------------------------------------------------------------------
#  'VGrid' class:
#-------------------------------------------------------------------------------

class VGrid ( VGroup ):
    """ A group whose items are laid out in 2 columns.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    columns = 2

#-------------------------------------------------------------------------------
#  'HFlow' class:
#-------------------------------------------------------------------------------

class HFlow ( HGroup ):
    """ A group in which items are laid out horizontally, and "wrap" when 
    they exceed the available horizontal space..
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    layout      = 'flow'
    show_labels = false

#-------------------------------------------------------------------------------
#  'VFlow' class:
#-------------------------------------------------------------------------------

class VFlow ( VGroup ):
    """ A group in which items are laid out vertically, and "wrap" when they
    exceed the available vertical space.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    layout      = 'flow'
    show_labels = false

#-------------------------------------------------------------------------------
#  'HSplit' class:
#-------------------------------------------------------------------------------

class HSplit ( Group ):
    """ A horizontal group with splitter bars to separate it from other groups.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    layout      = 'split'
    orientation = 'horizontal'

#-------------------------------------------------------------------------------
#  'VSplit' class:
#-------------------------------------------------------------------------------

class VSplit ( Group ):
    """ A vertical group with splitter bars to separate it from other groups.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    layout      = 'split'
    orientation = 'vertical'

#-------------------------------------------------------------------------------
#  'Tabbed' class:
#-------------------------------------------------------------------------------

class Tabbed ( Group ):
    """ A group that is shown as a tab in a notebook.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    layout = 'tabbed'

#-------------------------------------------------------------------------------
#  'ShadowGroup' class:
#-------------------------------------------------------------------------------

class ShadowGroup ( Group ):
    """ Corresponds to a Group object, but with all embedded Include
    objects resolved, and with all embedded Group objects replaced by 
    corresponding ShadowGroup objects.
    """
    #---------------------------------------------------------------------------
    # Trait definitions:
    #---------------------------------------------------------------------------

    # Group object this is a "shadow" for
    shadow = ReadOnly

    # Number of ShadowGroups in **content**
    groups = ReadOnly

    # Name of the group
    id = ShadowDelegate

    # User interface label for the group
    label = ShadowDelegate

    # Default context object for group items
    object = ShadowDelegate

    # Default style of items in the group
    style = ShadowDelegate

    # Default docking style of items in the group
    dock = ShadowDelegate

    # Default image to display on notebook tabs
    image = ShadowDelegate

    # Category of elements dragged from the view
    export = ShadowDelegate

    # Spatial orientation of the group
    orientation = ShadowDelegate

    # Layout style of the group
    layout = ShadowDelegate

    # The number of columns in the group
    columns = ShadowDelegate

    # Should a border be drawn around group?
    show_border = ShadowDelegate

    # Should labels be added to items in group?
    show_labels = ShadowDelegate

    # Should labels be shown to the left of items (vs. the right)?
    show_left = ShadowDelegate

    # Is group the initially selected page?
    selected = ShadowDelegate

    # Should the group use extra space along its parent group's layout
    # orientation?
    springy = ShadowDelegate

    # Optional help text (for top-level group)
    help = ShadowDelegate

    # Pre-condition for defining the group
    defined_when = ShadowDelegate

    # Pre-condition for showing the group
    visible_when = ShadowDelegate

    # Pre-condition for enabling the group
    enabled_when = ShadowDelegate

    # Amount of padding to add around each item
    padding = ShadowDelegate

    #---------------------------------------------------------------------------
    #  Returns the contents of the ShadowGroup within a specified user interface
    #  building context. This makes sure that all Group types are of the same
    #  type (i.e. Group or Item) and that all Include objects have been replaced
    #  by their substituted values:
    #---------------------------------------------------------------------------

    def get_content ( self, allow_groups = True ):
        """ Returns the contents of the Group within a specified context for
        building a user interface.

        This method makes sure that all Group types are of the same type (i.e.,
        Group or Item) and that all Include objects have been replaced by their
        substituted values.
        """
        # Make a copy of the content:
        result = self.content[:]

        # If result includes any ShadowGroups and they are not allowed,
        # replace them:
        if self.groups != 0:
            if not allow_groups:
                i = 0
                while i < len( result ):
                    value = result[i]
                    if isinstance( value, ShadowGroup ):
                        items         = value.get_content( False )
                        result[i:i+1] = items
                        i += len( items )
                    else:
                        i += 1
            elif (self.groups != len( result )) and (self.layout == 'normal'):
                items   = []
                content = []
                for item in result:
                    if isinstance( item, ShadowGroup ):
                        self._flush_items( content, items )
                        content.append( item )
                    else:
                        items.append( item )
                self._flush_items( content, items )
                result = content

        # Return the resulting list of objects:
        return result

    #---------------------------------------------------------------------------
    #  Returns an id used to identify the group:
    #---------------------------------------------------------------------------

    def get_id ( self ):
        """ Returns an ID for the group.
        """
        if self.id != '':
            return self.id

        return ':'.join( [ item.get_id() for item in self.get_content() ] )

    #---------------------------------------------------------------------------
    #  Sets the correct container for the content:
    #---------------------------------------------------------------------------

    def set_container ( self ):
        """ Sets the correct container for the content.
        """
        pass

    #---------------------------------------------------------------------------
    #  Creates a sub-Group for any items contained in a specified list:
    #---------------------------------------------------------------------------

    def _flush_items ( self, content, items ):
        """ Creates a sub-group for any items contained in a specified list.
        """
        if len( items ) > 0:
            content.append( ShadowGroup( shadow      = self.shadow,
                                         groups      = 0,
                                         label       = '',
                                         show_border = False,
                                         content     = items ).set(
                                         show_labels = self.show_labels,
                                         show_left   = self.show_left,
                                         springy     = self.springy,
                                         orientation = self.orientation ) )
            del items[:]


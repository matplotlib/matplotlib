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
# Date: 07/01/2005
#
#  Symbols defined: TableFilter
#
#------------------------------------------------------------------------------
""" Defines the filter object used to filter items displayed in a table editor.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasPrivateTraits, Str, Any, Instance, Trait, List, Property, Event, \
           Expression, Enum, false, true, Callable

from enthought.traits.ui.api \
    import View, Group, Item, Include, Handler, EnumEditor, EditorFactory

from enthought.traits.ui.menu \
    import Action

from enthought.traits.ui.table_column \
    import ObjectColumn

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

GenericTableFilterRuleOperation = Trait( '=', {
    '=':           'eq',
    '<>':          'ne',
    '<':           'lt',
    '<=':          'le',
    '>':           'gt',
    '>=':          'ge',
    'contains':    'contains',
    'starts with': 'starts_with',
    'ends with':   'ends_with'
} )

#-------------------------------------------------------------------------------
#  'TableFilter' class:
#-------------------------------------------------------------------------------

class TableFilter ( HasPrivateTraits ):
    """ Filter for items displayed in a table.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # UI name of this filter (so the user can identify it in the UI)
    name = Str( 'Default filter' )
    
    # Default name that can be automatically overridden
    _name = Str( 'Default filter' )
    
    # A user-readable description of what kind of object satisfies the filter
    desc = Str( 'All items' )
    
    # A callable function that returns whether the passed object is allowed
    # by the filter
    allowed = Callable( lambda object: True )

    # Is the filter a template (i.e., non-deletable, non-editable)?
    template = false

    #---------------------------------------------------------------------------
    #  Class constants:
    #---------------------------------------------------------------------------

    # Traits that are ignored by the _anytrait_changed() handler
    ignored_traits = [ '_name', 'template', 'desc' ]

    #---------------------------------------------------------------------------
    #  Traits view definitions:
    #---------------------------------------------------------------------------

    traits_view = View( 
        'name{Filter name}', '_', 
        Include( 'filter_view' ),
        title   = 'Edit Filter',
        width   = 0.2,
        buttons = [ 'OK',
                    'Cancel',
                    Action( 
                        name         = 'Help',
                        action       = 'show_help',
                        defined_when = "ui.view_elements.content['filter_view']"
                                       ".help_id != ''"
                    )
                  ]
    )

    searchable_view = View( [
        [ Include( 'search_view' ), '|[]' ],
        [ 'handler.status~', '|[]<>' ],
        [ 'handler.find_next`Find the next matching item`',
          'handler.find_previous`Find the previous matching item`',
          'handler.select`Select all matching items`',
          'handler.OK`Exit search`', '-<>'   ],
        '|<>' ],
        title  = 'Search for',
        kind   = 'livemodal',
        undo   = False,
        revert = False,
        ok     = False,
        cancel = False,
        help   = False,
        width  = 0.25 )

    search_view = Group( Include( 'filter_view' ) )

    filter_view = Group()

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter or search 
        criteria.
        """
        return self.allowed( object )

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user-readable description of what kind of object 
        satisfies the filter.
        """
        return self.desc

    #---------------------------------------------------------------------------
    #  Edits the contents of the filter:
    #---------------------------------------------------------------------------

    def edit ( self, object ):
        """ Edits the contents of the filter.

        The ''object'' parameter is a sample object for the table that the 
        filter will be applied to. It is supplied in case the filter needs to
        extract data or metadata from the object. If the table is empty, the 
        ''object'' argument is None.
        """
        return self.edit_traits( kind = 'livemodal' )

    #---------------------------------------------------------------------------
    #  'object' interface:
    #---------------------------------------------------------------------------

    def __str__ ( self ):
        return self.name

    #---------------------------------------------------------------------------
    #  Event handlers:
    #---------------------------------------------------------------------------

    def _anytrait_changed ( self, name, old, new ):
        if ((name not in self.ignored_traits) and
            ((self.name == self._name) or (self.name == ''))):
            self.name = self._name = self.description()

#-------------------------------------------------------------------------------
#  'EvalTableFilter' class:
#-------------------------------------------------------------------------------

class EvalTableFilter ( TableFilter ):
    """ A table filter based on evaluating an expression.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Override the standard **name** trait
    name = 'Default evaluation filter'

    # Python expression which will be applied to each table item
    expression = Expression

    #---------------------------------------------------------------------------
    #  Traits view definitions:
    #---------------------------------------------------------------------------

    filter_view = Group( 'expression' )

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter or search 
        criteria.
        """
        if self._traits is None:
            self._traits = object.trait_names()
        try:
            return eval( self.expression_, globals(),
                         object.get( *self._traits ) )
        except:
            return False

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user readable description of what kind of object
            satisfies the filter.
        """
        return self.expression

#-------------------------------------------------------------------------------
#  'GenericTableFilterRule' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRule ( HasPrivateTraits ):
    """ A general rule used by a table filter.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Filter this rule is part of
    filter = Instance( 'RuleTableFilter' )

    # Is this rule enabled?
    enabled = false

    # Is this rule an 'and' rule or an 'or' rule?
    and_or = Enum( 'and', 'or' )

    # EnumEditor used to edit the **name** trait:
    name_editor = Instance( EditorFactory )

    # Name of the object trait that this rule applies to
    name = Str

    # Operation to be applied in the rule
    operation = GenericTableFilterRuleOperation

    # Editor used to edit the **value** trait
    value_editor = Instance( EditorFactory )

    # Value to use in the operation when applying the rule to an object
    value = Any

    #---------------------------------------------------------------------------
    #  Class constants:
    #---------------------------------------------------------------------------

    # Traits that are ignored by the _anytrait_changed() handler
    ignored_traits = [ 'filter', 'name_editor', 'value_editor' ]

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        super( GenericTableFilterRule, self ).__init__( **traits )
        if self.name == '':
            names = self.filter._trait_values.keys()
            if len( names ) > 0:
                names.sort()
                self.name    = names[0]
                self.enabled = False

    #---------------------------------------------------------------------------
    #  Handles the value of the 'name' trait changing:
    #---------------------------------------------------------------------------

    def _name_changed ( self, name ):
        """ Handles a change to the value of the **name** trait.
        """
        filter = self.filter
        if filter is not None:
            self.value        = filter._trait_values.get( name )
            self.value_editor = filter._object.base_trait( name ).get_editor()

    #---------------------------------------------------------------------------
    #  Event handlers:
    #---------------------------------------------------------------------------

    def _anytrait_changed ( self, name, old, new ):
        if (name not in self.ignored_traits) and (self.filter is not None):
            self.filter.modified = True
            if name != 'enabled':
                self.enabled = True

    #---------------------------------------------------------------------------
    #  Clones a new object from this one, optionally copying only a specified
    #  set of traits:
    #---------------------------------------------------------------------------

    def clone_traits ( self, traits = None, memo = None, copy = None, 
                             **metadata ):
        """ Clones a new object from this one, optionally copying only a 
        specified set of traits."""
        return super( GenericTableFilterRule, self ).clone_traits( traits,
                       memo, copy, **metadata ).set( 
                       enabled = self.enabled,
                       name    = self.name )

    #---------------------------------------------------------------------------
    #  Returns a description of the filter:
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a description of the filter.
        """
        return '%s %s %s' % ( self.name, self.operation, self.value )

    #---------------------------------------------------------------------------
    #  Returns whether the rule is true for a specified object:
    #---------------------------------------------------------------------------

    def is_true ( self, object ):
        """ Returns whether the rule is true for a specified object.
        """
        try:
            value1 = getattr( object, self.name )
            type1  = type( value1 )
            value2 = self.value
            if type1 is not type( value2 ):
                value2 = type1( value2 )
            return getattr( self, self.operation_ )( value1, value2 )
        except:
            return False

    #---------------------------------------------------------------------------
    #  Implemenations of the various rule operations:
    #---------------------------------------------------------------------------

    def eq ( self, value1, value2 ):
        return (value1 == value2)

    def ne ( self, value1, value2 ):
        return (value1 != value2)

    def lt ( self, value1, value2 ):
        return (value1 < value2)

    def le ( self, value1, value2 ):
        return (value1 <= value2)

    def gt ( self, value1, value2 ):
        return (value1 > value2)

    def ge ( self, value1, value2 ):
        return (value1 >= value2)

    def contains ( self, value1, value2 ):
        return (value1.lower().find( value2.lower() ) >= 0)

    def starts_with ( self, value1, value2 ):
        return (value1[ : len( value2 ) ].lower() == value2.lower())

    def ends_with ( self, value1, value2 ):
        return (value1[ -len( value2 ): ].lower() == value2.lower())

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleEnabledColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleEnabledColumn ( ObjectColumn ):
    """ Table column that indicates whether a filter rule is enabled.
    """
    #---------------------------------------------------------------------------
    #  Returns the value of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_value ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        return [ '', '==>' ][ object.enabled ]

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleAndOrColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleAndOrColumn ( ObjectColumn ):
    """ Table column that displays whether a filter rule is conjoining ('and')
    or disjoining ('or').
    """
    #---------------------------------------------------------------------------
    #  Returns the value of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_value ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        if object.and_or == 'or':
            return 'or'
        return ''

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleNameColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleNameColumn ( ObjectColumn ):
    """ Table column for the name of an object trait.
    """
    #---------------------------------------------------------------------------
    #  Returns the traits editor of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_editor ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        return object.name_editor

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleValueColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleValueColumn ( ObjectColumn ):
    """ Table column for the value of an object trait.
    """
    #---------------------------------------------------------------------------
    #  Returns the traits editor of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_editor ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        return object.value_editor

#-------------------------------------------------------------------------------
#  Defines the columns to display in the generic filter rule table:
#-------------------------------------------------------------------------------

# Columns to display in the table for generic filter rules.
generic_table_filter_rule_columns = [
    GenericTableFilterRuleAndOrColumn( name = 'and_or', label = 'or' ),
    GenericTableFilterRuleNameColumn(  name = 'name' ),
    ObjectColumn(                      name = 'operation' ),
    GenericTableFilterRuleValueColumn( name = 'value' )
]

#-------------------------------------------------------------------------------
#  'RuleTableFilter' class:
#-------------------------------------------------------------------------------

class RuleTableFilter ( TableFilter ):
    """ A table filter based on rules.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Overrides the default **name** trait
    name = 'Default rule-based filter'

    # List of the filter rules to be applied
    rules = List( GenericTableFilterRule )

    # Event fired when the contents of the filter have changed
    modified = Event
    
    # Persistence ID of the view
    view_id = Str( 'enthought.traits.ui.table_filter.RuleTableFilter' )

    # Sample object that the filter will apply to
    _object = Any

    # Map of trait names and default values
    _trait_values = Any

    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------

    error_view = View(
        Item( label = 'A menu or rule based filter can only be created for '
                      'tables with at least one entry'
        ),
        title        = 'Error Creating Filter',
        kind         = 'livemodal',
        close_result = False,
        buttons      = [ 'Cancel' ]
    )

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter or search 
        criteria.
        """
        is_first = is_true = True
        for rule in self.rules:
            if rule.and_or == 'or':
                if is_true and (not is_first):
                    return True
                is_true = True
            if is_true:
                is_true = rule.is_true( object )
            is_first = False
        return is_true

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user-readable description of the kind of object that 
            satisfies the filter.
        """
        ors  = []
        ands = []
        if len( self.rules ) > 0:
            for rule in self.rules:
                if rule.and_or == 'or':
                    if len( ands ) > 0:
                        ors.append( ' and '.join( ands ) )
                        ands = []
                ands.append( rule.description() )
        if len( ands ) > 0:
            ors.append( ' and '.join( ands ) )
        if len( ors ) == 1:
            return ors[0]
        if len( ors ) > 1:
            return ' or '.join( [ '(%s)' % t for t in ors ] )
        return super( RuleTableFilter, self ).description()

    #---------------------------------------------------------------------------
    #  Edits the contents of the filter:
    #---------------------------------------------------------------------------

    def edit ( self, object ):
        """ Edits the contents of the filter.

        The ''object'' parameter is a sample object for the table that the 
        filter will be applied to. It is supplied in case the filter needs to
        extract data or metadata from the object. If the table is empty, the 
        ''object'' argument is None.
        """
        self._object = object
        if object is None:
            return self.edit_traits( view = 'error_view' )
            
        names              = object.editable_traits()
        self._trait_values = object.get( names )
        return self.edit_traits( view = View( [
                        [ 'name{Filter name}', '_' ],
                        [ Item( 'rules',
                                id     = 'rules_table',
                                editor = self._get_table_editor( names ) ),
                          '|<>' ] ],
                        id        = self.view_id,
                        title     = 'Edit Filter',
                        kind      = 'livemodal',
                        resizable = True,
                        undo      = False,
                        revert    = False,
                        help      = False,
                        width     = 0.4,
                        height    = 0.3 ) )

    #---------------------------------------------------------------------------
    #  Returns a table editor to use for editing the filter:
    #---------------------------------------------------------------------------

    def _get_table_editor ( self, names ):
        """ Returns a table editor to use for editing the filter.
        """
        from enthought.traits.ui.api import TableEditor

        return TableEditor( columns        = generic_table_filter_rule_columns,
                            orientation    = 'vertical',
                            deletable      = True,
                            sortable       = False,
                            configurable   = False,
                            auto_size      = False,
                            auto_add       = True,
                            row_factory    = GenericTableFilterRule,
                            row_factory_kw = {
                                'filter':      self,
                                'name_editor': EnumEditor( values = names ) } )

    #---------------------------------------------------------------------------
    #  Returns the state to be pickled (override of object):
    #---------------------------------------------------------------------------

    def __getstate__ ( self ):
        """ Returns the state to be pickled. 
        
        This definition overrides **object**.
        """
        dict = self.__dict__.copy()
        if '_object' in dict:
            del dict[ '_object' ]
            del dict[ '_trait_values' ]
        return dict
        
    #---------------------------------------------------------------------------
    #  Handles the 'rules' trait being changed:  
    #---------------------------------------------------------------------------

    def _rules_changed ( self, rules ):
        """ Handles a change to the **rules** trait.
        """
        for rule in rules:
            rule.filter = self

#-------------------------------------------------------------------------------
#  Defines the columns to display in the menu filter rule table:
#-------------------------------------------------------------------------------

# Columns to display in the table for menu filters.
menu_table_filter_rule_columns = [
    GenericTableFilterRuleEnabledColumn( name = 'enabled', label = '' ),
    GenericTableFilterRuleNameColumn(    name = 'name' ),
    ObjectColumn(                        name = 'operation' ),
    GenericTableFilterRuleValueColumn(   name = 'value' )
]

#-------------------------------------------------------------------------------
#  'MenuTableFilter' class:
#-------------------------------------------------------------------------------

class MenuTableFilter ( RuleTableFilter ):
    """ A table filter based on a menu of rules.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Overrides the default **name** trait
    name = 'Default menu-based filter'
    
    # Overrides the persistence ID of the view
    view_id = Str( 'enthought.traits.ui.table_filter.MenuTableFilter' )

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter or search 
        criteria.
        """
        for rule in self.rules:
            if rule.enabled and (not rule.is_true( object )):
                return False
        return True

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user8readable description of what kind of object 
            satisfies the filter.
        """
        result = ' and '.join( [ rule.description() for rule in self.rules
                                 if rule.enabled ] )
        if result != '':
            return result
        return 'All items'

    #---------------------------------------------------------------------------
    #  Returns a table editor to use for editing the filter:
    #---------------------------------------------------------------------------

    def _get_table_editor ( self, names ):
        """ Returns a table editor to use for editing the filter.
        """
        from enthought.traits.ui.api import TableEditor

        names       = self._object.editable_traits()
        name_editor = EnumEditor( values = names )
        if len( self.rules ) == 0:
            self.rules  = [ GenericTableFilterRule(
                                filter      = self,
                                name_editor = name_editor ).set(
                                name        = name )
                            for name in names ]
            for rule in self.rules:
                rule.enabled = False

        return TableEditor( columns        = menu_table_filter_rule_columns,
                            orientation    = 'vertical',
                            deletable      = True,
                            sortable       = False,
                            configurable   = False,
                            auto_size      = False,
                            auto_add       = True,
                            row_factory    = GenericTableFilterRule,
                            row_factory_kw = {
                                'filter':      self,
                                'name_editor': name_editor  } )

#-------------------------------------------------------------------------------
#  Define some standard template filters:
#-------------------------------------------------------------------------------

EvalFilterTemplate = EvalTableFilter( name     = 'Evaluation filter template',
                                      template = True )
RuleFilterTemplate = RuleTableFilter( name     = 'Rule-based filter template',
                                      template = True )
MenuFilterTemplate = MenuTableFilter( name     = 'Menu-based filter template',
                                      template = True )


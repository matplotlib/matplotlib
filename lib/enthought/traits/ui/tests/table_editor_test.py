#-------------------------------------------------------------------------------
#    
#  TableEditor test case for Traits UI 
#    
#  Written by: David C. Morrill
#    
#  Date: 07/05/2005
#    
#  (c) Copyright 2005 by Enthought, Inc.  
#    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:  
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasStrictTraits, Str, Int, Regex, List, Instance
    
from enthought.traits.ui.api \
    import View, Group, Item, TableEditor, EnumEditor
    
from enthought.traits.ui.menu \
    import NoButtons      
    
from enthought.traits.ui.table_column \
    import ObjectColumn
    
from enthought.traits.ui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
           MenuFilterTemplate, EvalFilterTemplate
        
#-------------------------------------------------------------------------------
#  'Person' class:  
#-------------------------------------------------------------------------------



                   
class Person ( HasStrictTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    name  = Str
    age   = Int
    phone = Regex( value = '000-0000', regex = '\d\d\d[-]\d\d\d\d' )
    state = Str
    
    #---------------------------------------------------------------------------
    #  Traits view definition:  
    #---------------------------------------------------------------------------
    
    traits_view = View( 'name', 'age', 'phone', 'state',
                        title  = 'Create new person',
                        width  = 0.18,
                        undo   = False, 
                        revert = False, 
                        help   = False )    
    
#-------------------------------------------------------------------------------
#  Sample data:  
#-------------------------------------------------------------------------------
    
people = [
   Person( name = 'Dave',   age = 39, phone = '555-1212' ),
   Person( name = 'Mike',   age = 28, phone = '555-3526' ),
   Person( name = 'Joe',    age = 34, phone = '555-6943' ),
   Person( name = 'Tom',    age = 22, phone = '555-7586' ),
   Person( name = 'Dick',   age = 63, phone = '555-3895' ),
   Person( name = 'Harry',  age = 46, phone = '555-3285' ),
   Person( name = 'Sally',  age = 43, phone = '555-8797' ),
   Person( name = 'Fields', age = 31, phone = '555-3547' )
]

#-------------------------------------------------------------------------------
#  'AgeFilter' class:  
#-------------------------------------------------------------------------------

class AgeFilter ( TableFilter ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------

    name = "Age filter"
    _name = "Age filter"

    age = Int( 0 )
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:    
    #---------------------------------------------------------------------------
        
    #filter_view = Group( 'age{Age >=}' )
    
    #---------------------------------------------------------------------------
    #  Returns whether an object passes the filter or not:  
    #---------------------------------------------------------------------------
        
    def filter ( self, person ):
        """ Returns whether an object passes the filter or not.
        """
        return (person.age >= self.age)
        
    #---------------------------------------------------------------------------
    #  Returns a user readable description of what the filter does:    
    #---------------------------------------------------------------------------
    
    def description ( self ):
        """ Returns a user readable description of what the filter does.
        """
        return 'Age >= %d' % self.age     

    def _age_changed(self, old, new):
        self.name = self.description()
        print 'AgeFilter _age_changed', self.name
        
#-------------------------------------------------------------------------------
#  'NameFilter' class:  
#-------------------------------------------------------------------------------

class NameFilter ( TableFilter ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    mname = Str
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:    
    #---------------------------------------------------------------------------
        
    filter_view = Group( 'mname{Name contains}' )
    
    #---------------------------------------------------------------------------
    #  Returns whether an object passes the filter or not:  
    #---------------------------------------------------------------------------
        
    def filter ( self, person ):
        """ Returns whether an object passes the filter or not.
        """
        return (person.name.lower().find( self.mname.lower() ) >= 0)
        
    #---------------------------------------------------------------------------
    #  Returns a user readable description of what the filter does:    
    #---------------------------------------------------------------------------
    
    def description ( self ):
        """ Returns a user readable description of what the filter does.
        """
        return "Name contains '%s'" % self.mname     

#-------------------------------------------------------------------------------
#  Table editor definition:  
#-------------------------------------------------------------------------------

filters      = [ AgeFilter( age = 30 ), NameFilter( mname = 'd' ), EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate,
                 ]

def evaluate_value(v):
    print 'evaluate_value', v
    return str(v)
#-------------------------------------------------------------------------------
#  'TableTest' class:  
#-------------------------------------------------------------------------------

class TableTest ( HasStrictTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    #people = Instance( Person )
    people = List( Person )
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------

    _valid_states = List(["AL", "AR", "AZ", "AK"])
    
    _state_editor = EnumEditor(
        name     = "_valid_states", 
        evaluate = evaluate_value,
        object   = 'table_editor_object'
    )
    
    
    table_editor = TableEditor(
        columns            = [ ObjectColumn( name = 'name' ),
                               ObjectColumn( name = 'age' ),
                               ObjectColumn( name = 'phone' ),
                               ObjectColumn( name = 'state', 
                                             editor=_state_editor), ],
        editable           = True,
        deletable          = True,
        sortable           = True,
        sort_model         = True,
        show_lines         = True,
        orientation        = 'vertical',
        show_column_labels = True,
        edit_view          = View( [ 'name', 'age', 'phone', 'state', '|[]' ], 
                                   resizable = True ),
        filter             = None,
        filters            = filters,
        row_factory        = Person
    )
    
        
    traits_view = View(
        [ Item( 'people',
                id     = 'people',
                editor = table_editor ),
          '|[]<>' ],
        title     = 'Table Editor Test',
        id        = 'enthought.traits.ui.tests.table_editor_test',
        dock      = 'horizontal',
        width     = .4,
        height    = .3,
        resizable = True,
        buttons   = NoButtons,
        kind      = 'live' )

    
#-------------------------------------------------------------------------------
#  Run the tests:  
#-------------------------------------------------------------------------------
                 
if __name__ == '__main__':
    tt = TableTest( people = people )
    tt.configure_traits()
    for p in tt.people:
        p.print_traits()
        print '--------------'

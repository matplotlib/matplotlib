from enthought.traits.api \
    import *
    
from enthought.traits.ui.api \
    import *
    
from enthought.traits.ui.menu \
    import *

#-------------------------------------------------------------------------------
#  'Person' class:  
#-------------------------------------------------------------------------------
                        
class Person ( Handler ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    name  = Str
    age   = Int
    phone = Regex( value = '000-0000', regex = '\d\d\d[-]\d\d\d\d' )
    notes = Str
    
    #---------------------------------------------------------------------------
    #  Handles the 'Annoy' button being clicked:  
    #---------------------------------------------------------------------------
        
    def _annoy_clicked ( self, info ):
        self.edit_traits( view = View( title   = 'Annoying',
                                       kind    = 'modal',
                                       buttons = [ 'OK' ] ) )

#-------------------------------------------------------------------------------
#  Run the tests:  
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    AnnoyButton = Action( name         = 'Annoy', 
                          tooltip      = 'Click me to be annoyed',
                          enabled_when = 'age >= 40' )
    
    person = Person( name = 'Bill', age = 42, phone = '555-1212' )
    
    fields = Group( 'name', 'age', 'phone', 'notes~' )
             
    person.notes = ("Should have 6 standard 'live' buttons: Undo, Redo, "
                    "Revert, OK, Cancel, Help") 
    person.configure_traits( view = View( fields,
                             kind    = 'livemodal',
                             buttons = LiveButtons ) )
    
    person.notes = ("Should have 5 standard 'modal' buttons: Apply, Revert, "
                    "OK, Cancel, Help") 
    person.configure_traits( view = View( fields,
                                          buttons = ModalButtons ) )
                                          
    person.notes = "Should have 2 standard buttons: OK, Cancel"
    person.configure_traits( 
               view = View( fields,
                            buttons = [ OKButton, CancelButton ] ) )
                                          
    person.notes = "Should have 1 standard button: OK (enabled when age >= 40)"
    person.configure_traits(
               view = View( fields,
                            buttons      = [ Action( name = 'OK', 
                            enabled_when = 'age >= 40' ) ] ) )
                                          
    person.notes = "Should have 1 standard button: OK (visible when age >= 40)"
    person.configure_traits( 
               view = View( fields,
                            buttons      = [ Action( name = 'OK', 
                            visible_when = 'age >= 40' ) ] ) )
                                          
    person.notes = ("Should have 2 standard buttons: OK, Help (defined when "
                    "age >= 50)")
    person.configure_traits( 
               view = View( fields,
                            buttons      = [ 'OK', Action( name = 'Help',
                            defined_when = 'age >= 50' ) ] ) )
                                          
    person.notes = ("Should have 1 user and 5 standard buttons: Annoy (enabled "
                    "when age >= 40), Apply, Revert, OK, Cancel, Help")
    person.configure_traits( 
               view    = View( fields,
               buttons = [ AnnoyButton ] + ModalButtons ) )
                                          

from enthought.traits.api    import *
from enthought.traits.ui.api import *

def evaluate_value(v):
    print 'evaluate_value', v
    return str(v)

class Team ( HasTraits ):

    captain = Str( 'Dick' )
    players = List( [ 'Tom', 'Dick', 'Harry', 'Sally' ], Str )
    
    captain_editor = EnumEditor( name = 'players', evaluate=evaluate_value ) 
    
    view = View( Item( 'captain', editor = captain_editor),
                 '_', 
                 'players@',
                 height=200 )
                 
if __name__ == '__main__':
    team = Team()
    team.configure_traits()  
    team.print_traits()
                   

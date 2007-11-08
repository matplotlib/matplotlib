#-------------------------------------------------------------------------------
#  
#  Traits UI DataBase manager  
#  
#  Written by: David C. Morrill
#  
#  Date: 12/13/2005
#  
#  (c) Copyright 2005 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------
""" Defines the Traits UI database manager.
"""
#-------------------------------------------------------------------------------
#  Imports:  
#-------------------------------------------------------------------------------

import shelve
import os
import shutil

from enthought.traits.api \
    import HasPrivateTraits, Enum, Instance, List, Str, Button, View, Item, \
           SetEditor, Handler, VGroup, HGroup
    
from enthought.traits.trait_base \
    import traits_home                             
    
from enthought.traits.ui.menu \
    import NoButtons
    
#-------------------------------------------------------------------------------
#  Returns the name of the traits UI database:  
#-------------------------------------------------------------------------------
        
def ui_db_name ( ):
    """ Returns the name of the traits UI database.
    """
    return os.path.join( traits_home(), 'traits_ui' )
    
#-------------------------------------------------------------------------------
#  Opens the traits UI database:  
#-------------------------------------------------------------------------------
        
def get_ui_db ( mode = 'r' ):
    """ Opens the traits UI database.
    """
    try:
        return shelve.open( ui_db_name(), flag = mode, protocol = -1 )
    except:
        return None
                                           
#-------------------------------------------------------------------------------
#  'TUIDB' class:  
#-------------------------------------------------------------------------------

class TUIDB ( Handler ):
    """ Handles the Traits UI database.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # All items currently in the database
    all_items = List( Str )
    
    # All items to be discarded from the database
    discard = List( Str )
    
    # Status message
    status = Str
    
    # Action buttons:
    
    backup  = Button( 'Backup' )
    restore = Button( 'Restore' )
    delete  = Button( 'Delete' )
    update  = Button( 'Update' )
    exit    = Button( 'Exit' )
    
    #---------------------------------------------------------------------------
    #  Trait view definitions:  
    #---------------------------------------------------------------------------
        
    content = Item( 'discard{}', editor = SetEditor(
                                 name               = 'all_items',
                                 left_column_title  = 'Keep',
                                 right_column_title = 'Delete' ) )
    
    traits_view = View( 
        VGroup( 
            '<content>',
            '_',
            HGroup( 'backup', 'restore', 'update', 'delete', 'exit', 
                    '10', 'status~',
                    show_labels = False )
        ),
        title     = 'Traits UI Database Utility',
        id        = 'enthought.traits.ui.tuidb',
        width     = 0.4,
        height    = 0.3,
        resizable = True,
        buttons   = NoButtons
    )
        
    view = View(
        VGroup(
            '<content>',
            '_',
            HGroup( 'backup', 'restore', 'update', 'delete', '10', 'status~',
                    show_labels = False )
        ),
        title = 'Traits UI DB',
        id    = 'enthought.traits.ui.tuidb_plugin'
    )
                 
    #---------------------------------------------------------------------------
    #  Initializes the object:  
    #---------------------------------------------------------------------------
    
    def __init__ ( self, **traits ):
        """ Initializes the Traits UI database manager.
        """
        super( TUIDB, self ).__init__( **traits )
        self.update_all_items()
        
    #---------------------------------------------------------------------------
    #  Determines the set of available database keys:  
    #---------------------------------------------------------------------------
                
    def update_all_items ( self ):
        """ Determines the set of available database keys.
        """
        db = get_ui_db()
        if db is not None:
            keys = db.keys()
            db.close()
            keys.sort()
            self.all_items = keys 
            
    #---------------------------------------------------------------------------
    #  Handles the 'discard' list being changed:  
    #---------------------------------------------------------------------------
                        
    def object_discard_changed ( self, info ):
        """ Handles the **discard** list being changed.
        """
        info.delete.enabled = (len( self.discard ) > 0)
        
    #---------------------------------------------------------------------------
    #  Backs up the current traits UI database:
    #---------------------------------------------------------------------------
                
    def _backup_changed ( self ):
        """ Backs up the current traits UI database.
        """
        name = ui_db_name()
        try:
            shutil.copy( name, name + '.bak' )
            self.status = 'The Traits UI database has been backed up'
        except:
            self.status = 'Could not back up the Traits UI database'
        
    #---------------------------------------------------------------------------
    #  Restores the current backup of the traits UI database:  
    #---------------------------------------------------------------------------
                
    def _restore_changed ( self ):
        """ Restores the current backup of the traits UI database.
        """
        name = ui_db_name()
        try:
            shutil.copy( name + '.bak', name )
            self.update_all_items()
            self.status = 'The Traits UI database has been restored'
        except:
            self.status = 'Could not restore the Traits UI database'
        
    #---------------------------------------------------------------------------
    #  Deletes the specified items from the traits UI database:  
    #---------------------------------------------------------------------------
                
    def _delete_changed ( self ):
        """ Deletes the specified items from the traits UI database.
        """
        db = get_ui_db( mode = 'c' )
        if db is not None:
            all_items = self.all_items
            n         = len( self.discard )
            for item in self.discard:
                del db[ item ]
                all_items.remove( item )
            db.close()
            self.status  = ('%d items deleted from the Traits UI database' % n)
            self.discard = []
        else:
            self.status = 'Cannot access the Traits UI database'
            
    #---------------------------------------------------------------------------
    #  Updates the list of defined items in the traits UI database:  
    #---------------------------------------------------------------------------
                        
    def _update_changed ( self ):
        """ Updates the list of defined items in the traits UI database.
        """
        self.update_all_items()
        
    #---------------------------------------------------------------------------
    #  Exits the utility:
    #---------------------------------------------------------------------------
                
    def object_exit_changed ( self, info ):
        """ Exits the utility.
        """
        if info.initialized:
            info.ui.dispose()
        
#-------------------------------------------------------------------------------
#  Create export objects:
#-------------------------------------------------------------------------------

# Exported instance of TUIDB
tuidb = TUIDB()

#-------------------------------------------------------------------------------
#  Run the utility:  
#-------------------------------------------------------------------------------
                   
if __name__ == '__main__':
    tuidb.configure_traits()
    

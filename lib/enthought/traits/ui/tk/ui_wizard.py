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
# Date: 12/02/2004
# Description: Create a wizard-based Tkinter user interface for a specified UI
#              object.
#
#  Symbols defined: ui_wizard
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk
import wx.wizard as wz

from ui_panel         import fill_panel_for_group
from editor           import Editor
from enthought.traits.api import Trait, Str

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Only allow 'None' or a string value:
none_str_trait = Trait( '', None, str )

#-------------------------------------------------------------------------------
#  Creates a wizard-based Tkinter user interface for a specified UI object:
#-------------------------------------------------------------------------------

def ui_wizard ( ui, parent ):
        
    # Create the copy of the 'context' we will need while editing:
    context     = ui.context
    ui._context = context
    new_context = {}
    for name, value in context.items():
        new_context[ name ] = value.clone_traits()
    ui.context = new_context
    
    # Now bind the context values to the 'info' object:
    ui.info.bind_context()
    
    # Create the Tkinter wizard window:
    ui.control = wizard = wz.Wizard( parent, -1, ui.view.title )
    
    # Create all of the wizard pages:
    pages        = []
    editor_pages = []
    info         = ui.info
    shadow_group = ui.view.content.get_shadow( ui )
    min_dx = min_dy = 0
    for group in shadow_group.get_content():
        page = UIWizardPage( wizard, editor_pages )
        pages.append( page )
        fill_panel_for_group( page, group, ui )
        
        # Size the page correctly, then calculate cumulative minimum size:
        sizer = page.GetSizer()
        sizer.Fit( page )
        size   = sizer.CalcMin()
        min_dx = max( min_dx, size.GetWidth() )
        min_dy = max( min_dy, size.GetHeight() )
        
        # If necessary, create a GroupEditor and attach it to the right places:
        id = group.id
        if id or group.enabled_when:
            page.editor = editor = GroupEditor( control = page )
            if id:
                page.id = id
                editor_pages.append( page )
                info.bind( id, editor )
            if group.enabled_when:
                ui.add_enabled( group.enabled_when, editor )
                
    # Size the wizard correctly:                
    wizard.SetPageSize( wx.Size( min_dx, min_dy ) )
    
    # Set up the wizard 'page changing' event handler:
    wz.EVT_WIZARD_PAGE_CHANGING( wizard, wizard.GetId(), page_changing )
    
    # Size the wizard and the individual pages appropriately:
    prev_page = pages[0]
    wizard.FitToPage( prev_page )
    
    # Link the pages together:
    for page in pages[1:]:
        page.SetPrev( prev_page )
        prev_page.SetNext( page )
        prev_page = page
    
    # Finalize the display of the wizard:
    try:
        ui.prepare_ui()
    except:
        ui.control.Destroy()
        ui.control.ui = None
        ui.control    = None
        ui.result     = False
        raise
        
    # Position the wizard on the display:
    ui.handler.position( ui.info )
    
    # Run the wizard:
    if wizard.RunWizard( pages[0] ):
        # If successful, apply the modified context to the original context: 
        original = ui._context
        for name, value in ui.context.items():
            original[ name ].copy_traits( value )
        ui.result = True
    else:
        ui.result = False
    
    # Clean up loose ends, like restoring the original context:
    ui.control.Destroy()
    ui.control  = None
    ui.context  = ui._context
    ui._context = {}
    
#-------------------------------------------------------------------------------
#  Handles the user attempting to change the current wizard page:
#-------------------------------------------------------------------------------
    
def page_changing ( event ):
    """ Handles the user attempting to change the current wizard page.
    """
    # Get the page the user is trying to go to:
    page = event.GetPage()
    if event.GetDirection():
       new_page = page.GetNext()
    else:
       new_page = page.GetPrev()
       
    # If the page has a disabled GroupEditor object, veto the page change:
    if ((new_page is not None) and 
        (new_page.editor is not None) and 
        (not new_page.editor.enabled)):
        event.Veto()
	
	# If their is a message associated with the editor, display it:
        msg = new_page.editor.msg
        if msg != '':
            wx.MessageBox( msg )
        
#-------------------------------------------------------------------------------
#  'UIWizardPage' class:
#-------------------------------------------------------------------------------
        
class UIWizardPage ( wz.PyWizardPage ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, wizard, pages ):
        wz.PyWizardPage.__init__ ( self, wizard )
        self.next  = self.previous = self.editor = None
        self.pages = pages

    #---------------------------------------------------------------------------
    #  Sets the next page after this one:
    #---------------------------------------------------------------------------
    
    def SetNext ( self, page ):
        self.next = page

    #---------------------------------------------------------------------------
    #  Sets the previous page before this one:
    #---------------------------------------------------------------------------
    
    def SetPrev ( self, page ):
        self.previous = page
        
    #---------------------------------------------------------------------------
    #  Returns the next page after this one:
    #---------------------------------------------------------------------------
    
    def GetNext ( self ):
        """ Returns the next page after this one.
        """
        editor = self.editor
        if (editor is not None) and (editor.next != ''):
            next = editor.next
            if next == None:
                return None
            for page in self.pages:
                if page.id == next:
                    return page
        return self.next
        
    #---------------------------------------------------------------------------
    #  Returns the previous page before this one:
    #---------------------------------------------------------------------------
    
    def GetPrev ( self ):
        """ Returns the previous page before this one.
        """
        editor = self.editor
        if (editor is not None) and (editor.previous != ''):
            previous = editor.previous
            if previous is None:
                return None
            for page in self.pages:
                if page.id == previous:
                    return page
        return self.previous
    
#-------------------------------------------------------------------------------
#  'GroupEditor' class:
#-------------------------------------------------------------------------------
        
class GroupEditor ( Editor ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    next     = none_str_trait  # Id of next page to display 
    previous = none_str_trait  # Id of previous page to display
    msg      = Str             # Message to display if user can't link to page
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, **traits ):
        """ Initializes the object.
        """
        self.set( **traits )
    

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
# Description: Create a panel-based Tkinter user interface for a specified UI
#              object.
#
#  Symbols defined: ui_panel
#                   panel
#                   fill_panel_for_group
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk
import wx.html as wh
import re

from enthought.traits.ui.api               import Group
from enthought.traits.trait_base       import user_name_for
from enthought.traits.ui.undo          import UndoHistory
from enthought.traits.ui.help_template import help_template
from helper                            import position_near
from constants                         import screen_dx, screen_dy, WindowColor

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Pattern of all digits:    
all_digits = re.compile( r'\d+' )

#-------------------------------------------------------------------------------
#  Creates a panel-based Tkinter user interface for a specified UI object:
#-------------------------------------------------------------------------------

def ui_panel ( ui, parent ):
    """ Creates a panel-based Tkinter user interface for a specified UI object.
    """
    ui_panel_for( ui, parent, True )

#-------------------------------------------------------------------------------
#  Creates a subpanel-based Tkinter user interface for a specified UI object:
#-------------------------------------------------------------------------------

def ui_subpanel ( ui, parent ):
    """ Creates a subpanel-based Tkinter user interface for a specified UI 
        object.
    """
    ui_panel_for( ui, parent, False )

#-------------------------------------------------------------------------------
#  Creates a panel-based Tkinter user interface for a specified UI object:
#-------------------------------------------------------------------------------

def ui_panel_for ( ui, parent, buttons ):
    """ Creates a panel-based Tkinter user interface for a specified UI object.
    """
    ui.control = Panel( ui, parent, buttons ).control
    try:
        ui.prepare_ui()
    except:
        ui.control.Destroy()
        ui.control = None
        ui.result  = False
        raise
    ui.result = True
    
#-------------------------------------------------------------------------------
#  'Panel' class:
#-------------------------------------------------------------------------------

class Panel ( object ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object: 
    #---------------------------------------------------------------------------
        
    def __init__ ( self, ui, parent, allow_buttons ):
        """ Initializes the object.
        """
        self.ui = ui
        history = None
        view    = ui.view
        cpanel  = parent
        buttons = False
        if allow_buttons:
            buttons = view.undo or view.revert
            if buttons:
                ui.history = history = UndoHistory()
            buttons |= view.help
            if buttons:
                cpanel = wx.Panel( parent, -1 )
        
        # Create the actual trait sheet panel and imbed it in a scrollable 
        # window:
        sizer       = wx.BoxSizer( wx.VERTICAL )
        sw          = wx.ScrolledWindow( cpanel )
        trait_sheet = panel( ui, sw )
        sizer.Add( trait_sheet, 1, wx.EXPAND )
        
        sw.SetAutoLayout( True )
        sw.SetSizer( sizer )
        sizer.Fit( sw )
        sw.SetScrollRate( 16, 16 )
        
        if not buttons:
            self.control = sw
            return
            
        self.control = cpanel
        
        sw_sizer = wx.BoxSizer( wx.VERTICAL )
        sw_sizer.Add( sw, 1, wx.EXPAND )
        
        # Add the special function buttons:
        sw_sizer.Add( wx.StaticLine( cpanel, -1 ), 0, wx.EXPAND )
        b_sizer = wx.BoxSizer( wx.HORIZONTAL )
        if view.undo:
            self.undo = self._add_button( 'Undo', self._on_undo, b_sizer, 
                                          False )
            self.redo = self._add_button( 'Redo', self._on_redo, b_sizer, 
                                          False )
            history.on_trait_change( self._on_undoable, 'undoable', 
                                     dispatch = 'ui' )
            history.on_trait_change( self._on_redoable, 'redoable', 
                                     dispatch = 'ui' )
        if view.revert:
            self.revert = self._add_button( 'Revert', self._on_revert, b_sizer, 
                                            False )
            history.on_trait_change( self._on_revertable, 'undoable', 
                                     dispatch = 'ui' )
        if view.help:
            self._add_button( 'Help', self._on_help, b_sizer )
            
        sw_sizer.Add( b_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5 )
        
        cpanel.SetAutoLayout( True )
        cpanel.SetSizer( sw_sizer )
        sw_sizer.Fit( cpanel )
   
    #---------------------------------------------------------------------------
    #  Handles an 'Undo' change request:
    #---------------------------------------------------------------------------
           
    def _on_undo ( self, event ):
        """ Handles an 'Undo' change request.
        """
        self.ui.history.undo()
    
    #---------------------------------------------------------------------------
    #  Handles a 'Redo' change request:
    #---------------------------------------------------------------------------
           
    def _on_redo ( self, event ):
        """ Handles a 'Redo' change request.
        """
        self.ui.history.redo()
    
    #---------------------------------------------------------------------------
    #  Handles a 'Revert' all changes request:
    #---------------------------------------------------------------------------
           
    def _on_revert ( self, event ):
        """ Handles a 'Revert' all changes request.
        """
        self.ui.history.revert()
    
    #---------------------------------------------------------------------------
    #  Handles the 'Help' button being clicked:
    #---------------------------------------------------------------------------
           
    def _on_help ( self, event ):
        """ Handles the 'Help' button being clicked.
        """
        show_help( self.ui, event.GetEventObject() )
            
    #-----------------------------------------------------------------------
    #  Handles the undo history 'undoable' state changing:
    #-----------------------------------------------------------------------
            
    def _on_undoable ( self, state ):
        """ Handles the undo history 'undoable' state changing.
        """
        self.undo.Enable( state )
            
    #---------------------------------------------------------------------------
    #  Handles the undo history 'redoable' state changing:
    #---------------------------------------------------------------------------
            
    def _on_redoable ( self, state ):
        """ Handles the undo history 'redoable' state changing.
        """
        self.redo.Enable( state )
            
    #---------------------------------------------------------------------------
    #  Handles the 'revert' state changing:
    #---------------------------------------------------------------------------
            
    def _on_revertable ( self, state ):
        """ Handles the 'revert' state changing.
        """
        self.revert.Enable( state )
    
    #---------------------------------------------------------------------------
    #  Creates a new dialog button:
    #---------------------------------------------------------------------------
    
    def _add_button ( self, label, action, sizer, enabled = True ):
        """ Creates a new dialog button.
        """
        button = wx.Button( self.control, -1, label )
        wx.EVT_BUTTON( self.control, button.GetId(), action )
        sizer.Add( button, 0, wx.LEFT, 5 )
        button.Enable( enabled )
        return button
    
#-------------------------------------------------------------------------------
#  Creates a panel-based Tkinter user interface for a specified UI object:
#
#  Note: This version does not modify the UI object passed to it.
#-------------------------------------------------------------------------------

def panel ( ui, parent ):
    """ Creates a panel-based Tkinter user interface for a specified UI object.
    """
    # Bind the context values to the 'info' object:
    ui.info.bind_context()
    
    # Get the content that will be displayed in the user interface:
    shadow_group = ui.view.content.get_shadow( ui )
    ui._groups   = content = shadow_group.get_content()
    
    # If there is 0 or 1 Groups in the content, create a single panel for it:
    if len( content ) <= 1:
        panel = wx.Panel( parent, -1 )
        if len( content ) == 1:
            # Fill the panel with the Group's content:
            fill_panel_for_group( panel, content[0], ui )
        
        # Make sure the panel and its contents have been laid out properly:
        panel.GetSizer().Fit( panel )
        
        # Return the panel that was created:
        return panel
        
    # Create a notebook which will contain a page for each group in the content:
    nb    = wx.Notebook( parent, -1 )
    nbs   = wx.NotebookSizer( nb )
    nb.ui = ui
    wx.EVT_NOTEBOOK_PAGE_CHANGED( parent, nb.GetId(), _page_changed )
    
    count = 0
    
    # Create a notebook page for each group in the content:
    for group in content:
        page_name = group.label
        count    += 1
        if (page_name is None) or (page_name == ''):
           page_name = 'Page %d' % count
        wrapper = wx.Panel( nb, -1 )
        panel   = wx.Panel( wrapper, -1 )
        fill_panel_for_group( panel, group, ui )
        panel.GetSizer().Fit( panel )
        sizer = wx.BoxSizer( wx.VERTICAL )
        sizer.Add( panel, 0, wx.EXPAND | wx.ALL, 5 )
        wrapper.SetSizer( sizer )
        nb.AddPage( wrapper, page_name, group.selected )
        
    # Finish laying out the notebook and set its size correctly:
    nbs.Fit( nb )
    dx, dy = nb.GetSizeTuple()
    size   = wx.Size( max( len( content ) * 54, 260, dx ), dy )
    nb.SetSize( size )
    
    # Return the notebook as the result:
    return nb
    
#-------------------------------------------------------------------------------
#  Handles a notebook page being 'turned':
#-------------------------------------------------------------------------------
    
def _page_changed ( event ):
    nb = event.GetEventObject()
    nb.ui._active_group = event.GetSelection()
    
#-------------------------------------------------------------------------------
#  Displays a help window for the specified UI's active Group:
#-------------------------------------------------------------------------------
    
def show_help ( ui, button ):
    """ Displays a help window for the specified UI's active Group.
    """
    group    = ui._groups[ ui._active_group ]
    template = help_template()
    if group.help is not None:
        header = template.group_help % group.help
    else:
        header = template.no_group_help
    fields = []
    for item in group.get_content( False ):
        if not item.is_spacer():
            fields.append( template.item_help % ( 
                           item.get_label( ui ), item.get_help( ui ) ) )
    html = template.group_html % ( header, '\n'.join( fields ) ) 
    HTMLHelpWindow( button, html, .25, .33 )
    
#-------------------------------------------------------------------------------
#  Displays a pop-up help window for a single trait:
#-------------------------------------------------------------------------------
    
def show_help_popup ( event ):
    """ Displays a pop-up help window for a single trait.
    """
    control  = event.GetEventObject()
    template = help_template()
    html     = template.item_html % ( control.GetLabel(), 
                                      control.trait.get_help() )
    HTMLHelpWindow( control, html, .25, .13 )
    
#-------------------------------------------------------------------------------
#  Builds the user interface for a specified Group within a specified Panel:
#-------------------------------------------------------------------------------
    
def fill_panel_for_group ( panel, group, ui ):
    """ Builds the user interface for a specified Group within a specified 
        Panel.
    """
    fp = FillPanel( panel, group, ui )
    return ( fp.sizer, fp.resizable )
    
#-------------------------------------------------------------------------------
#  'FillPanel' class:
#-------------------------------------------------------------------------------
    
class FillPanel ( object ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, panel, group, ui ):
        """ Initializes the object.
        """
        self.panel = panel
        self.group = group
        self.ui    = ui
        
        # Determine the horizontal/vertical orientation of the group:
        self.is_horizontal = (group.orientation == 'horizontal')
        if self.is_horizontal:
            orientation = wx.HORIZONTAL
        else:
            orientation = wx.VERTICAL
            
        # Set up a group with or without a border around its contents:
        if group.show_border:
            self.sizer = wx.StaticBoxSizer( wx.StaticBox( panel, -1,
                                            group.label or '' ), orientation )
        else:
            self.sizer = wx.BoxSizer( orientation )
            
        # If no sizer has been specified for the panel yet, make the new sizer 
        # the layout sizer for the panel:        
        if panel.GetSizer() is None:
            panel.SetAutoLayout( True )
            panel.SetSizer( self.sizer )
        
        # Get the contents of the group:
        content = group.get_content()
        
        # Assume our contents are not resizable:
        self.resizable = False
        
        if len( content ) > 0:
            # Check if content is all Group objects:
            if isinstance( content[0], Group ):
                # If so, add them to the panel and exit:
                self.add_groups( content )
            else:
                # Otherwise, the content is a list of Item objects...
                self.add_items( content )

    #---------------------------------------------------------------------------
    #  Adds a list of Group objects to the panel:
    #---------------------------------------------------------------------------
        
    def add_groups ( self, content ):
        """ Adds a list of Group objects to the panel.
        """
        sizer = self.sizer
        
        # Process each group:
        for subgroup in content:
            # Add the sub-group to the panel:
            sg_sizer, sg_resizable = fill_panel_for_group( self.panel, subgroup,
                                                           self.ui )
            
            # If the sub-group is resizable:
            if sg_resizable:
                
                # Then so are we:
                self.resizable = True
                
                # Add the sub-group so that it can be resized by the layout:
                sizer.Add( sg_sizer, 1, wx.EXPAND | wx.ALL, 2 )
                
            # For horizontal layout, or a group with no border:
            elif self.is_horizontal or (not subgroup.show_border):
                
                # Do not allow the sub-group to be resized at all:
                sizer.Add( sg_sizer, 0, wx.ALL, 2 )
            else:
                # Otherwise, allow it to be resized horizontally to allow the
                # group box borders to line up neatly:
                sizer.Add( sg_sizer, 0, wx.EXPAND | wx.ALL, 2 )
        
    #---------------------------------------------------------------------------
    #  Adds a list of Item objects to the panel:
    #---------------------------------------------------------------------------
        
    def add_items ( self, content ):
        """ Adds a list of Item objects to the panel.
        """
        # Get local references to various objects we need:
        panel   = self.panel
        sizer   = self.sizer
        ui      = self.ui
        info    = ui.info
        handler = ui.handler
        
        show_labels      = self.group.show_labels
        show_left        = self.group.show_left
        row              = -1
        self.label_flags = 0
        if (not self.is_horizontal) and show_labels:
            # For a vertical list of Items with labels, use a FlexGridSizer:
            self.label_pad = 0
            cols           = 2
            flags          = 0
            border_size    = 0
            item_sizer     = wx.FlexGridSizer( 0, 2, 5, 5 )
            if show_left:
                self.label_flags = wx.ALIGN_RIGHT
                item_sizer.AddGrowableCol( 1 )
        else:
            # Otherwise, the current sizer will work as is:
            self.label_pad   = 3
            cols             = 1
            flags            = wx.ALL
            border_size      = 4
            item_sizer       = sizer
            
        # Process each Item in the list:
        for item in content:
            
            # Get the name in order to determine its type:
            name = item.name or ''
            
            # Check if is a label:
            if name == '':
                label = item.label
                if label is not None:
                    # Indicate a row is added to the sizer:
                    row += 1
                    
                    # Add the label to the sizer:
                    item_sizer.Add( wx.StaticText( panel, -1, label ), 0, 
                                    wx.ALIGN_CENTER )
                                    
                    # If we are building a two-column layout, just add space in
                    # the second column:
                    if cols > 1:
                        item_sizer.Add( 1, 1 )
                    
                # Continue on to the next Item in the list:
                continue
            
            # Indicate a row is added to the sizer:
            row += 1
            
            # Check if it is a separator:
            if name == '_':
                for i in range( cols ):
                    if self.is_horizontal:
                        # Add a vertical separator:
                        item_sizer.Add( wx.StaticLine( panel, -1, 
                                                       style = wx.LI_VERTICAL ), 
                                        0, wx.LEFT | wx.RIGHT | wx.EXPAND, 2 )
                    else:
                        # Add a horizontal separator:
                        item_sizer.Add( wx.StaticLine( panel, -1,
                                                  style = wx.LI_HORIZONTAL ), 
                                        0, wx.TOP | wx.BOTTOM | wx.EXPAND, 2 )
                # Continue on to the next Item in the list:
                continue
               
            # Convert a blank to a 5 pixel spacer:
            if name == ' ':
                name = '5'
               
            # Check if it is a spacer:
            if all_digits.match( name ):
                
                # If so, add the appropriate amount of space to the sizer:
                n = int( name )
                for i in range( cols ):
                    item_sizer.Add( ( n, n ) )
                    
                # Continue on to the next Item in the list:
                continue
               
            # Otherwise, it must be a trait Item:
            object = ui.context[ item.object ]
            trait  = object.base_trait( name )
            desc   = trait.desc or ''
            
            # If we are displaying labels on the left, add the label to the 
            # user interface:
            if show_labels and show_left:
                self.create_label( item.get_label( ui ), desc, panel, 
                                   item_sizer, trait )
                           
            # Get the editor factory associated with the Item:                          
            editor_factory = item.editor
            if editor_factory is None:
                editor_factory = trait.get_editor()

                # If the item has formatting traits set them
                # in the editor_factory.
                if item.format_func is not None:
                    editor_factory.format_func = item.format_func
                if item.format_str != '':
                    editor_factory.format_str = item.format_str
                
            # Create the requested type of editor from the editor factory: 
            factory_method = getattr( editor_factory, item.style + '_editor' )
            editor         = factory_method( ui, object, name, desc, panel )
                
            # Bind the editor into the UIInfo object name space so it can be 
            # referred to by a Handler while the user interface is active:
            id = item.id or name
            info.bind( id, editor )
            
            # Also, add the editors to the list of editors used to construct 
            # the user interface:
            ui._editors.append( editor )
            
            # If the handler wants to be notified when the editor is defined, 
            # add it to the list of methods to be called when the UI is 
            # complete:
            created = getattr( handler, id + '_defined', None )
            if created is not None:
                ui.add_created( created )
            
            # If the editor is conditionally defined, add the defining 
            # 'expression' and the editor to the UI object's list of monitored 
            # objects: 
            if item.defined_when:
                ui.add_defined( item.defined_when, editor )
            
            # If the editor is conditionally enabled, add the enabling 
            # 'expression' and the editor to the UI object's list of monitored 
            # objects: 
            if item.enabled_when:
                ui.add_enabled( item.enabled_when, editor )
            
            # Add the created editor control to the sizer with the appropriate
            # layout flags and values:
            growable = 0
            if item.resizable:
                growable       = 1
                self.resizable = True
            item_sizer.Add( editor.control, growable, 
                            flags | editor.layout_style, border_size )
            
            # If we are displaying labels on the right, add the label to the 
            # user interface:
            if show_labels and (not show_left):
                self.create_label( item.get_label( ui ), desc, panel, 
                                   item_sizer, trait, '' )
                            
            # If the Item is resizable, and we are using a two-column grid:                        
            if item.resizable and (cols == 2):
                # Mark the new row as growable:
                item_sizer.AddGrowableRow( row )
                
        # If we created a grid sizer, add it to the original sizer:
        if item_sizer is not sizer:
            growable = 0
            if self.resizable:
                growable = 1
            sizer.Add( item_sizer, growable, wx.EXPAND | wx.ALL, 4 )

    #---------------------------------------------------------------------------
    #  Creates an item label:
    #---------------------------------------------------------------------------
        
    def create_label ( self, label, desc, parent, sizer, trait, suffix = ':' ):    
        """ Creates an item label.
        """
        if (label == '') or (label[-1:] == '?'):
            suffix = ''
        control = wx.StaticText( parent, -1, label + suffix,
                                 style = wx.ALIGN_RIGHT )
        wx.EVT_LEFT_UP( control, show_help_popup )
        control.trait = trait
        sizer.Add( control, 0, self.label_flags | wx.ALIGN_CENTER_VERTICAL | 
                               wx.LEFT, self.label_pad )
        if desc != '':
            control.SetToolTipString( 'Specifies ' + desc )

#-------------------------------------------------------------------------------
#  'HTMLHelpWindow' class:
#-------------------------------------------------------------------------------
            
class HTMLHelpWindow ( wx.Frame ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, parent, html, scale_dx, scale_dy ):
        """ Initializes the object.
        """
        wx.Frame.__init__( self, parent, -1, 'Help' )
        self.SetBackgroundColour( WindowColor )
        
        # Wrap the dialog around the image button panel:
        sizer        = wx.BoxSizer( wx.VERTICAL )
        html_control = wh.HtmlWindow( self )
        html_control.SetBorders( 2 )
        html_control.SetPage( html )
        sizer.Add( html_control, 1, wx.EXPAND )
        sizer.Add( wx.StaticLine( self, -1 ), 0, wx.EXPAND )
        b_sizer = wx.BoxSizer( wx.HORIZONTAL )
        button  = wx.Button( self, -1, 'OK' )
        wx.EVT_BUTTON( self, button.GetId(), self._on_ok )
        b_sizer.Add( button, 0 )
        sizer.Add( b_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5 )
        self.SetSizer( sizer )
        self.SetAutoLayout( True )
        self.SetSize( wx.Size( int( scale_dx * screen_dx ), 
                               int( scale_dy * screen_dy ) ) )
 
        # Position and show the dialog:
        position_near( parent, self, align_y = -1 )
        self.Show()
        
    #---------------------------------------------------------------------------
    #  Handles the window being closed:
    #---------------------------------------------------------------------------
        
    def _on_ok ( self, event ):
        """ Handles the window being closed.
        """
        self.Destroy()
        

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
# Date: 12/04/2004
# Description: Test case for the traits tree editor.
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasTraits, Str, Regex, List, Instance
    
from enthought.traits.ui.api \
    import TreeEditor, TreeNode, View, Group, Item, Handler, InstanceEditor
    
from enthought.traits.ui.instance_choice \
    import InstanceDropChoice
                      
from enthought.traits.ui.menu \
    import Menu, Action, Separator
    
from enthought.traits.ui.wx.tree_editor \
    import NewAction, CopyAction, CutAction, PasteAction, DeleteAction, \
           RenameAction

#-------------------------------------------------------------------------------
#  'Employee' class:
#-------------------------------------------------------------------------------

class Employee ( HasTraits ):
    name  = Str( '<unknown>' )
    title = Str
    phone = Regex( regex = r'\d\d\d-\d\d\d\d' )
    
    view = View( 'title', 'phone' )
    
    def default_title ( self ):
        self.title = 'Senior Engineer'
        
#-------------------------------------------------------------------------------
#  'Department' class:
#-------------------------------------------------------------------------------

class Department ( HasTraits ):
    name      = Str( '<unknown>' )
    employees = List( Employee )
    
    view = View( [ 'employees', '|<>' ] )

#-------------------------------------------------------------------------------
#  'Company' class:
#-------------------------------------------------------------------------------

class Company ( HasTraits ):
    name        = Str( '<unknown>' )
    departments = List( Department )
    employees   = List( Employee )
    
#-------------------------------------------------------------------------------
#  'Partner' class:  
#-------------------------------------------------------------------------------
        
class Partner ( HasTraits ):
    name    = Str( '<unknown>' )
    company = Instance( Company )
    eom     = Instance( Employee )
    dom     = Instance( Department )

#-------------------------------------------------------------------------------
#  Create a hierarchy:
#-------------------------------------------------------------------------------

jason = Employee( 
     name  = 'Jason',
     title = 'Sr. Engineer', 
     phone = '536-1057' )
     
mike = Employee( 
     name  = 'Mike',
     title = 'Sr. Engineer', 
     phone = '536-1057' )
     
dave = Employee(
     name  = 'Dave',
     title = 'Sr. Engineer',
     phone = '536-1057' )
     
martin = Employee(
     name  = 'Martin',
     title = 'Sr. Engineer',
     phone = '536-1057' )
     
duncan = Employee(
     name  = 'Duncan',
     title = 'Sr. Engineer' )
        
partner = Partner(
    name    = 'eric',
    company = Company( 
        name        = 'Enthought, Inc.',
        departments = [
            Department( 
                name      = 'Business',
                employees = [ jason, mike ]
            ),
            Department(
                name      = 'Scientific',
                employees = [ dave, martin, duncan ] 
            )
        ],
        employees = [ dave, martin, mike, duncan, jason ]
    )
)

#-------------------------------------------------------------------------------
#  Define the tree trait editor:
#-------------------------------------------------------------------------------

no_view = View()

tree_editor = TreeEditor( 
    editable = False,
    nodes = [
        TreeNode( node_for  = [ Company ],
                  auto_open = True,
                  children  = '',
                  label     = 'name',
                  view      = View( [ 'name', '|<' ] ) ),
        TreeNode( node_for  = [ Company ],
                  auto_open = True,
                  children  = 'departments',
                  label     = '=Departments',
                  view      = no_view,
                  add       = [ Department ] ),
        TreeNode( node_for  = [ Company ],
                  auto_open = True,
                  children  = 'employees',
                  label     = '=Employees',
                  view      = no_view,
                  add       = [ Employee ] ),
        TreeNode( node_for  = [ Department ],
                  auto_open = True,
                  children  = 'employees',
                  label     = 'name',
                  menu      = Menu( NewAction,
                                    Separator(),
                                    DeleteAction,
                                    Separator(),
                                    RenameAction,
                                    Separator(),
                                    CopyAction, 
                                    CutAction, 
                                    PasteAction ),
                  view      = View( [ 'name', '|<' ] ),
                  add       = [ Employee ] ),
        TreeNode( node_for  = [ Employee ],
                  auto_open = True,
                  label     = 'name',
                  menu      = Menu( NewAction,
                                    Separator(),
                                    Action( name   = 'Default title',
                                            action = 'object.default_title' ),
                                    Action( name   = 'Department',
                                            action = 'handler.employee_department(editor,object)' ),
                                    Separator(),
                                    CopyAction, 
                                    CutAction, 
                                    PasteAction,
                                    Separator(),
                                    DeleteAction,
                                    Separator(),
                                    RenameAction ),
                  view      = View( [ 'name', 'title', 'phone', '|<' ] ) )
    ]
)

#-------------------------------------------------------------------------------
#  'TreeHandler' class:  
#-------------------------------------------------------------------------------

class TreeHandler ( Handler ):
    
    def employee_department ( self, editor, object ):
        dept = editor.get_parent( object )
        print '%s works in the %s department.' % ( object.name, dept.name )
 
#-------------------------------------------------------------------------------
#  Define the View to use:
#-------------------------------------------------------------------------------

view = View(
    Group(
        [ Item( 'company', 
                editor    = tree_editor, 
                resizable = True ),
          '|<>' ],
        Group( 
            [ '{Employee of the Month}@',
              Item( 'eom@', 
                    editor = InstanceEditor( values = [ 
                                 InstanceDropChoice( klass      = Employee,
                                                     selectable = True ) ] ),
                    resizable = True ),
              '|<>' ],
            [ '{Department of the Month}@',
              Item( 'dom@', 
                    editor = InstanceEditor( values = [ 
                                 InstanceDropChoice( klass = Department ) ] ),
                    resizable = True ),
              '|<>' ],
            show_labels = False,
            layout      = 'split' ),
        orientation = 'horizontal',
        show_labels = False,
        layout      = 'split' ),
    title     = 'Company Structure',
    handler   = TreeHandler(),
    buttons   = [ 'OK', 'Cancel' ],
    resizable = True,
    width     = .5,
    height    = .5
)
             
#-------------------------------------------------------------------------------
#  Edit it:  
#-------------------------------------------------------------------------------
 
if __name__ == '__main__':
    partner.configure_traits( view = view )


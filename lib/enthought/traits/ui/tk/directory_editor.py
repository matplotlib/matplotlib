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
# Description: Define the Tkinter implementation of the various directory
#              editors and the directory editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from file_editor import ToolkitEditorFactory as EditorFactory
from file_editor import SimpleEditor         as SimpleFileEditor

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description ) 
                                      
#-------------------------------------------------------------------------------
#  'SimpleEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleEditor ( SimpleFileEditor ):
           
    #---------------------------------------------------------------------------
    #  Creates the correct type of file dialog:
    #---------------------------------------------------------------------------
           
    def create_file_dialog ( self ):
        """ Creates the correct type of file dialog.
        """
        dlg =  wx.DirDialog( self.control, message = 'Select a directory' )
        dlg.SetPath( self._filename.GetValue() )
        return dlg
        

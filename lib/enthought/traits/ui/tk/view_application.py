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
# Description: Creates a Tkinter specific modal dialog user interface that runs
#              as a complete application using information from the specified UI
#              object.
#  
#  Symbols defined: view_application
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

def view_application ( context, view, kind ):
    if (kind == 'panel') or ((kind is None) and (view.kind == 'panel')):
        kind = 'modal'
    return ViewApplication( context, view, kind ).ui.result
    
#-------------------------------------------------------------------------------
#  'ViewApplication' class:
#-------------------------------------------------------------------------------

class ViewApplication ( wx.App ):
    
   #----------------------------------------------------------------------------
   #  Initializes the object:
   #----------------------------------------------------------------------------
   
   def __init__ ( self, context, view, kind ):
       """ Initializes the object.
       """
       self.context = context
       self.view    = view
       self.kind    = kind
       wx.InitAllImageHandlers()
       wx.App.__init__( self, 1, 'debug.log' )
       self.MainLoop()
   
   #----------------------------------------------------------------------------
   #  Handles application initialization:
   #----------------------------------------------------------------------------

   def OnInit ( self ):
       self.ui = self.view.ui( self.context, kind = self.kind )
       return True


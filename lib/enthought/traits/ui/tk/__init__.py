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
# Description: Define the concrete implementations of the traits Toolkit
#              interface for the Tkinter user interface toolkit.
#
#  Symbols defined: toolkit
#                   TkColor
#                   TkClearColor
#                   RGBColor
#                   RGBClearColor
#                   TkFont
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Define the reference to the exported GUIToolkit object:
#-------------------------------------------------------------------------------
    
import toolkit
toolkit = toolkit.GUIToolkit()

from color_editor     import TkColor, TkClearColor
from rgb_color_editor import RGBColor, RGBClearColor
from font_editor      import TkFont     
       

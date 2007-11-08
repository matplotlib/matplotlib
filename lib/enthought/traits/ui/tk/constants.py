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
# Description: Define constants used by the Tkinter implementation of the
#              various text editors and text editor factories.
#
#  Symbols defined: OKColor
#                   ErrorColor
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import wx

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Color used to mark valid input:
OKColor = '#FFFFFF'

# Color used to highlight input errors:
ErrorColor = '#FFC0C0'

# Color used for background of 'read-only' fields:
ReadonlyColor = '#ECE9D8'

# Color used for background of windows (like dialog background color):
WindowColor = '#ECE9D8'

# Standard width of an image bitmap:
standard_bitmap_width = 120

# Width of a scrollbar:
scrollbar_dx = wx.SystemSettings_GetMetric( wx.SYS_VSCROLL_X )    

# Screen size:
screen_dx = wx.SystemSettings_GetMetric( wx.SYS_SCREEN_X )
screen_dy = wx.SystemSettings_GetMetric( wx.SYS_SCREEN_Y )



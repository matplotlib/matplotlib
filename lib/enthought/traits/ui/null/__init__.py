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
# Date: 02/14/2005
#
#  Symbols defined: toolkit
#
#------------------------------------------------------------------------------
""" Define the concrete implementations of the traits Toolkit interface for the
'null' (do nothing) user interface toolkit. This toolkit is provided to handle
situations where no recognized traits-compatible UI toolkit is installed, but
users still want to use traits for non-UI related tasks.
"""
#-------------------------------------------------------------------------------
#  Define the reference to the exported GUIToolkit object:
#-------------------------------------------------------------------------------
    
import toolkit
toolkit = toolkit.GUIToolkit()
       

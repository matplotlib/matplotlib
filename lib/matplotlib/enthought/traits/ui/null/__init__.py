#-------------------------------------------------------------------------------
#
#  Define the concrete implementations of the traits Toolkit interface for the 
#  'null' (do nothing) user interface toolkit. This toolkit is provided to
#  handle situations where no recognized traits-compatible UI toolkit is
#  installed, but users still want to use traits for non-UI related tasks.
#
#  Written by: David C. Morrill
#
#  Date: 02/14/2005
#
#  Symbols defined: toolkit
#
#  (c) Copyright 2005 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Define the reference to the exported GUIToolkit object:
#-------------------------------------------------------------------------------
    
import toolkit
toolkit = toolkit.GUIToolkit()
       

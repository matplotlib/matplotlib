#-------------------------------------------------------------------------------
#
#  Define various helper functions that are useful for creating traits based
#  user interfaces.
#
#  Written by: David C. Morrill
#
#  Date: 10/25/2004
#
#  Symbols defined: user_name_for
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from string import uppercase
    
#----------------------------------------------------------------------------
#  Return a 'user-friendly' name for a specified trait:
#----------------------------------------------------------------------------

def user_name_for ( name ):
    name       = name.replace( '_', ' ' ).capitalize()
    result     = ''
    last_lower = 0
    for c in name:
        if (c in uppercase) and last_lower:
           result += ' '
        last_lower = (c in lowercase)
        result    += c
    return result


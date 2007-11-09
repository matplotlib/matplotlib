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
# Date: 06/21/2002
# ------------------------------------------------------------------------------
""" Defines the standard exceptions raised by the Traits package.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import exceptions

from trait_base \
    import class_of

#-------------------------------------------------------------------------------
#  'TraitError' class:
#-------------------------------------------------------------------------------

class TraitError ( exceptions.Exception ):

    def __init__ ( self, args = None, name = None, info = None, value = None ):
        if name is None:
           # If the given args is not a tuple then assume that the user intended
           # it to be the single item in a one-element tuple.
           if not isinstance(args, tuple):
               args = args,
           self.args = args
        else:
           # Save the information, in case the 'args' object is not the correct
           # one, and we need to regenerate the message later:
           self.name   = name
           self.info   = info
           self.value  = value
           self.desc   = None
           self.prefix = 'The'
           self.set_desc( None, args )
 
    def set_desc ( self, desc, object = None ):
        if hasattr( self, 'desc' ):
           if desc is not None:
              self.desc = desc
           if object is not None:
              self.object = object
           self.set_args()
           
    def set_prefix ( self, prefix ):
        if hasattr( self, 'prefix' ):
           self.prefix = prefix
           self.set_args()
           
    def set_args ( self ):
        if self.desc is None:
           extra = ''
        else:
           extra = ' specifies %s and' % self.desc
        obj = getattr( self, 'object', None )

        # Note: self.args must be a tuple so be sure to leave the trailing
        # commas.
        if obj is not None:
            self.args = ( "%s '%s' trait of %s instance%s must be %s, "
                          "but a value of %s was specified." % (
                          self.prefix, self.name, class_of( obj ), extra,
                          self.info, self.value ) ),
        else:
            self.args = ( "%s '%s' trait%s must be %s, but a value of %s was "
                          "specified." % ( self.prefix, self.name, extra, 
                                           self.info, self.value ) ),
   
#-------------------------------------------------------------------------------
#  'TraitNotificationError' class:
#-------------------------------------------------------------------------------

class TraitNotificationError ( exceptions.Exception ):

    pass
   
#-------------------------------------------------------------------------------
#  'DelegationError' class:
#-------------------------------------------------------------------------------

class DelegationError ( TraitError ):

    def __init__ ( self, args ):
        self.args = args
       
#-------------------------------------------------------------------------------
#  Export the defined exceptions to the C-base traits module:
#-------------------------------------------------------------------------------
       
import ctraits
ctraits._exceptions( TraitError, DelegationError )

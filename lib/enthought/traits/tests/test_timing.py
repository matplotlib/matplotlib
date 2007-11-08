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
# Date: 03/03/2003
# Description: Perform timing tests on various trait styles to determine the
#              amount of overhead that traits add.
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from time             import time
from enthought.traits.api import *

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Number of iterations to perform:
n = 1000000

# Loop overhead time (actual value determined first time a measurement is made):
t0 = -1.0

#-------------------------------------------------------------------------------
#  Measure how long it takes to execute a specified function: 
#-------------------------------------------------------------------------------

def measure ( func ):
    now = time()
    func()
    return time() - now

#-------------------------------------------------------------------------------
#  'Old style' Python attribute get/set:
#-------------------------------------------------------------------------------
    
class old_style_value:
       
   def measure ( self ):
       global t0
       self.init()
       
       if t0 < 0.0:
          t0 = measure( self.null )
       t1    = measure( self.do_get )
       t2    = measure( self.do_set )
       scale = 1.0e6 / n
       print self.__class__.__name__ + ':'
       print '  get: %.2f usec' % (max( t1 - t0, 0.0 ) * scale)
       print '  set: %.2f usec' % (max( t2 - t0, 0.0 ) * scale)
       print
       
   def null ( self ):
       for i in range(n):
           pass

   def init ( self ):
       self.value = -1

   def do_set ( self ):
       for i in range(n):
           self.value = i
   
   def do_get ( self ):
       for i in range(n):
           self.value   

#-------------------------------------------------------------------------------
#  'New style' Python attribute get/set:
#-------------------------------------------------------------------------------
    
class new_style_value ( object ):
       
   def measure ( self ):
       global t0
       self.init()
       
       if t0 < 0.0:
          t0 = measure( self.null )
       t1    = measure( self.do_get )
       t2    = measure( self.do_set )
       scale = 1.0e6 / n
       print self.__class__.__name__ + ':'
       print '  get: %.2f usec' % (max( t1 - t0, 0.0 ) * scale)
       print '  set: %.2f usec' % (max( t2 - t0, 0.0 ) * scale)
       print
       
   def null ( self ):
       for i in range(n):
           pass

   def init ( self ):
       self.value = -1

   def do_set ( self ):
       for i in range(n):
           self.value = i
   
   def do_get ( self ):
       for i in range(n):
           self.value   

#-------------------------------------------------------------------------------
#  Python 'property' get/set:
#-------------------------------------------------------------------------------
    
class property_value ( new_style_value ):
    
   def get_value ( self ):
       return self._value

   def set_value ( self, value ):
       self._value = value
   
   value = property( get_value, set_value )

#-------------------------------------------------------------------------------
#  Python 'global' get/set:
#-------------------------------------------------------------------------------
    
class global_value ( new_style_value ):

   def init ( self ):
       global gvalue
       gvalue = -1

   def do_set ( self ):
       global gvalue
       for i in range(n):
           gvalue = i
   
   def do_get ( self ):
       global gvalue
       for i in range(n):
           gvalue   
 
#-------------------------------------------------------------------------------
#  Trait that can have any value:
#-------------------------------------------------------------------------------
  
class any_value ( HasTraits, new_style_value ):
    
   value = Any
   
#-------------------------------------------------------------------------------
#  Trait that can only have 'float' values: 
#-------------------------------------------------------------------------------
   
class int_value ( any_value ):
    
   value = Int
   
#-------------------------------------------------------------------------------
#  Trait that can only have 'range' values: 
#-------------------------------------------------------------------------------
   
class range_value ( any_value ):
    
   value = Range( -1, 2000000000 )
       
#-------------------------------------------------------------------------------
#  Executes method when float trait is changed:
#-------------------------------------------------------------------------------
       
class change_value ( int_value ):
    
   def _value_changed ( self, old, new ):
       pass

#-------------------------------------------------------------------------------
#  Notifies handler when float trait is changed:
#-------------------------------------------------------------------------------

class monitor_value ( int_value ):
    
   def init ( self ):
       self.on_trait_change( self.on_value_change, 'value' )

   def on_value_change ( self, object, trait_name, old, new ):
       pass
   
#-------------------------------------------------------------------------------
#  Float trait is delegated to another object:
#-------------------------------------------------------------------------------
   
class delegate_value ( HasTraits, new_style_value ):
    
   value    = Delegate( 'delegate' )
   delegate = Any
   
   def init ( self ):
       self.delegate = int_value()
   
#-------------------------------------------------------------------------------
#  Float trait is delegated through one object to another object:
#-------------------------------------------------------------------------------
   
class delegate_2_value ( delegate_value ):
    
   def init ( self ):
       self.delegate = delegate_value()
       self.delegate.init()
         
#-------------------------------------------------------------------------------
#  Float trait is delegated through two objects to another object:
#-------------------------------------------------------------------------------
         
class delegate_3_value ( delegate_value ):
    
   def init ( self ):
       self.delegate = delegate_2_value()
       self.delegate.init()
       
#-------------------------------------------------------------------------------
#  Run the timing measurements:
#-------------------------------------------------------------------------------

if __name__ == '__main__':
   old_style_value().measure()
   new_style_value().measure()
   property_value().measure()
   global_value().measure()
   any_value().measure()
   int_value().measure()
   range_value().measure()
   change_value().measure()
   monitor_value().measure()
   delegate_value().measure()
   delegate_2_value().measure()
   delegate_3_value().measure()

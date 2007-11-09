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
# Description: <Traits component>
#------------------------------------------------------------------------------
import unittest

from enthought.traits.api import HasTraits, Category, Str, TraitError

class Base( HasTraits ) :
    y = Str("Base y")
    z = Str("Base z")
    
class BaseExtra( Category, Base ) :
    x = Str("BaseExtra x")

class BasePlus( Category, Base ) :
    p = Str("BasePlus p")
#   z = Str("BasePlus z")    overrides not allowed.
    
class BasePlusPlus( BasePlus ) :
    pp = Str("BasePlusPlus pp")
    
class CategoryTestCase( unittest.TestCase ) :
    """ Test cases for traits category """
    
    def setUp( self ) :
        self.base = Base()
        return
        
    def test_base_category(self) :
        """ Base class with traits """
        self.assertEqual( self.base.y, "Base y", msg="y != 'Base y'" )
        self.assertEqual( self.base.z, "Base z", msg="z != 'Base z'" )
        return
    
    def test_extra_extension_category(self) :
        """ Base class extended with a category subclass """
        self.assertEqual( self.base.x, "BaseExtra x", msg="x != 'BaseExtra x'" )
        return
        
    def test_plus_extension_category(self) :
        """ Base class extended with two category subclasses """
        self.assertEqual( self.base.x, "BaseExtra x", msg="x != 'BaseExtra x'" )
        self.assertEqual( self.base.p, "BasePlus p", msg="p != 'BasePlus p'" )
        return
    
    def test_subclass_extension_category(self) :
        """ Category subclass does not extend base class.
        This test demonstrates that traits allows subclassing of a category
        class, but that the traits from the subclass are not actually added
        to the base class of the Category.
        Seems like the declaration of the subclass (BasePlusPlus) should fail.
        """
        try :
            x = self.base.pp
            self.fail( msg="base.pp should have thrown AttributeError "
                        "as Category subclassing is not supported." )
        except AttributeError :
            pass
        
        basepp = BasePlusPlus()
        return

    def test_subclass_instance_category(self) :
        """ Category subclass instantiation not supportted.
        This test demonstrates that traits allows subclassing of a category
        class, that subclass can be instantiated, but the traits of the parent
        class are not inherited.
        Seems like the declaration of the subclass (BasePlusPlus) should fail.
        """
        bpp = BasePlusPlus()
        self.assertEqual( bpp.pp, "BasePlusPlus pp", 
                        msg="pp != 'BasePlusPlus pp'" )
        
        try :
            self.assertEqual( bpp.p, "BasePlus p", msg="p != 'BasePlus p'" )
            self.fail( msg="bpp.p should have thrown SystemError as "
                "instantiating a subclass of a category is not supported." )
        except SystemError :
            pass
        return

#
# support running this test individually, from the command-line as a script
#
if __name__ == '__main__':
    unittest.main()
    
#### EOF ######################################################################

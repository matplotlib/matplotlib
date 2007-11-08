#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
import unittest

from enthought.traits.api import Delegate, HasTraits, Instance, Str, Any

# global because event handlers are being called with wrong value for self
baz_s_handler_self = None
baz_sd_handler_self = None
foo_s_handler_self = None
foo_t_handler_self = None

class Foo( HasTraits ):
    s = Str( 'foo' )
    t = Str( 'foo.t')
    
    def _s_changed(self, name, old, new):
        print 'Foo._s_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global foo_s_handler_self
        foo_s_handler_self = self
        return

    def _t_changed(self, name, old, new):
        print 'Foo._t_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global foo_t_handler_self
        foo_t_handler_self = self
        return

class Bar( HasTraits ):
    foo = Instance( Foo, () )
    s = Delegate( 'foo' )

class BazModify( HasTraits ):
    foo = Instance( Foo, () )
    sd = Delegate( 'foo', prefix='s', modify=True )
    t = Delegate( 'foo', modify=True )
    
    def _s_changed(self, name, old, new):
        # should never be called
        print 'BazModify._s_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global baz_s_handler_self
        baz_s_handler_self = self
        return
    
    def _sd_changed(self, name, old, new):
        print 'BazModify._sd_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global baz_sd_handler_self
        baz_sd_handler_self = self
        return

    def _t_changed(self, name, old, new):
        print 'BazModify._t_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global baz_t_handler_self
        baz_t_handler_self = self
        return
    

class BazNoModify( HasTraits ):
    foo = Instance( Foo, () )
    sd = Delegate( 'foo', prefix='s' )
    t = Delegate( 'foo' )
    
    def _s_changed(self, name, old, new):
        print 'BazNoModify._s_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global baz_s_handler_self
        baz_s_handler_self = self
        return
    
    def _sd_changed(self, name, old, new):
        print 'BazNoModify._sd_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global baz_sd_handler_self
        baz_sd_handler_self = self
        return

    def _t_changed(self, name, old, new):
        print 'BazNoModify._t_changed( %s, %s, %s, %s)' % (self, name, old, new)
        global baz_t_handler_self
        baz_t_handler_self = self
        return

class DelegateTestCase( unittest.TestCase ):
    """ Test cases for delegated traits. """


    def test_reset(self):
        """ Test that a delegated trait may be reset.

        Deleting the attribute should reset the trait back to its initial
        delegation behavior.
        """
        
        f = Foo()
        b = Bar(foo=f)

        # Check initial delegation.
        self.assertEqual( f.s, b.s )

        # Check that an override works.
        b.s = 'bar'
        self.assertNotEqual( f.s, b.s )

        # Check that we can reset back to delegation.  This is what we are
        # really testing for.
        del b.s
        self.assertEqual( f.s, b.s )

        return


    # Below are 8 tests to check the calling of change notification handlers.
    # There are 8 cases for the 2x2x2 matrix with axes:
    # Delegate with prefix or not
    # Delegate with modify write through or not
    # Handler in the delegator and delegatee
    #
    def test_modify_prefix_handler_on_delegator(self):
        f = Foo()
        b = BazModify(foo=f)
        
        self.assertEqual( f.s, b.sd )

        global baz_s_handler_self        
        global baz_sd_handler_self        
        baz_sd_handler_self = None
        baz_s_handler_self = None
        
        b.sd = 'changed'
        self.assertEqual( f.s, b.sd )
        
        # Don't expect _s_changed to be called because from Baz's perspective
        # the triat is named 'sd'
        self.assertEqual( baz_s_handler_self, None )
        
        # Do expect '_sd_changed' to be called with b as self
        self.assertEqual( baz_sd_handler_self, b )

        return
    
    def test_modify_prefix_handler_on_delegatee(self):
        f = Foo()
        b = BazModify(foo=f)
        
        self.assertEqual( f.s, b.sd )
        
        global foo_s_handler_self
        foo_s_handler_self = None
        
        b.sd = 'changed'
        self.assertEqual( f.s, b.sd )
        
        # Foo expects its '_s_changed' handler to be called with f as self
        self.assertEqual( foo_s_handler_self, f )
        
        return



    def test_no_modify_prefix_handler_on_delegator(self):
        f = Foo()
        b = BazNoModify(foo=f)
        
        self.assertEqual( f.s, b.sd )

        global baz_s_handler_self        
        global baz_sd_handler_self        
        baz_sd_handler_self = None
        baz_s_handler_self = None
        
        b.sd = 'changed'
        self.assertNotEqual( f.s, b.sd )
        
        # Don't expect _s_changed to be called because from Baz's perspective
        # the triat is named 'sd'
        self.assertEqual( baz_s_handler_self, None )
        
        # Do expect '_sd_changed' to be called with b as self
        self.assertEqual( baz_sd_handler_self, b )

        return
    
    def test_no_modify_prefix_handler_on_delegatee_not_called(self):
        f = Foo()
        b = BazNoModify(foo=f)
        
        self.assertEqual( f.s, b.sd )
        
        global foo_s_handler_self
        foo_s_handler_self = None
        
        b.sd = 'changed'
        self.assertNotEqual( f.s, b.sd )
        
        # Foo expects its '_s_changed' handler to be called with f as self
        self.assertEqual( foo_s_handler_self, None )
        
        return




    def test_modify_handler_on_delegator(self):
        f = Foo()
        b = BazModify(foo=f)
        
        self.assertEqual( f.t, b.t )

        global baz_t_handler_self        
        baz_t_handler_self = None
        
        b.t = 'changed'
        self.assertEqual( f.t, b.t )
        
        # Do expect '_t_changed' to be called with b as self
        self.assertEqual( baz_t_handler_self, b )

        return
    
    def test_modify_handler_on_delegatee(self):
        f = Foo()
        b = BazModify(foo=f)
        
        self.assertEqual( f.t, b.t )
        
        global foo_t_handler_self
        foo_t_handler_self = None
        
        b.t = 'changed'
        self.assertEqual( f.t, b.t )
        
        # Foo t did change so '_t_changed' handler should be called
        self.assertEqual( foo_t_handler_self, f)
        
        return


    def test_no_modify_handler_on_delegator(self):
        f = Foo()
        b = BazNoModify(foo=f)
        
        self.assertEqual( f.t, b.t )

        global baz_t_handler_self        
        baz_t_handler_self = None
        
        b.t = 'changed'
        self.assertNotEqual( f.t, b.t )
        
        # Do expect '_t_changed' to be called with b as self
        self.assertEqual( baz_t_handler_self, b )

        return
    
    def test_no_modify_handler_on_delegatee_not_called(self):
        f = Foo()
        b = BazNoModify(foo=f)
        
        self.assertEqual( f.t, b.t )

        global foo_t_handler_self        
        foo_t_handler_self = None
        
        b.t = 'changed'
        self.assertNotEqual( f.t, b.t )
        
        # Foo t did not change so '_t_changed' handler should not be called
        self.assertEqual( foo_t_handler_self, None)
        
        return



    # Below are 4 tests for notification when the delegated trait is changed
    # directly rather than through the delegator.

    def test_no_modify_handler_on_delegatee_direct_change(self):
        f = Foo()
        b = BazNoModify(foo=f)
        
        self.assertEqual( f.t, b.t )
        
        global foo_t_handler_self
        foo_t_handler_self = None
        
        f.t = 'changed'
        self.assertEqual( f.t, b.t )
        
        # Foo t did change so '_t_changed' handler should be called
        self.assertEqual( foo_t_handler_self, f)
        
        return

    def test_no_modify_handler_on_delegator_direct_change(self):
        f = Foo()
        b = BazNoModify(foo=f)
        
        self.assertEqual( f.t, b.t )
        
        global baz_t_handler_self        
        baz_t_handler_self = None
        
        f.t = 'changed'
        self.assertEqual( f.t, b.t )
        
        # Do expect '_t_changed' to be called with b as self
        self.assertEqual( baz_t_handler_self, b )
        
        return






    def test_modify_handler_on_delegatee_direct_change(self):
        f = Foo()
        b = BazModify(foo=f)
        
        self.assertEqual( f.t, b.t )
        
        global foo_t_handler_self
        foo_t_handler_self = None
        
        f.t = 'changed'
        self.assertEqual( f.t, b.t )
        
        # Foo t did change so '_t_changed' handler should be called
        self.assertEqual( foo_t_handler_self, f)
        
        return

    def test_modify_handler_on_delegator_direct_change(self):
        f = Foo()
        b = BazModify(foo=f)
        
        self.assertEqual( f.t, b.t )
        
        global baz_t_handler_self        
        baz_t_handler_self = None
        
        f.t = 'changed'
        self.assertEqual( f.t, b.t )
        
        # Do expect '_t_changed' to be called with b as self
        self.assertEqual( baz_t_handler_self, b )
        
        return


#### EOF ######################################################################
        

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
# Author: David C. Morrill Date: 10/22/2003 Description: Unit test case for
# Traits event notification handling.
# ------------------------------------------------------------------------------

from enthought.traits.api import *

#------------------------------------------------------------------------------
# Change to True if the version of Traits under test has the defer_trait_change
# functionality:
#------------------------------------------------------------------------------

HAVE_DEFER_TRAIT_CHANGE = False


#------------------------------------------------------------------------------

class TestBase ( HasTraits ):

    __traits__ = {
        't1': 0,
        't2': 0
    }

    def test ( self ):
        print '---------- Begin %s test ----------' % self.__class__.__name__
        print 'normal changes'
        self.t1 = 1
        self.t2 = 2

        if ( HAVE_DEFER_TRAIT_CHANGE ) :
            print 'deferring trait changes'
            self.defer_trait_change()

            self.t1 = 3
            self.t2 = 4

            print 'enabling trait changes'
            self.defer_trait_change( False )

            print 'making normal changes again'

            self.t1 = 5
            self.t2 = 6

            print 'deferring trait changes'
            self.defer_trait_change()

            self.t1 = 7
            self.t2 = 8
            self.t1 = 9
            self.t2 = 10

            print 'enabling trait changes'
            self.defer_trait_change( False )

            print 'making normal changes again'

            self.t1 = 11
            self.t2 = 12

        print '---------- End %s test ----------\n' % self.__class__.__name__

#------------------------------------------------------------------------------

class Test1 ( TestBase ):

    def t1_changed ( self, old, new ):
        print 't1 changed:', old, new

    def t2_changed ( self, old, new ):
        print 't2 changed:', old, new

#------------------------------------------------------------------------------

class Test2 ( Test1 ):

    def anytrait_changed ( self, name, old, new ):
        print 'anytrait changed:', name, old, new

#------------------------------------------------------------------------------

class Test3 ( TestBase ):

    def anytrait_changed ( self, name, old, new ):
        print 'anytrait changed:', name, old, new

#------------------------------------------------------------------------------

class Test4 ( TestBase ):

    def __init__ ( self, **traits ):
        TestBase.__init__( self, **traits )
        self.on_trait_change( self.on_anytrait )

    def on_anytrait ( self, object, name, old, new ):
        print 'on anytrait changed:', name, old, new

#------------------------------------------------------------------------------

class Test5 ( TestBase ):

    def __init__ ( self, **traits ):
        TestBase.__init__( self, **traits )
        self.on_trait_change( self.t1_trait, 't1' )
        self.on_trait_change( self.t2_trait, 't2' )

    def t1_trait ( self, object, name, old, new ):
        print 'on t1 changed:', old, new

    def t2_trait ( self, object, name, old, new ):
        print 'on t2 changed:', old, new

#------------------------------------------------------------------------------

class Test6 ( Test5 ):

    def __init__ ( self, **traits ):
        Test5.__init__( self, **traits )
        self.on_trait_change( self.on_anytrait )

    def on_anytrait ( self, object, name, old, new ):
        print 'on anytrait changed:', name, old, new

#------------------------------------------------------------------------------

class Test7 ( Test1 ):

    def __init__ ( self, **traits ):
        Test1.__init__( self, **traits )
        self.on_trait_change( self.t1_trait, 't1' )
        self.on_trait_change( self.t2_trait, 't2' )

    def t1_trait ( self, object, name, old, new ):
        print 'on t1 changed:', old, new

    def t2_trait ( self, object, name, old, new ):
        print 'on t2 changed:', old, new

#------------------------------------------------------------------------------

class Test8 ( Test2 ):

    def __init__ ( self, **traits ):
        Test1.__init__( self, **traits )
        self.on_trait_change( self.t1_trait, 't1' )
        self.on_trait_change( self.t2_trait, 't2' )
        self.on_trait_change( self.on_anytrait )

    def on_anytrait ( self, object, name, old, new ):
        print 'on anytrait changed:', name, old, new

    def t1_trait ( self, object, name, old, new ):
        print 'on t1 changed:', old, new

    def t2_trait ( self, object, name, old, new ):
        print 'on t2 changed:', old, new

#------------------------------------------------------------------------------

class Test9 ( Test2 ):

    def __init__ ( self, **traits ):
        Test1.__init__( self, **traits )
        if ( HAVE_DEFER_TRAIT_CHANGE ) :
            print 'deferring trait changes in __init__'
            self.defer_trait_change()
            self.t1 = -1
            self.t2 = -2
            self.on_trait_change( self.t1_trait, 't1' )
            self.on_trait_change( self.t2_trait, 't2' )
            self.on_trait_change( self.on_anytrait )
            self.t1 = -3
            self.t2 = -4
            print 'enabling trait changes in __init__'
            self.defer_trait_change( False )

    def on_anytrait ( self, object, name, old, new ):
        print 'on anytrait changed:', name, old, new

    def t1_trait ( self, object, name, old, new ):
        print 'on t1 changed:', old, new

    def t2_trait ( self, object, name, old, new ):
        print 'on t2 changed:', old, new

#------------------------------------------------------------------------------

class Test10 ( Test2 ):

    def __init__ ( self, **traits ):
        Test1.__init__( self, **traits )
        print 'deferring trait changes in __init__'
        if ( HAVE_DEFER_TRAIT_CHANGE ) :
            self.defer_trait_change()
            self.t1 = -1
            self.t2 = -2
            self.on_trait_change( self.t1_trait, 't1' )
            self.on_trait_change( self.t2_trait, 't2' )
            self.on_trait_change( self.on_anytrait )
            self.defer_trait_change()
            self.t1 = -3
            self.t2 = -4
            self.defer_trait_change( False )
            self.on_trait_change( self.t1_trait, 't1', remove = True )
            self.on_trait_change( self.t2_trait, 't2', remove = True )
            self.on_trait_change( self.on_anytrait,    remove = True )
            print 'enabling trait changes in __init__'
            self.defer_trait_change( False )

    def on_anytrait ( self, object, name, old, new ):
        print 'on anytrait changed:', name, old, new

    def t1_trait ( self, object, name, old, new ):
        print 'on t1 changed:', old, new

    def t2_trait ( self, object, name, old, new ):
        print 'on t2 changed:', old, new

#------------------------------------------------------------------------------

test1 = Test1()
test1.test()

test2 = Test2()
test2.test()

test3 = Test3()
test3.test()

test4 = Test4()
test4.test()

test5 = Test5()
test5.test()

test6 = Test6()
test6.test()

test7 = Test7()
test7.test()

test8 = Test8()
test8.test()

if ( HAVE_DEFER_TRAIT_CHANGE ) :
    test9 = Test9()
    test9.test()

    test10 = Test10()
    test10.test()


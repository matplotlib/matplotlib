
import unittest

from enthought.traits.api import HasTraits, Int, Range, Str

class WithFloatRange(HasTraits):
    r = Range(0.0, 100.0)
    r_copied_on_change = Str
    
    _changed_handler_calls = Int
    
    def _r_changed(self, old, new):
        self._changed_handler_calls += 1
        self.r_copied_on_change = str(self.r)

        if (self.r % 10) > 0:
            self.r += 10-(self.r % 10)
        
    
class WithLargeIntRange(HasTraits):
    r = Range(0, 1000)
    r_copied_on_change = Str
    
    _changed_handler_calls = Int
    
    def _r_changed(self, old, new):
        self._changed_handler_calls += 1
        self.r_copied_on_change = str(self.r)

        if self.r > 100:
            self.r = 0
        


class RangeTestCase(unittest.TestCase):

    def test_non_ui_events(self):
        
        obj = WithFloatRange()
        obj._changed_handler_calls = 0

        obj.r = 10
        self.failUnlessEqual(1, obj._changed_handler_calls)

        obj._changed_handler_calls = 0
        obj.r = 34.56
        self.failUnlessEqual(2, obj._changed_handler_calls)
        self.failUnlessEqual(40, obj.r)

        return
    
    def test_non_ui_int_events(self):
        
        # Even thou the range is configured for 0..1000, the handler resets
        # the value to 0 when it exceeds 100.
        obj = WithLargeIntRange()
        obj._changed_handler_calls = 0

        obj.r = 10
        self.failUnlessEqual(1, obj._changed_handler_calls)
        self.failUnlessEqual(10, obj.r)

        obj.r = 100
        self.failUnlessEqual(2, obj._changed_handler_calls)
        self.failUnlessEqual(100, obj.r)

        obj.r = 101
        self.failUnlessEqual(4, obj._changed_handler_calls)
        self.failUnlessEqual(0, obj.r)

        return
    
    def ui_test_events(self):
        print
        print 'enter the value 34.56 in the range text box and tab out or enter.'
        print 'Notice that the changed handler call count is 2.'
        print 'Notice the slider is at 34.56 and the text box still shows 34.56'
        print 'Notice that r_copied_on_change shows 40.0'
        print 'Click OK to close the window.'
        print 'The test will not fail, because the range value was rounded by the event handler.'
        print 'However, the range widget did not show that change.'
        
        obj = WithFloatRange()
        obj._changed_handler_calls = 0

        obj.edit_traits(kind='livemodal', )
        
        self.failUnlessEqual( obj.r % 10, 0 )
        
        return
    
    def ui_test_int_events(self):
        print
        print 'enter the value 95 in the range text box.'
        print 'Notice that the changed handler call count is 2.'
        print 'Notice that r_copied_on_change shows 95'
        print 'Click the up arrow 5 times. Each time the handler call count will increment by one.'
        print 'The R value is now 100 and the change handler call count is 7.'
        print 'Click the up array 1 time. The call count is 11, R is 101 (wrong), and R copied on change is 0 (correct)'
        print 'Click OK to close the window.'
        print 'The test will not fail, because the range value kept below 101 by the event handler.'
        print 'However, the range widget did not show that change.'
        
        obj = WithLargeIntRange()
        obj._changed_handler_calls = 0

        obj.edit_traits(kind='livemodal', )
        
        self.failUnless( obj.r <= 100 )
        
        return

### EOF

"""
Tests for Dict and List items_changed events
"""

import unittest

from enthought.traits.api import HasTraits, Dict, List

class ListEventTestCase(unittest.TestCase):
    pass



class MyClass(HasTraits):
    """ A dummy HasTraits class with a Dict """
    d = Dict({"a": "apple", "b": "banana", "c": "cherry", "d": "durian" })

    def __init__(self, callback):
        "The callback is called with the TraitDictEvent instance"
        self.callback = callback
        return

    def _d_items_changed(self, event):
        if self.callback:
            self.callback(event)
        return

class MyOtherClass(HasTraits):
    """ A dummy HasTraits class with a Dict """
    d = Dict({"a": "apple", "b": "banana", "c": "cherry", "d": "durian" })

class Callback:
    """
    A stateful callback that gets initialized with the values to check for
    """
    def __init__(self, obj, added={}, changed={}, removed={}):
        self.obj = obj        
        self.added = added
        self.changed = changed
        self.removed = removed
        self.called = False
        return
    
    def __call__(self, event):
        if event.added != self.added:
            print "\n\n******Error\nevent.added:", event.added
        else:
            self.obj.assert_(event.added == self.added)
        self.obj.assert_(event.changed == self.changed)
        self.obj.assert_(event.removed == self.removed)
        self.called = True
        return


class DictEventTestCase(unittest.TestCase):

    def test_setitem(self):
        # overwriting an existing item
        cb = Callback(self, changed={"c":"cherry"})
        foo = MyClass(cb)
        foo.d["c"] = "coconut"
        self.assert_(cb.called)
        # adding a new item
        cb = Callback(self, added={"g":"guava"})
        bar = MyClass(cb)
        bar.d["g"] = "guava"
        self.assert_(cb.called)
        return
    
    def test_delitem(self):
        cb = Callback(self, removed={"b":"banana"})
        foo = MyClass(cb)
        del foo.d["b"]
        self.assert_(cb.called)
        return
    
    def test_clear(self):
        removed = MyClass(None).d.copy()
        cb = Callback(self, removed=removed)
        foo = MyClass(cb)
        foo.d.clear()
        self.assert_(cb.called)
        return

    def test_update(self):
        update_dict = {"a":"artichoke", "f": "fig"}
        cb = Callback(self, changed={"a":"apple"}, added={"f":"fig"})
        foo = MyClass(cb)
        foo.d.update(update_dict)
        self.assert_(cb.called)
        return

    def test_setdefault(self):
        # Test retrieving an existing value
        cb = Callback(self)
        foo = MyClass(cb)
        self.assert_(foo.d.setdefault("a", "dummy") == "apple")
        self.assert_(not cb.called)
        
        # Test adding a new value
        cb = Callback(self, added={"f":"fig"})
        bar = MyClass(cb)
        self.assert_(bar.d.setdefault("f", "fig") == "fig")
        self.assert_(cb.called)
        return

    def test_pop(self):
        # Test popping a non-existent key
        cb = Callback(self)
        foo = MyClass(cb)
        self.assert_(foo.d.pop("x", "dummy") == "dummy")
        self.assert_(not cb.called)
        
        # Test popping a regular item
        cb = Callback(self, removed={"c": "cherry"})
        bar = MyClass(cb)
        self.assert_(bar.d.pop("c") == "cherry")
        self.assert_(cb.called)
        return

    def test_popitem(self):
        foo = MyClass(None)
        foo.d.clear()
        foo.d["x"] = "xylophone"
        cb = Callback(self, removed={"x":"xylophone"})
        foo.callback = cb
        self.assert_(foo.d.popitem() == ("x", "xylophone"))
        self.assert_(cb.called)
        return

    def test_dynamic_listener(self):
        foo = MyOtherClass()
        # Test adding
        func = Callback(self, added={"g":"guava"})
        foo.on_trait_change(func.__call__, "d_items")
        foo.d["g"] = "guava"
        foo.on_trait_change(func.__call__, "d_items", remove=True)
        self.assert_(func.called)
        
        # Test removing
        func2 = Callback(self, removed={"a":"apple"})
        foo.on_trait_change(func2.__call__, "d_items")
        del foo.d["a"]
        foo.on_trait_change(func2.__call__, "d_items", remove=True)
        self.assert_(func2.called)
        
        # Test changing
        func3 = Callback(self, changed={"b":"banana"})
        foo.on_trait_change(func3.__call__, "d_items")
        foo.d["b"] = "broccoli"
        foo.on_trait_change(func3.__call__, "d_items", remove=True)
        self.assert_(func3.called)
        return

# EOF

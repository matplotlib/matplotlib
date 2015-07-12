from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from traitlets.config import (Configurable, TraitType) as (ipyConfigurbale, ipyTraitType)

from traitlets import (Int, Float, Bool, Dict, List, Instance,
        Union, TraitError, HasTraits, NoDefaultSpecified)

class Configurable(ipyConfigurable): pass

class OverloadMixin(object):

    def validate(self, obj, value):
        try:
            return super(OverloadMixin,self).validate(obj,value)
        except TraitError:
            if self.name:
                ohandle = '_%s_overload'%self.name
                if hasattr(obj, ohandle):
                    return getattr(obj, ohandle)(self, value)
            self.error(obj, value)

    def info(self):
        i = super(OverloadMixin,self).info()
        return 'overload resolvable, ' + i

class String(TraitType):
    """A string trait"""

    info_text = 'a string'

    def validate(self, obj, value):
        if isinstance(value, str):
            return value
        self.error(obj, value)
        

class Callable(TraitType):
    """A trait which is callable.

    Notes
    -----
    Classes are callable, as are instances
    with a __call__() method."""

    info_text = 'a callable'

    def validate(self, obj, value):
        if callable(value):
            return value
        else:
            self.error(obj, value)

class oInstance(OverloadMixin,Instance): pass
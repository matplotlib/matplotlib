from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

try:
    # IPython4 imports
    from traitlets.config import Configurable, Config
    from traitlets import (TraitType, Int, Float, Bool,
                           Dict, List, Instance, Union,
                           Unicode, Tuple, TraitError,
                           Undefined, BaseDescriptor,
                           getargspec)
except ImportError:
    # IPython3 imports
    from IPython.utils.traitlets.config import Configurable, Config
    from IPython.utils.traitlets import (TraitType, Int, Float, Bool,
                            Dict, List, Instance, Union, Unicode,
                            Tuple, TraitError, Undefined, BaseDescriptor,
                            getargspec)

import re
import types
import numpy as np
from matplotlib.externals import six
from .transforms import IdentityTransform, Transform
import contextlib


class exdict(dict):

    def __init__(self, *args, **kwargs):
        super(exdict, self).__init__(*args, **kwargs)
        self._memory = dict()

    def __setitem__(self, key, new):
        try:
            old = self[key]
            if old != new:
                self._memory[key] = old
        except KeyError:
            pass
        super(exdict, self).__setitem__(key, new)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError("update expected at most 1 arguments, got %d" % len(args))
        other = dict(*args, **kwargs)
        for key in other:
            self[key] = other[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]

    @property
    def ex(self):
        return self._memory.copy()


class PrivateMethodMixin(object):

    def __new__(cls, *args, **kwargs):
        inst = super(PrivateMethodMixin,cls).__new__(cls, *args, **kwargs)
        inst._trait_values = exdict(inst._trait_values)
        return inst

    def force_callback(self, name, cross_validate=True):
        if name not in self.traits():
            msg = "'%s' is not a trait of a %s class"
            raise TraitError(msg % (name, self.__class__))

        trait = self.traits()[name]

        new = self._trait_values[name]
        try:
            old = self._trait_values.ex[name]
        except KeyError:
            trait = getattr(self.__class__, name)
            old = trait.default_value

        self._notify_trait(name, old, new)
        if cross_validate:
            trait = self._retrieve_trait(name)
            # note value is updated via cross validation
            new = trait._cross_validate(self, new)
            self.private(name, new)

    def private(self, name, value=Undefined):
        trait = self._retrieve_trait(name)

        if value is not Undefined:
            trait._cross_validation_lock = True
            _notify_trait = self._notify_trait
            self._notify_trait = lambda *a: None
            setattr(self, name, value)
            self._notify_trait = _notify_trait
            trait._cross_validation_lock = False
            if isinstance(_notify_trait, types.MethodType):
                self.__dict__.pop('_notify_trait', None)

        if hasattr(trait, '__base_get__'):
            return trait.__base_get__(self)
        return getattr(self, name)

    def _retrieve_trait(self, name):
        try:
            trait = getattr(self.__class__, name)
            if not isinstance(trait, BaseDescriptor):
                msg = "'%s' is a standard attribute, not a trait, of a %s instance"
                raise TraitError(msg % (name, self.__class__.__name__))
        except AttributeError:
            msg = "'%s' is not a trait of a %s instance"
            raise TraitError(msg % (name, self.__class__.__name__))
        return trait

class OnGetMixin(object):

    def __init__(self, *args, **kwargs):
        super_obj = super(OnGetMixin,self)
        self.__base_get__ = super_obj.__get__
        self.__base_set__ = super_obj.__set__
        super_obj.__init__(*args, **kwargs)

    def __get__(self, obj, cls=None):
        value = self.__base_get__(obj,cls)

        if hasattr(obj, '_'+self.name+'_getter'):
            meth = getattr(obj, '_'+self.name+'_getter')
            if not callable(meth):
                raise TraitError(("""a trait getter method
                                   must be callable"""))
            argspec = len(getargspec(meth)[0])
            if isinstance(meth, types.MethodType):
                argspec -= 1
            if argspec==0:
                args = ()
            elif argspec==1:
                args = (value,)
            elif argspec==2:
                args = (value, self)
            elif argspec==3:
                args = (value, self, cls)
            else:
                raise TraitError(("""a trait getter method must
                                   have 3 or fewer arguments"""))
            value = meth(*args)
        
        return value

    def __set__(self, obj, value):
        if self.read_only:
            raise TraitError('The "%s" trait is read-only.' % self.name)
        elif hasattr(obj, '_'+self.name+'_setter'):
            meth = getattr(obj, '_'+self.name+'_setter')
            if not callable(meth):
                raise TraitError(("""a trait setter method
                                   must be callable"""))
            argspec = len(getargspec(meth)[0])
            if isinstance(meth, types.MethodType):
                argspec -= 1
            if argspec==0:
                args = ()
            elif argspec==1:
                args = (value,)
            elif argspec==2:
                args = (obj._trait_values[self.name], value)
            elif argspec==3:
                args = (obj._trait_values[self.name], value, self)
            else:
                raise TraitError(("""a trait setter method must
                                   have 2 or fewer arguments"""))
            value = meth(*args)

            if value is not obj._trait_values[self.name]:
                self.set(obj, value)
        else:
            self.set(obj, value)


class TransformInstance(TraitType):

    info_text = ('a Transform instance or have an'
                 ' `_as_mpl_transform` method')

    def __init__(self, *args, **kwargs):
        super(TransformInstance,self).__init__(*args, **kwargs)
        self._conversion_method = False

    def _validate(self, obj, value):
        if hasattr(self, 'validate'):
            value = self.validate(obj, value)
        if obj._cross_validation_lock is False:
            value = self._cross_validate(obj, value)
        return value

    def validate(self, obj, value):
        if value is None:
            return IdentityTransform()
        if isinstance(value, Transform):
            self._conversion_method = False
            return value
        elif hasattr(value, '_as_mpl_transform'):
            self._conversion_method = True
            return value._as_mpl_transform
        trait.error(obj, value)

class gTransformInstance(OnGetMixin,TransformInstance): pass

#!Note : this is what the transform instance would
# look like if getters were to be avoided entirely.
# `_name_validate` would handle "on set" events
# while standard change handlers would accomodate
# any "on get" requirements. This could be hairy
# to implement, but in principle it seems possible.
# For now though, getters will remain a crutch to
# make it through testing.

# class TransformInstance(TraitType):

#     info_text = ('None, a Transform instance or have an'
#                  ' `_as_mpl_transform` method')
#     allow_none = True

#     def __init__(self, *args, **kwargs):
#         super(TransformInstance,self).__init__(*args, **kwargs)
#         self._conversion_value = Undefined

#     def __get__(self, obj, cls=None):
#         value = super(TransformInstance,self).__get__(obj,cls)
#         if self._conversion_value is not Undefined:
#             return self._conversion_value
#         return value

#     def _validate(self, obj, value):
#         if value is None:
#             return IdentityTransform()
#         if hasattr(self, 'validate'):
#             value = self.validate(obj, value)
#         if obj._cross_validation_lock is False:
#             value = self._cross_validate(obj, value)
#         return value

#     def validate(self, obj, value):
#         if isinstance(value, Transform):
#             if self._conversion_value is not Undefined:
#                 self._conversion_value = Undefined
#             return value
#         elif hasattr(value, '_as_mpl_transform'):
#             method = value._as_mpl_transform
#             try:
#                 self._conversion_value = method(obj.axes)
#             except:
#                 self._conversion_value = None
#         trait.error(obj, value)

class Callable(TraitType):
    """A trait which is callable.

    Notes
    -----
    Classes are callable, as are instances
    with a __call__() method."""

    info_text = 'a callable'

    def validate(self, obj, value):
        if six.callable(value):
            return value
        else:
            self.error(obj, value)

class Stringlike(Unicode):

    info_text = 'string or unicode interpretable'

    def validate(self, obj, value):
        if not isinstance(value, (str,unicode)):
            if hasattr(value,'__unicode__'):
                value = unicode(value)
            elif hasattr(value, '__str__'):
                value = str(value)
        return super(Stringlike,self).validate(obj,value)

class Color(TraitType):
    """A trait representing a color, can be either in RGB, or RGBA format.

    Arguments:
        force_rgb: bool: Force the return in RGB format instead of RGB. Default: False
        as_hex: bool: Return the hex value instead. Default: False
        default_alpha: float (0.0-1.0) or integer (0-255). Default alpha value (1.0)

    Accepts:
        string: a valid hex color string (i.e. #FFFFFF). With 4 or 7 chars.
        tuple: a tuple of ints (0-255), or tuple of floats (0.0-1.0)
        float: A gray shade (0-1)
        integer: A gray shade (0-255)

    Defaults: RGBA tuple, color black (0.0, 0.0, 0.0, 1.0)

    Return:
       A tuple of floats (r,g,b,a), (r,g,b) or a hex color string. i.e. "#FFFFFF".

    """
    metadata = {
        'force_rgb': False,
        'as_hex' : False,
        'default_alpha' : 1.0,
        }
    allow_none = True
    info_text = 'float, int, tuple of float or int, or a hex string color'
    default_value = (0.0,0.0,0.0, metadata['default_alpha'])
    named_colors = {}
    _re_color_hex = re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$')

    def _int_to_float(self, value):
        as_float = (np.array(value)/255).tolist()
        return as_float

    def _float_to_hex(self, value):
        as_hex = '#%02x%02x%02x' % tuple([int(np.round(v * 255)) for v in\
                                                                 value[:3]])
        return as_hex

    def _int_to_hex(self, value):
        as_hex = '#%02x%02x%02x' % value[:3]
        return as_hex

    def _hex_to_float(self, value):
        if len(value) == 7:
            split_hex = (value[1:3],value[3:5],value[5:7])
            as_float = (np.array([int(v,16) for v in split_hex])/255.0).tolist()
        elif len(value) == 4:
            as_float = (np.array([int(v+v,16) for v in value[1:]])/255.0).tolist()
        return as_float

    def _float_to_shade(self, value):
        grade = value*255.0
        return (grade,grade,grade)

    def _int_to_shade(self, value):
        grade = value/255.0
        return (grade,grade,grade)

    def validate(self, obj, value):
        in_range = False
        if value is True:
            self.error(obj, value)

        elif value is None or value is False or value in ['none','']:
            value = (0.0, 0.0, 0.0, 0.0)
            in_range = True

        elif isinstance(value, float):
            if 0 <= value <= 1:
                value = self._float_to_shade(value)
                in_range = True
            else:
                in_range = False

        elif isinstance(value, int):
            if 0 <= value <= 255:
                value = self._int_to_shade(value)
                in_range = True
            else:
                in_range = False

        elif isinstance(value, (tuple, list)) and len(value) in (3,4):
            is_all_float = np.prod([isinstance(v, (float)) for v in value])
            in_range = np.prod([(0 <= v <= 1) for v in value])
            if is_all_float and in_range:
                value = value
            else:
                is_all_int = np.prod([isinstance(v, int) for v in value])
                in_range = np.prod([(0 <= v <= 255) for v in value])
                if is_all_int and in_range:
                    value = self._int_to_float(value)

        elif isinstance(value, str) and len(value) in [4,7] and value[0] == '#':
            if self._re_color_hex.match(value):
                value = self._hex_to_float(value)
                in_range = np.prod([(0 <= v <= 1) for v in value])
                if in_range:
                    value = value

        elif isinstance(value, str) and value in self.named_colors:
            value = self.validate(obj, self.named_colors[value])
            in_range = True
            
        if in_range:
            # Convert to hex color string
            if self._metadata['as_hex']:
                return self._float_to_hex(value)

            # Ignores alpha and return rgb
            if self._metadata['force_rgb'] and in_range:
                return tuple(np.round(value[:3],5).tolist())

            # If no alpha provided, use default_alpha, also round the output
            if len(value) == 3:
                value = tuple(np.round((value[0], value[1], value[2], 
                             self._metadata['default_alpha']),5).tolist())
            elif len(value) == 4:
            # If no alpha provided, use default_alpha
                value = tuple(np.round(value,5).tolist())

            return value

        self.error(obj, value)
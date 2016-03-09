from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# IPython4 imports
from traitlets.config import Configurable, Config
from traitlets import (TraitType, Int, Float, Bool,
                       Dict, List, Instance, Union,
                       Unicode, Tuple, TraitError,
                       Undefined, BaseDescriptor,
                       getargspec, observe, default,
                       validate, EventHandler)

import re
import types
import numpy as np
from matplotlib.externals import six
from matplotlib.colors import cnames
from .transforms import IdentityTransform, Transform
import contextlib


class PrivateMethodMixin(object):

    def force_notify_changes(self, *changes):
        for change in changes:
            self.notify_change(change)

    @contextlib.contextmanager
    def mute_trait_notifications(self, cross_validate=True):
        """Context manager for muting trait change notifications and cross
        validation.
        Use this when doing multiple trait assignments (init, config), to avoid
        race conditions in trait notifiers requesting other trait values.
        All trait notifications will fire after all values have been assigned.
        """
        if self._cross_validation_lock is True:
            yield {}
            return
        else:
            cache = {}
            notify_change = self.notify_change

            def compress(past_changes, change):
                """Merges the provided change with the last if possible."""
                if past_changes is None:
                    return [change]
                else:
                    if past_changes[-1]['type'] == 'change' and change['type'] == 'change':
                        past_changes[-1]['new'] = change['new']
                    else:
                        # In case of changes other than 'change', append the notification.
                        past_changes.append(change)
                    return past_changes

            def hold(change):
                name = change['name']
                cache[name] = compress(cache.get(name), change)

            try:
                # Replace notify_change with `hold`, caching and compressing
                # notifications, disable cross-validation and yield.
                self.notify_change = hold
                if not cross_validate:
                    self._cross_validation_lock = True
            except TraitError as e:
                # Roll back in case of TraitError during final cross validation.
                self.notify_change = lambda x: None
                for name, changes in cache.items():
                    for change in changes[::-1]:
                        # TODO: Separate in a rollback function per notification type.
                        if change['type'] == 'change':
                            if change['old'] is not Undefined:
                                self.set_trait(name, change['old'])
                            else:
                                self._trait_values.pop(name)
                cache.clear()
                raise e
            finally:
                self.notify_change = notify_change
                if not cross_validate:
                    self._cross_validation_lock = False
                if isinstance(notify_change, types.MethodType):
                    # FIXME: remove when support is bumped to 3.4.
                    # when original method is restored,
                    # remove the redundant value from __dict__
                    # (only used to preserve pickleability on Python < 3.4)
                    self.__dict__.pop('notify_change', None)

    @contextlib.contextmanager
    def hold_trait_notifications(self):
        """Context manager for bundling trait change notifications and cross
        validation.
        Use this when doing multiple trait assignments (init, config), to avoid
        race conditions in trait notifiers requesting other trait values.
        All trait notifications will fire after all values have been assigned.
        """
        try:
            with self.mute_trait_notifications() as cache:
                yield
        finally:
            for c in cache.values():
                self.notify_change(c)

    def private(self, name, value=Undefined):
        trait = self._retrieve_trait(name)

        if value is not Undefined:
            self._trait_values[name] = value
        else:
            return trait.get(self, None)

    def _retrieve_trait(self, name):
        try:
            trait = getattr(self.__class__, name)
            if not isinstance(trait, BaseDescriptor):
                msg = ("'%s' is a standard attribute, not a traitlet, of"
                       " %s instances" % (name, self.__class__.__name__))
                raise TraitError(msg)
        except AttributeError:
            msg = "'%s' is not a traitlet of %s instances"
            raise TraitError(msg % (name, self.__class__.__name__))
        return trait


def retrieve(name):
    return RetrieveHandler(name)


class RetrieveHandler(EventHandler):

    def __init__(self, name):
        self._name = name

    def instance_init(self, inst):
        if not hasattr(inst, '_retrieve_handlers'):
            inst._retrieve_handlers = {}
        handler = inst._retrieve_handlers.get(self._name)
        if handler and hasattr(handler, 'func'):
                raise TraitError("A retriever for the trait '%s' has "
                                 "already been registered" % self._name)
        inst._retrieve_handlers[self._name] = self

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('func', None)
        return d


class OnGetMixin(object):

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        try:
            value = super(OnGetMixin, self).get(obj, cls)
            value_found = True
        except TraitError, e:
            value_found = False
        finally:
            has_retrievers = hasattr(obj, '_retrieve_handlers')
            if has_retrievers and self.name in obj._retrieve_handlers:
                handler = obj._retrieve_handlers[self.name]
                if not value_found:
                    value = Undefined
                pull = {'value': value, 'owner': obj, 'trait': self}
                value = handler(obj, pull)
            elif not value_found:
                raise TraitError(e)
        return value


class TransformInstance(OnGetMixin, TraitType):

    info_text = ('a Transform instance or have an'
                 ' `_as_mpl_transform` method')

    def __init__(self, *args, **kwargs):
        super(TransformInstance, self).__init__(*args, **kwargs)
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
        if not isinstance(value, six.text_type):
            if hasattr(value, '__unicode__'):
                value = six.text_type(value)
            elif hasattr(value, '__str__'):
                value = str(value)
        return super(Stringlike, self).validate(obj, value)


class Color(TraitType):
    """A trait representing a color, can be either in RGB, or RGBA format.

    Arguments:
        as_rgb: bool: coerce to RGB. Default: False
        as_hex: bool: coerce to hex value. Default: False
        default_alpha: float (0.0-1.0) or integer (0-255). Default (1.0)

    Accepts:
        string: a valid hex color string (i.e. #FFFFFF). With 4 or 7 chars.
        tuple: a tuple of ints (0-255), or tuple of floats (0.0-1.0)
        float: A gray shade (0-1)
        integer: A gray shade (0-255)

    Defaults: RGBA tuple, color black (0.0, 0.0, 0.0, 1.0)

    Return:
       A tuple of floats (r,g,b,a), (r,g,b) or a hex color string.

    """
    metadata = {
        'as_rgb': False,
        'as_hex': False,
        'default_alpha': 1.0,
        }
    info_text = 'float, int, tuple of float or int, or a hex string color'
    default_value = (0.0, 0.0, 0.0, metadata['default_alpha'])
    named_colors = cnames
    _re_color_hex = re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$')

    def __init__(self, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)

    def _int_to_float(self, value):
        as_float = (np.array(value)/255).tolist()
        return as_float

    def _float_to_hex(self, value):
        as_hex = '#%02x%02x%02x' % tuple([int(np.round(v * 255))
                                          for v in value[:3]])
        return as_hex

    def _int_to_hex(self, value):
        as_hex = '#%02x%02x%02x' % value[:3]
        return as_hex

    def _hex_to_float(self, value):
        if len(value) == 7:
            split_hex = (value[1:3], value[3:5], value[5:7])
            as_float = (np.array([int(v, 16) for v in split_hex]) / 255.0)
        elif len(value) == 4:
            as_float = (np.array([int(v+v, 16) for v in value[1:]]) / 255.0)
        return as_float.tolist()

    def _float_to_shade(self, value):
        grade = value*255.0
        return (grade, grade, grade)

    def _int_to_shade(self, value):
        grade = value/255.0
        return (grade, grade, grade)

    def validate(self, obj, value):
        in_range = False
        if value is True:
            self.error(obj, value)

        elif value is None or value is False or value in ['none', '']:
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

        elif isinstance(value, (tuple, list)) and len(value) in (3, 4):
            is_all_float = np.prod([isinstance(v, (float)) for v in value])
            in_range = np.prod([(0 <= v <= 1) for v in value])
            if is_all_float and in_range:
                value = value
            else:
                is_all_int = np.prod([isinstance(v, int) for v in value])
                in_range = np.prod([(0 <= v <= 255) for v in value])
                if is_all_int and in_range:
                    value = self._int_to_float(value)

        elif isinstance(value, (str, unicode)):
            if value[0] == '#' and len(value) in (4, 7):
                if self._re_color_hex.match(value):
                    value = self._hex_to_float(value)
                    in_range = np.prod([(0 <= v <= 1) for v in value])
            elif value in self.named_colors:
                return self.validate(obj, self.named_colors[value])

        if in_range:
            # Convert to hex color string
            if self.metadata['as_hex']:
                return self._float_to_hex(value)

            # Ignores alpha and return rgb
            if self.metadata['as_rgb'] and in_range:
                return tuple(np.round(value[:3], 5).tolist())

            # If no alpha provided, use default_alpha, also round the output
            if len(value) == 3:
                value = tuple(np.round((value[0], value[1], value[2],
                             self.metadata['default_alpha']), 5).tolist())
            elif len(value) == 4:
            # If no alpha provided, use default_alpha
                value = tuple(np.round(value, 5).tolist())

            return value

        self.error(obj, value)


def _traitlets_deprecation_msg(name):
    msg = ("This has been deprecated to make way for Traitlets. Please"
           " use the '%s' TraitType and Traitlet event decorators.")
    return msg % name

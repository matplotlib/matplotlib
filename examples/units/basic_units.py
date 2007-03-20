
import numpy as N
import numpy.core.ma as ma
from matplotlib.units import *
from matplotlib.axes import Axes
from matplotlib.ticker import AutoLocator, ScalarFormatter
    
class ProxyDelegate(object):
  def __init__(self, fn_name, proxy_type):
    self.proxy_type = proxy_type
    self.fn_name = fn_name
  def __get__(self, obj, objtype=None):
    return self.proxy_type(self.fn_name, obj)

class TaggedValueMeta (type):
  def __init__(cls, name, bases, dict):
    for fn_name in cls._proxies.keys(): 
      try:
        dummy = getattr(cls, fn_name)
      except AttributeError:
        setattr(cls, fn_name, ProxyDelegate(fn_name, cls._proxies[fn_name]))

class PassThroughProxy(object):
  def __init__(self, fn_name, obj):
    self.fn_name = fn_name
    self.target = obj.proxy_target
  def __call__(self, *args):
    fn = getattr(self.target, self.fn_name)
    ret = fn(*args)
    return ret

class ConvertArgsProxy(PassThroughProxy):
  def __init__(self, fn_name, obj):
    PassThroughProxy.__init__(self, fn_name, obj)
    self.unit = obj.unit
  def __call__(self, *args):
    converted_args = []
    for a in args:
      try:
        converted_args.append(a.convert_to(self.unit))
      except AttributeError:
        converted_args.append(TaggedValue(a, self.unit)) 
    converted_args = tuple([c.get_value() for c in converted_args])
    return PassThroughProxy.__call__(self, *converted_args)

class ConvertReturnProxy(PassThroughProxy):
  def __init__(self, fn_name, obj):
    PassThroughProxy.__init__(self, fn_name, obj)
    self.unit = obj.unit
  def __call__(self, *args):
    ret = PassThroughProxy.__call__(self, *args)
    if (type(ret) == type(NotImplemented)):
      return NotImplemented
    return TaggedValue(ret, self.unit)

class ConvertAllProxy(PassThroughProxy):
  def __init__(self, fn_name, obj):
    PassThroughProxy.__init__(self, fn_name, obj)
    self.unit = obj.unit
  def __call__(self, *args):
    converted_args = []
    arg_units = [self.unit]
    for a in args:
      if hasattr(a, 'get_unit') and not hasattr(a, 'convert_to'):
        # if this arg has a unit type but no conversion ability,
        # this operation is prohibited
        return NotImplemented

      if hasattr(a, 'convert_to'):
        try:
          a = a.convert_to(self.unit)
        except:
          pass
        arg_units.append(a.get_unit())
        converted_args.append(a.get_value()) 
      else:
        converted_args.append(a)
        if hasattr(a, 'get_unit'):
          arg_units.append(a.get_unit())
        else:
          arg_units.append(None)
    converted_args = tuple(converted_args)
    ret = PassThroughProxy.__call__(self, *converted_args)
    if (type(ret) == type(NotImplemented)):
      return NotImplemented
    ret_unit = unit_resolver(self.fn_name, arg_units)
    if (ret_unit == NotImplemented):
      return NotImplemented
    return TaggedValue(ret, ret_unit)

class TaggedValue (UnitsTagInterfaceWithMA, object):

  __metaclass__ = TaggedValueMeta
  _proxies = {'__add__':ConvertAllProxy, 
              '__mul__':ConvertAllProxy,
              '__rmul__':ConvertAllProxy,
              '__len__':PassThroughProxy}

  def __new__(cls, value, unit):
    # generate a new subclass for value
    value_class = type(value)
    try:
        subcls = type('TaggedValue_of_%s' % (`value_class.__name__`),
                      tuple([cls, value_class]),
                      {})
        return object.__new__(subcls, value, unit)
    except:
        return object.__new__(cls, value, unit)

  def __init__(self, value, unit):
    self.value = value
    self.unit  = unit
    self.proxy_target = self.value

  def get_compressed_copy(self, mask):
    compressed_value = ma.masked_array(self.value, mask=mask).compressed()
    return TaggedValue(compressed_value, self.unit)

  def __getattribute__(self, name):
    if (name.startswith('__')):
       return object.__getattribute__(self, name)
    variable = object.__getattribute__(self, 'value')
    if (hasattr(variable, name) and name not in self.__class__.__dict__):
      return getattr(variable, name)
    return object.__getattribute__(self, name)

  def __array__(self, t = None, context = None):
    if t:
      return N.asarray(self.value).astype(t)
    else:
      return N.asarray(self.value)

  def __array_wrap__(self, array, context):
    return TaggedValue(array, self.unit)
 
  def __repr__(self):
    return 'TaggedValue(' + repr(self.value) + ', ' + repr(self.unit) + ')'

  def __str__(self):
    return str(self.value) + ' in ' + str(self.unit)

  def __iter__(self):
    class IteratorProxy(object):
      def __init__(self, iter, unit):
        self.iter = iter
        self.unit = unit
      def next(self):
        value = self.iter.next()
        return TaggedValue(value, self.unit) 
    return IteratorProxy(iter(self.value), self.unit)

  def get_compressed_copy(self, mask):
    new_value = ma.masked_array(self.value, mask=mask).compressed()
    return TaggedValue(new_value, self.unit)

  def convert_to(self, unit):
    if (unit == self.unit or not unit):
      return self
    new_value = self.unit.convert_value_to(self.value, unit)
    return TaggedValue(new_value, unit)

  def get_value(self):
    return self.value

  def convert_to_value(self, unit):
    return self.convert_to(unit).get_value()

  def get_default_unit_tag(self):
    return self.unit

  def get_unit(self):
    return self.unit

class BasicUnit(object):
  def __init__(self, name, full_name=None, tick_locators=None, tick_formatters=None):
    self.name = name
    if (not full_name):
      full_name = name
    self.full_name = full_name
    self.conversions = dict()
    self.tick_locators = tick_locators
    self.tick_formatters = tick_formatters

  def set_tick_locators(self, major_locator, minor_locator):
    self.tick_locators = (major_locator, minor_locator)

  def get_tick_locators(self):
    return self.tick_locators

  def set_tick_formatter(self, major_formatter, minor_formatter):
    self.tick_formatters = (major_formatter, minor_formatter)

  def get_tick_formatters(self):
    return self.tick_formatters

  def __repr__(self):
    return 'BasicUnit(' + `self.name` + ')'

  def __str__(self):
    return self.full_name

  def __call__(self, value):
    return TaggedValue(value, self)

  def __mul__(self, rhs):
    value = rhs
    unit  = self
    if hasattr(rhs, 'get_unit'):
      value = rhs.get_value()
      unit  = rhs.get_unit()
      unit  = unit_resolver('__mul__', (self, unit))
    if (unit == NotImplemented):
      return NotImplemented
    return TaggedValue(value, unit)

  def __rmul__(self, lhs):
    return self*lhs

  def __array_wrap__(self, array, context):
    return TaggedValue(array, self)

  def __array__(self, t=None, context=None):
    ret = N.array([1])
    if (t):
      return ret.astype(t)
    else:
      return ret
    
  def add_conversion_factor(self, unit, factor):
    def convert(x):
      return x*factor
    self.conversions[unit] = convert

  def add_conversion_fn(self, unit, fn):
    self.conversions[unit] = fn

  def get_conversion_fn(self, unit):
    return self.conversions[unit]

  def convert_value_to(self, value, unit):
    conversion_fn = self.conversions[unit]
    ret = conversion_fn(value)
    return ret


  def get_unit(self):
    return self

class UnitResolver(object):
  def addition_rule(self, units):
    for unit_1, unit_2 in zip(units[:-1], units[1:]):
      if (unit_1 != unit_2):
        return NotImplemented
    return units[0]
  def multiplication_rule(self, units):
    non_null = [u for u in units if u]
    if (len(non_null) > 1):
      return NotImplemented
    return non_null[0]

  op_dict = {
    '__mul__':multiplication_rule,
    '__rmul__':multiplication_rule,
    '__add__':addition_rule,
    '__radd__':addition_rule,
    '__sub__':addition_rule,
    '__rsub__':addition_rule,
  }
 
  def __call__(self, operation, units):
    if (operation not in self.op_dict):
      return NotImplemented

    return self.op_dict[operation](self, units)

unit_resolver = UnitResolver()

def locator_map(u):
  return u.get_tick_locators()
def formatter_map(u):
  return u.get_tick_formatters()

Axes.set_default_unit_to_locator_map(locator_map)
Axes.set_default_unit_to_formatter_map(formatter_map)


import math


import matplotlib.units as units
import matplotlib.ticker as ticker
import matplotlib.numerix as nx
from matplotlib.axes import Axes
from matplotlib.cbook import iterable

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
        #print 'passthrough', self.target, self.fn_name
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

class TaggedValue (object):

  __metaclass__ = TaggedValueMeta
  _proxies = {'__add__':ConvertAllProxy,
              '__sub__':ConvertAllProxy,
              '__mul__':ConvertAllProxy,
              '__rmul__':ConvertAllProxy,
              '__len__':PassThroughProxy}

  def __new__(cls, value, unit):
    # generate a new subclass for value
    value_class = type(value)
    try:
        subcls = type('TaggedValue_of_%s' % (value_class.__name__),
                      tuple([cls, value_class]),
                      {})
        if subcls not in units.registry:
            units.registry[subcls] = basicConverter
        return object.__new__(subcls, value, unit)
    except TypeError:
        if cls not in units.registry:
            units.registry[cls] = basicConverter
        return object.__new__(cls, value, unit)

  def __init__(self, value, unit):
    self.value = value
    self.unit  = unit
    self.proxy_target = self.value

  def get_compressed_copy(self, mask):
    compressed_value = nx.ma.masked_array(self.value, mask=mask).compressed()
    return TaggedValue(compressed_value, self.unit)

  def  __getattribute__(self, name):
    if (name.startswith('__')):
       return object.__getattribute__(self, name)
    variable = object.__getattribute__(self, 'value')
    if (hasattr(variable, name) and name not in self.__class__.__dict__):
      return getattr(variable, name)
    return object.__getattribute__(self, name)

  def __array__(self, t = None, context = None):
    if t is not None:
      return nx.asarray(self.value).astype(t)
    else:
      return nx.asarray(self.value, 'O')

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
    new_value = nx.ma.masked_array(self.value, mask=mask).compressed()
    return TaggedValue(new_value, self.unit)

  def convert_to(self, unit):
      #print 'convert to', unit, self.unit
      if (unit == self.unit or not unit):
          return self
      new_value = self.unit.convert_value_to(self.value, unit)
      return TaggedValue(new_value, unit)

  def get_value(self):
    return self.value

  def get_unit(self):
    return self.unit


class BasicUnit(object):
  def __init__(self, name, fullname=None):
    self.name = name
    if fullname is None: fullname = name
    self.fullname = fullname
    self.conversions = dict()


  def __repr__(self):
    return 'BasicUnit(%s)'%self.name

  def __str__(self):
    return self.fullname

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
    ret = nx.array([1])
    if t is not None:
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
      #print 'convert value to: value ="%s", unit="%s"'%(value, type(unit)), self.conversions
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


cm = BasicUnit('cm', 'centimeters')
inch = BasicUnit('inch', 'inches')
inch.add_conversion_factor(cm, 2.54)
cm.add_conversion_factor(inch, 1/2.54)

radians = BasicUnit('rad', 'radians')
degrees = BasicUnit('deg', 'degrees')
radians.add_conversion_factor(degrees, 180.0/nx.pi)
degrees.add_conversion_factor(radians, nx.pi/180.0)

secs = BasicUnit('s', 'seconds')
hertz = BasicUnit('Hz', 'Hertz')
minutes = BasicUnit('min', 'minutes')

secs.add_conversion_fn(hertz, lambda x:1./x)
secs.add_conversion_factor(minutes, 1/60.0)

# radians formatting
def rad_fn(x,pos=None):
  n = int((x / nx.pi) * 2.0 + 0.25)
  if n == 0:
    return '0'
  elif n == 1:
    return r'$\pi/2$'
  elif n == 2:
    return r'$\pi$'
  elif n % 2 == 0:
    return r'$%s\pi$' % (n/2,)
  else:
    return r'$%s\pi/2$' % (n,)


class BasicUnitConverter(units.ConversionInterface):

    def axisinfo(unit):
        'return AxisInfo instance for x and unit'

        if unit==radians:
            return units.AxisInfo(
              majloc=ticker.MultipleLocator(base=nx.pi/2),
              majfmt=ticker.FuncFormatter(rad_fn),
              label=unit.fullname,
                )
        elif unit==degrees:
            return units.AxisInfo(
              majloc=ticker.AutoLocator(),
              majfmt=ticker.FormatStrFormatter(r'$%i^\circ$'),
              label=unit.fullname,
                )
        elif unit is not None:
            if hasattr(unit, 'fullname'):
                return units.AxisInfo(label=unit.fullname)
            elif hasattr(unit, 'unit'):
                return units.AxisInfo(label=unit.unit.fullname)
        return None

    axisinfo = staticmethod(axisinfo)

    def convert(val, unit):
        if units.ConversionInterface.is_numlike(val):
            return val
        #print 'convert checking iterable'
        if iterable(val):
            return [thisval.convert_to(unit).get_value() for thisval in val]
        else:
            return val.convert_to(unit).get_value()
    convert = staticmethod(convert)

    def default_units(x):
        'return the default unit for x or None'
        if iterable(x):
            for thisx in x:
                return thisx.unit
        return x.unit
    default_units = staticmethod(default_units)



def cos( x ):
   if ( iterable(x) ):
      result = []
      for val in x:
         result.append( math.cos( val.convert_to( radians ).get_value() ) )
      return result
   else:
      return math.cos( x.convert_to( radians ).get_value() )

basicConverter = BasicUnitConverter()
units.registry[BasicUnit] = basicConverter
units.registry[TaggedValue] = basicConverter


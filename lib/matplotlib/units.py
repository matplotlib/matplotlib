
from cbook import iterable, flatten
import copy

class UnitsTagInterface:
    """
       This interface describes the expected protocol for types implementing
       values tagged with measurement units.

       In addition for an implementation that represents values along with
       an associated measurement unit, matplotlib needs a consistent 
       mechanism for passing unit types.  To matplotlib, units are 
       merely identifiers, and the actual objects can be of any 
       Python type, as long as the implementation of convert_to()
       handles that Python type.  So, units could be unique strings
       or specialized objects, depending on the implementation of the
       value class.

       For custom TickLocator and TickFormatter instances, one must define
       a function that returns locators and formatter pairs corresponding
       to a unit.  The function should return a tuple containing a major
       locator/formatter object and a minor locator/formatter object.
       Example:
           def simple_locator_map(unit_object):
               'returns (major locator, minor locator) tuple for unit'
               return (AutoLocator(), NullLocator())

       Once defined, the function must be passed to the current Axes
       object in one of two ways.

       First, the implementations of get_unit_to_[formatter|locator]_map()
       can return this function.

       Second, default maps can be specified using the static methods
       in the Axes class, set_default_unit_to_[locator|formatter]_map().

       When determining the locator/formatter, the first valid
       locator/formatter pair is used.  All supplied data is queried
       for locator/formatter functions, and the default map is checked
       only when a check of the data results in no valid locator/formatter
       pairs.

       Lastly, duplicate functions (duplicate Python objects) are not
       checked multiple times.  Thus, whenever possible, returning the
       same Python object as the locator/formatter function may improve
       efficiency.
    """
    def convert_to_value(self, unit):
        """
           Converts the existing units object to the specified units
           object and strips the target unit, leaving the unit-less
           values.  If convert_to() and convert_to_value() are
           both implemented, val.convert_to_value() should be equivalent
           to val.convert_to().get_value().
           Parameters:
             unit - unit of the desired type
           Returns:
             values converted to the requested units
        """
        raise NotImplemented

    def get_default_unit_tag(self):
        """
           Returns the default unit tag if not unit is specified.
           This method will be invoked when no desired unit is
           specified using xunits/yunits parameters.  The result
           of querying this method is used to set xunits/yunits.
           Returns:
             default unit tag when no unit is specified
        """
        raise NotImplemented

    def get_unit_to_locator_map(self):
        """
           If a custom locators are desired, this method should return
           a function with the profile
             fn(unit) => (<major locator>, <minor locator>)
           In the absence of a valid return, the default locators
           are used.
        """
        return None

    def get_unit_to_formatter_map(self):
        """
           If a custom formatters are desired, this method should return
           a function with the profile
             fn(unit) => (<major formatter>, <minor formatter>)
           In the absence of a valid return, the default formatters
           are used.
        """
        return None

class UnitsTagInterfaceWithMA(UnitsTagInterface):
    """
       Adds the one method required to implement unit classes which
       encapsulate masked arrays.
    """
    def get_compressed_copy(self, mask):
        """
           Returns the equivalent of ma.masked_array(self, mask).compressed()
           with this tagged value object as x.

           Implement this to provide for encapsulation of masked arrays
           if necessary.
        """
        raise NotImplemented

class UnitsTagConversionInterface:
    """
       This interface describes the expected protocol for defining conversions
       between Python types implementing values tagged with measurement units.

       In addition for an implementation that represents values along with
       an associated measurement unit, matplotlib needs a consistent 
       mechanism for passing unit types.  To matplotlib, units are 
       merely identifiers, and the actual objects can be of any 
       Python type, as long as the implementation of convert_to_value()
       handles that Python type.  So, units could be unique strings
       or specialized objects, depending on the implementation of the
       value class.

       Note: the methods convert_to() and get_value() are currently
       unused by the calling matplotlib code.

       For custom TickLocator and TickFormatter instances, one must define
       a function that returns locators and formatter pairs corresponding
       to a unit.  The function should return a tuple containing a major
       locator/formatter object and a minor locator/formatter object.
       Example:
           def simple_locator_map(unit_object):
               'returns (major locator, minor locator) tuple for unit'
               return (AutoLocator(), NullLoc#        temp_x, temp_y = self._convert_units((x, self._xunits),
#                                             (y, self._yunits))
ator())

       Once defined, the function must be passed to the current Axes
       object in one of two ways.

       First, the implementations of get_unit_to_[formatter|locator]_map()
       can return this function.

       Second, default maps can be specified using the static methods
       in the Axes class, set_default_unit_to_[locator|formatter]_map().

       When determining the locator/formatter, the first valid
       locator/formatter pair is used.  All supplied data is queried
       for locator/formatter functions, and the default map is checked
       only when a check of the data results in no valid locator/formatter
       pairs.

       Lastly, duplicate functions (duplicate Python objects) are not
       checked multiple times.  Thus, whenever possible, returning the
       same Python object as the locator/formatter function may improve
       efficiency.
    """
    def convert_to(self, tagged_value, unit):
        """Converts the tagged_value parameter to the specified units object.
           (Currently unused)
           Parameters:
             unit - unit of the desired type
           Returns:
             object converted to the requested units (should be of a type
             that supports this interface)
        """
        raise NotImplemented

    def get_value(self, tagged_value):
        """Returns the tagged_value stripped of its unit tag.  (Currently
           unused)
        """
        raise NotImplemented

    def convert_to_value(self, tagged_value, unit):
        """
           Converts the tagged_value object to the specified units
           object and strips the target unit, leaving the unit-less
           values.  If convert_to() and convert_to_value() are
           both implemented, val.convert_to_value() should be equivalent
           to val.convert_to().get_value().
           Parameters:
             unit - unit of the desired type
           Returns:
             values converted to the requested units
        """
        raise NotImplemented

    def get_default_unit_tag(self, tagged_value):
        """
           Returns the default unit tag if not unit is specified.
           This method will be invoked when no desired unit is
           specified using xunits/yunits parameters.  The result
           of querying this method is used to set xunits/yunits.
           Returns:
             default unit tag when no unit is specified
        """
        raise NotImplemented

    def get_unit_to_locator_map(self, tagged_value):
        """
           If a custom locators are desired, this method should return
           a function with the profile
             fn(unit) => (<major locator>, <minor locator>)
           In the absence of a valid return, the default locators
           are used.
        """
        return None
    def get_unit_to_formatter_map(self, tagged_value):
        """
           If a custom formatters are desired, this method should return
           a function with the profile
             fn(unit) => (<major formatter>, <minor formatter>)
           In the absence of a valid return, the default formatters
           are used.
        """
        return None
    
class UnitsManager:
    def __init__(self):
        self.unit_conversions = {}

    def register_unit_conversion(self, python_type, conversion):
        """      
        Register a unit conversion class
        
        ACCEPTS: a Unit instance
        """
        self._unit_conversions[python_type] = conversion

    def unregister_unit_conversion(self, python_type):
        """
        Unregister a unit conversion class
        
        ACCEPTS: any Python type
        """
        self._unit_conversions.remove(python_type)

    def _get_unit_conversion(self, python_type):
        """
        Get a unit conversion corresponding to a python type
        """
        for current in classes:
            if (current in self._unit_conversions):
                # found it!
                #print 'Found unit conversion for %s!' % (`python_type`)
                return self._unit_conversions[current]
        return None 
      
    def _invoke_units_method(self, method_name, value_arg_seq, \
                             distinct_lookup=False):
        """
           value_arg_seq should be a sequence of tuples.
           The contents of each tuple varies with the value of
           distinct_lookup and the parameters of the method
           being called.

           If distinct_lookup == True, the first element in
           the tuple is the lookup variable, followed by
           the value variable, then parameters for the method
           invocation.

           If distinct_lookup == False, the first element
           in the tuple is the value variable followed by
           the parameters of the method invocation.

           With distinct_lookup set to True, the lookup
           parameter is used for resolving the method name.
           Currently, this feature is unused and was added
           during earlier implementation.

           The real benefit of this routine is its encapsulation
           of the two types of conversion mechanism.  Whether
           the conversion implementation is internal or
           external to the units class, this method
           handles it.
        """
        class working_copy:
            """Working copy allows partial modification of
               sequences.  This is useful in the case of
               a heterogenous sequence, where only one element
               might need conversion.
            """
            def __init__(self, value, lookup=None):
                self.working_copy = value
                self.lookup = lookup
                self.copied = False
            def get_lookup(self, indices):
                if (not distinct_lookup):
                    return self.get_value(indices)
                position = self.lookup
                for index in indices[:-1]:
                    position = position[index]
                return position[indices[-1]]
            def get_value(self, indices):
                position = self.working_copy
                for index in indices[:-1]:
                    position = position[index]
                return position[indices[-1]]
            def set_value(self, indices, value):
                if (indices == [0]):
                    self.working_copy = [value]
                    self.copied = True
                    return
                if (not self.copied):
                    # first level is an enclosing list
                    self.working_copy = [copy.copy(self.working_copy[0])]
                    self.copied = True
                position = self.working_copy
                for index in indices[:-1]:
                    position = position[index]
                position[indices[-1]] = value

        def invoke_on_elem(working, current_index, previous_indices, args):
            #print 'in invoke_on_elem, args = %s' % (`args`)
            position = previous_indices + [current_index]
            value = working.get_value(position)
            lookup = working.get_lookup(position)
            # check for internal implementation
            if (hasattr(lookup, method_name)):
                #print 'in invoke_on_elem, args = %s' % (`args`)
                arg_list = args
                if (distinct_lookup):
                    arg_list = (value,) + arg_list
                value = getattr(lookup, method_name)(*arg_list)
                # copy check and replace
                working.set_value(position, value)
            else:
                conversion_class = None
                try:
                    conversion_class = \
                        self._get_unit_conversion(lookup.__class__)
                except: pass
                if (conversion_class):
                    # copy check
                    arg_list = [value]
                    arg_list.extend(args)
                    arg_list = tuple(arg_list)
                    value = getattr(conversion_class, method_name)(*arg_list)
                    # copy check and replace
                    working.set_value(position, value)
                elif (iterable(value)):
                    for v_index in range(len(value)):
                        invoke_on_elem(working, v_index, position, args)

        def invoke_once(args):
            if (distinct_lookup):
                # lookup is first
                lookup = args[0]
                value  = args[1]
                args   = args[2:]
                working = working_copy([value], lookup=[lookup])
            else:
                # value is first
                value  = args[0]
                args   = args[1:]
                working = working_copy([value])
            invoke_on_elem(working, 0, [], args)
            return working.get_value([0])

        ret = tuple([invoke_once(args) for args in value_arg_seq])
        return ret

    def _convert_units(self, *args):
        #print 'in _convert_units(%s)' % (`args`)
        ret = []
        # ML XXX replace me with direct calls
        return self._invoke_units_method('convert_to_value', args)
 

def getattr_nested1(x, attr):
    """
    try to getattr on x or the first element of x if x is iterable
    The value of None is returned if nothing is found
    """
    val = None
    val = getattr(x, attr, None)
    if val is not None: return val
    if iterable(x):
        for thisx in x:
            val = getattr(thisx, attr, None)
            break
    return val
    

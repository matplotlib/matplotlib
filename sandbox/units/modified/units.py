
class UnitTaggedInterface:
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
       object using set_units_locator_map().
       Example:
           gca().set_units_locator_map(simple_locator_map)

       The locator function can be set globally using
       Axes.set_default_units_locator_map().  An analogous function
       exists for formatter functions, Axes.set_default_units_formatter_map().
       A local function takes precedence over a globally defined locator/
       formatter function.
    """
    def convert_to(self, unit):
        """Converts the existing units object to the specified units object.
           Parameters:
             unit - unit of the desired type
           Returns:
             object converted to the requested units (should be of a type
             that supports this interface)
        """
        raise NotImplemented
    def get_value(self):
        """Returns the quantities stripped of unit.
        """
        raise NotImplemented


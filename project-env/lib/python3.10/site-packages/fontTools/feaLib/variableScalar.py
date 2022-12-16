from fontTools.varLib.models import VariationModel, normalizeValue


def Location(loc):
    return tuple(sorted(loc.items()))


class VariableScalar:
    """A scalar with different values at different points in the designspace."""

    # We will often use exactly the same locations (i.e. the font's
    # masters) for a large number of variable scalars. Instead of
    # creating a model for each, let's share the models.
    model_pool = {}

    @classmethod
    def clear_cache(cls):
        cls.model_pool = {}

    def __init__(self, location_value={}):
        self.values = {}
        self.axes = {}
        for location, value in location_value.items():
            self.add_value(location, value)

    def __repr__(self):
        items = []
        for location, value in self.values.items():
            loc = ",".join(["%s=%i" % (ax, loc) for ax, loc in location])
            items.append("%s:%i" % (loc, value))
        return "(" + (" ".join(items)) + ")"

    @property
    def does_vary(self):
        values = list(self.values.values())
        return any(v != values[0] for v in values[1:])

    @property
    def axes_dict(self):
        if not self.axes:
            raise ValueError(
                ".axes must be defined on variable scalar before interpolating"
            )
        return {ax.axisTag: ax for ax in self.axes}

    def _normalized_location(self, location):
        location = self.fix_location(location)
        normalized_location = {}
        for axtag in location.keys():
            if axtag not in self.axes_dict:
                raise ValueError("Unknown axis %s in %s" % (axtag, location))
            axis = self.axes_dict[axtag]
            normalized_location[axtag] = normalizeValue(
                location[axtag], (axis.minValue, axis.defaultValue, axis.maxValue)
            )

        return Location(normalized_location)

    def fix_location(self, location):
        location = dict(location)
        for tag, axis in self.axes_dict.items():
            if tag not in location:
                location[tag] = axis.defaultValue
        return location

    def add_value(self, location, value):
        if self.axes:
            location = self.fix_location(location)

        self.values[Location(location)] = value

    def fix_all_locations(self):
        self.values = {
            Location(self.fix_location(l)): v for l, v in self.values.items()
        }

    @property
    def default(self):
        self.fix_all_locations()
        key = Location({ax.axisTag: ax.defaultValue for ax in self.axes})
        if key not in self.values:
            raise ValueError("Default value could not be found")
            # I *guess* we could interpolate one, but I don't know how.
        return self.values[key]

    def value_at_location(self, location):
        loc = location
        if loc in self.values.keys():
            return self.values[loc]
        values = list(self.values.values())
        return self.model.interpolateFromMasters(loc, values)

    @property
    def model(self):
        key = tuple(self.values.keys())
        if key in self.model_pool:
            return self.model_pool[key]
        locations = [dict(self._normalized_location(k)) for k in self.values.keys()]
        m = VariationModel(locations)
        self.model_pool[key] = m
        return m

    def get_deltas_and_supports(self):
        values = list(self.values.values())
        return self.model.getDeltasAndSupports(values)

    def add_to_variation_store(self, store_builder):
        deltas, supports = self.get_deltas_and_supports()
        store_builder.setSupports(supports)
        index = store_builder.storeDeltas(deltas)
        return int(self.default), index

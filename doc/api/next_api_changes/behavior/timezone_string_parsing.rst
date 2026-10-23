Timezone strings are now parsed using `~zoneinfo.ZoneInfo`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Timezone strings are now parsed using `~zoneinfo.ZoneInfo` rather than
`~dateutil.tz.gettz`. This change affects the following methods:

- `matplotlib.axis.Axis.axis_date`
- `matplotlib.dates.num2date`
- `matplotlib.dates.DateFormatter`
- `matplotlib.dates.DateLocator` and subclasses

Timezone identifiers must be valid IANA time zone keys (``"America/Los_Angeles"``).

Other strings will raise a `~zoneinfo.ZoneInfoNotFoundError`. This includes
abbreviations (``"EST"``), POSIX-style names (``"EST5EDT"``), and invalid time zone
keys (``"Invalid/Zone"``).

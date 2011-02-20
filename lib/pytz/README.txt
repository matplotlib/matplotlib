pytz - World Timezone Definitions for Python
============================================

:Author: Stuart Bishop <stuart@stuartbishop.net>

Introduction
~~~~~~~~~~~~

pytz brings the Olson tz database into Python. This library allows
accurate and cross platform timezone calculations using Python 2.3
or higher. It also solves the issue of ambiguous times at the end
of daylight savings, which you can read more about in the Python
Library Reference (datetime.tzinfo).

Amost all (over 540) of the Olson timezones are supported [*]_.

Note that if you perform date arithmetic on local times that cross
DST boundaries, the results may be in an incorrect timezone (ie.
subtract 1 minute from 2002-10-27 1:00 EST and you get 2002-10-27
0:59 EST instead of the correct 2002-10-27 1:59 EDT). This cannot
be resolved without modifying the Python datetime implementation.
However, these tzinfo classes provide a normalize() method which
allows you to correct these values.


Installation
~~~~~~~~~~~~

This is a standard Python distutils distribution. To install the
package, run the following command as an administrative user::

    python setup.py install


Example & Usage
~~~~~~~~~~~~~~~

>>> from datetime import datetime, timedelta
>>> from pytz import timezone
>>> import pytz
>>> utc = pytz.utc
>>> utc.zone
'UTC'
>>> eastern = timezone('US/Eastern')
>>> eastern.zone
'US/Eastern'
>>> fmt = '%Y-%m-%d %H:%M:%S %Z%z'

The preferred way of dealing with times is to always work in UTC,
converting to localtime only when generating output to be read
by humans.

>>> utc_dt = datetime(2002, 10, 27, 6, 0, 0, tzinfo=utc)
>>> loc_dt = utc_dt.astimezone(eastern)
>>> loc_dt.strftime(fmt)
'2002-10-27 01:00:00 EST-0500'

This library also allows you to do date arithmetic using local
times, although it is more complicated than working in UTC as you
need to use the `normalize` method to handle daylight savings time
and other timezone transitions. In this example, `loc_dt` is set
to the instant when daylight savings time ends in the US/Eastern
timezone.

>>> before = loc_dt - timedelta(minutes=10)
>>> before.strftime(fmt)
'2002-10-27 00:50:00 EST-0500'
>>> eastern.normalize(before).strftime(fmt)
'2002-10-27 01:50:00 EDT-0400'
>>> after = eastern.normalize(before + timedelta(minutes=20))
>>> after.strftime(fmt)
'2002-10-27 01:10:00 EST-0500'

Creating localtimes is also tricky, and the reason why working with
local times is not recommended. Unfortunately, you cannot just pass
a 'tzinfo' argument when constructing a datetime (see the next section
for more details)

>>> dt = datetime(2002, 10, 27, 1, 30, 0)
>>> dt1 = eastern.localize(dt, is_dst=True)
>>> dt1.strftime(fmt)
'2002-10-27 01:30:00 EDT-0400'
>>> dt2 = eastern.localize(dt, is_dst=False)
>>> dt2.strftime(fmt)
'2002-10-27 01:30:00 EST-0500'

Converting between timezones also needs special attention. This also needs
to use the normalize method to ensure the conversion is correct.

>>> utc_dt = utc.localize(datetime.utcfromtimestamp(1143408899))
>>> utc_dt.strftime(fmt)
'2006-03-26 21:34:59 UTC+0000'
>>> au_tz = timezone('Australia/Sydney')
>>> au_dt = au_tz.normalize(utc_dt.astimezone(au_tz))
>>> au_dt.strftime(fmt)
'2006-03-27 08:34:59 EST+1100'
>>> utc_dt2 = utc.normalize(au_dt.astimezone(utc))
>>> utc_dt2.strftime(fmt)
'2006-03-26 21:34:59 UTC+0000'

You can also take shortcuts when dealing with the UTC side of timezone
conversions. Normalize and localize are not really necessary because there
are no daylight savings time transitions to deal with.

>>> utc_dt = datetime.utcfromtimestamp(1143408899).replace(tzinfo=utc)
>>> utc_dt.strftime(fmt)
'2006-03-26 21:34:59 UTC+0000'
>>> au_tz = timezone('Australia/Sydney')
>>> au_dt = au_tz.normalize(utc_dt.astimezone(au_tz))
>>> au_dt.strftime(fmt)
'2006-03-27 08:34:59 EST+1100'
>>> utc_dt2 = au_dt.astimezone(utc)
>>> utc_dt2.strftime(fmt)
'2006-03-26 21:34:59 UTC+0000'


Problems with Localtime
~~~~~~~~~~~~~~~~~~~~~~~

The major problem we have to deal with is that certain datetimes
may occur twice in a year. For example, in the US/Eastern timezone
on the last Sunday morning in October, the following sequence
happens:

    - 01:00 EDT occurs
    - 1 hour later, instead of 2:00am the clock is turned back 1 hour
      and 01:00 happens again (this time 01:00 EST)

In fact, every instant between 01:00 and 02:00 occurs twice. This means
that if you try and create a time in the US/Eastern timezone using
the standard datetime syntax, there is no way to specify if you meant
before of after the end-of-daylight-savings-time transition.

>>> loc_dt = datetime(2002, 10, 27, 1, 30, 00, tzinfo=eastern)
>>> loc_dt.strftime(fmt)
'2002-10-27 01:30:00 EST-0500'

As you can see, the system has chosen one for you and there is a 50%
chance of it being out by one hour. For some applications, this does
not matter. However, if you are trying to schedule meetings with people
in different timezones or analyze log files it is not acceptable. 

The best and simplest solution is to stick with using UTC.  The pytz package
encourages using UTC for internal timezone representation by including a
special UTC implementation based on the standard Python reference 
implementation in the Python documentation.  This timezone unpickles to be
the same instance, and pickles to a relatively small size.  The UTC 
implementation can be obtained as pytz.utc, pytz.UTC, or 
pytz.timezone('UTC').  Note that this instance is not the same 
instance (or implementation) as other timezones with the same meaning 
(GMT, Greenwich, Universal, etc.).

>>> import pickle, pytz
>>> dt = datetime(2005, 3, 1, 14, 13, 21, tzinfo=utc)
>>> naive = dt.replace(tzinfo=None)
>>> p = pickle.dumps(dt, 1)
>>> naive_p = pickle.dumps(naive, 1)
>>> len(p), len(naive_p), len(p) - len(naive_p)
(60, 43, 17)
>>> new = pickle.loads(p)
>>> new == dt
True
>>> new is dt
False
>>> new.tzinfo is dt.tzinfo
True
>>> pytz.utc is pytz.UTC is pytz.timezone('UTC')
True
>>> utc is pytz.timezone('GMT')
False

If you insist on working with local times, this library provides a
facility for constructing them almost unambiguously.

>>> loc_dt = datetime(2002, 10, 27, 1, 30, 00)
>>> est_dt = eastern.localize(loc_dt, is_dst=True)
>>> edt_dt = eastern.localize(loc_dt, is_dst=False)
>>> print est_dt.strftime(fmt), '/', edt_dt.strftime(fmt)
2002-10-27 01:30:00 EDT-0400 / 2002-10-27 01:30:00 EST-0500

Note that although this handles many cases, it is still not possible
to handle all. In cases where countries change their timezone definitions,
cases like the end-of-daylight-savings-time occur with no way of resolving
the ambiguity. For example, in 1915 Warsaw switched from Warsaw time to
Central European time. So at the stroke of midnight on August 4th 1915
the clocks were wound back 24 minutes creating a ambiguous time period
that cannot be specified without referring to the timezone abbreviation
or the actual UTC offset.

The 'Standard' Python way of handling all these ambiguities is not to,
such as demonstrated in this example using the US/Eastern timezone
definition from the Python documentation (Note that this implementation
only works for dates between 1987 and 2006 - it is included for tests only!):

>>> from pytz.reference import Eastern # pytz.reference only for tests
>>> dt = datetime(2002, 10, 27, 0, 30, tzinfo=Eastern)
>>> str(dt)
'2002-10-27 00:30:00-04:00'
>>> str(dt + timedelta(hours=1))
'2002-10-27 01:30:00-05:00'
>>> str(dt + timedelta(hours=2))
'2002-10-27 02:30:00-05:00'
>>> str(dt + timedelta(hours=3))
'2002-10-27 03:30:00-05:00'

Notice the first two results? At first glance you might think they are
correct, but taking the UTC offset into account you find that they are
actually two hours appart instead of the 1 hour we asked for.

>>> from pytz.reference import UTC # pytz.reference only for tests
>>> str(dt.astimezone(UTC))
'2002-10-27 04:30:00+00:00'
>>> str((dt + timedelta(hours=1)).astimezone(UTC))
'2002-10-27 06:30:00+00:00'


What is UTC
~~~~~~~~~~~

`UTC` is Universal Time, formerly known as Greenwich Mean Time or GMT.
All other timezones are given as offsets from UTC. No daylight savings
time occurs in UTC, making it a useful timezone to perform date arithmetic
without worrying about the confusion and ambiguities caused by daylight
savings time transitions, your country changing its timezone, or mobile
computers that move roam through multiple timezones.


Helpers
~~~~~~~

There are two lists of timezones provided.

`all_timezones` is the exhaustive list of the timezone names that can be used.

>>> from pytz import all_timezones
>>> len(all_timezones) >= 500
True
>>> 'Etc/Greenwich' in all_timezones
True

`common_timezones` is a list of useful, current timezones. It doesn't
contain deprecated zones or historical zones. It is also a sequence of
strings.

>>> from pytz import common_timezones
>>> len(common_timezones) < len(all_timezones)
True
>>> 'Etc/Greenwich' in common_timezones
False

You can also retrieve lists of timezones used by particular countries
using the `country_timezones()` method. It requires an ISO-3166 two letter
country code.

>>> from pytz import country_timezones
>>> country_timezones('ch')
['Europe/Zurich']
>>> country_timezones('CH')
['Europe/Zurich']

License
~~~~~~~

MIT license.

This code is also available as part of Zope 3 under the Zope Public
License,  Version 2.1 (ZPL).

I'm happy to relicense this code if necessary for inclusion in other
open source projects.

Latest Versions
~~~~~~~~~~~~~~~

This package will be updated after releases of the Olson timezone database.
The latest version can be downloaded from the Python Cheeseshop_ or
Sourceforge_. The code that is used to generate this distribution is
available using the Bazaar_ revision control system using::

    bzr branch http://bazaar.launchpad.net/~stub/pytz/devel

.. _Cheeseshop: http://cheeseshop.python.org/pypi/pytz/
.. _Sourceforge: http://sourceforge.net/projects/pytz/
.. _Bazaar: http://bazaar-vcs.org/

Bugs, Feature Requests & Patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bugs can be reported using Launchpad at
https://bugs.launchpad.net/products/pytz

Issues & Limitations
~~~~~~~~~~~~~~~~~~~~

- Offsets from UTC are rounded to the nearest whole minute, so timezones
  such as Europe/Amsterdam pre 1937 will be up to 30 seconds out. This is 
  a limitation of the Python datetime library.

- If you think a timezone definition is incorrect, I probably can't fix
  it. pytz is a direct translation of the Olson timezone database, and
  changes to the timezone definitions need to be made to this source.
  If you find errors they should be reported to the time zone mailing
  list, linked from http://www.twinsun.com/tz/tz-link.htm

Further Reading
~~~~~~~~~~~~~~~

More info than you want to know about timezones:
http://www.twinsun.com/tz/tz-link.htm


Contact
~~~~~~~

Stuart Bishop <stuart@stuartbishop.net>

.. [*]  The missing few are for Riyadh Solar Time in 1987, 1988 and 1989.
	As Saudi Arabia gave up trying to cope with their timezone
	definition, I see no reason to complicate my code further
	to cope with them.  (I understand the intention was to set
	sunset to 0:00 local time, the start of the Islamic day.
	In the best case caused the DST offset to change daily and
	worst case caused the DST offset to change each instant
	depending on how you interpreted the ruling.)



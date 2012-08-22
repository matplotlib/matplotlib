## This file is in the moin format. The latest version is found
## at https://moin.conectiva.com.br/DateUtil

== Contents ==
[[TableOfContents]]

== Description ==
The '''dateutil''' module provides powerful extensions to
the standard '''datetime''' module, available in Python 2.3+.

== Features ==

  * Computing of relative deltas (next month, next year,
  next monday, last week of month, etc);

  * Computing of relative deltas between two given
  date and/or datetime objects;

  * Computing of dates based on very flexible recurrence rules,
  using a superset of the
  [ftp://ftp.rfc-editor.org/in-notes/rfc2445.txt iCalendar]
  specification. Parsing of RFC strings is supported as well.

  * Generic parsing of dates in almost any string format;

  * Timezone (tzinfo) implementations for tzfile(5) format
  files (/etc/localtime, /usr/share/zoneinfo, etc), TZ
  environment string (in all known formats), iCalendar
  format files, given ranges (with help from relative deltas),
  local machine timezone, fixed offset timezone, UTC timezone,
  and Windows registry-based time zones.

  * Internal up-to-date world timezone information based on
  Olson's database.

  * Computing of Easter Sunday dates for any given year,
  using Western, Orthodox or Julian algorithms;

  * More than 400 test cases.

== Quick example ==
Here's a snapshot, just to give an idea about the power of the
package. For more examples, look at the documentation below.

Suppose you want to know how much time is left, in
years/months/days/etc, before the next easter happening on a
year with a Friday 13th in August, and you want to get today's
date out of the "date" unix system command. Here is the code:
{{{
from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
import commands
import os
now = parse(commands.getoutput("date"))
today = now.date()
year = rrule(YEARLY,bymonth=8,bymonthday=13,byweekday=FR)[0].year
rdelta = relativedelta(easter(year), today)
print "Today is:", today
print "Year with next Aug 13th on a Friday is:", year
print "How far is the Easter of that year:", rdelta
print "And the Easter of that year is:", today+rdelta
}}}

And here's the output:
{{{
Today is: 2003-10-11
Year with next Aug 13th on a Friday is: 2004
How far is the Easter of that year: relativedelta(months=+6)
And the Easter of that year is: 2004-04-11
}}}

{i} Being exactly 6 months ahead was '''really''' a coincidence :)

== Download ==
The following files are available.
  * attachment:python-dateutil-1.0.tar.bz2
  * attachment:python-dateutil-1.0-1.noarch.rpm

== Author ==
The dateutil module was written by GustavoNiemeyer <gustavo@niemeyer.net>.

== Documentation ==
The following modules are available.

=== relativedelta ===
This module offers the '''relativedelta''' type, which is based 
on the specification of the excelent work done by M.-A. Lemburg in his
[http://www.egenix.com/files/python/mxDateTime.html mxDateTime]
extension. However, notice that this type '''does not''' implement the
same algorithm as his work. Do not expect it to behave like
{{{mxDateTime}}}'s counterpart.

==== relativedelta type ====

There's two different ways to build a relativedelta instance. The
first one is passing it two {{{date}}}/{{{datetime}}} instances:
{{{
relativedelta(datetime1, datetime2)
}}}

This will build the relative difference between {{{datetime1}}} and
{{{datetime2}}}, so that the following constraint is always true:
{{{
datetime2+relativedelta(datetime1, datetime2) == datetime1
}}}

Notice that instead of {{{datetime}}} instances, you may use
{{{date}}} instances, or a mix of both.

And the other way is to use any of the following keyword arguments:

  year, month, day, hour, minute, second, microsecond::
  Absolute information.

  years, months, weeks, days, hours, minutes, seconds, microseconds::
  Relative information, may be negative.

  weekday::
  One of the weekday instances ({{{MO}}}, {{{TU}}}, etc). These
  instances may receive a parameter {{{n}}}, specifying the {{{n}}}th
  weekday, which could be positive or negative (like {{{MO(+2)}}} or
  {{{MO(-3)}}}. Not specifying it is the same as specifying {{{+1}}}.
  You can also use an integer, where {{{0=MO}}}. Notice that,
  for example, if the calculated date is already Monday, using
  {{{MO}}} or {{{MO(+1)}}} (which is the same thing in this context),
  won't change the day.

  leapdays::
  Will add given days to the date found, but only if the computed
  year is a leap year and the computed date is post 28 of february.

  yearday, nlyearday::
  Set the yearday or the non-leap year day (jump leap days).
  These are converted to {{{day}}}/{{{month}}}/{{{leapdays}}}
  information.

==== Behavior of operations ====
If you're curious about exactly how the relative delta will act
on operations, here is a description of its behavior.

  1. Calculate the absolute year, using the {{{year}}} argument, or the
  original datetime year, if the argument is not present.
  1. Add the relative {{{years}}} argument to the absolute year.
  1. Do steps 1 and 2 for {{{month}}}/{{{months}}}.
  1. Calculate the absolute day, using the {{{day}}} argument, or the
  original datetime day, if the argument is not present. Then, subtract
  from the day until it fits in the year and month found after their
  operations.
  1. Add the relative {{{days}}} argument to the absolute day. Notice
  that the {{{weeks}}} argument is multiplied by 7 and added to {{{days}}}.
  1. If {{{leapdays}}} is present, the computed year is a leap year, and
  the computed month is after february, remove one day from the found date.
  1. Do steps 1 and 2 for {{{hour}}}/{{{hours}}}, {{{minute}}}/{{{minutes}}},
  {{{second}}}/{{{seconds}}}, {{{microsecond}}}/{{{microseconds}}}.
  1. If the {{{weekday}}} argument is present, calculate the {{{n}}}th
  occurrence of the given weekday.

==== Examples ====

Let's begin our trip.
{{{
>>> from datetime import *; from dateutil.relativedelta import *
>>> import calendar
}}}

Store some values.
{{{
>>> NOW = datetime.now()
>>> TODAY = date.today()
>>> NOW
datetime.datetime(2003, 9, 17, 20, 54, 47, 282310)
>>> TODAY
datetime.date(2003, 9, 17)
}}}

Next month.
{{{
>>> NOW+relativedelta(months=+1)
datetime.datetime(2003, 10, 17, 20, 54, 47, 282310)
}}}

Next month, plus one week.
{{{
>>> NOW+relativedelta(months=+1, weeks=+1)
datetime.datetime(2003, 10, 24, 20, 54, 47, 282310)
}}}

Next month, plus one week, at 10am.
{{{
>>> TODAY+relativedelta(months=+1, weeks=+1, hour=10)
datetime.datetime(2003, 10, 24, 10, 0)
}}}

Let's try the other way around. Notice that the
hour setting we get in the relativedelta is relative,
since it's a difference, and the weeks parameter
has gone.
{{{
>>> relativedelta(datetime(2003, 10, 24, 10, 0), TODAY)
relativedelta(months=+1, days=+7, hours=+10)
}}}

One month before one year.
{{{
>>> NOW+relativedelta(years=+1, months=-1)
datetime.datetime(2004, 8, 17, 20, 54, 47, 282310)
}}}

How does it handle months with different numbers of days?
Notice that adding one month will never cross the month
boundary.
{{{
>>> date(2003,1,27)+relativedelta(months=+1)
datetime.date(2003, 2, 27)
>>> date(2003,1,31)+relativedelta(months=+1)
datetime.date(2003, 2, 28)
>>> date(2003,1,31)+relativedelta(months=+2)
datetime.date(2003, 3, 31)
}}}

The logic for years is the same, even on leap years.
{{{
>>> date(2000,2,28)+relativedelta(years=+1)
datetime.date(2001, 2, 28)
>>> date(2000,2,29)+relativedelta(years=+1)
datetime.date(2001, 2, 28)

>>> date(1999,2,28)+relativedelta(years=+1)
datetime.date(2000, 2, 28)
>>> date(1999,3,1)+relativedelta(years=+1)
datetime.date(2000, 3, 1)

>>> date(2001,2,28)+relativedelta(years=-1)
datetime.date(2000, 2, 28)
>>> date(2001,3,1)+relativedelta(years=-1)
datetime.date(2000, 3, 1)
}}}

Next friday.
{{{
>>> TODAY+relativedelta(weekday=FR)
datetime.date(2003, 9, 19)

>>> TODAY+relativedelta(weekday=calendar.FRIDAY)
datetime.date(2003, 9, 19)
}}}

Last friday in this month.
{{{
>>> TODAY+relativedelta(day=31, weekday=FR(-1))
datetime.date(2003, 9, 26)
}}}

Next wednesday (it's today!).
{{{
>>> TODAY+relativedelta(weekday=WE(+1))
datetime.date(2003, 9, 17)
}}}

Next wednesday, but not today.
{{{
>>> TODAY+relativedelta(days=+1, weekday=WE(+1))
datetime.date(2003, 9, 24)
}}}

Following
[http://www.cl.cam.ac.uk/~mgk25/iso-time.html ISO year week number notation]
find the first day of the 15th week of 1997.
{{{
>>> datetime(1997,1,1)+relativedelta(day=4, weekday=MO(-1), weeks=+14)
datetime.datetime(1997, 4, 7, 0, 0)
}}}

How long ago has the millennium changed?
{{{
>>> relativedelta(NOW, date(2001,1,1))
relativedelta(years=+2, months=+8, days=+16,
	      hours=+20, minutes=+54, seconds=+47, microseconds=+282310)
}}}

How old is John?
{{{
>>> johnbirthday = datetime(1978, 4, 5, 12, 0)
>>> relativedelta(NOW, johnbirthday)
relativedelta(years=+25, months=+5, days=+12,
	      hours=+8, minutes=+54, seconds=+47, microseconds=+282310)
}}}

It works with dates too.
{{{
>>> relativedelta(TODAY, johnbirthday)
relativedelta(years=+25, months=+5, days=+11, hours=+12)
}}}

Obtain today's date using the yearday:
{{{
>>> date(2003, 1, 1)+relativedelta(yearday=260)
datetime.date(2003, 9, 17)
}}}

We can use today's date, since yearday should be absolute
in the given year:
{{{
>>> TODAY+relativedelta(yearday=260)
datetime.date(2003, 9, 17)
}}}

Last year it should be in the same day:
{{{
>>> date(2002, 1, 1)+relativedelta(yearday=260)
datetime.date(2002, 9, 17)
}}}

But not in a leap year:
{{{
>>> date(2000, 1, 1)+relativedelta(yearday=260)
datetime.date(2000, 9, 16)
}}}

We can use the non-leap year day to ignore this:
{{{
>>> date(2000, 1, 1)+relativedelta(nlyearday=260)
datetime.date(2000, 9, 17)
}}}

=== rrule ===
The rrule module offers a small, complete, and very fast, implementation
of the recurrence rules documented in the 
[ftp://ftp.rfc-editor.org/in-notes/rfc2445.txt iCalendar RFC], including
support for caching of results.

==== rrule type ====
That's the base of the rrule operation. It accepts all the keywords
defined in the RFC as its constructor parameters (except {{{byday}}},
which was renamed to {{{byweekday}}}) and more. The constructor
prototype is:
{{{
rrule(freq)
}}}

Where {{{freq}}} must be one of {{{YEARLY}}}, {{{MONTHLY}}},
{{{WEEKLY}}}, {{{DAILY}}}, {{{HOURLY}}}, {{{MINUTELY}}},
or {{{SECONDLY}}}.

Additionally, it supports the following keyword arguments:

    cache::
    If given, it must be a boolean value specifying to enable
    or disable caching of results. If you will use the same
    {{{rrule}}} instance multiple times, enabling caching will
    improve the performance considerably.

    dtstart::
    The recurrence start. Besides being the base for the
    recurrence, missing parameters in the final recurrence
    instances will also be extracted from this date. If not
    given, {{{datetime.now()}}} will be used instead.

    interval::
    The interval between each {{{freq}}} iteration. For example,
    when using {{{YEARLY}}}, an interval of {{{2}}} means
    once every two years, but with {{{HOURLY}}}, it means
    once every two hours. The default interval is {{{1}}}.

    wkst::
    The week start day. Must be one of the {{{MO}}}, {{{TU}}},
    {{{WE}}} constants, or an integer, specifying the first day
    of the week. This will affect recurrences based on weekly
    periods. The default week start is got from
    {{{calendar.firstweekday()}}}, and may be modified by
    {{{calendar.setfirstweekday()}}}.

    count::
    How many occurrences will be generated.

    until::
    If given, this must be a {{{datetime}}} instance, that will
    specify the limit of the recurrence. If a recurrence instance
    happens to be the same as the {{{datetime}}} instance given
    in the {{{until}}} keyword, this will be the last occurrence.

    bysetpos::
    If given, it must be either an integer, or a sequence of
    integers, positive or negative. Each given integer will
    specify an occurrence number, corresponding to the nth
    occurrence of the rule inside the frequency period. For
    example, a {{{bysetpos}}} of {{{-1}}} if combined with a
    {{{MONTHLY}}} frequency, and a {{{byweekday}}} of
    {{{(MO, TU, WE, TH, FR)}}}, will result in the last work
    day of every month.

    bymonth::
    If given, it must be either an integer, or a sequence of
    integers, meaning the months to apply the recurrence to.

    bymonthday::
    If given, it must be either an integer, or a sequence of
    integers, meaning the month days to apply the recurrence to.

    byyearday::
    If given, it must be either an integer, or a sequence of
    integers, meaning the year days to apply the recurrence to.

    byweekno::
    If given, it must be either an integer, or a sequence of
    integers, meaning the week numbers to apply the recurrence
    to. Week numbers have the meaning described in ISO8601,
    that is, the first week of the year is that containing at
    least four days of the new year.
    
    byweekday::
    If given, it must be either an integer ({{{0 == MO}}}), a
    sequence of integers, one of the weekday constants
    ({{{MO}}}, {{{TU}}}, etc), or a sequence of these constants.
    When given, these variables will define the weekdays where
    the recurrence will be applied. It's also possible to use
    an argument {{{n}}} for the weekday instances, which will
    mean the {{{n}}}''th'' occurrence of this weekday in the
    period. For example, with {{{MONTHLY}}}, or with
    {{{YEARLY}}} and {{{BYMONTH}}}, using {{{FR(+1)}}}
    in {{{byweekday}}} will specify the first friday of the
    month where the recurrence happens. Notice that in the RFC
    documentation, this is specified as {{{BYDAY}}}, but was
    renamed to avoid the ambiguity of that keyword.

    byhour::
    If given, it must be either an integer, or a sequence of
    integers, meaning the hours to apply the recurrence to.

    byminute::
    If given, it must be either an integer, or a sequence of
    integers, meaning the minutes to apply the recurrence to.

    bysecond::
    If given, it must be either an integer, or a sequence of
    integers, meaning the seconds to apply the recurrence to.

    byeaster::
    If given, it must be either an integer, or a sequence of
    integers, positive or negative. Each integer will define
    an offset from the Easter Sunday. Passing the offset
    {{{0}}} to {{{byeaster}}} will yield the Easter Sunday
    itself. This is an extension to the RFC specification.

==== rrule methods ====
The following methods are available in {{{rrule}}} instances:

    rrule.before(dt, inc=False)::
    Returns the last recurrence before the given {{{datetime}}}
    instance. The {{{inc}}} keyword defines what happens if
    {{{dt}}} '''is''' an occurrence. With {{{inc == True}}},
    if {{{dt}}} itself is an occurrence, it will be returned.

    rrule.after(dt, inc=False)::
    Returns the first recurrence after the given {{{datetime}}}
    instance. The {{{inc}}} keyword defines what happens if
    {{{dt}}} '''is''' an occurrence. With {{{inc == True}}},
    if {{{dt}}} itself is an occurrence, it will be returned.

    rrule.between(after, before, inc=False)::
    Returns all the occurrences of the rrule between {{{after}}}
    and {{{before}}}. The {{{inc}}} keyword defines what happens
    if {{{after}}} and/or {{{before}}} are themselves occurrences.
    With {{{inc == True}}}, they will be included in the list,
    if they are found in the recurrence set.

    rrule.count()::
    Returns the number of recurrences in this set. It will have
    go trough the whole recurrence, if this hasn't been done
    before.

Besides these methods, {{{rrule}}} instances also support
the {{{__getitem__()}}} and {{{__contains__()}}} special methods,
meaning that these are valid expressions:
{{{
rr = rrule(...)
if datetime(...) in rr:
    ...
print rr[0]
print rr[-1]
print rr[1:2]
print rr[::-2]
}}}

The getitem/slicing mechanism is smart enough to avoid getting the whole
recurrence set, if possible.

==== Notes ====

  * The rrule type has no {{{byday}}} keyword. The equivalent keyword
  has been replaced by the {{{byweekday}}} keyword, to remove the
  ambiguity present in the original keyword.

  * Unlike documented in the RFC, the starting datetime ({{{dtstart}}})
  is not the first recurrence instance, unless it does fit in the
  specified rules. In a python module context, this behavior makes more
  sense than otherwise. Notice that you can easily get the original
  behavior by using a rruleset and adding the {{{dtstart}}} as an
  {{{rdate}}} recurrence.

  * Unlike documented in the RFC, every keyword is valid on every
  frequency (the RFC documents that {{{byweekno}}} is only valid
  on yearly frequencies, for example).

  * In addition to the documented keywords, a {{{byeaster}}} keyword
  was introduced, making it easy to compute recurrent events relative
  to the Easter Sunday.

==== rrule examples ====
These examples were converted from the RFC.

Prepare the environment.
{{{
>>> from dateutil.rrule import *
>>> from dateutil.parser import *
>>> from datetime import *

>>> import pprint
>>> import sys
>>> sys.displayhook = pprint.pprint
}}}

Daily, for 10 occurrences.
{{{
>>> list(rrule(DAILY, count=10,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 3, 9, 0),
 datetime.datetime(1997, 9, 4, 9, 0),
 datetime.datetime(1997, 9, 5, 9, 0),
 datetime.datetime(1997, 9, 6, 9, 0),
 datetime.datetime(1997, 9, 7, 9, 0),
 datetime.datetime(1997, 9, 8, 9, 0),
 datetime.datetime(1997, 9, 9, 9, 0),
 datetime.datetime(1997, 9, 10, 9, 0),
 datetime.datetime(1997, 9, 11, 9, 0)]
}}}

Daily until December 24, 1997
{{{
>>> list(rrule(DAILY,
	       dtstart=parse("19970902T090000"),
	       until=parse("19971224T000000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 3, 9, 0),
 datetime.datetime(1997, 9, 4, 9, 0),
 (...)
 datetime.datetime(1997, 12, 21, 9, 0),
 datetime.datetime(1997, 12, 22, 9, 0),
 datetime.datetime(1997, 12, 23, 9, 0)]
}}}

Every other day, 5 occurrences.
{{{
>>> list(rrule(DAILY, interval=2, count=5,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 4, 9, 0),
 datetime.datetime(1997, 9, 6, 9, 0),
 datetime.datetime(1997, 9, 8, 9, 0),
 datetime.datetime(1997, 9, 10, 9, 0)]
}}}

Every 10 days, 5 occurrences.
{{{
>>> list(rrule(DAILY, interval=10, count=5,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 12, 9, 0),
 datetime.datetime(1997, 9, 22, 9, 0),
 datetime.datetime(1997, 10, 2, 9, 0),
 datetime.datetime(1997, 10, 12, 9, 0)]
}}}

Everyday in January, for 3 years.
{{{
>>> list(rrule(YEARLY, bymonth=1, byweekday=range(7),
	       dtstart=parse("19980101T090000"),
	       until=parse("20000131T090000")))
[datetime.datetime(1998, 1, 1, 9, 0),
 datetime.datetime(1998, 1, 2, 9, 0),
 (...)
 datetime.datetime(1998, 1, 30, 9, 0),
 datetime.datetime(1998, 1, 31, 9, 0),
 datetime.datetime(1999, 1, 1, 9, 0),
 datetime.datetime(1999, 1, 2, 9, 0),
 (...)
 datetime.datetime(1999, 1, 30, 9, 0),
 datetime.datetime(1999, 1, 31, 9, 0),
 datetime.datetime(2000, 1, 1, 9, 0),
 datetime.datetime(2000, 1, 2, 9, 0),
 (...)
 datetime.datetime(2000, 1, 29, 9, 0),
 datetime.datetime(2000, 1, 31, 9, 0)]
}}} 

Same thing, in another way.
{{{
>>> list(rrule(DAILY, bymonth=1,
               dtstart=parse("19980101T090000"),
	       until=parse("20000131T090000")))
(...)
}}}

Weekly for 10 occurrences.
{{{
>>> list(rrule(WEEKLY, count=10,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 9, 9, 0),
 datetime.datetime(1997, 9, 16, 9, 0),
 datetime.datetime(1997, 9, 23, 9, 0),
 datetime.datetime(1997, 9, 30, 9, 0),
 datetime.datetime(1997, 10, 7, 9, 0),
 datetime.datetime(1997, 10, 14, 9, 0),
 datetime.datetime(1997, 10, 21, 9, 0),
 datetime.datetime(1997, 10, 28, 9, 0),
 datetime.datetime(1997, 11, 4, 9, 0)]
}}}

Every other week, 6 occurrences.
{{{
>>> list(rrule(WEEKLY, interval=2, count=6,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 16, 9, 0),
 datetime.datetime(1997, 9, 30, 9, 0),
 datetime.datetime(1997, 10, 14, 9, 0),
 datetime.datetime(1997, 10, 28, 9, 0),
 datetime.datetime(1997, 11, 11, 9, 0)]
}}}

Weekly on Tuesday and Thursday for 5 weeks.
{{{
>>> list(rrule(WEEKLY, count=10, wkst=SU, byweekday=(TU,TH),
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 4, 9, 0),
 datetime.datetime(1997, 9, 9, 9, 0),
 datetime.datetime(1997, 9, 11, 9, 0),
 datetime.datetime(1997, 9, 16, 9, 0),
 datetime.datetime(1997, 9, 18, 9, 0),
 datetime.datetime(1997, 9, 23, 9, 0),
 datetime.datetime(1997, 9, 25, 9, 0),
 datetime.datetime(1997, 9, 30, 9, 0),
 datetime.datetime(1997, 10, 2, 9, 0)]
}}}

Every other week on Tuesday and Thursday, for 8 occurrences.
{{{
>>> list(rrule(WEEKLY, interval=2, count=8,
	       wkst=SU, byweekday=(TU,TH),
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 4, 9, 0),
 datetime.datetime(1997, 9, 16, 9, 0),
 datetime.datetime(1997, 9, 18, 9, 0),
 datetime.datetime(1997, 9, 30, 9, 0),
 datetime.datetime(1997, 10, 2, 9, 0),
 datetime.datetime(1997, 10, 14, 9, 0),
 datetime.datetime(1997, 10, 16, 9, 0)]
}}}

Monthly on the 1st Friday for ten occurrences.
{{{
>>> list(rrule(MONTHLY, count=10, byweekday=FR(1),
	       dtstart=parse("19970905T090000")))
[datetime.datetime(1997, 9, 5, 9, 0),
 datetime.datetime(1997, 10, 3, 9, 0),
 datetime.datetime(1997, 11, 7, 9, 0),
 datetime.datetime(1997, 12, 5, 9, 0),
 datetime.datetime(1998, 1, 2, 9, 0),
 datetime.datetime(1998, 2, 6, 9, 0),
 datetime.datetime(1998, 3, 6, 9, 0),
 datetime.datetime(1998, 4, 3, 9, 0),
 datetime.datetime(1998, 5, 1, 9, 0),
 datetime.datetime(1998, 6, 5, 9, 0)]
}}}

Every other month on the 1st and last Sunday of the month for 10 occurrences.
{{{
>>> list(rrule(MONTHLY, interval=2, count=10,
	       byweekday=(SU(1), SU(-1)),
	       dtstart=parse("19970907T090000")))
[datetime.datetime(1997, 9, 7, 9, 0),
 datetime.datetime(1997, 9, 28, 9, 0),
 datetime.datetime(1997, 11, 2, 9, 0),
 datetime.datetime(1997, 11, 30, 9, 0),
 datetime.datetime(1998, 1, 4, 9, 0),
 datetime.datetime(1998, 1, 25, 9, 0),
 datetime.datetime(1998, 3, 1, 9, 0),
 datetime.datetime(1998, 3, 29, 9, 0),
 datetime.datetime(1998, 5, 3, 9, 0),
 datetime.datetime(1998, 5, 31, 9, 0)]
}}}

Monthly on the second to last Monday of the month for 6 months.
{{{
>>> list(rrule(MONTHLY, count=6, byweekday=MO(-2),
	       dtstart=parse("19970922T090000")))
[datetime.datetime(1997, 9, 22, 9, 0),
 datetime.datetime(1997, 10, 20, 9, 0),
 datetime.datetime(1997, 11, 17, 9, 0),
 datetime.datetime(1997, 12, 22, 9, 0),
 datetime.datetime(1998, 1, 19, 9, 0),
 datetime.datetime(1998, 2, 16, 9, 0)]
}}}

Monthly on the third to the last day of the month, for 6 months.
{{{
>>> list(rrule(MONTHLY, count=6, bymonthday=-3,
	       dtstart=parse("19970928T090000")))
[datetime.datetime(1997, 9, 28, 9, 0),
 datetime.datetime(1997, 10, 29, 9, 0),
 datetime.datetime(1997, 11, 28, 9, 0),
 datetime.datetime(1997, 12, 29, 9, 0),
 datetime.datetime(1998, 1, 29, 9, 0),
 datetime.datetime(1998, 2, 26, 9, 0)]
}}}

Monthly on the 2nd and 15th of the month for 5 occurrences.
{{{
>>> list(rrule(MONTHLY, count=5, bymonthday=(2,15),
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 15, 9, 0),
 datetime.datetime(1997, 10, 2, 9, 0),
 datetime.datetime(1997, 10, 15, 9, 0),
 datetime.datetime(1997, 11, 2, 9, 0)]
}}}

Monthly on the first and last day of the month for 3 occurrences.
{{{
>>> list(rrule(MONTHLY, count=5, bymonthday=(-1,1,),
               dtstart=parse("1997090
2T090000")))
[datetime.datetime(1997, 9, 30, 9, 0),
 datetime.datetime(1997, 10, 1, 9, 0),
 datetime.datetime(1997, 10, 31, 9, 0),
 datetime.datetime(1997, 11, 1, 9, 0),
 datetime.datetime(1997, 11, 30, 9, 0)]
}}}

Every 18 months on the 10th thru 15th of the month for 10 occurrences.
{{{
>>> list(rrule(MONTHLY, interval=18, count=10,
	       bymonthday=range(10,16),
	       dtstart=parse("19970910T090000")))
[datetime.datetime(1997, 9, 10, 9, 0),
 datetime.datetime(1997, 9, 11, 9, 0),
 datetime.datetime(1997, 9, 12, 9, 0),
 datetime.datetime(1997, 9, 13, 9, 0),
 datetime.datetime(1997, 9, 14, 9, 0),
 datetime.datetime(1997, 9, 15, 9, 0),
 datetime.datetime(1999, 3, 10, 9, 0),
 datetime.datetime(1999, 3, 11, 9, 0),
 datetime.datetime(1999, 3, 12, 9, 0),
 datetime.datetime(1999, 3, 13, 9, 0)]
}}}

Every Tuesday, every other month, 6 occurences.
{{{
>>> list(rrule(MONTHLY, interval=2, count=6, byweekday=TU,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 9, 9, 0),
 datetime.datetime(1997, 9, 16, 9, 0),
 datetime.datetime(1997, 9, 23, 9, 0),
 datetime.datetime(1997, 9, 30, 9, 0),
 datetime.datetime(1997, 11, 4, 9, 0)]
}}}

Yearly in June and July for 10 occurrences.
{{{
>>> list(rrule(YEARLY, count=4, bymonth=(6,7),
	       dtstart=parse("19970610T0900
00")))
[datetime.datetime(1997, 6, 10, 9, 0),
 datetime.datetime(1997, 7, 10, 9, 0),
 datetime.datetime(1998, 6, 10, 9, 0),
 datetime.datetime(1998, 7, 10, 9, 0)]
}}}

Every 3rd year on the 1st, 100th and 200th day for 4 occurrences.
{{{
>>> list(rrule(YEARLY, count=4, interval=3, byyearday=(1,100,200),
	       dtstart=parse("19970101T090000")))
[datetime.datetime(1997, 1, 1, 9, 0),
 datetime.datetime(1997, 4, 10, 9, 0),
 datetime.datetime(1997, 7, 19, 9, 0),
 datetime.datetime(2000, 1, 1, 9, 0)]
}}}

Every 20th Monday of the year, 3 occurrences.
{{{
>>> list(rrule(YEARLY, count=3, byweekday=MO(20),
	       dtstart=parse("19970519T090000")))
[datetime.datetime(1997, 5, 19, 9, 0),
 datetime.datetime(1998, 5, 18, 9, 0),
 datetime.datetime(1999, 5, 17, 9, 0)]
}}}

Monday of week number 20 (where the default start of the week is Monday),
3 occurrences.
{{{
>>> list(rrule(YEARLY, count=3, byweekno=20, byweekday=MO,
	       dtstart=parse("19970512T090000")))
[datetime.datetime(1997, 5, 12, 9, 0),
 datetime.datetime(1998, 5, 11, 9, 0),
 datetime.datetime(1999, 5, 17, 9, 0)]
}}}

The week number 1 may be in the last year.
{{{
>>> list(rrule(WEEKLY, count=3, byweekno=1, byweekday=MO,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 12, 29, 9, 0),
 datetime.datetime(1999, 1, 4, 9, 0),
 datetime.datetime(2000, 1, 3, 9, 0)]
}}}

And the week numbers greater than 51 may be in the next year.
{{{
>>> list(rrule(WEEKLY, count=3, byweekno=52, byweekday=SU,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 12, 28, 9, 0),
 datetime.datetime(1998, 12, 27, 9, 0),
 datetime.datetime(2000, 1, 2, 9, 0)]
}}}

Only some years have week number 53:
{{{
>>> list(rrule(WEEKLY, count=3, byweekno=53, byweekday=MO,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1998, 12, 28, 9, 0),
 datetime.datetime(2004, 12, 27, 9, 0),
 datetime.datetime(2009, 12, 28, 9, 0)]
}}}

Every Friday the 13th, 4 occurrences.
{{{
>>> list(rrule(YEARLY, count=4, byweekday=FR, bymonthday=13,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1998, 2, 13, 9, 0),
 datetime.datetime(1998, 3, 13, 9, 0),
 datetime.datetime(1998, 11, 13, 9, 0),
 datetime.datetime(1999, 8, 13, 9, 0)]
}}}

Every four years, the first Tuesday after a Monday in November,
3 occurrences (U.S. Presidential Election day):
{{{
>>> list(rrule(YEARLY, interval=4, count=3, bymonth=11,
	       byweekday=TU, bymonthday=(2,3,4,5,6,7,8),
	       dtstart=parse("19961105T090000")))
[datetime.datetime(1996, 11, 5, 9, 0),
 datetime.datetime(2000, 11, 7, 9, 0),
 datetime.datetime(2004, 11, 2, 9, 0)]
}}}

The 3rd instance into the month of one of Tuesday, Wednesday or
Thursday, for the next 3 months:
{{{
>>> list(rrule(MONTHLY, count=3, byweekday=(TU,WE,TH),
	       bysetpos=3, dtstart=parse("19970904T090000")))
[datetime.datetime(1997, 9, 4, 9, 0),
 datetime.datetime(1997, 10, 7, 9, 0),
 datetime.datetime(1997, 11, 6, 9, 0)]
}}}

The 2nd to last weekday of the month, 3 occurrences.
{{{
>>> list(rrule(MONTHLY, count=3, byweekday=(MO,TU,WE,TH,FR),
	       bysetpos=-2, dtstart=parse("19970929T090000")))
[datetime.datetime(1997, 9, 29, 9, 0),
 datetime.datetime(1997, 10, 30, 9, 0),
 datetime.datetime(1997, 11, 27, 9, 0)]
}}}

Every 3 hours from 9:00 AM to 5:00 PM on a specific day.
{{{
>>> list(rrule(HOURLY, interval=3,
	       dtstart=parse("19970902T090000"),
	       until=parse("19970902T170000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 2, 12, 0),
 datetime.datetime(1997, 9, 2, 15, 0)]
}}}

Every 15 minutes for 6 occurrences.
{{{
>>> list(rrule(MINUTELY, interval=15, count=6,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 2, 9, 15),
 datetime.datetime(1997, 9, 2, 9, 30),
 datetime.datetime(1997, 9, 2, 9, 45),
 datetime.datetime(1997, 9, 2, 10, 0),
 datetime.datetime(1997, 9, 2, 10, 15)]
}}}

Every hour and a half for 4 occurrences.
{{{
>>> list(rrule(MINUTELY, interval=90, count=4,
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 2, 10, 30),
 datetime.datetime(1997, 9, 2, 12, 0),
 datetime.datetime(1997, 9, 2, 13, 30)]
}}}

Every 20 minutes from 9:00 AM to 4:40 PM for two days.
{{{
>>> list(rrule(MINUTELY, interval=20, count=48,
	       byhour=range(9,17), byminute=(0,20,40),
	       dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 2, 9, 20),
 (...)
 datetime.datetime(1997, 9, 2, 16, 20),
 datetime.datetime(1997, 9, 2, 16, 40),
 datetime.datetime(1997, 9, 3, 9, 0),
 datetime.datetime(1997, 9, 3, 9, 20),
 (...)
 datetime.datetime(1997, 9, 3, 16, 20),
 datetime.datetime(1997, 9, 3, 16, 40)]
}}}

An example where the days generated makes a difference because of {{{wkst}}}.
{{{
>>> list(rrule(WEEKLY, interval=2, count=4,
	       byweekday=(TU,SU), wkst=MO,
	       dtstart=parse("19970805T090000")))
[datetime.datetime(1997, 8, 5, 9, 0),
 datetime.datetime(1997, 8, 10, 9, 0),
 datetime.datetime(1997, 8, 19, 9, 0),
 datetime.datetime(1997, 8, 24, 9, 0)]

>>> list(rrule(WEEKLY, interval=2, count=4,
	       byweekday=(TU,SU), wkst=SU,
	       dtstart=parse("19970805T090000")))
[datetime.datetime(1997, 8, 5, 9, 0),
 datetime.datetime(1997, 8, 17, 9, 0),
 datetime.datetime(1997, 8, 19, 9, 0),
 datetime.datetime(1997, 8, 31, 9, 0)]
}}}

==== rruleset type ====
The {{{rruleset}}} type allows more complex recurrence setups, mixing
multiple rules, dates, exclusion rules, and exclusion dates.
The type constructor takes the following keyword arguments:

    cache::
    If True, caching of results will be enabled, improving performance
    of multiple queries considerably.

==== rruleset methods ====
The following methods are available:

    rruleset.rrule(rrule)::
    Include the given {{{rrule}}} instance in the recurrence set
    generation.

    rruleset.rdate(dt)::
    Include the given {{{datetime}}} instance in the recurrence
    set generation.
    
    rruleset.exrule(rrule)::
    Include the given {{{rrule}}} instance in the recurrence set
    exclusion list. Dates which are part of the given recurrence
    rules will not be generated, even if some inclusive {{{rrule}}}
    or {{{rdate}}} matches them.

    rruleset.exdate(dt)::
    Include the given {{{datetime}}} instance in the recurrence set
    exclusion list. Dates included that way will not be generated,
    even if some inclusive {{{rrule}}} or {{{rdate}}} matches them.

    rruleset.before(dt, inc=False)::
    Returns the last recurrence before the given {{{datetime}}}
    instance. The {{{inc}}} keyword defines what happens if
    {{{dt}}} '''is''' an occurrence. With {{{inc == True}}},
    if {{{dt}}} itself is an occurrence, it will be returned.

    rruleset.after(dt, inc=False)::
    Returns the first recurrence after the given {{{datetime}}}
    instance. The {{{inc}}} keyword defines what happens if
    {{{dt}}} '''is''' an occurrence. With {{{inc == True}}},
    if {{{dt}}} itself is an occurrence, it will be returned.

    rruleset.between(after, before, inc=False)::
    Returns all the occurrences of the rrule between {{{after}}}
    and {{{before}}}. The {{{inc}}} keyword defines what happens
    if {{{after}}} and/or {{{before}}} are themselves occurrences.
    With {{{inc == True}}}, they will be included in the list,
    if they are found in the recurrence set.

    rruleset.count()::
    Returns the number of recurrences in this set. It will have
    go trough the whole recurrence, if this hasn't been done
    before.

Besides these methods, {{{rruleset}}} instances also support
the {{{__getitem__()}}} and {{{__contains__()}}} special methods,
meaning that these are valid expressions:
{{{
set = rruleset(...)
if datetime(...) in set:
    ...
print set[0]
print set[-1]
print set[1:2]
print set[::-2]
}}}

The getitem/slicing mechanism is smart enough to avoid getting the whole
recurrence set, if possible.

==== rruleset examples ====
Daily, for 7 days, jumping Saturday and Sunday occurrences.
{{{
>>> set = rruleset()
>>> set.rrule(rrule(DAILY, count=7,
		    dtstart=parse("19970902T090000")))
>>> set.exrule(rrule(YEARLY, byweekday=(SA,SU),
		     dtstart=parse("19970902T090000")))
>>> list(set)
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 3, 9, 0),
 datetime.datetime(1997, 9, 4, 9, 0),
 datetime.datetime(1997, 9, 5, 9, 0),
 datetime.datetime(1997, 9, 8, 9, 0)]
}}}

Weekly, for 4 weeks, plus one time on day 7, and not on day 16.
{{{
>>> set = rruleset()
>>> set.rrule(rrule(WEEKLY, count=4,
		    dtstart=parse("19970902T090000")))
>>> set.rdate(datetime.datetime(1997, 9, 7, 9, 0))
>>> set.exdate(datetime.datetime(1997, 9, 16, 9, 0))
>>> list(set)
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 7, 9, 0),
 datetime.datetime(1997, 9, 9, 9, 0),
 datetime.datetime(1997, 9, 23, 9, 0)]
}}}

==== rrulestr() function ====
The {{{rrulestr()}}} function is a parser for ''RFC-like'' syntaxes.
The function prototype is:
{{{
rrulestr(str)
}}}

The string passed as parameter may be a multiple line string, a
single line string, or just the {{{RRULE}}} property value.

Additionally, it accepts the following keyword arguments:

    cache::
    If {{{True}}}, the {{{rruleset}}} or {{{rrule}}} created instance
    will cache its results. Default is not to cache.

    dtstart::
    If given, it must be a {{{datetime}}} instance that will be used
    when no {{{DTSTART}}} property is found in the parsed string. If
    it is not given, and the property is not found, {{{datetime.now()}}}
    will be used instead.

    unfold::
    If set to {{{True}}}, lines will be unfolded following the RFC
    specification. It defaults to {{{False}}}, meaning that spaces
    before every line will be stripped.

    forceset::
    If set to {{{True}}} a {{{rruleset}}} instance will be returned,
    even if only a single rule is found. The default is to return an
    {{{rrule}}} if possible, and an {{{rruleset}}} if necessary.

    compatible::
    If set to {{{True}}}, the parser will operate in RFC-compatible
    mode. Right now it means that {{{unfold}}} will be turned on,
    and if a {{{DTSTART}}} is found, it will be considered the first     
    recurrence instance, as documented in the RFC.

    ignoretz::
    If set to {{{True}}}, the date parser will ignore timezone
    information available in the {{{DTSTART}}} property, or the
    {{{UNTIL}}} attribute.

    tzinfos::
    If set, it will be passed to the datetime string parser to
    resolve unknown timezone settings. For more information about
    what could be used here, check the parser documentation.

==== rrulestr() examples ====

Every 10 days, 5 occurrences.
{{{
>>> list(rrulestr("""
... DTSTART:19970902T090000
... RRULE:FREQ=DAILY;INTERVAL=10;COUNT=5
... """))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 12, 9, 0),
 datetime.datetime(1997, 9, 22, 9, 0),
 datetime.datetime(1997, 10, 2, 9, 0),
 datetime.datetime(1997, 10, 12, 9, 0)]
}}}

Same thing, but passing only the {{{RRULE}}} value.
{{{
>>> list(rrulestr("FREQ=DAILY;INTERVAL=10;COUNT=5",
		  dtstart=parse("19970902T090000")))
[datetime.datetime(1997, 9, 2, 9, 0),
 datetime.datetime(1997, 9, 12, 9, 0),
 datetime.datetime(1997, 9, 22, 9, 0),
 datetime.datetime(1997, 10, 2, 9, 0),
 datetime.datetime(1997, 10, 12, 9, 0)]
}}}

Notice that when using a single rule, it returns an
{{{rrule}}} instance, unless {{{forceset}}} was used.
{{{
>>> rrulestr("FREQ=DAILY;INTERVAL=10;COUNT=5")
<dateutil.rrule.rrule instance at 0x30269f08>

>>> rrulestr("""
... DTSTART:19970902T090000
... RRULE:FREQ=DAILY;INTERVAL=10;COUNT=5
... """)
<dateutil.rrule.rrule instance at 0x302699e0>

>>> rrulestr("FREQ=DAILY;INTERVAL=10;COUNT=5", forceset=True)
<dateutil.rrule.rruleset instance at 0x30269f08>
}}}

But when an {{{rruleset}}} is needed, it is automatically used.
{{{
>>> rrulestr("""
... DTSTART:19970902T090000
... RRULE:FREQ=DAILY;INTERVAL=10;COUNT=5
... RRULE:FREQ=DAILY;INTERVAL=5;COUNT=3
... """)
<dateutil.rrule.rruleset instance at 0x302699e0>
}}}

=== parser ===
This module offers a generic date/time string parser which is
able to parse most known formats to represent a date and/or
time.

==== parse() function ====
That's probably the only function you'll need from this module.
It offers you an interface to access the parser functionality and
extract a {{{datetime}}} type out of a string.

The prototype of this function is:
{{{
parse(timestr)
}}}

Additionally, the following keyword arguments are available:

    default::
    If given, this must be a {{{datetime}}} instance. Any fields
    missing in the parsed date will be copied from this instance.
    The default value is the current date, at 00:00:00am.

    ignoretz::
    If this is true, even if a timezone is found in the string,
    the parser will not use it.

    tzinfos::
    Using this keyword argument you may provide custom timezones
    to the parser. If given, it must be either a dictionary with
    the timezone abbreviation as key, or a function accepting a
    timezone abbreviation and offset as argument. The dictionary
    values and the function return must be a timezone offset
    in seconds, a tzinfo subclass, or a string defining the
    timezone (in the TZ environment variable format).

    dayfirst::
    This option allow one to change the precedence in which
    days are parsed in date strings. The default is given in the
    parserinfo instance (the default parserinfo has it set to
    False). If {{{dayfirst}}} is False, the {{{MM-DD-YYYY}}}
    format will have precedence over {{{DD-MM-YYYY}}} in an
    ambiguous date.

    yearfirst::
    This option allow one to change the precedence in which
    years are parsed in date strings. The default is given in
    the parserinfo instance (the default parserinfo has it set
    to False). If {{{yearfirst}}} is false, the {{{MM-DD-YY}}}
    format will have precedence over {{{YY-MM-DD}}} in an
    ambiguous date.

    fuzzy::
    If {{{fuzzy}}} is set to True, unknown tokens in the string
    will be ignored.

    parserinfo::
    This parameter allows one to change how the string is parsed,
    by using a different parserinfo class instance. Using it you
    may, for example, intenationalize the parser strings, or make
    it ignore additional words.

==== Format precedence ====
Whenever an ambiguous date is found, the {{{dayfirst}}} and
{{{yearfirst}}} parameters will control how the information
is processed. Here is the precedence in each case:

If {{{dayfirst}}} is {{{False}}} and {{{yearfirst}}} is {{{False}}},
(default, if no parameter is given):

    * {{{MM-DD-YY}}}
    * {{{DD-MM-YY}}}
    * {{{YY-MM-DD}}}

If {{{dayfirst}}} is {{{True}}} and {{{yearfirst}}} is {{{False}}}:

    * {{{DD-MM-YY}}}
    * {{{MM-DD-YY}}}
    * {{{YY-MM-DD}}}

If {{{dayfirst}}} is {{{False}}} and {{{yearfirst}}} is {{{True}}}:

    * {{{YY-MM-DD}}}
    * {{{MM-DD-YY}}}
    * {{{DD-MM-YY}}}

If {{{dayfirst}}} is {{{True}}} and {{{yearfirst}}} is {{{True}}}:

    * {{{YY-MM-DD}}}
    * {{{DD-MM-YY}}}
    * {{{MM-DD-YY}}}

==== Converting two digit years ====
When a two digit year is found, it is processed considering
the current year, so that the computed year is never more
than 49 years after the current year, nor 50 years before the
current year. In other words, if we are in year 2003, and the
year 30 is found, it will be considered as 2030, but if the
year 60 is found, it will be considered 1960.

==== Examples ====
The following code will prepare the environment:
{{{
>>> from dateutil.parser import *
>>> from dateutil.tz import *
>>> from datetime import *
>>> TZOFFSETS = {"BRST": -10800}
>>> BRSTTZ = tzoffset(-10800, "BRST")
>>> DEFAULT = datetime(2003, 9, 25)
}}}

Some simple examples based on the {{{date}}} command, using the
{{{TZOFFSET}}} dictionary to provide the BRST timezone offset.
{{{
>>> parse("Thu Sep 25 10:36:28 BRST 2003", tzinfos=TZOFFSETS)
datetime.datetime(2003, 9, 25, 10, 36, 28,
		  tzinfo=tzoffset('BRST', -10800))

>>> parse("2003 10:36:28 BRST 25 Sep Thu", tzinfos=TZOFFSETS)
datetime.datetime(2003, 9, 25, 10, 36, 28,
		  tzinfo=tzoffset('BRST', -10800))
}}}

Notice that since BRST is my local timezone, parsing it without
further timezone settings will yield a {{{tzlocal}}} timezone.
{{{
>>> parse("Thu Sep 25 10:36:28 BRST 2003")
datetime.datetime(2003, 9, 25, 10, 36, 28, tzinfo=tzlocal())
}}}

We can also ask to ignore the timezone explicitly:
{{{
>>> parse("Thu Sep 25 10:36:28 BRST 2003", ignoretz=True)
datetime.datetime(2003, 9, 25, 10, 36, 28)
}}}

That's the same as processing a string without timezone:
{{{
>>> parse("Thu Sep 25 10:36:28 2003")
datetime.datetime(2003, 9, 25, 10, 36, 28)
}}}

Without the year, but passing our {{{DEFAULT}}} datetime to return
the same year, no mattering what year we currently are in:
{{{
>>> parse("Thu Sep 25 10:36:28", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 36, 28)
}}}

Strip it further:
{{{
>>> parse("Thu Sep 10:36:28", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 36, 28)

>>> parse("Thu 10:36:28", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 36, 28)

>>> parse("Thu 10:36", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 36)

>>> parse("10:36", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 36)
>>> 
}}}

Strip in a different way:
{{{
>>> parse("Thu Sep 25 2003")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("Sep 25 2003")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("Sep 2003", default=DEFAULT)
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("Sep", default=DEFAULT)
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("2003", default=DEFAULT)
datetime.datetime(2003, 9, 25, 0, 0)
}}}

Another format, based on {{{date -R}}} (RFC822):
{{{
>>> parse("Thu, 25 Sep 2003 10:49:41 -0300")
datetime.datetime(2003, 9, 25, 10, 49, 41,
		  tzinfo=tzoffset(None, -10800))
}}}

ISO format:
{{{
>>> parse("2003-09-25T10:49:41.5-03:00")
datetime.datetime(2003, 9, 25, 10, 49, 41, 500000,
		  tzinfo=tzoffset(None, -10800))
}}}

Some variations:
{{{
>>> parse("2003-09-25T10:49:41")
datetime.datetime(2003, 9, 25, 10, 49, 41)

>>> parse("2003-09-25T10:49")
datetime.datetime(2003, 9, 25, 10, 49)

>>> parse("2003-09-25T10")
datetime.datetime(2003, 9, 25, 10, 0)

>>> parse("2003-09-25")
datetime.datetime(2003, 9, 25, 0, 0)
}}}

ISO format, without separators:
{{{
>>> parse("20030925T104941.5-0300")
datetime.datetime(2003, 9, 25, 10, 49, 41, 500000,
		  tzinfo=tzinfo=tzoffset(None, -10800))

>>> parse("20030925T104941-0300")
datetime.datetime(2003, 9, 25, 10, 49, 41,
		  tzinfo=tzoffset(None, -10800))

>>> parse("20030925T104941")
datetime.datetime(2003, 9, 25, 10, 49, 41)

>>> parse("20030925T1049")
datetime.datetime(2003, 9, 25, 10, 49)

>>> parse("20030925T10")
datetime.datetime(2003, 9, 25, 10, 0)

>>> parse("20030925")
datetime.datetime(2003, 9, 25, 0, 0)
}}}

Everything together.
{{{
>>> parse("199709020900")
datetime.datetime(1997, 9, 2, 9, 0)
>>> parse("19970902090059")
datetime.datetime(1997, 9, 2, 9, 0, 59)
}}}

Different date orderings:
{{{
>>> parse("2003-09-25")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("2003-Sep-25")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("25-Sep-2003")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("Sep-25-2003")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("09-25-2003")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("25-09-2003")
datetime.datetime(2003, 9, 25, 0, 0)
}}}

Check some ambiguous dates:
{{{
>>> parse("10-09-2003")
datetime.datetime(2003, 10, 9, 0, 0)

>>> parse("10-09-2003", dayfirst=True)
datetime.datetime(2003, 9, 10, 0, 0)

>>> parse("10-09-03")
datetime.datetime(2003, 10, 9, 0, 0)

>>> parse("10-09-03", yearfirst=True)
datetime.datetime(2010, 9, 3, 0, 0)
}}}

Other date separators are allowed:
{{{
>>> parse("2003.Sep.25")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("2003/09/25")
datetime.datetime(2003, 9, 25, 0, 0)
}}}

Even with spaces:
{{{
>>> parse("2003 Sep 25")
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("2003 09 25")
datetime.datetime(2003, 9, 25, 0, 0)
}}}

Hours with letters work:
{{{
>>> parse("10h36m28.5s", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 36, 28, 500000)

>>> parse("01s02h03m", default=DEFAULT)
datetime.datetime(2003, 9, 25, 2, 3, 1)

>>> parse("01h02m03", default=DEFAULT)
datetime.datetime(2003, 9, 3, 1, 2)

>>> parse("01h02", default=DEFAULT)
datetime.datetime(2003, 9, 2, 1, 0)

>>> parse("01h02s", default=DEFAULT)
datetime.datetime(2003, 9, 25, 1, 0, 2)
}}}

With AM/PM:
{{{
>>> parse("10h am", default=DEFAULT)
datetime.datetime(2003, 9, 25, 10, 0)

>>> parse("10pm", default=DEFAULT)
datetime.datetime(2003, 9, 25, 22, 0)

>>> parse("12:00am", default=DEFAULT)
datetime.datetime(2003, 9, 25, 0, 0)

>>> parse("12pm", default=DEFAULT)
datetime.datetime(2003, 9, 25, 12, 0)
}}}

Some special treating for ''pertain'' relations:
{{{
>>> parse("Sep 03", default=DEFAULT)
datetime.datetime(2003, 9, 3, 0, 0)

>>> parse("Sep of 03", default=DEFAULT)
datetime.datetime(2003, 9, 25, 0, 0)
}}}

Fuzzy parsing:
{{{
>>> s = "Today is 25 of September of 2003, exactly " \
...     "at 10:49:41 with timezone -03:00."
>>> parse(s, fuzzy=True)
datetime.datetime(2003, 9, 25, 10, 49, 41,
		  tzinfo=tzoffset(None, -10800))
}}}

Other random formats:
{{{
>>> parse("Wed, July 10, '96")
datetime.datetime(1996, 7, 10, 0, 0)

>>> parse("1996.07.10 AD at 15:08:56 PDT", ignoretz=True)
datetime.datetime(1996, 7, 10, 15, 8, 56)

>>> parse("Tuesday, April 12, 1952 AD 3:30:42pm PST", ignoretz=True)
datetime.datetime(1952, 4, 12, 15, 30, 42)

>>> parse("November 5, 1994, 8:15:30 am EST", ignoretz=True)
datetime.datetime(1994, 11, 5, 8, 15, 30)

>>> parse("3rd of May 2001")
datetime.datetime(2001, 5, 3, 0, 0)

>>> parse("5:50 A.M. on June 13, 1990")
datetime.datetime(1990, 6, 13, 5, 50)
}}}

=== easter ===
This module offers a generic easter computing method for
any given year, using Western, Orthodox or Julian algorithms.

==== easter() function ====
This method was ported from the work done by
[http://users.chariot.net.au/~gmarts/eastalg.htm GM Arts],
on top of the algorithm by
[http://www.tondering.dk/claus/calendar.html Claus Tondering],
which was based in part on the algorithm of Ouding (1940),
as quoted in "Explanatory Supplement to the Astronomical
Almanac", P.  Kenneth Seidelmann, editor.

This algorithm implements three different easter
calculation methods:

    1. Original calculation in Julian calendar, valid in
    dates after 326 AD
    1. Original method, with date converted to Gregorian
    calendar, valid in years 1583 to 4099
    1. Revised method, in Gregorian calendar, valid in
    years 1583 to 4099 as well

These methods are represented by the constants:
{{{
EASTER_JULIAN   = 1
EASTER_ORTHODOX = 2
EASTER_WESTERN  = 3
}}}

The default method is method 3.

=== tz ===
This module offers timezone implementations subclassing
the abstract {{{datetime.tzinfo}}} type. There are
classes to handle [http://www.twinsun.com/tz/tz-link.htm tzfile]
format files (usually are in /etc/localtime,
/usr/share/zoneinfo, etc), TZ environment string (in all
known formats), given ranges (with help from relative
deltas), local machine timezone, fixed offset timezone,
and UTC timezone.

==== tzutc type ====
This type implements a basic UTC timezone. The constructor of this
type accepts no parameters.

==== tzutc examples ====
{{{
>>> from datetime import *
>>> from dateutil.tz import *

>>> datetime.now()
datetime.datetime(2003, 9, 27, 9, 40, 1, 521290)

>>> datetime.now(tzutc())
datetime.datetime(2003, 9, 27, 12, 40, 12, 156379, tzinfo=tzutc())

>>> datetime.now(tzutc()).tzname()
'UTC'
}}}

==== tzoffset type ====
This type implements a fixed offset timezone, with no
support to daylight saving times. Here is the prototype of the
type constructor:
{{{
tzoffset(name, offset)
}}}

The {{{name}}} parameter may be optionally set to {{{None}}}, and
{{{offset}}} must be given in seconds.

==== tzoffset examples ====
{{{
>>> from datetime import *
>>> from dateutil.tz import *

>>> datetime.now(tzoffset("BRST", -10800))
datetime.datetime(2003, 9, 27, 9, 52, 43, 624904,
		  tzinfo=tzinfo=tzoffset('BRST', -10800))

>>> datetime.now(tzoffset("BRST", -10800)).tzname()
'BRST'

>>> datetime.now(tzoffset("BRST", -10800)).astimezone(tzutc())
datetime.datetime(2003, 9, 27, 12, 53, 11, 446419,
		  tzinfo=tzutc())
}}}

==== tzlocal type ====
This type implements timezone settings as known by the
operating system. The constructor of this type accepts no
parameters.

==== tzlocal examples ====
{{{
>>> from datetime import *
>>> from dateutil.tz import *

>>> datetime.now(tzlocal())
datetime.datetime(2003, 9, 27, 10, 1, 43, 673605,
		  tzinfo=tzlocal())

>>> datetime.now(tzlocal()).tzname()
'BRST'

>>> datetime.now(tzlocal()).astimezone(tzoffset(None, 0))
datetime.datetime(2003, 9, 27, 13, 3, 0, 11493,
		  tzinfo=tzoffset(None, 0))
}}}

==== tzstr type ====
This type implements timezone settings extracted from a
string in known TZ environment variable formats. Here is the prototype
of the constructor:
{{{
tzstr(str)
}}}

==== tzstr examples ====
Here are examples of the recognized formats:

  * {{{EST5EDT}}}
  * {{{EST5EDT,4,0,6,7200,10,0,26,7200,3600}}}
  * {{{EST5EDT,4,1,0,7200,10,-1,0,7200,3600}}}
  * {{{EST5EDT4,M4.1.0/02:00:00,M10-5-0/02:00}}}
  * {{{EST5EDT4,95/02:00:00,298/02:00}}}
  * {{{EST5EDT4,J96/02:00:00,J299/02:00}}}

Notice that if daylight information is not present, but a
daylight abbreviation was provided, {{{tzstr}}} will follow the
convention of using the first sunday of April to start daylight
saving, and the last sunday of October to end it. If start or
end time is not present, 2AM will be used, and if the daylight
offset is not present, the standard offset plus one hour will
be used. This convention is the same as used in the GNU libc.

This also means that some of the above examples are exactly
equivalent, and all of these examples are equivalent
in the year of 2003.

Here is the example mentioned in the
[http://www.python.org/doc/current/lib/module-time.html time module documentation].
{{{
>>> os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
>>> time.tzset()
>>> time.strftime('%X %x %Z')
'02:07:36 05/08/03 EDT'
>>> os.environ['TZ'] = 'AEST-10AEDT-11,M10.5.0,M3.5.0'
>>> time.tzset()
>>> time.strftime('%X %x %Z')
'16:08:12 05/08/03 AEST'
}}}

And here is an example showing the same information using {{{tzstr}}},
without touching system settings.
{{{
>>> tz1 = tzstr('EST+05EDT,M4.1.0,M10.5.0')
>>> tz2 = tzstr('AEST-10AEDT-11,M10.5.0,M3.5.0')
>>> dt = datetime(2003, 5, 8, 2, 7, 36, tzinfo=tz1)
>>> dt.strftime('%X %x %Z')
'02:07:36 05/08/03 EDT'
>>> dt.astimezone(tz2).strftime('%X %x %Z')
'16:07:36 05/08/03 AEST'
}}}

Are these really equivalent?
{{{
>>> tzstr('EST5EDT') == tzstr('EST5EDT,4,1,0,7200,10,-1,0,7200,3600')
True
}}}

Check the daylight limit.
{{{
>>> datetime(2003, 4, 6, 1, 59, tzinfo=tz).tzname()
'EST'
>>> datetime(2003, 4, 6, 2, 00, tzinfo=tz).tzname()
'EDT'
>>> datetime(2003, 10, 26, 0, 59, tzinfo=tz).tzname()
'EDT'
>>> datetime(2003, 10, 26, 1, 00, tzinfo=tz).tzname()
'EST'
}}}  

==== tzrange type ====
This type offers the same functionality as the {{{tzstr}}} type, but
instead of timezone strings, information is passed using
{{{relativedelta}}}s which are applied to a datetime set to the first
day of the year. Here is the prototype of this type's constructor:
{{{
tzrange(stdabbr, stdoffset=None, dstabbr=None, dstoffset=None,
	start=None, end=None):
}}}

Offsets must be given in seconds. Information not provided will be
set to the defaults, as explained in the {{{tzstr}}} section above.

==== tzrange examples ====
{{{
>>> tzstr('EST5EDT') == tzrange("EST", -18000, "EDT")
True

>>> from dateutil.relativedelta import *
>>> range1 = tzrange("EST", -18000, "EDT")
>>> range2 = tzrange("EST", -18000, "EDT", -14400,
...                  relativedelta(hours=+2, month=4, day=1,
				   weekday=SU(+1)),
...                  relativedelta(hours=+1, month=10, day=31,
				   weekday=SU(-1)))
>>> tzstr('EST5EDT') == range1 == range2
True
}}}

Notice a minor detail in the last example: while the DST should end
at 2AM, the delta will catch 1AM. That's because the daylight saving
time should end at 2AM standard time (the difference between STD and
DST is 1h in the given example) instead of the DST time. That's how
the {{{tzinfo}}} subtypes should deal with the extra hour that happens
when going back to the standard time. Check
[http://www.python.org/doc/current/lib/datetime-tzinfo.html tzinfo documentation]
for more information.

==== tzfile type ====
This type allows one to use tzfile(5) format timezone files to extract
current and historical zone information. Here is the type constructor
prototype:
{{{
tzfile(fileobj)
}}}

Where {{{fileobj}}} is either a filename or a file-like object with
a {{{read()}}} method.

==== tzfile examples ====
{{{
>>> tz = tzfile("/etc/localtime")
>>> datetime.now(tz)
datetime.datetime(2003, 9, 27, 12, 3, 48, 392138,
		  tzinfo=tzfile('/etc/localtime'))

>>> datetime.now(tz).astimezone(tzutc())
datetime.datetime(2003, 9, 27, 15, 3, 53, 70863,
		  tzinfo=tzutc())

>>> datetime.now(tz).tzname()
'BRST'
>>> datetime(2003, 1, 1, tzinfo=tz).tzname()
'BRDT'
}}}

Check the daylight limit.
{{{
>>> tz = tzfile('/usr/share/zoneinfo/EST5EDT')
>>> datetime(2003, 4, 6, 1, 59, tzinfo=tz).tzname()
'EST'
>>> datetime(2003, 4, 6, 2, 00, tzinfo=tz).tzname()
'EDT'
>>> datetime(2003, 10, 26, 0, 59, tzinfo=tz).tzname()
'EDT'
>>> datetime(2003, 10, 26, 1, 00, tzinfo=tz).tzname()
'EST'
}}}  

==== tzical type ====
This type is able to parse
[ftp://ftp.rfc-editor.org/in-notes/rfc2445.txt iCalendar]
style {{{VTIMEZONE}}} sessions into a Python timezone object.
The constuctor prototype is:
{{{
tzical(fileobj)
}}}

Where {{{fileobj}}} is either a filename or a file-like object with
a {{{read()}}} method.

==== tzical methods ====

    tzical.get(tzid=None)::
    Since a single iCalendar file may contain more than one timezone,
    you must ask for the timezone you want with this method. If there's
    more than one timezone in the parsed file, you'll need to pass the
    {{{tzid}}} parameter. Otherwise, leaving it empty will yield the only
    available timezone.

==== tzical examples ====
Here is a sample file extracted from the RFC. This file defines
the {{{EST5EDT}}} timezone, and will be used in the following example.
{{{
BEGIN:VTIMEZONE
TZID:US-Eastern
LAST-MODIFIED:19870101T000000Z
TZURL:http://zones.stds_r_us.net/tz/US-Eastern
BEGIN:STANDARD
DTSTART:19671029T020000
RRULE:FREQ=YEARLY;BYDAY=-1SU;BYMONTH=10
TZOFFSETFROM:-0400
TZOFFSETTO:-0500
TZNAME:EST
END:STANDARD
BEGIN:DAYLIGHT
DTSTART:19870405T020000
RRULE:FREQ=YEARLY;BYDAY=1SU;BYMONTH=4
TZOFFSETFROM:-0500
TZOFFSETTO:-0400
TZNAME:EDT
END:DAYLIGHT
END:VTIMEZONE
}}}

And here is an example exploring a {{{tzical}}} type:
{{{
>>> from dateutil.tz import *; from datetime import *

>>> tz = tzical('EST5EDT.ics')
>>> tz.keys()
['US-Eastern']

>>> est = tz.get('US-Eastern')
>>> est
<tzicalvtz 'US-Eastern'>

>>> datetime.now(est)
datetime.datetime(2003, 10, 6, 19, 44, 18, 667987,
		  tzinfo=<tzicalvtz 'US-Eastern'>)

>>> est == tz.get()
True
}}}

Let's check the daylight ranges, as usual:
{{{
>>> datetime(2003, 4, 6, 1, 59, tzinfo=est).tzname()
'EST'
>>> datetime(2003, 4, 6, 2, 00, tzinfo=est).tzname()
'EDT'

>>> datetime(2003, 10, 26, 0, 59, tzinfo=est).tzname()
'EDT'
>>> datetime(2003, 10, 26, 1, 00, tzinfo=est).tzname()
'EST'
}}}

==== tzwin type ====
This type offers access to internal registry-based Windows timezones.
The constuctor prototype is:
{{{
tzwin(name)
}}}

Where {{{name}}} is the timezone name. There's a static {{{tzwin.list()}}}
method to check the available names,

==== tzwin methods ====

    tzwin.display()::
    This method returns the timezone extended name.

    tzwin.list()::
    This static method lists all available timezone names.

==== tzwin examples ====
{{{
>>> tz = tzwin("E. South America Standard Time")
}}}

==== tzwinlocal type ====
This type offers access to internal registry-based Windows timezones.
The constructor accepts no parameters, so the prototype is:
{{{
tzwinlocal()
}}}

==== tzwinlocal methods ====

    tzwinlocal.display()::
    This method returns the timezone extended name, and returns
    {{{None}}} if one is not available.

==== tzwinlocal examples ====
{{{
>>> tz = tzwinlocal()
}}}

==== gettz() function ====
This function is a helper that will try its best to get the right
timezone for your environment, or for the given string. The prototype
is as follows:
{{{
gettz(name=None)
}}}

If given, the parameter may be a filename, a path relative to the base
of the timezone information path (the base could be
{{{/usr/share/zoneinfo}}}, for example), a string timezone
specification, or a timezone abbreviation. If {{{name}}} is not given,
and the {{{TZ}}} environment variable is set, it's used instead. If the
parameter is not given, and {{{TZ}}} is not set, the default tzfile
paths will be tried. Then, if no timezone information is found,
an internal compiled database of timezones is used. When running
on Windows, the internal registry-based Windows timezones are also
considered.

Example:
{{{
>>> from dateutil.tz import *
>>> gettz()
tzfile('/etc/localtime')

>>> gettz("America/Sao Paulo")
tzfile('/usr/share/zoneinfo/America/Sao_Paulo')

>>> gettz("EST5EDT")
tzfile('/usr/share/zoneinfo/EST5EDT')

>>> gettz("EST5")
tzstr('EST5')

>>> gettz('BRST')
tzlocal()

>>> os.environ["TZ"] = "America/Sao Paulo"
>>> gettz()
tzfile('/usr/share/zoneinfo/America/Sao_Paulo')

>>> os.environ["TZ"] = "BRST"
>>> gettz()
tzlocal()

>>> gettz("Unavailable")
>>> 
}}}

=== zoneinfo ===
This module provides direct access to the internal compiled
database of timezones. The timezone data and the compiling tools
are obtained from the following project:

  http://www.twinsun.com/tz/tz-link.htm

==== gettz() function ====
This function will try to retrieve the given timezone information
from the internal compiled database, and will cache its results.

Example:
{{{
>>> from dateutil import zoneinfo
>>> zoneinfo.gettz("Brazil/East")
tzfile('Brazil/East')
}}}

## vim:ft=moin

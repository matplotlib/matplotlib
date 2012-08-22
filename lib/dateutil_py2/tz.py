"""
Copyright (c) 2003-2007  Gustavo Niemeyer <gustavo@niemeyer.net>

This module offers extensions to the standard python 2.3+
datetime module.
"""
__author__ = "Gustavo Niemeyer <gustavo@niemeyer.net>"
__license__ = "PSF License"

import datetime
import struct
import time
import sys
import os

relativedelta = None
parser = None
rrule = None

__all__ = ["tzutc", "tzoffset", "tzlocal", "tzfile", "tzrange",
           "tzstr", "tzical", "tzwin", "tzwinlocal", "gettz"]

try:
    from dateutil.tzwin import tzwin, tzwinlocal
except (ImportError, OSError):
    tzwin, tzwinlocal = None, None

ZERO = datetime.timedelta(0)
EPOCHORDINAL = datetime.datetime.utcfromtimestamp(0).toordinal()

class tzutc(datetime.tzinfo):

    def utcoffset(self, dt):
        return ZERO
     
    def dst(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def __eq__(self, other):
        return (isinstance(other, tzutc) or
                (isinstance(other, tzoffset) and other._offset == ZERO))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__

class tzoffset(datetime.tzinfo):

    def __init__(self, name, offset):
        self._name = name
        self._offset = datetime.timedelta(seconds=offset)

    def utcoffset(self, dt):
        return self._offset

    def dst(self, dt):
        return ZERO

    def tzname(self, dt):
        return self._name

    def __eq__(self, other):
        return (isinstance(other, tzoffset) and
                self._offset == other._offset)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               `self._name`,
                               self._offset.days*86400+self._offset.seconds)

    __reduce__ = object.__reduce__

class tzlocal(datetime.tzinfo):

    _std_offset = datetime.timedelta(seconds=-time.timezone)
    if time.daylight:
        _dst_offset = datetime.timedelta(seconds=-time.altzone)
    else:
        _dst_offset = _std_offset

    def utcoffset(self, dt):
        if self._isdst(dt):
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        if self._isdst(dt):
            return self._dst_offset-self._std_offset
        else:
            return ZERO

    def tzname(self, dt):
        return time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        # We can't use mktime here. It is unstable when deciding if
        # the hour near to a change is DST or not.
        # 
        # timestamp = time.mktime((dt.year, dt.month, dt.day, dt.hour,
        #                         dt.minute, dt.second, dt.weekday(), 0, -1))
        # return time.localtime(timestamp).tm_isdst
        #
        # The code above yields the following result:
        #
        #>>> import tz, datetime
        #>>> t = tz.tzlocal()
        #>>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        #'BRDT'
        #>>> datetime.datetime(2003,2,16,0,tzinfo=t).tzname()
        #'BRST'
        #>>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        #'BRST'
        #>>> datetime.datetime(2003,2,15,22,tzinfo=t).tzname()
        #'BRDT'
        #>>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        #'BRDT'
        #
        # Here is a more stable implementation:
        #
        timestamp = ((dt.toordinal() - EPOCHORDINAL) * 86400
                     + dt.hour * 3600
                     + dt.minute * 60
                     + dt.second)
        return time.localtime(timestamp+time.timezone).tm_isdst

    def __eq__(self, other):
        if not isinstance(other, tzlocal):
            return False
        return (self._std_offset == other._std_offset and
                self._dst_offset == other._dst_offset)
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__

class _ttinfo(object):
    __slots__ = ["offset", "delta", "isdst", "abbr", "isstd", "isgmt"]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def __repr__(self):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, `value`))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(l))

    def __eq__(self, other):
        if not isinstance(other, _ttinfo):
            return False
        return (self.offset == other.offset and
                self.delta == other.delta and
                self.isdst == other.isdst and
                self.abbr == other.abbr and
                self.isstd == other.isstd and
                self.isgmt == other.isgmt)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        state = {}
        for name in self.__slots__:
            state[name] = getattr(self, name, None)
        return state

    def __setstate__(self, state):
        for name in self.__slots__:
            if name in state:
                setattr(self, name, state[name])

class tzfile(datetime.tzinfo):

    # http://www.twinsun.com/tz/tz-link.htm
    # ftp://elsie.nci.nih.gov/pub/tz*.tar.gz
    
    def __init__(self, fileobj):
        if isinstance(fileobj, basestring):
            self._filename = fileobj
            fileobj = open(fileobj)
        elif hasattr(fileobj, "name"):
            self._filename = fileobj.name
        else:
            self._filename = `fileobj`

        # From tzfile(5):
        #
        # The time zone information files used by tzset(3)
        # begin with the magic characters "TZif" to identify
        # them as time zone information files, followed by
        # sixteen bytes reserved for future use, followed by
        # six four-byte values of type long, written in a
        # ``standard'' byte order (the high-order  byte
        # of the value is written first).

        if fileobj.read(4) != "TZif":
            raise ValueError, "magic not found"

        fileobj.read(16)

        (
         # The number of UTC/local indicators stored in the file.
         ttisgmtcnt,

         # The number of standard/wall indicators stored in the file.
         ttisstdcnt,
         
         # The number of leap seconds for which data is
         # stored in the file.
         leapcnt,

         # The number of "transition times" for which data
         # is stored in the file.
         timecnt,

         # The number of "local time types" for which data
         # is stored in the file (must not be zero).
         typecnt,

         # The  number  of  characters  of "time zone
         # abbreviation strings" stored in the file.
         charcnt,

        ) = struct.unpack(">6l", fileobj.read(24))

        # The above header is followed by tzh_timecnt four-byte
        # values  of  type long,  sorted  in ascending order.
        # These values are written in ``standard'' byte order.
        # Each is used as a transition time (as  returned  by
        # time(2)) at which the rules for computing local time
        # change.

        if timecnt:
            self._trans_list = struct.unpack(">%dl" % timecnt,
                                             fileobj.read(timecnt*4))
        else:
            self._trans_list = []

        # Next come tzh_timecnt one-byte values of type unsigned
        # char; each one tells which of the different types of
        # ``local time'' types described in the file is associated
        # with the same-indexed transition time. These values
        # serve as indices into an array of ttinfo structures that
        # appears next in the file.
        
        if timecnt:
            self._trans_idx = struct.unpack(">%dB" % timecnt,
                                            fileobj.read(timecnt))
        else:
            self._trans_idx = []
        
        # Each ttinfo structure is written as a four-byte value
        # for tt_gmtoff  of  type long,  in  a  standard  byte
        # order, followed  by a one-byte value for tt_isdst
        # and a one-byte  value  for  tt_abbrind.   In  each
        # structure, tt_gmtoff  gives  the  number  of
        # seconds to be added to UTC, tt_isdst tells whether
        # tm_isdst should be set by  localtime(3),  and
        # tt_abbrind serves  as an index into the array of
        # time zone abbreviation characters that follow the
        # ttinfo structure(s) in the file.

        ttinfo = []

        for i in range(typecnt):
            ttinfo.append(struct.unpack(">lbb", fileobj.read(6)))

        abbr = fileobj.read(charcnt)

        # Then there are tzh_leapcnt pairs of four-byte
        # values, written in  standard byte  order;  the
        # first  value  of  each pair gives the time (as
        # returned by time(2)) at which a leap second
        # occurs;  the  second  gives the  total  number of
        # leap seconds to be applied after the given time.
        # The pairs of values are sorted in ascending order
        # by time.

        # Not used, for now
        if leapcnt:
            leap = struct.unpack(">%dl" % (leapcnt*2),
                                 fileobj.read(leapcnt*8))

        # Then there are tzh_ttisstdcnt standard/wall
        # indicators, each stored as a one-byte value;
        # they tell whether the transition times associated
        # with local time types were specified as standard
        # time or wall clock time, and are used when
        # a time zone file is used in handling POSIX-style
        # time zone environment variables.

        if ttisstdcnt:
            isstd = struct.unpack(">%db" % ttisstdcnt,
                                  fileobj.read(ttisstdcnt))

        # Finally, there are tzh_ttisgmtcnt UTC/local
        # indicators, each stored as a one-byte value;
        # they tell whether the transition times associated
        # with local time types were specified as UTC or
        # local time, and are used when a time zone file
        # is used in handling POSIX-style time zone envi-
        # ronment variables.

        if ttisgmtcnt:
            isgmt = struct.unpack(">%db" % ttisgmtcnt,
                                  fileobj.read(ttisgmtcnt))

        # ** Everything has been read **

        # Build ttinfo list
        self._ttinfo_list = []
        for i in range(typecnt):
            gmtoff, isdst, abbrind =  ttinfo[i]
            # Round to full-minutes if that's not the case. Python's
            # datetime doesn't accept sub-minute timezones. Check
            # http://python.org/sf/1447945 for some information.
            gmtoff = (gmtoff+30)//60*60
            tti = _ttinfo()
            tti.offset = gmtoff
            tti.delta = datetime.timedelta(seconds=gmtoff)
            tti.isdst = isdst
            tti.abbr = abbr[abbrind:abbr.find('\x00', abbrind)]
            tti.isstd = (ttisstdcnt > i and isstd[i] != 0)
            tti.isgmt = (ttisgmtcnt > i and isgmt[i] != 0)
            self._ttinfo_list.append(tti)

        # Replace ttinfo indexes for ttinfo objects.
        trans_idx = []
        for idx in self._trans_idx:
            trans_idx.append(self._ttinfo_list[idx])
        self._trans_idx = tuple(trans_idx)

        # Set standard, dst, and before ttinfos. before will be
        # used when a given time is before any transitions,
        # and will be set to the first non-dst ttinfo, or to
        # the first dst, if all of them are dst.
        self._ttinfo_std = None
        self._ttinfo_dst = None
        self._ttinfo_before = None
        if self._ttinfo_list:
            if not self._trans_list:
                self._ttinfo_std = self._ttinfo_first = self._ttinfo_list[0]
            else:
                for i in range(timecnt-1,-1,-1):
                    tti = self._trans_idx[i]
                    if not self._ttinfo_std and not tti.isdst:
                        self._ttinfo_std = tti
                    elif not self._ttinfo_dst and tti.isdst:
                        self._ttinfo_dst = tti
                    if self._ttinfo_std and self._ttinfo_dst:
                        break
                else:
                    if self._ttinfo_dst and not self._ttinfo_std:
                        self._ttinfo_std = self._ttinfo_dst

                for tti in self._ttinfo_list:
                    if not tti.isdst:
                        self._ttinfo_before = tti
                        break
                else:
                    self._ttinfo_before = self._ttinfo_list[0]

        # Now fix transition times to become relative to wall time.
        #
        # I'm not sure about this. In my tests, the tz source file
        # is setup to wall time, and in the binary file isstd and
        # isgmt are off, so it should be in wall time. OTOH, it's
        # always in gmt time. Let me know if you have comments
        # about this.
        laststdoffset = 0
        self._trans_list = list(self._trans_list)
        for i in range(len(self._trans_list)):
            tti = self._trans_idx[i]
            if not tti.isdst:
                # This is std time.
                self._trans_list[i] += tti.offset
                laststdoffset = tti.offset
            else:
                # This is dst time. Convert to std.
                self._trans_list[i] += laststdoffset
        self._trans_list = tuple(self._trans_list)

    def _find_ttinfo(self, dt, laststd=0):
        timestamp = ((dt.toordinal() - EPOCHORDINAL) * 86400
                     + dt.hour * 3600
                     + dt.minute * 60
                     + dt.second)
        idx = 0
        for trans in self._trans_list:
            if timestamp < trans:
                break
            idx += 1
        else:
            return self._ttinfo_std
        if idx == 0:
            return self._ttinfo_before
        if laststd:
            while idx > 0:
                tti = self._trans_idx[idx-1]
                if not tti.isdst:
                    return tti
                idx -= 1
            else:
                return self._ttinfo_std
        else:
            return self._trans_idx[idx-1]

    def utcoffset(self, dt):
        if not self._ttinfo_std:
            return ZERO
        return self._find_ttinfo(dt).delta

    def dst(self, dt):
        if not self._ttinfo_dst:
            return ZERO
        tti = self._find_ttinfo(dt)
        if not tti.isdst:
            return ZERO

        # The documentation says that utcoffset()-dst() must
        # be constant for every dt.
        return tti.delta-self._find_ttinfo(dt, laststd=1).delta

        # An alternative for that would be:
        #
        # return self._ttinfo_dst.offset-self._ttinfo_std.offset
        #
        # However, this class stores historical changes in the
        # dst offset, so I belive that this wouldn't be the right
        # way to implement this.
        
    def tzname(self, dt):
        if not self._ttinfo_std:
            return None
        return self._find_ttinfo(dt).abbr

    def __eq__(self, other):
        if not isinstance(other, tzfile):
            return False
        return (self._trans_list == other._trans_list and
                self._trans_idx == other._trans_idx and
                self._ttinfo_list == other._ttinfo_list)

    def __ne__(self, other):
        return not self.__eq__(other)


    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, `self._filename`)

    def __reduce__(self):
        if not os.path.isfile(self._filename):
            raise ValueError, "Unpickable %s class" % self.__class__.__name__
        return (self.__class__, (self._filename,))

class tzrange(datetime.tzinfo):

    def __init__(self, stdabbr, stdoffset=None,
                 dstabbr=None, dstoffset=None,
                 start=None, end=None):
        global relativedelta
        if not relativedelta:
            from dateutil import relativedelta
        self._std_abbr = stdabbr
        self._dst_abbr = dstabbr
        if stdoffset is not None:
            self._std_offset = datetime.timedelta(seconds=stdoffset)
        else:
            self._std_offset = ZERO
        if dstoffset is not None:
            self._dst_offset = datetime.timedelta(seconds=dstoffset)
        elif dstabbr and stdoffset is not None:
            self._dst_offset = self._std_offset+datetime.timedelta(hours=+1)
        else:
            self._dst_offset = ZERO
        if dstabbr and start is None:
            self._start_delta = relativedelta.relativedelta(
                    hours=+2, month=4, day=1, weekday=relativedelta.SU(+1))
        else:
            self._start_delta = start
        if dstabbr and end is None:
            self._end_delta = relativedelta.relativedelta(
                    hours=+1, month=10, day=31, weekday=relativedelta.SU(-1))
        else:
            self._end_delta = end

    def utcoffset(self, dt):
        if self._isdst(dt):
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        if self._isdst(dt):
            return self._dst_offset-self._std_offset
        else:
            return ZERO

    def tzname(self, dt):
        if self._isdst(dt):
            return self._dst_abbr
        else:
            return self._std_abbr

    def _isdst(self, dt):
        if not self._start_delta:
            return False
        year = datetime.datetime(dt.year,1,1)
        start = year+self._start_delta
        end = year+self._end_delta
        dt = dt.replace(tzinfo=None)
        if start < end:
            return dt >= start and dt < end
        else:
            return dt >= start or dt < end

    def __eq__(self, other):
        if not isinstance(other, tzrange):
            return False
        return (self._std_abbr == other._std_abbr and
                self._dst_abbr == other._dst_abbr and
                self._std_offset == other._std_offset and
                self._dst_offset == other._dst_offset and
                self._start_delta == other._start_delta and
                self._end_delta == other._end_delta)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s(...)" % self.__class__.__name__

    __reduce__ = object.__reduce__

class tzstr(tzrange):
    
    def __init__(self, s):
        global parser
        if not parser:
            from dateutil import parser
        self._s = s

        res = parser._parsetz(s)
        if res is None:
            raise ValueError, "unknown string format"

        # Here we break the compatibility with the TZ variable handling.
        # GMT-3 actually *means* the timezone -3.
        if res.stdabbr in ("GMT", "UTC"):
            res.stdoffset *= -1

        # We must initialize it first, since _delta() needs
        # _std_offset and _dst_offset set. Use False in start/end
        # to avoid building it two times.
        tzrange.__init__(self, res.stdabbr, res.stdoffset,
                         res.dstabbr, res.dstoffset,
                         start=False, end=False)

        if not res.dstabbr:
            self._start_delta = None
            self._end_delta = None
        else:
            self._start_delta = self._delta(res.start)
            if self._start_delta:
                self._end_delta = self._delta(res.end, isend=1)

    def _delta(self, x, isend=0):
        kwargs = {}
        if x.month is not None:
            kwargs["month"] = x.month
            if x.weekday is not None:
                kwargs["weekday"] = relativedelta.weekday(x.weekday, x.week)
                if x.week > 0:
                    kwargs["day"] = 1
                else:
                    kwargs["day"] = 31
            elif x.day:
                kwargs["day"] = x.day
        elif x.yday is not None:
            kwargs["yearday"] = x.yday
        elif x.jyday is not None:
            kwargs["nlyearday"] = x.jyday
        if not kwargs:
            # Default is to start on first sunday of april, and end
            # on last sunday of october.
            if not isend:
                kwargs["month"] = 4
                kwargs["day"] = 1
                kwargs["weekday"] = relativedelta.SU(+1)
            else:
                kwargs["month"] = 10
                kwargs["day"] = 31
                kwargs["weekday"] = relativedelta.SU(-1)
        if x.time is not None:
            kwargs["seconds"] = x.time
        else:
            # Default is 2AM.
            kwargs["seconds"] = 7200
        if isend:
            # Convert to standard time, to follow the documented way
            # of working with the extra hour. See the documentation
            # of the tzinfo class.
            delta = self._dst_offset-self._std_offset
            kwargs["seconds"] -= delta.seconds+delta.days*86400
        return relativedelta.relativedelta(**kwargs)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, `self._s`)

class _tzicalvtzcomp:
    def __init__(self, tzoffsetfrom, tzoffsetto, isdst,
                       tzname=None, rrule=None):
        self.tzoffsetfrom = datetime.timedelta(seconds=tzoffsetfrom)
        self.tzoffsetto = datetime.timedelta(seconds=tzoffsetto)
        self.tzoffsetdiff = self.tzoffsetto-self.tzoffsetfrom
        self.isdst = isdst
        self.tzname = tzname
        self.rrule = rrule

class _tzicalvtz(datetime.tzinfo):
    def __init__(self, tzid, comps=[]):
        self._tzid = tzid
        self._comps = comps
        self._cachedate = []
        self._cachecomp = []

    def _find_comp(self, dt):
        if len(self._comps) == 1:
            return self._comps[0]
        dt = dt.replace(tzinfo=None)
        try:
            return self._cachecomp[self._cachedate.index(dt)]
        except ValueError:
            pass
        lastcomp = None
        lastcompdt = None
        for comp in self._comps:
            if not comp.isdst:
                # Handle the extra hour in DST -> STD
                compdt = comp.rrule.before(dt-comp.tzoffsetdiff, inc=True)
            else:
                compdt = comp.rrule.before(dt, inc=True)
            if compdt and (not lastcompdt or lastcompdt < compdt):
                lastcompdt = compdt
                lastcomp = comp
        if not lastcomp:
            # RFC says nothing about what to do when a given
            # time is before the first onset date. We'll look for the
            # first standard component, or the first component, if
            # none is found.
            for comp in self._comps:
                if not comp.isdst:
                    lastcomp = comp
                    break
            else:
                lastcomp = comp[0]
        self._cachedate.insert(0, dt)
        self._cachecomp.insert(0, lastcomp)
        if len(self._cachedate) > 10:
            self._cachedate.pop()
            self._cachecomp.pop()
        return lastcomp

    def utcoffset(self, dt):
        return self._find_comp(dt).tzoffsetto

    def dst(self, dt):
        comp = self._find_comp(dt)
        if comp.isdst:
            return comp.tzoffsetdiff
        else:
            return ZERO

    def tzname(self, dt):
        return self._find_comp(dt).tzname

    def __repr__(self):
        return "<tzicalvtz %s>" % `self._tzid`

    __reduce__ = object.__reduce__

class tzical:
    def __init__(self, fileobj):
        global rrule
        if not rrule:
            from dateutil import rrule

        if isinstance(fileobj, basestring):
            self._s = fileobj
            fileobj = open(fileobj)
        elif hasattr(fileobj, "name"):
            self._s = fileobj.name
        else:
            self._s = `fileobj`

        self._vtz = {}

        self._parse_rfc(fileobj.read())

    def keys(self):
        return self._vtz.keys()

    def get(self, tzid=None):
        if tzid is None:
            keys = self._vtz.keys()
            if len(keys) == 0:
                raise ValueError, "no timezones defined"
            elif len(keys) > 1:
                raise ValueError, "more than one timezone available"
            tzid = keys[0]
        return self._vtz.get(tzid)

    def _parse_offset(self, s):
        s = s.strip()
        if not s:
            raise ValueError, "empty offset"
        if s[0] in ('+', '-'):
            signal = (-1,+1)[s[0]=='+']
            s = s[1:]
        else:
            signal = +1
        if len(s) == 4:
            return (int(s[:2])*3600+int(s[2:])*60)*signal
        elif len(s) == 6:
            return (int(s[:2])*3600+int(s[2:4])*60+int(s[4:]))*signal
        else:
            raise ValueError, "invalid offset: "+s

    def _parse_rfc(self, s):
        lines = s.splitlines()
        if not lines:
            raise ValueError, "empty string"

        # Unfold
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                del lines[i]
            elif i > 0 and line[0] == " ":
                lines[i-1] += line[1:]
                del lines[i]
            else:
                i += 1

        tzid = None
        comps = []
        invtz = False
        comptype = None
        for line in lines:
            if not line:
                continue
            name, value = line.split(':', 1)
            parms = name.split(';')
            if not parms:
                raise ValueError, "empty property name"
            name = parms[0].upper()
            parms = parms[1:]
            if invtz:
                if name == "BEGIN":
                    if value in ("STANDARD", "DAYLIGHT"):
                        # Process component
                        pass
                    else:
                        raise ValueError, "unknown component: "+value
                    comptype = value
                    founddtstart = False
                    tzoffsetfrom = None
                    tzoffsetto = None
                    rrulelines = []
                    tzname = None
                elif name == "END":
                    if value == "VTIMEZONE":
                        if comptype:
                            raise ValueError, \
                                  "component not closed: "+comptype
                        if not tzid:
                            raise ValueError, \
                                  "mandatory TZID not found"
                        if not comps:
                            raise ValueError, \
                                  "at least one component is needed"
                        # Process vtimezone
                        self._vtz[tzid] = _tzicalvtz(tzid, comps)
                        invtz = False
                    elif value == comptype:
                        if not founddtstart:
                            raise ValueError, \
                                  "mandatory DTSTART not found"
                        if tzoffsetfrom is None:
                            raise ValueError, \
                                  "mandatory TZOFFSETFROM not found"
                        if tzoffsetto is None:
                            raise ValueError, \
                                  "mandatory TZOFFSETFROM not found"
                        # Process component
                        rr = None
                        if rrulelines:
                            rr = rrule.rrulestr("\n".join(rrulelines),
                                                compatible=True,
                                                ignoretz=True,
                                                cache=True)
                        comp = _tzicalvtzcomp(tzoffsetfrom, tzoffsetto,
                                              (comptype == "DAYLIGHT"),
                                              tzname, rr)
                        comps.append(comp)
                        comptype = None
                    else:
                        raise ValueError, \
                              "invalid component end: "+value
                elif comptype:
                    if name == "DTSTART":
                        rrulelines.append(line)
                        founddtstart = True
                    elif name in ("RRULE", "RDATE", "EXRULE", "EXDATE"):
                        rrulelines.append(line)
                    elif name == "TZOFFSETFROM":
                        if parms:
                            raise ValueError, \
                                  "unsupported %s parm: %s "%(name, parms[0])
                        tzoffsetfrom = self._parse_offset(value)
                    elif name == "TZOFFSETTO":
                        if parms:
                            raise ValueError, \
                                  "unsupported TZOFFSETTO parm: "+parms[0]
                        tzoffsetto = self._parse_offset(value)
                    elif name == "TZNAME":
                        if parms:
                            raise ValueError, \
                                  "unsupported TZNAME parm: "+parms[0]
                        tzname = value
                    elif name == "COMMENT":
                        pass
                    else:
                        raise ValueError, "unsupported property: "+name
                else:
                    if name == "TZID":
                        if parms:
                            raise ValueError, \
                                  "unsupported TZID parm: "+parms[0]
                        tzid = value
                    elif name in ("TZURL", "LAST-MODIFIED", "COMMENT"):
                        pass
                    else:
                        raise ValueError, "unsupported property: "+name
            elif name == "BEGIN" and value == "VTIMEZONE":
                tzid = None
                comps = []
                invtz = True

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, `self._s`)

if sys.platform != "win32":
    TZFILES = ["/etc/localtime", "localtime"]
    TZPATHS = ["/usr/share/zoneinfo", "/usr/lib/zoneinfo", "/etc/zoneinfo"]
else:
    TZFILES = []
    TZPATHS = []

def gettz(name=None):
    tz = None
    if not name:
        try:
            name = os.environ["TZ"]
        except KeyError:
            pass
    if name is None or name == ":":
        for filepath in TZFILES:
            if not os.path.isabs(filepath):
                filename = filepath
                for path in TZPATHS:
                    filepath = os.path.join(path, filename)
                    if os.path.isfile(filepath):
                        break
                else:
                    continue
            if os.path.isfile(filepath):
                try:
                    tz = tzfile(filepath)
                    break
                except (IOError, OSError, ValueError):
                    pass
        else:
            tz = tzlocal()
    else:
        if name.startswith(":"):
            name = name[:-1]
        if os.path.isabs(name):
            if os.path.isfile(name):
                tz = tzfile(name)
            else:
                tz = None
        else:
            for path in TZPATHS:
                filepath = os.path.join(path, name)
                if not os.path.isfile(filepath):
                    filepath = filepath.replace(' ','_')
                    if not os.path.isfile(filepath):
                        continue
                try:
                    tz = tzfile(filepath)
                    break
                except (IOError, OSError, ValueError):
                    pass
            else:
                tz = None
                if tzwin:
                    try:
                        tz = tzwin(name)
                    except OSError:
                        pass
                if not tz:
                    from dateutil.zoneinfo import gettz
                    tz = gettz(name)
                if not tz:
                    for c in name:
                        # name must have at least one offset to be a tzstr
                        if c in "0123456789":
                            try:
                                tz = tzstr(name)
                            except ValueError:
                                pass
                            break
                    else:
                        if name in ("GMT", "UTC"):
                            tz = tzutc()
                        elif name in time.tzname:
                            tz = tzlocal()
    return tz

# vim:ts=4:sw=4:et

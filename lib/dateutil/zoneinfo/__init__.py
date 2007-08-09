"""
Copyright (c) 2003-2005  Gustavo Niemeyer <gustavo@niemeyer.net>

This module offers extensions to the standard python 2.3+
datetime module.
"""
from dateutil.tz import tzfile
from tarfile import TarFile
import os

__author__ = "Gustavo Niemeyer <gustavo@niemeyer.net>"
__license__ = "PSF License"

__all__ = ["setcachesize", "gettz", "rebuild"]

CACHE = []
CACHESIZE = 10

class tzfile(tzfile):
    def __reduce__(self):
        return (gettz, (self._filename,))

def getzoneinfofile():
    filenames = os.listdir(os.path.join(os.path.dirname(__file__)))
    filenames.sort()
    filenames.reverse()
    for entry in filenames:
        if entry.startswith("zoneinfo") and ".tar." in entry:
            return os.path.join(os.path.dirname(__file__), entry)
    return None

ZONEINFOFILE = getzoneinfofile()

del getzoneinfofile

def setcachesize(size):
    global CACHESIZE, CACHE
    CACHESIZE = size
    del CACHE[size:]

def gettz(name):
    tzinfo = None
    if ZONEINFOFILE:
        for cachedname, tzinfo in CACHE:
            if cachedname == name:
                break
        else:
            tf = TarFile.open(ZONEINFOFILE)
            try:
                zonefile = tf.extractfile(name)
            except KeyError:
                tzinfo = None
            else:
                tzinfo = tzfile(zonefile)
            tf.close()
            CACHE.insert(0, (name, tzinfo))
            del CACHE[CACHESIZE:]
    return tzinfo

def rebuild(filename, tag=None, format="gz"):
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    zonedir = os.path.join(tmpdir, "zoneinfo")
    moduledir = os.path.dirname(__file__)
    if tag: tag = "-"+tag
    targetname = "zoneinfo%s.tar.%s" % (tag, format)
    try:
        tf = TarFile.open(filename)
        for name in tf.getnames():
            if not (name.endswith(".sh") or
                    name.endswith(".tab") or
                    name == "leapseconds"):
                tf.extract(name, tmpdir)
                filepath = os.path.join(tmpdir, name)
                os.system("zic -d %s %s" % (zonedir, filepath))
        tf.close()
        target = os.path.join(moduledir, targetname)
        for entry in os.listdir(moduledir):
            if entry.startswith("zoneinfo") and ".tar." in entry:
                os.unlink(os.path.join(moduledir, entry))
        tf = TarFile.open(target, "w:%s" % format)
        for entry in os.listdir(zonedir):
            entrypath = os.path.join(zonedir, entry)
            tf.add(entrypath, entry)
        tf.close()
    finally:
        shutil.rmtree(tmpdir)

import datetime, StringIO, unittest
import matplotlib.mlab as mlab
import numpy

class TestMlab(unittest.TestCase):
    def test_csv2rec_closefile(self):
        # If passed a file-like object, rec2csv should not close it.
        ra=numpy.rec.array([(123, 1197346475.0137341), (456, 123.456)],
                           dtype=[('a', '<i8'), ('b', '<f8')])
        fh = StringIO.StringIO()
        mlab.rec2csv( ra, fh )
        self.failIf( fh.closed )

    def test_csv2rec_roundtrip(self):

        # Make sure double-precision floats and strings pass through a
        # roundtrip unaltered.

        # A bug in numpy (fixed in r4602) meant that numpy scalars
        # lost precision when passing through repr(). csv2rec was
        # affected by this. This test will only pass on numpy >=
        # 1.0.5.
        delta = datetime.timedelta(days=1)
        date0 = datetime.date(2007,12,16)
        date1 = date0 + delta
        date2 = date1 + delta

        delta = datetime.timedelta(days=1)
        datetime0 = datetime.datetime(2007,12,16,22,29,34,924122)
        datetime1 = datetime0 + delta
        datetime2 = datetime1 + delta
        ra=numpy.rec.fromrecords([
                (123, date0, datetime0, 1197346475.0137341, 'a,bc'),
                (456, date1, datetime1, 123.456, 'd\'ef'),
                (789, date2, datetime2, 0.000000001, 'ghi'),
                            ],
            names='intdata,datedata,datetimedata,floatdata,stringdata')

        fh = StringIO.StringIO()
        mlab.rec2csv( ra, fh )
        fh.seek(0)
        if 0:
            print 'CSV contents:','-'*40
            print fh.read()
            print '-'*40
            fh.seek(0)
        ra2 = mlab.csv2rec(fh)
        fh.close()
        #print 'ra', ra
        #print 'ra2', ra2
        for name in ra.dtype.names:
            if 0:
                print name, repr(ra[name]), repr(ra2[name])
                dt = ra.dtype[name]
                print 'repr(dt.type)',repr(dt.type)
            self.failUnless( numpy.all(ra[name] == ra2[name]) ) # should not fail with numpy 1.0.5

if __name__=='__main__':
    unittest.main()

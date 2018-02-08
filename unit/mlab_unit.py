from __future__ import print_function
import datetime
import StringIO
import unittest
import matplotlib.mlab as mlab
import numpy


class TestMlab(unittest.TestCase):
    def test_csv2rec_closefile(self):
        # If passed a file-like object, rec2csv should not close it.
        ra = numpy.rec.array([(123, 1197346475.0137341), (456, 123.456)],
                             dtype=[('a', '<i8'), ('b', '<f8')])
        fh = StringIO.StringIO()
        mlab.rec2csv(ra, fh)
        self.failIf(fh.closed)

    def test_csv2rec_roundtrip(self):

        # Make sure double-precision floats and strings pass through a
        # roundtrip unaltered.

        # A bug in numpy (fixed in r4602) meant that numpy scalars
        # lost precision when passing through repr(). csv2rec was
        # affected by this. This test will only pass on numpy >=
        # 1.0.5.
        delta = datetime.timedelta(days=1)
        date0 = datetime.date(2007, 12, 16)
        date1 = date0 + delta
        date2 = date1 + delta

        delta = datetime.timedelta(days=1)
        datetime0 = datetime.datetime(2007, 12, 16, 22, 29, 34, 924122)
        datetime1 = datetime0 + delta
        datetime2 = datetime1 + delta
        ra = numpy.rec.fromrecords([(123, date0, datetime0, 1197346475.0137341, 'a,bc'),
                                    (456, date1, datetime1, 123.456, 'd\'ef'),
                                    (789, date2, datetime2, 0.000000001, 'ghi'),
                                    ],
                                   names='intdata,datedata,datetimedata,floatdata,stringdata')

        fh = StringIO.StringIO()
        mlab.rec2csv(ra, fh)
        fh.seek(0)
        if 0:
            print('CSV contents:', '-' * 40)
            print(fh.read())
            print('-' * 40)
            fh.seek(0)
        ra2 = mlab.csv2rec(fh)
        fh.close()
        for name in ra.dtype.names:
            if 0:
                print(name, repr(ra[name]), repr(ra2[name]))
                dt = ra.dtype[name]
                print('repr(dt.type)', repr(dt.type))
            self.failUnless(numpy.all(ra[name] == ra2[name]))  # should not fail with numpy 1.0.5

    def test_csv2rec_masks(self):
        # Make sure masked entries survive roundtrip

        csv = """date,age,weight,name
2007-01-01,12,32.2,"jdh1"
0000-00-00,0,23,"jdh2"
2007-01-03,,32.5,"jdh3"
2007-01-04,12,NaN,"jdh4"
2007-01-05,-1,NULL,"""
        missingd = dict(date='0000-00-00', age='-1', weight='NULL')
        fh = StringIO.StringIO(csv)
        r1 = mlab.csv2rec(fh, missingd=missingd)
        fh = StringIO.StringIO()
        mlab.rec2csv(r1, fh, missingd=missingd)
        fh.seek(0)
        r2 = mlab.csv2rec(fh, missingd=missingd)

        self.failUnless(numpy.all(r2['date'].mask == [0, 1, 0, 0, 0]))
        self.failUnless(numpy.all(r2['age'].mask == [0, 0, 1, 0, 1]))
        self.failUnless(numpy.all(r2['weight'].mask == [0, 0, 0, 0, 1]))
        self.failUnless(numpy.all(r2['name'].mask == [0, 0, 0, 0, 1]))
        self.failUnless(numpy.all(r2['name'].mask == [0, 0, 0, 0, 1]))

if __name__ == '__main__':
    unittest.main()

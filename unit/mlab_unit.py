import unittest
import matplotlib.mlab as mlab
import numpy
import StringIO

class TestMlab(unittest.TestCase):
    def test_csv2rec_closefile(self):
        # If passed a file-like object, rec2csv should not close it.
        ra=numpy.rec.array([(123, 1197346475.0137341), (456, 123.456)],
                           dtype=[('a', '<i8'), ('b', '<f8')])
        fh = StringIO.StringIO()
        mlab.rec2csv( ra, fh )
        self.failIf( fh.closed )

    def test_csv2rec_roundtrip(self):
        # Make sure double-precision floats pass through.

        # A bug in numpy (fixed in r4602) meant that numpy scalars
        # lost precision when passing through repr(). csv2rec was
        # affected by this. This test will only pass on numpy >=
        # 1.0.5.
        ra=numpy.rec.array([(123, 1197346475.0137341), (456, 123.456)],
                           dtype=[('a', '<i8'), ('b', '<f8')])
        rec2csv_closes_files = True
        if rec2csv_closes_files:
            fh = 'mlab_unit_tmp.csv'
        else:
            fh = StringIO.StringIO()
        mlab.rec2csv( ra, fh )
        if not rec2csv_closes_files:
            fh.seek(0)
        ra2 = mlab.csv2rec(fh)
        for name in ra.dtype.names:
            #print name, repr(ra[name]), repr(ra2[name])
            self.failUnless( numpy.all(ra[name] == ra2[name]) ) # should not fail with numpy 1.0.5

if __name__=='__main__':
    unittest.main()

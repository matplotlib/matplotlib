import unittest
import numpy as np
import matplotlib.cbook as cbook

class TestAxes(unittest.TestCase):
    def test_delete_masked_points_arrays(self):
        input = (   [1,2,3,np.nan,5],
                    np.array((1,2,3,4,5)),
                    )
        expected = [np.array((1,2,3,5))]*2
        actual = cbook.delete_masked_points(*input)
        assert np.allclose(actual, expected)

        input = (   np.ma.array( [1,2,3,4,5], mask=[False,False,False,True,False] ),
                    np.array((1,2,3,4,5)),
                    )
        expected = [np.array((1,2,3,5))]*2
        actual = cbook.delete_masked_points(*input)
        assert np.allclose(actual, expected)

        input = (   [1,2,3,np.nan,5],
                    np.ma.array( [1,2,3,4,5], mask=[False,False,False,True,False] ),
                    np.array((1,2,3,4,5)),
                    )
        expected = [np.array((1,2,3,5))]*3
        actual = cbook.delete_masked_points(*input)
        assert np.allclose(actual, expected)

        input = ()
        expected = ()
        actual = cbook.delete_masked_points(*input)
        assert np.allclose(actual, expected)


        input = (   [1,2,3,np.nan,5],
                    )
        expected = [np.array((1,2,3,5))]*1
        actual = cbook.delete_masked_points(*input)
        assert np.allclose(actual, expected)

        input = (   np.array((1,2,3,4,5)),
                    )
        expected = [np.array((1,2,3,4,5))]*1
        actual = cbook.delete_masked_points(*input)
        assert np.allclose(actual, expected)

    def test_delete_masked_points_strings(self):
        input = (   'hello',
                    )
        expected = ('hello',)
        actual = cbook.delete_masked_points(*input)
        assert actual == expected

        input = (   u'hello',
                    )
        expected = (u'hello',)
        actual = cbook.delete_masked_points(*input)
        assert actual == expected


if __name__=='__main__':
    unittest.main()

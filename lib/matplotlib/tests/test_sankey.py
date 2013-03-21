from matplotlib.sankey import Sankey


def test_sankey():
    # lets just create a sankey instance and check the code runs
    sankey = Sankey()
    sankey.add()


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)

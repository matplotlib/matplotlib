import builtins

import matplotlib


def test_simple():
    assert 1 + 1 == 2


def test_override_builtins():
    import pylab

    ok_to_override = {
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        'any',
        'all',
        'sum',
        'divmod'
    }
    overridden = False
    for key in dir(pylab):
        if key in dir(builtins):
            if (getattr(pylab, key) != getattr(builtins, key) and
                    key not in ok_to_override):
                print("'%s' was overridden in globals()." % key)
                overridden = True

    assert not overridden


def test_verbose():
    assert isinstance(matplotlib.verbose, matplotlib.Verbose)

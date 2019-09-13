import matplotlib
import matplotlib.rcsetup


def test_use_doc_standard_backends():
    """
    Test that the standard backends mentioned in the docstring of
    matplotlib.use() are the same as in matplotlib.rcsetup.
    """
    def parse(key):
        backends = []
        for line in matplotlib.use.__doc__.split(key)[1].split('\n'):
            if not line.strip():
                break
            backends += [e.strip() for e in line.split(',') if e]
        return backends

    assert (set(parse('- interactive backends:\n')) ==
            set(matplotlib.rcsetup.interactive_bk))
    assert (set(parse('- non-interactive backends:\n')) ==
            set(matplotlib.rcsetup.non_interactive_bk))

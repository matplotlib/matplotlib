import pytest
"""
So pytest is trying to load tests in numpy, and generates this error

____________________________ ERROR at setup of test ____________________________
file /home/travis/build/matplotlib/matplotlib/venv/lib/python2.7/site-packages/
numpy/testing/nosetester.py, line 249
      def test(self, label='fast', verbose=1, extra_argv=None, doctests=False,
        fixture 'self' not found
        available fixtures: tmpdir_factory, pytestconfig, cov, cache, recwarn,
        monkeypatch, record_xml_property, capfd, capsys, tmpdir

        use 'py.test --fixtures [testpath]' for help on them.
/home/travis/build/matplotlib/matplotlib/venv/lib/python2.7/site-packages/
 numpy/testing/nosetester.py:249

this file is intended to stop that behaviour, by blocking files from
being considered for collection if "numpy" is in the path
"""

def pytest_ignore_collect(path, config):
    if 'numpy' not in path.basename:
        print('allowing ', path)
        return True
    else:
        print('blocking ', path)
        return False

#    return 'numpy' not in path.basename:
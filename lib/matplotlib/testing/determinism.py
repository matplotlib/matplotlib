"""
Provides utilities to test output reproducibility.
"""

import io
import os
import re

from matplotlib import pyplot as plt


def _test_determinism_save(filename, objects='mhi', format="pdf"):
    # save current value of SOURCE_DATE_EPOCH and set it
    # to a constant value, so that time difference is not
    # taken into account
    sde = os.environ.pop('SOURCE_DATE_EPOCH', None)
    os.environ['SOURCE_DATE_EPOCH'] = "946684800"

    fig = plt.figure()

    if 'm' in objects:
        # use different markers...
        ax1 = fig.add_subplot(1, 6, 1)
        x = range(10)
        ax1.plot(x, [1] * 10, marker=u'D')
        ax1.plot(x, [2] * 10, marker=u'x')
        ax1.plot(x, [3] * 10, marker=u'^')
        ax1.plot(x, [4] * 10, marker=u'H')
        ax1.plot(x, [5] * 10, marker=u'v')

    if 'h' in objects:
        # also use different hatch patterns
        ax2 = fig.add_subplot(1, 6, 2)
        bars = ax2.bar(range(1, 5), range(1, 5)) + \
            ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5))
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

    if 'i' in objects:
        # also use different images
        A = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        fig.add_subplot(1, 6, 3).imshow(A, interpolation='nearest')
        A = [[1, 3, 2], [1, 2, 3], [3, 1, 2]]
        fig.add_subplot(1, 6, 4).imshow(A, interpolation='bilinear')
        A = [[2, 3, 1], [1, 2, 3], [2, 1, 3]]
        fig.add_subplot(1, 6, 5).imshow(A, interpolation='bicubic')

    x = range(5)
    fig.add_subplot(1, 6, 6).plot(x, x)

    fig.savefig(filename, format=format)

    # Restores SOURCE_DATE_EPOCH
    if sde is None:
        os.environ.pop('SOURCE_DATE_EPOCH', None)
    else:
        os.environ['SOURCE_DATE_EPOCH'] = sde


def _test_determinism(objects='mhi', format="pdf", uid=""):
    """
    Output three times the same graphs and checks that the outputs are exactly
    the same.

    Parameters
    ----------
    objects : str
        contains characters corresponding to objects to be included in the test
        document: 'm' for markers, 'h' for hatch patterns, 'i' for images. The
        default value is "mhi", so that the test includes all these objects.
    format : str
        format string. The default value is "pdf".
    uid : str
        some string to add to the filename used to store the output. Use it to
        allow parallel execution of two tests with the same objects parameter.
    """
    import sys
    from subprocess import check_call
    from nose.tools import assert_equal
    filename = 'determinism_O%s%s.%s' % (objects, uid, format)
    plots = []
    for i in range(3):
        check_call([sys.executable, '-R', '-c',
                    'import matplotlib; '
                    'matplotlib.use(%r); '
                    'from matplotlib.testing.determinism '
                    'import _test_determinism_save;'
                    '_test_determinism_save(%r,%r,%r)'
                    % (format, filename, objects, format)])
        with open(filename, 'rb') as fd:
            plots.append(fd.read())
        os.unlink(filename)
    for p in plots[1:]:
        assert_equal(p, plots[0])


def _test_source_date_epoch(format, string, keyword=b"CreationDate"):
    """
    Test SOURCE_DATE_EPOCH support. Output a document with the envionment
    variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the
    document contains the timestamp that corresponds to this date (given as an
    argument).

    Parameters
    ----------
    format : str
        format string, such as "pdf".
    string : str
        timestamp string for 2000-01-01 00:00 UTC.
    keyword : str
        a string to look at when searching for the timestamp in the document
        (used in case the test fails).
    """
    try:
        # save current value of SOURCE_DATE_EPOCH
        sde = os.environ.pop('SOURCE_DATE_EPOCH', None)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = [1, 2, 3, 4, 5]
        ax.plot(x, x)
        os.environ['SOURCE_DATE_EPOCH'] = "946684800"
        find_keyword = re.compile(b".*" + keyword + b".*")
        with io.BytesIO() as output:
            fig.savefig(output, format=format)
            output.seek(0)
            buff = output.read()
            key = find_keyword.search(buff)
            if key:
                print(key.group())
            else:
                print("Timestamp keyword (%s) not found!" % keyword)
            assert string in buff
        os.environ.pop('SOURCE_DATE_EPOCH', None)
        with io.BytesIO() as output:
            fig.savefig(output, format=format)
            output.seek(0)
            buff = output.read()
            assert string not in buff
    finally:
        # Restores SOURCE_DATE_EPOCH
        if sde is None:
            os.environ.pop('SOURCE_DATE_EPOCH', None)
        else:
            os.environ['SOURCE_DATE_EPOCH'] = sde

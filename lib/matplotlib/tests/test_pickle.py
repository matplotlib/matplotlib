from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use('tkagg')

from matplotlib.testing.decorators import cleanup, image_comparison
import matplotlib.pyplot as plt

from nose.tools import assert_equal, assert_not_equal

# cpickle is faster, pickle gives better exceptions
import cPickle as pickle
import pickle

from cStringIO import StringIO


def recursive_pickle(obj, nested_info='top level object', memo=None):
    """
    Pickle the object's attributes recursively, storing a memo of the object
    which have already been pickled.
    
    If any pickling issues occur, a pickle.Pickle error will be raised with details.
    
    This is not a completely general purpose routine, but will be useful for
    debugging some pickle issues. HINT: cPickle is less verbose than Pickle
    
    
    """
    if memo is None:
        memo = {}
    
    if id(obj) in memo:
        return
    
    # put this object in the memo
    memo[id(obj)] = obj

    # start by pickling all of the object's attributes/contents
    
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            recursive_pickle(item, memo=memo, nested_info='list item #%s in (%s)' % (i, nested_info))
    else:
        if isinstance(obj, dict):
            state = obj
        elif hasattr(obj, '__getstate__'):
            state = obj.__getstate__()
            if not isinstance(state, dict):
                state = {}
        elif hasattr(obj, '__dict__'):
            state = obj.__dict__
        else:
            state = {}
            
        for key, value in state.iteritems():
            recursive_pickle(value, memo=memo, nested_info='attribute "%s" in (%s)' % (key, nested_info))
        
#    print(id(obj), type(obj), nested_info)
    
    # finally, try picking the object itself
    try:
        pickle.dump(obj, StringIO())#, pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, AssertionError), err:
        print(pickle.PickleError('Pickling failed with nested info: [(%s) %s].'
                                 '\nException: %s' % (type(obj), 
                                                        nested_info, 
                                                        err)))
        # re-raise the exception for full traceback
        raise


@cleanup
def test_simple():
    fig = plt.figure()
    # un-comment to debug
    recursive_pickle(fig, 'figure')
    pickle.dump(fig, StringIO(), pickle.HIGHEST_PROTOCOL)

    ax = plt.subplot(121)
#    recursive_pickle(ax, 'ax')
    pickle.dump(ax, StringIO(), pickle.HIGHEST_PROTOCOL)

    ax = plt.axes(projection='polar')
#    recursive_pickle(ax, 'ax')
    pickle.dump(ax, StringIO(), pickle.HIGHEST_PROTOCOL)
    
#    ax = plt.subplot(121, projection='hammer')
#    recursive_pickle(ax, 'figure')
#    pickle.dump(ax, StringIO(), pickle.HIGHEST_PROTOCOL)


@image_comparison(baseline_images=['multi_pickle'], 
                  extensions=['png'])
def test_complete():
    fig = plt.figure('Figure with a label?')
    
    plt.suptitle('Can you fit any more in a figure?')
    
    # make some arbitrary data
    x, y = np.arange(8), np.arange(10)
    data = u = v = np.linspace(0, 10, 80).reshape(10, 8)
    v = np.sin(v * -0.6)
    
    plt.subplot(3,3,1)
    plt.plot(range(10))
    
    plt.subplot(3, 3, 2)
    plt.contourf(data, hatches=['//', 'ooo'])
#    plt.colorbar()  # sadly, colorbar is currently failing. This might be an easy fix once
    # its been identified what the problem is. (lambda functions in colorbar)
    
    plt.subplot(3, 3, 3)
    plt.pcolormesh(data)
#    cb = plt.colorbar()
    
    plt.subplot(3, 3, 4)
    plt.imshow(data)
    
    plt.subplot(3, 3, 5)
    plt.pcolor(data)
    
    plt.subplot(3, 3, 6)
    plt.streamplot(x, y, u, v)
    
    plt.subplot(3, 3, 7)
    plt.quiver(x, y, u, v)
    
    plt.subplot(3, 3, 8)
    plt.scatter(x, x**2, label='$x^2$')
#    plt.legend()
    
    plt.subplot(3, 3, 9)
    plt.errorbar(x, x * -0.5, xerr=0.2, yerr=0.4)
    
    
    result_fh = StringIO()
#    recursive_pickle(fig, 'figure')
    pickle.dump(fig, result_fh, pickle.HIGHEST_PROTOCOL)
    
    plt.close('all')
    
    # make doubly sure that there are no figures left
    assert_equal(plt._pylab_helpers.Gcf.figs, {})
    
    # wind back the fh and load in the figure
    result_fh.seek(0)
    fig = pickle.load(result_fh)
    
    # make sure there is now a figure manager
    assert_not_equal(plt._pylab_helpers.Gcf.figs, {})
    
    assert_equal(fig.get_label(), 'Figure with a label?')
    
    
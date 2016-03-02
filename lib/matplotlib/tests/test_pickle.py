from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import cPickle as pickle
from matplotlib.externals.six.moves import xrange

from io import BytesIO

from nose.tools import assert_equal, assert_not_equal
import numpy as np

from matplotlib.testing.decorators import cleanup, image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def depth_getter(obj,
                 current_depth=0,
                 depth_stack=None,
                 nest_info='top level object'):
    """
    Returns a dictionary mapping:

        id(obj): (shallowest_depth, obj, nest_info)

    for the given object (and its subordinates).

    This, in conjunction with recursive_pickle, can be used to debug
    pickling issues, although finding others is sometimes a case of
    trial and error.

    """
    if depth_stack is None:
        depth_stack = {}

    if id(obj) in depth_stack:
        stack = depth_stack[id(obj)]
        if stack[0] > current_depth:
            del depth_stack[id(obj)]
        else:
            return depth_stack

    depth_stack[id(obj)] = (current_depth, obj, nest_info)

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            depth_getter(item, current_depth=current_depth + 1,
                         depth_stack=depth_stack,
                         nest_info=('list/tuple item #%s in '
                                    '(%s)' % (i, nest_info)))
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

        for key, value in six.iteritems(state):
            depth_getter(value, current_depth=current_depth + 1,
                         depth_stack=depth_stack,
                         nest_info=('attribute "%s" in '
                                    '(%s)' % (key, nest_info)))

    return depth_stack


def recursive_pickle(top_obj):
    """
    Recursively pickle all of the given objects subordinates, starting with
    the deepest first. **Very** handy for debugging pickling issues, but
    also very slow (as it literally pickles each object in turn).

    Handles circular object references gracefully.

    """
    objs = depth_getter(top_obj)
    # sort by depth then by nest_info
    objs = sorted(six.itervalues(objs), key=lambda val: (-val[0], val[2]))

    for _, obj, location in objs:
        try:
            pickle.dump(obj, BytesIO(), pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            print(obj)
            print('Failed to pickle %s. \n Type: %s. Traceback '
                  'follows:' % (location, type(obj)))
            raise


@cleanup
def test_simple():
    fig = plt.figure()
    # un-comment to debug
#    recursive_pickle(fig)
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)

    ax = plt.subplot(121)
    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

    ax = plt.axes(projection='polar')
    plt.plot(list(xrange(10)), label='foobar')
    plt.legend()

    # Uncomment to debug any unpicklable objects. This is slow so is not
    # uncommented by default.
#    recursive_pickle(fig)
    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

#    ax = plt.subplot(121, projection='hammer')
#    recursive_pickle(ax, 'figure')
#    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

    plt.figure()
    plt.bar(left=list(xrange(10)), height=list(xrange(10)))
    pickle.dump(plt.gca(), BytesIO(), pickle.HIGHEST_PROTOCOL)

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(list(xrange(10)))
    ax.set_yscale('log')
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)


@cleanup
@image_comparison(baseline_images=['multi_pickle'],
                  extensions=['png'], remove_text=True)
def test_complete():
    fig = plt.figure('Figure with a label?', figsize=(10, 6))

    plt.suptitle('Can you fit any more in a figure?')

    # make some arbitrary data
    x, y = np.arange(8), np.arange(10)
    data = u = v = np.linspace(0, 10, 80).reshape(10, 8)
    v = np.sin(v * -0.6)

    plt.subplot(3, 3, 1)
    plt.plot(list(xrange(10)))

    plt.subplot(3, 3, 2)
    plt.contourf(data, hatches=['//', 'ooo'])
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.pcolormesh(data)

    plt.subplot(3, 3, 4)
    plt.imshow(data)

    plt.subplot(3, 3, 5)
    plt.pcolor(data)

    ax = plt.subplot(3, 3, 6)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.streamplot(x, y, u, v)

    ax = plt.subplot(3, 3, 7)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.quiver(x, y, u, v)

    plt.subplot(3, 3, 8)
    plt.scatter(x, x**2, label='$x^2$')
    plt.legend(loc='upper left')

    plt.subplot(3, 3, 9)
    plt.errorbar(x, x * -0.5, xerr=0.2, yerr=0.4)

    ###### plotting is done, now test its pickle-ability #########

    # Uncomment to debug any unpicklable objects. This is slow (~200 seconds).
#    recursive_pickle(fig)

    result_fh = BytesIO()
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


@cleanup
def test_no_pyplot():
    # tests pickle-ability of a figure not created with pyplot
    from matplotlib.backends.backend_pdf import FigureCanvasPdf as fc
    from matplotlib.figure import Figure

    fig = Figure()
    _ = fc(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)


@cleanup
def test_renderer():
    from matplotlib.backends.backend_agg import RendererAgg
    renderer = RendererAgg(10, 20, 30)
    pickle.dump(renderer, BytesIO())


@cleanup
def test_image():
    # Prior to v1.4.0 the Image would cache data which was not picklable
    # once it had been drawn.
    from matplotlib.backends.backend_agg import new_figure_manager
    manager = new_figure_manager(1000)
    fig = manager.canvas.figure
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.arange(12).reshape(3, 4))
    manager.canvas.draw()
    pickle.dump(fig, BytesIO())


@cleanup
def test_grid():
    from matplotlib.backends.backend_agg import new_figure_manager
    manager = new_figure_manager(1000)
    fig = manager.canvas.figure
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    # Drawing the grid triggers instance methods to be attached
    # to the Line2D object (_lineFunc).
    manager.canvas.draw()

    pickle.dump(ax, BytesIO())


@cleanup
def test_polar():
    ax = plt.subplot(111, polar=True)
    fig = plt.gcf()
    result = BytesIO()
    pf = pickle.dumps(fig)
    pickle.loads(pf)
    plt.draw()


class TransformBlob(object):
    def __init__(self):
        self.identity = mtransforms.IdentityTransform()
        self.identity2 = mtransforms.IdentityTransform()
        # Force use of the more complex composition.
        self.composite = mtransforms.CompositeGenericTransform(
            self.identity,
            self.identity2)
        # Check parent -> child links of TransformWrapper.
        self.wrapper = mtransforms.TransformWrapper(self.composite)
        # Check child -> parent links of TransformWrapper.
        self.composite2 = mtransforms.CompositeGenericTransform(
            self.wrapper,
            self.identity)


def test_transform():
    obj = TransformBlob()
    pf = pickle.dumps(obj)
    del obj

    obj = pickle.loads(pf)
    # Check parent -> child links of TransformWrapper.
    assert_equal(obj.wrapper._child, obj.composite)
    # Check child -> parent links of TransformWrapper.
    assert_equal(
        [v() for v in obj.wrapper._parents.values()], [obj.composite2])
    # Check input and output dimensions are set as expected.
    assert_equal(obj.wrapper.input_dims, obj.composite.input_dims)
    assert_equal(obj.wrapper.output_dims, obj.composite.output_dims)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s'])

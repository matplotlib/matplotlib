from matplotlib.backend_bases import (
    FigureCanvasBase, LocationEvent, RendererBase, MouseEvent)
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.path as path
from matplotlib.testing.decorators import check_figures_equal
from mpl_toolkits.axes_grid1 import AxesGrid

import numpy as np
import pytest


def test_uses_per_path():
    id = transforms.Affine2D()
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]
    tforms = [id.rotate(i) for i in range(1, 5)]
    offsets = np.arange(20).reshape((10, 2))
    facecolors = ['red', 'green']
    edgecolors = ['red', 'green']

    def check(master_transform, paths, all_transforms,
              offsets, facecolors, edgecolors):
        rb = RendererBase()
        raw_paths = list(rb._iter_collection_raw_paths(
            master_transform, paths, all_transforms))
        gc = rb.new_gc()
        ids = [path_id for xo, yo, path_id, gc0, rgbFace in
               rb._iter_collection(gc, master_transform, all_transforms,
                                   range(len(raw_paths)), offsets,
                                   transforms.IdentityTransform(),
                                   facecolors, edgecolors, [], [], [False],
                                   [], 'data')]
        uses = rb._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        if raw_paths:
            seen = np.bincount(ids, minlength=len(raw_paths))
            assert set(seen).issubset([uses - 1, uses])

    check(id, paths, tforms, offsets, facecolors, edgecolors)
    check(id, paths[0:1], tforms, offsets, facecolors, edgecolors)
    check(id, [], tforms, offsets, facecolors, edgecolors)
    check(id, paths, tforms[0:1], offsets, facecolors, edgecolors)
    check(id, paths, [], offsets, facecolors, edgecolors)
    for n in range(0, offsets.shape[0]):
        check(id, paths, tforms, offsets[0:n, :], facecolors, edgecolors)
    check(id, paths, tforms, offsets, [], edgecolors)
    check(id, paths, tforms, offsets, facecolors, [])
    check(id, paths, tforms, offsets, [], [])
    check(id, paths, tforms, offsets, facecolors[0:1], edgecolors)


def test_get_default_filename(tmpdir):
    plt.rcParams['savefig.directory'] = str(tmpdir)
    fig = plt.figure()
    canvas = FigureCanvasBase(fig)
    filename = canvas.get_default_filename()
    assert filename == 'image.png'


@pytest.mark.backend('pdf')
def test_non_gui_warning():
    plt.subplots()
    with pytest.warns(UserWarning) as rec:
        plt.show()
        assert len(rec) == 1
        assert ('Matplotlib is currently using pdf, which is a non-GUI backend'
                in str(rec[0].message))

    with pytest.warns(UserWarning) as rec:
        plt.gcf().show()
        assert len(rec) == 1
        assert ('Matplotlib is currently using pdf, which is a non-GUI backend'
                in str(rec[0].message))


def test_location_event_position():
    # LocationEvent should cast its x and y arguments
    # to int unless it is None
    fig = plt.figure()
    canvas = FigureCanvasBase(fig)
    test_positions = [(42, 24), (None, 42), (None, None),
                      (200, 100.01), (205.75, 2.0)]
    for x, y in test_positions:
        event = LocationEvent("test_event", canvas, x, y)
        if x is None:
            assert event.x is None
        else:
            assert event.x == int(x)
            assert isinstance(event.x, int)
        if y is None:
            assert event.y is None
        else:
            assert event.y == int(y)
            assert isinstance(event.y, int)


class ButtonTest():
    def __init__(self, fig_test, fig_ref, button=None, plots=None,
                 layout='none'):

        """
        Class for testing the home, back and forwards buttons togheter with
        layout managers for several types of "subplots".

        Parameters
        ----------
        fig_test, fig_ref: `.figure.figures`
            The figure to compare

        button_call: {'home', 'back'}
            Which buttons to test

        plots: {'subplots', 'subplots_gridspec', 'add_subplopt', 'add_axes' \
'axesgrid'}
            Which plot configuration to test.

        pause : float
            The pause time between events.
        """
        self.rows_cols = 1, 2
        self.figs = fig_test, fig_ref
        self.trans = None, None
        self.layout = layout
        self.zoom0 = (0.5, 0.5), (0.5, 0.5)
        self.zoom1 = (0.5, 0.5), (0.6, 0.54)
        self.zoom2 = (0.52, 0.52), (0.58, 0.53)

        self.set_layout(True)
        self.add_subplots(plots)
        self.add_suptitle('Test button: {}, axes: {}, layout manager: {}'.
                          format(button, plots, layout))

        if button == 'home':
            self.home_button()
        elif button == 'back':
            self.back_button()

        # set layout managers to False so that the figures don't change
        # during saving.
        self.set_layout(False)

    def add_suptitle(self, title):
        for fig in self.figs:
            fig.suptitle(title)

    def set_layout(self, bool_):
        for fig in self.figs:
            if self.layout == 'constrained':
                fig.set_constrained_layout(bool_)
            elif self.layout == 'tight':
                arg = dict(rect=(0, 0, 1, 0.95)) if bool_ else bool_
                fig.set_tight_layout(arg)

    def add_subplots(self, plots):
        rows, cols = self.rows_cols
        self.trans = []
        for fig in self.figs:
            if plots == 'subplots':
                ax = fig.subplots(rows, cols, squeeze=False)[rows-1, cols-1]
            elif plots == 'subplots_gridspec':
                ax = fig.subplots(rows, cols, squeeze=False,
                                  gridspec_kw={'left': 0.01})[rows-1, cols-1]
            elif plots == 'add_subplot':
                for i in range(1, rows*cols + 1):
                    ax = fig.add_subplot(rows, cols, i)
            elif plots == 'add_axes':
                width = (1-0.1)/cols-0.1
                height = (1-0.1)/rows-0.1
                for i in range(rows):
                    for j in range(cols):
                        x0 = j*(width+0.1)+0.1
                        y0 = i*(height+0.1)+0.1
                        ax = fig.add_axes((x0, y0, width, height))
            elif plots == 'axesgrid':
                ax = AxesGrid(fig, 111, nrows_ncols=(2, 2),
                              axes_pad=0.1)[-1]

            self.trans.append(ax.transData.transform)
        self.draw()

    def draw(self):
        for fig in self.figs:
            fig.canvas.flush_events()

    def home_button(self):
        """Zoom twice and get back to home with the home button."""

        fig_test, fig_ref = self.figs
        trans_test, trans_ref = self.trans
        #No zoom happens but this is sometimes necessary to get equal results
        self.zoom_event(fig_ref, trans_ref, self.zoom0)
        self.zoom_event(fig_test, trans_test, self.zoom1)
        self.zoom_event(fig_test, trans_test, self.zoom2)

        self.move(fig_test, 'home')

    def back_button(self):
        """
        Zoom once in the ref figure, zoom twice in the test figure and use
        the back button twice followed by the forward button
        """
        fig_test, fig_ref = self.figs
        trans_test, trans_ref = self.trans

        self.zoom_event(fig_ref, trans_ref, self.zoom1)
        self.zoom_event(fig_test, trans_test, self.zoom1)
        self.zoom_event(fig_test, trans_test, self.zoom2)

        self.move(fig_test, 'back')
        self.move(fig_test, 'back')
        self.move(fig_test, 'forward')

    def zoom_event(self, fig, trans, zoom):
        """Simulate a zoom event from zoom[0] to zoom[1]"""
        xy1, xy2 = trans(zoom[0]), trans(zoom[1])
        press_zoom = MouseEvent('', fig.canvas, xy1[0], xy1[1], button=1)
        drag = MouseEvent('', fig.canvas, xy2[0], xy2[1], button=1)
        release_zoom = MouseEvent('', fig.canvas, xy2[0], xy2[1], button=1)

        fig.canvas.toolbar.press_zoom(press_zoom)
        fig.canvas.toolbar.drag_zoom(drag)
        fig.canvas.toolbar.release_zoom(release_zoom)
        self.draw()

    def move(self, fig, direction):
        """Simulate the back or forward button on fig"""
        getattr(fig.canvas.toolbar, direction)()
        self.draw()


"""
The tests are designed to test some known failures at writting time and that
there are no problem with the actual implementation for different use cases

test_back_button 2 and 4 and test_home_button 2 fails on master.
The home_button tests would probably fail on v2.1.0
"""

list_tests = [('back', 'subplots', 'none'),
              ('back', 'subplots', 'tight'),
              ('back', 'subplots', 'constrained'),
              ('back', 'add_subplot', 'tight'),
              ('back', 'add_axes', 'none'),
              ('back', 'axesgrid', 'none'),
              ('back', 'subplots_gridspec', 'none'),
              ('home', 'subplots', 'none'),
              ('home', 'subplots', 'tight'),
              ('home', 'subplots', 'constrained')]


for button, plots, layout in list_tests:
    @pytest.mark.backend('QT5Agg')
    @check_figures_equal(extensions=["png"])
    def b(fig_test, fig_ref):
        ButtonTest(fig_test, fig_ref, button=button,
                   plots=plots, layout=layout)
    name = 'test_{}_{}_{}'.format(button, plots, layout)
    locals()[name]=b
#    test_b.__name__ = 'test_{}_{}_{}'.format(button, plots, layout)



#@pytest.mark.parametrize('button, subplots, layout', list_tests)
#def test_wrapper(fig_test, fig_ref, button, subplots, layout):
#    def test_buttons(fig_test, fig_ref):
#        ButtonTest(fig_test, fig_ref, 'back', plots='subplots', layout='none')
#    return test_buttons
##def test_back_button1(fig_test, fig_ref, button, subplots, layout):
##    ButtonTest(fig_test, fig_ref, 'back', plots='subplots', layout='none')


#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_back_button2(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'back', plots='subplots', layout='tight')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_back_button3(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'back', plots='subplots',
#               layout='constrained')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_back_button4(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'back', plots='add_subplot', layout='tight')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_back_button5(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'back', plots='add_axes', layout='none')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_back_button6(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'back', plots='axesgrid', layout='none')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_back_button7(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'back', plots='subplots_gridspec',
#               layout='none')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_home_button1(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'home', plots='subplots', layout='none')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_home_button2(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'home', plots='subplots', layout='tight')
#
#
#@pytest.mark.backend('QT5Agg')
#@check_figures_equal(extensions=["png"])
#def test_home_button3(fig_test, fig_ref):
#    ButtonTest(fig_test, fig_ref, 'home', plots='subplots',
#               layout='constrained')

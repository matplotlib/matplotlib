from matplotlib import rcParams, rcParamsDefault
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase


def _get_cmap_norms():
    """
    Define a colormap and appropriate norms for each of the four
    possible settings of the extend keyword.

    Helper function for _colorbar_extension_shape and
    colorbar_extension_length.
    """
    # Create a color map and specify the levels it represents.
    cmap = get_cmap("RdBu", lut=5)
    clevs = [-5., -2.5, -.5, .5, 1.5, 3.5]
    # Define norms for the color maps.
    norms = dict()
    norms['neither'] = BoundaryNorm(clevs, len(clevs)-1)
    norms['min'] = BoundaryNorm([-10]+clevs[1:], len(clevs)-1)
    norms['max'] = BoundaryNorm(clevs[:-1]+[10], len(clevs)-1)
    norms['both'] = BoundaryNorm([-10]+clevs[1:-1]+[10], len(clevs)-1)
    return cmap, norms


def _colorbar_extension_shape(spacing):
    '''
    Produce 4 colorbars with rectangular extensions for either uniform
    or proportional spacing.

    Helper function for test_colorbar_extension_shape.
    '''
    # Get a colormap and appropriate norms for each extension type.
    cmap, norms = _get_cmap_norms()
    # Create a figure and adjust whitespace for subplots.
    fig = plt.figure()
    fig.subplots_adjust(hspace=4)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # Get the appropriate norm and use it to get colorbar boundaries.
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        # Create a subplot.
        cax = fig.add_subplot(4, 1, i+1)
        # Turn off text and ticks.
        for item in cax.get_xticklabels() + cax.get_yticklabels() +\
                cax.get_xticklines() + cax.get_yticklines():
            item.set_visible(False)
        # Generate the colorbar.
        cb = ColorbarBase(cax, cmap=cmap, norm=norm,
                boundaries=boundaries, values=values,
                extend=extension_type, extendrect=True,
                orientation='horizontal', spacing=spacing)
    # Return the figure to the caller.
    return fig


def _colorbar_extension_length(spacing):
    '''
    Produce 12 colorbars with variable length extensions for either
    uniform or proportional spacing.

    Helper function for test_colorbar_extension_length.
    '''
    # Get a colormap and appropriate norms for each extension type.
    cmap, norms = _get_cmap_norms()
    # Create a figure and adjust whitespace for subplots.
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # Get the appropriate norm and use it to get colorbar boundaries.
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        for j, extendfrac in enumerate((None, 'auto', 0.1)):
            # Create a subplot.
            cax = fig.add_subplot(12, 1, i*3+j+1)
            # Turn off text and ticks.
            for item in cax.get_xticklabels() + cax.get_yticklabels() +\
                    cax.get_xticklines() + cax.get_yticklines():
                item.set_visible(False)
            # Generate the colorbar.
            cb = ColorbarBase(cax, cmap=cmap, norm=norm,
                    boundaries=boundaries, values=values,
                    extend=extension_type, extendfrac=extendfrac,
                    orientation='horizontal', spacing=spacing)
    # Return the figure to the caller.
    return fig


@image_comparison(
        baseline_images=['colorbar_extensions_shape_uniform',
                         'colorbar_extensions_shape_proportional'],
        extensions=['png'])
def test_colorbar_extension_shape():
    '''Test rectangular colorbar extensions.'''
    # Use default params so matplotlibrc doesn't cause the test to fail.
    rcParams.update(rcParamsDefault)
    # Create figures for uniform and proportionally spaced colorbars.
    fig1 = _colorbar_extension_shape('uniform')
    fig2 = _colorbar_extension_shape('proportional')


@image_comparison(
        baseline_images=['colorbar_extensions_uniform', 'colorbar_extensions_proportional'],
        extensions=['png'])
def test_colorbar_extension_length():
    '''Test variable length colorbar extensions.'''
    # Use default params so matplotlibrc doesn't cause the test to fail.
    rcParams.update(rcParamsDefault)
    # Create figures for uniform and proportionally spaced colorbars.
    fig1 = _colorbar_extension_length('uniform')
    fig2 = _colorbar_extension_length('proportional')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)


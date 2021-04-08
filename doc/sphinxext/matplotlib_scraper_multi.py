import sphinx_gallery.scrapers as scrapers
import os
from textwrap import indent
from distutils.version import LooseVersion

def _anim_rst(anim, image_path, gallery_conf):
    import matplotlib
    from matplotlib.animation import FFMpegWriter, ImageMagickWriter
    # output the thumbnail as the image, as it will just be copied
    # if it's the file thumbnail
    fig = anim._fig
    image_path = image_path.replace('.png', '.gif')
    fig_size = fig.get_size_inches()
    thumb_size = gallery_conf['thumbnail_size']
    use_dpi = round(
        min(t_s / f_s for t_s, f_s in zip(thumb_size, fig_size)))
    # FFmpeg is buggy for GIFs before Matplotlib 3.3.1
    if LooseVersion(matplotlib.__version__) >= LooseVersion('3.3.1') and \
            FFMpegWriter.isAvailable():
        writer = 'ffmpeg'
    elif ImageMagickWriter.isAvailable():
        writer = 'imagemagick'
    else:
        writer = None
    anim.save(image_path, writer=writer, dpi=use_dpi)
    html = anim._repr_html_()
    if html is None:  # plt.rcParams['animation.html'] == 'none'
        html = anim.to_jshtml()
    html = indent(html, '     ')
    return _ANIMATION_RST.format(html)

_ANIMATION_RST = '''
.. container:: sphx-glr-animation

    .. raw:: html

        {0}
'''

def matplotlib_scraper_multi(block, block_vars, gallery_conf, **kwargs):
    """Scrape Matplotlib images, but with both high and low-def...

    Parameters
    ----------
    block : tuple
        A tuple containing the (label, content, line_number) of the block.
    block_vars : dict
        Dict of block variables.
    gallery_conf : dict
        Contains the configuration of Sphinx-Gallery
    **kwargs : dict
        Additional keyword arguments to pass to
        :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.
        The ``format`` kwarg in particular is used to set the file extension
        of the output file (currently only 'png', 'jpg', and 'svg' are
        supported).

    Returns
    -------
    rst : str
        The ReSTructuredText that will be rendered to HTML containing
        the images. This is often produced by :func:`figure_rst`.
    """
    matplotlib, plt = scrapers._import_matplotlib()
    from matplotlib.animation import Animation
    image_path_iterator = block_vars['image_path_iterator']
    image_rsts = []
    # Check for animations
    anims = list()
    if gallery_conf.get('matplotlib_animations', False):
        for ani in block_vars['example_globals'].values():
            if isinstance(ani, Animation):
                anims.append(ani)
    # Then standard images
    for fig_num, image_path in zip(plt.get_fignums(), image_path_iterator):
        if 'format' in kwargs:
            image_path = '%s.%s' % (os.path.splitext(image_path)[0],
                                    kwargs['format'])
        # Set the fig_num figure as the current figure as we can't
        # save a figure that's not the current figure.
        fig = plt.figure(fig_num)
        # Deal with animations
        cont = False
        for anim in anims:
            if anim._fig is fig:
                image_rsts.append(_anim_rst(anim, image_path, gallery_conf))
                cont = True
                break
        if cont:
            continue
        # get fig titles
        fig_titles = scrapers._matplotlib_fig_titles(fig)
        to_rgba = matplotlib.colors.colorConverter.to_rgba
        # shallow copy should be fine here, just want to avoid changing
        # "kwargs" for subsequent figures processed by the loop
        these_kwargs = kwargs.copy()
        hikwargs = kwargs.copy()
        hikwargs['dpi']=200
        
        for attr in ['facecolor', 'edgecolor']:
            fig_attr = getattr(fig, 'get_' + attr)()
            default_attr = matplotlib.rcParams['figure.' + attr]
            if to_rgba(fig_attr) != to_rgba(default_attr) and \
                    attr not in kwargs:
                these_kwargs[attr] = fig_attr
        try:
            hipath = image_path[:-4]+'_hidpi' + image_path[-4:]
            fig.savefig(hipath, **hikwargs)
            fig.savefig(image_path, **these_kwargs)
            
        except Exception:
            plt.close('all')
            raise
        if 'images' in gallery_conf['compress_images']:
            optipng(image_path, gallery_conf['compress_images_args'])
        image_rsts.append(
            figure_rst([image_path], gallery_conf['src_dir'], fig_titles))
    plt.close('all')
    rst = ''
    if len(image_rsts) == 1:
        rst = image_rsts[0]
    elif len(image_rsts) > 1:
        image_rsts = [re.sub(r':class: sphx-glr-single-img',
                             ':class: sphx-glr-multi-img',
                             image) for image in image_rsts]
        image_rsts = [HLIST_IMAGE_MATPLOTLIB + indent(image, u' ' * 6)
                      for image in image_rsts]
        rst = HLIST_HEADER + ''.join(image_rsts)
    return rst

import re

def figure_rst(figure_list, sources_dir, fig_titles=''):
    """Generate RST for a list of image filenames.

    Depending on whether we have one or more figures, we use a
    single rst call to 'image' or a horizontal list.

    Parameters
    ----------
    figure_list : list
        List of strings of the figures' absolute paths.
    sources_dir : str
        absolute path of Sphinx documentation sources
    fig_titles : str
        Titles of figures, empty string if no titles found. Currently
        only supported for matplotlib figures, default = ''.

    Returns
    -------
    images_rst : str
        rst code to embed the images in the document
    """

    figure_paths = [os.path.relpath(figure_path, sources_dir)
                    .replace(os.sep, '/').lstrip('/')
                    for figure_path in figure_list]
    # Get alt text
    alt = ''
    if fig_titles:
        alt = fig_titles
    elif figure_list:
        file_name = os.path.split(figure_list[0])[1]
        # remove ext & 'sphx_glr_' from start & n#'s from end
        file_name_noext = os.path.splitext(file_name)[0][9:-4]
        # replace - & _ with \s
        file_name_final = re.sub(r'[-,_]', ' ', file_name_noext)
        alt = file_name_final
    alt = scrapers._single_line_sanitize(alt)

    images_rst = ""
    if len(figure_paths) == 1:
        figure_name = '/' + figure_paths[0]
        figure_pre = figure_name[:-4]
        images_rst = SINGLE_IMAGE % (figure_pre, alt, figure_pre)
    elif len(figure_paths) > 1:
        images_rst = HLIST_HEADER
        for figure_name in figure_paths:
            figure_pre = figure_name[:-4]
            images_rst += HLIST_IMAGE_TEMPLATE % (figure_name, alt)
    return images_rst

HLIST_HEADER = """
.. rst-class:: sphx-glr-horizontal

"""

HLIST_IMAGE_MATPLOTLIB = """
    *
"""

HLIST_IMAGE_TEMPLATE = """
    *

      .. image:: /%s
          :alt: %s
          :class: sphx-glr-multi-img
"""

SINGLE_IMAGE = """
.. image-srcset:: %s.png
   :alt: %s
   :hidpi: %s_hidpi.png
   :class: sphx-glr-single-img
"""

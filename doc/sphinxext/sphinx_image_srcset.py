from pathlib import Path
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("sphinx-image-srcset")
except ImportError:
    __version__ = "0+unknown"

from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image
from docutils import statemachine

import os
from os.path import relpath

import shutil


def setup(app):
    ImageSrcset.app = app
    #setup.confdir = app.confdir
    setup.app = app
    app.add_directive("image-srcset", ImageSrcset)


class ImageSrcset(Directive):
    """
    Impliments a directive to allow an optional hidpi image.  If one is not
    available, it just uses the low-dpi image.

    e.g.:

    .. image-srcset:: /plot_types/basic/images/sphx_glr_bar_001.png
        :alt: bar
        :hidpi: /plot_types/basic/images/sphx_glr_bar_001_hidpi.png
        :class: sphx-glr-single-img
    
    The result is

    <img srcset="_images/sphx_glr_bar_001.png, sphx_glr_bar_001_hidpi.png 2x",
        src="sphx_glr_bar_001_hidpi.png"
        alt="bar"
        class="sphx-glr-single-img>

    See https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Responsive_images#resolution_switching_same_size_different_resolutions

    """
    has_content = True
    required_arguments = 1
    optional_arguments = 3
    final_argument_whitespace = False
    option_spec = {
        'alt': directives.unchanged,
        'hidpi': directives.unchanged,
        'class': directives.unchanged
    }
    def run(self):
        args = self.arguments
        imagenm = args[0]
        options = self.options
        hidpiimagenm = options.pop('hidpi', None)
        document = self.state_machine.document

        source_file_name = os.path.join(setup.app.builder.srcdir,
                                            directives.uri(args[0]))
        
        current_dir = os.path.dirname(self.state.document.current_source)

        # where will the html file go...
        source_rel_name = relpath(document.attributes['source'], setup.app.confdir)
        source_rel_dir = os.path.dirname(source_rel_name)
        dest_dir = os.path.abspath(os.path.join(self.app.builder.outdir,
                                            source_rel_dir))
        
        # where will the images go...
        image_dir = os.path.abspath(os.path.join(self.app.builder.outdir,
                                            '_images'))
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # get relative path between images and where the html will be:
        image_rel = relpath(image_dir, dest_dir)

        # Copy image files:    
        for fn in [imagenm, hidpiimagenm]:
            if fn is not None:          
                base = os.path.basename(fn)  
                sourceimg = os.path.join(current_dir, 'images', base)
                destimg = os.path.join(image_dir, base)
                shutil.copyfile(sourceimg, destimg)
            
        #    <img srcset="elva-fairy-320w.jpg,
        #            elva-fairy-480w.jpg 1.5x,
        #             elva-fairy-640w.jpg 2x"
        #     src="elva-fairy-640w.jpg"
        #     alt="Elva dressed as a fairy">
        
        # write html...
        imagebase = os.path.basename(imagenm)
        baseimage = os.path.join(image_rel, imagebase)
        hiimage = os.path.join(image_rel, os.path.basename(hidpiimagenm))
        alt = options.pop('alt', '')
        classst = options.pop('class', None)
        if classst is not None:
            classst = f'class="{classst}"'
        else:
            classst = ''
        result = f'.. raw:: html\n\n    <img srcset="{baseimage}, {hiimage} 2x" src="{hiimage}" alt="{alt}" {classst}>'
        result = statemachine.string2lines(result, convert_whitespace=True)
        self.state_machine.insert_input(result, source=source_file_name)

        return []



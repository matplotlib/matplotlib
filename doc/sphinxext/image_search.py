from __future__ import division, print_function, absolute_import
import codecs
import copy
from datetime import timedelta, datetime
from difflib import get_close_matches
from importlib import import_module
import re
import os
import pathlib
from xml.sax.saxutils import quoteattr, escape
from itertools import chain
from collections import defaultdict
import json
import logging
from pathlib import Path

from docutils.utils import get_source_line
from docutils import nodes

from sphinx.util import logging as sphinx_logging
from sphinx.errors import ExtensionError
from sphinx_gallery import gen_gallery
from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.gen_rst import extract_intro_and_title
from sphinx_gallery.backreferences import BACKREF_THUMBNAIL_TEMPLATE, _thumbnail_div, THUMBNAIL_PARENT_DIV, THUMBNAIL_PARENT_DIV_CLOSE
from sphinx_gallery.scrapers import _find_image_ext

logger = sphinx_logging.getLogger(__name__)


# id="imgsearchref-{ref_name}" attribute is used by the js file later
# to programmatically hide or unhide thumbnails depending on search result
THUMBNAIL_TEMPLATE = """
.. raw:: html

    <div class="sphx-glr-imgsearch-resultelement" id="imgsearchref-({ref_name})">
    <div class="sphx-glr-thumbcontainer" tooltip="{snippet}">

.. only:: html

  .. image:: /{thumbnail}
    :alt:

  :ref:`sphx_glr_{ref_name}`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">{title}</div>
    </div>
    </div>

"""

def _thumbnail_div(target_dir, src_dir, fname, snippet, title,
                   is_backref=False, check=True):
    """Generate RST to place a thumbnail in a gallery."""
    thumb, _ = _find_image_ext(
        os.path.join(target_dir, 'images', 'thumb',
                     'sphx_glr_%s_thumb.png' % fname[:-3]))
    if check and not os.path.isfile(thumb):
        # This means we have done something wrong in creating our thumbnail!
        raise ExtensionError('Could not find internal Sphinx-Gallery thumbnail'
                             ' file:\n%s' % (thumb,))
    thumb = os.path.relpath(thumb, src_dir)
    full_dir = os.path.relpath(target_dir, src_dir)

    # Inside rst files forward slash defines paths
    thumb = thumb.replace(os.sep, "/")

    ref_name = os.path.join(full_dir, fname).replace(os.path.sep, '_')

    template = BACKREF_THUMBNAIL_TEMPLATE if is_backref else THUMBNAIL_TEMPLATE
    return template.format(snippet=escape(snippet),
                           thumbnail=thumb, title=title, ref_name=ref_name)


def generate_search_page(app):
    """
    fetches all generated example images and adds links to them
    in image_search/index.recommendations file
    """

    gallery_conf = app.config.sphinx_gallery_conf   

    workdirs = gen_gallery._prepare_sphx_glr_dirs(gallery_conf,
                                      app.builder.srcdir)

    # imageSearch = ImageSearch()

    src_dir = app.builder.srcdir
    heading = "Image Search page"

    image_search_path = os.path.join(src_dir, "image_search")

    try:
        os.mkdir(image_search_path)
    except FileExistsError:
        pass
    
    f = open(os.path.join(image_search_path, "index.recommendations"), "w")
    f.write("\n\n" + heading + "\n")
    f.write("^" * len(heading) + "\n")

    # THUMBNAIL_PARENT_DIV can be modified to include search page speecific classnames
    # for applying custom css
    f.write(THUMBNAIL_PARENT_DIV)

    for examples_dir, gallery_dir in workdirs:

        examples_dir_abs_path = os.path.join(app.builder.srcdir, examples_dir)
        gallery_dir_abs_path = os.path.join(app.builder.srcdir, gallery_dir)

        # list all paths to subsection index files in this array
        subsecs = gen_gallery.get_subsections(app.builder.srcdir,
                                  examples_dir_abs_path, gallery_conf,
                                  check_for_index=True)

        directory_explore = [gallery_dir_abs_path] + subsecs
        # logger.info("directory_explore")
        # logger.info(directory_explore)

        rst_content = ""

        # loop through every subfolder
        for subsection in directory_explore:
            src_dir = os.path.join(gallery_dir_abs_path, subsection)
        
            # get filenames of files with .py extension
            listdir = [fname for fname in os.listdir(src_dir) if fname.endswith('.py')]
            fullpath = [os.path.join(src_dir, file_name) for file_name in listdir]
            

            for example in fullpath:

                # the name of example is determined by the name of the .py file
                example_name = example.split('/')[-1].replace('.py','')

                # name of the example image is generated in the following format
                # sphx_glr_{example_name}_{i where i = index of image in example}.png
                # for now i just tested with the first example
                example_image_path = os.path.join(os.path.join(src_dir, "images"), f"sphx_glr_{example_name}_001.png")
                
                if os.path.isfile(example_image_path):

                    _, script = split_code_and_text_blocks(example)
                    intro, title = extract_intro_and_title(example, script[0][1])
                    
                    # generates rst text with thumbnail and link to the example   
                    # _thumbnail_div needs to be modified to keep the thumbnail and link
                    # hidden by default so that it can be made visible by the js script later
                    thumbnail_rst = _thumbnail_div(
                        src_dir,
                        app.builder.srcdir,
                        f"{example_name}.py",
                        intro,
                        title
                    )

                    # add the thumbnail 
                    rst_content += thumbnail_rst


        f.write(rst_content)
        f.write(THUMBNAIL_PARENT_DIV_CLOSE)
        # f.close()



def setup(app):
    
    # need to decide on the priority
    app.connect("builder-inited", generate_search_page, priority=100)
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

import sphinx_gallery.gen_gallery as gen_gallery


from sphinx.util import logging as sphinx_logging


from collections import defaultdict
import json
import logging
from pathlib import Path

from docutils.utils import get_source_line
from docutils import nodes

from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.gen_rst import extract_intro_and_title
from sphinx_gallery.backreferences import _thumbnail_div, THUMBNAIL_PARENT_DIV, THUMBNAIL_PARENT_DIV_CLOSE


logger = sphinx_logging.getLogger(__name__)


class ImageSearch():
    def __init__(self, gallery_dir):
        self.gallery_dir = gallery_dir
        self.items = []

    def add_item(self, src_path, taget_path, image_dir):
        self.items.append((src_path, taget_path, image_dir))
        logger.info("Image path")
        logger.info(image_dir)
        # self.items.extend(item_dirs)

    def generate_files(self, src_dir):
        # src_dir = gallery_conf["src_dir"]
        
        include_path = os.path.join(self.gallery_dir, "%s.recommendationsssssss" % self.gallery_dir.split('/')[-1])
        logger.info("Include Path")
        logger.info(include_path)

        logger.info("self items")
        logger.info(self.items)

        with open(include_path, "w", encoding="utf-8") as ex_file:
            heading = "More related examples"
            ex_file.write("\n\n" + heading + "\n")
            ex_file.write("^" * len(heading) + "\n")
            ex_file.write(THUMBNAIL_PARENT_DIV)
            for src_path, taget_path, image_dir in self.items:
                rec_path, rec_name = taget_path.rsplit("/", maxsplit=1)
                
                _, script = split_code_and_text_blocks(taget_path)
                intro, title = extract_intro_and_title(taget_path, script[0][1])

                logger.info("rec_path")
                logger.info(rec_path)
                logger.info("rec_path")
                logger.info(rec_name)
                logger.info("src_dir")
                logger.info(src_dir)
                
                ex_file.write(
                    _thumbnail_div(
                        rec_path,
                        src_dir,
                        rec_name,
                        intro,
                        title,
                        is_backref=True,
                    )
                )
            ex_file.write(THUMBNAIL_PARENT_DIV_CLOSE)
    # def fit(self):



def generate_search_page(app):

    gallery_conf = app.config.sphinx_gallery_conf   

    workdirs = gen_gallery._prepare_sphx_glr_dirs(gallery_conf,
                                      app.builder.srcdir)

    # Check for duplicate filenames to make sure linking works as expected
    examples_dirs = [ex_dir for ex_dir, _ in workdirs]

    # imageSearch = ImageSearch()

    src_dir = app.builder.srcdir
    heading = "Search results"

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

        all_py_examples = []
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
                    # imageSearch.add_item(example_image_path)
                    # logger.info("OK")
                    
                    # logger.info(example_image_path)
                    # logger.info(example)
                    # logger.info(app.builder.srcdir)
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

            all_py_examples.append(fullpath)

        f.write(rst_content)
        f.write(THUMBNAIL_PARENT_DIV_CLOSE)
        # f.close()

        all_py_examples = list(chain.from_iterable(all_py_examples))

        logger.info("all py examples")
        logger.info(all_py_examples)
        logger.info(examples_dir_abs_path)
        logger.info(gallery_dir_abs_path)

        


def setup(app):
    app.connect("builder-inited", generate_search_page)
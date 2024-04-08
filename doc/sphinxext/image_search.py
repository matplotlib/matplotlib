import os
import json
import pandas as pd
import numpy as np
import torch
import timm

from xml.sax.saxutils import escape
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable

from sphinx.util import logging as sphinx_logging
from sphinx.errors import ExtensionError
from sphinx_gallery import gen_gallery
from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.gen_rst import extract_intro_and_title
from sphinx_gallery.backreferences import BACKREF_THUMBNAIL_TEMPLATE, _thumbnail_div, THUMBNAIL_PARENT_DIV, THUMBNAIL_PARENT_DIV_CLOSE
from sphinx_gallery.scrapers import _find_image_ext


logger = sphinx_logging.getLogger(__name__)



class SearchSetup:
    """ A class for setting up and generating image vectors."""
    def __init__(self, model_name='vgg19', pretrained=True):
        """
        Parameters:
        -----------
        image_list : list
        A list of images to be indexed and searched.
        model_name : str, optional (default='vgg19')
        The name of the pre-trained model to use for feature extraction.
        pretrained : bool, optional (default=True)
        Whether to use the pre-trained weights for the chosen model.
        image_count : int, optional (default=None)
        The number of images to be indexed and searched. If None, all images in the image_list will be used.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_data = pd.DataFrame()
        self.d = None
        self.queue = []

        base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()   # disables gradient computation


    def _extract(self, img):
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        # Extract features
        feature = self.model(x)
        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)

    def _get_feature(self, image_data: list):
        self.image_data = image_data
        features = []
        for img_path in tqdm(self.image_data):  # Iterate through images
            # Extract features from the image
            try:
                feature = self._extract(img=Image.open(img_path))
                print(feature)
                features.append(feature)
            except:
                # If there is an error, append None to the feature list
                features.append(None)
                continue
        return features
    
    def add_image( self, thumbnail_id, image_path ):
        
        self.queue.append( (thumbnail_id, image_path) )

    def start_feature_extraction(self):
        data_df = pd.DataFrame()

        image_paths = list( map( lambda x:x[1], self.queue ) )
        data_df['image_path'] = image_paths

        features = self._get_feature(image_paths)
        data_df['feature'] = features

        data_df['thumbnail_id'] = list( map( lambda x:x[0], self.queue ) )

        f = open('./_static/data.json', "w")
        data_json = []
        for i in range(len(data_df)):
            data_json.append( [ data_df.loc[i, "thumbnail_id"], data_df.loc[i, "feature"].tolist() ] )

        f.write(json.dumps(data_json))





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
    return ( ref_name, template.format(snippet=escape(snippet),
                           thumbnail=thumb, title=title, ref_name=ref_name) )


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
    
    search_setup = SearchSetup(model_name='vgg19', pretrained=True)
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
                    ( ref_name, thumbnail_rst ) = _thumbnail_div(
                        src_dir,
                        app.builder.srcdir,
                        f"{example_name}.py",
                        intro,
                        title
                    )

                    search_setup.add_image( f"imgsearchref-({ref_name})", example_image_path )

                    # add the thumbnail 
                    rst_content += thumbnail_rst

        logger.info("STARTING FEATURE EXTRACTION")
        search_setup.start_feature_extraction()
        f.write(rst_content)
        f.write(THUMBNAIL_PARENT_DIV_CLOSE)
        # f.close()



def setup(app):
    
    # need to decide on the priority
    app.connect("builder-inited", generate_search_page, priority=100)

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
import numpy as np
from matplotlib.transforms import Affine2D, IdentityTransform

import matplotlib.font_manager as font_manager
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT, LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
import matplotlib.patches as mpatches

from matplotlib.offsetbox import AnnotationBbox,\
     AnchoredOffsetbox, AuxTransformBox

#from matplotlib.offsetbox import

from matplotlib.cbook import get_sample_data


class TextPatch(mpatches.PathPatch):

    FONT_SCALE = 100.

    def __init__(self, xy, s, size=None, prop=None, bbox_image=None,
                 *kl, **kwargs):
        if prop is None:
            prop = FontProperties()

        if size is None:
            size = prop.get_size_in_points()

        self._xy = xy
        self.set_size(size)

        self.text_path = self.text_get_path(prop, s)

        mpatches.PathPatch.__init__(self, self.text_path, *kl, **kwargs)

        self._init_bbox_image(bbox_image)


    def _init_bbox_image(self, im):

        if im is None:
            self.bbox_image = None
        else:
            bbox_image = BboxImage(self.get_window_extent,
                                   norm = None,
                                   origin=None,
                                   )
            bbox_image.set_transform(IdentityTransform())

            bbox_image.set_data(im)
            self.bbox_image = bbox_image

    def draw(self, renderer=None):


        if self.bbox_image is not None:
            # the clip path must be updated every draw. any solution? -JJ
            self.bbox_image.set_clip_path(self.text_path, self.get_transform())
            self.bbox_image.draw(renderer)

        mpatches.PathPatch.draw(self, renderer)


    def set_size(self, size):
        self._size = size

    def get_size(self):
        return self._size

    def get_patch_transform(self):
        tr = Affine2D().scale(self._size/self.FONT_SCALE, self._size/self.FONT_SCALE)
        return tr.translate(*self._xy)

    def glyph_char_path(self, glyph, currx=0.):

        verts, codes = [], []
        for step in glyph.path:
            if step[0] == 0:   # MOVE_TO
                verts.append((step[1], step[2]))
                codes.append(Path.MOVETO)
            elif step[0] == 1: # LINE_TO
                verts.append((step[1], step[2]))
                codes.append(Path.LINETO)
            elif step[0] == 2: # CURVE3
                verts.extend([(step[1], step[2]),
                               (step[3], step[4])])
                codes.extend([Path.CURVE3, Path.CURVE3])
            elif step[0] == 3: # CURVE4
                verts.extend([(step[1], step[2]),
                              (step[3], step[4]),
                              (step[5], step[6])])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            elif step[0] == 4: # ENDPOLY
                verts.append((0, 0,))
                codes.append(Path.CLOSEPOLY)

        verts = [(x+currx, y) for (x,y) in verts]

        return verts, codes


    def text_get_path(self, prop, s):

        fname = font_manager.findfont(prop)
        font = FT2Font(str(fname))

        font.set_size(self.FONT_SCALE, 72)

        cmap = font.get_charmap()
        lastgind = None

        currx = 0

        verts, codes = [], []

        for c in s:

            ccode = ord(c)
            gind = cmap.get(ccode)
            if gind is None:
                ccode = ord('?')
                gind = 0
            glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)


            if lastgind is not None:
                kern = font.get_kerning(lastgind, gind, KERNING_DEFAULT)
            else:
                kern = 0
            currx += (kern / 64.0) #/ (self.FONT_SCALE)

            verts1, codes1 = self.glyph_char_path(glyph, currx)
            verts.extend(verts1)
            codes.extend(codes1)


            currx += (glyph.linearHoriAdvance / 65536.0) #/ (self.FONT_SCALE)
            lastgind = gind

        return Path(verts, codes)

if 1:

    fig = plt.figure(1)

    # EXAMPLE 1

    ax = plt.subplot(211)

    from matplotlib._png import read_png
    fn = get_sample_data("lena.png", asfileobj=False)
    arr = read_png(fn)
    p = TextPatch((0, 0), "!?", size=150, fc="none", ec="k",
                  bbox_image=arr,
                  transform=IdentityTransform())
    p.set_clip_on(False)

    # make offset box
    offsetbox = AuxTransformBox(IdentityTransform())
    offsetbox.add_artist(p)

    # make anchored offset box
    ao = AnchoredOffsetbox(loc=2, child=offsetbox, frameon=True, borderpad=0.2)

    ax.add_artist(ao)



    # EXAMPLE 2

    ax = plt.subplot(212)

    shadow1 = TextPatch((3, -2), "TextPath", size=70, fc="none", ec="0.6", lw=3,
                   transform=IdentityTransform())
    shadow2 = TextPatch((3, -2), "TextPath", size=70, fc="0.3", ec="none",
                   transform=IdentityTransform())

    arr = np.arange(256).reshape(1,256)/256.
    text_path = TextPatch((0, 0), "TextPath", size=70, fc="none", ec="none", lw=1,
                          bbox_image=arr,
                          transform=IdentityTransform())

    # make offset box
    offsetbox = AuxTransformBox(IdentityTransform())
    offsetbox.add_artist(shadow1)
    offsetbox.add_artist(shadow2)
    offsetbox.add_artist(text_path)

    # place the anchored offset box using AnnotationBbox
    ab = AnnotationBbox(offsetbox, (0.5, 0.5),
                        xycoords='data',
                        boxcoords="offset points",
                        box_alignment=(0.5,0.5),
                        )


    ax.add_artist(ab)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)





    plt.draw()
    plt.show()

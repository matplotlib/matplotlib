import os
import matplotlib
from matplotlib.ft2font import FT2Font
import matplotlib.agg as agg

MOVETO, LINETO, CURVE3, CURVE4, ENDPOLY = range(5)

def glyph_to_agg_path(glyph):
    path = agg.path_storage()
    for tup in glyph.path:
        print tup
        code = tup[0]
        if code == MOVETO:
            x,y = tup[1:]
            path.move_to(x,y)
        elif code == LINETO:
            x,y = tup[1:]
            path.line_to(x,y)
        elif code == CURVE3:
            xctl, yctl, xto, yto= tup[1:]
            path.curve3(xctl, yctl, xto, yto)

        elif code == CURVE4:
            xctl1, yctl1, xctl2, yctl2, xto, yto= tup[1:]
            path.curve4(xctl1, yct1, xctl2, yctl2, xto, yto)
        elif code == ENDPOLY:
            path.end_poly()

    return path

width, height = 300,300
fname = os.path.join(matplotlib.get_data_path(), 'fonts/ttf/Vera.ttf')
font = FT2Font(fname)
glyph = font.load_char(ord('y'))
path = glyph_to_agg_path(glyph)

curve = agg.conv_curve_path(path)


scaling = agg.trans_affine_scaling(20,20)
translation = agg.trans_affine_translation(4,4)
rotation = agg.trans_affine_rotation(3.1415926)
mtrans = translation*scaling # cannot use this as a temporary
tpath = agg.conv_transform_path(path, mtrans)

curve = agg.conv_curve_trans(tpath)

stride = width*4
buffer = agg.buffer(width, height, stride)

rbuf = agg.rendering_buffer()
rbuf.attachb(buffer)

red = agg.rgba8(255,0,0,255)
blue = agg.rgba8(0,0,255,255)
white = agg.rgba8(255,255,255,255)

pf = agg.pixel_format_rgba(rbuf)
rbase = agg.renderer_base_rgba(pf)
rbase.clear_rgba8(white)

renderer =  agg.renderer_scanline_aa_solid_rgba(rbase);


rasterizer = agg.rasterizer_scanline_aa()
scanline = agg.scanline_p8()

# first fill
rasterizer.add_path(curve)
renderer.color_rgba8(blue)
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

# then stroke
stroke = agg.conv_stroke_curvetrans(curve)
stroke.width(2.0)
renderer.color_rgba8( red )
rasterizer.add_path(stroke)
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

s = buffer.to_string()
import Image
im = Image.fromstring( "RGBA", (width, height), s)
im.show()


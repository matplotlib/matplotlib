# this example uses the agg python module directly there is no
# documentation -- you have to know how to use the agg c++ API to use
# it
import matplotlib.agg as agg
from math import pi

## Define some colors
red = agg.rgba8(255,0,0,255)
blue = agg.rgba8(0,0,255,255)
green = agg.rgba8(0,255,0,255)
black = agg.rgba8(0,0,0,255)
white = agg.rgba8(255,255,255,255)
yellow = agg.rgba8(192,192,255,255)

## Create the rendering buffer, rasterizer, etc
width, height = 600,400
stride = width*4
buffer = agg.buffer(width, height, stride)

rbuf = agg.rendering_buffer()
rbuf.attachb(buffer)

pf = agg.pixel_format_rgba(rbuf)
rbase = agg.renderer_base_rgba(pf)
rbase.clear_rgba8(blue)

renderer =  agg.renderer_scanline_aa_solid_rgba(rbase);
renderer.color_rgba8( red )
rasterizer = agg.rasterizer_scanline_aa()
scanline = agg.scanline_p8()

## A polygon path
path = agg.path_storage()
path.move_to(10,10)
path.line_to(100,100)
path.line_to(200,200)
path.line_to(100,200)
path.close_polygon()

# stroke it
stroke = agg.conv_stroke_path(path)
stroke.width(3.0)
rasterizer.add_path(stroke)
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

## A curved path
path = agg.path_storage()
path.move_to(200,10)
path.line_to(350,50)
path.curve3(150,200)
path.curve3(100,70)
path.close_polygon()
curve = agg.conv_curve_path(path)

# fill it
rasterizer.add_path(curve)
renderer.color_rgba8( green )
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

# and stroke it
stroke = agg.conv_stroke_curve(curve)
stroke.width(5.0)
rasterizer.add_path(stroke)
renderer.color_rgba8( yellow )
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

## Transforming a path
path = agg.path_storage()
path.move_to(0,0)
path.line_to(1,0)
path.line_to(1,1)
path.line_to(0,1)
path.close_polygon()

rotation = agg.trans_affine_rotation(pi/4)
scaling = agg.trans_affine_scaling(30,30)
translation = agg.trans_affine_translation(300,300)
trans = rotation*scaling*translation

transpath = agg.conv_transform_path(path, trans)
stroke = agg.conv_stroke_transpath(transpath)
stroke.width(2.0)
rasterizer.add_path(stroke)
renderer.color_rgba8( black )
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

## Converting a transformed path to a curve
path = agg.path_storage()
path.move_to(0,0)
path.curve3(1,0)
path.curve3(1,1)
path.curve3(0,1)
path.close_polygon()

rotation = agg.trans_affine_rotation(pi/4)
scaling = agg.trans_affine_scaling(30,30)
translation = agg.trans_affine_translation(300,250)
trans = rotation*scaling*translation
trans.flip_y()

transpath = agg.conv_transform_path(path, trans)
curvetrans = agg.conv_curve_trans(transpath)
stroke = agg.conv_stroke_curvetrans(curvetrans)
stroke.width(2.0)
rasterizer.add_path(stroke)
renderer.color_rgba8( white )
agg.render_scanlines_rgba(rasterizer, scanline, renderer);

if 0:
    ## Copy a rectangle from the buffer the rectangle defined by
    ## x0,y0->x1,y1 and paste it at xdest, ydest
    x0, y0 = 10, 50
    x1, y1 = 110, 190
    xdest, ydest = 350, 200



    widthr, heightr = x1-x0, y1-y0
    strider = widthr*4
    copybuffer = agg.buffer(widthr, heightr, strider)


    rbufcopy = agg.rendering_buffer()
    rbufcopy.attachb(copybuffer)
    pfcopy = agg.pixel_format_rgba(rbufcopy)
    rbasecopy = agg.renderer_base_rgba(pfcopy)

    rect = agg.rect(x0, y0, x1, y1)
    print rect.is_valid()
    rectp = agg.rectPtr(rect)
    #print dir(rbasecopy)

    # agg is funny about the arguments to copy from; the last 2 args are
    # dx, dy.  If the src and dest buffers are the same size and you omit
    # the dx and dy args, the position of the copy in the dest buffer is
    # the same as in the src.  Since our dest buffer is smaller than our
    # src buffer, we have to offset the location by -x0, -y0
    rbasecopy.copy_from(rbuf, rect, -x0, -y0);

    # paste the rectangle at a new location xdest, ydest
    rbase.copy_from(rbufcopy, None, xdest, ydest);



## Display it with PIL
s = buffer.to_string()
print len(s)
import Image
im = Image.fromstring( "RGBA", (width, height), s)
im.show()





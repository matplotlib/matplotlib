# this example uses the agg python module directly there is no
# documentation -- you have to know how to use the agg c++ API to use
# it
import matplotlib.agg as agg

width, height = 600,400
stride = width*4
buffer = agg.buffer(width, height, stride)

rbuf = agg.rendering_buffer()
rbuf.attachb(buffer)

red = agg.rgba(1,0,0,1)
blue = agg.rgba(0,0,1,1)
green = agg.rgba(0,1,0,1)
black = agg.rgba(0,0,0,1)
white = agg.rgba(1,1,1,1)
yellow = agg.rgba(.75,.75,1,1)

path = agg.path_storage()
path.move_to(10,10)
path.line_to(100,100)
path.line_to(200,200)
path.line_to(100,200)
path.close_polygon()

stroke = agg.conv_stroke(path)
stroke.width(3.0)

pf = agg.pixel_format(rbuf)
rbase = agg.renderer_base(pf)
rbase.clear(blue)

renderer =  agg.renderer_scanline_aa_solid(rbase);
renderer.color( red )

rasterizer = agg.rasterizer_scanline_aa()
rasterizer.add_path(stroke)

scanline = agg.scanline_p8()

agg.render_scanlines(rasterizer, scanline, renderer);

s = buffer.to_string()
print len(s)
import Image
im = Image.fromstring( "RGBA", (width, height), s)
im.show()





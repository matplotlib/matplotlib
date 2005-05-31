# this example uses the agg python module directly there is no
# documentation -- you have to know how to use the agg c++ API to use
# it
import matplotlib.agg as agg

width, height = 600,400
stride = width*4
buffer = agg.buffer(width, height, stride)

rbuf = agg.rendering_buffer()
rbuf.attachb(buffer)

red = agg.rgba8(255,0,0,255)
blue = agg.rgba8(0,0,255,255)
green = agg.rgba8(0,255,0,255)
black = agg.rgba8(0,0,0,255)
white = agg.rgba8(255,255,255,255)
yellow = agg.rgba8(192,192,255,255)


path = agg.path_storage()
path.move_to(10,10)
path.line_to(100,100)
path.line_to(200,200)
path.line_to(100,200)
path.close_polygon()

stroke = agg.conv_stroke_path(path)
stroke.width(3.0)

pf = agg.pixel_format_rgba(rbuf)
rbase = agg.renderer_base_rgba(pf)
rbase.clear_rgba8(blue) 

renderer =  agg.renderer_scanline_aa_solid_rgba(rbase);
renderer.color_rgba8( red )

rasterizer = agg.rasterizer_scanline_aa()
rasterizer.add_path(stroke)

scanline = agg.scanline_p8()

agg.render_scanlines_rgba(rasterizer, scanline, renderer);

s = buffer.to_string()
print len(s)
import Image
im = Image.fromstring( "RGBA", (width, height), s)
im.show()





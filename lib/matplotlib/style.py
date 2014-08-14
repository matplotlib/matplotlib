from matplotlib.backend_bases import GraphicsContextBase

def style_property(name = 'style', expected_type = Style):
    """ Property for style attributes. Performs type checking """
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)
    @prop.setter
    def prop(self, value):
        if isinstance(value, expected_type) or value is None:
            setattr(self, storage_name, value)
        else:
            raise TypeError('{} must be a {}'.format(name, expected_type))
         
    return prop
    
    
class Style(GraphicsContextBase):

    _lineStyles = {
    '-':    'solid',
     '--':   'dashed',
    '-.':   'dashdot',
    ':':    'dotted'}

    def get_graylevel(self):
        'Just returns the foreground color'
        return self._rgb

    alpha = property(GraphicsContextBase.get_alpha, GraphicsContextBase.set_alpha)
    antialiased = property(GraphicsContextBase.get_antialiased, GraphicsContextBase.set_antialiased)
    capstyle = property(GraphicsContextBase.get_capstyle, GraphicsContextBase.set_capstyle)
    clip_rectangle = property(GraphicsContextBase.get_clip_rectangle, GraphicsContextBase.set_clip_rectangle)
    clip_path = property(GraphicsContextBase.get_clip_path, GraphicsContextBase.set_clip_path)
    graylevel = property(get_graylevel, GraphicsContextBase.set_graylevel)
    joinstyle = property(GraphicsContextBase.get_joinstyle, GraphicsContextBase.set_joinstyle)
    linewidth = property(GraphicsContextBase.get_linewidth, GraphicsContextBase.set_linewidth)
    linestyle = property(GraphicsContextBase.get_linestyle, GraphicsContextBase.set_linestyle)
    url = property(GraphicsContextBase.get_url, GraphicsContextBase.set_url)
    gid = property(GraphicsContextBase.get_gid, GraphicsContextBase.set_gid)
    snap = property(GraphicsContextBase.get_snap, GraphicsContextBase.set_snap)
    hatch = property(GraphicsContextBase.get_hatch, GraphicsContextBase.set_hatch)
 
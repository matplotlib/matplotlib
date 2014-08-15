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


    # Refactoring of set_dashes into two properties..
    @property
    def dash_offset(self):
        return self._dashes[0]
    @dash_offset.setter
    def dash_offset(self, value):
        self._dashes[0] = value

    @property
    def dashes(self):
        return self._dashes[1]

    @dashes.setter
    def dashes(self, value):
        if value is not None:
            dl = np.asarray(value)
            if np.any(dl <= 0.0):
                raise ValueError("All values in the dash list must be positive")

        self._dashed[1] = value

    #Color property is an alternative to 'set_foreground'. It does the same thing, but makes no allowances for providing a colour already in RGBA format..
    @property
    def color(self):
        return self._rgb
    @color.setter
    def color(self, value):
        self.set_foreground(value)

    # Defining 3 properties for sketch params
    @property
    def sketch_scale(self):
        return self._sketch[0]
    @sketch_scale.setter
    def sketch_scale(self, value):        
        self.set_sketch_params(scale = value)

    @property
    def sketch_length(self):
        return self._sketch[1]
    @sketch_length.setter
    def sketch_length(self, value):
        self.set_sketch_params(length = value)

    @property
    def sketch_randomness(self):
        return self._sketch[2]
    @sketch_randomness.setter
    def sketch_randomness(self, value):
        self.set_sketch_params(randomness = value)

    @classmethod
    def from_dict(cls, styleDict):
        """ Generate a style class from a dictionary """
        st = cls()
        for key, value in styleDict.iteritems():
            setattr(st, key, value)

        return st
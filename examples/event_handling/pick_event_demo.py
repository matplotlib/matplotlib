"""
===============
Pick Event Demo
===============


You can enable picking by setting the "picker" property of an artist
(for example, a matplotlib Line2D, Text, Patch, Polygon, AxesImage,
etc...)

There are a variety of meanings of the picker property

    None -  picking is disabled for this artist (default)

    boolean - if True then picking will be enabled and the
      artist will fire a pick event if the mouse event is over
      the artist

    float - if picker is a number it is interpreted as an
      epsilon tolerance in points and the artist will fire
      off an event if it's data is within epsilon of the mouse
      event.  For some artists like lines and patch collections,
      the artist may provide additional data to the pick event
"test.py" 186L, 6570C                                         1,1           Top

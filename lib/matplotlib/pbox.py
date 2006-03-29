from __future__ import division

class PBox(list):
    '''
    A left-bottom-width-height (lbwh) specification of a bounding box,
    such as is used to specify the position of an Axes object within
    a Figure.
    It is a 4-element list with methods for changing the size, shape,
    and position relative to its container.
    '''
    coefs = {'C':  (0.5, 0.5),
             'SW': (0,0),
             'S':  (0.5, 0),
             'SE': (1.0, 0),
             'E':  (1.0, 0.5),
             'NE': (1.0, 1.0),
             'N':  (0.5, 1.0),
             'NW': (0, 1.0),
             'W':  (0, 0.5)}
    def __init__(self, box, container=None, llur=False):
        if len(box) != 4:
            raise ValueError("Argument must be iterable of length 4")
        if llur:
            box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        list.__init__(self, box)
        self.set_container(container)

    def as_llur(self):
        return [self[0], self[1], self[0]+self[2], self[1]+self[3]]

    def set_container(self, box=None):
        if box is None:
            box = self
        if len(box) != 4:
            raise ValueError("Argument must be iterable of length 4")
        self._container = list(box)

    def get_container(self, box):
        return self._container

    def anchor(self, c, container=None):
        '''
        Shift to position c within its container.

        c can be a sequence (cx, cy) where cx, cy range from 0 to 1,
        where 0 is left or bottom and 1 is right or top.

        Alternatively, c can be a string: C for centered,
        S for bottom-center, SE for bottom-left, E for left, etc.

        Optional arg container is the lbwh box within which the
        PBox is positioned; it defaults to the initial
        PBox.
        '''
        if container is None:
            container = self._container
        l,b,w,h = container
        if isinstance(c, str):
            cx, cy = self.coefs[c]
        else:
            cx, cy = c
        W,H = self[2:]
        self[:2] = l + cx * (w-W), b + cy * (h-H)
        return self

    def shrink(self, mx, my):
        '''
        Shrink the box by mx in the x direction and my in the y direction.
        The lower left corner of the box remains unchanged.
        Normally mx and my will be <= 1, but this is not enforced.
        '''
        assert mx >= 0 and my >= 0
        self[2:] = mx * self[2], my * self[3]
        return self

    def shrink_to_aspect(self, box_aspect, fig_aspect = 1):
        '''
        Shrink the box so that it is as large as it can be while
        having the desired aspect ratio, box_aspect.
        If the box coordinates are relative--that is, fractions of
        a larger box such as a figure--then the physical aspect
        ratio of that figure is specified with fig_aspect, so
        that box_aspect can also be given as a ratio of the
        absolute dimensions, not the relative dimensions.
        '''
        assert box_aspect > 0 and fig_aspect > 0
        l,b,w,h = self._container
        H = w * box_aspect/fig_aspect
        if H <= h:
            W = w
        else:
            W = h * fig_aspect/box_aspect
            H = h
        self[2:] = W,H
        return self

    def splitx(self, *args):
        '''
        e.g., PB.splitx(f1, f2, ...)

        Returns a list of new PBoxes formed by
        splitting the original one (PB) with vertical lines
        at fractional positions f1, f2, ...
        '''
        boxes = []
        xf = [0] + list(args) + [1]
        l,b,w,h = self[:]
        for xf0, xf1 in zip(xf[:-1], xf[1:]):
            boxes.append(PBox([l+xf0*w, b, (xf1-xf0)*w, h]))
        return boxes

    def splity(self, *args):
        '''
        e.g., PB.splity(f1, f2, ...)

        Returns a list of new PBoxes formed by
        splitting the original one (PB) with horizontal lines
        at fractional positions f1, f2, ..., with y measured
        positive up.
        '''
        boxes = []
        yf = [0] + list(args) + [1]
        l,b,w,h = self[:]
        for yf0, yf1 in zip(yf[:-1], yf[1:]):
            boxes.append(PBox([l, b+yf0*h, w, (yf1-yf0)*h]))
        return boxes



import numpy as np
import matplotlib
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image

matplotlib.rc('image', origin='upper')

class MenuItem(artist.Artist):
    parser = mathtext.MathTextParser("Bitmap")
    padx = 5
    pady =5
    def __init__(self, fig, labelstr):
        artist.Artist.__init__(self)
        self.set_figure(fig)


        x, self.depth = self.parser.to_rgba(
            labelstr, color='black', fontsize=14, dpi=100)
        xHover, depth = self.parser.to_rgba(
            labelstr, color='white', fontsize=14, dpi=100)


        self.labelwidth = x.shape[1]
        self.labelheight = x.shape[0]
        print 'h', self.labelheight
        self.label = image.FigureImage(fig)
        self.label.set_array(x.astype(float)/255.)

        self.labelHover = image.FigureImage(fig)
        self.labelHover.set_array(xHover.astype(float)/255.)



        # we'll update these later
        self.rect = patches.Rectangle((0,0), 1,1, facecolor='yellow', alpha=0.2)
        self.rectHover = patches.Rectangle((0,0), 1,1, facecolor='blue', alpha=0.2)



    def set_extent(self, x, y, w, h):
        print x, y, w, h
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)

        self.rectHover.set_x(x)
        self.rectHover.set_y(y)
        self.rectHover.set_width(w)
        self.rectHover.set_height(h)

        self.label.ox = x+self.padx
        self.label.oy = y-self.depth+self.pady/2.

        self.rect._update_patch_transform()
        self.rectHover._update_patch_transform()
        self.labelHover.ox = x+self.padx
        self.labelHover.oy = y-self.depth+self.pady/2.
        self.hover = False

        self.activeRect = self.rect
        self.activeLabel = self.label

    def draw(self, renderer):
        self.activeRect.draw(renderer)
        self.activeLabel.draw(renderer)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b,junk = self.rect.contains(event)
        if b:
            self.activeRect = self.rectHover
            self.activeLabel = self.labelHover
        else:
            self.activeRect = self.rect
            self.activeLabel = self.label

        h = self.hover
        self.hover = b
        return b!=h

class Menu:

    def __init__(self, fig, labels):
        self.figure = fig
        fig.suppressComposite = True
        menuitems = []
        self.numitems = len(labels)
        for label in labels:
            menuitems.append(MenuItem(fig, label))

        self.menuitems = menuitems


        maxw = max([item.labelwidth for item in menuitems])
        maxh = max([item.labelheight for item in menuitems])


        totalh = self.numitems*maxh + (self.numitems+1)*2*MenuItem.pady

        x0 = 100
        y0 = 400

        width = maxw + 2*MenuItem.padx
        height = maxh+MenuItem.pady
        for item in menuitems:
            left = x0
            bottom = y0-maxh-MenuItem.pady


            item.set_extent(left, bottom, width, height)

            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady


        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems:
            b = item.set_hover(event)
            draw = b

        if draw:
            print 'draw'
            self.figure.canvas.draw()


fig = plt.figure()
menu = Menu(fig, ('open', 'close', 'save', 'save as', 'quit'))

plt.show()





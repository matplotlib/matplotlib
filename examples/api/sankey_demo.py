#!/usr/bin/env python
# Time-stamp: <2010-02-10 01:49:08 ycopin>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

def sankey(ax, losses, labels=None,
           dx=40, dy=10, angle=45, w=3, dip=10, offset=2, **kwargs):
    """Draw a Sankey diagram.

    losses: array of losses, should sum up to 100%
    labels: loss labels (same length as losses),
            or None (use default labels) or '' (no labels)
    dx: horizontal elongation
    dy: vertical elongation
    angle: arrow angle [deg]
    w: arrow shoulder
    dip: input dip
    offset: text offset
    **kwargs: propagated to Patch (e.g. fill=False)

    Return (patch,texts)."""

    assert sum(losses)==100, "Input losses don't sum up to 100%"

    def add_loss(loss, last=False):
        h = (loss/2+w)*np.tan(angle/180.*np.pi) # Arrow tip height
        move,(x,y) = path[-1]           # Use last point as reference
        if last:                        # Final loss (horizontal)
            path.extend([(Path.LINETO,[x+dx,y]),
                         (Path.LINETO,[x+dx,y+w]),
                         (Path.LINETO,[x+dx+h,y-loss/2]), # Tip
                         (Path.LINETO,[x+dx,y-loss-w]),
                         (Path.LINETO,[x+dx,y-loss])])
            tips.append(path[-3][1])
        else:                           # Intermediate loss (vertical)
            path.extend([(Path.LINETO,[x+dx/2,y]),
                        (Path.CURVE3,[x+dx,y]),
                        (Path.CURVE3,[x+dx,y+dy]),
                        (Path.LINETO,[x+dx-w,y+dy]),
                        (Path.LINETO,[x+dx+loss/2,y+dy+h]), # Tip
                        (Path.LINETO,[x+dx+loss+w,y+dy]),
                        (Path.LINETO,[x+dx+loss,y+dy]),
                        (Path.CURVE3,[x+dx+loss,y-loss]),
                        (Path.CURVE3,[x+dx/2+loss,y-loss])])
            tips.append(path[-5][1])

    tips = []                           # Arrow tip positions
    path = [(Path.MOVETO,[0,100])]      # 1st point
    for i,loss in enumerate(losses):
        add_loss(loss, last=(i==(len(losses)-1)))
    path.extend([(Path.LINETO,[0,0]),
                 (Path.LINETO,[dip,50]), # Dip
                 (Path.CLOSEPOLY,[0,100])])
    codes,verts = zip(*path)
    verts = np.array(verts)

    # Path patch
    path = Path(verts,codes)
    patch = mpatches.PathPatch(path, **kwargs)
    ax.add_patch(patch)

    # Labels
    if labels=='':                      # No labels
        pass
    elif labels is None:                # Default labels
        labels = [ '%2d%%' % loss for loss in losses ]
    else:
        assert len(labels)==len(losses)

    texts = []
    for i,label in enumerate(labels):
        x,y = tips[i]                   # Label position
        last = (i==(len(losses)-1))
        if last:
            t = ax.text(x+offset,y,label, ha='left', va='center')
        else:
            t = ax.text(x,y+offset,label, ha='center', va='bottom')
        texts.append(t)

    # Axes management
    ax.set_xlim(verts[:,0].min()-10, verts[:,0].max()+40)
    ax.set_ylim(verts[:,1].min()-10, verts[:,1].max()+20)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([])
    ax.set_yticks([])

    return patch,texts

if __name__=='__main__':

    losses = [10.,20.,5.,15.,10.,40.]
    labels = ['First','Second','Third','Fourth','Fifth','Hurray!']
    labels = [ s+'\n%d%%' % l for l,s in zip(losses,labels) ]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    patch,texts = sankey(ax, losses, labels, fc='g', alpha=0.2)
    texts[1].set_color('r')
    texts[-1].set_fontweight('bold')

    plt.show()

from matplotlib import pylab

DATA = ((1, 3),
        (2, 4),
        (3, 1),
        (4, 2))
# dash_style =
#     direction, length, (text)rotation, dashrotation, push
# (The parameters are varied to show their effects,
# not for visual appeal).
dash_style = (
    (0, 20, -15, 30, 10),
    (1, 30, 0, 15, 10),
    (0, 40, 15, 15, 10),
    (1, 20, 30, 60, 10),
    )

def test_dashpointlabel(save=False):
    pylab.clf()
    (x,y) = zip(*DATA)
    pylab.plot(x, y, marker='o')
    for i in xrange(len(DATA)):
        (x,y) = DATA[i]
        (dd, dl, r, dr, dp) = dash_style[i]
        pylab.text(x, y, str((x,y)), withdash=True,
                   dashdirection=dd,
                   dashlength=dl,
                   rotation=r,
                   dashrotation=dr,
                   dashpush=dp,
                   )
    axis = pylab.gca()
    axis.set_xlim((0.0, 5.0))
    axis.set_ylim((0.0, 5.0))
    if save:
        pylab.savefig('dashpointlabel')
    pylab.show()

if __name__ == '__main__':
    test_dashpointlabel()

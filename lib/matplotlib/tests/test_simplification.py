from __future__ import print_function

import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
import matplotlib.pyplot as plt

from pylab import *
import numpy as np
from matplotlib import patches, path, transforms

from nose.tools import raises
import io

nan = np.nan
Path = path.Path

# NOTE: All of these tests assume that path.simplify is set to True
# (the default)

@image_comparison(baseline_images=['clipping'], remove_text=True)
def test_clipping():
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*pi*t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, s, linewidth=1.0)
    ax.set_ylim((-0.20, -0.28))

@image_comparison(baseline_images=['overflow'], remove_text=True)
def test_overflow():
    x = np.array([1.0,2.0,3.0,2.0e5])
    y = np.arange(len(x))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    ax.set_xlim(xmin=2,xmax=6)

@image_comparison(baseline_images=['clipping_diamond'], remove_text=True)
def test_diamond():
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    y = np.array([1.0, 0.0, -1.0, 0.0, 1.0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlim(xmin=-0.6, xmax=0.6)
    ax.set_ylim(ymin=-0.6, ymax=0.6)

@cleanup
def test_noise():
    np.random.seed(0)
    x = np.random.uniform(size=(5000,)) * 50

    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)

    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = list(path.iter_segments(simplify=(800, 600)))

    assert len(simplified) == 3884

@cleanup
def test_sine_plus_noise():
    np.random.seed(0)
    x = np.sin(np.linspace(0, np.pi * 2.0, 1000)) + np.random.uniform(size=(1000,)) * 0.01

    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.plot(x, solid_joinstyle='round', linewidth=2.0)

    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = list(path.iter_segments(simplify=(800, 600)))

    assert len(simplified) == 876

@image_comparison(baseline_images=['simplify_curve'], remove_text=True)
def test_simplify_curve():
    pp1 = patches.PathPatch(
        Path([(0, 0), (1, 0), (1, 1), (nan, 1), (0, 0), (2, 0), (2, 2), (0, 0)],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
        fc="none")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_patch(pp1)
    ax.set_xlim((0, 2))
    ax.set_ylim((0, 2))

@image_comparison(baseline_images=['hatch_simplify'], remove_text=True)
def test_hatch():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, hatch="/"))
    ax.set_xlim((0.45, 0.55))
    ax.set_ylim((0.45, 0.55))

@image_comparison(baseline_images=['fft_peaks'], remove_text=True)
def test_fft_peaks():
    fig = plt.figure()
    t = arange(65536)
    ax = fig.add_subplot(111)
    p1 = ax.plot(abs(fft(sin(2*pi*.01*t)*blackman(len(t)))))

    path = p1[0].get_path()
    transform = p1[0].get_transform()
    path = transform.transform_path(path)
    simplified = list(path.iter_segments(simplify=(800, 600)))

    assert len(simplified) == 20

@cleanup
def test_start_with_moveto():
    # Should be entirely clipped away to a single MOVETO
    data = b"""
ZwAAAAku+v9UAQAA+Tj6/z8CAADpQ/r/KAMAANlO+v8QBAAAyVn6//UEAAC6ZPr/2gUAAKpv+v+8
BgAAm3r6/50HAACLhfr/ewgAAHyQ+v9ZCQAAbZv6/zQKAABepvr/DgsAAE+x+v/lCwAAQLz6/7wM
AAAxx/r/kA0AACPS+v9jDgAAFN36/zQPAAAF6Pr/AxAAAPfy+v/QEAAA6f36/5wRAADbCPv/ZhIA
AMwT+/8uEwAAvh77//UTAACwKfv/uRQAAKM0+/98FQAAlT/7/z0WAACHSvv//RYAAHlV+/+7FwAA
bGD7/3cYAABea/v/MRkAAFF2+//pGQAARIH7/6AaAAA3jPv/VRsAACmX+/8JHAAAHKL7/7ocAAAP
rfv/ah0AAAO4+/8YHgAA9sL7/8QeAADpzfv/bx8AANzY+/8YIAAA0OP7/78gAADD7vv/ZCEAALf5
+/8IIgAAqwT8/6kiAACeD/z/SiMAAJIa/P/oIwAAhiX8/4QkAAB6MPz/HyUAAG47/P+4JQAAYkb8
/1AmAABWUfz/5SYAAEpc/P95JwAAPmf8/wsoAAAzcvz/nCgAACd9/P8qKQAAHIj8/7cpAAAQk/z/
QyoAAAWe/P/MKgAA+aj8/1QrAADus/z/2isAAOO+/P9eLAAA2Mn8/+AsAADM1Pz/YS0AAMHf/P/g
LQAAtur8/10uAACr9fz/2C4AAKEA/f9SLwAAlgv9/8ovAACLFv3/QDAAAIAh/f+1MAAAdSz9/ycx
AABrN/3/mDEAAGBC/f8IMgAAVk39/3UyAABLWP3/4TIAAEFj/f9LMwAANm79/7MzAAAsef3/GjQA
ACKE/f9+NAAAF4/9/+E0AAANmv3/QzUAAAOl/f+iNQAA+a/9/wA2AADvuv3/XDYAAOXF/f+2NgAA
29D9/w83AADR2/3/ZjcAAMfm/f+7NwAAvfH9/w44AACz/P3/XzgAAKkH/v+vOAAAnxL+//04AACW
Hf7/SjkAAIwo/v+UOQAAgjP+/905AAB5Pv7/JDoAAG9J/v9pOgAAZVT+/606AABcX/7/7zoAAFJq
/v8vOwAASXX+/207AAA/gP7/qjsAADaL/v/lOwAALZb+/x48AAAjof7/VTwAABqs/v+LPAAAELf+
/788AAAHwv7/8TwAAP7M/v8hPQAA9df+/1A9AADr4v7/fT0AAOLt/v+oPQAA2fj+/9E9AADQA///
+T0AAMYO//8fPgAAvRn//0M+AAC0JP//ZT4AAKsv//+GPgAAojr//6U+AACZRf//wj4AAJBQ///d
PgAAh1v///c+AAB+Zv//Dz8AAHRx//8lPwAAa3z//zk/AABih///TD8AAFmS//9dPwAAUJ3//2w/
AABHqP//ej8AAD6z//+FPwAANb7//48/AAAsyf//lz8AACPU//+ePwAAGt///6M/AAAR6v//pj8A
AAj1//+nPwAA/////w=="""

    import base64
    if hasattr(base64, 'encodebytes'):
        # Python 3 case
        decodebytes = base64.decodebytes
    else:
        # Python 2 case
        decodebytes = base64.decodestring

    verts = np.fromstring(decodebytes(data), dtype='<i4')
    verts = verts.reshape((len(verts) / 2, 2))
    path = Path(verts)
    segs = path.iter_segments(transforms.IdentityTransform(), clip=(0.0, 0.0, 100.0, 100.0))
    segs = list(segs)
    assert len(segs) == 1
    assert segs[0][1] == Path.MOVETO

@cleanup
@raises(OverflowError)
def test_throw_rendering_complexity_exceeded():
    rcParams['path.simplify'] = False
    xx = np.arange(200000)
    yy = np.random.rand(200000)
    yy[1000] = np.nan
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, yy)
    try:
        fig.savefig(io.BytesIO())
    finally:
        rcParams['path.simplify'] = True

@image_comparison(baseline_images=['clipper_edge'], remove_text=True)
def test_clipper():
    dat = (0, 1, 0, 2, 0, 3, 0, 4, 0, 5)
    fig = plt.figure(figsize=(2, 1))
    fig.subplots_adjust(left = 0, bottom = 0, wspace = 0, hspace = 0)

    ax = fig.add_axes((0, 0, 1.0, 1.0), ylim = (0, 5), autoscale_on = False)
    ax.plot(dat)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlim(5, 9)

@image_comparison(baseline_images=['para_equal_perp'], remove_text=True)
def test_para_equal_perp():
    x = np.array([0, 1, 2, 1, 0, -1, 0, 1] + [1] * 128)
    y = np.array([1, 1, 2, 1, 0, -1, 0, 0] + [0] * 128)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x + 1, y + 1)
    ax.plot(x + 1, y + 1, 'ro')

@image_comparison(baseline_images=['clipping_with_nans'])
def test_clipping_with_nans():
    x = np.linspace(0, 3.14 * 2, 3000)
    y = np.sin(x)
    x[::100] = np.nan

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_ylim(-0.25, 0.25)


if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)

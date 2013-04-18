import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib

def draw_arrow(ax, t, r):
    ax.annotate('', xy=(0.5, 0.5+r), xytext=(0.5, 0.5), size=30,
                arrowprops=dict(arrowstyle=t,
                                fc="b", ec='k'))

@image_comparison(baseline_images=['fancyarrow_test_image'])
def test_fancyarrow():
    r = [0.4, 0.3, 0.2, 0.1]
    t = ["fancy", "simple", matplotlib.patches.ArrowStyle.Fancy()]

    fig, axes = plt.subplots(len(t), len(r), squeeze=False,
                             subplot_kw=dict(aspect=True),
                         figsize=(8, 4.5))

    for i_r, r1 in enumerate(r):
        for i_t, t1 in enumerate(t):
            ax = axes[i_t, i_r]
            draw_arrow(ax, t1, r1)
            ax.tick_params(labelleft=False, labelbottom=False)


if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)


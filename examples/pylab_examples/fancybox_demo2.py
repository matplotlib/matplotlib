import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

styles = mpatch.BoxStyle.get_styles()

figheight = (len(styles)+.5)
fig1 = plt.figure(1, (4, figheight))
fontsize = 0.4 * fig1.dpi

for i, (stylename, styleclass) in enumerate(styles.items()):
    fig1.text(0.5, (float(len(styles)) - 0.5 - i)/figheight, stylename,
              ha="center",
              size=fontsize,
              transform=fig1.transFigure,
              bbox=dict(boxstyle=stylename, fc="w", ec="k"))
plt.draw()
plt.show()

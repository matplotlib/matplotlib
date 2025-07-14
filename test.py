import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(0, 1, 10)
y = np.random.randn(10)

def reset():
    plt.close('all')
    plt.figure(figsize=(6, 4))

def show(title):
    plt.legend(labelcolor='linecolor')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2)
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    # Top Left: Filled Histogram (default color C0)
    axes[0, 0].hist(x, histtype='bar', label="spam")
    leg00 = axes[0, 0].legend(labelcolor='linecolor')
    assert np.allclose(leg00.get_texts()[0].get_color()[:3], mcolors.to_rgb('C0'))

    # Top Right: Step Histogram (default color C0)
    axes[0, 1].hist(x, histtype='step', label="spam")
    leg01 = axes[0, 1].legend(labelcolor='linecolor')
    assert np.allclose(leg01.get_texts()[0].get_color()[:3], mcolors.to_rgb('C0'))

    # Bottom Left: Scatter (filled, default color C0)
    axes[1, 0].scatter(x, y, label="spam")
    leg10 = axes[1, 0].legend(labelcolor='linecolor')
    assert np.allclose(leg10.get_texts()[0].get_color()[:3], mcolors.to_rgb('C0'))

    # Bottom Right: Scatter (outline, default edge color C0)
    axes[1, 1].scatter(x, y, label="spam")
    leg11 = axes[1, 1].legend(labelcolor='linecolor')
    assert np.allclose(leg11.get_texts()[0].get_color()[:3], mcolors.to_rgb('C0'))

    plt.close(fig)

# 1. plot c=None (invisible)
reset()
plt.plot(x, y, 'o', c='None', label="plot – c=None")
show("1. Plot – c=None (invisible)")

# 2. plot with full color
reset()
plt.plot(x, y, 'o', c='orange', label="plot – orange")
show("2. Plot – filled (orange)")

# 3. plot with alpha=0
reset()
plt.plot(x, y, 'o', color=(1, 0, 0, 0), label="plot – alpha=0 (transparent)")
show("3. Plot – RGBA alpha=0")

# 4. plot with mec only
reset()
plt.plot(x, y, 'o', mec='red', mfc='None', label="plot – mec='red', mfc=None")
show("4. Plot – red edge only")

# 5. plot with mfc only
reset()
plt.plot(x, y, 'o', mfc='cyan', mec='None', label="plot – mfc='cyan', mec=None")
show("5. Plot – cyan fill only")

# 6. scatter c=None
reset()
plt.scatter(x, y, c='None', label="scatter – c=None")
show("6. Scatter – c=None (invisible)")

# 7. scatter with colormap
reset()
plt.scatter(x, y, c=np.linspace(0, 1, 10), cmap='viridis', label="scatter – colormap")
show("7. Scatter – colormap (viridis)")

# 8. scatter fc=None, ec=blue
reset()
plt.scatter(x, y, fc='None', ec='blue', label="scatter – fc=None, ec=blue")
show("8. Scatter – blue edge")

# 9. scatter fc=magenta, ec=None
reset()
plt.scatter(x, y, fc='magenta', ec='None', label="scatter – fc=magenta, ec=None")
show("9. Scatter – magenta fill only")

# 10. histtype='bar'
reset()
plt.hist(np.random.randn(100), bins=10, label="hist – bar", color='C3', histtype='bar')
show("10. Histogram – bar (facecolor)")

# 11. histtype='step'
reset()
plt.hist(np.random.randn(100), bins=10, label="hist – step", color='C4', histtype='step')
show("11. Histogram – step (edgecolor)")

# 12. Mixed plot (c=None), scatter (valid), hist (valid)
reset()
plt.plot(x, y, 'o', c='None', label="spam – invisible")
plt.scatter(x, y, c='limegreen', label="ham – visible")
plt.hist(np.random.randn(100), label="eggs – bar hist", color='teal')
show("12. Mixed – spam/ham/eggs")

# 13. plot with colormap (valid RGBA)
reset()
rgba = plt.cm.plasma(0.5)
plt.plot(x, y, color=rgba, label="plot – RGBA colormapped")
show("13. Plot – RGBA colormap")

# 14. scatter with multiple unique colors (array)
reset()
colors_array = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink'])
plt.scatter(x, y, c=colors_array, label="scatter – multiple facecolors")
show("14. Scatter – mixed facecolors")

# 15. scatter with zero-size color array (simulate empty input)
reset()
try:
    plt.scatter(x, y, c=np.array([]), label="scatter – empty color array")
    show("15. Scatter – empty color array")
except Exception as e:
    print("15. Expected error:", e)

# 16. plot with no label (legend shouldn't show it)
reset()
plt.plot(x, y, 'o', c='red')  # No label
plt.plot(x, y+1, 'o', c='green', label="visible")
show("16. One line has no label")

# 17. scatter with transparent RGBA
reset()
plt.scatter(x, y, color=(0, 0, 1, 0), label="scatter – RGBA alpha=0")
show("17. Scatter – fully transparent")

# 18. invalid labelcolor string
reset()
try:
    plt.plot(x, y, label="invalid color test")
    plt.legend(labelcolor='not-a-color')
except ValueError as e:
    print("18. Expected error:", e)

# 19. labelcolor = list of colors (must match label count)
reset()
plt.plot(x, y, label="spam")
plt.plot(x, y+1, label="ham")
plt.plot(x, y+2, label="eggs")
plt.legend(labelcolor=['red', 'green', 'blue'])
plt.title("19. labelcolor=['r', 'g', 'b']")
plt.tight_layout()
plt.show()

# 20. labelcolor = 'none'
reset()
plt.plot(x, y, label="spam")
plt.plot(x, y+1, label="ham")
plt.legend(labelcolor='none')
plt.title("20. labelcolor='none' (hide labels)")
plt.tight_layout()
plt.show()
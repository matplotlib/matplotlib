import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

fig=plt.figure(1, figsize=(3,3))
ax = plt.subplot(111)

at = AnchoredText("Figure 1a",
                  prop=dict(size=15), frameon=True,
                  loc=2, 
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

plt.show()

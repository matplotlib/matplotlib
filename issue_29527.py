import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.__file__)
print(matplotlib.get_backend())

# matplotlib.use("TkAgg")
fig, ax = plt.subplots()

txt = "$k$1||\n1||"
# txt = "k1||\n1||"
ax.text(1, 0.5, txt, ha='right', ma='right', transform=ax.transAxes, fontsize=21)
# ax.axvline(0.97, color='red', linestyle='--')
# ax.axvline(0.98, color='red', linestyle='--')
# ax.axvline(0.99, color='red', linestyle='--')

# fig.show()
# fig.savefig("/tmp/bla.pdf")
fig.savefig("./new.png")
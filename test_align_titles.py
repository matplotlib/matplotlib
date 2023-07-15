import matplotlib as plt
fig, axs = plt.subplots(2, 2,
                        subplot_kw={"xlabel": "x", "ylabel": "", "title": "t"})
print(axs.shape)
axs[0][0].imshow(plt.zeros((3, 5)))
axs[0][1].imshow(plt.zeros((5, 3)))
axs[1][0].imshow(plt.zeros((1, 2)))
axs[1][1].imshow(plt.zeros((2, 1)))
axs[0][0].set_title('t2')
rowspan1 = axs[0][0].get_subplotspec().rowspan
print(rowspan1, rowspan1.start, rowspan1.stop)
rowspan2 = axs[1][1].get_subplotspec().rowspan
print(rowspan2, rowspan2.start, rowspan2.stop)

fig.align_labels()
fig.align_titles()
plt.show()
print("DONE")

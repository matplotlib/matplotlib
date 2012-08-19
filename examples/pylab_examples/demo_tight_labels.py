import matplotlib.pyplot as plt

for n in [1, 2, 3, 5]:
    fig, axes_list = plt.subplots(n, n)
    plt.tight_labels(0.5, 0.3)
    plt.tight_layout()

plt.show()

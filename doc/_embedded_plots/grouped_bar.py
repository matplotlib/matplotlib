import matplotlib.pyplot as plt

categories = ['A', 'B']
data0 = [1.0, 3.0]
data1 = [1.4, 3.4]
data2 = [1.8, 3.8]

fig, ax = plt.subplots(figsize=(4, 2.2))
ax.grouped_bar(
    [data0, data1, data2],
    tick_labels=categories,
    labels=['dataset 0', 'dataset 1', 'dataset 2'],
    colors=['#1f77b4', '#58a1cf', '#abd0e6'],
)
ax.legend()

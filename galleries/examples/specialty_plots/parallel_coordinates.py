"""
=====================
Parallel coordinates
=====================

A parallel coordinate plot visualizes multi-dimensional data by drawing
vertical axes for each feature and connecting individual data points as
polylines across these axes. This is useful for exploring patterns,
clusters, and relationships in high-dimensional datasets.

The `~.Axes.parallel_coordinates` method normalizes each dimension to the
same scale, so the absolute values are not visible in this representation.
"""

import matplotlib.pyplot as plt
import numpy as np

# Create a multi-dimensional dataset with 3 clusters
np.random.seed(19680801)

n_points = 50
centers = [[0, 0, 0, 0], [3, 3, 3, 3], [-1, 2, -2, 1]]
labels = ["Cluster A", "Cluster B", "Cluster C"]

data_list = []
label_list = []
for center, label in zip(centers, labels):
    data_list.append(np.random.randn(n_points, 4) + center)
    label_list.extend([label] * n_points)

data = np.vstack(data_list)
column_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]

# Use a structured array with class labels
structured_data = np.column_stack([data, label_list])

fig, ax = plt.subplots(layout="constrained")
ax.parallel_coordinates(structured_data, class_column=4, cols=column_names,
                        alpha=0.4, linewidth=0.8)
ax.set_title("Parallel coordinates showing three clusters")

plt.show()

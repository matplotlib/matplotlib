import pandas as pd
import matplotlib.pyplot as plt


def test_3d_timestamp_support():
    t = pd.Timestamp.now()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Test scatter
    ax.scatter(t, t, t)

    # Test set_zlim
    ax.set_zlim(t, t)

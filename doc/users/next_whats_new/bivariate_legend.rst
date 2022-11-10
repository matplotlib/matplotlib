Bivariate Legend
--------------------

Ebru Ayyorgun has added a new feature where users can map their data based on 2 variables
instead of being able to map by just 1 variable using colorbars. This allows for users
to visualize new correlations and extract more meaning from their datasets that 
previously couldn't be seen. This results in a "2D Colorbar". The user passes in 2 lists of 
data that must be the same size. Then each element of the lists are mapped to their individual 
colormaps and these colors are blended to result in a bivarate colormap of that single data point.
The function produces a list of rgb colors for each data point, which the user can then use
to pass into the color paramter of a matplotlib mappable plotting function. For more information 
and paramter details, see
:func:`~matplotlib.pyplot.bivariate_legend` and
:class:`matplotlib.bivariate_legend.Bivariate_legend`.

.. plot::
    import matplotlib
    from matplotlib import pyplot as plt
    import pandas as pd
    import math
    import numpy as np
    import matplotlib.colors as mcolors

    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data)
    df.columns = iris.feature_names

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    blegend = plt.bivariate_legend(
        df['petal length (cm)'], df['petal width (cm)'], d1num_bins=4, d2num_bins=4)
    scatter = ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
                        df['petal length (cm)'], c=blegend.get_mapped_colors() , alpha=1)
    plt.show()

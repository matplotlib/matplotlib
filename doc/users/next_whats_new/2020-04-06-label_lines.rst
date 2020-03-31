New `~.axes.Axes.label_lines` method
------------------------------------

A new `~.axes.Axes.label_lines` method has been added to label the end of lines on an axes.
Previously, the user had to go through the hassle of positioning each label individually
like the Bachelors degrees by gender example.

https://matplotlib.org/gallery/showcase/bachelors_degrees_by_gender.html

Now, to achieve the same effect, a user can simply call

    ax.label_lines()

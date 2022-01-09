from matplotlib.patches import BoxStyle, Rectangle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np

from cursorxlib import MyStyle2, SnaptoCursorEx

if __name__ == "__main__": 

 
    BoxStyle._style_list["lpointy"] = MyStyle2  # Register the custom style.

    from matplotlib.cbook import get_sample_data

    fname = get_sample_data('percent_bachelors_degrees_women_usa.csv',
                            asfileobj=False)
    gender_degree_data = np.genfromtxt(fname, delimiter=',', names=True)

    # You typically want your plot to be ~1.33x wider than tall. This plot
    # is a rare exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    fig, ax = plt.subplots(1, 1, figsize=(6, 7))

    # These are the colors that will be used in the plot
    ax.set_prop_cycle(color=[
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5'])

    # Remove the plot frame lines. They are unnecessary here.
    ax.spines[:].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary.
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    fig.subplots_adjust(left=.1, right=.75, bottom=.1, top=.94)
    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    ax.set_xlim(1969.5, 2011.1)
    ax.set_ylim(-0.25, 90)

    # Set a fixed location and format for ticks.
    ax.set_xticks(range(1970, 2011, 10))
    ax.set_yticks(range(0, 91, 10))
    # Use automatic StrMethodFormatter creation
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_major_formatter('{x:.0f}%')

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just
    # plotted. Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    ax.tick_params(axis='both', which='both', labelsize=10,
                   bottom=False, top=False, labelbottom=True,
                   left=False, right=False, labelleft=True)

    # Now that the plot is prepared, it's time to actually plot the data!
    # Note that I plotted the majors in order of the highest % in the final year.
    majors = ['Health Professions', 'Public Administration', 'Education',
              'Psychology', 'Foreign Languages', 'English',
              'Communications\nand Journalism', 'Art and Performance', 'Biology',
              'Agriculture', 'Social Sciences and History', 'Business',
              'Math and Statistics', 'Architecture', 'Physical Sciences',
              'Computer Science', 'Engineering']

    y_offsets = {'Foreign Languages': 0.5, 'English': -0.5,
                 'Communications\nand Journalism': 0.75,
                 'Art and Performance': -0.25, 'Agriculture': 1.25,
                 'Social Sciences and History': 0.25, 'Business': -0.75,
                 'Math and Statistics': 0.75, 'Architecture': -0.75,
                 'Computer Science': 0.75, 'Engineering': -0.25}

    for column in majors:
        # Plot each line separately with its own color.
        column_rec_name = column.replace('\n', '_').replace(' ', '_')

        line, = ax.plot('Year', column_rec_name, data=gender_degree_data,
                        lw=1)

        # Add a text label to the right end of every line. Most of the code below
        # is adding specific offsets y position because some labels overlapped.
        y_pos = gender_degree_data[column_rec_name][-1] - 0.5

        if column in y_offsets:
            y_pos += y_offsets[column]

        # Again, make sure that all labels are large enough to be easily read
        # by the viewer.
        ax.text(2011.5, y_pos, column, fontsize=8, color=line.get_color())

    # Make the title big enough so it spans the entire plot, but don't make it
    # so big that it requires two lines to show.

    # Note that if the title is descriptive enough, it is unnecessary to include
    # axis labels; they are self-evident, in this plot's case.
    fig.suptitle("Percentage of Bachelor's degrees conferred to women in\n"
                 "the U.S.A. by major (1970-2011)", fontsize=10, ha="center")

    # Always include your data source(s) and copyright notice! 
    ax.text(1966, -8, "Data source: https://matplotlib.org/stable/gallery/showcase/bachelors_degrees_by_gender.html", fontsize=8)

    # Finally, save the figure as a PNG.
    # You can also save it as a PDF, JPEG, etc.
    # Just change the file extension in this call.
    # fig.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')

    cursor = SnaptoCursorEx(ax, ax.lines[9].get_xdata(), ax.lines[9].get_ydata())
    
    # ax.autoscale(False)             ## autoscale OFF
    plt.gcf().canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

    plt.show()